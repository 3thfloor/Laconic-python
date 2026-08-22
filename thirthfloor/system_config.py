"""Hardware detection and safe load settings.

A standalone utility module. The Engine does not import this and does not
know it exists: the Engine stays explicit and unopinionated, and the caller
decides. Applications that need to run on unknown hardware (AODE) ask this
module what to use, then pass the answer to engine.load() themselves.

    from thirthfloor.system_config import SystemConfig
    from thirthfloor.engine import Engine

    profile = SystemConfig.detect()
    engine = Engine()
    engine.load("assistant", model_path,
                ctx=profile.recommended_ctx,
                gpu_layers=profile.recommended_gpu_layers)

The point: a student on a 4GB Chromebook and a developer on a 24GB GPU box
should both get settings that work, without anyone learning KV cache math.

Detection is stdlib only. psutil is used if it happens to be installed,
otherwise we fall back to platform-specific calls. Nothing here is required.

Author: Justin Bench, 3th Floor AI.
"""

import logging
import os
import platform
import shutil
import subprocess

logger = logging.getLogger("thirthfloor")

# Tier boundaries, in GB of system RAM.
CONSTRAINED_MAX_RAM_GB = 6.0

# A discrete GPU needs at least this much VRAM before we trust it with a
# whole model. Below this it is usually integrated graphics borrowing RAM.
GPU_TIER_MIN_VRAM_GB = 6.0

# Apple Silicon shares one pool between CPU and GPU. Metal offload is still
# worth it, but only once there is enough RAM to hold a real model.
UNIFIED_GPU_TIER_MIN_RAM_GB = 16.0

DEFAULT_CTX = 4096

BYTES_PER_GB = 1024 ** 3


class SystemConfig:
    """A snapshot of the machine the engine is running on.

    Build one with SystemConfig.detect(). The result is cached, so calling
    detect() on every load() is free after the first call.
    """

    _cached = None

    def __init__(self, tier, ram_gb, ram_free_gb, vram_gb, unified, gpu_name, os_name):
        self.tier = tier                # "constrained" | "mid" | "gpu"
        self.ram_gb = ram_gb
        self.ram_free_gb = ram_free_gb
        self.vram_gb = vram_gb
        self.unified = unified          # True when GPU and CPU share one memory pool
        self.gpu_name = gpu_name        # None when no usable GPU was found
        self.os_name = os_name

    # ---------------------------------------------------------------- detect

    @classmethod
    def detect(cls, refresh=False):
        """Inspect the machine and return a SystemConfig. Cached after the
        first call unless refresh=True."""
        if cls._cached is not None and not refresh:
            return cls._cached

        os_name = platform.system()
        ram_gb, ram_free_gb = _detect_ram_gb()
        vram_gb, unified, gpu_name = _detect_gpu(os_name, ram_gb)
        tier = _classify(ram_gb, vram_gb, unified)

        cls._cached = cls(
            tier=tier,
            ram_gb=ram_gb,
            ram_free_gb=ram_free_gb,
            vram_gb=vram_gb,
            unified=unified,
            gpu_name=gpu_name,
            os_name=os_name,
        )
        return cls._cached

    @classmethod
    def reset(cls):
        """Forget the cached detection. Mostly useful in tests."""
        cls._cached = None

    # ------------------------------------------------------------ suggestions

    @property
    def recommended_ctx(self):
        """Context window to pass as engine.load(..., ctx=...)."""
        return recommend_ctx(self.tier, self.ram_gb, self.vram_gb)

    @property
    def recommended_gpu_layers(self):
        """Layer count to pass as engine.load(..., gpu_layers=...).
        -1 offloads everything, 0 is pure CPU, a positive int is partial."""
        return recommend_gpu_layers(self.tier, self.vram_gb, self.unified)

    def load_kwargs(self):
        """Both recommendations as kwargs, for engine.load(alias, path, **kw)."""
        return {
            "ctx": self.recommended_ctx,
            "gpu_layers": self.recommended_gpu_layers,
        }

    # ------------------------------------------------------------- reporting

    def describe(self):
        """One human-readable line. This is what gets logged."""
        if self.gpu_name and self.unified:
            gpu = f"{self.gpu_name} unified memory"
        elif self.gpu_name:
            gpu = f"{self.gpu_name}, {self.vram_gb:.1f}GB VRAM"
        else:
            gpu = "no GPU"
        return f"{self.ram_gb:.1f}GB RAM, {gpu}, {self.os_name} (tier={self.tier})"

    def as_dict(self):
        return {
            "tier": self.tier,
            "ram_gb": round(self.ram_gb, 2),
            "ram_free_gb": round(self.ram_free_gb, 2),
            "vram_gb": round(self.vram_gb, 2),
            "unified": self.unified,
            "gpu_name": self.gpu_name,
            "os": self.os_name,
            "recommended_ctx": self.recommended_ctx,
            "recommended_gpu_layers": self.recommended_gpu_layers,
        }

    def __repr__(self):
        return f"<SystemConfig {self.describe()}>"


# --------------------------------------------------------------- tier policy


def _classify(ram_gb, vram_gb, unified):
    """Pick a tier. Deliberately blunt: three buckets, safe defaults."""
    if unified:
        # Shared pool. The GPU is only useful once there is RAM to spare.
        if ram_gb >= UNIFIED_GPU_TIER_MIN_RAM_GB:
            return "gpu"
        return "constrained" if ram_gb < CONSTRAINED_MAX_RAM_GB else "mid"
    if vram_gb >= GPU_TIER_MIN_VRAM_GB:
        return "gpu"
    if ram_gb < CONSTRAINED_MAX_RAM_GB:
        return "constrained"
    return "mid"


def recommend_ctx(tier, ram_gb, vram_gb):
    """Return a context window that will not take the machine down.

    These are floors, not ceilings. A caller who knows their model can always
    pass ctx explicitly and it wins.
    """
    if tier == "constrained":
        # Under 3GB there is not enough room for weights plus KV plus the OS.
        return 1024 if ram_gb < 3.0 else 2048
    if tier == "mid":
        return 4096
    # GPU tier: scale with the card.
    if vram_gb >= 24.0:
        return 16384
    if vram_gb >= 12.0:
        return 8192
    return 4096


def recommend_gpu_layers(tier, vram_gb, unified=False):
    """-1 means offload everything, 0 means pure CPU, a positive int is partial.

    Partial offload is for the awkward middle: a 4GB laptop GPU that can hold
    some layers but not a whole model.
    """
    if tier == "gpu":
        return -1
    if unified:
        # Metal on a small Mac. Offloading is still the right call; the pool
        # is shared either way.
        return -1
    if vram_gb >= 2.0:
        # Enough for a slice. 20 layers is a conservative guess that helps
        # without risking an allocation failure mid-load.
        return 20
    return 0


# ------------------------------------------------------------ RAM detection


def _detect_ram_gb():
    """Return (total_gb, available_gb). Falls back hard rather than raising."""
    try:
        import psutil  # optional, not a dependency

        vm = psutil.virtual_memory()
        return vm.total / BYTES_PER_GB, vm.available / BYTES_PER_GB
    except Exception:
        pass

    total = _total_ram_bytes_stdlib()
    if total:
        # Without psutil we cannot see "available" reliably. Assume 70% is
        # usable, which is pessimistic on a fresh boot and about right on a
        # laptop with a browser open.
        return total / BYTES_PER_GB, (total * 0.7) / BYTES_PER_GB

    # Total unknown. Claim 8GB so we land in "mid" rather than guessing big.
    logger.debug("Could not detect system RAM; assuming 8GB")
    return 8.0, 5.6


def _total_ram_bytes_stdlib():
    system = platform.system()

    # Linux and most Unixes.
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass

    if system == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=3, check=True,
            )
            return int(out.stdout.strip())
        except Exception:
            pass

    if system == "Windows":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys)
        except Exception:
            pass

    return None


# ------------------------------------------------------------ GPU detection


def _detect_gpu(os_name, ram_gb):
    """Return (vram_gb, unified, gpu_name)."""
    nvidia = _detect_nvidia()
    if nvidia:
        vram_gb, name = nvidia
        return vram_gb, False, name

    if os_name == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        # Apple Silicon. One pool. Metal can address roughly 70% of it by
        # default, so that is what we report as usable VRAM.
        return ram_gb * 0.7, True, "Apple Silicon"

    return 0.0, False, None


def _detect_nvidia():
    """Query nvidia-smi for the largest visible card. None if unavailable.

    Largest, not first: on a mixed box like Behemoth (V100 / TITAN / P100)
    the sensible default is the biggest card, not whichever one enumerates
    at index 0.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=memory.total,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
    except Exception:
        return None

    best = None
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            mib = float(parts[0])
        except ValueError:
            continue
        gb = mib / 1024.0
        name = parts[1]
        if best is None or gb > best[0]:
            best = (gb, name)
    return best


# ------------------------------------------- post-load architecture analysis


def inspect_architecture(model, ctx):
    """Read GGUF metadata off a loaded model and estimate its KV cache cost.

    Returns a dict, or None when the metadata is not exposed. This runs after
    load, so it cannot change the context that was already allocated. Its job
    is to tell the user when the ctx we picked was too aggressive for an old
    MHA model, and what to use next time.

    MHA models (TinyLlama 1.1B, early Llama) give every attention head its own
    K/V tables. GQA models (Qwen3, Gemma 3, Llama 3+) share them across groups
    and use 4-8x less memory at the same ctx.
    """
    meta = getattr(model, "metadata", None)
    if not meta:
        return None

    def find(suffix):
        for key, value in meta.items():
            if key.endswith(suffix):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
        return None

    heads = find(".attention.head_count")
    kv_heads = find(".attention.head_count_kv")
    layers = find(".block_count")
    embed = find(".embedding_length")
    key_len = find(".attention.key_length")

    if not heads or not layers:
        return None
    if not kv_heads:
        kv_heads = heads

    head_dim = key_len or (embed // heads if embed else None)
    if not head_dim:
        return None

    # K and V, f16, across every layer and every context slot.
    kv_bytes = 2 * layers * ctx * kv_heads * head_dim * 2
    kv_mb = kv_bytes / (1024 ** 2)
    is_gqa = kv_heads < heads

    return {
        "architecture": meta.get("general.architecture", "unknown"),
        "attention": "GQA" if is_gqa else "MHA",
        "is_gqa": is_gqa,
        "heads": heads,
        "kv_heads": kv_heads,
        "layers": layers,
        "head_dim": head_dim,
        "kv_cache_mb": round(kv_mb, 1),
        "train_ctx": find(".context_length"),
    }


def warn_if_ctx_too_aggressive(info, ctx, config, logger_=logger):
    """Log a warning when the auto-chosen ctx costs more KV cache than the
    tier can comfortably spare, and suggest a ctx that fits.

    Never raises. The model is already loaded; this is advice, not enforcement.
    """
    if not info:
        return None

    # How much memory we are willing to hand to the KV cache alone.
    if config.tier == "constrained":
        budget_mb = 400.0
    elif config.tier == "mid":
        budget_mb = 1024.0
    else:
        budget_mb = 4096.0

    kv_mb = info["kv_cache_mb"]
    if kv_mb <= budget_mb:
        logger_.debug(
            "Model uses %s, KV cache ~%.0fMB at ctx=%d",
            info["attention"], kv_mb, ctx,
        )
        return None

    # Scale ctx down to fit the budget, rounded to a power-of-two-ish step.
    safe_ctx = max(512, int((ctx * budget_mb / kv_mb) // 512) * 512)
    logger_.warning(
        "Model uses %s attention (%d heads, %d KV heads): KV cache is ~%.0fMB "
        "at ctx=%d, above the ~%.0fMB budget for tier=%s. It loaded, but it "
        "will be tight. Consider ctx=%d on this machine.",
        info["attention"], info["heads"], info["kv_heads"], kv_mb,
        ctx, budget_mb, config.tier, safe_ctx,
    )
    return safe_ctx
