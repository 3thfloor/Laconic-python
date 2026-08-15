import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelManager:
    """
    Manages the local model registry for the 3th Floor AI engine.

    Registry lives at {models_dir}/registry.json and tracks all known
    models by alias. Loaded state is checked live against the engine.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._registry_path: Path = engine.models_dir / "registry.json"

    # ------------------------------------------------------------------
    # Internal registry I/O
    # ------------------------------------------------------------------

    def _read(self) -> List[Dict[str, Any]]:
        """Load registry.json and return its entries. Return [] if missing or corrupt."""
        if not self._registry_path.exists():
            return []
        try:
            with open(self._registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return []
            return data
        except (json.JSONDecodeError, OSError):
            return []

    def _write(self, entries: List[Dict[str, Any]]) -> None:
        """Persist entries to registry.json with 2-space indent."""
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._registry_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list(self) -> List[Dict[str, Any]]:
        """Return all registry entries."""
        return self._read()

    def add(
        self,
        alias: str,
        path: str,
        ctx: int = 4096,
    ) -> Dict[str, Any]:
        """
        Add a model to the registry, or update it if the alias already exists.

        Resolves the path to an absolute location, computes size_mb from the
        file size on disk, and stores added_at as a Unix timestamp.

        Returns the entry dict.
        """
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Model file not found: {resolved}")

        size_mb = round(resolved.stat().st_size / (1024 * 1024), 2)

        entry: Dict[str, Any] = {
            "alias": alias,
            "path": str(resolved),
            "ctx": ctx,
            "added_at": time.time(),
            "size_mb": size_mb,
        }

        entries = self._read()
        entries = [e for e in entries if e.get("alias") != alias]
        entries.append(entry)
        self._write(entries)
        return entry

    def remove(self, alias: str) -> None:
        """
        Remove a model from the registry by alias.

        Raises KeyError if the alias is not found.
        """
        entries = self._read()
        filtered = [e for e in entries if e.get("alias") != alias]
        if len(filtered) == len(entries):
            raise KeyError(f"No model registered with alias: {alias!r}")
        self._write(filtered)

    def info(self, alias: str) -> Dict[str, Any]:
        """
        Return the registry entry for the given alias.

        Adds a "loaded" key (bool) reflecting whether the model is currently
        loaded in the engine. Raises KeyError if not in the registry.
        """
        entries = self._read()
        for entry in entries:
            if entry.get("alias") == alias:
                result = dict(entry)
                result["loaded"] = alias in self._engine._loaded
                return result
        raise KeyError(f"No model registered with alias: {alias!r}")

    def download(
        self,
        repo_id: str,
        filename: Optional[str] = None,
        quantization: str = "Q4_K_M",
    ) -> str:
        """
        Download a GGUF file from HuggingFace and register it automatically.

        If filename is not provided, lists the repo files and picks the first
        one whose name contains the quantization string (case-insensitive).

        The model is saved to engine.models_dir and registered under an alias
        derived from the filename stem: lowercased, non-alphanumeric chars
        replaced with dashes, leading/trailing dashes stripped.

        Returns the local path as a string.

        Requires huggingface_hub. If missing, raises ImportError with an
        install hint.
        """
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for model downloads. "
                'Install it with: pip install "thirthfloor[hf]"'
            ) from exc

        if filename is None:
            repo_files = list(list_repo_files(repo_id))
            gguf_files = [
                f for f in repo_files
                if f.lower().endswith(".gguf")
                and quantization.lower() in f.lower()
            ]
            if not gguf_files:
                all_gguf = [f for f in repo_files if f.lower().endswith(".gguf")]
                if not all_gguf:
                    raise ValueError(
                        f"No GGUF files found in repo: {repo_id}"
                    )
                raise ValueError(
                    f"No GGUF file matching quantization {quantization!r} in "
                    f"{repo_id}. Available: {all_gguf}"
                )
            filename = gguf_files[0]

        local_path_str = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(self._engine.models_dir),
        )

        local_path = Path(local_path_str)

        # Derive a clean alias from the filename stem.
        stem = local_path.stem
        alias_chars = []
        for ch in stem.lower():
            if ch.isalnum():
                alias_chars.append(ch)
            else:
                alias_chars.append("-")
        alias = "".join(alias_chars).strip("-")
        # Collapse consecutive dashes.
        while "--" in alias:
            alias = alias.replace("--", "-")

        self.add(alias, str(local_path))
        return str(local_path)
