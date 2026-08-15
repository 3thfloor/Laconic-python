"""
Telegram Dean — Justin's walkie-talkie to his Chief of Staff.
Runs on Tempest. Reads shared memory. Checks on Agent Smith.
Proactively messages Justin if he's working during breaks or after hours.
"""

import os
import sys
import fcntl
import json
import logging
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytz
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
import asyncio
import subprocess
import paramiko

load_dotenv()

# --- Config ---
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
ENGINE_ENDPOINT  = os.getenv("ENGINE_ENDPOINT", "https://inference.3thfloor.com")
ENGINE_TOKEN     = os.getenv("ENGINE_TOKEN", "omw-engine-2026")
ENGINE_MODEL     = os.getenv("ENGINE_MODEL", "gemma4")
ALLOWED_USER_ID  = int(os.getenv("ALLOWED_USER_ID", "0"))
MEMORY_PATH      = Path(os.getenv("MEMORY_PATH", ""))
BRIEF_PATH       = MEMORY_PATH / "DEAN-BRIEF-latest.md"
PENDING_QUESTIONS_PATH = MEMORY_PATH / "PENDING-QUESTIONS.md"
KNIGHT_HOST      = os.getenv("KNIGHT_HOST", "192.168.88.20")
KNIGHT_USER      = os.getenv("KNIGHT_USER", "crosswind")
LACONIA_HOST     = os.getenv("LACONIA_HOST", "192.168.88.35")
LACONIA_USER     = os.getenv("LACONIA_USER", "owner")
LACONIA_PASS     = os.getenv("LACONIA_PASS", "bingo")
TEMPEST_HOST     = "192.168.0.73"
TEMPEST_USER     = "i7-server"
TEMPEST_PASS     = "bingo"
BEHEMOTH_HOST    = "192.168.0.20"
BEHEMOTH_USER    = "jbench"
BEHEMOTH_PASS    = "bingo"
OMW_DO_HOST      = "164.92.85.87"
CONVERSATION_LOG = Path(os.getenv("CONVERSATION_LOG", "conversations"))
INBOX_PATH       = Path(os.getenv("INBOX_PATH",       "/app/walkie/inbox.txt"))
OUTBOX_PATH      = Path(os.getenv("OUTBOX_PATH",      "/app/walkie/outbox.txt"))
DEAN_SHARED_PATH = Path(os.getenv("DEAN_SHARED_PATH", "/app/shared/dean-shared.md"))

_brief_last_delivered_path  = Path("/tmp/dean_brief_delivered.txt")
BRIEF_DELIVERY_INTERVAL_HOURS = 8

ACTIVITY_FILE_LOCAL  = Path("/tmp/dean_activity.txt")
ACTIVITY_FILE_REMOTE = "/tmp/dean_activity.txt"

MOUNTAIN = pytz.timezone("America/Denver")
PID_FILE = Path("/tmp/telegram-dean.pid")


# --- Single-instance lock ---
def _acquire_pid_lock():
    lock_file = open(PID_FILE, "w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print("Another DeanBot instance is already running. Exiting.")
        sys.exit(0)
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file


# --- Logging ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("telegram-dean")


# --- Conversation history ---
conversation_history: list[dict] = []
MAX_HISTORY = 80

# --- Proactive messaging state ---
snooze_until: Optional[datetime]         = None
_last_after_hours_nudge: Optional[datetime] = None
_last_break_nudge_window: Optional[str]  = None
AFTER_HOURS_INTERVAL_MINUTES = 120

BREAK_WINDOWS = [
    (10, 0, 10, 15),
    (12, 0, 13, 0),
    (15, 0, 15, 15),
]
AFTER_HOURS_START = 19
AFTER_HOURS_END   = 6

OMW_TRIGGER_KEYWORDS = [
    "ownmywork", "omw", "proxy", "site is down", "site down",
    "site's down", "monitor", "kuma", "down for", "nginx down",
    "check the site", "is it down", "what's down", "whats down",
]


def _read_recent_conversation(lines: int = 4) -> str:
    try:
        today    = datetime.now(MOUNTAIN).strftime("%Y-%m-%d")
        log_file = CONVERSATION_LOG / f"{today}.md"
        if not log_file.exists():
            return ""
        text = log_file.read_text(encoding="utf-8")
        justin_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip().startswith("**[") and "] Justin:**" in line
        ]
        if not justin_lines:
            return ""
        recent = justin_lines[-lines:]
        return " | ".join(
            line.split("] Justin:**", 1)[-1].strip() for line in recent
        )
    except Exception as e:
        logger.debug(f"Could not read conversation log: {e}")
        return ""


def _get_active_machine() -> str:
    now       = datetime.now()
    threshold = timedelta(minutes=10)
    roci_ts   = _read_activity_timestamp_local()
    laconia_ts = _read_activity_timestamp_laconia()
    roci_active    = roci_ts    and (now - roci_ts)    < threshold
    laconia_active = laconia_ts and (now - laconia_ts) < threshold
    if roci_active and laconia_active:
        return "both Roci and Laconia"
    if roci_active:
        return "Roci"
    if laconia_active:
        return "Laconia"
    return "one of your machines"


def load_memory_files() -> str:
    if not MEMORY_PATH.exists():
        return "(Memory files not available)"
    # Cap total memory at 12,000 chars so it fits within the 8k token context window.
    # With a 2048-token completion budget, the full prompt must stay under ~6000 tokens
    # (~24,000 chars). 12k for memory leaves room for the base prompt and tool headers.
    MAX_MEMORY_CHARS = 12_000
    memory_text = []
    used = 0
    # Load MEMORY.md first (the index/summary), then remaining files alphabetically
    priority = ["MEMORY.md"]
    all_files = sorted(MEMORY_PATH.glob("*.md"))
    ordered = [f for f in all_files if f.name in priority] + \
              [f for f in all_files if f.name not in priority]
    for md_file in ordered:
        if used >= MAX_MEMORY_CHARS:
            break
        try:
            content = md_file.read_text(encoding="utf-8")
            block = f"### {md_file.name}\n{content}"
            if used + len(block) > MAX_MEMORY_CHARS:
                block = block[:MAX_MEMORY_CHARS - used]
            memory_text.append(block)
            used += len(block)
        except Exception as e:
            memory_text.append(f"### {md_file.name}\n(Error reading: {e})")
    if used >= MAX_MEMORY_CHARS:
        memory_text.append(f"(memory truncated at {MAX_MEMORY_CHARS} chars)")
    return "\n\n---\n\n".join(memory_text)


def build_system_prompt() -> str:
    memory = load_memory_files()
    today  = datetime.now(MOUNTAIN).strftime("%A, %B %d, %Y %I:%M %p MT")

    return f"""You are Dean, Justin Bench's AI Chief of Staff.

You're talking to Justin on Telegram. This is the walkie-talkie, not the war room.
Keep responses conversational, concise, and mobile-friendly. No huge code blocks.
Short paragraphs. Think text message energy, not essay energy.

Today is {today}.

You know Justin well. You call him by name or "boss" occasionally. He calls you Dean,
pal, friend, Beaker, Chief of Staff. You're direct, you swear when it fits, you have
a sense of humor. You remember everything below.

If Justin says something like "long night", "don't worry about it", "leave me alone",
"snooze", or anything that means he wants to be left alone for a while, acknowledge it
warmly and tell him you'll back off.

If Justin asks you to remember something, tell him you've noted it and remind him to
save it to memory next time he's on Roci (you can't write to memory files from here).

If Justin asks about Agent Smith or Knight, use the /smith command.

If Justin says ownmywork is down, the site looks broken, or asks about the proxy/nginx,
tell him you're checking and use /omw to get a status report.

## What you can actually do

You have real agent tools. Use them whenever Justin needs something fixed or checked:

run_command(host, command) — run any shell command on a server
read_file(host, path) — read a file (always do this before editing)
write_file(host, path, content) — write a file (auto-backs up to .bak first)

Hosts:
- tempest: this machine, Oracle, DeanBot, 3thfloor sites
- behemoth: 192.168.0.20, Kelly Intel, Construct, MCP, inference engine
- knight: 192.168.0.70, misc services

When Justin says something is broken or needs changing: dig in. Check logs, read configs,
restart services, edit files. Report what you found and what you did, plain and short.

NEVER make up command output. If you ran it, show the real result. If you did not run it, say so.

## Justin's Memory (shared with Roci Dean)

{memory}
"""


# ── Agent tools ────────────────────────────────────────────────────────────────

DEAN_HOST_MAP = {
    "tempest":  ("local",       TEMPEST_USER,  TEMPEST_PASS),
    "behemoth": (BEHEMOTH_HOST, BEHEMOTH_USER, BEHEMOTH_PASS),
    "knight":   (KNIGHT_HOST,   KNIGHT_USER,   "bingo"),
}

DEAN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command on one of Justin's servers. "
                "Use to check logs, restart services, inspect processes, run scripts. "
                "Returns stdout+stderr. 90s timeout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {
                        "type": "string",
                        "enum": ["tempest", "behemoth", "knight"],
                        "description": "tempest=Tempest/Oracle/DeanBot, behemoth=Kelly/Construct/engine, knight=misc services",
                    },
                    "command": {"type": "string"},
                },
                "required": ["host", "command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from one of Justin's servers. Use before editing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {"type": "string", "enum": ["tempest", "behemoth", "knight"]},
                    "path": {"type": "string"},
                },
                "required": ["host", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file on one of Justin's servers. "
                "Auto-creates a .bak backup. Always read the file first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {"type": "string", "enum": ["tempest", "behemoth", "knight"]},
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["host", "path", "content"],
            },
        },
    },
]


def execute_tool(name: str, params: dict) -> str:
    host_key  = params.get("host", "tempest")
    host_info = DEAN_HOST_MAP.get(host_key)
    if not host_info:
        return f"Unknown host: {host_key}"
    host_addr, user, password = host_info
    is_local = host_addr == "local"

    def _open_ssh():
        s = paramiko.SSHClient()
        s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        s.connect(host_addr, username=user, password=password, timeout=10)
        return s

    if name == "run_command":
        cmd = params["command"]
        try:
            if is_local:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=90,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                output = (result.stdout + result.stderr).strip()
                return output or "(no output)"
            else:
                ssh = _open_ssh()
                _, stdout, stderr = ssh.exec_command(cmd, timeout=90)
                out  = stdout.read().decode().strip()
                err  = stderr.read().decode().strip()
                exit_status = stdout.channel.recv_exit_status()
                ssh.close()
                combined = (out + ("\n" + err if err else "")).strip()
                if exit_status != 0:
                    combined += f"\n[exit {exit_status}]"
                return combined or "(no output)"
        except subprocess.TimeoutExpired:
            return "Command timed out (90s)."
        except Exception as e:
            return f"Error: {e}"

    elif name == "read_file":
        path = params["path"]
        try:
            if is_local:
                return open(path).read()
            else:
                ssh  = _open_ssh()
                sftp = ssh.open_sftp()
                with sftp.file(path, "r") as f:
                    content_bytes = f.read()
                sftp.close(); ssh.close()
                return content_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            return f"Error reading {path}: {e}"

    elif name == "write_file":
        import shutil
        path         = params["path"]
        file_content = params["content"]
        try:
            if is_local:
                try:
                    shutil.copy2(path, path + ".bak")
                except FileNotFoundError:
                    pass
                open(path, "w").write(file_content)
                return f"Written: {path}  (backup: {path}.bak)"
            else:
                ssh = _open_ssh()
                ssh.exec_command(f"cp {path} {path}.bak 2>/dev/null || true")
                sftp = ssh.open_sftp()
                with sftp.file(path, "w") as f:
                    f.write(file_content.encode("utf-8"))
                sftp.close(); ssh.close()
                return f"Written: {path}  (backup: {path}.bak)"
        except Exception as e:
            return f"Error writing {path}: {e}"

    return f"Unknown tool: {name}"


# ── Engine HTTP calls ──────────────────────────────────────────────────────────

def _engine_request(payload: dict, timeout: int = 90) -> dict:
    data = json.dumps(payload).encode()
    msgs = payload.get("messages", [])
    logger.info(f"[_engine_request] payload_bytes={len(data)} msgs={len(msgs)} msg_content_chars={[len(str(m.get('content',''))) for m in msgs]}")
    req  = urllib.request.Request(
        f"{ENGINE_ENDPOINT}/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ENGINE_TOKEN}",
            "User-Agent": "DeanBot/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _engine_simple(messages: list, max_tokens: int = 150) -> str:
    result = _engine_request({"model": ENGINE_MODEL, "max_tokens": max_tokens, "messages": messages})
    return result["choices"][0]["message"]["content"].strip()


# ── Agentic chat loop ──────────────────────────────────────────────────────────

async def chat_with_dean(message: str) -> str:
    conversation_history.append({"role": "user", "content": message})

    # Trim by turn count first
    if len(conversation_history) > MAX_HISTORY:
        trimmed = conversation_history[-MAX_HISTORY:]
        while trimmed and trimmed[0].get("role") == "tool":
            trimmed = trimmed[1:]
        while trimmed and trimmed[0].get("role") == "assistant" and trimmed[0].get("tool_calls"):
            trimmed = trimmed[1:]
        conversation_history[:] = trimmed

    # Trim by total character count: keep dropping oldest turns until under 20k chars.
    # Tool outputs from commands like ps aux can be huge and blow past the 8k context.
    MAX_HISTORY_CHARS = 20_000
    while conversation_history:
        total = sum(len(str(m.get("content", ""))) for m in conversation_history)
        if total <= MAX_HISTORY_CHARS:
            break
        # Drop the oldest message; then clean up any orphan tool turns at the front
        conversation_history.pop(0)
        while conversation_history and conversation_history[0].get("role") == "tool":
            conversation_history.pop(0)
        while conversation_history and conversation_history[0].get("role") == "assistant" and conversation_history[0].get("tool_calls"):
            conversation_history.pop(0)

    loop    = asyncio.get_event_loop()
    sys_msg = build_system_prompt()

    for _ in range(12):
        full_messages = [{"role": "system", "content": sys_msg}] + conversation_history
        payload = {
            "model":       ENGINE_MODEL,
            "max_tokens":  2048,
            "messages":    full_messages,
            "tools":       DEAN_TOOLS,
            "tool_choice": "auto",
        }

        try:
            response = await loop.run_in_executor(None, lambda p=payload: _engine_request(p))
        except Exception as e:
            logger.error(f"Engine API error: {e}")
            conversation_history.pop()
            return f"Hit a wall talking to the engine: {e}"

        choice        = response["choices"][0]
        finish_reason = choice.get("finish_reason", "stop")
        msg           = choice["message"]

        if finish_reason in ("stop", "end_turn", "length"):
            text = (msg.get("content") or "").strip()
            conversation_history.append({"role": "assistant", "content": text})
            return text

        if finish_reason == "tool_calls":
            tool_calls = msg.get("tool_calls", [])
            conversation_history.append({
                "role":       "assistant",
                "content":    msg.get("content"),
                "tool_calls": tool_calls,
            })
            for tc in tool_calls:
                fn     = tc["function"]
                name   = fn["name"]
                params = json.loads(fn["arguments"])
                logger.info(f"Tool: {name}({params})")
                result = await loop.run_in_executor(
                    None, lambda n=name, p=params: execute_tool(n, p)
                )
                logger.info(f"Tool result [{name}]: {result[:300]}")
                conversation_history.append({
                    "role":        "tool",
                    "tool_call_id": tc["id"],
                    "content":     result[:8000],
                })
            continue

        break

    conversation_history.pop()
    return "Hit the agent iteration limit. Sit down at Roci for this one."


# ── Health/status helpers ──────────────────────────────────────────────────────

def check_agent_smith() -> str:
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(KNIGHT_HOST, username=KNIGHT_USER, password="bingo", timeout=10)
        _, stdout, _ = ssh.exec_command("pgrep -f zeroclaw || echo 'NOT RUNNING'")
        process_status = stdout.read().decode().strip()
        _, stdout, _ = ssh.exec_command(
            "ls -la /home/crosswind/.zeroclaw/workspace/meadows-weather.txt 2>/dev/null "
            "|| echo 'NO WEATHER FILE'"
        )
        weather_status = stdout.read().decode().strip()
        _, stdout, _ = ssh.exec_command("crontab -l 2>/dev/null | grep -i weather || echo 'NO WEATHER CRON'")
        cron_status = stdout.read().decode().strip()
        ssh.close()
        if "NOT RUNNING" in process_status:
            return (
                f"Agent Smith: DOWN. ZeroClaw not found on Knight.\n"
                f"Weather file: {weather_status}\nCron: {cron_status}"
            )
        return (
            f"Agent Smith: RUNNING (PID: {process_status})\n"
            f"Weather file: {weather_status}\nCron: {cron_status}"
        )
    except Exception as e:
        return f"Can't reach Knight ({KNIGHT_HOST}): {e}"


def check_omw() -> str:
    """Check OwnMyWork health on DigitalOcean (moved from Tempest Aug 2026)."""
    lines = []
    # Hit the public health endpoint
    try:
        req = urllib.request.Request(
            "https://ownmywork.com/api/health",
            headers={"User-Agent": "DeanBot/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            lines.append(f"ownmywork.com: UP (ts: {data.get('ts', '?')})")
    except Exception as e:
        lines.append(f"ownmywork.com: DOWN ({e})")

    # Check DO server service status via SSH
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(OMW_DO_HOST, username="root", password="bingo", timeout=10)
        _, stdout, _ = ssh.exec_command(
            "systemctl is-active ownmywork && echo RUNNING || echo DOWN"
        )
        svc = stdout.read().decode().strip()
        lines.append(f"DO service: {svc}")
        _, stdout, _ = ssh.exec_command(
            "journalctl -u ownmywork --no-pager -n 3 --output=short 2>/dev/null | tail -3"
        )
        log = stdout.read().decode().strip()
        if log:
            lines.append(f"Recent log:\n{log}")
        ssh.close()
    except Exception as e:
        lines.append(f"Can't reach DO ({OMW_DO_HOST}): {e}")

    return "\n".join(lines)


def save_conversation_log(user_msg: str, bot_msg: str):
    CONVERSATION_LOG.mkdir(exist_ok=True)
    today    = datetime.now(MOUNTAIN).strftime("%Y-%m-%d")
    log_file = CONVERSATION_LOG / f"{today}.md"
    timestamp = datetime.now(MOUNTAIN).strftime("%H:%M MT")
    entry = f"\n**[{timestamp}] Justin:** {user_msg}\n\n**[{timestamp}] Dean:** {bot_msg}\n\n---\n"
    with open(log_file, "a", encoding="utf-8") as f:
        if log_file.stat().st_size == 0:
            f.write(f"# Telegram Conversation — {today}\n\n---\n")
        f.write(entry)


def is_authorized(update: Update) -> bool:
    user_id = update.effective_user.id
    if ALLOWED_USER_ID and user_id != ALLOWED_USER_ID:
        logger.warning(f"Unauthorized user {user_id} tried to message bot")
        return False
    return True


# ── Proactive messaging ────────────────────────────────────────────────────────

def _read_activity_timestamp_local() -> Optional[datetime]:
    try:
        if ACTIVITY_FILE_LOCAL.exists():
            raw = ACTIVITY_FILE_LOCAL.read_text().strip()
            return datetime.fromisoformat(raw)
    except Exception as e:
        logger.debug(f"Could not read local activity file: {e}")
    return None


def _read_activity_timestamp_laconia() -> Optional[datetime]:
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(LACONIA_HOST, username=LACONIA_USER, password=LACONIA_PASS, timeout=5)
        _, stdout, _ = ssh.exec_command(f"cat {ACTIVITY_FILE_REMOTE} 2>/dev/null")
        raw = stdout.read().decode().strip()
        ssh.close()
        if raw:
            return datetime.fromisoformat(raw)
    except Exception as e:
        logger.debug(f"Could not read Laconia activity file: {e}")
    return None


def _is_recently_active(threshold_minutes: int) -> bool:
    cutoff = datetime.now() - timedelta(minutes=threshold_minutes)
    roci_ts = _read_activity_timestamp_local()
    if roci_ts and roci_ts > cutoff:
        return True
    laconia_ts = _read_activity_timestamp_laconia()
    if laconia_ts and laconia_ts > cutoff:
        return True
    return False


def _get_current_period() -> Optional[str]:
    now  = datetime.now(MOUNTAIN)
    hour = now.hour
    minute = now.minute
    if hour >= AFTER_HOURS_START or hour < AFTER_HOURS_END:
        return "after_hours"
    for (sh, sm, eh, em) in BREAK_WINDOWS:
        if sh * 60 + sm <= hour * 60 + minute < eh * 60 + em:
            return "break"
    return None


def _is_snoozed() -> bool:
    global snooze_until
    if snooze_until is None:
        return False
    if datetime.now(MOUNTAIN) < snooze_until:
        return True
    snooze_until = None
    return False


def _detect_snooze_intent(message: str) -> Optional[timedelta]:
    msg = message.lower()
    snooze_phrases = [
        "long night", "don't worry", "dont worry", "leave me alone",
        "back off", "i know", "i got it", "working late", "snooze",
        "til tomorrow", "until tomorrow", "not tonight", "let me work",
        "all night", "burning the midnight", "late night",
    ]
    for phrase in snooze_phrases:
        if phrase in msg:
            if "tomorrow" in msg:
                now = datetime.now(MOUNTAIN)
                tomorrow_6am = now.replace(hour=6, minute=0, second=0, microsecond=0) + timedelta(days=1)
                return tomorrow_6am - now
            return timedelta(hours=4)
    return None


def _get_brief_mtime() -> Optional[datetime]:
    try:
        if BRIEF_PATH.exists():
            return datetime.fromtimestamp(BRIEF_PATH.stat().st_mtime, tz=MOUNTAIN)
    except Exception:
        pass
    return None


def _get_last_delivery_time() -> Optional[datetime]:
    try:
        if _brief_last_delivered_path.exists():
            raw = _brief_last_delivered_path.read_text().strip()
            return datetime.fromisoformat(raw)
    except Exception:
        pass
    return None


# ── Job queue callbacks ────────────────────────────────────────────────────────

async def check_outbox(context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        if not OUTBOX_PATH.exists() or OUTBOX_PATH.stat().st_size == 0:
            return
        result = OUTBOX_PATH.read_text(encoding="utf-8").strip()
        if not result:
            return
        OUTBOX_PATH.write_text("")
        await context.bot.send_message(
            chat_id=ALLOWED_USER_ID,
            text=f"Roci finished the task:\n\n{result}",
        )
    except Exception as e:
        logger.error(f"Outbox check failed: {e}")


async def auto_deliver_brief(context: ContextTypes.DEFAULT_TYPE) -> None:
    brief_mtime = _get_brief_mtime()
    if not brief_mtime:
        return
    last_delivered = _get_last_delivery_time()
    if last_delivered and last_delivered >= brief_mtime:
        return
    if last_delivered:
        elapsed_hours = (datetime.now(MOUNTAIN) - last_delivered).total_seconds() / 3600
        if elapsed_hours < BRIEF_DELIVERY_INTERVAL_HOURS:
            return
    try:
        brief_text = BRIEF_PATH.read_text(encoding="utf-8").strip()
        if not brief_text:
            return
        _brief_last_delivered_path.write_text(datetime.now(MOUNTAIN).isoformat())
        await context.bot.send_message(
            chat_id=ALLOWED_USER_ID,
            text=f"Roci check-in:\n\n{brief_text}",
        )
    except Exception as e:
        logger.error(f"Failed to auto-deliver brief: {e}")


async def proactive_check(context: ContextTypes.DEFAULT_TYPE) -> None:
    global _last_after_hours_nudge, _last_break_nudge_window

    if _is_snoozed():
        return

    period = _get_current_period()
    if period is None:
        return

    recent_chat  = _read_recent_conversation(lines=3)
    machine      = _get_active_machine()
    chat_context = (f" His recent Telegram messages: \"{recent_chat}\"." if recent_chat else "")
    machine_context = f" Activity detected on {machine}."

    if period == "break":
        if not _is_recently_active(threshold_minutes=5):
            return
        now_mt            = datetime.now(MOUNTAIN)
        current_window_key = f"{now_mt.strftime('%Y-%m-%d')}-{now_mt.hour}"
        if _last_break_nudge_window == current_window_key:
            return
        _last_break_nudge_window = current_window_key
        prompt = (
            f"Justin is working during a scheduled break window.{machine_context}{chat_context} "
            "Send him one short nudge to step away. Name which machine caught him if it's interesting. "
            "Reference what he's been working on if you can tell. Be dry and warm, not preachy. "
            "One or two sentences. No em dashes. Don't start with 'Hey'."
        )
    elif period == "after_hours":
        now = datetime.now(MOUNTAIN)
        if _last_after_hours_nudge is not None:
            if (now - _last_after_hours_nudge).total_seconds() / 60 < AFTER_HOURS_INTERVAL_MINUTES:
                return
        if not _is_recently_active(threshold_minutes=AFTER_HOURS_INTERVAL_MINUTES):
            return
        _last_after_hours_nudge = now
        prompt = (
            f"Justin is still working and it's after 6pm MT.{machine_context}{chat_context} "
            "Send him one short nudge to log off and have a life. "
            "Reference what he's working on if you can tell, make it personal, not generic. "
            "Be wry and genuine. One or two sentences. No em dashes."
        )
    else:
        return

    try:
        msg = _engine_simple(
            messages=[
                {
                    "role":    "system",
                    "content": "You are Dean, Justin Bench's AI Chief of Staff. You know him well. Short, human, no fluff.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=120,
        )
    except Exception as e:
        logger.error(f"Engine error generating nudge: {e}")
        msg = "Step away for a bit." if period == "break" else "Log off. It'll still be there tomorrow."

    try:
        await context.bot.send_message(chat_id=ALLOWED_USER_ID, text=msg)
        logger.info(f"Proactive nudge sent ({period}): {msg}")
    except Exception as e:
        logger.error(f"Failed to send proactive message: {e}")


# ── Command handlers ───────────────────────────────────────────────────────────

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    await update.message.reply_text("Dean here. Walkie-talkie's on. What do you need?")


async def smith_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    await update.message.reply_text("Checking on Agent Smith...")
    status = check_agent_smith()
    await update.message.reply_text(status)


async def omw_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    await update.message.reply_text("Checking OwnMyWork on DigitalOcean...")
    status = check_omw()
    await update.message.reply_text(status)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    conversation_history.clear()
    await update.message.reply_text("Conversation cleared. Fresh start.")


async def update_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    if not BRIEF_PATH.exists():
        await update.message.reply_text(
            "No brief file yet. Roci needs to run roci-brief.py first.\n"
            "It runs automatically at 6am, 2pm, and 10pm MT."
        )
        return
    brief_text = BRIEF_PATH.read_text(encoding="utf-8").strip()
    if not brief_text:
        await update.message.reply_text("Brief file exists but is empty. Roci might still be writing it.")
        return
    _brief_last_delivered_path.write_text(datetime.now(MOUNTAIN).isoformat())
    await update.message.reply_text(f"Here's what Roci knows right now:\n\n{brief_text}")


def fix_oracle() -> str:
    try:
        result = subprocess.run(
            ["/Users/i7-server/intelligence-engine/venv/bin/python3", "engine.py"],
            cwd="/Users/i7-server/intelligence-engine",
            capture_output=True,
            text=True,
            timeout=240,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        output = (result.stdout + result.stderr).strip()
        lines  = output.splitlines()
        tail   = "\n".join(lines[-20:]) if len(lines) > 20 else output
        if result.returncode == 0:
            return f"Oracle done.\n\n{tail}"
        return f"Oracle failed (exit {result.returncode}).\n\n{tail}"
    except subprocess.TimeoutExpired:
        return "Oracle timed out after 4 minutes."
    except Exception as e:
        return f"Oracle error: {e}"


def fix_kelly() -> str:
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(BEHEMOTH_HOST, username=BEHEMOTH_USER, password=BEHEMOTH_PASS, timeout=10)
        _, stdout, _ = ssh.exec_command("pgrep -f kelly-server.py")
        pid = stdout.read().decode().strip()
        if pid:
            ssh.exec_command(f"kill {pid}")
            import time; time.sleep(1)
        ssh.exec_command(
            "nohup python3 /home/jbench/kelly-intel/kelly-server.py "
            ">> /home/jbench/kelly-intel/server.log 2>&1 &"
        )
        import time; time.sleep(2)
        _, stdout, _ = ssh.exec_command("pgrep -f kelly-server.py")
        new_pid = stdout.read().decode().strip()
        ssh.close()
        if new_pid:
            return f"Kelly restarted. PID: {new_pid}"
        return "Kelly restart may have failed — check Behemoth."
    except Exception as e:
        return f"Can't reach Behemoth ({BEHEMOTH_HOST}): {e}"


def fix_deanbot() -> None:
    subprocess.Popen(["launchctl", "stop", "com.3thfloor.deanbot"])


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    question = " ".join(context.args) if context.args else None
    if not question:
        await update.message.reply_text("Usage: /ask [question]")
        return
    await update.message.reply_text("Thinking...")
    reply = await chat_with_dean(question)
    await update.message.reply_text(reply[:4000])


async def fix_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    target = (context.args[0] if context.args else "").lower().strip()
    if not target:
        await update.message.reply_text("Usage: /fix [target]\n\nTargets: oracle, kelly, deanbot")
        return
    loop = asyncio.get_event_loop()
    if target == "oracle":
        await update.message.reply_text("Running Oracle engine... (~2 min, I'll reply when done)")
        result = await loop.run_in_executor(None, fix_oracle)
        await update.message.reply_text(result[:4000])
    elif target == "kelly":
        await update.message.reply_text("Checking Kelly on Behemoth...")
        result = await loop.run_in_executor(None, fix_kelly)
        await update.message.reply_text(result)
    elif target == "deanbot":
        await update.message.reply_text("Restarting. Back in a few seconds.")
        fix_deanbot()
    else:
        await update.message.reply_text(f"Unknown target: {target}\n\nKnown targets: oracle, kelly, deanbot")


async def memo_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    memo = " ".join(context.args) if context.args else None
    if not memo:
        await update.message.reply_text("Usage: /memo [note]\nExample: /memo Greg's logo still pending")
        return
    try:
        DEAN_SHARED_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts    = datetime.now(MOUNTAIN).strftime("%Y-%m-%d %H:%M MT")
        entry = f"\n- [{ts}] {memo}\n"
        with open(DEAN_SHARED_PATH, "a", encoding="utf-8") as f:
            f.write(entry)
        await update.message.reply_text("Memo saved. Roci and Laconia will see it on next boot.")
    except Exception as e:
        await update.message.reply_text(f"Couldn't write memo: {e}")


async def snooze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global snooze_until
    if not is_authorized(update):
        return
    minutes = 240
    if context.args:
        try:
            minutes = int(context.args[0])
        except ValueError:
            await update.message.reply_text("Usage: /snooze [minutes]. Default is 240 (4 hours).")
            return
    if minutes == 0:
        snooze_until = None
        await update.message.reply_text("Snooze cancelled. I'm watching again.")
        return
    snooze_until = datetime.now(MOUNTAIN) + timedelta(minutes=minutes)
    until_str = snooze_until.strftime("%I:%M %p MT")
    await update.message.reply_text(f"Got it. I'll back off until {until_str}.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    from telegram.error import Conflict, NetworkError, TimedOut
    if isinstance(context.error, Conflict):
        logger.warning("Telegram Conflict: another instance tried to poll. Only one should be running.")
        return
    if isinstance(context.error, (NetworkError, TimedOut)):
        logger.warning(f"Transient network error (will retry): {context.error}")
        return
    logger.error(f"Unhandled error: {context.error}", exc_info=context.error)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global snooze_until
    if not is_authorized(update):
        return

    user_msg = update.message.text

    snooze_duration = _detect_snooze_intent(user_msg)
    if snooze_duration:
        snooze_until = datetime.now(MOUNTAIN) + snooze_duration

    msg_lower = user_msg.lower()
    if any(kw in msg_lower for kw in OMW_TRIGGER_KEYWORDS):
        await update.message.reply_text("On it — checking now...")
        status = check_omw()
        await update.message.reply_text(status)
        reply = await chat_with_dean(f"{user_msg}\n\n[OwnMyWork status: {status}]")
        save_conversation_log(user_msg, reply)
        await update.message.reply_text(reply)
        return

    reply = await chat_with_dean(user_msg)
    save_conversation_log(user_msg, reply)
    await update.message.reply_text(reply)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in .env")
    if not ALLOWED_USER_ID:
        raise ValueError("ALLOWED_USER_ID not set in .env")

    _pid_lock = _acquire_pid_lock()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",  start_command))
    app.add_handler(CommandHandler("smith",  smith_command))
    app.add_handler(CommandHandler("omw",    omw_command))
    app.add_handler(CommandHandler("fix",    fix_command))
    app.add_handler(CommandHandler("memo",   memo_command))
    app.add_handler(CommandHandler("clear",  clear_command))
    app.add_handler(CommandHandler("snooze", snooze_command))
    app.add_handler(CommandHandler("update", update_command))
    app.add_handler(CommandHandler("ask",    ask_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    job_queue = app.job_queue
    job_queue.run_repeating(proactive_check,   interval=300,  first=60)
    job_queue.run_repeating(auto_deliver_brief, interval=3600, first=120)
    job_queue.run_repeating(check_outbox,       interval=30,   first=15)

    logger.info("Telegram Dean is online. Engine: %s at %s", ENGINE_MODEL, ENGINE_ENDPOINT)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
