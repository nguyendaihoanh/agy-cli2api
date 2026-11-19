"""Runtime configuration for agy2api."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from pathlib import Path

logger = logging.getLogger("agy2api.config")

def _strip_inline_comment(val: str) -> str:
    """Strip outer matching quotes or unquoted inline '#' comments."""
    val = val.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
        return val[1:-1]
    if "#" in val:
        # Treat '#' preceded by whitespace (or at index 0) as an inline comment.
        for idx, ch in enumerate(val):
            if ch == "#" and (idx == 0 or val[idx - 1].isspace()):
                val = val[:idx].strip()
                break
    return val


def _load_dotenv() -> None:
    """Load variables from .env file into os.environ if present, without external dependencies."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, val = line.partition("=")
        key = key.strip()
        val = _strip_inline_comment(val)
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()


def _resolve_agy() -> str:
    """Find the agy executable regardless of the launching shell's PATH.

    Order: explicit AGY_PATH env > on PATH > known per-user install location.
    Falls back to the bare name so the original FileNotFoundError still surfaces
    a clear message if nothing is found.
    """
    explicit = os.getenv("AGY_PATH")
    if explicit:
        return explicit
    found = shutil.which("agy")
    if found:
        return found
    candidates = [
        Path.home() / ".local" / "bin" / "agy",
        Path.home() / "bin" / "agy",
        Path("/usr/local/bin/agy"),
        Path("/opt/homebrew/bin/agy"),
        Path.home() / "AppData" / "Local" / "agy" / "bin" / "agy.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "agy"


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str, default: List[str]) -> List[str]:
    value = os.getenv(name)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


# Fallback mapping from clean OpenAI-style model ids to the exact display labels
# that `agy models` prints and `agy --model` accepts verbatim.
# At runtime, `discover_model_catalog()` queries `agy models` dynamically so new
# models released server-side appear automatically without updating this map.
MODEL_MAP: Dict[str, str] = {
    # Gemini 4 Neon
    "gemini-4-neon-xhigh": "Gemini 4 Neon (XHigh)",
    "gemini-4-neon-high": "Gemini 4 Neon (High)",
    "gemini-4-neon-medium": "Gemini 4 Neon (Medium)",
    "gemini-4-neon-low": "Gemini 4 Neon (Low)",
    # Gemini 4 Argon
    "gemini-4-argon-medium": "Gemini 4 Argon (Medium)",
    "gemini-4-argon-high": "Gemini 4 Argon (High)",
    "gemini-4-argon-low": "Gemini 4 Argon (Low)",
    # Gemini 4 Hydrogen
    "gemini-4-hydrogen-low": "Gemini 4 Hydrogen (Low)",
    "gemini-4-hydrogen-medium": "Gemini 4 Hydrogen (Medium)",
    "gemini-4-hydrogen-high": "Gemini 4 Hydrogen (High)",
    "gemini-4-hydrogen-xhigh": "Gemini 4 Hydrogen (XHigh)",
    # Orchestrators
    "gemini-pro-orchestrator-neon": "Orchestrator (Neon)",
    "gemini-pro-orchestrator-g1-pro": "Orchestrator (Argon)",
    # Gemini 3.8 Flash
    "gemini-3.8-flash-high": "Gemini 3.8 Flash (High)",
    "gemini-3.8-flash-medium": "Gemini 3.8 Flash (Medium)",
    "gemini-3.8-flash-low": "Gemini 3.8 Flash (Low)",
    # Claude 5.5
    "claude-opus-5-5-low": "Claude Opus 5.5 (Low)",
    "claude-opus-5-5-medium": "Claude Opus 5.5 (Medium)",
    "claude-opus-5-5-high": "Claude Opus 5.5 (High)",
    "claude-sonnet-5-5-low": "Claude Sonnet 5.5 (Low)",
    "claude-sonnet-5-5-medium": "Claude Sonnet 5.5 (Medium)",
    "claude-sonnet-5-5-high": "Claude Sonnet 5.5 (High)",
    # Open-weights hosted on Antigravity
    "gpt-oss-120b-medium": "GPT-OSS 120B (Medium)",
    # Backward compatibility aliases
    "gemini-3.5-flash": "Gemini 3.8 Flash (Medium)",
    "gemini-3.1-pro": "Gemini 4 Argon (High)",
    "claude-sonnet-4.6": "Claude Sonnet 5.5 (Medium)",
    "claude-opus-4.6": "Claude Opus 5.5 (Medium)",
    "gpt-oss-120b": "GPT-OSS 120B (Medium)",
}

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{1,127}$")
_BULLET_PREFIX_RE = re.compile(r"^(?:[*•-]\s+)+")
_IGNORED_PREFIXES = (
    "fetching ",
    "available models",
    "loading ",
    "usage:",
    "error:",
    "warning:",
)

_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE_TS: float = 0.0
_MODEL_CACHE_TTL_ACTIVE: float = 0.0
_MODEL_CACHE_DATA: Dict[str, str] = {}
_FALLBACK_BACKOFF_SEC: float = 60.0


def parse_agy_models_output(raw: str) -> Dict[str, str]:
    """Parse `agy models` stdout into an ordered `{model_id: display_label}` map.

    Supports:
    - Tab-separated two-column lines (`gemini-4-neon-xhigh\\tGemini 4 Neon (XHigh)`).
    - Multi-space separated two-column lines (`gemini-4-neon-xhigh   Gemini 4 Neon (XHigh)`).
    - Known static display-label-only lines.
    - Bare model slug lines containing `-` or `.`.
    """
    label_to_id = {v.lower(): k for k, v in MODEL_MAP.items()}
    discovered: Dict[str, str] = {}

    for line in raw.splitlines():
        cleaned = _BULLET_PREFIX_RE.sub("", line.strip()).strip()
        if not cleaned or set(cleaned) <= {"-", "=", "_", " "}:
            continue
        low = cleaned.lower()
        if any(low.startswith(prefix) for prefix in _IGNORED_PREFIXES):
            continue

        # Primary format: `model-id\tDisplay Label` or `model-id   Display Label`
        parts: List[str] = []
        if "\t" in cleaned:
            parts = [p.strip() for p in cleaned.split("\t", 1)]
        else:
            split_ws = re.split(r"\s{2,}", cleaned, maxsplit=1)
            if len(split_ws) == 2:
                parts = [p.strip() for p in split_ws]

        if len(parts) == 2 and parts[0] and parts[1] and _MODEL_ID_RE.match(parts[0]):
            discovered.setdefault(parts[0], parts[1])
            continue

        # Fallback: display label matching a known static entry
        if low in label_to_id:
            mid = label_to_id[low]
            discovered.setdefault(mid, MODEL_MAP[mid])
            continue

        # Fallback: single-column model slug (e.g. `gemini-5-future-high`)
        if ("-" in cleaned or "." in cleaned) and _MODEL_ID_RE.match(cleaned):
            discovered.setdefault(cleaned, MODEL_MAP.get(cleaned, cleaned))

    return discovered


def clear_model_cache() -> None:
    """Clear the cached live model catalog."""
    global _MODEL_CACHE_TS, _MODEL_CACHE_TTL_ACTIVE, _MODEL_CACHE_DATA
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE_TS = 0.0
        _MODEL_CACHE_TTL_ACTIVE = 0.0
        _MODEL_CACHE_DATA = {}


def _run_agy_models_cli(agy_path: str, timeout: float, env: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Invoke `agy models` safely without blocking on stdin or console windows."""
    if not agy_path or not (shutil.which(agy_path) or os.path.exists(agy_path)):
        return None
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    try:
        proc = subprocess.run(
            [agy_path, "models"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
            creationflags=creationflags,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            parsed = parse_agy_models_output(proc.stdout)
            if parsed:
                return parsed
        else:
            logger.debug("agy models exited with rc=%s", proc.returncode)
    except Exception as exc:
        logger.debug("agy models discovery failed: %s", exc)
    return None


def discover_model_catalog(force_refresh: bool = False) -> Dict[str, str]:
    """Return the active `{model_id: display_label}` catalog with TTL caching.

    Queries `agy models` dynamically when `AGY_MODEL_DISCOVERY` is enabled; falls
    back cleanly to `MODEL_MAP` if the CLI is missing, fails, or times out.
    """
    global _MODEL_CACHE_TS, _MODEL_CACHE_TTL_ACTIVE, _MODEL_CACHE_DATA

    if not settings.model_discovery:
        return dict(MODEL_MAP)

    now = time.monotonic()
    with _MODEL_CACHE_LOCK:
        if not force_refresh and _MODEL_CACHE_DATA and (now - _MODEL_CACHE_TS) < _MODEL_CACHE_TTL_ACTIVE:
            return dict(_MODEL_CACHE_DATA)

    parsed = _run_agy_models_cli(
        settings.agy_path,
        timeout=settings.model_discovery_timeout,
        env=settings.agy_env(),
    )

    now_after = time.monotonic()
    with _MODEL_CACHE_LOCK:
        if parsed:
            _MODEL_CACHE_DATA = dict(parsed)
            _MODEL_CACHE_TS = now_after
            _MODEL_CACHE_TTL_ACTIVE = max(1.0, settings.model_cache_ttl)
            logger.info("Discovered %d live models from agy CLI", len(_MODEL_CACHE_DATA))
            return dict(_MODEL_CACHE_DATA)

        # CLI unavailable/timed out: keep previous live cache if any, else static fallback,
        # with short backoff so repeated /v1/models calls don't stall on every request.
        fallback = dict(_MODEL_CACHE_DATA) if _MODEL_CACHE_DATA else dict(MODEL_MAP)
        _MODEL_CACHE_DATA = fallback
        _MODEL_CACHE_TS = now_after
        _MODEL_CACHE_TTL_ACTIVE = min(_FALLBACK_BACKOFF_SEC, max(1.0, settings.model_cache_ttl))
        return dict(fallback)


def get_exposed_models(force_refresh: bool = False) -> List[str]:
    """Return model IDs to expose via `/v1/models`.

    Honors an explicit `AGY_MODELS` override when configured; otherwise uses
    dynamic discovery (falling back to static `MODEL_MAP`).
    """
    if settings.models_override is not None:
        return list(settings.models_override)
    return list(discover_model_catalog(force_refresh=force_refresh).keys())


def resolve_model(model: Optional[str]) -> Optional[str]:
    """Translate an exposed model id to the agy label.

    Consults dynamically discovered labels first, then static `MODEL_MAP` aliases,
    and passes unknown values through unchanged so new/raw `agy` IDs work directly.
    """
    if not model:
        return None
    with _MODEL_CACHE_LOCK:
        live_label = _MODEL_CACHE_DATA.get(model)
    if live_label:
        return live_label
    return MODEL_MAP.get(model, model)


@dataclass(frozen=True)
class Settings:
    agy_path: str = _resolve_agy()
    api_key: str = os.getenv("AGY2API_KEY", os.getenv("API_KEY", "")).strip()
    require_auth: bool = _bool_env("AGY2API_REQUIRE_AUTH", True)
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "7862"))
    request_timeout: float = float(os.getenv("AGY_TIMEOUT", "300"))
    poll_interval: float = float(os.getenv("AGY_POLL_INTERVAL", "0.25"))
    chunk_size: int = int(os.getenv("AGY2API_CHUNK_SIZE", "10"))
    stream_delay: float = float(os.getenv("AGY2API_STREAM_DELAY", "0.03"))
    expose_reasoning: bool = _bool_env("AGY2API_EXPOSE_REASONING", True)
    # Safety: refuse to bind a non-loopback address unless explicitly allowed.
    # Exposing this endpoint shares your personal Google quota with others.
    allow_remote: bool = _bool_env("AGY2API_ALLOW_REMOTE", False)
    # Concurrent agy runs. Safe now that each run reads its OWN conversation DB
    # (resolved from the per-run --log-file conversation id), not the newest by
    # mtime. Kept modest to stay human-paced / avoid abuse-like quota bursts.
    max_concurrency: int = int(os.getenv("AGY2API_MAX_CONCURRENCY", "3"))
    # Stateless wrapper: full history is resent in each prompt, so a finished
    # run's conversation DB + brain/ dir are throwaway. Delete them after a
    # successful read to keep local session files from piling up. Set false to
    # retain them for debugging. (Does NOT reduce server-side conversation count.)
    cleanup_db: bool = _bool_env("AGY2API_CLEANUP_DB", True)
    # Keep agy's `last_check.timestamp` fresh before each run so agy's background
    # auto-updater never thinks a check is due. This stops the `agy --bg-updater`
    # child process from spawning (it flashes its own console window despite our
    # CREATE_NO_WINDOW) AND prevents agy silently auto-updating underneath us
    # (which could change the DB/protobuf layout our parser depends on). Set
    # false to let agy auto-update normally.
    suppress_autoupdate: bool = _bool_env("AGY2API_SUPPRESS_AUTOUPDATE", True)
    # Experimental: keep a persistent agy conversation per chat and send only the
    # new turn each request (instead of resending the full history). Smaller
    # per-turn payloads finish faster and are less likely to hit an upstream
    # connection timeout on long chats. Opt-in; changes the cleanup lifecycle.
    stateful: bool = _bool_env("AGY2API_STATEFUL", False)
    max_sessions: int = int(os.getenv("AGY2API_MAX_SESSIONS", "200"))
    # When stateful, run agy inside its OWN home directory so our startup/exit
    # wipes never touch the user's real ~/.gemini (TUI conversations stay safe).
    # agy honors USERPROFILE for its home root, so we point it here. Default is a
    # `stateful_home/` folder INSIDE this project (gitignored), so it's easy to
    # find and manage. Set AGY2API_STATEFUL_HOME to override.
    stateful_home: Path = Path(
        os.getenv(
            "AGY2API_STATEFUL_HOME",
            str(Path(__file__).resolve().parent / "stateful_home"),
        )
    )
    conversations_dir: Path = Path(
        os.getenv(
            "AGY_CONVERSATIONS_DIR",
            str(Path.home() / ".gemini" / "antigravity-cli" / "conversations"),
        )
    )
    workdir: Path = Path(
        os.getenv("AGY_WORKDIR", str(Path(__file__).resolve().parent))
    )
    model_discovery: bool = _bool_env("AGY_MODEL_DISCOVERY", True)
    model_cache_ttl: float = float(os.getenv("AGY_MODEL_CACHE_TTL", "1800"))
    model_discovery_timeout: float = float(os.getenv("AGY_MODEL_DISCOVERY_TIMEOUT", "8.0"))
    models_override: Optional[List[str]] = None
    models: Optional[List[str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        raw_models = os.getenv("AGY_MODELS", "").strip()
        override = [item.strip() for item in raw_models.split(",") if item.strip()] if raw_models else None
        object.__setattr__(self, "models_override", override)
        object.__setattr__(
            self,
            "models",
            override if override is not None else list(MODEL_MAP.keys()),
        )
        # In stateful mode, redirect conversations_dir into the isolated home
        # so agy reads/writes only there (and our sweeps stay self-contained).
        if self.stateful:
            sandbox_conv = self.stateful_home / ".gemini" / "antigravity-cli" / "conversations"
            object.__setattr__(self, "conversations_dir", sandbox_conv)

    def agy_env(self) -> Dict[str, str]:
        """Environment dict to pass to the agy subprocess.

        In stateful mode, set USERPROFILE to the isolated home so agy's whole
        data tree (conversations, brain, auth) lives there, fully separated
        from the user's real ~/.gemini. In stateless mode, pass the current
        environment unchanged.
        """
        if not self.stateful:
            return os.environ.copy()
        env = os.environ.copy()
        env["USERPROFILE"] = str(self.stateful_home)
        env["HOME"] = str(self.stateful_home)
        return env


settings = Settings()


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", ""}


def is_loopback_host(host: str) -> bool:
    return host.strip().lower() in _LOOPBACK_HOSTS
