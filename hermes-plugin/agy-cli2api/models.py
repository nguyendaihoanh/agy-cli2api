"""Model catalog, aliases, effort mapping, and dynamic discovery for agy-cli2api."""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("agy_cli2api.models")

MODEL_MAP: Dict[str, str] = {
    "gemini-4-argon-medium": "Gemini 4 Argon (Medium)",
    "gemini-4-argon-high": "Gemini 4 Argon (High)",
    "gemini-4-argon-low": "Gemini 4 Argon (Low)",
    "gemini-4-hydrogen-low": "Gemini 4 Hydrogen (Low)",
    "gemini-4-hydrogen-medium": "Gemini 4 Hydrogen (Medium)",
    "gemini-4-hydrogen-high": "Gemini 4 Hydrogen (High)",
    "gemini-4-hydrogen-xhigh": "Gemini 4 Hydrogen (XHigh)",
    "gemini-pro-orchestrator-g1-pro": "Orchestrator (Argon)",
    "gemini-3.8-flash-high": "Gemini 3.8 Flash (High)",
    "gemini-3.8-flash-medium": "Gemini 3.8 Flash (Medium)",
    "gemini-3.8-flash-low": "Gemini 3.8 Flash (Low)",
    "gemini-3.7-flash-high": "Gemini 3.7 Flash (High)",
    "gemini-3.7-flash-medium": "Gemini 3.7 Flash (Medium)",
    "gemini-3.7-flash-low": "Gemini 3.7 Flash (Low)",
    "gemini-3.1-pro-high": "Gemini 3.1 Pro (High)",
    "gemini-3.1-pro-low": "Gemini 3.1 Pro (Low)",
    "claude-opus-5-5-low": "Claude Opus 5.5 (Low)",
    "claude-opus-5-5-medium": "Claude Opus 5.5 (Medium)",
    "claude-opus-5-5-high": "Claude Opus 5.5 (High)",
    "claude-sonnet-5-5-low": "Claude Sonnet 5.5 (Low)",
    "claude-sonnet-5-5-medium": "Claude Sonnet 5.5 (Medium)",
    "claude-sonnet-5-5-high": "Claude Sonnet 5.5 (High)",
    "claude-sonnet-4-6": "Claude Sonnet 4.6 (Thinking)",
    "claude-opus-4-6-thinking": "Claude Opus 4.6 (Thinking)",
    "gpt-oss-120b-medium": "GPT-OSS 120B (Medium)",
}

MODEL_ALIASES: Dict[str, str] = {
    "hydrogen": "gemini-4-hydrogen-xhigh",
    "hydrogen-xhigh": "gemini-4-hydrogen-xhigh",
    "gemini-4-hydrogen": "gemini-4-hydrogen-xhigh",
    "argon": "gemini-4-argon-high",
    "gemini-4-argon": "gemini-4-argon-high",
    "flash": "gemini-3.8-flash-medium",
    "gemini-3.8-flash": "gemini-3.8-flash-medium",
    "gemini-3.5-flash": "gemini-3.8-flash-medium",
    "pro": "gemini-4-argon-high",
    "gemini-3.1-pro": "gemini-3.1-pro-high",
    "opus": "claude-opus-5-5-high",
    "claude-opus-5.5": "claude-opus-5-5-high",
    "claude-opus-4.6": "claude-opus-4-6-thinking",
    "sonnet": "claude-sonnet-5-5-medium",
    "claude-sonnet-5.5": "claude-sonnet-5-5-medium",
    "claude-sonnet-4.6": "claude-sonnet-4-6",
    "gpt-oss-120b": "gpt-oss-120b-medium",
}

DEFAULT_MODEL = "gemini-3.8-flash-medium"
DEFAULT_CONTEXT_LENGTH = 239_000
_CACHE_TTL_SECONDS = 300.0
_CACHE_LOCK = threading.Lock()
_CACHED_MODELS: Tuple[float, List[str]] = (0.0, [])


def resolve_model_id(model: Optional[str]) -> str:
    """Normalize aliases to a canonical exposed model id."""
    cleaned = (model or "").strip()
    if not cleaned:
        return DEFAULT_MODEL
    return MODEL_ALIASES.get(cleaned.lower(), cleaned)


def resolve_agy_label(model: Optional[str], reasoning_effort: Optional[str] = None) -> str:
    """Resolve an OpenAI-style model id + optional effort into an `agy --model` label."""
    canonical = resolve_model_id(model)
    if reasoning_effort:
        eff = reasoning_effort.strip().lower()
        for suffix in ("-low", "-medium", "-high", "-xhigh"):
            if canonical.endswith(suffix):
                base = canonical[: -len(suffix)]
                candidate = f"{base}-{eff}"
                if candidate in MODEL_MAP:
                    canonical = candidate
                break
        else:
            candidate = f"{canonical}-{eff}"
            if candidate in MODEL_MAP:
                canonical = candidate
    return MODEL_MAP.get(canonical, canonical)


def supported_efforts_for(model: Optional[str]) -> Optional[Tuple[str, ...]]:
    """Return declared effort levels accepted by the given model family."""
    name = resolve_model_id(model).lower()
    if name.startswith("gemini-4-hydrogen"):
        return ("low", "medium", "high")
    if name.startswith(("gemini-4-argon", "gemini-3.8-flash", "gemini-3.7-flash")):
        return ("low", "medium", "high")
    if name.startswith(("claude-opus-5-5", "claude-sonnet-5-5")):
        return ("low", "medium", "high")
    if name.startswith("gemini-3.1-pro"):
        return ("low", "high")
    if name.startswith("gpt-oss-120b"):
        return ("medium",)
    if name in ("claude-sonnet-4-6", "claude-opus-4-6-thinking"):
        return ()
    return None


def get_context_length(model: Optional[str]) -> int:
    name = resolve_model_id(model).lower()
    if name.startswith("gpt-oss"):
        return 128_000
    return DEFAULT_CONTEXT_LENGTH


def _parse_agy_models_output(raw: str) -> List[str]:
    """Parse `agy models` stdout into exposed model ids when recognized."""
    label_to_id = {v.lower(): k for k, v in MODEL_MAP.items()}
    discovered: List[str] = []
    for line in raw.splitlines():
        stripped = line.strip().lstrip("*-• ").strip()
        if not stripped:
            continue
        low = stripped.lower()
        if low in label_to_id:
            mid = label_to_id[low]
            if mid not in discovered:
                discovered.append(mid)
            continue
        for lbl, mid in label_to_id.items():
            if lbl in low and mid not in discovered:
                discovered.append(mid)
    return discovered


def discover_models(agy_path: str, timeout: float = 10.0, force_refresh: bool = False) -> List[str]:
    """Query `agy models` with in-memory TTL caching; falls back to MODEL_MAP."""
    now = time.monotonic()
    with _CACHE_LOCK:
        ts, cached = _CACHED_MODELS
        if not force_refresh and cached and (now - ts) < _CACHE_TTL_SECONDS:
            return list(cached)

    if agy_path and (shutil.which(agy_path) or __import__("os").path.exists(agy_path)):
        try:
            proc = subprocess.run(
                [agy_path, "models"],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout:
                parsed = _parse_agy_models_output(proc.stdout)
                if parsed:
                    with _CACHE_LOCK:
                        globals()["_CACHED_MODELS"] = (time.monotonic(), list(parsed))
                    return parsed
        except Exception as exc:
            logger.debug("agy models discovery failed: %s", exc)

    return list(MODEL_MAP.keys())


def clear_model_cache() -> None:
    with _CACHE_LOCK:
        globals()["_CACHED_MODELS"] = (0.0, [])
