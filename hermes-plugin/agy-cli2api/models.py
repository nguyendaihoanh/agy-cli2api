"""Model catalog, aliases, effort mapping, and dynamic discovery for agy-cli2api."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("agy_cli2api.models")

MODEL_MAP: Dict[str, str] = {
    "gemini-4-neon-xhigh": "Gemini 4 Neon (XHigh)",
    "gemini-4-neon-high": "Gemini 4 Neon (High)",
    "gemini-4-neon-medium": "Gemini 4 Neon (Medium)",
    "gemini-4-neon-low": "Gemini 4 Neon (Low)",
    "gemini-4-argon-medium": "Gemini 4 Argon (Medium)",
    "gemini-4-argon-high": "Gemini 4 Argon (High)",
    "gemini-4-argon-low": "Gemini 4 Argon (Low)",
    "gemini-4-hydrogen-low": "Gemini 4 Hydrogen (Low)",
    "gemini-4-hydrogen-medium": "Gemini 4 Hydrogen (Medium)",
    "gemini-4-hydrogen-high": "Gemini 4 Hydrogen (High)",
    "gemini-4-hydrogen-xhigh": "Gemini 4 Hydrogen (XHigh)",
    "gemini-pro-orchestrator-neon": "Orchestrator (Neon)",
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
    "neon": "gemini-4-neon-xhigh",
    "neon-xhigh": "gemini-4-neon-xhigh",
    "gemini-4-neon": "gemini-4-neon-xhigh",
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
_CACHE_TTL_SECONDS = 1800.0
_FALLBACK_BACKOFF_SECONDS = 60.0
_CACHE_LOCK = threading.Lock()
_CACHED_MODELS: Tuple[float, float, List[str]] = (0.0, 0.0, [])
_DISCOVERED_LABELS: Dict[str, str] = {}

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


def _known_catalog() -> Dict[str, str]:
    merged = dict(MODEL_MAP)
    with _CACHE_LOCK:
        merged.update(_DISCOVERED_LABELS)
    return merged


def get_model_label(model_id: str) -> str:
    """Return display label from live discovered catalog or static fallback."""
    with _CACHE_LOCK:
        live = _DISCOVERED_LABELS.get(model_id)
    if live:
        return live
    return MODEL_MAP.get(model_id, model_id)


def resolve_model_id(model: Optional[str]) -> str:
    """Normalize aliases to a canonical exposed model id."""
    cleaned = (model or "").strip()
    if not cleaned:
        return DEFAULT_MODEL
    return MODEL_ALIASES.get(cleaned.lower(), cleaned)


def resolve_agy_label(model: Optional[str], reasoning_effort: Optional[str] = None) -> str:
    """Resolve an OpenAI-style model id + optional effort into an `agy --model` label."""
    canonical = resolve_model_id(model)
    catalog = _known_catalog()
    if reasoning_effort:
        eff = reasoning_effort.strip().lower()
        for suffix in ("-low", "-medium", "-high", "-xhigh"):
            if canonical.endswith(suffix):
                base = canonical[: -len(suffix)]
                candidate = f"{base}-{eff}"
                if candidate in catalog:
                    canonical = candidate
                break
        else:
            candidate = f"{canonical}-{eff}"
            if candidate in catalog:
                canonical = candidate
    return catalog.get(canonical, canonical)


def supported_efforts_for(model: Optional[str]) -> Optional[Tuple[str, ...]]:
    """Return declared effort levels accepted by the given model family."""
    name = resolve_model_id(model).lower()
    if name.startswith(("gemini-4-neon", "gemini-4-hydrogen")):
        return ("low", "medium", "high", "xhigh")
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


def _parse_agy_models_catalog(raw: str) -> Dict[str, str]:
    """Parse `agy models` stdout into an ordered `{model_id: display_label}` dict.

    Supports newly released models even before they are added to `MODEL_MAP`.
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

        if low in label_to_id:
            mid = label_to_id[low]
            discovered.setdefault(mid, MODEL_MAP[mid])
            continue

        if ("-" in cleaned or "." in cleaned) and _MODEL_ID_RE.match(cleaned):
            discovered.setdefault(cleaned, MODEL_MAP.get(cleaned, cleaned))

    return discovered


def _parse_agy_models_output(raw: str) -> List[str]:
    """Parse `agy models` stdout into exposed model ids (including newly released ones)."""
    return list(_parse_agy_models_catalog(raw).keys())


def discover_models(agy_path: str, timeout: float = 8.0, force_refresh: bool = False) -> List[str]:
    """Query `agy models` with in-memory TTL caching; falls back to MODEL_MAP."""
    now = time.monotonic()
    with _CACHE_LOCK:
        ts, ttl, cached = _CACHED_MODELS
        if not force_refresh and cached and (now - ts) < ttl:
            return list(cached)

    if agy_path and (shutil.which(agy_path) or os.path.exists(agy_path)):
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
                creationflags=creationflags,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout:
                catalog = _parse_agy_models_catalog(proc.stdout)
                if catalog:
                    ids = list(catalog.keys())
                    with _CACHE_LOCK:
                        _DISCOVERED_LABELS.clear()
                        _DISCOVERED_LABELS.update(catalog)
                        globals()["_CACHED_MODELS"] = (time.monotonic(), _CACHE_TTL_SECONDS, list(ids))
                    return ids
        except Exception as exc:
            logger.debug("agy models discovery failed: %s", exc)

    fallback_ids = list(MODEL_MAP.keys())
    with _CACHE_LOCK:
        _, _, prev = _CACHED_MODELS
        result = list(prev) if prev else fallback_ids
        globals()["_CACHED_MODELS"] = (time.monotonic(), _FALLBACK_BACKOFF_SECONDS, list(result))
    return result


def clear_model_cache() -> None:
    with _CACHE_LOCK:
        _DISCOVERED_LABELS.clear()
        globals()["_CACHED_MODELS"] = (0.0, 0.0, [])
