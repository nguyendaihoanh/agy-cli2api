"""Hermes Agent model-provider plugin entry point for nguyendaihoanh/agy-cli2api."""

from __future__ import annotations

import os
from dataclasses import fields as _dataclass_fields
from typing import Any, Dict, List, Optional, Tuple

from providers import register_provider
from providers.base import ProviderProfile

from .account_manager import setup_status_summary
from .models import (
    DEFAULT_MODEL,
    MODEL_ALIASES,
    MODEL_MAP,
    clear_model_cache,
    discover_models,
    get_context_length,
    get_model_label,
    resolve_model_id,
    supported_efforts_for,
)
from .runner import has_quota_marker, resolve_agy_executable

_BASE_URL_SENTINEL = "agy-cli2api://local"


class AgyCli2ApiProfile(ProviderProfile):
    """Multi-account Google Antigravity CLI provider backed by agy-cli2api."""

    def create_client(self, **client_kwargs: Any) -> Any:
        from .client import AgyCli2ApiClient

        return AgyCli2ApiClient(**client_kwargs)

    def setup_status(self, **kwargs: Any) -> Dict[str, Any]:
        return setup_status_summary(resolve_agy_executable())

    def external_auth_evidence(self) -> Optional[Tuple[bool, Optional[str]]]:
        info = self.setup_status()
        return bool(info.get("logged_in")), info.get("plan") or None

    def discover_models(self, **kwargs: Any) -> List[Dict[str, Any]]:
        exe = resolve_agy_executable() or "agy"
        ids = discover_models(exe, force_refresh=bool(kwargs.get("force_refresh", False)))
        return [
            {
                "id": mid,
                "label": get_model_label(mid),
                "note": "Multi-account pool" if "hydrogen" not in mid else "Hydrogen",
            }
            for mid in ids
        ]

    def fetch_models(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 8.0,
    ) -> Optional[List[str]]:
        exe = resolve_agy_executable() or "agy"
        return discover_models(exe, timeout=timeout)

    def invalidate_catalog_cache(self) -> None:
        clear_model_cache()

    def get_model_context_length(self, model: str) -> Optional[int]:
        return get_context_length(model)

    def supported_reasoning_efforts(self, model: Optional[str]) -> Optional[Tuple[str, ...]]:
        return supported_efforts_for(model)

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: Optional[dict] = None,
        model: Optional[str] = None,
        **context: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        effort = None
        if isinstance(reasoning_config, dict):
            effort = "low" if reasoning_config.get("enabled") is False else reasoning_config.get("effort")
        top_level: Dict[str, Any] = {}
        if effort:
            top_level["reasoning_effort"] = str(effort).strip().lower()
        return {}, top_level

    def classify_api_error(
        self,
        error: Exception,
        *,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        message: str = "",
        body: Any = None,
        model: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        combined = f"{error} {message}".lower()
        if status_code == 429 or has_quota_marker(combined):
            return {"reason": "rate_limit", "retryable": True}
        if "timed out" in combined:
            return {"reason": "timeout", "retryable": True}
        if "not found" in combined or "not authenticated" in combined:
            return {"reason": "auth", "retryable": False}
        return None


_SUPPORTED_PROFILE_FIELDS = {f.name for f in _dataclass_fields(ProviderProfile)}

_OPTIONAL_PROFILE_KWARGS: Dict[str, Any] = {
    "aliases": ("agy2api",),
    "display_name": "Antigravity CLI (agy-cli2api Multi-Account)",
    "description": "Multi-account Google Antigravity CLI provider with 0.2s auto-swap failover",
    "api_mode": "chat_completions",
    "auth_type": "external_process",
    "base_url": os.getenv("AGY_CLI2API_BASE_URL", _BASE_URL_SENTINEL).strip() or _BASE_URL_SENTINEL,
    "default_aux_model": DEFAULT_MODEL,
    "process_command": "agy",
    "process_args": ("--print", ""),
    "process_command_env_vars": ("AGY_PATH", "ANTIGRAVITY_COMMAND", "ANTIGRAVITY_CLI_PATH"),
    "fallback_models": tuple(MODEL_MAP.keys()),
    "model_aliases": MODEL_ALIASES,
    "supports_streaming": True,
    "supports_health_check": False,
    "supports_model_listing": True,
    "supports_vision": False,
    "unsupported_response_formats": ("json_schema",),
    "static_models": tuple(MODEL_MAP.keys()),
    "base_url_env_var": "AGY_CLI2API_BASE_URL",
}

PROFILE = AgyCli2ApiProfile(
    name="agy-cli2api",
    **{k: v for k, v in _OPTIONAL_PROFILE_KWARGS.items() if k in _SUPPORTED_PROFILE_FIELDS},
)

register_provider(PROFILE)
