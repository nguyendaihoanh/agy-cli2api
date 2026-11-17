"""Multi-account pool discovery and instant rotation for Antigravity CLI profiles."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("agy_cli2api.account_manager")
_ACCOUNT_LOCK = threading.RLock()
_TOKEN_PREFIX = "antigravity-oauth-token."


def mask_email(email: str) -> str:
    """Mask email address for privacy in logs."""
    if not email or "@" not in email:
        return email
    name, domain = email.split("@", 1)
    masked = (name[:2] + "***") if len(name) > 2 else "***"
    return f"{masked}@{domain}"


def _base_home() -> Path:
    home = Path.home()
    if ".gemini_profiles" in home.parts:
        return Path(*home.parts[: home.parts.index(".gemini_profiles")])
    return home


def _profiles_dir() -> Path:
    override = os.getenv("AGY_PROFILES_DIR", "").strip()
    return Path(override) if override else (_base_home() / ".gemini_profiles")


def _legacy_cli_dir() -> Path:
    override = os.getenv("AGY_CLI_DIR", "").strip()
    return Path(override) if override else (_base_home() / ".gemini" / "antigravity-cli")


def _active_acc_file() -> Path:
    state_name = os.getenv("AGY_STATE_FILE", ".active-account").strip() or ".active-account"
    p = Path(state_name)
    return p if p.is_absolute() else (_legacy_cli_dir() / state_name)


def _excluded_accounts() -> Set[str]:
    raw = os.getenv("AGY_EXCLUDED_ACCOUNTS", "")
    return {acc.strip() for acc in raw.split(",") if acc.strip()}


def _preferred_accounts() -> Tuple[str, ...]:
    raw = os.getenv("AGY_PREFERRED_ACCOUNTS", "")
    return tuple(acc.strip() for acc in raw.split(",") if acc.strip())


def _recovery_seconds() -> int:
    try:
        return max(0, int(os.getenv("AGY_PRIMARY_RECOVERY_SEC", "900")))
    except ValueError:
        return 900


def get_profile_dir(email: str) -> Path:
    """Return isolated HOME directory for a named profile account."""
    p_dir = _profiles_dir() / email
    (p_dir / ".gemini" / "antigravity-cli").mkdir(parents=True, exist_ok=True)
    return p_dir


def _has_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def get_available_accounts() -> List[str]:
    """Discover local Antigravity CLI accounts without reading raw token contents."""
    profiles_root = _profiles_dir()
    legacy_root = _legacy_cli_dir()
    excluded = _excluded_accounts()
    preferred = _preferred_accounts()

    accounts: Set[str] = set()
    if profiles_root.is_dir():
        try:
            for entry in profiles_root.iterdir():
                if entry.is_dir() and "@" in entry.name:
                    token_file = entry / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
                    if _has_nonempty_file(token_file):
                        accounts.add(entry.name.strip())
        except OSError:
            pass

    if legacy_root.is_dir():
        try:
            for entry in legacy_root.iterdir():
                if entry.is_file() and entry.name.startswith(_TOKEN_PREFIX):
                    email = entry.name[len(_TOKEN_PREFIX) :].strip()
                    if "@" in email:
                        accounts.add(email)
                        prof_token = get_profile_dir(email) / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
                        if not _has_nonempty_file(prof_token):
                            try:
                                shutil.copy2(entry, prof_token)
                            except OSError:
                                pass
        except OSError:
            pass

    active = [a for a in accounts if a not in excluded]
    pref_rank = {acc: idx for idx, acc in enumerate(preferred)}
    return sorted(active, key=lambda a: (pref_rank.get(a, 999), a))


def has_default_cli_session() -> bool:
    """Check whether standard single-account `agy` home is initialized on disk."""
    legacy_root = _legacy_cli_dir()
    return (
        _has_nonempty_file(legacy_root / "antigravity-oauth-token")
        or (legacy_root / "conversations").is_dir()
    )


def _resolve_primary_account(available: List[str]) -> str:
    custom_default = os.getenv("AGY_DEFAULT_ACCOUNT", "").strip()
    if custom_default and custom_default in available:
        return custom_default
    for pref in _preferred_accounts():
        if pref in available:
            return pref
    return available[0]


def _write_active_account(account: str) -> None:
    acc_file = _active_acc_file()
    acc_file.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".active-account-", dir=str(acc_file.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(account)
        os.replace(tmp_name, acc_file)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def get_active_account() -> str:
    """Return currently active account email, recovering primary after cooldown."""
    with _ACCOUNT_LOCK:
        available = get_available_accounts()
        if not available:
            return "default"

        primary = _resolve_primary_account(available)
        acc_file = _active_acc_file()
        recovery_sec = _recovery_seconds()

        if acc_file.exists():
            try:
                current = acc_file.read_text(encoding="utf-8").strip()
                if current in available:
                    if current != primary and recovery_sec > 0:
                        age = time.time() - acc_file.stat().st_mtime
                        if age >= recovery_sec:
                            logger.info(
                                "Cooldown elapsed (%.0fs >= %ds). Restoring primary: %s -> %s",
                                age,
                                recovery_sec,
                                mask_email(current),
                                mask_email(primary),
                            )
                            _write_active_account(primary)
                            return primary
                    return current
            except OSError:
                pass

        try:
            _write_active_account(primary)
        except OSError:
            pass
        return primary


def rotate_to_next_account(current_account: str) -> str:
    """Switch atomically to the next available account in the pool."""
    with _ACCOUNT_LOCK:
        available = get_available_accounts()
        if len(available) <= 1:
            return current_account

        acc_file = _active_acc_file()
        if acc_file.exists():
            try:
                recorded = acc_file.read_text(encoding="utf-8").strip()
                if recorded in available and recorded != current_account:
                    return recorded
            except OSError:
                pass

        try:
            idx = available.index(current_account)
            next_acc = available[(idx + 1) % len(available)]
        except ValueError:
            next_acc = available[0]

        try:
            _write_active_account(next_acc)
            logger.warning(
                "Instant account swap: %s -> %s (pool size: %d)",
                mask_email(current_account),
                mask_email(next_acc),
                len(available),
            )
        except OSError as exc:
            logger.error("Failed updating active account marker: %s", exc)
        return next_acc


def setup_status_summary(agy_exe: Optional[str]) -> Dict[str, Any]:
    """Return non-secret setup status for Hermes UI/CLI readiness checks."""
    available_bin = bool(agy_exe)
    pool = get_available_accounts()
    logged_in = bool(pool) or has_default_cli_session()
    proxy_url = os.getenv("AGY_CLI2API_BASE_URL", "").strip()
    if proxy_url:
        detail = f"HTTP gateway mode configured ({proxy_url})"
        available_bin = True
        logged_in = True
    elif pool:
        detail = f"{len(pool)} local Antigravity profile(s) available for auto-swap"
    elif logged_in:
        detail = "Signed in via default agy CLI session"
    elif available_bin:
        detail = "agy CLI found, not signed in yet; run `agy` to authenticate"
    else:
        detail = "agy CLI not found on PATH; install Antigravity CLI or set AGY_PATH"

    return {
        "available": available_bin,
        "logged_in": logged_in,
        "plan": f"pool:{len(pool)}" if pool else ("single" if logged_in else ""),
        "detail": detail,
        "login_command": "agy",
    }
