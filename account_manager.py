"""Multi-account pool management and instant rotation for Antigravity CLI."""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

logger = logging.getLogger("agy2api.account_manager")

def mask_email(email: str) -> str:
    """Mask email address for privacy in logs (e.g. user@domain.com -> us***@domain.com)."""
    if not email or "@" not in email:
        return email
    name, domain = email.split("@", 1)
    masked_name = (name[:2] + "***") if len(name) > 2 else "***"
    return f"{masked_name}@{domain}"


def _get_base_home() -> Path:
    home = Path.home()
    if ".gemini_profiles" in home.parts:
        idx = home.parts.index(".gemini_profiles")
        return Path(*home.parts[:idx])
    return home


BASE_HOME = _get_base_home()
LEGACY_CLI_DIR = BASE_HOME / ".gemini" / "antigravity-cli"
PROFILES_DIR = Path(os.getenv("AGY_PROFILES_DIR", str(BASE_HOME / ".gemini_profiles")))
ACTIVE_ACC_FILE = BASE_HOME / ".gemini" / "antigravity-cli" / os.getenv("AGY_STATE_FILE", ".active-account")
TOKEN_PREFIX = "antigravity-oauth-token."
PRIMARY_RECOVERY_SECONDS = int(os.getenv("AGY_PRIMARY_RECOVERY_SEC", "900"))

# Accounts explicitly excluded (configurable via comma-separated emails in env)
EXCLUDED_ACCOUNTS: Set[str] = {
    acc.strip() for acc in os.getenv("AGY_EXCLUDED_ACCOUNTS", "").split(",") if acc.strip()
}

# Preferred account ordering (configurable via comma-separated emails in env)
PREFERRED_ACCOUNTS: Tuple[str, ...] = tuple(
    acc.strip() for acc in os.getenv("AGY_PREFERRED_ACCOUNTS", "").split(",") if acc.strip()
)


def extract_email_from_token(token_path: Path) -> Optional[str]:
    """Extract email from OAuth token JSON by inspecting email, user_email, or id_token JWT claims."""
    if not token_path.exists() or token_path.stat().st_size == 0:
        return None
    try:
        data = json.loads(token_path.read_text(encoding="utf-8"))
        for key in ("email", "user_email"):
            val = data.get(key)
            if val and "@" in str(val):
                return str(val).strip()
        id_token = data.get("id_token")
        if id_token and isinstance(id_token, str) and "." in id_token:
            parts = id_token.split(".")
            if len(parts) >= 2:
                payload = parts[1]
                payload += "=" * (4 - len(payload) % 4) if len(payload) % 4 != 0 else ""
                claims = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8", errors="ignore"))
                email = claims.get("email")
                if email and "@" in str(email):
                    return str(email).strip()
    except Exception as e:
        logger.debug("Could not extract email from %s: %s", token_path.name, e)
    return None


def get_profile_dir(email: str) -> Path:
    """Get the isolated profile HOME directory for a specific account."""
    p_dir = PROFILES_DIR / email
    p_dir.mkdir(parents=True, exist_ok=True)
    cli_dir = p_dir / ".gemini" / "antigravity-cli"
    cli_dir.mkdir(parents=True, exist_ok=True)
    return p_dir


def get_available_accounts() -> List[str]:
    """Scan PROFILES_DIR and LEGACY_CLI_DIR for available accounts, filtering out excluded ones."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    accounts = set()

    for p in PROFILES_DIR.iterdir():
        if p.is_dir() and "@" in p.name:
            token_file = p / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
            if token_file.exists() and token_file.stat().st_size > 0:
                accounts.add(p.name.strip())

    if LEGACY_CLI_DIR.exists():
        for p in LEGACY_CLI_DIR.iterdir():
            if p.is_file() and p.name.startswith(TOKEN_PREFIX):
                email = p.name[len(TOKEN_PREFIX):].strip()
                if "@" in email:
                    accounts.add(email)
                    prof_token = get_profile_dir(email) / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
                    if not prof_token.exists() or prof_token.stat().st_size == 0:
                        try:
                            shutil.copy2(p, prof_token)
                        except OSError:
                            pass

    active_accounts = [a for a in accounts if a not in EXCLUDED_ACCOUNTS]
    pref_rank = {acc: idx for idx, acc in enumerate(PREFERRED_ACCOUNTS)}
    return sorted(active_accounts, key=lambda a: (pref_rank.get(a, 999), a))


def _resolve_primary_account(available: List[str]) -> str:
    custom_default = os.getenv("AGY_DEFAULT_ACCOUNT")
    if custom_default and custom_default in available:
        return custom_default
    for preferred in PREFERRED_ACCOUNTS:
        if preferred in available:
            return preferred
    return available[0]


def get_active_account() -> str:
    """Get currently active account email, auto-recovering back to primary account after cooldown."""
    available = get_available_accounts()
    if not available:
        return "default"

    primary_acc = _resolve_primary_account(available)

    if ACTIVE_ACC_FILE.exists():
        try:
            current = ACTIVE_ACC_FILE.read_text(encoding="utf-8").strip()
            if current in available:
                if current != primary_acc and PRIMARY_RECOVERY_SECONDS > 0:
                    age = time.time() - ACTIVE_ACC_FILE.stat().st_mtime
                    if age >= PRIMARY_RECOVERY_SECONDS:
                        logger.info(
                            "Cooldown elapsed (%.0fs >= %ds). Restoring primary account: %s -> %s",
                            age, PRIMARY_RECOVERY_SECONDS, mask_email(current), mask_email(primary_acc)
                        )
                        ACTIVE_ACC_FILE.write_text(primary_acc, encoding="utf-8")
                        return primary_acc
                return current
        except OSError:
            pass

    try:
        ACTIVE_ACC_FILE.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_ACC_FILE.write_text(primary_acc, encoding="utf-8")
    except OSError:
        pass
    return primary_acc


def rotate_to_next_account(current_account: str) -> str:
    """Instantly switch to the next available account in the pool."""
    available = get_available_accounts()
    if len(available) <= 1:
        return current_account

    try:
        idx = available.index(current_account)
        next_acc = available[(idx + 1) % len(available)]
    except ValueError:
        next_acc = available[0]

    try:
        ACTIVE_ACC_FILE.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_ACC_FILE.write_text(next_acc, encoding="utf-8")
        logger.warning(
            "Instant account swap: %s -> %s (pool size: %d)",
            mask_email(current_account), mask_email(next_acc), len(available)
        )
    except OSError as e:
        logger.error("Failed to write active account file: %s", e)

    return next_acc
