"""Subprocess execution, multi-account auto-swap, and artifact cleanup for agy."""

from __future__ import annotations

import logging
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .account_manager import (
    get_active_account,
    get_available_accounts,
    get_profile_dir,
    mask_email,
    rotate_to_next_account,
)
from .conversation_reader import AgResponse, UpstreamError, read_response

logger = logging.getLogger("agy_cli2api.runner")

_PASSTHROUGH_ENV_VARS = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LANGUAGE",
    "TZ",
    "TERM",
    "TMPDIR",
    "TEMP",
    "TMP",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "DBUS_SESSION_BUS_ADDRESS",
    "SYSTEMROOT",
    "APPDATA",
    "LOCALAPPDATA",
)

_QUOTA_MARKERS = (
    "resource_exhausted",
    "code 429",
    "individual quota reached",
    "quota exceeded",
    "quota reached",
    "quota exhausted",
    "permission_denied",
    "code 403",
)

_BENIGN_QUOTA_SUBSTRINGS = (
    "fetchconfig",
    "access_token_scope_insufficient",
    "telemetry.go",
    "repo.go",
)

_PERMANENT_MARKERS = (
    "safety",
    "block_reason",
    "content policy",
    "policy violation",
    "prohibited",
)

_CONVERSATION_ID_RE = re.compile(
    r"conversation[\"=:\s]+([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)
_NOISY_LOG_RE = re.compile(
    r"(reconcile_summaries|failed to generate summary|oversized step|conversationsearch)",
    re.IGNORECASE,
)
_DB_SIDECARS = ("-wal", "-shm", "-journal")


@dataclass(frozen=True)
class AgyRunResult:
    db_path: Path
    response: AgResponse
    account: str
    stderr: str = ""


def resolve_agy_executable() -> Optional[str]:
    """Locate the local `agy` binary."""
    for var in ("AGY_PATH", "ANTIGRAVITY_COMMAND", "ANTIGRAVITY_CLI_PATH"):
        explicit = (os.getenv(var) or "").strip()
        if explicit and (Path(explicit).exists() or shutil.which(explicit)):
            return explicit
    found = shutil.which("agy")
    if found:
        return found
    for candidate in (
        Path.home() / ".local" / "bin" / "agy",
        Path.home() / "bin" / "agy",
        Path("/usr/local/bin/agy"),
        Path("/opt/homebrew/bin/agy"),
        Path.home() / "AppData" / "Local" / "agy" / "bin" / "agy.exe",
    ):
        if candidate.exists():
            return str(candidate)
    return None


def _build_child_env(profile_home: Optional[Path]) -> Dict[str, str]:
    """Construct minimal clean child environment without leaking host API tokens."""
    env: Dict[str, str] = {}
    for key in _PASSTHROUGH_ENV_VARS:
        val = os.getenv(key)
        if val is not None:
            env[key] = val
    env["AGY_CLI_DISABLE_AUTO_UPDATE"] = "true"
    if profile_home is not None:
        env["HOME"] = str(profile_home)
        env["USERPROFILE"] = str(profile_home)
    return env


def _keep_updater_quiet(profile_home: Optional[Path]) -> None:
    """Touch last_check.timestamp so agy does not launch a background updater window."""
    if (os.getenv("AGY2API_SUPPRESS_AUTOUPDATE", "true") or "").lower() not in ("1", "true", "yes", "on"):
        return
    root = profile_home or Path.home()
    ts_file = root / ".gemini" / "antigravity-cli" / "last_check.timestamp"
    try:
        ts_file.parent.mkdir(parents=True, exist_ok=True)
        ts_file.write_text(str(int(time.time())), encoding="utf-8")
    except OSError:
        pass


def has_quota_marker(text: str) -> bool:
    for raw_line in text.splitlines():
        line = raw_line.lower()
        if any(b in line for b in _BENIGN_QUOTA_SUBSTRINGS):
            continue
        if any(m in line for m in _QUOTA_MARKERS):
            return True
    return False


def _cleanup_db(db_path: Path) -> None:
    for _ in range(8):
        try:
            if db_path.exists():
                db_path.unlink()
            break
        except PermissionError:
            time.sleep(0.15)
        except OSError:
            break
    for suffix in _DB_SIDECARS:
        side = db_path.with_name(db_path.name + suffix)
        if side.exists():
            try:
                side.unlink()
            except OSError:
                pass
    brain_dir = db_path.parent.parent / "brain" / db_path.stem
    if brain_dir.is_dir():
        shutil.rmtree(brain_dir, ignore_errors=True)


def _extract_db_from_log(log_path: Path, conv_dir: Path) -> Optional[Path]:
    deadline = time.time() + 2.0
    while time.time() <= deadline:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        ids: List[str] = []
        for line in text.splitlines():
            if _NOISY_LOG_RE.search(line):
                continue
            for cid in _CONVERSATION_ID_RE.findall(line):
                if cid not in ids:
                    ids.append(cid)
        for cid in reversed(ids):
            candidate = conv_dir / f"{cid}.db"
            try:
                if candidate.exists() and candidate.stat().st_size > 0:
                    return candidate
            except OSError:
                continue
        time.sleep(0.2)
    return None


def _run_once(
    *,
    agy_exe: str,
    prompt: str,
    model_label: Optional[str],
    conversation_id: Optional[str],
    active_account: str,
    timeout_seconds: float,
    keep_db: bool,
    cancel_event: Optional[threading.Event] = None,
) -> AgyRunResult:
    import subprocess

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Request cancelled before launching agy.")

    p_dir: Optional[Path] = None
    if active_account and active_account != "default":
        p_dir = get_profile_dir(active_account)
        conv_dir = p_dir / ".gemini" / "antigravity-cli" / "conversations"
    else:
        conv_dir = Path.home() / ".gemini" / "antigravity-cli" / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)

    _keep_updater_quiet(p_dir)
    run_env = _build_child_env(p_dir)
    log_fd, raw_log = tempfile.mkstemp(prefix="agy_plugin_", suffix=".log")
    os.close(log_fd)
    log_path = Path(raw_log)

    workdir = Path(tempfile.gettempdir())
    print_sec = max(1, int(timeout_seconds))
    cmd = [agy_exe, "--print", "", "--log-file", str(log_path), "--print-timeout", f"{print_sec}s"]
    if conversation_id:
        cmd.extend(["--conversation", conversation_id])
    if model_label:
        cmd.extend(["--model", model_label])

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    cleanup_enabled = (os.getenv("AGY2API_CLEANUP_DB", "true") or "").lower() in ("1", "true", "yes", "on")

    try:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workdir),
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds + 20.0,
                check=False,
                env=run_env,
                creationflags=creationflags,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"agy timed out after {timeout_seconds:.0f}s") from exc

        db_path = _extract_db_from_log(log_path, conv_dir)
        if db_path is not None:
            deadline = time.time() + 3.0
            while True:
                try:
                    response = read_response(db_path)
                    break
                except UpstreamError:
                    if cleanup_enabled and conversation_id is None:
                        _cleanup_db(db_path)
                    raise
                except (ValueError, sqlite3.OperationalError) as exc:
                    if time.time() >= deadline:
                        if cleanup_enabled and conversation_id is None:
                            _cleanup_db(db_path)
                        diag = "\n".join(p for p in (proc.stderr, proc.stdout) if p)
                        if has_quota_marker(diag):
                            raise UpstreamError(f"Quota/permission error: {diag[:300]}") from exc
                        raise RuntimeError(f"{exc} (agy rc={proc.returncode})") from exc
                    time.sleep(0.2)

            if cleanup_enabled and not keep_db and proc.returncode == 0 and not response.truncated:
                _cleanup_db(db_path)
            return AgyRunResult(db_path=db_path, response=response, account=active_account, stderr=proc.stderr or "")

        diag = "\n".join(p for p in (proc.stderr, proc.stdout, log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else "") if p)
        if has_quota_marker(diag):
            raise UpstreamError(f"Antigravity quota/permission rejected: {diag[:300]}")
        raise RuntimeError(f"agy exited with rc={proc.returncode}: {(proc.stderr or proc.stdout or 'no output')[:400]}")
    finally:
        try:
            log_path.unlink(missing_ok=True)
        except OSError:
            pass


def run_agy_with_autoswap(
    *,
    prompt: str,
    model_label: Optional[str],
    conversation_id: Optional[str] = None,
    full_prompt_fallback: Optional[str] = None,
    timeout_seconds: float = 300.0,
    keep_db: bool = False,
    cancel_event: Optional[threading.Event] = None,
) -> AgyRunResult:
    """Execute `agy` with transient retries and 0.2s multi-account rotation on 429/403."""
    agy_exe = resolve_agy_executable()
    if not agy_exe:
        raise RuntimeError("agy CLI binary not found. Install Antigravity CLI or set AGY_PATH.")

    available = get_available_accounts()
    max_swaps = max(1, len(available))
    target_cid = conversation_id
    current_prompt = prompt
    last_exc: Optional[Exception] = None

    for swap_round in range(max_swaps):
        active_acc = get_active_account()
        if target_cid and full_prompt_fallback:
            p_dir = get_profile_dir(active_acc) if active_account_is_profile(active_acc) else Path.home()
            conv_db = p_dir / ".gemini" / "antigravity-cli" / "conversations" / f"{target_cid}.db"
            if not conv_db.exists():
                target_cid = None
                current_prompt = full_prompt_fallback

        swapped = False
        for attempt in range(2):
            try:
                return _run_once(
                    agy_exe=agy_exe,
                    prompt=current_prompt,
                    model_label=model_label,
                    conversation_id=target_cid,
                    active_account=active_acc,
                    timeout_seconds=timeout_seconds,
                    keep_db=keep_db,
                    cancel_event=cancel_event,
                )
            except UpstreamError as exc:
                last_exc = exc
                err_str = str(exc)
                if has_quota_marker(err_str) and len(available) > 1 and swap_round + 1 < max_swaps:
                    next_acc = rotate_to_next_account(active_acc)
                    logger.warning(
                        "Quota reached on %s; auto-swapped to %s (round %d/%d).",
                        mask_email(active_acc),
                        mask_email(next_acc),
                        swap_round + 1,
                        max_swaps,
                    )
                    if target_cid and full_prompt_fallback:
                        target_cid = None
                        current_prompt = full_prompt_fallback
                    time.sleep(0.2)
                    swapped = True
                    break
                if any(m in err_str.lower() for m in _PERMANENT_MARKERS):
                    raise
                if attempt == 0:
                    time.sleep(1.5)
                    continue
                raise
            except Exception as exc:
                if target_cid and full_prompt_fallback and attempt == 0:
                    target_cid = None
                    current_prompt = full_prompt_fallback
                    continue
                raise

        if not swapped and last_exc:
            raise last_exc

    if last_exc:
        raise last_exc
    raise RuntimeError("agy execution failed without result.")


def active_account_is_profile(account: str) -> bool:
    return bool(account and account != "default")
