"""Read Antigravity conversation SQLite files and extract model responses."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from proto_decoder import ProtoField, decode_message, find_fields, raw_text, walk_text


USER_STEP_TYPE = 14
MODEL_STEP_TYPE = 15
TOOL_STEP_TYPES = {8, 9}
ERROR_STEP_TYPE = 17
DONE_STATUS = 3

# Keywords that mark an agy error step (type 17) as an upstream/transport
# failure worth surfacing verbatim to the caller.
_ERROR_KEYWORDS = (
    "model unreachable",
    "forcibly closed",
    "terminated due to error",
    "retryable error",
    "failed to get load code assist",
    "stream reading error",
    "resource_exhausted",
    "code 429",
    "quota",
    "permission_denied",
    "code 403",
    "rate limit",
    "timed out",
    "deadline exceeded",
    "connection refused",
    "connection reset",
    "broken pipe",
    "request failed",
)


@dataclass(frozen=True)
class AgResponse:
    answer: str
    reasoning: str = ""
    tool_summaries: tuple[str, ...] = ()
    db_path: str = ""
    truncated: bool = False


class UpstreamError(ValueError):
    """agy/Google transport-side failure worth retrying (rate-limit, transient
    backend drop, model unreachable). Distinct from parse errors and from
    content-safety rejections, which are NOT retryable.

    Carried by read_response so agy_runner can decide retry-vs-surface without
    re-string-matching the message text."""


def read_response(db_path: str | Path) -> AgResponse:
    path = Path(db_path)
    conn = sqlite3.connect(path, timeout=5.0)
    try:
        rows = conn.execute(
            "SELECT idx, step_type, status, step_payload FROM steps ORDER BY idx"
        ).fetchall()
    finally:
        conn.close()

    # Isolate steps belonging to the latest user turn so a failed resumed turn
    # never leaks a previous turn's answer from the same DB.
    last_user_idx = -1
    for idx, step_type, _status, _ in rows:
        if step_type == USER_STEP_TYPE:
            last_user_idx = idx
    turn_rows = [r for r in rows if r[0] > last_user_idx] if last_user_idx != -1 else rows

    tool_summaries = _extract_tool_summaries(turn_rows)

    # Scan completed model rows (newest first) and preserve reasoning even when
    # agy splits reasoning and final answer across adjacent completed rows.
    completed = [
        payload
        for _, step_type, status, payload in turn_rows
        if step_type == MODEL_STEP_TYPE and status == DONE_STATUS and isinstance(payload, bytes)
    ]
    best_answer = ""
    best_reasoning = ""
    for payload in reversed(completed):
        answer, reasoning = _extract_model_response(payload)
        if answer and not best_answer:
            best_answer = answer
            best_reasoning = reasoning
        elif reasoning and not best_reasoning:
            best_reasoning = reasoning
        if best_answer and best_reasoning:
            break
    if best_answer:
        return AgResponse(
            answer=best_answer,
            reasoning=best_reasoning,
            tool_summaries=tuple(tool_summaries),
            db_path=str(path),
        )

    # No answer anywhere. If agy logged an upstream/transport error (type 17),
    # surface it verbatim — this is usually a network/proxy drop to Google, not
    # a parsing problem.
    error = _extract_error(turn_rows)
    if error:
        raise UpstreamError(error)

    # Degraded read: agy killed mid-generation may leave a partial (non-DONE)
    # row whose text is still better than a blank response.
    incomplete = [
        payload
        for _, step_type, status, payload in turn_rows
        if step_type == MODEL_STEP_TYPE and status != DONE_STATUS and isinstance(payload, bytes)
    ]
    for payload in reversed(incomplete):
        answer, reasoning = _extract_model_response(payload)
        if answer or reasoning:
            return AgResponse(
                answer=answer,
                reasoning=reasoning,
                tool_summaries=tuple(tool_summaries),
                db_path=str(path),
                truncated=True,
            )

    # Last resort: reasoning was produced but no answer and no explicit error.
    if best_reasoning:
        return AgResponse(
            answer="",
            reasoning=best_reasoning,
            tool_summaries=tuple(tool_summaries),
            db_path=str(path),
            truncated=True,
        )

    raise ValueError(f"no model response found in {path}")


def read_user_prompt(db_path: str | Path) -> str:
    path = Path(db_path)
    conn = sqlite3.connect(path, timeout=5.0)
    try:
        rows = conn.execute(
            """
            SELECT step_payload
            FROM steps
            WHERE step_type = ? AND status = ?
            ORDER BY idx
            """,
            (USER_STEP_TYPE, DONE_STATUS),
        ).fetchall()
    finally:
        conn.close()

    for (payload,) in rows:
        if not isinstance(payload, bytes):
            continue
        fields = decode_message(payload)
        for field19 in find_fields(fields, 19):
            if isinstance(field19.value, list):
                prompt = raw_text(field19.value, 2)
                if prompt:
                    return prompt
    return ""


def _extract_model_response(payload: bytes) -> tuple[str, str]:
    fields = decode_message(payload)
    response_fields = [
        field.value
        for field in find_fields(fields, 20)
        if isinstance(field.value, list)
    ]
    if not response_fields:
        return "", ""

    answer = ""
    reasoning = ""
    for response in reversed(response_fields):
        if not answer:
            answer = raw_text(response, 1)
        if not reasoning:
            reasoning = raw_text(response, 3)
        if answer and reasoning:
            break
    return answer, reasoning


def _extract_error(rows: list[tuple[int, int, int, bytes]]) -> str:
    """Pull a concise upstream-error message out of agy error steps (type 17)."""
    found: list[str] = []
    fallback: list[str] = []
    for _, step_type, _status, payload in rows:
        if step_type != ERROR_STEP_TYPE or not isinstance(payload, bytes):
            continue
        for text in walk_text(decode_message(payload)):
            normalized = " ".join(text.split())
            if len(normalized) < 10:
                continue
            low = normalized.lower()
            snippet = normalized[:200]
            if any(kw in low for kw in _ERROR_KEYWORDS):
                if snippet not in found:
                    found.append(snippet)
            elif snippet not in fallback and len(fallback) < 2:
                fallback.append(snippet)
    if found:
        return " | ".join(found[:2])
    return " | ".join(fallback[:2])


def _extract_tool_summaries(rows: list[tuple[int, int, int, bytes]]) -> list[str]:
    summaries: list[str] = []
    for _, step_type, status, payload in rows:
        if step_type not in TOOL_STEP_TYPES or status != DONE_STATUS or not isinstance(payload, bytes):
            continue
        fields = decode_message(payload)
        summaries.extend(_pick_tool_texts(fields))
    return summaries


def _pick_tool_texts(fields: list[ProtoField]) -> list[str]:
    seen: set[str] = set()
    picked: list[str] = []
    for text in walk_text(fields):
        clean = text.strip()
        if not clean or clean in seen:
            continue
        if len(clean) > 240:
            clean = clean[:237] + "..."
        if any(marker in clean.lower() for marker in ("view_file", "list_dir", "tool", "file://")):
            picked.append(clean)
            seen.add(clean)
    return picked
