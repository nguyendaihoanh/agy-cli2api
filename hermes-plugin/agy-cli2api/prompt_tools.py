"""Prompt serialization and <tool_call> extraction for Hermes Agent <-> agy CLI."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

_RECENT_TOOL_RESULTS_KEPT = 8
_OLD_TOOL_RESULT_CAP = 600
_RECENT_TOOL_RESULT_CAP = 15000

_OPEN_TAG = "<tool_call>"
_CLOSE_TAG = "</tool_call>"
_NOISE_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)


def content_to_text(content: Any) -> str:
    """Flatten OpenAI-style text/content-block arrays into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: List[str] = []
        for item in content:
            if isinstance(item, dict):
                kind = str(item.get("type", ""))
                if kind == "text" or "text" in item:
                    pieces.append(str(item.get("text", "")))
                elif kind == "image_url":
                    url = (item.get("image_url") or {}).get("url", "")
                    if url and not str(url).startswith("data:"):
                        pieces.append(f"[Image]({url})")
            else:
                pieces.append(str(item))
        return "\n".join(p for p in pieces if p)
    return str(content)


def format_tools_preamble(tools: Optional[Sequence[Dict[str, Any]]], tool_choice: Any = None) -> str:
    """Build concise instruction forcing `agy` to delegate tool execution back to Hermes."""
    if not tools or tool_choice == "none":
        return ""

    specs: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = str(fn.get("name", "")).strip()
        if not name:
            continue
        specs.append({
            "name": name,
            "description": str(fn.get("description", ""))[:400],
            "parameters": fn.get("parameters", {}),
        })

    if not specs:
        return ""

    choice_hint = ""
    if tool_choice == "required":
        choice_hint = "You MUST invoke at least one declared tool in this turn using <tool_call>.\n"
    elif isinstance(tool_choice, dict):
        fn_name = ((tool_choice.get("function") or {}).get("name")) or ""
        if fn_name:
            choice_hint = f"You MUST call the `{fn_name}` tool using <tool_call>.\n"

    return (
        "### HOST TOOL PROTOCOL (MANDATORY)\n"
        "Do NOT run shell commands, file edits, or internal agy tools yourself.\n"
        "Whenever you need to execute an action or read external data, emit one or more "
        "exact XML blocks using ONLY the host tools listed below:\n"
        '<tool_call>{"name": "<tool_name>", "arguments": {<json_args>}}</tool_call>\n'
        f"{choice_hint}"
        "Wait for the host to return tool results before continuing.\n"
        f"Available Host Tools (JSON):\n{json.dumps(specs, ensure_ascii=False)}"
    )


def format_messages_prompt(
    messages: Sequence[Dict[str, Any]],
    *,
    tools: Optional[Sequence[Dict[str, Any]]] = None,
    tool_choice: Any = None,
) -> str:
    """Serialize full OpenAI chat messages and tools into a prompt for agy stdin."""
    sections: List[str] = []
    tools_block = format_tools_preamble(tools, tool_choice)
    if tools_block:
        sections.append(tools_block)

    tool_indices = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "tool"]
    cutoff_idx = tool_indices[-_RECENT_TOOL_RESULTS_KEPT] if len(tool_indices) > _RECENT_TOOL_RESULTS_KEPT else -1

    transcript: List[str] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "user")).strip().lower()
        body = content_to_text(msg.get("content")).strip()

        if role == "tool":
            cap = _OLD_TOOL_RESULT_CAP if idx < cutoff_idx else _RECENT_TOOL_RESULT_CAP
            if len(body) > cap:
                body = body[: cap // 2] + f"\n[... truncated {len(body) - cap} chars ...]\n" + body[-cap // 2 :]
            call_id = msg.get("tool_call_id") or msg.get("name") or "tool"
            transcript.append(f"Tool Result ({call_id}):\n{body}")
            continue

        if role == "assistant":
            parts: List[str] = []
            if body:
                parts.append(body)
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments", "{}")
                try:
                    parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    parsed_args = raw_args
                parts.append(
                    f"{_OPEN_TAG}{json.dumps({'name': fn.get('name', ''), 'arguments': parsed_args}, ensure_ascii=False)}{_CLOSE_TAG}"
                )
            if parts:
                transcript.append("Assistant:\n" + "\n".join(parts))
            continue

        label = "System" if role in ("system", "developer") else ("User" if role == "user" else "Context")
        if body:
            transcript.append(f"{label}:\n{body}")

    if transcript:
        sections.append("\n\n".join(transcript))
    return "\n\n".join(s for s in sections if s).strip()


def format_delta_prompt(messages: Sequence[Dict[str, Any]]) -> str:
    """Serialize only newly appended turns when resuming via --conversation."""
    return format_messages_prompt(messages, tools=None, tool_choice=None)


def _find_tag_close(text: str, start: int) -> int:
    """Find closing `</tool_call>` outside JSON string literals."""
    in_string = False
    escaped = False
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if text.startswith(_CLOSE_TAG, i):
            return i
        i += 1
    return text.find(_CLOSE_TAG, start)


def _decode_tool_json(payload_str: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """Parse JSON body inside `<tool_call>` into `(name, arguments_json, call_id)`."""
    raw = payload_str.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2:
            raw = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    call_id = str(data.get("id") or "").strip() or None
    fn_block = data.get("function")
    if isinstance(fn_block, dict):
        name = str(fn_block.get("name") or data.get("name") or "").strip()
        args = fn_block.get("arguments", data.get("arguments", {}))
    else:
        name = str(data.get("name") or "").strip()
        args = data.get("arguments", {})

    if not name:
        return None
    if isinstance(args, str):
        args_json = args
    else:
        args_json = json.dumps(args if args is not None else {}, ensure_ascii=False)
    return name, args_json, call_id


def extract_tool_calls(text: str) -> Tuple[str, List[SimpleNamespace]]:
    """Extract `<tool_call>` blocks into OpenAI-compatible tool call objects + clean text."""
    if not text or _OPEN_TAG not in text:
        return _NOISE_TAG_RE.sub("", text or "").strip(), []

    calls: List[SimpleNamespace] = []
    kept_segments: List[str] = []
    cursor = 0
    call_idx = 0

    while cursor < len(text):
        open_idx = text.find(_OPEN_TAG, cursor)
        if open_idx == -1:
            kept_segments.append(text[cursor:])
            break

        kept_segments.append(text[cursor:open_idx])
        body_start = open_idx + len(_OPEN_TAG)
        close_idx = _find_tag_close(text, body_start)
        if close_idx == -1:
            kept_segments.append(text[open_idx:])
            break

        raw_block = text[body_start:close_idx]
        parsed = _decode_tool_json(raw_block)
        if parsed is None:
            kept_segments.append(text[open_idx : close_idx + len(_CLOSE_TAG)])
        else:
            name, args_json, raw_id = parsed
            call_id = raw_id or f"call_{call_idx + 1}"
            calls.append(
                SimpleNamespace(
                    index=call_idx,
                    id=call_id,
                    type="function",
                    function=SimpleNamespace(name=name, arguments=args_json),
                )
            )
            call_idx += 1
        cursor = close_idx + len(_CLOSE_TAG)

    cleaned = _NOISE_TAG_RE.sub("", "".join(kept_segments)).strip()
    return cleaned, calls
