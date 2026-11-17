"""OpenAI-compatible client facade for Hermes Agent (CLI Direct + Optional HTTP Gateway)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any, Iterator, List, Optional

from .models import DEFAULT_MODEL, resolve_agy_label, resolve_model_id
from .prompt_tools import extract_tool_calls, format_delta_prompt, format_messages_prompt
from .runner import run_agy_with_autoswap
from .session_store import SessionStore, fingerprint

_DEFAULT_TIMEOUT_SECONDS = 300.0
_PII_REDACT_RE = re.compile(
    r"([A-Za-z]:[/\\][^\s'\"]+|/(?:Users|home|root|var|tmp|opt)/[^\s'\"]+|[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+|Bearer\s+[A-Za-z0-9._~+/-]+=*)"
)


def sanitize_error_text(text: str) -> str:
    return _PII_REDACT_RE.sub("[REDACTED]", text or "")


async def _ready(value: Any) -> Any:
    return value


class _AwaitableCompletion(SimpleNamespace):
    """ChatCompletion result that supports both sync calls and `await create(...)`."""

    def __await__(self) -> Any:
        return _ready(self).__await__()


class _ChunkStream(Iterator[Any]):
    """Yields OpenAI-compatible ChatCompletionChunk objects across sync and async loops."""

    response: Any = None

    def __init__(self, chunks: List[Any]):
        self._chunks = iter(chunks)

    def __iter__(self) -> "_ChunkStream":
        return self

    def __next__(self) -> Any:
        return next(self._chunks)

    def __await__(self) -> Any:
        return _ready(self).__await__()

    def __aiter__(self) -> "_ChunkStream":
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    def close(self) -> None:
        return None


def _build_completion_objects(
    *,
    completion_id: str,
    model: str,
    content: str,
    reasoning: str,
    tool_calls: List[SimpleNamespace],
    finish_reason: str,
) -> _AwaitableCompletion:
    msg = SimpleNamespace(
        role="assistant",
        content=content or None,
        tool_calls=tool_calls if tool_calls else None,
        reasoning=reasoning or None,
        reasoning_content=reasoning or None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(index=0, message=msg, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=0,
        completion_tokens=len(content or ""),
        total_tokens=len(content or ""),
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )
    return _AwaitableCompletion(
        id=completion_id,
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=usage,
    )


def _completion_to_stream(completion: _AwaitableCompletion) -> _ChunkStream:
    choice = completion.choices[0]
    msg = choice.message
    cid = completion.id
    model = completion.model
    created = getattr(completion, "created", int(time.time()))

    chunks: List[Any] = [
        SimpleNamespace(
            id=cid,
            object="chat.completion.chunk",
            created=created,
            model=model,
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(role="assistant", content=None, reasoning_content=None, tool_calls=None),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
    ]

    if msg.reasoning_content:
        chunks.append(
            SimpleNamespace(
                id=cid,
                object="chat.completion.chunk",
                created=created,
                model=model,
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(role=None, content=None, reasoning_content=msg.reasoning_content, tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
        )

    if msg.content:
        chunks.append(
            SimpleNamespace(
                id=cid,
                object="chat.completion.chunk",
                created=created,
                model=model,
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(role=None, content=msg.content, reasoning_content=None, tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
        )

    if msg.tool_calls:
        chunks.append(
            SimpleNamespace(
                id=cid,
                object="chat.completion.chunk",
                created=created,
                model=model,
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(role=None, content=None, reasoning_content=None, tool_calls=msg.tool_calls),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
        )

    chunks.append(
        SimpleNamespace(
            id=cid,
            object="chat.completion.chunk",
            created=created,
            model=model,
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(role=None, content=None, reasoning_content=None, tool_calls=None),
                    finish_reason=choice.finish_reason,
                )
            ],
            usage=completion.usage,
        )
    )
    return _ChunkStream(chunks)


class _CompletionsResource:
    def __init__(self, client: "AgyCli2ApiClient") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ChatNamespace:
    def __init__(self, client: "AgyCli2ApiClient") -> None:
        self.completions = _CompletionsResource(client)


class AgyCli2ApiClient:
    """OpenAI-compatible client backing `agy-cli2api` inside Hermes."""

    # Tell Hermes core not to replace this client with a generic HTTP AsyncOpenAI
    # or transport adapter, since it drives local `agy` CLI profiles directly.
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, **kwargs: Any) -> None:
        self.base_url = str(kwargs.get("base_url") or os.getenv("AGY_CLI2API_BASE_URL", "")).strip()
        self.api_key = str(kwargs.get("api_key") or os.getenv("AGY2API_KEY", "")).strip()
        self.timeout = kwargs.get("timeout")
        self._stateful = (os.getenv("AGY2API_STATEFUL", "false") or "").lower() in ("1", "true", "yes", "on")
        self._sessions = SessionStore(200) if self._stateful else None
        self._lock = threading.Lock()
        self.is_closed = False
        self.chat = _ChatNamespace(self)

    def __enter__(self) -> "AgyCli2ApiClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        self.is_closed = True

    def _effective_timeout(self, call_timeout: Any) -> float:
        raw = call_timeout if call_timeout is not None else self.timeout
        if hasattr(raw, "read"):
            read_val = getattr(raw, "read")
            raw = 1800.0 if read_val is None else read_val
        try:
            val = float(raw)
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass
        env_t = os.getenv("AGY_TIMEOUT", "")
        if env_t:
            try:
                val = float(env_t)
                if val > 0:
                    return val
            except ValueError:
                pass
        return _DEFAULT_TIMEOUT_SECONDS

    def _http_gateway_url(self) -> Optional[str]:
        env_url = os.getenv("AGY_CLI2API_BASE_URL", "").strip()
        candidate = env_url or self.base_url
        if candidate.startswith(("http://", "https://")):
            return candidate.rstrip("/")
        return None

    def _create_via_http(
        self,
        *,
        gateway_url: str,
        model: str,
        messages: List[dict[str, Any]],
        tools: Optional[List[dict[str, Any]]],
        tool_choice: Any,
        stream: bool,
        timeout: float,
    ) -> Any:
        prompt_text = format_messages_prompt(messages, tools=tools, tool_choice=tool_choice)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
            "stream": False,
        }
        req = urllib.request.Request(
            f"{gateway_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self.api_key and self.api_key != "agy-cli2api":
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"HTTP {exc.code} from agy-cli2api gateway: {sanitize_error_text(body)}") from exc

        choices = data.get("choices") or []
        msg = (choices[0].get("message") or {}) if choices else {}
        raw_answer = str(msg.get("content") or "")
        reasoning = str(msg.get("reasoning_content") or "")
        clean_text, parsed_calls = extract_tool_calls(raw_answer)
        finish_reason = "tool_calls" if parsed_calls else str(choices[0].get("finish_reason") or "stop") if choices else "stop"

        completion = _build_completion_objects(
            completion_id=str(data.get("id") or f"agy-{int(time.time() * 1000)}"),
            model=model,
            content=clean_text,
            reasoning=reasoning,
            tool_calls=parsed_calls,
            finish_reason=finish_reason,
        )
        return _completion_to_stream(completion) if stream else completion

    def _create_chat_completion(
        self,
        *,
        model: Optional[str] = None,
        messages: Optional[List[dict[str, Any]]] = None,
        tools: Optional[List[dict[str, Any]]] = None,
        tool_choice: Any = None,
        stream: bool = False,
        timeout: Any = None,
        reasoning_effort: Optional[str] = None,
        **extra_kwargs: Any,
    ) -> Any:
        if self.is_closed:
            raise RuntimeError("AgyCli2ApiClient is closed.")

        exposed_model = resolve_model_id(model or DEFAULT_MODEL)
        effort = reasoning_effort or extra_kwargs.get("reasoning_effort")
        agy_label = resolve_agy_label(exposed_model, effort)
        msg_list = list(messages or [])
        eff_timeout = self._effective_timeout(timeout)

        gateway_url = self._http_gateway_url()
        if gateway_url:
            return self._create_via_http(
                gateway_url=gateway_url,
                model=exposed_model,
                messages=msg_list,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream,
                timeout=eff_timeout,
            )

        full_prompt = format_messages_prompt(msg_list, tools=tools, tool_choice=tool_choice)
        prompt_to_send = full_prompt
        resume_cid: Optional[str] = None
        sigs: List[str] = []

        if self._sessions is not None:
            sigs = [
                fingerprint(str(m.get("role", "user")), json.dumps(m.get("content", ""), sort_keys=True, default=str))
                for m in msg_list
                if isinstance(m, dict)
            ]
            from .account_manager import get_active_account

            cur_acc = get_active_account()
            with self._lock:
                plan = self._sessions.lookup(sigs, cur_acc)
            if plan.conversation_id and plan.prefix_len < len(msg_list):
                delta = msg_list[plan.prefix_len :]
                while delta and str(delta[0].get("role", "")).lower() == "assistant":
                    delta = delta[1:]
                if delta:
                    resume_cid = plan.conversation_id
                    prompt_to_send = format_delta_prompt(delta)

        try:
            result = run_agy_with_autoswap(
                prompt=prompt_to_send,
                model_label=agy_label,
                conversation_id=resume_cid,
                full_prompt_fallback=full_prompt,
                timeout_seconds=eff_timeout,
                keep_db=self._stateful,
            )
        except Exception as exc:
            if self._sessions is not None and resume_cid:
                with self._lock:
                    self._sessions.forget(resume_cid)
            raise RuntimeError(sanitize_error_text(str(exc))) from exc

        if self._sessions is not None and sigs and result.db_path:
            with self._lock:
                self._sessions.remember(result.db_path.stem, result.account, sigs)

        clean_text, parsed_calls = extract_tool_calls(result.response.answer)
        if parsed_calls:
            finish_reason = "tool_calls"
        elif result.response.truncated or not clean_text:
            finish_reason = "length" if result.response.truncated else "stop"
        else:
            finish_reason = "stop"

        completion = _build_completion_objects(
            completion_id=result.db_path.stem if result.db_path else f"agy-{int(time.time() * 1000)}",
            model=exposed_model,
            content=clean_text,
            reasoning=result.response.reasoning,
            tool_calls=parsed_calls,
            finish_reason=finish_reason,
        )
        return _completion_to_stream(completion) if stream else completion
