"""Pydantic models for the small OpenAI-compatible surface."""

from __future__ import annotations

from typing import Any, Literal, Optional, List, Dict

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    model: str = "gemini-2.5-pro"
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "agy2api"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelInfo]


class ErrorBody(BaseModel):
    error: Dict[str, Any] = Field(default_factory=dict)
