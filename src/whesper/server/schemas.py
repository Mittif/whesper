from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    detail: dict | None = None


class HealthResponse(BaseModel):
    ok: bool = True
    version: str


class CapabilitiesResponse(BaseModel):
    version: str
    features: list[str]
    route_modes: list[str]
    websearch_providers: list[str]


class SessionCreateRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)


class SessionSummaryResponse(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    pinned_model: str
    message_count: int
    has_pending_ask_user: bool


class SessionListResponse(BaseModel):
    items: list[SessionSummaryResponse]
    next_before: str | None = None


class OkResponse(BaseModel):
    ok: bool = True


class TurnRequest(BaseModel):
    text: str = Field(..., min_length=1)
    mode: str = Field(default="auto")


class RouteDecisionModel(BaseModel):
    model_alias: str
    mode: str
    reason: str


class AskUserOptionModel(BaseModel):
    label: str
    value: str
    description: str | None = None


class AskUserActionModel(BaseModel):
    prompt: str
    options: list[AskUserOptionModel] = Field(default_factory=list)
    allow_free_text: bool = True
    field_name: str | None = None


class ChatMessageModel(BaseModel):
    role: str
    content: str
    created_at: str
    model_alias: str | None = None
    route_reason: str | None = None
    reasoning_content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    source_profile: str | None = None


class TurnResponse(BaseModel):
    decision: RouteDecisionModel
    assistant_message: ChatMessageModel
    ask_user: AskUserActionModel | None = None


class MessageListResponse(BaseModel):
    items: list[ChatMessageModel]
    next_before: str | None = None
