from __future__ import annotations

from dataclasses import asdict
from typing import Any

from whesper.agent_types import AgentStep, AskUserAction
from whesper.chat import ChatTurnResult
from whesper.router import RouteDecision
from whesper.session import ChatMessage


# Event names are the v1 SSE wire contract. Clients pin on these strings.
EVENT_ROUTE = "route"
EVENT_STEP = "step"
EVENT_CHUNK = "chunk"
EVENT_ASK_USER = "ask_user"
EVENT_FINAL = "final"
EVENT_ERROR = "error"
EVENT_DONE = "done"


def chat_message_to_dict(message: ChatMessage) -> dict[str, Any]:
    return asdict(message)


def ask_user_to_dict(action: AskUserAction) -> dict[str, Any]:
    return asdict(action)


def decision_to_dict(decision: RouteDecision) -> dict[str, Any]:
    return asdict(decision)


def step_to_dict(step: AgentStep) -> dict[str, Any]:
    return asdict(step)


def turn_result_to_dict(result: ChatTurnResult) -> dict[str, Any]:
    return {
        "decision": decision_to_dict(result.decision),
        "assistant_message": chat_message_to_dict(result.assistant_message),
        "ask_user": ask_user_to_dict(result.ask_user) if result.ask_user is not None else None,
    }
