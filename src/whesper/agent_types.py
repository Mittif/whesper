from __future__ import annotations

from dataclasses import dataclass
import json


@dataclass(slots=True, frozen=True)
class ToolInvocation:
    tool_call_id: str
    name: str
    arguments_json: str

    def arguments(self) -> dict[str, object]:
        if not self.arguments_json:
            return {}
        parsed = json.loads(self.arguments_json)
        if not isinstance(parsed, dict):
            raise TypeError("Tool arguments must decode to a JSON object.")
        return parsed


@dataclass(slots=True)
class AgentCompletion:
    content: str
    raw_response: dict
    tool_calls: tuple[ToolInvocation, ...] = ()
    reasoning_content: str | None = None


@dataclass(slots=True, frozen=True)
class ToolExecutionMeta:
    parallel_safe: bool = True
    side_effectful: bool = False
    retryable: bool = False
    timeout_seconds: float = 15.0
    source: str = "local"
    visible_to_model: bool = True


@dataclass(slots=True)
class ToolObservation:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(slots=True)
class AgentStep:
    """A single step in the agentic tool-calling loop, used for observability."""

    round_index: int
    kind: str  # "tool_call", "tool_result", "planning_retry", "error_recovery", "final"
    tool_name: str | None = None
    summary: str = ""
    is_error: bool = False
