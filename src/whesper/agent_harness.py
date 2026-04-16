from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Callable

from whesper.agent_types import (
    AgentCompletion,
    AgentStep,
    AskUserAction,
    AskUserOption,
    ToolInvocation,
)
from whesper.message_builder import SessionMessageBuilder
from whesper.router import RouteDecision
from whesper.session import ChatMessage, ConversationSession, utc_now_iso
from whesper.tool_executor import ToolExecutor
from whesper.tools import ToolExecutionError


@dataclass(slots=True)
class HarnessRunResult:
    completion: AgentCompletion
    used_tools: bool = False
    final_messages: list[dict[str, object]] | None = None
    ask_user: AskUserAction | None = None
    steps: list[AgentStep] = field(default_factory=list)


@dataclass(slots=True)
class AgentRunState:
    working_messages: list[dict[str, object]]
    tool_choice: str | dict[str, object] | None
    used_tools: bool = False
    consecutive_errors: int = 0
    steps: list[AgentStep] = field(default_factory=list)
    last_completion: AgentCompletion | None = None
    ensure_reasoning_content: bool = False


StepCallback = Callable[[AgentStep], None]
CompletionRequester = Callable[..., AgentCompletion]
TraceEmitter = Callable[..., None]
ToolChoiceBuilder = Callable[..., str | dict[str, object] | None]
ReasoningContentResolver = Callable[[RouteDecision, AgentCompletion], str | None]
ToolCallPayloadBuilder = Callable[[ToolInvocation], dict[str, object]]

# Injected after tool errors so the model can adapt its strategy.
_ERROR_RECOVERY_PROMPT = (
    "The tool call above returned an error. Consider the following recovery strategies:\n"
    "1. Try an alternative tool that can provide similar information.\n"
    "2. Use information already obtained in earlier steps.\n"
    "3. If the user's memory or conversation history contains relevant context, use it.\n"
    "4. If no alternative exists, inform the user what went wrong and ask for manual input.\n"
    "Do NOT repeat the exact same tool call that just failed."
)

# Planning guidance appended to the system prompt when tools are available.
AGENTIC_PLANNING_PROMPT = """\

## Task Decomposition & Tool Use

When the user's request requires external data or multiple pieces of information:
1. Identify what information you need and in what order to obtain it.
2. Call tools step by step. Each tool result informs the next step.
3. If a tool call fails, adapt:
   - Try an alternative tool or parameter.
   - Fall back to information already available (memory, conversation history, earlier tool results).
   - Tell the user what failed and ask them to provide the missing information.
4. After gathering all needed data, synthesize a concise final answer.

Examples of multi-step workflows:
- "最近天气怎么样" -> get_local_weather -> answer
- "帮我查一下美元兑日元汇率和东京天气" -> lookup_exchange_rate + get_weather_by_location in sequence -> answer

Always proceed to the next tool call directly. Do NOT narrate your plan or say "让我查一下" without actually calling a tool."""

ASK_USER_FORMAT_PROMPT = """\

## Asking The User To Continue

If you need the user to choose or provide a missing value before you can continue:
- Do not ask in plain prose.
- Output exactly one <ask_user>...</ask_user> block and nothing else.
- Inside the block, emit valid JSON with this shape:
  {"prompt":"short question","options":[{"label":"short option","value":"text to continue with","description":"optional short hint"}],"allow_free_text":true,"field_name":"optional field name"}
- Keep options short and actionable. Use 2-4 options when they would help.
- Set allow_free_text to true unless the user must choose one of the options exactly.
"""

INTERNAL_CONTINUE_PROMPT = (
    "Continue the same turn internally. Do not narrate that you will search or check. "
    "If fresh information is needed, call an appropriate tool now. Otherwise provide the "
    "final user-facing answer directly."
)

INTERIM_TOOL_RESPONSE_MARKERS = (
    "我来帮你查",
    "我来查",
    "让我看看",
    "让我查",
    "让我先搜索",
    "让我搜索",
    "为你搜索",
    "帮你搜索",
    "帮你查一下",
    "搜索一下",
    "正在搜索",
    "正在查找",
    "马上帮你",
    "马上为你",
    "稍等",
    "稍等一下",
    "正在帮你",
    "i'll check",
    "i'll search",
    "i'll look",
    "let me check",
    "let me look",
    "let me search",
    "let me find",
    "one moment",
    "searching for",
)

TOOLLESS_CONTINUE_PROMPT = (
    "You do not have search or browsing tools in this session. "
    "Please answer the user's question directly based on your existing knowledge. "
    "Do not say you will search or check — provide the answer now."
)

_ASK_USER_TAG_PATTERN = re.compile(r"<ask_user>\s*(\{.*?\})\s*</ask_user>", re.DOTALL)


def parse_ask_user_action(content: str) -> AskUserAction | None:
    stripped = content.strip()
    if not stripped:
        return None

    payload_text: str | None = None
    tagged_match = _ASK_USER_TAG_PATTERN.fullmatch(stripped)
    if tagged_match is not None:
        payload_text = tagged_match.group(1)
    elif stripped.startswith("{") and stripped.endswith("}"):
        payload_text = stripped

    if payload_text is None:
        return None

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("type") not in {None, "ask_user"}:
        return None

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None

    options_raw = payload.get("options", [])
    options: list[AskUserOption] = []
    if isinstance(options_raw, list):
        for item in options_raw:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            if not isinstance(label, str) or not label.strip():
                continue
            value = item.get("value")
            if not isinstance(value, str) or not value.strip():
                value = label
            description = item.get("description")
            options.append(
                AskUserOption(
                    label=label.strip(),
                    value=value.strip(),
                    description=(
                        str(description).strip()
                        if description is not None and str(description).strip()
                        else None
                    ),
                )
            )

    allow_free_text = payload.get("allow_free_text", True)
    field_name = payload.get("field_name")
    return AskUserAction(
        prompt=prompt.strip(),
        options=tuple(options),
        allow_free_text=bool(allow_free_text),
        field_name=str(field_name).strip() if isinstance(field_name, str) and field_name.strip() else None,
    )


class AgentHarness:
    def __init__(
        self,
        *,
        message_builder: SessionMessageBuilder,
        tool_executor: ToolExecutor,
        completion_requester: CompletionRequester,
        tool_call_payload_builder: ToolCallPayloadBuilder,
        tool_choice_builder: ToolChoiceBuilder,
        reasoning_content_resolver: ReasoningContentResolver,
        trace_emitter: TraceEmitter | None = None,
        max_rounds: int = 10,
    ) -> None:
        self.message_builder = message_builder
        self.tool_executor = tool_executor
        self.completion_requester = completion_requester
        self.tool_call_payload_builder = tool_call_payload_builder
        self.tool_choice_builder = tool_choice_builder
        self.reasoning_content_resolver = reasoning_content_resolver
        self.trace_emitter = trace_emitter
        self.max_rounds = max_rounds

    def run_until_final(
        self,
        session: ConversationSession,
        decision: RouteDecision,
        *,
        provider,
        model,
        model_system_prompt: str | None,
        user_text: str,
        route_mode: str,
        target_profile,
        tools: list[dict[str, object]],
        on_step: StepCallback | None = None,
        ensure_reasoning_content: bool = False,
    ) -> HarnessRunResult:
        state = AgentRunState(
            working_messages=self.message_builder.build_messages(
                session,
                model_system_prompt,
                target_profile=target_profile,
                user_text=user_text,
                route_mode=route_mode,
                planning_prompt=f"{AGENTIC_PLANNING_PROMPT}\n\n{ASK_USER_FORMAT_PROMPT}",
                include_live_context=False,
                ensure_reasoning_content=ensure_reasoning_content,
            ),
            tool_choice=self.tool_choice_builder(
                session,
                user_text,
                tools,
                route_mode=route_mode,
            ),
            ensure_reasoning_content=ensure_reasoning_content,
        )

        for round_index in range(self.max_rounds):
            completion = self.completion_requester(
                session_id=session.session_id,
                decision=decision,
                provider=provider,
                model=model,
                messages=state.working_messages,
                tools=tools,
                tool_choice=state.tool_choice,
                streamed=False,
            )
            state.last_completion = completion

            if completion.tool_calls:
                should_stop = self._handle_tool_calls(
                    session,
                    state,
                    decision,
                    provider_name=provider.name,
                    round_index=round_index,
                    completion=completion,
                    tools=tools,
                    model_system_prompt=model_system_prompt,
                    user_text=user_text,
                    route_mode=route_mode,
                    target_profile=target_profile,
                    on_step=on_step,
                )
                if should_stop:
                    break
                continue

            ask_user = parse_ask_user_action(completion.content)
            if ask_user is not None:
                self._emit_trace(
                    kind="ask_user",
                    session_id=session.session_id,
                    decision=decision,
                    provider_name=provider.name,
                    streamed=False,
                    tools_enabled=True,
                    tool_choice=self._stringify_tool_choice(state.tool_choice),
                    preview=ask_user.prompt,
                )
                self._record_step(
                    state,
                    AgentStep(
                        round_index=round_index,
                        kind="ask_user",
                        summary=ask_user.prompt,
                    ),
                    on_step=on_step,
                )
                return HarnessRunResult(
                    completion=completion,
                    used_tools=state.used_tools,
                    ask_user=ask_user,
                    steps=state.steps,
                )

            if self.should_continue_tool_loop(
                completion.content,
                route_mode=route_mode,
                round_index=round_index,
            ):
                self._emit_trace(
                    kind="planning_retry",
                    session_id=session.session_id,
                    decision=decision,
                    provider_name=provider.name,
                    streamed=False,
                    tools_enabled=True,
                    tool_choice=self._stringify_tool_choice(state.tool_choice),
                    preview=completion.content,
                )
                state.working_messages = [
                    *state.working_messages,
                    {"role": "assistant", "content": completion.content},
                    {"role": "user", "content": INTERNAL_CONTINUE_PROMPT},
                ]
                self._record_step(
                    state,
                    AgentStep(
                        round_index=round_index,
                        kind="planning_retry",
                        summary="model narrated intent without calling tool, nudging",
                    ),
                    on_step=on_step,
                )
                continue

            self._record_step(
                state,
                AgentStep(
                    round_index=round_index,
                    kind="final",
                    summary="final answer",
                ),
                on_step=on_step,
            )
            return HarnessRunResult(
                completion=completion,
                used_tools=state.used_tools,
                final_messages=list(state.working_messages) if state.used_tools else None,
                steps=state.steps,
            )

        self._emit_trace(
            kind="tool_loop_limit",
            session_id=session.session_id,
            decision=decision,
            provider_name=provider.name,
            streamed=False,
            tools_enabled=True,
            tool_choice=self._stringify_tool_choice(state.tool_choice),
            note=f"max_rounds={self.max_rounds}",
        )

        if state.last_completion is not None and state.last_completion.content.strip():
            return HarnessRunResult(
                completion=state.last_completion,
                used_tools=state.used_tools,
                steps=state.steps,
            )

        return HarnessRunResult(
            completion=AgentCompletion(
                content="我刚才没有顺利拿到完整结果。要不要我换个方式继续帮你查？",
                raw_response={},
            ),
            used_tools=state.used_tools,
            steps=state.steps,
        )

    def should_continue_tool_loop(
        self,
        content: str,
        *,
        route_mode: str,
        round_index: int,
    ) -> bool:
        if round_index >= self.max_rounds - 1:
            return False
        normalized = " ".join(content.split()).casefold()
        if not normalized:
            return False
        for marker in INTERIM_TOOL_RESPONSE_MARKERS:
            if marker.casefold() in normalized:
                return True
        if len(normalized) < 60 and route_mode in ("search", "auto", "chat"):
            planning_cues = (
                "首先",
                "第一步",
                "接下来",
                "然后",
                "step 1",
                "first",
                "next",
                "i need to",
            )
            if any(cue in normalized for cue in planning_cues):
                return True
        return False

    def _handle_tool_calls(
        self,
        session: ConversationSession,
        state: AgentRunState,
        decision: RouteDecision,
        *,
        provider_name: str,
        round_index: int,
        completion: AgentCompletion,
        tools: list[dict[str, object]],
        model_system_prompt: str | None,
        user_text: str,
        route_mode: str,
        target_profile,
        on_step: StepCallback | None,
    ) -> bool:
        state.used_tools = True
        source_profile_id = (
            getattr(target_profile, "profile_id", None)
            if target_profile is not None
            else None
        )
        self._append_assistant_tool_message(
            session,
            decision,
            completion,
            source_profile_id=source_profile_id,
        )
        self._emit_trace(
            kind="tool_calls",
            session_id=session.session_id,
            decision=decision,
            provider_name=provider_name,
            streamed=False,
            tools_enabled=True,
            tool_choice=self._stringify_tool_choice(state.tool_choice),
            note=", ".join(item.name for item in completion.tool_calls),
            preview=completion.content or None,
        )

        for tool_call in completion.tool_calls:
            self._record_step(
                state,
                AgentStep(
                    round_index=round_index,
                    kind="tool_call",
                    tool_name=tool_call.name,
                    summary=f"call {tool_call.name}",
                ),
                on_step=on_step,
            )

        has_error = False
        for message in self._execute_tool_calls(
            session,
            decision,
            provider_name=provider_name,
            tool_calls=completion.tool_calls,
            tool_choice=state.tool_choice,
            source_profile_id=source_profile_id,
        ):
            is_error = self._is_error_content(message.content)
            has_error = has_error or is_error
            self._record_step(
                state,
                AgentStep(
                    round_index=round_index,
                    kind="tool_result",
                    tool_name=message.name,
                    summary=message.content[:120] if message.content else "",
                    is_error=is_error,
                ),
                on_step=on_step,
            )

        state.consecutive_errors = state.consecutive_errors + 1 if has_error else 0
        if state.consecutive_errors >= 3:
            self._record_step(
                state,
                AgentStep(
                    round_index=round_index,
                    kind="error_recovery",
                    summary="too many consecutive tool errors, stopping loop",
                    is_error=True,
                ),
                on_step=on_step,
            )
            return True

        state.working_messages = self.message_builder.build_messages(
            session,
            model_system_prompt,
            target_profile=target_profile,
            user_text=user_text,
            route_mode=route_mode,
            planning_prompt=f"{AGENTIC_PLANNING_PROMPT}\n\n{ASK_USER_FORMAT_PROMPT}",
            include_live_context=False,
            ensure_reasoning_content=state.ensure_reasoning_content,
        )
        if has_error:
            state.working_messages = [
                *state.working_messages,
                {"role": "user", "content": _ERROR_RECOVERY_PROMPT},
            ]
            self._record_step(
                state,
                AgentStep(
                    round_index=round_index,
                    kind="error_recovery",
                    summary="injected recovery prompt after tool error",
                ),
                on_step=on_step,
            )
        return False

    def _append_assistant_tool_message(
        self,
        session: ConversationSession,
        decision: RouteDecision,
        completion: AgentCompletion,
        *,
        source_profile_id: str | None = None,
    ) -> None:
        session.append(
            ChatMessage(
                role="assistant",
                content="",
                created_at=utc_now_iso(),
                model_alias=decision.model_alias,
                route_reason=decision.reason,
                reasoning_content=self.reasoning_content_resolver(decision, completion),
                tool_calls=[self.tool_call_payload_builder(item) for item in completion.tool_calls],
                source_profile=source_profile_id,
            )
        )

    def _execute_tool_calls(
        self,
        session: ConversationSession,
        decision: RouteDecision,
        *,
        provider_name: str,
        tool_calls: tuple[ToolInvocation, ...],
        tool_choice: str | dict[str, object] | None,
        source_profile_id: str | None = None,
    ) -> list[ChatMessage]:
        tool_messages: list[ChatMessage] = []
        for tool_call in tool_calls:
            try:
                result = self.tool_executor.execute(tool_call)
                content = result.content
                name = result.name
                tool_call_id = result.tool_call_id
            except ToolExecutionError as exc:
                content = str(exc)
                name = tool_call.name
                tool_call_id = tool_call.tool_call_id
            message = ChatMessage(
                role="tool",
                content=content,
                created_at=utc_now_iso(),
                name=name,
                tool_call_id=tool_call_id,
                source_profile=source_profile_id,
            )
            session.append(message)
            tool_messages.append(message)
            self._emit_trace(
                kind="tool_result",
                session_id=session.session_id,
                decision=decision,
                provider_name=provider_name,
                streamed=False,
                tools_enabled=True,
                tool_choice=self._stringify_tool_choice(tool_choice),
                note=name,
                preview=content,
            )
        return tool_messages

    def _record_step(
        self,
        state: AgentRunState,
        step: AgentStep,
        *,
        on_step: StepCallback | None,
    ) -> None:
        state.steps.append(step)
        if on_step is not None:
            on_step(step)

    def _emit_trace(self, **kwargs: object) -> None:
        if self.trace_emitter is None:
            return
        self.trace_emitter(**kwargs)

    @staticmethod
    def _stringify_tool_choice(tool_choice: str | dict[str, object] | None) -> str | None:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            return tool_choice
        return str(tool_choice)

    @staticmethod
    def _is_error_content(content: str) -> bool:
        if not content:
            return False
        lowered = content.casefold()
        return (
            "ToolExecutionError" in content
            or "failed:" in lowered
            or "error:" in lowered
            or '"is_error": true' in lowered
        )
