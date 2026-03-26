from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from whesper.agent_types import AgentCompletion, AgentStep, ToolInvocation
from whesper.message_builder import SessionMessageBuilder
from whesper.router import RouteDecision
from whesper.session import ChatMessage, ConversationSession, utc_now_iso
from whesper.tool_executor import ToolExecutor
from whesper.tools import ToolExecutionError


@dataclass(slots=True)
class HarnessRunResult:
    completion: AgentCompletion
    used_tools: bool = False
    steps: list[AgentStep] = field(default_factory=list)


@dataclass(slots=True)
class AgentRunState:
    working_messages: list[dict[str, object]]
    tool_choice: str | dict[str, object] | None
    used_tools: bool = False
    consecutive_errors: int = 0
    steps: list[AgentStep] = field(default_factory=list)
    last_completion: AgentCompletion | None = None


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
- "最近天气怎么样" -> get_public_ip -> get_ip_location(ip) -> get_weather_by_location(city) -> answer
- "帮我查一下美元兑日元汇率和东京天气" -> lookup_exchange_rate + get_weather_by_location in sequence -> answer

Always proceed to the next tool call directly. Do NOT narrate your plan or say "让我查一下" without actually calling a tool."""

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
    "稍等",
    "稍等一下",
    "正在帮你",
    "i'll check",
    "let me check",
    "let me look",
    "let me search",
    "one moment",
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
        tool_message_format,
        tools: list[dict[str, object]],
        on_step: StepCallback | None = None,
    ) -> HarnessRunResult:
        state = AgentRunState(
            working_messages=self.message_builder.build_messages(
                session,
                model_system_prompt,
                tool_message_format=tool_message_format,
                user_text=user_text,
                route_mode=route_mode,
                planning_prompt=AGENTIC_PLANNING_PROMPT,
            ),
            tool_choice=self.tool_choice_builder(tools, route_mode=route_mode),
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
                    tool_message_format=tool_message_format,
                    on_step=on_step,
                )
                if should_stop:
                    break
                continue

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
        tool_message_format,
        on_step: StepCallback | None,
    ) -> bool:
        state.used_tools = True
        self._append_assistant_tool_message(session, decision, completion)
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
            tool_message_format=tool_message_format,
            user_text=user_text,
            route_mode=route_mode,
            planning_prompt=AGENTIC_PLANNING_PROMPT,
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
