from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from whesper.agent_harness import (
    AGENTIC_PLANNING_PROMPT,
    AgentHarness,
    StepCallback,
    TOOLLESS_CONTINUE_PROMPT,
    parse_ask_user_action,
)
from whesper.agent_types import AskUserAction
from whesper.client import CompletionResult, OpenAICompatibleClient, ProviderError, ToolCall
from whesper.config import AppConfig
from whesper.live_data import LiveContextService
from whesper.message_builder import SessionMessageBuilder
from whesper.memory import MemoryService
from whesper.provider_profile import ProviderProfile, resolve_profile
from whesper.router import RouteDecision, select_model
from whesper.session import ChatMessage, ConversationSession, utc_now_iso
from whesper.tool_executor import ToolExecutor
from whesper.tool_protocol import DefaultToolProtocolAdapter, sanitize_tool_call_artifacts
from whesper.trace import TraceStore, make_trace_event
from whesper.tools import (
    ToolRegistry,
    _matches_cup_control_intent,
    _matches_cup_scene_followup,
)


@dataclass(slots=True)
class ChatTurnResult:
    decision: RouteDecision
    assistant_message: ChatMessage
    ask_user: AskUserAction | None = None


class GenerationInterrupted(RuntimeError):
    def __init__(self, partial_result: ChatTurnResult | None = None) -> None:
        super().__init__("Generation interrupted.")
        self.partial_result = partial_result


class ChatService:
    MAX_AGENT_ROUNDS = 10

    def __init__(
        self,
        config: AppConfig,
        client: OpenAICompatibleClient | None = None,
        memory_service: MemoryService | None = None,
        live_context_service: LiveContextService | None = None,
        tool_registry: ToolRegistry | None = None,
        trace_store: TraceStore | None = None,
    ) -> None:
        self.config = config
        self.client = client or OpenAICompatibleClient()
        self.memory_service = memory_service
        self.live_context_service = live_context_service or LiveContextService.from_config(config)
        self.tool_registry = tool_registry or ToolRegistry.default(config)
        self.tool_protocol_adapter = DefaultToolProtocolAdapter()
        self.message_builder = SessionMessageBuilder(
            config,
            memory_service=self.memory_service,
            live_context_service=self.live_context_service,
            tool_registry=self.tool_registry,
        )
        self.tool_executor = ToolExecutor(self.tool_registry)
        self.trace_store = trace_store
        self._tool_unsupported_providers: set[str] = set()
        self.harness = AgentHarness(
            message_builder=self.message_builder,
            tool_executor=self.tool_executor,
            completion_requester=self._client_create_chat_completion,
            tool_call_payload_builder=self._tool_call_payload,
            tool_choice_builder=self._tool_choice_for_request,
            reasoning_content_resolver=self._tool_message_reasoning_content,
            trace_emitter=self._append_trace,
            max_rounds=self.MAX_AGENT_ROUNDS,
        )

    def send(
        self,
        session: ConversationSession,
        user_text: str,
        *,
        mode_override: str = "auto",
        on_step: StepCallback | None = None,
    ) -> ChatTurnResult:
        resolved_user_text = self._resolve_pending_ask_response(session, user_text)
        self._append_user_message(session, resolved_user_text)
        session.pending_ask_user = None
        if self.memory_service is not None:
            self.memory_service.capture_user_message(session.session_id, resolved_user_text)
        return self._complete_turn(
            session,
            resolved_user_text,
            mode_override=mode_override,
            on_step=on_step,
        )

    def send_stream(
        self,
        session: ConversationSession,
        user_text: str,
        *,
        mode_override: str = "auto",
        on_chunk: Callable[[str], None] | None = None,
        on_step: StepCallback | None = None,
    ) -> ChatTurnResult:
        resolved_user_text = self._resolve_pending_ask_response(session, user_text)
        self._append_user_message(session, resolved_user_text)
        session.pending_ask_user = None
        if self.memory_service is not None:
            self.memory_service.capture_user_message(session.session_id, resolved_user_text)
        return self._complete_turn_stream(
            session,
            resolved_user_text,
            mode_override=mode_override,
            on_chunk=on_chunk,
            on_step=on_step,
        )

    def retry_stream(
        self,
        session: ConversationSession,
        *,
        mode_override: str = "auto",
        on_chunk: Callable[[str], None] | None = None,
        on_step: StepCallback | None = None,
    ) -> ChatTurnResult:
        last_user_index = self._last_user_index(session)
        if last_user_index is None:
            raise ValueError("No user message is available to retry yet.")

        last_user_message = session.messages[last_user_index]
        del session.messages[last_user_index + 1 :]
        transcript_last_user_index = self._last_user_index_in_messages(session.transcript_messages)
        if transcript_last_user_index is not None:
            del session.transcript_messages[transcript_last_user_index + 1 :]
        self._sync_session_updated_at(session)
        return self._complete_turn_stream(
            session,
            last_user_message.content,
            mode_override=mode_override,
            on_chunk=on_chunk,
            on_step=on_step,
        )

    def _complete_turn(
        self,
        session: ConversationSession,
        user_text: str,
        *,
        mode_override: str = "auto",
        on_step: StepCallback | None = None,
    ) -> ChatTurnResult:
        decision = select_model(
            self.config,
            user_text,
            pinned_model=session.pinned_model,
            mode_override=mode_override,
        )

        model_config = self.config.get_model(decision.model_alias)
        provider_config = self.config.get_provider(model_config.provider)
        target_profile = resolve_profile(provider_config, model_config)
        tools = self._tools_for_request(
            session=session,
            provider=provider_config,
            model=model_config,
            user_text=user_text,
            route_mode=decision.mode,
        )

        needs_reasoning = self._requires_reasoning_content_replay(target_profile, model_config)
        messages = self._build_messages(
            session,
            model_config.system_prompt,
            target_profile=target_profile,
            user_text=user_text,
            route_mode=decision.mode,
            include_live_context=tools is None,
            ensure_reasoning_content=needs_reasoning,
        )
        if tools is not None:
            run_result = self.harness.run_until_final(
                session,
                decision,
                provider=provider_config,
                model=model_config,
                model_system_prompt=model_config.system_prompt,
                user_text=user_text,
                route_mode=decision.mode,
                tools=tools,
                target_profile=target_profile,
                on_step=on_step,
                ensure_reasoning_content=needs_reasoning,
            )
            completion = run_result.completion
            ask_user = run_result.ask_user
        else:
            completion = self._client_create_chat_completion(
                session_id=session.session_id,
                decision=decision,
                provider=provider_config,
                model=model_config,
                messages=messages,
                streamed=False,
            )
            if self.harness.should_continue_tool_loop(
                completion.content, route_mode=decision.mode, round_index=0,
            ):
                messages = [
                    *messages,
                    {"role": "assistant", "content": completion.content},
                    {"role": "user", "content": TOOLLESS_CONTINUE_PROMPT},
                ]
                completion = self._client_create_chat_completion(
                    session_id=session.session_id,
                    decision=decision,
                    provider=provider_config,
                    model=model_config,
                    messages=messages,
                    streamed=False,
                )
            ask_user = parse_ask_user_action(completion.content)
        return self._append_assistant_message(
            session,
            decision,
            ask_user.prompt if ask_user is not None else completion.content,
            ask_user=ask_user,
            target_profile=target_profile,
        )

    def _complete_turn_stream(
        self,
        session: ConversationSession,
        user_text: str,
        *,
        mode_override: str = "auto",
        on_chunk: Callable[[str], None] | None = None,
        on_step: StepCallback | None = None,
    ) -> ChatTurnResult:

        decision = select_model(
            self.config,
            user_text,
            pinned_model=session.pinned_model,
            mode_override=mode_override,
        )

        model_config = self.config.get_model(decision.model_alias)
        provider_config = self.config.get_provider(model_config.provider)
        target_profile = resolve_profile(provider_config, model_config)
        tools = self._tools_for_request(
            session=session,
            provider=provider_config,
            model=model_config,
            user_text=user_text,
            route_mode=decision.mode,
        )
        needs_reasoning = self._requires_reasoning_content_replay(target_profile, model_config)
        messages = self._build_messages(
            session,
            model_config.system_prompt,
            target_profile=target_profile,
            user_text=user_text,
            route_mode=decision.mode,
            include_live_context=tools is None,
            ensure_reasoning_content=needs_reasoning,
        )
        if tools is None:
            result = self._stream_final_answer(
                session,
                decision,
                provider=provider_config,
                model=model_config,
                messages=messages,
                on_chunk=on_chunk,
                target_profile=target_profile,
            )
            if self.harness.should_continue_tool_loop(
                result.assistant_message.content, route_mode=decision.mode, round_index=0,
            ):
                messages = [
                    *messages,
                    {"role": "assistant", "content": result.assistant_message.content},
                    {"role": "user", "content": TOOLLESS_CONTINUE_PROMPT},
                ]
                if on_chunk is not None:
                    on_chunk("\n\n")
                result = self._stream_final_answer(
                    session,
                    decision,
                    provider=provider_config,
                    model=model_config,
                    messages=messages,
                    on_chunk=on_chunk,
                    target_profile=target_profile,
                )
            return result
        run_result = self.harness.run_until_final(
            session,
            decision,
            provider=provider_config,
            model=model_config,
            model_system_prompt=model_config.system_prompt,
            user_text=user_text,
            route_mode=decision.mode,
            tools=tools,
            target_profile=target_profile,
            on_step=on_step,
            ensure_reasoning_content=needs_reasoning,
        )
        if run_result.final_messages is not None:
            return self._stream_final_answer(
                session,
                decision,
                provider=provider_config,
                model=model_config,
                messages=run_result.final_messages,
                on_chunk=on_chunk,
                target_profile=target_profile,
            )
        completion = run_result.completion
        ask_user = run_result.ask_user
        if ask_user is not None:
            if on_chunk is not None and ask_user.prompt:
                on_chunk(ask_user.prompt)
            return self._append_assistant_message(
                session,
                decision,
                ask_user.prompt,
                ask_user=ask_user,
                target_profile=target_profile,
            )
        if on_chunk is not None and completion.content:
            on_chunk(completion.content)
        return self._append_assistant_message(
            session,
            decision,
            completion.content,
            target_profile=target_profile,
        )

    def _last_user_index(self, session: ConversationSession) -> int | None:
        return self._last_user_index_in_messages(session.messages)

    def _last_user_index_in_messages(self, messages: list[ChatMessage]) -> int | None:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == "user":
                return index
        return None

    def _sync_session_updated_at(self, session: ConversationSession) -> None:
        if session.messages:
            session.updated_at = session.messages[-1].created_at
            return
        session.updated_at = session.created_at

    def _resolve_pending_ask_response(
        self,
        session: ConversationSession,
        user_text: str,
    ) -> str:
        pending = session.pending_ask_user
        if pending is None:
            return user_text
        normalized = user_text.strip()
        if normalized.isdigit():
            index = int(normalized) - 1
            if 0 <= index < len(pending.options):
                return pending.options[index].value
        return user_text

    def _append_user_message(self, session: ConversationSession, user_text: str) -> ChatMessage:
        user_message = ChatMessage(
            role="user",
            content=user_text,
            created_at=utc_now_iso(),
        )
        session.append(user_message)
        session.append_transcript(
            ChatMessage(
                role="user",
                content=user_text,
                created_at=user_message.created_at,
            )
        )
        return user_message

    def _build_messages(
        self,
        session: ConversationSession,
        model_system_prompt: str | None,
        *,
        target_profile: ProviderProfile,
        user_text: str,
        route_mode: str,
        include_planning_prompt: bool = True,
        include_live_context: bool = True,
        ensure_reasoning_content: bool = False,
    ) -> list[dict[str, object]]:
        planning_prompt = AGENTIC_PLANNING_PROMPT if include_planning_prompt else None
        return self.message_builder.build_messages(
            session,
            model_system_prompt,
            target_profile=target_profile,
            user_text=user_text,
            route_mode=route_mode,
            planning_prompt=planning_prompt,
            include_live_context=include_live_context,
            ensure_reasoning_content=ensure_reasoning_content,
        )

    def _tool_call_payload(self, tool_call: ToolCall) -> dict[str, object]:
        return self.tool_protocol_adapter.tool_call_payload(tool_call)

    def _client_create_chat_completion(
        self,
        *,
        session_id: str,
        decision: RouteDecision,
        provider,
        model,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        streamed: bool,
        structured_tool_arguments: bool = False,
        allow_disable_thinking: bool = True,
        force_disable_thinking: bool = False,
    ) -> CompletionResult:
        messages = self.tool_protocol_adapter.messages_with_normalized_tool_call_ids(messages)
        self._append_trace(
            kind="completion_request",
            session_id=session_id,
            decision=decision,
            provider_name=provider.name,
            streamed=streamed,
            tools_enabled=tools is not None,
            tool_choice=self._stringify_tool_choice(tool_choice),
            preview=self._trace_preview_from_messages(messages),
        )
        try:
            request_kwargs: dict[str, object] = {}
            if tools is not None:
                request_kwargs["tools"] = tools
            if tool_choice is not None:
                request_kwargs["tool_choice"] = tool_choice
            if force_disable_thinking:
                request_kwargs["disable_thinking"] = True
            elif allow_disable_thinking and self._should_disable_thinking_for_request(
                provider, model, tools
            ):
                request_kwargs["disable_thinking"] = True
            completion = self.client.create_chat_completion(
                provider=provider,
                model=model,
                messages=messages,
                **request_kwargs,
            )
            self._append_trace(
                kind="completion_response",
                session_id=session_id,
                decision=decision,
                provider_name=provider.name,
                streamed=streamed,
                tools_enabled=tools is not None,
                tool_choice=self._stringify_tool_choice(tool_choice),
                note=f"tool_calls={len(completion.tool_calls)}",
                preview=completion.content or self._tool_call_names_preview(completion.tool_calls),
            )
            return completion
        except TypeError as exc:
            message = str(exc)
            if (
                request_kwargs.get("disable_thinking")
                and "unexpected keyword argument 'disable_thinking'" in message
            ):
                self._append_trace(
                    kind="fallback",
                    session_id=session_id,
                    decision=decision,
                    provider_name=provider.name,
                    streamed=streamed,
                    tools_enabled=tools is not None,
                    tool_choice=self._stringify_tool_choice(tool_choice),
                    note="client does not accept disable_thinking; retrying without it",
                )
                return self._client_create_chat_completion(
                    session_id=session_id,
                    decision=decision,
                    provider=provider,
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    streamed=streamed,
                    structured_tool_arguments=structured_tool_arguments,
                    allow_disable_thinking=False,
                )
            if tool_choice is not None and "unexpected keyword argument 'tool_choice'" in message:
                self._append_trace(
                    kind="fallback",
                    session_id=session_id,
                    decision=decision,
                    provider_name=provider.name,
                    streamed=streamed,
                    tools_enabled=tools is not None,
                    tool_choice=self._stringify_tool_choice(tool_choice),
                    note="client does not accept tool_choice; retrying without it",
                )
                return self._client_create_chat_completion(
                    session_id=session_id,
                    decision=decision,
                    provider=provider,
                    model=model,
                    messages=messages,
                    tools=tools,
                    streamed=streamed,
                    structured_tool_arguments=structured_tool_arguments,
                )
            if "unexpected keyword argument 'tools'" not in message:
                raise
            self._tool_unsupported_providers.add(provider.name)
            self._append_trace(
                kind="fallback",
                session_id=session_id,
                decision=decision,
                provider_name=provider.name,
                streamed=streamed,
                tools_enabled=tools is not None,
                tool_choice=self._stringify_tool_choice(tool_choice),
                note="client does not accept tools; retrying without tools",
            )
            return self._client_create_chat_completion(
                session_id=session_id,
                decision=decision,
                provider=provider,
                model=model,
                messages=messages,
                tools=tools,
                streamed=streamed,
                structured_tool_arguments=structured_tool_arguments,
            )
        except ProviderError as exc:
            if (
                tool_choice is not None
                and tools is not None
                and "tool_choice" in str(exc).casefold()
            ):
                error_text = str(exc).casefold()
                if (
                    not request_kwargs.get("disable_thinking")
                    and "thinking enabled" in error_text
                ):
                    self._append_trace(
                        kind="fallback",
                        session_id=session_id,
                        decision=decision,
                        provider_name=provider.name,
                        streamed=streamed,
                        tools_enabled=True,
                        tool_choice=self._stringify_tool_choice(tool_choice),
                        note=(
                            "provider rejected tool_choice with thinking enabled; "
                            f"retrying with disable_thinking ({exc})"
                        ),
                    )
                    return self._client_create_chat_completion(
                        session_id=session_id,
                        decision=decision,
                        provider=provider,
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        streamed=streamed,
                        structured_tool_arguments=structured_tool_arguments,
                        allow_disable_thinking=False,
                        force_disable_thinking=True,
                    )
                self._append_trace(
                    kind="fallback",
                    session_id=session_id,
                    decision=decision,
                    provider_name=provider.name,
                    streamed=streamed,
                    tools_enabled=True,
                    tool_choice=self._stringify_tool_choice(tool_choice),
                    note=f"provider rejected tool_choice; retrying without it ({exc})",
                )
                return self._client_create_chat_completion(
                    session_id=session_id,
                    decision=decision,
                    provider=provider,
                    model=model,
                    messages=messages,
                    tools=tools,
                    streamed=streamed,
                    structured_tool_arguments=structured_tool_arguments,
                )
            if (
                not structured_tool_arguments
                and not tools
                and self._should_retry_with_structured_tool_arguments(provider, messages, exc)
            ):
                normalized_messages = self._messages_with_structured_tool_arguments(messages)
                self._append_trace(
                    kind="fallback",
                    session_id=session_id,
                    decision=decision,
                    provider_name=provider.name,
                    streamed=streamed,
                    tools_enabled=False,
                    note="provider rejected string tool arguments; retrying with structured arguments",
                )
                return self._client_create_chat_completion(
                    session_id=session_id,
                    decision=decision,
                    provider=provider,
                    model=model,
                    messages=normalized_messages,
                    streamed=streamed,
                    structured_tool_arguments=True,
                )
            if not tools:
                raise
            if not self._should_retry_without_tools(provider, exc):
                raise
            self._tool_unsupported_providers.add(provider.name)
            self._append_trace(
                kind="fallback",
                session_id=session_id,
                decision=decision,
                provider_name=provider.name,
                streamed=streamed,
                tools_enabled=True,
                tool_choice=self._stringify_tool_choice(tool_choice),
                note=f"provider rejected tools; retrying without tools ({exc})",
            )
            return self._client_create_chat_completion(
                session_id=session_id,
                decision=decision,
                provider=provider,
                model=model,
                messages=messages,
                streamed=streamed,
                structured_tool_arguments=structured_tool_arguments,
            )

    def _tools_for_request(
        self,
        *,
        session: ConversationSession | None = None,
        provider,
        model,
        user_text: str,
        route_mode: str,
    ) -> list[dict[str, object]] | None:
        if provider.name in self._tool_unsupported_providers:
            return None
        tools = list(
            self.tool_registry.openai_tools_for_request(
                user_text,
                route_mode=route_mode,
            )
        )
        if not tools and session is not None:
            tools = self._hardware_followup_tools_for_request(session, user_text)
        profile = resolve_profile(provider, model)
        if "$web_search" in profile.builtin_tools:
            tools = [
                tool
                for tool in tools
                if not (
                    isinstance(tool, dict)
                    and isinstance(tool.get("function"), dict)
                    and tool["function"].get("name") == "web_search"
                )
            ]
            tools.append(self._kimi_web_search_tool())
        if not tools:
            return None
        return tools

    def _hardware_followup_tools_for_request(
        self,
        session: ConversationSession,
        user_text: str,
    ) -> list[dict[str, object]]:
        tool_names: list[str] = []
        if self._should_require_cup_followup_tool_choice(session, user_text):
            tool_names.append("control_cup")
        if not tool_names:
            return []
        allowed_names = set(tool_names)
        return [
            tool
            for tool in self.tool_registry.openai_tools()
            if isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and tool["function"].get("name") in allowed_names
        ]

    def _tool_choice_for_request(
        self,
        session: ConversationSession,
        user_text: str,
        tools: list[dict[str, object]] | None,
        *,
        route_mode: str,
    ) -> str | dict[str, object] | None:
        if tools is None:
            return None
        default_choice = self.tool_registry.default_tool_choice(
            user_text=user_text,
            route_mode=route_mode,
        )
        if default_choice is not None:
            return default_choice
        if self._should_require_hardware_followup_tool_choice(session, user_text):
            return "required"
        return None

    def _should_require_hardware_followup_tool_choice(
        self,
        session: ConversationSession,
        user_text: str,
    ) -> bool:
        return self._should_require_cup_followup_tool_choice(session, user_text)

    def _should_require_cup_followup_tool_choice(
        self,
        session: ConversationSession,
        user_text: str,
    ) -> bool:
        if _matches_cup_control_intent(user_text):
            return True
        if not _matches_cup_scene_followup(user_text):
            return False
        recent_messages = session.messages[-6:]
        context_keywords = ("cup", "飞机杯", "motor", "转速", "震动", "振动", "马达", "电机")
        for message in reversed(recent_messages):
            if message.role == "tool" and message.name == "control_cup":
                return True
            if message.role not in {"user", "assistant"}:
                continue
            normalized = (message.content or "").casefold()
            if not normalized:
                continue
            if _matches_cup_control_intent(message.content):
                return True
            if any(keyword in normalized for keyword in context_keywords):
                return True
        return False

    @staticmethod
    def _kimi_web_search_tool() -> dict[str, object]:
        return {
            "type": "builtin_function",
            "function": {
                "name": "$web_search",
            },
        }

    def _should_disable_thinking_for_request(
        self,
        provider,
        model,
        tools: list[dict[str, object]] | None,
    ) -> bool:
        # Kimi k2.5 with $web_search should keep thinking enabled.
        # Disabling thinking mid-conversation causes reasoning_content
        # mismatches that trigger HTTP 400 from the Kimi API.
        return False

    def _append_trace(
        self,
        *,
        kind: str,
        session_id: str,
        decision: RouteDecision,
        provider_name: str,
        streamed: bool,
        tools_enabled: bool,
        tool_choice: str | None = None,
        note: str | None = None,
        preview: str | None = None,
    ) -> None:
        if self.trace_store is None:
            return
        self.trace_store.append(
            make_trace_event(
                kind=kind,
                session_id=session_id,
                model_alias=decision.model_alias,
                provider_name=provider_name,
                route_mode=decision.mode,
                streamed=streamed,
                tools_enabled=tools_enabled,
                tool_choice=tool_choice,
                note=note,
                preview=self._short_preview(preview),
            )
        )

    def _short_preview(self, value: str | None, *, limit: int = 220) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if not normalized:
            return None
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."

    def _trace_preview_from_messages(self, messages: list[dict[str, object]]) -> str | None:
        for message in reversed(messages):
            content = message.get("content")
            role = message.get("role")
            if isinstance(content, str) and content.strip():
                prefix = str(role) if role is not None else "message"
                return f"{prefix}: {content}"
        return None

    def _tool_call_names_preview(self, tool_calls: tuple[ToolCall, ...]) -> str | None:
        if not tool_calls:
            return None
        return ", ".join(tool_call.name for tool_call in tool_calls)

    def _stringify_tool_choice(self, tool_choice: str | dict[str, object] | None) -> str | None:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            return tool_choice
        return str(tool_choice)

    def _should_continue_tool_loop(
        self,
        content: str,
        *,
        route_mode: str,
        round_index: int,
    ) -> bool:
        return self.harness.should_continue_tool_loop(
            content,
            route_mode=route_mode,
            round_index=round_index,
        )

    def _should_retry_with_structured_tool_arguments(
        self,
        provider,
        messages: list[dict[str, object]],
        exc: ProviderError,
    ) -> bool:
        if not self._has_assistant_tool_calls(messages):
            return False
        error_text = str(exc).casefold()
        return (
            provider.kind == "ollama_native"
            or "ollama" in provider.name.casefold()
            or "can't find closing '}' symbol" in error_text
            or "value looks like object" in error_text
        )

    def _has_assistant_tool_calls(self, messages: list[dict[str, object]]) -> bool:
        return self.tool_protocol_adapter.has_assistant_tool_calls(messages)

    def _messages_with_structured_tool_arguments(
        self,
        messages: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        return self.tool_protocol_adapter.messages_with_structured_tool_arguments(messages)

    def _should_retry_without_tools(self, provider, exc: ProviderError) -> bool:
        error_text = str(exc).casefold()
        if any(
            marker in error_text
            for marker in (
                "unreachable",
                "ssl",
                "eof occurred in violation of protocol",
                "closed the connection unexpectedly",
                "timed out",
                "timeout",
                "connection reset",
                "remote end closed connection",
            )
        ):
            return False
        if "tool" in error_text or "function" in error_text:
            return True
        provider_name = str(getattr(provider, "name", "")).casefold()
        if "ollama" in provider_name and "400" in error_text:
            return True
        return False

    def _tool_message_reasoning_content(
        self,
        decision: RouteDecision,
        completion: CompletionResult,
    ) -> str | None:
        if completion.reasoning_content is not None:
            return completion.reasoning_content
        model_config = self.config.get_model(decision.model_alias)
        provider_config = self.config.get_provider(model_config.provider)
        profile = resolve_profile(provider_config, model_config)
        if not self._requires_reasoning_content_replay(profile, model_config):
            return None
        return profile.reasoning_content_empty_placeholder

    def _requires_reasoning_content_replay(
        self,
        profile: ProviderProfile,
        model,
    ) -> bool:
        if not profile.reasoning_content_required_when_thinking:
            return False
        if model.think is False:
            return False
        return True

    def _stream_final_answer(
        self,
        session: ConversationSession,
        decision: RouteDecision,
        *,
        provider,
        model,
        messages: list[dict[str, object]],
        on_chunk: Callable[[str], None] | None,
        target_profile: ProviderProfile | None = None,
    ) -> ChatTurnResult:
        messages = self.tool_protocol_adapter.messages_with_normalized_tool_call_ids(messages)
        content_parts: list[str] = []
        self._append_trace(
            kind="stream_request",
            session_id=session.session_id,
            decision=decision,
            provider_name=provider.name,
            streamed=True,
            tools_enabled=False,
            preview=self._trace_preview_from_messages(messages),
        )
        try:
            for chunk in self.client.create_chat_completion_stream(
                provider=provider,
                model=model,
                messages=messages,
            ):
                content_parts.append(chunk)
                if on_chunk is not None:
                    on_chunk(chunk)
        except KeyboardInterrupt as exc:
            partial_result = None
            partial_content = "".join(content_parts).strip()
            if partial_content:
                partial_result = self._append_assistant_message(
                    session,
                    decision,
                    partial_content,
                    target_profile=target_profile,
                )
            raise GenerationInterrupted(partial_result=partial_result) from exc
        except ProviderError:
            if content_parts:
                raise
            completion = self._client_create_chat_completion(
                session_id=session.session_id,
                decision=decision,
                provider=provider,
                model=model,
                messages=messages,
                streamed=True,
            )
            content_parts = [completion.content]
            if on_chunk is not None:
                on_chunk(completion.content)
        final_content = "".join(content_parts).strip()
        self._append_trace(
            kind="stream_response",
            session_id=session.session_id,
            decision=decision,
            provider_name=provider.name,
            streamed=True,
            tools_enabled=False,
            preview=final_content,
        )
        return self._append_assistant_message(
            session,
            decision,
            final_content,
            target_profile=target_profile,
        )

    def _append_assistant_message(
        self,
        session: ConversationSession,
        decision: RouteDecision,
        content: str,
        *,
        ask_user: AskUserAction | None = None,
        target_profile: ProviderProfile | None = None,
    ) -> ChatTurnResult:
        # Sanitise any model-specific tool-call text artifacts that may have
        # leaked through when text-based extraction failed.  This prevents
        # stored content from confusing a different model on subsequent turns.
        content = sanitize_tool_call_artifacts(content)
        source_profile_id = target_profile.profile_id if target_profile is not None else None
        assistant_message = ChatMessage(
            role="assistant",
            content=content,
            created_at=utc_now_iso(),
            model_alias=decision.model_alias,
            route_reason=decision.reason,
            source_profile=source_profile_id,
        )
        session.append(assistant_message)
        session.append_transcript(
            ChatMessage(
                role="assistant",
                content=content,
                created_at=assistant_message.created_at,
                model_alias=decision.model_alias,
                route_reason=decision.reason,
                source_profile=source_profile_id,
            )
        )
        session.pending_ask_user = ask_user

        return ChatTurnResult(
            decision=decision,
            assistant_message=assistant_message,
            ask_user=ask_user,
        )
