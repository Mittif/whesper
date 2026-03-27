from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from whesper.agent_harness import AgentHarness, AGENTIC_PLANNING_PROMPT, StepCallback
from whesper.client import CompletionResult, OpenAICompatibleClient, ProviderError, ToolCall
from whesper.config import AppConfig
from whesper.live_data import LiveContextService
from whesper.message_builder import SessionMessageBuilder
from whesper.memory import MemoryService
from whesper.router import RouteDecision, select_model
from whesper.session import ChatMessage, ConversationSession, utc_now_iso
from whesper.tool_executor import ToolExecutor
from whesper.tool_protocol import DefaultToolProtocolAdapter, ToolMessageFormat
from whesper.trace import TraceStore, make_trace_event
from whesper.tools import ToolRegistry


@dataclass(slots=True)
class ChatTurnResult:
    decision: RouteDecision
    assistant_message: ChatMessage


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
            tool_protocol_adapter=self.tool_protocol_adapter,
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
        self._append_user_message(session, user_text)
        if self.memory_service is not None:
            self.memory_service.capture_user_message(session.session_id, user_text)
        return self._complete_turn(
            session,
            user_text,
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
        self._append_user_message(session, user_text)
        if self.memory_service is not None:
            self.memory_service.capture_user_message(session.session_id, user_text)
        return self._complete_turn_stream(
            session,
            user_text,
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
        tool_message_format = self._tool_message_format(model_config, provider_config)

        messages = self._build_messages(
            session,
            model_config.system_prompt,
            tool_message_format=tool_message_format,
            user_text=user_text,
            route_mode=decision.mode,
        )
        tools = self._tools_for_request(provider_name=provider_config.name, user_text=user_text, route_mode=decision.mode)
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
                tool_message_format=tool_message_format,
                on_step=on_step,
            )
            completion = run_result.completion
        else:
            completion = self._client_create_chat_completion(
                session_id=session.session_id,
                decision=decision,
                provider=provider_config,
                model=model_config,
                messages=messages,
                streamed=False,
            )
        return self._append_assistant_message(
            session,
            decision,
            completion.content,
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
        tool_message_format = self._tool_message_format(model_config, provider_config)
        messages = self._build_messages(
            session,
            model_config.system_prompt,
            tool_message_format=tool_message_format,
            user_text=user_text,
            route_mode=decision.mode,
        )
        tools = self._tools_for_request(
            provider_name=provider_config.name,
            user_text=user_text,
            route_mode=decision.mode,
        )
        if tools is None:
            return self._stream_final_answer(
                session,
                decision,
                provider=provider_config,
                model=model_config,
                messages=messages,
                on_chunk=on_chunk,
            )
        run_result = self.harness.run_until_final(
            session,
            decision,
            provider=provider_config,
            model=model_config,
            model_system_prompt=model_config.system_prompt,
            user_text=user_text,
            route_mode=decision.mode,
            tools=tools,
            tool_message_format=tool_message_format,
            on_step=on_step,
        )
        if run_result.final_messages is not None:
            return self._stream_final_answer(
                session,
                decision,
                provider=provider_config,
                model=model_config,
                messages=run_result.final_messages,
                on_chunk=on_chunk,
            )
        completion = run_result.completion
        if on_chunk is not None and completion.content:
            on_chunk(completion.content)
        return self._append_assistant_message(session, decision, completion.content)

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
        tool_message_format: ToolMessageFormat,
        user_text: str,
        route_mode: str,
        include_planning_prompt: bool = True,
    ) -> list[dict[str, object]]:
        planning_prompt = AGENTIC_PLANNING_PROMPT if include_planning_prompt else None
        return self.message_builder.build_messages(
            session,
            model_system_prompt,
            tool_message_format=tool_message_format,
            user_text=user_text,
            route_mode=route_mode,
            planning_prompt=planning_prompt,
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
        provider_name: str,
        user_text: str,
        route_mode: str,
    ) -> list[dict[str, object]] | None:
        if provider_name in self._tool_unsupported_providers:
            return None
        tools = self.tool_registry.openai_tools()
        if not tools:
            return None
        # Always offer tools so the model can do multi-step agentic
        # decomposition for any query. The model decides whether to
        # actually call them based on the system prompt and user intent.
        return tools

    def _tool_choice_for_request(
        self,
        tools: list[dict[str, object]] | None,
        *,
        route_mode: str,
    ) -> str | dict[str, object] | None:
        if tools is None:
            return None
        return self.tool_registry.default_tool_choice(route_mode=route_mode)

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

    def _tool_message_format(self, model, provider) -> ToolMessageFormat:
        return self.tool_protocol_adapter.tool_message_format(model, provider)

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
        if not self._requires_reasoning_content_replay(provider_config, model_config):
            return None
        return ""

    def _requires_reasoning_content_replay(self, provider, model) -> bool:
        think = getattr(model, "think", None)
        if think in (None, False):
            return False
        provider_name = str(getattr(provider, "name", "")).casefold()
        model_name = str(getattr(model, "model", "")).casefold()
        return "kimi" in provider_name or "moonshot" in provider_name or "kimi" in model_name

    def _stream_final_answer(
        self,
        session: ConversationSession,
        decision: RouteDecision,
        *,
        provider,
        model,
        messages: list[dict[str, object]],
        on_chunk: Callable[[str], None] | None,
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
        )

    def _append_assistant_message(
        self,
        session: ConversationSession,
        decision: RouteDecision,
        content: str,
    ) -> ChatTurnResult:
        assistant_message = ChatMessage(
            role="assistant",
            content=content,
            created_at=utc_now_iso(),
            model_alias=decision.model_alias,
            route_reason=decision.reason,
        )
        session.append(assistant_message)
        session.append_transcript(
            ChatMessage(
                role="assistant",
                content=content,
                created_at=assistant_message.created_at,
                model_alias=decision.model_alias,
                route_reason=decision.reason,
            )
        )

        return ChatTurnResult(
            decision=decision,
            assistant_message=assistant_message,
        )
