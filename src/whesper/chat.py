from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable

from whesper.client import CompletionResult, OpenAICompatibleClient, ProviderError, ToolCall
from whesper.config import AppConfig
from whesper.live_data import LiveContextService
from whesper.memory import MemoryService
from whesper.router import RouteDecision, select_model
from whesper.session import ChatMessage, ConversationSession, utc_now_iso
from whesper.trace import TraceStore, make_trace_event
from whesper.tools import ToolExecutionError, ToolRegistry


@dataclass(slots=True)
class ChatTurnResult:
    decision: RouteDecision
    assistant_message: ChatMessage


@dataclass(slots=True, frozen=True)
class ToolMessageFormat:
    assistant_tool_content_null: bool = True
    tool_arguments_mode: str = "string"
    include_tool_name: bool = False


class GenerationInterrupted(RuntimeError):
    def __init__(self, partial_result: ChatTurnResult | None = None) -> None:
        super().__init__("Generation interrupted.")
        self.partial_result = partial_result


class ChatService:
    MAX_TOOL_ROUNDS = 3
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
    DEFAULT_TOOL_MESSAGE_FORMAT = ToolMessageFormat()
    KIMI_TOOL_MESSAGE_FORMAT = ToolMessageFormat(
        assistant_tool_content_null=True,
        tool_arguments_mode="string",
        include_tool_name=False,
    )
    QWEN_TOOL_MESSAGE_FORMAT = ToolMessageFormat(
        assistant_tool_content_null=True,
        tool_arguments_mode="object",
        include_tool_name=False,
    )

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
        self.tool_registry = tool_registry or ToolRegistry.default()
        self.trace_store = trace_store
        self._tool_unsupported_providers: set[str] = set()

    def send(
        self,
        session: ConversationSession,
        user_text: str,
        *,
        mode_override: str = "auto",
    ) -> ChatTurnResult:
        user_message = ChatMessage(
            role="user",
            content=user_text,
            created_at=utc_now_iso(),
        )
        session.append(user_message)
        if self.memory_service is not None:
            self.memory_service.capture_user_message(session.session_id, user_text)
        return self._complete_turn(
            session,
            user_text,
            mode_override=mode_override,
        )

    def send_stream(
        self,
        session: ConversationSession,
        user_text: str,
        *,
        mode_override: str = "auto",
        on_chunk: Callable[[str], None] | None = None,
    ) -> ChatTurnResult:
        user_message = ChatMessage(
            role="user",
            content=user_text,
            created_at=utc_now_iso(),
        )
        session.append(user_message)
        if self.memory_service is not None:
            self.memory_service.capture_user_message(session.session_id, user_text)
        return self._complete_turn_stream(
            session,
            user_text,
            mode_override=mode_override,
            on_chunk=on_chunk,
        )

    def retry_stream(
        self,
        session: ConversationSession,
        *,
        mode_override: str = "auto",
        on_chunk: Callable[[str], None] | None = None,
    ) -> ChatTurnResult:
        last_user_index = self._last_user_index(session)
        if last_user_index is None:
            raise ValueError("No user message is available to retry yet.")

        last_user_message = session.messages[last_user_index]
        del session.messages[last_user_index + 1 :]
        self._sync_session_updated_at(session)
        return self._complete_turn_stream(
            session,
            last_user_message.content,
            mode_override=mode_override,
            on_chunk=on_chunk,
        )

    def _complete_turn(
        self,
        session: ConversationSession,
        user_text: str,
        *,
        mode_override: str = "auto",
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
            completion = self._resolve_pre_tool_completion(
                session,
                decision,
                provider=provider_config,
                model=model_config,
                model_system_prompt=model_config.system_prompt,
                user_text=user_text,
                route_mode=decision.mode,
                tools=tools,
                tool_message_format=tool_message_format,
            )
        else:
            completion = self._client_create_chat_completion(
                session_id=session.session_id,
                decision=decision,
                provider=provider_config,
                model=model_config,
                messages=messages,
                streamed=False,
            )
        if completion.tool_calls:
            self._append_tool_interaction_messages(session, decision, completion)
            self._execute_tool_calls(session, decision, completion.tool_calls)
            messages = self._build_messages(
                session,
                model_config.system_prompt,
                tool_message_format=tool_message_format,
                user_text=user_text,
                route_mode=decision.mode,
            )
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
        completion = self._resolve_pre_tool_completion(
            session,
            decision,
            provider=provider_config,
            model=model_config,
            model_system_prompt=model_config.system_prompt,
            user_text=user_text,
            route_mode=decision.mode,
            tools=tools,
            tool_message_format=tool_message_format,
        )
        if completion.tool_calls:
            self._append_tool_interaction_messages(session, decision, completion)
            self._execute_tool_calls(session, decision, completion.tool_calls)
            messages = self._build_messages(
                session,
                model_config.system_prompt,
                tool_message_format=tool_message_format,
                user_text=user_text,
                route_mode=decision.mode,
            )
            return self._stream_final_answer(
                session,
                decision,
                provider=provider_config,
                model=model_config,
                messages=messages,
                on_chunk=on_chunk,
            )
        if on_chunk is not None and completion.content:
            on_chunk(completion.content)
        return self._append_assistant_message(session, decision, completion.content)

    def _last_user_index(self, session: ConversationSession) -> int | None:
        for index in range(len(session.messages) - 1, -1, -1):
            if session.messages[index].role == "user":
                return index
        return None

    def _sync_session_updated_at(self, session: ConversationSession) -> None:
        if session.messages:
            session.updated_at = session.messages[-1].created_at
            return
        session.updated_at = session.created_at

    def _build_messages(
        self,
        session: ConversationSession,
        model_system_prompt: str | None,
        *,
        tool_message_format: ToolMessageFormat,
        user_text: str,
        route_mode: str,
    ) -> list[dict[str, object]]:
        prompts = [self.config.persona.system_prompt.strip()]
        if model_system_prompt:
            prompts.append(model_system_prompt.strip())
        if self.tool_registry is not None:
            prompts.append(self.tool_registry.tool_prompt())
        if self.memory_service is not None:
            memory_prompt = self.memory_service.build_prompt_context(user_text)
            if memory_prompt:
                prompts.append(memory_prompt)
        if self.live_context_service is not None:
            live_context_prompt = self.live_context_service.build_prompt_context(
                user_text,
                route_mode=route_mode,
            )
            if live_context_prompt:
                prompts.append(live_context_prompt)
        system_message = "\n\n".join(part for part in prompts if part)

        result: list[dict[str, object]] = [{"role": "system", "content": system_message}]
        recent_messages = session.messages[-self.config.app.history_limit :]
        for message in recent_messages:
            payload: dict[str, object] = {"role": message.role}
            if (
                message.role == "assistant"
                and message.tool_calls is not None
                and not message.content.strip()
                and tool_message_format.assistant_tool_content_null
            ):
                payload["content"] = None
            else:
                payload["content"] = message.content
            if (
                message.name is not None
                and (message.role != "tool" or tool_message_format.include_tool_name)
            ):
                payload["name"] = message.name
            if message.tool_call_id is not None:
                payload["tool_call_id"] = message.tool_call_id
            if message.tool_calls is not None:
                payload["tool_calls"] = self._serialize_tool_calls(
                    message.tool_calls,
                    tool_message_format=tool_message_format,
                )
            result.append(payload)
        return result

    def _append_tool_interaction_messages(
        self,
        session: ConversationSession,
        decision: RouteDecision,
        completion: CompletionResult,
    ) -> None:
        assistant_message = ChatMessage(
            role="assistant",
            content="",
            created_at=utc_now_iso(),
            model_alias=decision.model_alias,
            route_reason=decision.reason,
            tool_calls=[self._tool_call_payload(item) for item in completion.tool_calls],
        )
        session.append(assistant_message)
        self._append_trace(
            kind="tool_calls",
            session_id=session.session_id,
            decision=decision,
            provider_name=self.config.get_provider(
                self.config.get_model(decision.model_alias).provider
            ).name,
            streamed=False,
            tools_enabled=True,
            tool_choice=self._tool_choice_for_request(
                self.tool_registry.openai_tools(),
                route_mode=decision.mode,
            ),
            note=", ".join(item.name for item in completion.tool_calls),
            preview=completion.content or None,
        )

    def _execute_tool_calls(
        self,
        session: ConversationSession,
        decision: RouteDecision,
        tool_calls: tuple[ToolCall, ...],
    ) -> list[ChatMessage]:
        tool_messages: list[ChatMessage] = []
        for tool_call in tool_calls:
            try:
                result = self.tool_registry.execute(tool_call)
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
            self._append_trace(
                kind="tool_result",
                session_id=session.session_id,
                decision=decision,
                provider_name=self.config.get_provider(
                    self.config.get_model(decision.model_alias).provider
                ).name,
                streamed=False,
                tools_enabled=True,
                tool_choice=self._tool_choice_for_request(
                    self.tool_registry.openai_tools(),
                    route_mode=decision.mode,
                ),
                note=name,
                preview=content,
            )
        return tool_messages

    def _tool_call_payload(self, tool_call: ToolCall) -> dict[str, object]:
        return {
            "id": tool_call.tool_call_id,
            "type": "function",
            "function": {
                "name": tool_call.name,
                "arguments": tool_call.arguments_json,
            },
        }

    def _serialize_tool_calls(
        self,
        tool_calls: list[dict[str, object]],
        *,
        tool_message_format: ToolMessageFormat,
    ) -> list[dict[str, object]]:
        serialized_calls: list[dict[str, object]] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                serialized_calls.append(tool_call)
                continue
            serialized_call = dict(tool_call)
            function = serialized_call.get("function")
            if isinstance(function, dict):
                serialized_function = dict(function)
                arguments = serialized_function.get("arguments")
                if (
                    tool_message_format.tool_arguments_mode == "object"
                    and isinstance(arguments, str)
                ):
                    try:
                        serialized_function["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError:
                        serialized_function["arguments"] = arguments
                elif (
                    tool_message_format.tool_arguments_mode == "string"
                    and isinstance(arguments, dict)
                ):
                    serialized_function["arguments"] = json.dumps(
                        arguments,
                        ensure_ascii=False,
                    )
                serialized_call["function"] = serialized_function
            serialized_calls.append(serialized_call)
        return serialized_calls

    def _resolve_pre_tool_completion(
        self,
        session: ConversationSession,
        decision: RouteDecision,
        *,
        provider,
        model,
        model_system_prompt: str | None,
        user_text: str,
        route_mode: str,
        tools: list[dict[str, object]],
        tool_message_format: ToolMessageFormat,
    ) -> CompletionResult:
        working_messages = self._build_messages(
            session,
            model_system_prompt,
            tool_message_format=tool_message_format,
            user_text=user_text,
            route_mode=route_mode,
        )
        tool_choice = self._tool_choice_for_request(tools, route_mode=route_mode)

        for round_index in range(self.MAX_TOOL_ROUNDS):
            completion = self._client_create_chat_completion(
                session_id=session.session_id,
                decision=decision,
                provider=provider,
                model=model,
                messages=working_messages,
                tools=tools,
                tool_choice=tool_choice,
                streamed=False,
            )
            if completion.tool_calls:
                return completion
            if self._should_continue_tool_loop(
                completion.content,
                route_mode=route_mode,
                round_index=round_index,
            ):
                self._append_trace(
                    kind="planning_retry",
                    session_id=session.session_id,
                    decision=decision,
                    provider_name=provider.name,
                    streamed=False,
                    tools_enabled=True,
                    tool_choice=self._stringify_tool_choice(tool_choice),
                    preview=completion.content,
                )
                working_messages = [
                    *working_messages,
                    {"role": "assistant", "content": completion.content},
                    {"role": "user", "content": self.INTERNAL_CONTINUE_PROMPT},
                ]
                continue
            return completion

        self._append_trace(
            kind="tool_loop_limit",
            session_id=session.session_id,
            decision=decision,
            provider_name=provider.name,
            streamed=False,
            tools_enabled=True,
            tool_choice=self._stringify_tool_choice(tool_choice),
            note=f"max_rounds={self.MAX_TOOL_ROUNDS}",
        )
        return CompletionResult(
            content="我刚才没有顺利拿到完整结果。要不要我换个方式继续帮你查？",
            raw_response={},
        )

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
        if not self.tool_registry.should_offer_tools(user_text, route_mode=route_mode):
            return None
        return self.tool_registry.openai_tools()

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
        candidates = (
            getattr(model, "name", ""),
            getattr(model, "model", ""),
            getattr(provider, "name", ""),
        )
        lowered = " ".join(str(item).casefold() for item in candidates if item)
        if "kimi" in lowered or "moonshot" in lowered:
            return self.KIMI_TOOL_MESSAGE_FORMAT
        if "qwen" in lowered:
            return self.QWEN_TOOL_MESSAGE_FORMAT
        if getattr(provider, "kind", "") == "ollama_native":
            return self.QWEN_TOOL_MESSAGE_FORMAT
        return self.DEFAULT_TOOL_MESSAGE_FORMAT

    def _should_continue_tool_loop(
        self,
        content: str,
        *,
        route_mode: str,
        round_index: int,
    ) -> bool:
        if round_index >= self.MAX_TOOL_ROUNDS - 1:
            return False
        if route_mode != "search":
            return False
        normalized = " ".join(content.split()).casefold()
        if not normalized:
            return False
        for marker in self.INTERIM_TOOL_RESPONSE_MARKERS:
            if marker.casefold() in normalized:
                return True
        return False

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
        for message in messages:
            if message.get("role") != "assistant":
                continue
            if isinstance(message.get("tool_calls"), list):
                return True
        return False

    def _messages_with_structured_tool_arguments(
        self,
        messages: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        normalized_messages: list[dict[str, object]] = []
        for message in messages:
            normalized_message = dict(message)
            raw_tool_calls = normalized_message.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                normalized_tool_calls: list[dict[str, object]] = []
                for tool_call in raw_tool_calls:
                    if not isinstance(tool_call, dict):
                        normalized_tool_calls.append(tool_call)
                        continue
                    normalized_call = dict(tool_call)
                    function = normalized_call.get("function")
                    if isinstance(function, dict):
                        normalized_function = dict(function)
                        arguments = normalized_function.get("arguments")
                        if isinstance(arguments, str):
                            try:
                                parsed_arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                parsed_arguments = arguments
                            normalized_function["arguments"] = parsed_arguments
                        normalized_call["function"] = normalized_function
                    normalized_tool_calls.append(normalized_call)
                normalized_message["tool_calls"] = normalized_tool_calls
            normalized_messages.append(normalized_message)
        return normalized_messages

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

        return ChatTurnResult(
            decision=decision,
            assistant_message=assistant_message,
        )
