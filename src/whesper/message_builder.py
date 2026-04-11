from __future__ import annotations

from whesper.config import AppConfig
from whesper.live_data import LiveContextService
from whesper.memory import MemoryService
from whesper.session import ChatMessage, ConversationSession
from whesper.tool_protocol import DefaultToolProtocolAdapter, ToolMessageFormat
from whesper.tools import ToolRegistry


class SessionMessageBuilder:
    def __init__(
        self,
        config: AppConfig,
        *,
        tool_protocol_adapter: DefaultToolProtocolAdapter,
        memory_service: MemoryService | None = None,
        live_context_service: LiveContextService | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.config = config
        self.tool_protocol_adapter = tool_protocol_adapter
        self.memory_service = memory_service
        self.live_context_service = live_context_service
        self.tool_registry = tool_registry

    def build_messages(
        self,
        session: ConversationSession,
        model_system_prompt: str | None,
        *,
        tool_message_format: ToolMessageFormat,
        user_text: str,
        route_mode: str,
        planning_prompt: str | None = None,
        include_live_context: bool = True,
    ) -> list[dict[str, object]]:
        prompts = [self.config.persona.system_prompt.strip()]
        if model_system_prompt:
            prompts.append(model_system_prompt.strip())
        if self.tool_registry is not None:
            tool_prompt = self.tool_registry.tool_prompt()
            if planning_prompt:
                tool_prompt += planning_prompt
            prompts.append(tool_prompt)
        if self.memory_service is not None:
            memory_prompt = self.memory_service.build_prompt_context(user_text)
            if memory_prompt:
                prompts.append(memory_prompt)
        if include_live_context and self.live_context_service is not None:
            live_context_prompt = self.live_context_service.build_prompt_context(
                user_text,
                route_mode=route_mode,
            )
            if live_context_prompt:
                prompts.append(live_context_prompt)
        system_message = "\n\n".join(part for part in prompts if part)

        messages: list[dict[str, object]] = [{"role": "system", "content": system_message}]
        for message in self.model_history_messages(session):
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
            if message.reasoning_content is not None:
                payload["reasoning_content"] = message.reasoning_content
            if (
                message.name is not None
                and (message.role != "tool" or tool_message_format.include_tool_name)
            ):
                payload["name"] = message.name
            if message.tool_call_id is not None:
                payload["tool_call_id"] = message.tool_call_id
            if message.tool_calls is not None:
                payload["tool_calls"] = self.tool_protocol_adapter.serialize_tool_calls(
                    message.tool_calls,
                    tool_message_format=tool_message_format,
                )
            messages.append(payload)
        return messages

    def model_history_messages(self, session: ConversationSession) -> list[ChatMessage]:
        current_turn_start = self._last_user_index(session.messages)
        if current_turn_start is None:
            return session.transcript_messages[-self.config.app.history_limit :]

        current_turn_messages = session.messages[current_turn_start:]
        current_user = session.messages[current_turn_start]
        transcript_cutoff = self.last_matching_transcript_index(session, current_user)
        if transcript_cutoff is None:
            historical_transcript = session.transcript_messages
        else:
            historical_transcript = session.transcript_messages[:transcript_cutoff]
        return [
            *historical_transcript[-self.config.app.history_limit :],
            *current_turn_messages,
        ]

    def last_matching_transcript_index(
        self,
        session: ConversationSession,
        message: ChatMessage,
    ) -> int | None:
        for index in range(len(session.transcript_messages) - 1, -1, -1):
            transcript_message = session.transcript_messages[index]
            if (
                transcript_message.role == message.role
                and transcript_message.created_at == message.created_at
                and transcript_message.content == message.content
            ):
                return index
        return None

    @staticmethod
    def _last_user_index(messages: list[ChatMessage]) -> int | None:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == "user":
                return index
        return None
