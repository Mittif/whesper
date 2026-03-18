from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from whesper.client import OpenAICompatibleClient, ProviderError
from whesper.config import AppConfig
from whesper.router import RouteDecision, select_model
from whesper.session import ChatMessage, ConversationSession, utc_now_iso


@dataclass(slots=True)
class ChatTurnResult:
    decision: RouteDecision
    assistant_message: ChatMessage


class GenerationInterrupted(RuntimeError):
    def __init__(self, partial_result: ChatTurnResult | None = None) -> None:
        super().__init__("Generation interrupted.")
        self.partial_result = partial_result


class ChatService:
    def __init__(self, config: AppConfig, client: OpenAICompatibleClient | None = None) -> None:
        self.config = config
        self.client = client or OpenAICompatibleClient()

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

        messages = self._build_messages(session, model_config.system_prompt)
        completion = self.client.create_chat_completion(
            provider=provider_config,
            model=model_config,
            messages=messages,
        )

        assistant_message = ChatMessage(
            role="assistant",
            content=completion.content,
            created_at=utc_now_iso(),
            model_alias=decision.model_alias,
            route_reason=decision.reason,
        )
        session.append(assistant_message)

        return ChatTurnResult(
            decision=decision,
            assistant_message=assistant_message,
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
        messages = self._build_messages(session, model_config.system_prompt)

        content_parts: list[str] = []
        try:
            for chunk in self.client.create_chat_completion_stream(
                provider=provider_config,
                model=model_config,
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
            completion = self.client.create_chat_completion(
                provider=provider_config,
                model=model_config,
                messages=messages,
            )
            content_parts = [completion.content]
            if on_chunk is not None:
                on_chunk(completion.content)

        return self._append_assistant_message(
            session,
            decision,
            "".join(content_parts).strip(),
        )

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
    ) -> list[dict[str, str]]:
        prompts = [self.config.persona.system_prompt.strip()]
        if model_system_prompt:
            prompts.append(model_system_prompt.strip())
        system_message = "\n\n".join(part for part in prompts if part)

        result: list[dict[str, str]] = [{"role": "system", "content": system_message}]
        recent_messages = session.messages[-self.config.app.history_limit :]
        for message in recent_messages:
            result.append({"role": message.role, "content": message.content})
        return result

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
