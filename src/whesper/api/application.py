from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from whesper import __version__
from whesper.agent_types import AgentStep
from whesper.api.events import (
    EVENT_ASK_USER,
    EVENT_CHUNK,
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_FINAL,
    EVENT_ROUTE,
    EVENT_STEP,
    ask_user_to_dict,
    chat_message_to_dict,
    decision_to_dict,
    step_to_dict,
)
from whesper.chat import ChatService, ChatTurnResult, GenerationInterrupted
from whesper.client import ProviderError
from whesper.commands import VALID_MODES
from whesper.config import AppConfig, ConfigError, VALID_SEARCH_PROVIDERS
from whesper.memory import MemoryService, MemoryStore
from whesper.router import select_model
from whesper.session import ConversationSession, SessionStore
from whesper.tools import ToolExecutionError
from whesper.trace import TraceStore


class SessionNotFound(KeyError):
    pass


class SessionAlreadyExists(ValueError):
    pass


class SessionBusy(RuntimeError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session is currently generating: {session_id}")
        self.session_id = session_id


_ERROR_CODE_BY_EXCEPTION: tuple[tuple[type[BaseException], str], ...] = (
    (GenerationInterrupted, "GENERATION_INTERRUPTED"),
    (ToolExecutionError, "TOOL_EXECUTION_ERROR"),
    (ProviderError, "PROVIDER_ERROR"),
    (ConfigError, "CONFIG_ERROR"),
    (ValueError, "VALIDATION_ERROR"),
)


def _classify_error(exc: BaseException) -> tuple[str, str]:
    for exc_type, code in _ERROR_CODE_BY_EXCEPTION:
        if isinstance(exc, exc_type):
            return code, str(exc) or exc.__class__.__name__
    return "INTERNAL_ERROR", str(exc) or exc.__class__.__name__


@dataclass(slots=True)
class SessionSummary:
    session_id: str
    created_at: str
    updated_at: str
    pinned_model: str
    message_count: int
    has_pending_ask_user: bool


class AgentApplication:
    """Pure-Python facade that every transport (CLI, HTTP, future clients) talks to.

    Owns the long-lived process state: configuration, storage, and (lazily) the
    ChatService. Methods on this class are the single source of truth for what
    the Whesper backend can do; transport layers only translate to/from this
    surface.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        storage_dir = Path(config.app.storage_dir)
        self.session_store = SessionStore(storage_dir)
        self.memory_store = MemoryStore(storage_dir)
        self.trace_store = TraceStore(storage_dir)
        self._memory_service = MemoryService(self.memory_store)
        self._chat_service: ChatService | None = None
        self._busy_sessions: set[str] = set()

    @property
    def version(self) -> str:
        return __version__

    @property
    def chat_service(self) -> ChatService:
        if self._chat_service is None:
            self._chat_service = ChatService(
                self.config,
                memory_service=self._memory_service,
                trace_store=self.trace_store,
            )
        return self._chat_service

    def capabilities(self) -> dict[str, object]:
        return {
            "version": self.version,
            "features": ["tools", "memory", "cup", "trace", "search"],
            "route_modes": list(VALID_MODES),
            "websearch_providers": list(VALID_SEARCH_PROVIDERS),
        }

    def list_sessions(self) -> list[SessionSummary]:
        summaries: list[SessionSummary] = []
        for session_id in self.session_store.list_sessions():
            session = self.session_store.load(session_id)
            summaries.append(self._summarize(session))
        return summaries

    def create_session(self, session_id: str) -> ConversationSession:
        normalized = self._normalize_session_id(session_id)
        if self.session_store.path_for(normalized).exists():
            raise SessionAlreadyExists(normalized)
        return self.session_store.load(normalized)

    def get_session(self, session_id: str) -> ConversationSession:
        normalized = self._normalize_session_id(session_id)
        if not self.session_store.path_for(normalized).exists():
            raise SessionNotFound(normalized)
        return self.session_store.load(normalized)

    def delete_session(self, session_id: str) -> bool:
        normalized = self._normalize_session_id(session_id)
        return self.session_store.delete_session(normalized)

    def summarize(self, session: ConversationSession) -> SessionSummary:
        return self._summarize(session)

    def list_messages(
        self,
        session_id: str,
        *,
        limit: int = 50,
        before: str | None = None,
        transcript: bool = False,
    ) -> tuple[list, str | None]:
        """Return messages newest-first, plus a cursor for the next page.

        `before` filters to messages strictly older than that ISO timestamp.
        `next_before` is the oldest returned message's created_at when the page
        filled exactly, else None.
        """

        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        session = self.get_session(session_id)
        source = session.transcript_messages if transcript else session.messages
        filtered = (
            [m for m in source if m.created_at < before]
            if before is not None
            else list(source)
        )
        filtered.sort(key=lambda m: m.created_at, reverse=True)
        page = filtered[:limit]
        next_before = page[-1].created_at if len(page) == limit else None
        return page, next_before

    def _normalize_session_id(self, session_id: str) -> str:
        value = (session_id or "").strip()
        if not value:
            raise ValueError("session_id must be a non-empty string")
        return value

    def _summarize(self, session: ConversationSession) -> SessionSummary:
        return SessionSummary(
            session_id=session.session_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            pinned_model=session.pinned_model,
            message_count=len(session.messages),
            has_pending_ask_user=session.pending_ask_user is not None,
        )

    async def send_turn(
        self,
        session_id: str,
        *,
        text: str,
        mode: str = "auto",
    ) -> ChatTurnResult:
        """Non-streaming turn. Serializes per-session; raises SessionBusy if a
        turn is already in flight for this session."""

        session = self.get_session(session_id)
        self._mark_busy(session.session_id)
        try:
            result = await asyncio.to_thread(
                self.chat_service.send,
                session,
                text,
                mode_override=mode,
            )
            self.session_store.save(session)
            return result
        finally:
            self._mark_free(session.session_id)

    async def send_turn_events(
        self,
        session_id: str,
        *,
        text: str,
        mode: str = "auto",
    ) -> AsyncIterator[tuple[str, dict]]:
        """Async generator yielding (event_name, data) pairs that map 1:1 to
        the SSE wire contract in docs/API.md. Always terminates with
        (EVENT_DONE, {})."""

        session = self.get_session(session_id)
        self._mark_busy(session.session_id)
        try:
            # Emit route first so the client knows the chosen model even if
            # the provider never produces tokens. ChatService will recompute
            # the decision internally; that's harmless and cheap.
            decision = select_model(
                self.config,
                text,
                pinned_model=session.pinned_model,
                mode_override=mode,
            )
            yield EVENT_ROUTE, {"decision": decision_to_dict(decision)}

            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()

            def on_chunk(chunk: str) -> None:
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("chunk", chunk),
                )

            def on_step(step: AgentStep) -> None:
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("step", step),
                )

            async def run_turn() -> None:
                try:
                    result = await asyncio.to_thread(
                        self.chat_service.send_stream,
                        session,
                        text,
                        mode_override=mode,
                        on_chunk=on_chunk,
                        on_step=on_step,
                    )
                    await queue.put(("__result__", result))
                except BaseException as exc:  # noqa: BLE001
                    await queue.put(("__error__", exc))

            task = asyncio.create_task(run_turn())

            try:
                while True:
                    kind, payload = await queue.get()
                    if kind == "chunk":
                        yield EVENT_CHUNK, {"text": payload}
                    elif kind == "step":
                        yield EVENT_STEP, {"step": step_to_dict(payload)}  # type: ignore[arg-type]
                    elif kind == "__result__":
                        result: ChatTurnResult = payload  # type: ignore[assignment]
                        if result.ask_user is not None:
                            yield EVENT_ASK_USER, {
                                "ask_user": ask_user_to_dict(result.ask_user),
                            }
                        else:
                            yield EVENT_FINAL, {
                                "assistant_message": chat_message_to_dict(
                                    result.assistant_message,
                                ),
                            }
                        break
                    elif kind == "__error__":
                        code, message = _classify_error(payload)  # type: ignore[arg-type]
                        yield EVENT_ERROR, {"code": code, "message": message}
                        break
            finally:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except BaseException:  # noqa: BLE001, S110
                        pass
                # Persist whatever state we ended up with (including partials).
                self.session_store.save(session)

            yield EVENT_DONE, {}
        finally:
            self._mark_free(session.session_id)

    def is_busy(self, session_id: str) -> bool:
        return session_id in self._busy_sessions

    def _mark_busy(self, session_id: str) -> None:
        if session_id in self._busy_sessions:
            raise SessionBusy(session_id)
        self._busy_sessions.add(session_id)

    def _mark_free(self, session_id: str) -> None:
        self._busy_sessions.discard(session_id)
