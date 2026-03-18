from __future__ import annotations

import io
import tempfile
import unittest

from whesper.cli import (
    apply_pinned_model_override,
    build_footer_meta,
    ensure_valid_session_model,
    handle_command,
)
from whesper.chat import ChatService, ChatTurnResult
from whesper.commands import ParsedCommand
from whesper.config import (
    AppConfig,
    AppSettings,
    ModelConfig,
    PersonaConfig,
    ProviderConfig,
    SchedulerConfig,
)
from whesper.router import RouteDecision
from whesper.session import ChatMessage, ConversationSession, SessionStore


class FakeStreamingClient:
    def create_chat_completion_stream(self, provider, model, messages):
        yield "new"
        yield " reply"

    def create_chat_completion(self, provider, model, messages):
        raise AssertionError("fallback should not be called in this test")


class InterruptingStreamingClient:
    def create_chat_completion_stream(self, provider, model, messages):
        yield "partial"
        raise KeyboardInterrupt()

    def create_chat_completion(self, provider, model, messages):
        raise AssertionError("fallback should not be called in this test")


def build_config() -> AppConfig:
    return AppConfig(
        app=AppSettings(),
        persona=PersonaConfig(),
        scheduler=SchedulerConfig(chat_model="local_chat"),
        providers={
            "local": ProviderConfig(
                name="local",
                kind="ollama_native",
                base_url="http://localhost:11434",
            )
        },
        models={
            "local_chat": ModelConfig(
                name="local_chat",
                provider="local",
                model="qwen",
            )
        },
        source_path=None,  # type: ignore[arg-type]
    )


class CliTests(unittest.TestCase):
    def build_chat_service(self, config: AppConfig) -> ChatService:
        return ChatService(config, client=FakeStreamingClient())

    def build_interrupting_chat_service(self, config: AppConfig) -> ChatService:
        return ChatService(config, client=InterruptingStreamingClient())

    def test_invalid_model_alias_is_handled_without_crash(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/model", arg="missing-model"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertFalse(outcome.should_exit)
        self.assertEqual(outcome.session.pinned_model, "auto")
        self.assertIn("Unknown model alias: missing-model", rendered)
        self.assertIn("available models: local_chat", rendered)

    def test_invalid_mode_is_handled_without_crash(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/mode", arg="12345"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertFalse(outcome.should_exit)
        self.assertEqual(outcome.mode_override, "auto")
        self.assertIn("Invalid mode: 12345", rendered)
        self.assertIn("available modes: auto, chat, reasoning, search", rendered)

    def test_invalid_saved_session_model_is_reset_to_auto(self) -> None:
        config = build_config()
        session = ConversationSession(
            session_id="main",
            created_at="2026-03-18T00:00:00+00:00",
            updated_at="2026-03-18T00:00:00+00:00",
            pinned_model="missing-model",
        )
        output = io.StringIO()

        changed = ensure_valid_session_model(
            config,
            session,
            output_stream=output,
        )

        rendered = output.getvalue()
        self.assertTrue(changed)
        self.assertEqual(session.pinned_model, "auto")
        self.assertIn("Unknown model alias: missing-model", rendered)
        self.assertIn("Session model was reset to auto.", rendered)

    def test_invalid_startup_model_override_keeps_auto(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            apply_pinned_model_override(
                config,
                store,
                session,
                "missing-model",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertEqual(session.pinned_model, "auto")
        self.assertIn("Unknown model alias: missing-model", rendered)
        self.assertIn("Starting with session model: auto.", rendered)

    def test_status_command_renders_current_state(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/status"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("session: main", rendered)
        self.assertIn("effective model: local_chat", rendered)

    def test_info_command_renders_model_details(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/info"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("alias: local_chat", rendered)
        self.assertIn("model id: qwen", rendered)

    def test_history_command_renders_colored_cards_for_roles(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            session.messages.extend(
                [
                    ChatMessage(
                        role="user",
                        content="hello there",
                        created_at="2026-03-18T00:00:00+00:00",
                    ),
                    ChatMessage(
                        role="assistant",
                        content="hi back",
                        created_at="2026-03-18T00:00:01+00:00",
                        model_alias="local_chat",
                        route_reason="default chat model",
                    ),
                ]
            )
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/history"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("History (2)", rendered)
        self.assertIn("You", rendered)
        self.assertIn("Whesper", rendered)
        self.assertIn("model local_chat", rendered)
        self.assertIn("default chat model", rendered)
        self.assertIn("hello there", rendered)
        self.assertIn("hi back", rendered)

    def test_history_command_handles_empty_session(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/history"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("Session is empty.", rendered)

    def test_retry_command_replaces_last_assistant_reply(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            session.messages.extend(
                [
                    ChatMessage(
                        role="user",
                        content="hello",
                        created_at="2026-03-18T00:00:00+00:00",
                    ),
                    ChatMessage(
                        role="assistant",
                        content="old reply",
                        created_at="2026-03-18T00:00:01+00:00",
                        model_alias="local_chat",
                    ),
                ]
            )
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/retry"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertEqual(session.messages[-1].content, "new reply")
        self.assertEqual(len(session.messages), 2)
        self.assertIn("Retrying the last user message...", rendered)
        self.assertIn("new", rendered)

    def test_retry_command_handles_keyboard_interrupt_without_crash(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            session.messages.extend(
                [
                    ChatMessage(
                        role="user",
                        content="hello",
                        created_at="2026-03-18T00:00:00+00:00",
                    ),
                    ChatMessage(
                        role="assistant",
                        content="old reply",
                        created_at="2026-03-18T00:00:01+00:00",
                        model_alias="local_chat",
                    ),
                ]
            )
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/retry"),
                config=config,
                store=store,
                chat_service=self.build_interrupting_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("Generation interrupted. Partial reply saved.", rendered)
        self.assertEqual(session.messages[-1].content, "partial")

    def test_copy_last_prints_last_assistant_reply(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            session.messages.append(
                ChatMessage(
                    role="assistant",
                    content="copied text",
                    created_at="2026-03-18T00:00:00+00:00",
                )
            )
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/copy-last"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("copied text", rendered)

    def test_rename_command_updates_session(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/rename", arg="renamed"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

            sessions = store.list_sessions()

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertEqual(outcome.session.session_id, "renamed")
        self.assertIn("Renamed session to: renamed", rendered)
        self.assertIn("renamed", sessions)

    def test_delete_session_removes_named_session(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            store.load("throwaway")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/delete-session", arg="throwaway"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

            sessions = store.list_sessions()

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("Deleted session: throwaway", rendered)
        self.assertNotIn("throwaway", sessions)

    def test_build_footer_meta_includes_development_details(self) -> None:
        config = build_config()
        session = ConversationSession(
            session_id="main",
            created_at="2026-03-18T00:00:00+00:00",
            updated_at="2026-03-18T00:00:01+00:00",
            messages=[
                ChatMessage(
                    role="user",
                    content="hello",
                    created_at="2026-03-18T00:00:00+00:00",
                ),
                ChatMessage(
                    role="assistant",
                    content="hello back",
                    created_at="2026-03-18T00:00:01+00:00",
                    model_alias="local_chat",
                ),
            ],
        )
        result = ChatTurnResult(
            decision=RouteDecision(
                model_alias="local_chat",
                mode="chat",
                reason="default chat model",
            ),
            assistant_message=session.messages[-1],
        )

        footer = build_footer_meta(
            config,
            result,
            total_seconds=4.8,
            first_token_seconds=1.2,
            session=session,
        )

        self.assertIn("done in 4.8s", footer)
        self.assertIn("first token 1.2s", footer)
        self.assertIn("model local_chat -> qwen", footer)
        self.assertIn("provider local/ollama_native", footer)
        self.assertIn("route chat", footer)
        self.assertIn("reason default chat model", footer)
        self.assertIn("10 chars", footer)
        self.assertIn("2 msgs", footer)


if __name__ == "__main__":
    unittest.main()
