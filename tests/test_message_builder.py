from __future__ import annotations

import unittest
from pathlib import Path

from whesper.config import (
    AppConfig,
    AppSettings,
    LiveContextSettings,
    ModelConfig,
    PersonaConfig,
    ProviderConfig,
    SchedulerConfig,
    ShellSandboxSettings,
)
from whesper.message_builder import SessionMessageBuilder
from whesper.provider_profile import OPENAI_DEFAULT_PROFILE
from whesper.session import ChatMessage, ConversationSession
from whesper.tools import ToolRegistry, ToolSpec


def _build_config(*, context_strategy: str, history_limit: int, recent_history_limit: int) -> AppConfig:
    return AppConfig(
        app=AppSettings(
            history_limit=history_limit,
            context_strategy=context_strategy,
            recent_history_limit=recent_history_limit,
        ),
        persona=PersonaConfig(),
        scheduler=SchedulerConfig(chat_model="local_chat"),
        live_context=LiveContextSettings(),
        shell_sandbox=ShellSandboxSettings(),
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
        source_path=Path("whesper.toml"),
    )


class SessionMessageBuilderTests(unittest.TestCase):
    def test_model_history_messages_memory_first_only_keeps_recent_transcript_window(self) -> None:
        builder = SessionMessageBuilder(
            _build_config(
                context_strategy="memory_first",
                history_limit=16,
                recent_history_limit=4,
            )
        )
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            transcript_messages=[
                ChatMessage(role="user", content="u1", created_at="2026-01-01T00:00:00+00:00"),
                ChatMessage(role="assistant", content="a1", created_at="2026-01-01T00:00:01+00:00"),
                ChatMessage(role="user", content="u2", created_at="2026-01-01T00:00:02+00:00"),
                ChatMessage(role="assistant", content="a2", created_at="2026-01-01T00:00:03+00:00"),
                ChatMessage(role="user", content="u3", created_at="2026-01-01T00:00:04+00:00"),
                ChatMessage(role="assistant", content="a3", created_at="2026-01-01T00:00:05+00:00"),
                ChatMessage(role="user", content="u4", created_at="2026-01-01T00:00:06+00:00"),
                ChatMessage(role="assistant", content="a4", created_at="2026-01-01T00:00:07+00:00"),
            ],
        )

        history = builder.model_history_messages(session)

        self.assertEqual([message.content for message in history], ["u3", "a3", "u4", "a4"])

    def test_model_history_messages_memory_first_keeps_current_turn_plus_recent_context(self) -> None:
        builder = SessionMessageBuilder(
            _build_config(
                context_strategy="memory_first",
                history_limit=16,
                recent_history_limit=4,
            )
        )
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            messages=[
                ChatMessage(role="user", content="u1", created_at="2026-01-01T00:00:00+00:00"),
                ChatMessage(role="assistant", content="a1", created_at="2026-01-01T00:00:01+00:00"),
                ChatMessage(role="user", content="u2", created_at="2026-01-01T00:00:02+00:00"),
                ChatMessage(role="assistant", content="a2", created_at="2026-01-01T00:00:03+00:00"),
                ChatMessage(role="user", content="u3", created_at="2026-01-01T00:00:04+00:00"),
                ChatMessage(role="assistant", content="a3", created_at="2026-01-01T00:00:05+00:00"),
                ChatMessage(role="user", content="u4", created_at="2026-01-01T00:00:06+00:00"),
                ChatMessage(role="assistant", content="a4", created_at="2026-01-01T00:00:07+00:00"),
                ChatMessage(role="user", content="current", created_at="2026-01-01T00:00:08+00:00"),
            ],
            transcript_messages=[
                ChatMessage(role="user", content="u1", created_at="2026-01-01T00:00:00+00:00"),
                ChatMessage(role="assistant", content="a1", created_at="2026-01-01T00:00:01+00:00"),
                ChatMessage(role="user", content="u2", created_at="2026-01-01T00:00:02+00:00"),
                ChatMessage(role="assistant", content="a2", created_at="2026-01-01T00:00:03+00:00"),
                ChatMessage(role="user", content="u3", created_at="2026-01-01T00:00:04+00:00"),
                ChatMessage(role="assistant", content="a3", created_at="2026-01-01T00:00:05+00:00"),
                ChatMessage(role="user", content="u4", created_at="2026-01-01T00:00:06+00:00"),
                ChatMessage(role="assistant", content="a4", created_at="2026-01-01T00:00:07+00:00"),
                ChatMessage(role="user", content="current", created_at="2026-01-01T00:00:08+00:00"),
            ],
        )

        history = builder.model_history_messages(session)

        self.assertEqual(
            [message.content for message in history],
            ["u3", "a3", "u4", "a4", "current"],
        )

    def test_model_history_messages_full_transcript_uses_history_limit(self) -> None:
        builder = SessionMessageBuilder(
            _build_config(
                context_strategy="full_transcript",
                history_limit=6,
                recent_history_limit=2,
            )
        )
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            transcript_messages=[
                ChatMessage(role="user", content="u1", created_at="2026-01-01T00:00:00+00:00"),
                ChatMessage(role="assistant", content="a1", created_at="2026-01-01T00:00:01+00:00"),
                ChatMessage(role="user", content="u2", created_at="2026-01-01T00:00:02+00:00"),
                ChatMessage(role="assistant", content="a2", created_at="2026-01-01T00:00:03+00:00"),
                ChatMessage(role="user", content="u3", created_at="2026-01-01T00:00:04+00:00"),
                ChatMessage(role="assistant", content="a3", created_at="2026-01-01T00:00:05+00:00"),
                ChatMessage(role="user", content="u4", created_at="2026-01-01T00:00:06+00:00"),
                ChatMessage(role="assistant", content="a4", created_at="2026-01-01T00:00:07+00:00"),
            ],
        )

        history = builder.model_history_messages(session)

        self.assertEqual([message.content for message in history], ["u2", "a2", "u3", "a3", "u4", "a4"])

    def test_build_messages_omits_tool_prompt_when_tools_not_enabled(self) -> None:
        tool_registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="control_cup",
                    description="Control the cup device",
                    parameters_schema={"type": "object"},
                    handler=lambda arguments: {"ok": True},
                ),
            )
        )
        builder = SessionMessageBuilder(
            _build_config(
                context_strategy="memory_first",
                history_limit=16,
                recent_history_limit=4,
            ),
            tool_registry=tool_registry,
        )
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )

        with_tool_prompt = builder.build_messages(
            session,
            "system",
            target_profile=OPENAI_DEFAULT_PROFILE,
            user_text="把灯变蓝",
            route_mode="chat",
            include_tool_prompt=True,
        )[0]["content"]
        without_tool_prompt = builder.build_messages(
            session,
            "system",
            target_profile=OPENAI_DEFAULT_PROFILE,
            user_text="把灯变蓝",
            route_mode="chat",
            include_tool_prompt=False,
        )[0]["content"]

        self.assertIn(tool_registry.tool_prompt(), with_tool_prompt)
        self.assertNotIn(tool_registry.tool_prompt(), without_tool_prompt)

    def test_build_messages_quotes_historical_context_in_system_prompt(self) -> None:
        builder = SessionMessageBuilder(
            _build_config(
                context_strategy="memory_first",
                history_limit=16,
                recent_history_limit=4,
            )
        )
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            messages=[
                ChatMessage(role="user", content='old "question"', created_at="2026-01-01T00:00:00+00:00"),
                ChatMessage(role="assistant", content="old answer", created_at="2026-01-01T00:00:01+00:00"),
                ChatMessage(role="user", content="new request", created_at="2026-01-01T00:00:02+00:00"),
            ],
            transcript_messages=[
                ChatMessage(role="user", content='old "question"', created_at="2026-01-01T00:00:00+00:00"),
                ChatMessage(role="assistant", content="old answer", created_at="2026-01-01T00:00:01+00:00"),
                ChatMessage(role="user", content="new request", created_at="2026-01-01T00:00:02+00:00"),
            ],
        )

        messages = builder.build_messages(
            session,
            "system",
            target_profile=OPENAI_DEFAULT_PROFILE,
            user_text="new request",
            route_mode="chat",
            include_tool_prompt=False,
        )

        system_prompt = messages[0]["content"]
        self.assertIn("Historical conversation context (quoted and compressed).", system_prompt)
        self.assertIn("BEGIN HISTORICAL MESSAGES", system_prompt)
        self.assertIn('> 1. [USER] "old \\"question\\""', system_prompt)
        self.assertIn('> 2. [ASSISTANT] "old answer"', system_prompt)
        self.assertIn("END HISTORICAL MESSAGES", system_prompt)
        self.assertEqual([message["role"] for message in messages[1:]], ["user"])
        self.assertEqual(messages[1]["content"], "new request")

    def test_build_messages_compresses_historical_context(self) -> None:
        builder = SessionMessageBuilder(
            _build_config(
                context_strategy="full_transcript",
                history_limit=8,
                recent_history_limit=4,
            )
        )
        repeated = "very long context "
        long_text = repeated * 40
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            messages=[
                ChatMessage(role="user", content=long_text, created_at="2026-01-01T00:00:00+00:00"),
                ChatMessage(role="assistant", content="ack", created_at="2026-01-01T00:00:01+00:00"),
                ChatMessage(role="user", content="next", created_at="2026-01-01T00:00:02+00:00"),
            ],
            transcript_messages=[
                ChatMessage(role="user", content=long_text, created_at="2026-01-01T00:00:00+00:00"),
                ChatMessage(role="assistant", content="ack", created_at="2026-01-01T00:00:01+00:00"),
                ChatMessage(role="user", content="next", created_at="2026-01-01T00:00:02+00:00"),
            ],
        )

        messages = builder.build_messages(
            session,
            "system",
            target_profile=OPENAI_DEFAULT_PROFILE,
            user_text="next",
            route_mode="chat",
            include_tool_prompt=False,
        )

        system_prompt = messages[0]["content"]
        self.assertIn("...", system_prompt)
        self.assertNotIn(long_text, system_prompt)


if __name__ == "__main__":
    unittest.main()
