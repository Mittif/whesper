from __future__ import annotations

import tempfile
import textwrap
import unittest

from whesper.chat import ChatService, GenerationInterrupted
from whesper.config import load_config
from whesper.session import ConversationSession


class FakeStreamingClient:
    def create_chat_completion_stream(self, provider, model, messages):
        yield "你好"
        yield "，世界"

    def create_chat_completion(self, provider, model, messages):
        raise AssertionError("fallback should not be called in this test")


class InterruptingStreamingClient:
    def create_chat_completion_stream(self, provider, model, messages):
        yield "partial"
        raise KeyboardInterrupt()

    def create_chat_completion(self, provider, model, messages):
        raise AssertionError("fallback should not be called in this test")


def make_config():
    content = textwrap.dedent(
        """
        [scheduler]
        chat_model = "chat"

        [providers.local]
        base_url = "http://localhost:11434/v1"

        [models.chat]
        provider = "local"
        model = "chat-model"
        """
    ).strip()

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write(content)
        temp_path = fh.name
    return load_config(temp_path)


class ChatTests(unittest.TestCase):
    def test_send_stream_builds_assistant_message(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        chunks: list[str] = []
        service = ChatService(config, client=FakeStreamingClient())

        result = service.send_stream(
            session,
            "你好",
            on_chunk=chunks.append,
        )

        self.assertEqual(chunks, ["你好", "，世界"])
        self.assertEqual(result.assistant_message.content, "你好，世界")
        self.assertEqual(session.messages[-1].content, "你好，世界")

    def test_retry_stream_replaces_last_assistant_message(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        service = ChatService(config, client=FakeStreamingClient())

        service.send_stream(session, "你好")
        session.messages[-1].content = "旧回复"

        chunks: list[str] = []
        result = service.retry_stream(
            session,
            on_chunk=chunks.append,
        )

        self.assertEqual(chunks, ["你好", "，世界"])
        self.assertEqual(result.assistant_message.content, "你好，世界")
        self.assertEqual(len(session.messages), 2)
        self.assertEqual(session.messages[0].role, "user")
        self.assertEqual(session.messages[1].role, "assistant")
        self.assertEqual(session.messages[1].content, "你好，世界")

    def test_send_stream_preserves_partial_reply_on_interrupt(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        service = ChatService(config, client=InterruptingStreamingClient())
        chunks: list[str] = []

        with self.assertRaises(GenerationInterrupted) as context:
            service.send_stream(
                session,
                "你好",
                on_chunk=chunks.append,
            )

        self.assertEqual(chunks, ["partial"])
        self.assertIsNotNone(context.exception.partial_result)
        assert context.exception.partial_result is not None
        self.assertEqual(context.exception.partial_result.assistant_message.content, "partial")
        self.assertEqual(len(session.messages), 2)
        self.assertEqual(session.messages[-1].content, "partial")


if __name__ == "__main__":
    unittest.main()
