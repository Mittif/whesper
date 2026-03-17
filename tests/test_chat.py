from __future__ import annotations

import tempfile
import textwrap
import unittest

from whesper.chat import ChatService
from whesper.config import load_config
from whesper.session import ConversationSession


class FakeStreamingClient:
    def create_chat_completion_stream(self, provider, model, messages):
        yield "你好"
        yield "，世界"

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


if __name__ == "__main__":
    unittest.main()
