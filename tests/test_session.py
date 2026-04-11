from __future__ import annotations

import tempfile
import unittest

from whesper.agent_types import AskUserAction, AskUserOption
from whesper.session import ChatMessage, SessionStore


class SessionStoreTests(unittest.TestCase):
    def test_save_and_load_transcript_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            session.messages.extend(
                (
                    ChatMessage(
                        role="user",
                        content="hello",
                        created_at="2026-01-01T00:00:00+00:00",
                    ),
                    ChatMessage(
                        role="assistant",
                        content="",
                        created_at="2026-01-01T00:00:01+00:00",
                        tool_calls=[
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "web_search", "arguments": "{}"},
                            }
                        ],
                    ),
                    ChatMessage(
                        role="tool",
                        content='{"result":"ok"}',
                        created_at="2026-01-01T00:00:02+00:00",
                        tool_call_id="call_1",
                    ),
                    ChatMessage(
                        role="assistant",
                        content="final reply",
                        created_at="2026-01-01T00:00:03+00:00",
                    ),
                )
            )
            session.transcript_messages.extend(
                (
                    ChatMessage(
                        role="user",
                        content="hello",
                        created_at="2026-01-01T00:00:00+00:00",
                    ),
                    ChatMessage(
                        role="assistant",
                        content="final reply",
                        created_at="2026-01-01T00:00:03+00:00",
                    ),
                )
            )

            store.save(session)
            loaded = store.load("main")

            self.assertEqual(len(loaded.messages), 4)
            self.assertEqual(len(loaded.transcript_messages), 2)
            self.assertEqual(loaded.transcript_messages[0].content, "hello")
            self.assertEqual(loaded.transcript_messages[1].content, "final reply")

    def test_delete_session_removes_transcript_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            session.transcript_messages.append(
                ChatMessage(
                    role="user",
                    content="hello",
                    created_at="2026-01-01T00:00:00+00:00",
                )
            )
            store.save(session)

            deleted = store.delete_session("main")

            self.assertTrue(deleted)
            self.assertFalse(store.path_for("main").exists())
            self.assertFalse(store.transcript_path_for("main").exists())

    def test_save_and_load_pending_ask_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            session.pending_ask_user = AskUserAction(
                prompt="你想查哪个城市？",
                options=(
                    AskUserOption(label="上海", value="上海"),
                    AskUserOption(label="东京", value="东京", description="更快一点"),
                ),
                allow_free_text=True,
                field_name="location",
            )

            store.save(session)
            loaded = store.load("main")

            self.assertIsNotNone(loaded.pending_ask_user)
            assert loaded.pending_ask_user is not None
            self.assertEqual(loaded.pending_ask_user.prompt, "你想查哪个城市？")
            self.assertEqual(loaded.pending_ask_user.options[0].label, "上海")
            self.assertEqual(loaded.pending_ask_user.options[1].description, "更快一点")
            self.assertEqual(loaded.pending_ask_user.field_name, "location")


if __name__ == "__main__":
    unittest.main()
