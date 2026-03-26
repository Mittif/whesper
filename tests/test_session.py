from __future__ import annotations

import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
