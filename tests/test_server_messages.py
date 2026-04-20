from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - server extras not installed
    TestClient = None  # type: ignore[assignment]

from whesper.config import load_config
from whesper.session import ChatMessage


SERVER_EXTRAS_AVAILABLE = TestClient is not None


def _write_config(storage_dir: Path, *, token: str = "t") -> object:
    body = textwrap.dedent(
        f"""
        [app]
        storage_dir = "{storage_dir}"

        [scheduler]
        chat_model = "local"

        [providers.main]
        base_url = "http://localhost:11434/v1"

        [models.local]
        provider = "main"
        model = "qwen"

        [server]
        enabled = true
        require_auth = true
        api_token = "{token}"
        """
    ).strip()
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write(body)
        temp_path = fh.name
    return load_config(temp_path)


def _seed_session(app_ref, session_id: str, n: int) -> None:
    session = app_ref.create_session(session_id)
    for i in range(n):
        # Timestamps are monotonically increasing within a session.
        stamp = f"2026-04-20T10:{i:02d}:00+00:00"
        session.messages.append(
            ChatMessage(role="user", content=f"u{i}", created_at=stamp)
        )
        session.transcript_messages.append(
            ChatMessage(role="user", content=f"u{i}", created_at=stamp)
        )
        session.messages.append(
            ChatMessage(
                role="assistant",
                content=f"a{i}",
                created_at=stamp,
                model_alias="local",
            )
        )
        # transcript only keeps clean assistant messages (simulate)
        session.transcript_messages.append(
            ChatMessage(
                role="assistant",
                content=f"a{i}",
                created_at=stamp,
                model_alias="local",
            )
        )
    app_ref.session_store.save(session)


@unittest.skipUnless(
    SERVER_EXTRAS_AVAILABLE, "fastapi extra not installed; skip message tests"
)
class ServerMessagesTests(unittest.TestCase):
    def _build(self, tmpdir: str):
        from whesper.api.application import AgentApplication
        from whesper.server.http import create_app

        config = _write_config(Path(tmpdir))
        app_ref = AgentApplication(config)
        app = create_app(app_ref)
        return app_ref, TestClient(app)

    def test_requires_auth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_ref, client = self._build(tmpdir)
            _seed_session(app_ref, "s1", 1)
            resp = client.get("/v1/sessions/s1/messages")
            self.assertEqual(resp.status_code, 401)

    def test_missing_session_returns_404(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, client = self._build(tmpdir)
            resp = client.get(
                "/v1/sessions/missing/messages",
                headers={"Authorization": "Bearer t"},
            )
            self.assertEqual(resp.status_code, 404)
            self.assertEqual(resp.json()["code"], "NOT_FOUND")

    def test_returns_newest_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_ref, client = self._build(tmpdir)
            _seed_session(app_ref, "s1", 3)  # 6 messages total
            resp = client.get(
                "/v1/sessions/s1/messages",
                headers={"Authorization": "Bearer t"},
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(len(body["items"]), 6)
            stamps = [m["created_at"] for m in body["items"]]
            self.assertEqual(stamps, sorted(stamps, reverse=True))
            self.assertIsNone(body["next_before"])

    def test_pagination_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_ref, client = self._build(tmpdir)
            _seed_session(app_ref, "s1", 5)  # 10 messages total

            resp = client.get(
                "/v1/sessions/s1/messages?limit=4",
                headers={"Authorization": "Bearer t"},
            )
            self.assertEqual(resp.status_code, 200)
            page1 = resp.json()
            self.assertEqual(len(page1["items"]), 4)
            self.assertIsNotNone(page1["next_before"])

            resp = client.get(
                f"/v1/sessions/s1/messages?limit=4&before={page1['next_before']}",
                headers={"Authorization": "Bearer t"},
            )
            self.assertEqual(resp.status_code, 200)
            page2 = resp.json()
            # All items in page2 strictly older than cursor.
            for m in page2["items"]:
                self.assertLess(m["created_at"], page1["next_before"])

    def test_limit_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_ref, client = self._build(tmpdir)
            _seed_session(app_ref, "s1", 1)
            resp = client.get(
                "/v1/sessions/s1/messages?limit=0",
                headers={"Authorization": "Bearer t"},
            )
            self.assertEqual(resp.status_code, 400)
            self.assertEqual(resp.json()["code"], "VALIDATION_ERROR")

    def test_transcript_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_ref, client = self._build(tmpdir)
            # Seed messages with an extra "tool" message that transcript excludes.
            session = app_ref.create_session("s1")
            session.messages.append(
                ChatMessage(role="user", content="hi", created_at="2026-04-20T10:00:00+00:00")
            )
            session.messages.append(
                ChatMessage(
                    role="tool",
                    content="internal",
                    created_at="2026-04-20T10:01:00+00:00",
                )
            )
            session.transcript_messages.append(
                ChatMessage(role="user", content="hi", created_at="2026-04-20T10:00:00+00:00")
            )
            app_ref.session_store.save(session)

            resp = client.get(
                "/v1/sessions/s1/messages",
                headers={"Authorization": "Bearer t"},
            )
            self.assertEqual(len(resp.json()["items"]), 2)

            resp = client.get(
                "/v1/sessions/s1/transcript",
                headers={"Authorization": "Bearer t"},
            )
            self.assertEqual(resp.status_code, 200)
            items = resp.json()["items"]
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["role"], "user")


if __name__ == "__main__":
    unittest.main()
