from __future__ import annotations

import json
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - server extras not installed
    TestClient = None  # type: ignore[assignment]

from whesper.agent_types import AgentStep, AskUserAction, AskUserOption
from whesper.chat import ChatTurnResult, GenerationInterrupted
from whesper.client import ProviderError
from whesper.config import ConfigError, load_config
from whesper.router import RouteDecision
from whesper.session import ChatMessage, utc_now_iso
from whesper.tools import ToolExecutionError


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


class _FakeChatService:
    """Stub that mimics ChatService.send / send_stream for tests."""

    def __init__(
        self,
        *,
        chunks: tuple[str, ...] = (),
        steps: tuple[AgentStep, ...] = (),
        ask_user: AskUserAction | None = None,
        error: BaseException | None = None,
        content: str = "done",
        delay_seconds: float = 0.0,
    ) -> None:
        self.chunks = chunks
        self.steps = steps
        self.ask_user = ask_user
        self.error = error
        self.content = content
        self.delay_seconds = delay_seconds
        self.calls = 0

    def _make_result(self) -> ChatTurnResult:
        decision = RouteDecision(model_alias="local", mode="chat", reason="test")
        assistant = ChatMessage(
            role="assistant",
            content=self.content,
            created_at=utc_now_iso(),
            model_alias="local",
        )
        return ChatTurnResult(
            decision=decision,
            assistant_message=assistant,
            ask_user=self.ask_user,
        )

    def send(self, session, user_text, *, mode_override="auto", on_step=None):
        self.calls += 1
        session.messages.append(
            ChatMessage(role="user", content=user_text, created_at=utc_now_iso())
        )
        if self.error is not None:
            raise self.error
        result = self._make_result()
        session.messages.append(result.assistant_message)
        return result

    def send_stream(
        self,
        session,
        user_text,
        *,
        mode_override="auto",
        on_chunk=None,
        on_step=None,
    ):
        self.calls += 1
        session.messages.append(
            ChatMessage(role="user", content=user_text, created_at=utc_now_iso())
        )
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        for step in self.steps:
            if on_step is not None:
                on_step(step)
        for chunk in self.chunks:
            if on_chunk is not None:
                on_chunk(chunk)
        if self.error is not None:
            raise self.error
        result = self._make_result()
        session.messages.append(result.assistant_message)
        return result


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        name: str | None = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if name is None:
            continue
        payload = json.loads("\n".join(data_lines)) if data_lines else {}
        events.append((name, payload))
    return events


@unittest.skipUnless(
    SERVER_EXTRAS_AVAILABLE, "fastapi extra not installed; skip turn tests"
)
class ServerTurnTests(unittest.TestCase):
    def _build(self, tmpdir: str, fake: _FakeChatService | None = None):
        from whesper.api.application import AgentApplication
        from whesper.server.http import create_app

        config = _write_config(Path(tmpdir))
        app_ref = AgentApplication(config)
        if fake is not None:
            app_ref._chat_service = fake  # type: ignore[assignment]
        app = create_app(app_ref)
        return app_ref, TestClient(app)

    def test_turn_happy_path_returns_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake = _FakeChatService(content="hello back")
            app_ref, client = self._build(tmpdir, fake)
            app_ref.create_session("s1")

            resp = client.post(
                "/v1/sessions/s1/turn",
                headers={"Authorization": "Bearer t"},
                json={"text": "hi"},
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(body["assistant_message"]["content"], "hello back")
            self.assertEqual(body["decision"]["model_alias"], "local")
            self.assertIsNone(body["ask_user"])

    def test_turn_returns_ask_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ask = AskUserAction(
                prompt="Which?",
                options=(AskUserOption(label="A", value="a"),),
                allow_free_text=False,
            )
            fake = _FakeChatService(ask_user=ask)
            app_ref, client = self._build(tmpdir, fake)
            app_ref.create_session("s1")

            resp = client.post(
                "/v1/sessions/s1/turn",
                headers={"Authorization": "Bearer t"},
                json={"text": "pick"},
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertIsNotNone(body["ask_user"])
            self.assertEqual(body["ask_user"]["prompt"], "Which?")
            self.assertEqual(body["ask_user"]["options"][0]["value"], "a")

    def test_turn_missing_session_returns_404(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, client = self._build(tmpdir, _FakeChatService())
            resp = client.post(
                "/v1/sessions/nope/turn",
                headers={"Authorization": "Bearer t"},
                json={"text": "hi"},
            )
            self.assertEqual(resp.status_code, 404)
            self.assertEqual(resp.json()["code"], "NOT_FOUND")

    def test_turn_provider_error_becomes_502(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake = _FakeChatService(error=ProviderError("upstream down"))
            app_ref, client = self._build(tmpdir, fake)
            app_ref.create_session("s1")

            resp = client.post(
                "/v1/sessions/s1/turn",
                headers={"Authorization": "Bearer t"},
                json={"text": "hi"},
            )
            self.assertEqual(resp.status_code, 502)
            self.assertEqual(resp.json()["code"], "PROVIDER_ERROR")

    def test_turn_tool_execution_error_becomes_500(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake = _FakeChatService(error=ToolExecutionError("tool failed"))
            app_ref, client = self._build(tmpdir, fake)
            app_ref.create_session("s1")

            resp = client.post(
                "/v1/sessions/s1/turn",
                headers={"Authorization": "Bearer t"},
                json={"text": "hi"},
            )
            self.assertEqual(resp.status_code, 500)
            self.assertEqual(resp.json()["code"], "TOOL_EXECUTION_ERROR")

    def test_turn_config_error_becomes_500(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake = _FakeChatService(error=ConfigError("bad config"))
            app_ref, client = self._build(tmpdir, fake)
            app_ref.create_session("s1")

            resp = client.post(
                "/v1/sessions/s1/turn",
                headers={"Authorization": "Bearer t"},
                json={"text": "hi"},
            )
            self.assertEqual(resp.status_code, 500)
            self.assertEqual(resp.json()["code"], "CONFIG_ERROR")

    def test_turn_generation_interrupted_returns_499(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake = _FakeChatService(error=GenerationInterrupted())
            app_ref, client = self._build(tmpdir, fake)
            app_ref.create_session("s1")

            resp = client.post(
                "/v1/sessions/s1/turn",
                headers={"Authorization": "Bearer t"},
                json={"text": "hi"},
            )
            self.assertEqual(resp.status_code, 499)
            self.assertEqual(resp.json()["code"], "GENERATION_INTERRUPTED")

    def test_turn_requires_auth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_ref, client = self._build(tmpdir, _FakeChatService())
            app_ref.create_session("s1")
            resp = client.post("/v1/sessions/s1/turn", json={"text": "hi"})
            self.assertEqual(resp.status_code, 401)

    def test_turn_rejects_blank_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_ref, client = self._build(tmpdir, _FakeChatService())
            app_ref.create_session("s1")
            resp = client.post(
                "/v1/sessions/s1/turn",
                headers={"Authorization": "Bearer t"},
                json={"text": ""},
            )
            self.assertEqual(resp.status_code, 422)

    def test_stream_emits_route_step_chunk_final_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake = _FakeChatService(
                chunks=("he", "llo"),
                steps=(
                    AgentStep(round_index=0, kind="tool_call", tool_name="search"),
                ),
                content="hello",
            )
            app_ref, client = self._build(tmpdir, fake)
            app_ref.create_session("s1")

            with client.stream(
                "POST",
                "/v1/sessions/s1/turn:stream",
                headers={"Authorization": "Bearer t"},
                json={"text": "hi"},
            ) as resp:
                self.assertEqual(resp.status_code, 200)
                self.assertTrue(resp.headers["content-type"].startswith("text/event-stream"))
                body = resp.read().decode("utf-8")

            events = _parse_sse(body)
            names = [name for name, _ in events]
            self.assertEqual(names[0], "route")
            self.assertEqual(names[-1], "done")
            self.assertIn("step", names)
            chunks = [payload["text"] for name, payload in events if name == "chunk"]
            self.assertEqual(chunks, ["he", "llo"])
            finals = [payload for name, payload in events if name == "final"]
            self.assertEqual(len(finals), 1)
            self.assertEqual(finals[0]["assistant_message"]["content"], "hello")

    def test_stream_emits_ask_user_instead_of_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake = _FakeChatService(
                ask_user=AskUserAction(prompt="?", allow_free_text=True),
            )
            app_ref, client = self._build(tmpdir, fake)
            app_ref.create_session("s1")

            with client.stream(
                "POST",
                "/v1/sessions/s1/turn:stream",
                headers={"Authorization": "Bearer t"},
                json={"text": "hi"},
            ) as resp:
                body = resp.read().decode("utf-8")

            events = _parse_sse(body)
            names = [name for name, _ in events]
            self.assertIn("ask_user", names)
            self.assertNotIn("final", names)
            self.assertEqual(names[-1], "done")

    def test_stream_emits_error_event_on_provider_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake = _FakeChatService(error=ProviderError("boom"))
            app_ref, client = self._build(tmpdir, fake)
            app_ref.create_session("s1")

            with client.stream(
                "POST",
                "/v1/sessions/s1/turn:stream",
                headers={"Authorization": "Bearer t"},
                json={"text": "hi"},
            ) as resp:
                body = resp.read().decode("utf-8")

            events = _parse_sse(body)
            names = [name for name, _ in events]
            errors = [payload for name, payload in events if name == "error"]
            self.assertEqual(names[-1], "done")
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["code"], "PROVIDER_ERROR")

    def test_stream_missing_session_returns_404(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, client = self._build(tmpdir, _FakeChatService())
            resp = client.post(
                "/v1/sessions/nope/turn:stream",
                headers={"Authorization": "Bearer t"},
                json={"text": "hi"},
            )
            self.assertEqual(resp.status_code, 404)

    def test_concurrent_turn_returns_409(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake = _FakeChatService(delay_seconds=0.3)
            app_ref, client = self._build(tmpdir, fake)
            app_ref.create_session("s1")

            results: list[int] = []

            def hit_stream() -> None:
                with client.stream(
                    "POST",
                    "/v1/sessions/s1/turn:stream",
                    headers={"Authorization": "Bearer t"},
                    json={"text": "hi"},
                ) as resp:
                    resp.read()
                    results.append(resp.status_code)

            t = threading.Thread(target=hit_stream)
            t.start()
            # Give the first request time to mark busy.
            time.sleep(0.1)
            resp = client.post(
                "/v1/sessions/s1/turn",
                headers={"Authorization": "Bearer t"},
                json={"text": "hi"},
            )
            t.join()
            self.assertEqual(resp.status_code, 409)
            self.assertEqual(resp.json()["code"], "CONFLICT")


if __name__ == "__main__":
    unittest.main()
