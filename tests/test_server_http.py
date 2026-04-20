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


SERVER_EXTRAS_AVAILABLE = TestClient is not None


def _write_config(storage_dir: Path, *, server_block: str = "") -> object:
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
        """
    ).strip()
    if server_block:
        body = body + "\n\n" + server_block.strip()
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write(body)
        temp_path = fh.name
    return load_config(temp_path)


@unittest.skipUnless(
    SERVER_EXTRAS_AVAILABLE, "fastapi extra not installed; skip HTTP tests"
)
class ServerHttpTests(unittest.TestCase):
    def _make_client(
        self, *, tmpdir: str, token: str = "test-token", require_auth: bool = True
    ):
        from whesper.api.application import AgentApplication
        from whesper.server.http import create_app

        storage_dir = Path(tmpdir)
        server_block = textwrap.dedent(
            f"""
            [server]
            enabled = true
            require_auth = {str(require_auth).lower()}
            api_token = "{token}"
            """
        )
        config = _write_config(storage_dir, server_block=server_block)
        app = create_app(AgentApplication(config))
        return TestClient(app)

    def test_health_is_public(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._make_client(tmpdir=tmpdir)
            resp = client.get("/v1/health")
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertTrue(body["ok"])
            self.assertIn("version", body)

    def test_capabilities_requires_auth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._make_client(tmpdir=tmpdir)
            resp = client.get("/v1/capabilities")
            self.assertEqual(resp.status_code, 401)
            self.assertEqual(resp.json()["code"], "UNAUTHORIZED")

    def test_capabilities_returns_schema_with_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._make_client(tmpdir=tmpdir, token="good")
            resp = client.get(
                "/v1/capabilities",
                headers={"Authorization": "Bearer good"},
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertIn("features", body)
            self.assertIn("route_modes", body)

    def test_wrong_token_returns_401(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._make_client(tmpdir=tmpdir, token="good")
            resp = client.get(
                "/v1/capabilities",
                headers={"Authorization": "Bearer wrong"},
            )
            self.assertEqual(resp.status_code, 401)

    def test_auth_disabled_skips_token_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._make_client(tmpdir=tmpdir, require_auth=False)
            resp = client.get("/v1/capabilities")
            self.assertEqual(resp.status_code, 200)

    def test_sessions_create_get_delete_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._make_client(tmpdir=tmpdir, token="t")
            auth = {"Authorization": "Bearer t"}

            resp = client.get("/v1/sessions", headers=auth)
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["items"], [])

            resp = client.post(
                "/v1/sessions",
                headers=auth,
                json={"session_id": "work"},
            )
            self.assertEqual(resp.status_code, 201)
            summary = resp.json()
            self.assertEqual(summary["session_id"], "work")
            self.assertEqual(summary["message_count"], 0)
            self.assertFalse(summary["has_pending_ask_user"])

            resp = client.post(
                "/v1/sessions",
                headers=auth,
                json={"session_id": "work"},
            )
            self.assertEqual(resp.status_code, 409)
            self.assertEqual(resp.json()["code"], "CONFLICT")

            resp = client.get("/v1/sessions/work", headers=auth)
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["session_id"], "work")

            resp = client.get("/v1/sessions/missing", headers=auth)
            self.assertEqual(resp.status_code, 404)
            self.assertEqual(resp.json()["code"], "NOT_FOUND")

            resp = client.delete("/v1/sessions/work", headers=auth)
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.json()["ok"])

            resp = client.delete("/v1/sessions/work", headers=auth)
            self.assertEqual(resp.status_code, 404)

    def test_create_session_rejects_blank_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._make_client(tmpdir=tmpdir, token="t")
            resp = client.post(
                "/v1/sessions",
                headers={"Authorization": "Bearer t"},
                json={"session_id": ""},
            )
            self.assertEqual(resp.status_code, 422)  # pydantic validation


if __name__ == "__main__":
    unittest.main()
