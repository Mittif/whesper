from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from whesper.api.application import (
    AgentApplication,
    SessionAlreadyExists,
    SessionNotFound,
)
from whesper.config import load_config


def _make_config(storage_dir: Path) -> object:
    toml_body = textwrap.dedent(
        f"""
        [app]
        storage_dir = "{storage_dir}"
        default_session = "main"

        [scheduler]
        chat_model = "local"

        [providers.main]
        base_url = "http://localhost:11434/v1"

        [models.local]
        provider = "main"
        model = "qwen"
        """
    ).strip()
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write(toml_body)
        temp_path = fh.name
    return load_config(temp_path)


class AgentApplicationTests(unittest.TestCase):
    def test_capabilities_lists_advertised_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            app = AgentApplication(config)
            capabilities = app.capabilities()

            self.assertEqual(capabilities["version"], app.version)
            self.assertIn("tools", capabilities["features"])
            self.assertEqual(
                sorted(capabilities["route_modes"]),
                ["auto", "chat", "reasoning", "search"],
            )
            self.assertIn("duckduckgo", capabilities["websearch_providers"])

    def test_create_and_list_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            app = AgentApplication(config)

            created = app.create_session("work")
            self.assertEqual(created.session_id, "work")

            summaries = {summary.session_id for summary in app.list_sessions()}
            self.assertIn("work", summaries)

    def test_create_existing_session_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            app = AgentApplication(config)

            app.create_session("main")
            with self.assertRaises(SessionAlreadyExists):
                app.create_session("main")

    def test_get_missing_session_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            app = AgentApplication(config)

            with self.assertRaises(SessionNotFound):
                app.get_session("does-not-exist")

    def test_delete_session_returns_false_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            app = AgentApplication(config)

            self.assertFalse(app.delete_session("absent"))

            app.create_session("main")
            self.assertTrue(app.delete_session("main"))
            self.assertEqual(app.list_sessions(), [])

    def test_rejects_blank_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(Path(tmpdir))
            app = AgentApplication(config)

            with self.assertRaises(ValueError):
                app.create_session("   ")


if __name__ == "__main__":
    unittest.main()
