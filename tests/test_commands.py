from __future__ import annotations

import tempfile
import textwrap
import unittest

from whesper.commands import command_completions, parse_command
from whesper.config import load_config
from whesper.session import SessionStore


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


class CommandTests(unittest.TestCase):
    def test_parse_command_with_arg(self) -> None:
        command = parse_command("/model local_chat")
        self.assertIsNotNone(command)
        assert command is not None
        self.assertEqual(command.name, "/model")
        self.assertEqual(command.arg, "local_chat")

    def test_parse_non_command(self) -> None:
        self.assertIsNone(parse_command("hello there"))

    def test_command_completions_include_models_and_sessions(self) -> None:
        config = make_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(temp_dir)
            store.load("demo")
            completions = command_completions(config, store)
            self.assertIn("/clear", completions)
            self.assertIn("/model", completions)
            self.assertIn("chat", completions["/model"])
            self.assertIn("/use", completions)
            self.assertIn("chat", completions["/use"])
            self.assertIn("/status", completions)
            self.assertIn("/info", completions)
            self.assertIn("/retry", completions)
            self.assertIn("/copy-last", completions)
            self.assertIn("/new", completions)
            self.assertIn("demo", completions["/new"])
            self.assertIn("/rename", completions)
            self.assertIn("demo", completions["/rename"])
            self.assertIn("/delete-session", completions)
            self.assertIn("demo", completions["/delete-session"])


if __name__ == "__main__":
    unittest.main()
