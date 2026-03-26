from __future__ import annotations

import json
from pathlib import Path
import tempfile
import textwrap
import unittest

from whesper.client import ToolCall
from whesper.config import ShellSandboxSettings, load_config
from whesper.shell_sandbox import ShellSandboxError, ShellSandboxExecutor
from whesper.tools import ToolRegistry


class ShellSandboxExecutorTests(unittest.TestCase):
    def test_execute_pwd_within_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            executor = ShellSandboxExecutor(
                ShellSandboxSettings(enabled=True),
                workspace_root=Path(workspace),
            )

            result = executor.execute("pwd")

            self.assertTrue(result.ok)
            self.assertEqual(Path(result.stdout.strip()).resolve(), Path(workspace).resolve())

    def test_rejects_shell_operators(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            executor = ShellSandboxExecutor(
                ShellSandboxSettings(enabled=True),
                workspace_root=Path(workspace),
            )

            with self.assertRaisesRegex(ShellSandboxError, "does not allow pipes"):
                executor.execute("pwd | cat")

    def test_rejects_paths_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            executor = ShellSandboxExecutor(
                ShellSandboxSettings(enabled=True),
                workspace_root=Path(workspace),
            )

            with self.assertRaisesRegex(ShellSandboxError, "outside the allowed shell sandbox roots"):
                executor.execute("cat ../secret.txt")

    def test_truncates_large_output(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            file_path = Path(workspace) / "large.txt"
            file_path.write_text("abcdefghij" * 20, encoding="utf-8")
            executor = ShellSandboxExecutor(
                ShellSandboxSettings(enabled=True, max_output_chars=30),
                workspace_root=Path(workspace),
            )

            result = executor.execute("cat large.txt")

            self.assertTrue(result.truncated)
            self.assertIn("[truncated]", result.stdout)


class ShellToolRegistryTests(unittest.TestCase):
    def test_registry_adds_shell_tool_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            config_path = Path(workspace) / "whesper.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [scheduler]
                    chat_model = "local"

                    [providers.main]
                    base_url = "http://localhost:11434/v1"

                    [models.local]
                    provider = "main"
                    model = "qwen"

                    [shell_sandbox]
                    enabled = true
                    """
                ).strip(),
                encoding="utf-8",
            )
            (Path(workspace) / "README.md").write_text("hello sandbox\n", encoding="utf-8")

            config = load_config(config_path)
            registry = ToolRegistry.default(config)

            self.assertIn("run_shell_command", {spec.name for spec in registry.specs})

            result = registry.execute(
                ToolCall(
                    tool_call_id="call_shell_1",
                    name="run_shell_command",
                    arguments_json=json.dumps({"command": "cat README.md"}),
                )
            )

            payload = json.loads(result.content)
            self.assertTrue(payload["ok"])
            self.assertIn("hello sandbox", payload["stdout"])

    def test_shell_tool_returns_policy_error_payload(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            config_path = Path(workspace) / "whesper.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [scheduler]
                    chat_model = "local"

                    [providers.main]
                    base_url = "http://localhost:11434/v1"

                    [models.local]
                    provider = "main"
                    model = "qwen"

                    [shell_sandbox]
                    enabled = true
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)
            registry = ToolRegistry.default(config)

            result = registry.execute(
                ToolCall(
                    tool_call_id="call_shell_2",
                    name="run_shell_command",
                    arguments_json=json.dumps({"command": "pwd | cat"}),
                )
            )

            payload = json.loads(result.content)
            self.assertFalse(payload["ok"])
            self.assertIn("does not allow pipes", payload["error"])


if __name__ == "__main__":
    unittest.main()
