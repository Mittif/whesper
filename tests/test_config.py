from __future__ import annotations

import tempfile
import textwrap
import unittest

from whesper.config import ConfigError, load_config


class ConfigTests(unittest.TestCase):
    def test_loads_basic_config(self) -> None:
        content = textwrap.dedent(
            """
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
            fh.write(content)
            temp_path = fh.name

        config = load_config(temp_path)
        self.assertEqual(config.scheduler.chat_model, "local")
        self.assertEqual(config.get_provider("main").base_url, "http://localhost:11434/v1")
        self.assertEqual(config.get_model("local").model, "qwen")

    def test_fails_with_helpful_message_when_no_models_are_defined(self) -> None:
        content = textwrap.dedent(
            """
            [scheduler]
            chat_model = "local"

            [providers.main]
            base_url = "http://localhost:11434/v1"

            [models]
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
            fh.write(content)
            temp_path = fh.name

        with self.assertRaisesRegex(ConfigError, "No models are configured"):
            load_config(temp_path)

    def test_fails_with_available_models_when_scheduler_points_to_missing_alias(self) -> None:
        content = textwrap.dedent(
            """
            [scheduler]
            chat_model = "missing_model"

            [providers.main]
            base_url = "http://localhost:11434/v1"

            [models.local]
            provider = "main"
            model = "qwen"
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
            fh.write(content)
            temp_path = fh.name

        with self.assertRaisesRegex(ConfigError, "available models: local"):
            load_config(temp_path)


if __name__ == "__main__":
    unittest.main()
