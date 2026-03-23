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
        self.assertEqual(config.live_context.custom_api, {})

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

    def test_loads_optional_live_context_endpoint_config(self) -> None:
        content = textwrap.dedent(
            """
            [scheduler]
            chat_model = "local"

            [providers.main]
            base_url = "http://localhost:11434/v1"

            [models.local]
            provider = "main"
            model = "qwen"

            [live_context.custom_api.device]
            url_template = "https://example.com/status"
            trigger_keywords = ["设备状态", "device status"]
            query_param = "q"
            instruction = "Use this internal status payload when relevant."
            api_key_env = "DEVICE_STATUS_KEY"
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
            fh.write(content)
            temp_path = fh.name

        config = load_config(temp_path)
        endpoint = config.live_context.custom_api["device"]
        self.assertEqual(endpoint.url_template, "https://example.com/status")
        self.assertEqual(endpoint.trigger_keywords, ("设备状态", "device status"))
        self.assertEqual(endpoint.query_param, "q")
        self.assertEqual(endpoint.api_key_env, "DEVICE_STATUS_KEY")


if __name__ == "__main__":
    unittest.main()
