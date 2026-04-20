from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from unittest import mock

from whesper.config import CUP_DEFAULT_BASE_URL, ConfigError, load_config


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
        self.assertEqual(config.app.context_strategy, "memory_first")
        self.assertEqual(config.app.recent_history_limit, 4)
        self.assertEqual(config.app.tool_exposure_strategy, "matched_only")

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

    def test_loads_optional_search_api_config(self) -> None:
        content = textwrap.dedent(
            """
            [scheduler]
            chat_model = "local"

            [providers.main]
            base_url = "http://localhost:11434/v1"

            [models.local]
            provider = "main"
            model = "qwen"

            [live_context.search_api]
            provider = "serpapi"
            api_key_env = "MY_SERPAPI_KEY"
            engine = "google"
            timeout_seconds = 12
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
            fh.write(content)
            temp_path = fh.name

        config = load_config(temp_path)
        self.assertEqual(config.live_context.search_api.provider, "serpapi")
        self.assertEqual(config.live_context.search_api.api_key_env, "MY_SERPAPI_KEY")
        self.assertEqual(config.live_context.search_api.engine, "google")
        self.assertEqual(config.live_context.search_api.timeout_seconds, 12)

    def test_loads_optional_brave_search_api_config_without_key(self) -> None:
        content = textwrap.dedent(
            """
            [scheduler]
            chat_model = "local"

            [providers.main]
            base_url = "http://localhost:11434/v1"

            [models.local]
            provider = "main"
            model = "qwen"

            [live_context.search_api]
            provider = "brave"
            timeout_seconds = 9
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
            fh.write(content)
            temp_path = fh.name

        config = load_config(temp_path)
        self.assertEqual(config.live_context.search_api.provider, "brave")
        self.assertEqual(config.live_context.search_api.api_key_env, "WHESPER_BRAVE_API_KEY")
        self.assertEqual(config.live_context.search_api.timeout_seconds, 9)

    def test_loads_optional_shell_sandbox_config(self) -> None:
        content = textwrap.dedent(
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
            timeout_seconds = 7
            max_output_chars = 2048
            allowed_roots = [".", "src"]
            allowed_command_prefixes = [["pwd"], ["git", "status"], ["rg"]]
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
            fh.write(content)
            temp_path = fh.name

        config = load_config(temp_path)
        self.assertTrue(config.shell_sandbox.enabled)
        self.assertEqual(config.shell_sandbox.timeout_seconds, 7)
        self.assertEqual(config.shell_sandbox.max_output_chars, 2048)
        self.assertEqual(config.shell_sandbox.allowed_roots, (".", "src"))
        self.assertEqual(
            config.shell_sandbox.allowed_command_prefixes,
            (("pwd",), ("git", "status"), ("rg",)),
        )

    def test_loads_optional_memory_first_context_config(self) -> None:
        content = textwrap.dedent(
            """
            [app]
            context_strategy = "memory_first"
            recent_history_limit = 2
            tool_exposure_strategy = "model_first"

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
        self.assertEqual(config.app.context_strategy, "memory_first")
        self.assertEqual(config.app.recent_history_limit, 2)
        self.assertEqual(config.app.tool_exposure_strategy, "model_first")

    def test_rejects_invalid_tool_exposure_strategy(self) -> None:
        content = textwrap.dedent(
            """
            [app]
            tool_exposure_strategy = "semantic_magic"

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

        with self.assertRaisesRegex(
            ConfigError,
            "app.tool_exposure_strategy must be either 'matched_only' or 'model_first'",
        ):
            load_config(temp_path)

    def test_loads_optional_cup_hardware_config(self) -> None:
        content = textwrap.dedent(
            """
            [scheduler]
            chat_model = "local"

            [providers.main]
            base_url = "http://localhost:11434/v1"

            [models.local]
            provider = "main"
            model = "qwen"

            [hardware.cup]
            enabled = true
            base_url = "%s"
            api_token = "secret-token"
            timeout_seconds = 8
            """
        ).strip() % CUP_DEFAULT_BASE_URL

        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
            fh.write(content)
            temp_path = fh.name

        config = load_config(temp_path)
        self.assertTrue(config.hardware.cup.enabled)
        self.assertEqual(config.hardware.cup.base_url, CUP_DEFAULT_BASE_URL)
        self.assertEqual(config.hardware.cup.timeout_seconds, 8)
        self.assertEqual(config.hardware.cup.resolved_api_token(), "secret-token")

    def test_loads_cup_hardware_base_url_from_config_variable(self) -> None:
        content = textwrap.dedent(
            """
            [variables]
            cup_api_base_url = "http://127.0.0.1:3901"

            [scheduler]
            chat_model = "local"

            [providers.main]
            base_url = "http://localhost:11434/v1"

            [models.local]
            provider = "main"
            model = "qwen"

            [hardware.cup]
            enabled = true
            base_url = "${cup_api_base_url}"
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
            fh.write(content)
            temp_path = fh.name

        config = load_config(temp_path)
        self.assertEqual(config.hardware.cup.base_url, "http://127.0.0.1:3901")

    def test_fails_with_invalid_provider_extra_headers_type(self) -> None:
        content = textwrap.dedent(
            """
            [scheduler]
            chat_model = "local"

            [providers.main]
            base_url = "http://localhost:11434/v1"
            extra_headers = []

            [models.local]
            provider = "main"
            model = "qwen"
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
            fh.write(content)
            temp_path = fh.name

        with self.assertRaisesRegex(ConfigError, "providers.main.extra_headers"):
            load_config(temp_path)

    def test_fails_with_invalid_live_context_extra_headers_type(self) -> None:
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
            extra_headers = []
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
            fh.write(content)
            temp_path = fh.name

        with self.assertRaisesRegex(
            ConfigError,
            "live_context.custom_api.device.extra_headers",
        ):
            load_config(temp_path)

    def test_loads_provider_base_url_from_environment_variable_default(self) -> None:
        content = textwrap.dedent(
            """
            [scheduler]
            chat_model = "local"

            [providers.ollama_local]
            kind = "ollama_native"
            base_url = "${WHESPER_OLLAMA_BASE_URL:-http://localhost:11434}"

            [models.local]
            provider = "ollama_local"
            model = "qwen"
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
            fh.write(content)
            temp_path = fh.name

        with mock.patch.dict(os.environ, {}, clear=True):
            config = load_config(temp_path)
        self.assertEqual(
            config.get_provider("ollama_local").base_url,
            "http://localhost:11434",
        )

    def test_server_settings_default_when_section_absent(self) -> None:
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
        self.assertFalse(config.server.enabled)
        self.assertEqual(config.server.host, "127.0.0.1")
        self.assertEqual(config.server.port, 8765)
        self.assertTrue(config.server.require_auth)
        self.assertEqual(config.server.cors_origins, ())
        self.assertEqual(config.server.request_timeout_seconds, 300)
        self.assertEqual(config.server.max_concurrent_turns, 4)
        self.assertEqual(config.server.api_token_env, "WHESPER_SERVER_TOKEN")
        self.assertIsNone(config.server.api_token)

    def test_loads_optional_server_settings(self) -> None:
        content = textwrap.dedent(
            """
            [scheduler]
            chat_model = "local"

            [providers.main]
            base_url = "http://localhost:11434/v1"

            [models.local]
            provider = "main"
            model = "qwen"

            [server]
            enabled = true
            host = "0.0.0.0"
            port = 9000
            require_auth = false
            api_token = "inline-secret"
            cors_origins = ["https://app.example.com", "https://other.example.com"]
            request_timeout_seconds = 120
            max_concurrent_turns = 8
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
            fh.write(content)
            temp_path = fh.name

        config = load_config(temp_path)
        self.assertTrue(config.server.enabled)
        self.assertEqual(config.server.host, "0.0.0.0")
        self.assertEqual(config.server.port, 9000)
        self.assertFalse(config.server.require_auth)
        self.assertEqual(
            config.server.cors_origins,
            ("https://app.example.com", "https://other.example.com"),
        )
        self.assertEqual(config.server.request_timeout_seconds, 120)
        self.assertEqual(config.server.max_concurrent_turns, 8)
        self.assertEqual(config.server.resolved_api_token(), "inline-secret")

    def test_server_api_token_resolves_from_environment(self) -> None:
        content = textwrap.dedent(
            """
            [scheduler]
            chat_model = "local"

            [providers.main]
            base_url = "http://localhost:11434/v1"

            [models.local]
            provider = "main"
            model = "qwen"

            [server]
            enabled = true
            api_token_env = "TEST_WHESPER_SERVER_TOKEN"
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
            fh.write(content)
            temp_path = fh.name

        with mock.patch.dict(
            os.environ,
            {"TEST_WHESPER_SERVER_TOKEN": "env-secret"},
            clear=False,
        ):
            config = load_config(temp_path)
            self.assertEqual(config.server.resolved_api_token(), "env-secret")

    def test_rejects_invalid_server_port(self) -> None:
        content = textwrap.dedent(
            """
            [scheduler]
            chat_model = "local"

            [providers.main]
            base_url = "http://localhost:11434/v1"

            [models.local]
            provider = "main"
            model = "qwen"

            [server]
            port = 99999
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
            fh.write(content)
            temp_path = fh.name

        with self.assertRaises(ConfigError):
            load_config(temp_path)

    def test_rejects_invalid_server_cors_origins(self) -> None:
        content = textwrap.dedent(
            """
            [scheduler]
            chat_model = "local"

            [providers.main]
            base_url = "http://localhost:11434/v1"

            [models.local]
            provider = "main"
            model = "qwen"

            [server]
            cors_origins = "https://app.example.com"
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
            fh.write(content)
            temp_path = fh.name

        with self.assertRaises(ConfigError):
            load_config(temp_path)

    def test_loads_provider_base_url_from_environment_variable_override(self) -> None:
        content = textwrap.dedent(
            """
            [scheduler]
            chat_model = "local"

            [providers.ollama_local]
            kind = "ollama_native"
            base_url = "${WHESPER_OLLAMA_BASE_URL:-http://localhost:11434}"

            [models.local]
            provider = "ollama_local"
            model = "qwen"
            """
        ).strip()

        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
            fh.write(content)
            temp_path = fh.name

        with mock.patch.dict(
            os.environ,
            {"WHESPER_OLLAMA_BASE_URL": "http://example.invalid:41434"},
            clear=False,
        ):
            config = load_config(temp_path)
        self.assertEqual(
            config.get_provider("ollama_local").base_url,
            "http://example.invalid:41434",
        )


if __name__ == "__main__":
    unittest.main()
