from __future__ import annotations

import socket
import unittest

from whesper.client import (
    OpenAICompatibleClient,
    ProviderError,
    _build_openai_payload,
    _effective_temperature,
    _extract_reasoning_content,
    _extract_stream_text,
    _extract_tool_calls,
    _extract_tool_calls_from_text,
    _iter_sse_events,
)
from whesper.config import ModelConfig, ProviderConfig
from whesper.provider_profile import (
    GLM_DEFAULT_PROFILE,
    OLLAMA_QWEN_PROFILE,
    OPENAI_DEFAULT_PROFILE,
    SILICONFLOW_QWEN_PROFILE,
)


class ClientTests(unittest.TestCase):
    def test_effective_temperature_uses_fixed_values_for_kimi_k25(self) -> None:
        provider = ProviderConfig(
            name="kimi",
            kind="openai_compatible",
            base_url="https://api.moonshot.cn/v1",
        )
        model = ModelConfig(
            name="kimi-k2.5",
            provider="kimi",
            model="kimi-k2.5",
            temperature=0.2,
        )

        self.assertEqual(
            _effective_temperature(provider, model, disable_thinking=False),
            1.0,
        )
        self.assertEqual(
            _effective_temperature(provider, model, disable_thinking=True),
            0.6,
        )

    def test_build_openai_payload_disables_thinking_with_fixed_kimi_temperature(self) -> None:
        provider = ProviderConfig(
            name="kimi",
            kind="openai_compatible",
            base_url="https://api.moonshot.cn/v1",
        )
        model = ModelConfig(
            name="kimi-k2.5",
            provider="kimi",
            model="kimi-k2.5",
            temperature=1.0,
        )

        payload = _build_openai_payload(
            provider,
            model,
            [{"role": "user", "content": "查一下最新 AI 新闻"}],
            stream=False,
            disable_thinking=True,
        )

        self.assertIn('"temperature": 0.6', payload.decode("utf-8"))
        self.assertIn('"thinking": {"type": "disabled"}', payload.decode("utf-8"))

    def test_iter_sse_events_groups_data_lines(self) -> None:
        lines = [
            b"data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n",
            b"\n",
            b"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n",
            b"\n",
            b"data: [DONE]\n",
            b"\n",
        ]
        events = list(_iter_sse_events(lines))
        self.assertEqual(
            events,
            [
                "{\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}",
                "{\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}",
                "[DONE]",
            ],
        )

    def test_extract_stream_text_from_delta(self) -> None:
        raw = {"choices": [{"delta": {"content": "hello"}}]}
        self.assertEqual(_extract_stream_text(raw), "hello")

    def test_extract_stream_text_from_message_fallback(self) -> None:
        raw = {"choices": [{"message": {"content": "full reply"}}]}
        self.assertEqual(_extract_stream_text(raw), "full reply")

    def test_extract_tool_calls_from_openai_response(self) -> None:
        provider = ProviderConfig(
            name="remote",
            kind="openai_compatible",
            base_url="https://example.com/v1",
        )
        raw = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": "{\"query\":\"latest release notes\"}",
                                },
                            }
                        ],
                    }
                }
            ]
        }
        tool_calls = _extract_tool_calls(provider, raw)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].tool_call_id, "call_123")
        self.assertEqual(tool_calls[0].name, "web_search")

    def test_extract_tool_calls_from_textual_function_call_block(self) -> None:
        content = """
        我来帮你查一下。

        <function_calls>
        <invoke name="web_search">
        <parameter name="query">今日原油价格走势 最新行情 Brent WTI 2025</parameter>
        </invoke>
        </function_calls>
        """

        tool_calls = _extract_tool_calls_from_text(content)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "web_search")
        self.assertIn("Brent WTI 2025", tool_calls[0].arguments_json)

    def test_extract_tool_calls_from_text_ignores_tool_arg_pairs_for_default_profile(self) -> None:
        content = """
        Here is a literal example:
        <tool>web_search</tool>
        <arg>{"query":"latest news"}</arg>
        """

        tool_calls = _extract_tool_calls_from_text(
            content,
            profile=OPENAI_DEFAULT_PROFILE,
        )

        self.assertEqual(tool_calls, ())

    def test_extract_tool_calls_from_text_supports_tool_arg_pairs_for_siliconflow_profile(self) -> None:
        content = """
        <tool>web_search</tool>
        <arg>{"query":"latest news"}</arg>
        """

        tool_calls = _extract_tool_calls_from_text(
            content,
            profile=SILICONFLOW_QWEN_PROFILE,
        )

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "web_search")
        self.assertIn("latest news", tool_calls[0].arguments_json)

    def test_extract_tool_calls_from_text_supports_action_json_for_ollama_qwen_profile(
        self,
    ) -> None:
        content = """
        好的，让我帮你搜一下“春潮黑客松”的相关信息！

        {"action": "search", "query": "春潮黑客松 2026"}
        """

        tool_calls = _extract_tool_calls_from_text(
            content,
            profile=OLLAMA_QWEN_PROFILE,
        )

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "web_search")
        self.assertIn("春潮黑客松 2026", tool_calls[0].arguments_json)

    def test_extract_tool_calls_from_text_supports_box_tool_call_for_glm_profile(
        self,
    ) -> None:
        content = """
        我需要先搜索一下“春潮黑客松”的最新信息。
        <tool_call>web_search(query="春潮黑客松")<|end_of_box|>
        <tool_call>web_search(query="春潮黑客松 2025")<|end_of_box|>
        """

        tool_calls = _extract_tool_calls_from_text(
            content,
            profile=GLM_DEFAULT_PROFILE,
        )

        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0].name, "web_search")
        self.assertIn("春潮黑客松", tool_calls[0].arguments_json)
        self.assertIn("春潮黑客松 2025", tool_calls[1].arguments_json)

    def test_extract_tool_calls_from_text_skips_fenced_code_blocks(self) -> None:
        content = """
        这里是调用示例（仅作说明，不是真的请求）：
        ```
        <function_calls>
        <invoke name="web_search">
        <parameter name="query">demo</parameter>
        </invoke>
        </function_calls>
        ```
        """

        tool_calls = _extract_tool_calls_from_text(
            content,
            profile=OPENAI_DEFAULT_PROFILE,
        )

        self.assertEqual(tool_calls, ())

    def test_extract_tool_calls_from_text_skips_inline_code_spans(self) -> None:
        content = (
            "你可以写 `<tool>web_search</tool><arg>{\"query\":\"demo\"}</arg>` "
            "来表达调用意图。"
        )

        tool_calls = _extract_tool_calls_from_text(
            content,
            profile=SILICONFLOW_QWEN_PROFILE,
        )

        self.assertEqual(tool_calls, ())

    def test_extract_tool_calls_from_text_filters_unregistered_tool_names(self) -> None:
        content = """
        <function_calls>
        <invoke name="fabricated_tool">
        <parameter name="query">demo</parameter>
        </invoke>
        </function_calls>
        """

        tool_calls = _extract_tool_calls_from_text(
            content,
            profile=OPENAI_DEFAULT_PROFILE,
            allowed_tool_names={"web_search"},
        )

        self.assertEqual(tool_calls, ())

    def test_extract_tool_calls_from_text_keeps_registered_tool_names(self) -> None:
        content = """
        <function_calls>
        <invoke name="web_search">
        <parameter name="query">demo</parameter>
        </invoke>
        </function_calls>
        """

        tool_calls = _extract_tool_calls_from_text(
            content,
            profile=OPENAI_DEFAULT_PROFILE,
            allowed_tool_names={"web_search"},
        )

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "web_search")

    def test_create_chat_completion_uses_profile_specific_text_tool_patterns(self) -> None:
        client = OpenAICompatibleClient()
        provider = ProviderConfig(
            name="remote",
            kind="openai_compatible",
            base_url="https://example.com/v1",
        )
        model = ModelConfig(
            name="chat",
            provider="remote",
            model="gpt-like",
        )

        raw = {
            "choices": [
                {
                    "message": {
                        "content": "<tool>web_search</tool><arg>{\"query\":\"latest news\"}</arg>",
                    }
                }
            ]
        }

        import json
        from unittest import mock

        class DummyResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(raw).encode("utf-8")

        with mock.patch("whesper.client.request.urlopen", return_value=DummyResponse()):
            result = client.create_chat_completion(
                provider,
                model,
                [{"role": "user", "content": "hello"}],
            )

        self.assertEqual(result.tool_calls, ())
        self.assertIn("<tool>web_search</tool>", result.content)

    def test_create_chat_completion_skips_text_extraction_when_tools_not_requested(
        self,
    ) -> None:
        client = OpenAICompatibleClient()
        provider = ProviderConfig(
            name="siliconflow",
            kind="openai_compatible",
            base_url="https://api.siliconflow.cn/v1",
        )
        model = ModelConfig(
            name="qwen",
            provider="siliconflow",
            model="Qwen/Qwen3-8B",
        )

        raw = {
            "choices": [
                {
                    "message": {
                        "content": "<tool>web_search</tool><arg>{\"query\":\"x\"}</arg>",
                    }
                }
            ]
        }

        import json
        from unittest import mock

        class DummyResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(raw).encode("utf-8")

        with mock.patch("whesper.client.request.urlopen", return_value=DummyResponse()):
            result = client.create_chat_completion(
                provider,
                model,
                [{"role": "user", "content": "hello"}],
            )

        self.assertEqual(result.tool_calls, ())
        self.assertIn("<tool>web_search</tool>", result.content)

    def test_create_chat_completion_drops_text_tool_calls_with_unknown_names(
        self,
    ) -> None:
        client = OpenAICompatibleClient()
        provider = ProviderConfig(
            name="siliconflow",
            kind="openai_compatible",
            base_url="https://api.siliconflow.cn/v1",
        )
        model = ModelConfig(
            name="qwen",
            provider="siliconflow",
            model="Qwen/Qwen3-8B",
        )

        raw = {
            "choices": [
                {
                    "message": {
                        "content": "<tool>fabricated</tool><arg>{\"query\":\"x\"}</arg>",
                    }
                }
            ]
        }

        import json
        from unittest import mock

        tools = [
            {
                "type": "function",
                "function": {"name": "web_search", "parameters": {"type": "object"}},
            }
        ]

        class DummyResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(raw).encode("utf-8")

        with mock.patch("whesper.client.request.urlopen", return_value=DummyResponse()):
            result = client.create_chat_completion(
                provider,
                model,
                [{"role": "user", "content": "hello"}],
                tools=tools,
            )

        self.assertEqual(result.tool_calls, ())
        self.assertIn("<tool>fabricated</tool>", result.content)

    def test_create_chat_completion_extracts_action_json_for_ollama_qwen_when_tools_requested(
        self,
    ) -> None:
        client = OpenAICompatibleClient()
        provider = ProviderConfig(
            name="ollama_local",
            kind="ollama_native",
            base_url="http://localhost:11434",
        )
        model = ModelConfig(
            name="qwen3.5:27b",
            provider="ollama",
            model="qwen3.5:27b",
        )

        raw = {
            "message": {
                "content": (
                    "好的，让我帮你搜一下“春潮黑客松”的相关信息！\n\n"
                    '{"action": "search", "query": "春潮黑客松 2026"}'
                ),
            }
        }

        import json
        from unittest import mock

        tools = [
            {
                "type": "function",
                "function": {"name": "web_search", "parameters": {"type": "object"}},
            }
        ]

        class DummyResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(raw).encode("utf-8")

        with mock.patch("whesper.client.request.urlopen", return_value=DummyResponse()):
            result = client.create_chat_completion(
                provider,
                model,
                [{"role": "user", "content": "hello"}],
                tools=tools,
            )

        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "web_search")
        self.assertEqual(result.content, "")

    def test_create_chat_completion_extracts_box_tool_calls_for_glm_when_tools_requested(
        self,
    ) -> None:
        client = OpenAICompatibleClient()
        provider = ProviderConfig(
            name="siliconflow",
            kind="openai_compatible",
            base_url="https://api.siliconflow.cn/v1",
        )
        model = ModelConfig(
            name="glm",
            provider="siliconflow",
            model="zai-org/GLM-4.6V",
        )

        raw = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "我需要先搜索一下“春潮黑客松”的最新信息。\n"
                            '<tool_call>web_search(query="春潮黑客松")<|end_of_box|>\n'
                            '<tool_call>web_search(query="春潮黑客松 2025")<|end_of_box|>'
                        )
                    }
                }
            ]
        }

        import json
        from unittest import mock

        tools = [
            {
                "type": "function",
                "function": {"name": "web_search", "parameters": {"type": "object"}},
            }
        ]

        class DummyResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(raw).encode("utf-8")

        with mock.patch("whesper.client.request.urlopen", return_value=DummyResponse()):
            result = client.create_chat_completion(
                provider,
                model,
                [{"role": "user", "content": "hello"}],
                tools=tools,
            )

        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.tool_calls[0].name, "web_search")
        self.assertEqual(result.content, "")

    def test_create_chat_completion_wraps_connection_reset_as_provider_error(self) -> None:
        client = OpenAICompatibleClient()
        provider = ProviderConfig(
            name="ollama_local",
            kind="ollama_native",
            base_url="http://localhost:11434",
        )
        model = ModelConfig(
            name="qwen3.5:27b",
            provider="ollama",
            model="qwen3.5:27b",
        )

        from unittest import mock

        with mock.patch(
            "whesper.client.request.urlopen",
            side_effect=ConnectionResetError(54, "Connection reset by peer"),
        ):
            with self.assertRaisesRegex(
                ProviderError,
                "Provider 'ollama_local' connection failed: \\[Errno 54\\] Connection reset by peer",
            ):
                client.create_chat_completion(
                    provider,
                    model,
                    [{"role": "user", "content": "hello"}],
                )

    def test_create_chat_completion_stream_wraps_socket_timeout_as_provider_error(self) -> None:
        client = OpenAICompatibleClient()
        provider = ProviderConfig(
            name="kimi",
            kind="openai_compatible",
            base_url="https://api.moonshot.cn/v1",
        )
        model = ModelConfig(
            name="kimi-k2.5",
            provider="kimi",
            model="kimi-k2.5",
        )

        from unittest import mock

        with mock.patch(
            "whesper.client.request.urlopen",
            side_effect=socket.timeout("timed out"),
        ):
            with self.assertRaisesRegex(
                ProviderError,
                "Provider 'kimi' connection failed: timed out",
            ):
                list(
                    client.create_chat_completion_stream(
                        provider,
                        model,
                        [{"role": "user", "content": "hello"}],
                    )
                )

    def test_extract_reasoning_content_from_openai_response(self) -> None:
        provider = ProviderConfig(
            name="kimi",
            kind="openai_compatible",
            base_url="https://api.moonshot.cn/v1",
        )
        raw = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning_content": "先搜索，再总结。",
                    }
                }
            ]
        }

        reasoning_content = _extract_reasoning_content(provider, raw)

        self.assertEqual(reasoning_content, "先搜索，再总结。")

    def test_extract_tool_calls_from_text_supports_box_tool_call_for_ollama_qwen_profile(
        self,
    ) -> None:
        content = (
            '我来搜索一下。\n'
            '<tool_call>web_search(query="春潮黑客松")<|end_of_box|>\n'
            '<tool_call>web_search(query="春潮 黑客松 2025")<|end_of_box|>'
        )

        tool_calls = _extract_tool_calls_from_text(
            content,
            profile=OLLAMA_QWEN_PROFILE,
        )

        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0].name, "web_search")
        self.assertIn("春潮黑客松", tool_calls[0].arguments_json)

    def test_extract_tool_calls_from_text_supports_glm_function_for_glm_profile(
        self,
    ) -> None:
        content = '✿FUNCTION✿{"name": "web_search", "arguments": {"query": "今日金价"}}✿RESULT✿'

        tool_calls = _extract_tool_calls_from_text(
            content,
            profile=GLM_DEFAULT_PROFILE,
        )

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "web_search")
        self.assertIn("今日金价", tool_calls[0].arguments_json)

    def test_extract_tool_calls_from_text_supports_plugin_for_qwen_profile(
        self,
    ) -> None:
        from whesper.provider_profile import QWEN_DEFAULT_PROFILE

        content = '<|plugin|>{"name": "web_search", "parameters": {"query": "天气预报"}}<|endoftext|>'

        tool_calls = _extract_tool_calls_from_text(
            content,
            profile=QWEN_DEFAULT_PROFILE,
        )

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "web_search")
        self.assertIn("天气预报", tool_calls[0].arguments_json)


class SanitizeToolCallArtifactsTests(unittest.TestCase):

    def test_strips_deepseek_tool_arg_pair(self) -> None:
        from whesper.tool_protocol import sanitize_tool_call_artifacts

        dirty = '<tool>web_search</tool>\n<arg>{"query": "最新新闻"}</arg>'
        clean = sanitize_tool_call_artifacts(dirty)
        self.assertEqual(clean, "")

    def test_strips_box_tool_call(self) -> None:
        from whesper.tool_protocol import sanitize_tool_call_artifacts

        dirty = (
            "我来搜索一下。\n"
            '<tool_call>web_search(query="test")<|end_of_box|>\n'
            '<tool_call>web_search(query="test2")<|end_of_box|>'
        )
        clean = sanitize_tool_call_artifacts(dirty)
        self.assertNotIn("tool_call", clean)
        self.assertNotIn("end_of_box", clean)
        self.assertIn("我来搜索一下", clean)

    def test_strips_glm_function_marker(self) -> None:
        from whesper.tool_protocol import sanitize_tool_call_artifacts

        dirty = '✿FUNCTION✿{"name": "web_search", "arguments": {"query": "test"}}✿RESULT✿'
        clean = sanitize_tool_call_artifacts(dirty)
        self.assertEqual(clean, "")

    def test_strips_qwen_plugin_format(self) -> None:
        from whesper.tool_protocol import sanitize_tool_call_artifacts

        dirty = '<|plugin|>{"name": "web_search", "parameters": {"query": "test"}}<|endoftext|>'
        clean = sanitize_tool_call_artifacts(dirty)
        self.assertEqual(clean, "")

    def test_strips_function_calls_xml(self) -> None:
        from whesper.tool_protocol import sanitize_tool_call_artifacts

        dirty = (
            "好的。\n"
            '<function_calls><invoke name="web_search">'
            '<parameter name="query">demo</parameter>'
            "</invoke></function_calls>"
        )
        clean = sanitize_tool_call_artifacts(dirty)
        self.assertNotIn("function_calls", clean)
        self.assertIn("好的", clean)

    def test_strips_minimax_wrapper(self) -> None:
        from whesper.tool_protocol import sanitize_tool_call_artifacts

        dirty = (
            "<minimax:tool_call>"
            '<invoke name="web_search">'
            '<parameter name="query">金价</parameter>'
            "</invoke>"
            "</minimax:tool_call>"
        )
        clean = sanitize_tool_call_artifacts(dirty)
        self.assertEqual(clean, "")

    def test_preserves_code_fenced_content(self) -> None:
        from whesper.tool_protocol import sanitize_tool_call_artifacts

        content = (
            "这是一个示例：\n"
            "```xml\n"
            '<function_calls><invoke name="web_search">'
            '<parameter name="query">demo</parameter>'
            "</invoke></function_calls>\n"
            "```\n"
            "以上是调用格式说明。"
        )
        clean = sanitize_tool_call_artifacts(content)
        self.assertIn("function_calls", clean)
        self.assertIn("这是一个示例", clean)
        self.assertIn("以上是调用格式说明", clean)

    def test_strips_multiple_patterns_in_one_message(self) -> None:
        from whesper.tool_protocol import sanitize_tool_call_artifacts

        dirty = (
            "我需要查一下。\n"
            '<tool>web_search</tool>\n<arg>{"query": "A"}</arg>\n'
            '<tool_call>web_search(query="B")<|end_of_box|>'
        )
        clean = sanitize_tool_call_artifacts(dirty)
        self.assertNotIn("<tool>", clean)
        self.assertNotIn("<tool_call>", clean)
        self.assertIn("我需要查一下", clean)

    def test_collapses_excess_blank_lines(self) -> None:
        from whesper.tool_protocol import sanitize_tool_call_artifacts

        dirty = (
            "前文。\n\n\n"
            '<tool>web_search</tool>\n<arg>{"query":"x"}</arg>\n\n\n'
            "后文。"
        )
        clean = sanitize_tool_call_artifacts(dirty)
        self.assertNotIn("\n\n\n", clean)
        self.assertIn("前文", clean)
        self.assertIn("后文", clean)


if __name__ == "__main__":
    unittest.main()
