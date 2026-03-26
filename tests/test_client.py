from __future__ import annotations

import unittest

from whesper.client import (
    _extract_reasoning_content,
    _extract_stream_text,
    _extract_tool_calls,
    _extract_tool_calls_from_text,
    _iter_sse_events,
)
from whesper.config import ProviderConfig


class ClientTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
