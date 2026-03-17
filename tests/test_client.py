from __future__ import annotations

import unittest

from whesper.client import _extract_stream_text, _iter_sse_events


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


if __name__ == "__main__":
    unittest.main()
