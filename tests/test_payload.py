from __future__ import annotations

import json
import unittest

from whesper.client import _request_payload
from whesper.config import ModelConfig, ProviderConfig


class PayloadTests(unittest.TestCase):
    def test_request_payload_includes_think_when_set(self) -> None:
        provider = ProviderConfig(
            name="local",
            kind="openai_compatible",
            base_url="http://localhost:11434/v1",
        )
        model = ModelConfig(
            name="chat",
            provider="local",
            model="qwen3.5:27b",
            think=False,
        )
        payload = json.loads(
            _request_payload(
                provider,
                model,
                [{"role": "user", "content": "hello"}],
                stream=True,
            ).decode("utf-8")
        )
        self.assertIn("think", payload)
        self.assertFalse(payload["think"])

    def test_ollama_payload_uses_options_block(self) -> None:
        provider = ProviderConfig(
            name="local",
            kind="ollama_native",
            base_url="http://localhost:11434",
        )
        model = ModelConfig(
            name="chat",
            provider="local",
            model="qwen3.5:27b",
            temperature=0.6,
            max_tokens=300,
            think=False,
        )
        payload = json.loads(
            _request_payload(
                provider,
                model,
                [{"role": "user", "content": "hello"}],
                stream=True,
            ).decode("utf-8")
        )
        self.assertEqual(payload["options"]["temperature"], 0.6)
        self.assertEqual(payload["options"]["num_predict"], 300)
        self.assertFalse(payload["think"])

    def test_request_payload_includes_tools_when_provided(self) -> None:
        provider = ProviderConfig(
            name="local",
            kind="openai_compatible",
            base_url="http://localhost:11434/v1",
        )
        model = ModelConfig(
            name="chat",
            provider="local",
            model="qwen3.5:27b",
        )
        payload = json.loads(
            _request_payload(
                provider,
                model,
                [{"role": "user", "content": "hello"}],
                stream=False,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "description": "Search the web",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            ).decode("utf-8")
        )
        self.assertIn("tools", payload)
        self.assertEqual(payload["tools"][0]["function"]["name"], "web_search")


if __name__ == "__main__":
    unittest.main()
