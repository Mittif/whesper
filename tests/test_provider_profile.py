from __future__ import annotations

import json
import unittest

from whesper.config import ModelConfig, ProviderConfig
from whesper.history_normalizer import HistoryNormalizer
from whesper.provider_profile import (
    KIMI_DEFAULT_PROFILE,
    KIMI_K25_PROFILE,
    OLLAMA_QWEN_PROFILE,
    OPENAI_DEFAULT_PROFILE,
    SILICONFLOW_DEEPSEEK_PROFILE,
    SILICONFLOW_QWEN_PROFILE,
    resolve_profile,
)
from whesper.session import ChatMessage


def _kimi_provider() -> ProviderConfig:
    return ProviderConfig(
        name="kimi",
        kind="openai_compatible",
        base_url="https://api.moonshot.cn/v1",
    )


def _siliconflow_provider() -> ProviderConfig:
    return ProviderConfig(
        name="siliconflow",
        kind="openai_compatible",
        base_url="https://api.siliconflow.cn/v1",
    )


def _ollama_provider() -> ProviderConfig:
    return ProviderConfig(
        name="ollama_local",
        kind="ollama_native",
        base_url="http://localhost:11434",
    )


def _model(provider: str, model: str, **kwargs: object) -> ModelConfig:
    return ModelConfig(name=model, provider=provider, model=model, **kwargs)


class ResolveProfileTests(unittest.TestCase):
    def test_resolves_kimi_k25_by_provider_and_family(self) -> None:
        profile = resolve_profile(_kimi_provider(), _model("kimi", "kimi-k2.5"))
        self.assertIs(profile, KIMI_K25_PROFILE)

    def test_resolves_kimi_default_for_other_kimi_models(self) -> None:
        profile = resolve_profile(_kimi_provider(), _model("kimi", "moonshot-v1-8k"))
        self.assertIs(profile, KIMI_DEFAULT_PROFILE)

    def test_resolves_siliconflow_qwen_family(self) -> None:
        profile = resolve_profile(
            _siliconflow_provider(),
            _model("siliconflow", "Qwen/Qwen3-8B"),
        )
        self.assertIs(profile, SILICONFLOW_QWEN_PROFILE)

    def test_resolves_siliconflow_deepseek_family(self) -> None:
        profile = resolve_profile(
            _siliconflow_provider(),
            _model("siliconflow", "deepseek-ai/DeepSeek-V3.2"),
        )
        self.assertIs(profile, SILICONFLOW_DEEPSEEK_PROFILE)

    def test_resolves_ollama_qwen_family(self) -> None:
        profile = resolve_profile(
            _ollama_provider(),
            _model("ollama", "qwen3.5:27b"),
        )
        self.assertIs(profile, OLLAMA_QWEN_PROFILE)

    def test_explicit_model_profile_override_wins(self) -> None:
        model = _model("kimi", "arbitrary")
        model.profile_override = "openai:default"
        profile = resolve_profile(_kimi_provider(), model)
        self.assertIs(profile, OPENAI_DEFAULT_PROFILE)

    def test_provider_override_selects_alternative_provider_key(self) -> None:
        provider = ProviderConfig(
            name="custom-kimi-gateway",
            kind="openai_compatible",
            base_url="https://example.com/v1",
            profile_override="kimi",
        )
        profile = resolve_profile(provider, _model("custom", "kimi-k2.5"))
        self.assertIs(profile, KIMI_K25_PROFILE)


class HistoryNormalizerTests(unittest.TestCase):
    def test_assistant_content_null_when_tool_calls_present(self) -> None:
        normalizer = HistoryNormalizer(KIMI_DEFAULT_PROFILE)
        message = ChatMessage(
            role="assistant",
            content="",
            created_at="2026-04-14T00:00:00+00:00",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query":"btc"}',
                    },
                }
            ],
            source_profile="kimi:default",
        )
        payloads = normalizer.normalize([message])
        self.assertIsNone(payloads[0]["content"])
        self.assertEqual(
            payloads[0]["tool_calls"][0]["function"]["arguments"],
            '{"query":"btc"}',
        )

    def test_tool_arguments_converted_to_object_for_object_mode_profile(self) -> None:
        normalizer = HistoryNormalizer(OLLAMA_QWEN_PROFILE)
        message = ChatMessage(
            role="assistant",
            content="",
            created_at="2026-04-14T00:00:00+00:00",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query":"btc"}',
                    },
                }
            ],
            source_profile="kimi:default",
        )
        payloads = normalizer.normalize([message])
        arguments = payloads[0]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(arguments, {"query": "btc"})

    def test_reasoning_content_dropped_across_profiles(self) -> None:
        normalizer = HistoryNormalizer(OPENAI_DEFAULT_PROFILE)
        message = ChatMessage(
            role="assistant",
            content="hello",
            created_at="2026-04-14T00:00:00+00:00",
            reasoning_content="internal kimi thought",
            source_profile="kimi:k2-5",
        )
        payloads = normalizer.normalize([message])
        self.assertNotIn("reasoning_content", payloads[0])

    def test_reasoning_content_preserved_within_same_profile(self) -> None:
        normalizer = HistoryNormalizer(KIMI_DEFAULT_PROFILE)
        message = ChatMessage(
            role="assistant",
            content="hello",
            created_at="2026-04-14T00:00:00+00:00",
            reasoning_content="internal kimi thought",
            source_profile="kimi:default",
        )
        payloads = normalizer.normalize([message])
        self.assertEqual(payloads[0]["reasoning_content"], "internal kimi thought")

    def test_reasoning_placeholder_injected_when_required(self) -> None:
        normalizer = HistoryNormalizer(KIMI_DEFAULT_PROFILE)
        message = ChatMessage(
            role="assistant",
            content="",
            created_at="2026-04-14T00:00:00+00:00",
            reasoning_content=None,
            source_profile="openai:default",
        )
        payloads = normalizer.normalize(
            [message],
            inject_reasoning_placeholder=True,
        )
        self.assertEqual(
            payloads[0]["reasoning_content"],
            KIMI_DEFAULT_PROFILE.reasoning_content_empty_placeholder,
        )

    def test_builtin_web_search_degraded_to_text_for_non_kimi_target(self) -> None:
        normalizer = HistoryNormalizer(OPENAI_DEFAULT_PROFILE)
        assistant_message = ChatMessage(
            role="assistant",
            content="",
            created_at="2026-04-14T00:00:00+00:00",
            tool_calls=[
                {
                    "id": "builtin-1",
                    "type": "builtin_function",
                    "function": {
                        "name": "$web_search",
                        "arguments": json.dumps({"query": "今日金价"}, ensure_ascii=False),
                    },
                }
            ],
            source_profile="kimi:k2-5",
        )
        tool_message = ChatMessage(
            role="tool",
            content='{"results":"gold price info"}',
            created_at="2026-04-14T00:00:01+00:00",
            name="$web_search",
            tool_call_id="builtin-1",
            source_profile="kimi:k2-5",
        )

        payloads = normalizer.normalize([assistant_message, tool_message])

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["role"], "assistant")
        self.assertNotIn("tool_calls", payloads[0])
        self.assertIn("previous $web_search", payloads[0]["content"])
        self.assertIn("今日金价", payloads[0]["content"])
        self.assertIn("gold price info", payloads[0]["content"])

    def test_builtin_web_search_preserved_when_target_supports_it(self) -> None:
        normalizer = HistoryNormalizer(KIMI_K25_PROFILE)
        assistant_message = ChatMessage(
            role="assistant",
            content="",
            created_at="2026-04-14T00:00:00+00:00",
            tool_calls=[
                {
                    "id": "builtin-1",
                    "type": "builtin_function",
                    "function": {
                        "name": "$web_search",
                        "arguments": json.dumps({"query": "今日金价"}, ensure_ascii=False),
                    },
                }
            ],
            source_profile="kimi:k2-5",
        )

        payloads = normalizer.normalize([assistant_message])

        self.assertIn("tool_calls", payloads[0])
        self.assertEqual(
            payloads[0]["tool_calls"][0]["function"]["name"],
            "$web_search",
        )


if __name__ == "__main__":
    unittest.main()
