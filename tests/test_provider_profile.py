from __future__ import annotations

import json
import unittest

from whesper.config import ModelConfig, ProviderConfig
from whesper.history_normalizer import HistoryNormalizer
from whesper.tool_protocol import sanitize_tool_call_artifacts
from whesper.provider_profile import (
    CLAUDE_DEFAULT_PROFILE,
    DEEPSEEK_DEFAULT_PROFILE,
    DEEPSEEK_R1_PROFILE,
    GEMINI_DEFAULT_PROFILE,
    GLM_DEFAULT_PROFILE,
    GROK_DEFAULT_PROFILE,
    KIMI_DEFAULT_PROFILE,
    KIMI_K25_PROFILE,
    LLAMA_DEFAULT_PROFILE,
    OLLAMA_DEFAULT_PROFILE,
    OLLAMA_QWEN_PROFILE,
    OPENAI_DEFAULT_PROFILE,
    QWEN_DEFAULT_PROFILE,
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

    # ------------------------------------------------------------------
    # New provider detection tests
    # ------------------------------------------------------------------

    def test_resolves_grok_by_xai_url(self) -> None:
        provider = ProviderConfig(
            name="xai",
            kind="openai_compatible",
            base_url="https://api.x.ai/v1",
        )
        profile = resolve_profile(provider, _model("xai", "grok-3"))
        self.assertIs(profile, GROK_DEFAULT_PROFILE)

    def test_resolves_gemini_by_googleapis_url(self) -> None:
        provider = ProviderConfig(
            name="google",
            kind="openai_compatible",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        )
        profile = resolve_profile(provider, _model("google", "gemini-2.5-pro"))
        self.assertIs(profile, GEMINI_DEFAULT_PROFILE)

    def test_resolves_glm_by_bigmodel_url(self) -> None:
        provider = ProviderConfig(
            name="zhipu",
            kind="openai_compatible",
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )
        profile = resolve_profile(provider, _model("zhipu", "glm-4"))
        self.assertIs(profile, GLM_DEFAULT_PROFILE)

    def test_resolves_deepseek_default_for_v3(self) -> None:
        provider = ProviderConfig(
            name="deepseek",
            kind="openai_compatible",
            base_url="https://api.deepseek.com/v1",
        )
        profile = resolve_profile(provider, _model("deepseek", "deepseek-chat"))
        self.assertIs(profile, DEEPSEEK_DEFAULT_PROFILE)

    def test_resolves_deepseek_r1_for_reasoner_model(self) -> None:
        provider = ProviderConfig(
            name="deepseek",
            kind="openai_compatible",
            base_url="https://api.deepseek.com/v1",
        )
        profile = resolve_profile(provider, _model("deepseek", "deepseek-reasoner"))
        self.assertIs(profile, DEEPSEEK_R1_PROFILE)

    def test_resolves_deepseek_r1_for_r1_model_name(self) -> None:
        provider = ProviderConfig(
            name="deepseek",
            kind="openai_compatible",
            base_url="https://api.deepseek.com/v1",
        )
        profile = resolve_profile(provider, _model("deepseek", "deepseek-r1"))
        self.assertIs(profile, DEEPSEEK_R1_PROFILE)

    def test_resolves_claude_by_anthropic_url(self) -> None:
        provider = ProviderConfig(
            name="anthropic",
            kind="openai_compatible",
            base_url="https://api.anthropic.com/v1",
        )
        profile = resolve_profile(provider, _model("anthropic", "claude-opus-4-6"))
        self.assertIs(profile, CLAUDE_DEFAULT_PROFILE)

    def test_resolves_qwen_by_dashscope_url(self) -> None:
        provider = ProviderConfig(
            name="dashscope",
            kind="openai_compatible",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        profile = resolve_profile(provider, _model("dashscope", "qwen-max"))
        self.assertIs(profile, QWEN_DEFAULT_PROFILE)

    def test_resolves_llama_by_meta_ai_url(self) -> None:
        provider = ProviderConfig(
            name="meta",
            kind="openai_compatible",
            base_url="https://api.llama.meta.ai/v1",
        )
        profile = resolve_profile(provider, _model("meta", "llama3.3-70b"))
        self.assertIs(profile, LLAMA_DEFAULT_PROFILE)


class NewProfileBehaviorTests(unittest.TestCase):
    """Verify that new profiles have the expected normalization behavior."""

    def test_gemini_tool_call_assistant_sends_empty_content_not_null(self) -> None:
        normalizer = HistoryNormalizer(GEMINI_DEFAULT_PROFILE)
        message = ChatMessage(
            role="assistant",
            content="",
            created_at="2026-04-14T00:00:00+00:00",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"test"}'},
                }
            ],
            source_profile="gemini:default",
        )
        payloads = normalizer.normalize([message])
        self.assertEqual(payloads[0]["content"], "")
        self.assertIsNotNone(payloads[0]["content"])

    def test_glm_tool_call_assistant_sends_empty_content_not_null(self) -> None:
        normalizer = HistoryNormalizer(GLM_DEFAULT_PROFILE)
        message = ChatMessage(
            role="assistant",
            content="",
            created_at="2026-04-14T00:00:00+00:00",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"test"}'},
                }
            ],
            source_profile="glm:default",
        )
        payloads = normalizer.normalize([message])
        self.assertEqual(payloads[0]["content"], "")
        self.assertIsNotNone(payloads[0]["content"])

    def test_llama_tool_call_assistant_sends_empty_content_not_null(self) -> None:
        normalizer = HistoryNormalizer(LLAMA_DEFAULT_PROFILE)
        message = ChatMessage(
            role="assistant",
            content="",
            created_at="2026-04-14T00:00:00+00:00",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"test"}'},
                }
            ],
            source_profile="llama:default",
        )
        payloads = normalizer.normalize([message])
        self.assertEqual(payloads[0]["content"], "")
        self.assertIsNotNone(payloads[0]["content"])

    def test_deepseek_reasoning_not_replayed_in_history(self) -> None:
        # DeepSeek R1 surfaces reasoning_content in responses but must NOT
        # receive it back in subsequent request history (causes 400 errors).
        normalizer = HistoryNormalizer(DEEPSEEK_R1_PROFILE)
        message = ChatMessage(
            role="assistant",
            content="final answer",
            created_at="2026-04-14T00:00:00+00:00",
            reasoning_content="chain of thought",
            source_profile="deepseek:r1",
        )
        payloads = normalizer.normalize([message])
        self.assertNotIn("reasoning_content", payloads[0])

    def test_grok_reasoning_not_replayed_in_history(self) -> None:
        normalizer = HistoryNormalizer(GROK_DEFAULT_PROFILE)
        message = ChatMessage(
            role="assistant",
            content="answer",
            created_at="2026-04-14T00:00:00+00:00",
            reasoning_content="thinking text",
            source_profile="grok:default",
        )
        payloads = normalizer.normalize([message])
        self.assertNotIn("reasoning_content", payloads[0])

    def test_qwen_tool_arguments_are_string_by_default(self) -> None:
        normalizer = HistoryNormalizer(QWEN_DEFAULT_PROFILE)
        message = ChatMessage(
            role="assistant",
            content="",
            created_at="2026-04-14T00:00:00+00:00",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"test"}'},
                }
            ],
            source_profile="qwen:default",
        )
        payloads = normalizer.normalize([message])
        # QWEN_DEFAULT_PROFILE uses tool_arguments_mode="string" (the default)
        self.assertIsInstance(
            payloads[0]["tool_calls"][0]["function"]["arguments"], str
        )


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
        # Kimi requires reasoning_content in input, so it must be preserved.
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

    def test_reasoning_content_not_replayed_to_provider_that_does_not_require_it(
        self,
    ) -> None:
        # SiliconFlow may extract reasoning_content from responses but does NOT
        # accept the field in request history. Replaying it would cause HTTP 400.
        normalizer = HistoryNormalizer(SILICONFLOW_QWEN_PROFILE)
        message = ChatMessage(
            role="assistant",
            content="some answer",
            created_at="2026-04-14T00:00:00+00:00",
            reasoning_content="siliconflow internal thinking",
            source_profile="siliconflow:qwen",
        )
        payloads = normalizer.normalize([message])
        self.assertNotIn("reasoning_content", payloads[0])

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


class MiniMaxToolCallTests(unittest.TestCase):
    """MiniMax models emit <minimax:tool_call> XML instead of <function_calls>."""

    _MINIMAX_BLOCK = (
        "<minimax:tool_call>\n"
        '<invoke name="web_search">\n'
        "<parameter name=\"query\">今日油价</parameter>\n"
        "</invoke>\n"
        "</minimax:tool_call>"
    )

    def test_tool_protocol_extracts_minimax_format_as_function_calls(self) -> None:
        from whesper.tool_protocol import DefaultToolProtocolAdapter

        adapter = DefaultToolProtocolAdapter()
        tool_calls = adapter.extract_tool_calls_from_text(self._MINIMAX_BLOCK)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "web_search")
        import json
        args = json.loads(tool_calls[0].arguments_json)
        self.assertEqual(args.get("query"), "今日油价")

    def test_strip_tool_call_xml_removes_minimax_block(self) -> None:
        dirty = f"我来帮你查一下。\n{self._MINIMAX_BLOCK}"
        clean = sanitize_tool_call_artifacts(dirty)
        self.assertNotIn("minimax:tool_call", clean)
        self.assertNotIn("invoke", clean)
        self.assertIn("我来帮你查一下", clean)

    def test_strip_tool_call_xml_removes_function_calls_block(self) -> None:
        dirty = (
            "好的。\n"
            "<function_calls>\n"
            '<invoke name="web_search">'
            '<parameter name="query">test</parameter>'
            "</invoke>\n"
            "</function_calls>"
        )
        clean = sanitize_tool_call_artifacts(dirty)
        self.assertNotIn("function_calls", clean)
        self.assertNotIn("invoke", clean)

    def test_history_normalizer_strips_minimax_xml_from_assistant_content(
        self,
    ) -> None:
        normalizer = HistoryNormalizer(OPENAI_DEFAULT_PROFILE)
        dirty_message = ChatMessage(
            role="assistant",
            content=self._MINIMAX_BLOCK,
            created_at="2026-04-14T00:00:00+00:00",
            source_profile="openai:default",
        )
        payloads = normalizer.normalize([dirty_message])
        self.assertEqual(payloads[0]["content"], "")
        self.assertNotIn("tool_calls", payloads[0])

    def test_history_normalizer_strips_inline_minimax_xml_preserving_surrounding_text(
        self,
    ) -> None:
        normalizer = HistoryNormalizer(OPENAI_DEFAULT_PROFILE)
        dirty_message = ChatMessage(
            role="assistant",
            content=f"好的，我来搜索。\n{self._MINIMAX_BLOCK}",
            created_at="2026-04-14T00:00:00+00:00",
            source_profile="openai:default",
        )
        payloads = normalizer.normalize([dirty_message])
        self.assertIn("好的，我来搜索", payloads[0]["content"])
        self.assertNotIn("minimax:tool_call", payloads[0]["content"])


class UniversalToolCallParserTests(unittest.TestCase):
    """Ensure every known text tool-call dialect is extractable + sanitizable."""

    def _adapter(self):
        from whesper.tool_protocol import DefaultToolProtocolAdapter

        return DefaultToolProtocolAdapter()

    # ---- Hermes / NousResearch / OpenChat ------------------------------

    def test_hermes_tool_call_extracts_with_nested_arguments(self) -> None:
        adapter = self._adapter()
        content = (
            'I will search.\n'
            '<tool_call>{"name": "web_search", "arguments": {"query": "btc price"}}</tool_call>'
        )
        calls = adapter.extract_tool_calls_from_text(
            content,
            profile=LLAMA_DEFAULT_PROFILE,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "web_search")
        args = json.loads(calls[0].arguments_json)
        self.assertEqual(args, {"query": "btc price"})

    def test_hermes_tool_call_multiple_blocks(self) -> None:
        adapter = self._adapter()
        content = (
            '<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>\n'
            '<tool_call>{"name":"b","arguments":{"y":2}}</tool_call>'
        )
        calls = adapter.extract_tool_calls_from_text(
            content,
            profile=LLAMA_DEFAULT_PROFILE,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].name, "a")
        self.assertEqual(calls[1].name, "b")

    def test_hermes_artifact_sanitized_from_history(self) -> None:
        normalizer = HistoryNormalizer(OPENAI_DEFAULT_PROFILE)
        dirty = ChatMessage(
            role="assistant",
            content='好的。\n<tool_call>{"name":"web_search","arguments":{"query":"a"}}</tool_call>',
            created_at="2026-04-15T00:00:00+00:00",
            source_profile="openai:default",
        )
        payloads = normalizer.normalize([dirty])
        self.assertIn("好的", payloads[0]["content"])
        self.assertNotIn("tool_call", payloads[0]["content"])

    # ---- Llama 3.x <|python_tag|> --------------------------------------

    def test_llama_python_tag_extracts(self) -> None:
        adapter = self._adapter()
        content = (
            '<|python_tag|>{"name": "web_search", "parameters": {"query": "今日金价"}}<|eom_id|>'
        )
        calls = adapter.extract_tool_calls_from_text(
            content,
            profile=LLAMA_DEFAULT_PROFILE,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "web_search")
        args = json.loads(calls[0].arguments_json)
        self.assertEqual(args, {"query": "今日金价"})

    def test_llama_python_tag_tolerates_missing_closer(self) -> None:
        adapter = self._adapter()
        content = '<|python_tag|>{"name": "calc", "arguments": {"a": 1}}'
        calls = adapter.extract_tool_calls_from_text(
            content,
            profile=LLAMA_DEFAULT_PROFILE,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "calc")

    def test_llama_python_tag_sanitized_from_history(self) -> None:
        normalizer = HistoryNormalizer(OPENAI_DEFAULT_PROFILE)
        dirty = ChatMessage(
            role="assistant",
            content='好的。\n<|python_tag|>{"name":"a","arguments":{}}<|eom_id|>',
            created_at="2026-04-15T00:00:00+00:00",
            source_profile="openai:default",
        )
        payloads = normalizer.normalize([dirty])
        self.assertNotIn("python_tag", payloads[0]["content"])
        self.assertNotIn("eom_id", payloads[0]["content"])

    # ---- Mistral [TOOL_CALLS][…] ---------------------------------------

    def test_mistral_tool_calls_extracts_list(self) -> None:
        adapter = self._adapter()
        content = (
            '[TOOL_CALLS][{"name": "web_search", "arguments": {"query": "x"}}]'
        )
        calls = adapter.extract_tool_calls_from_text(
            content,
            profile=OLLAMA_DEFAULT_PROFILE,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "web_search")

    def test_mistral_tool_calls_sanitized_from_history(self) -> None:
        normalizer = HistoryNormalizer(OPENAI_DEFAULT_PROFILE)
        dirty = ChatMessage(
            role="assistant",
            content='[TOOL_CALLS][{"name":"a","arguments":{}}]',
            created_at="2026-04-15T00:00:00+00:00",
            source_profile="openai:default",
        )
        payloads = normalizer.normalize([dirty])
        self.assertNotIn("TOOL_CALLS", payloads[0]["content"])

    # ---- Bare JSON fallback -------------------------------------------

    def test_bare_json_tool_call_requires_arguments_field(self) -> None:
        adapter = self._adapter()
        # A JSON with only "name" and no args-like field should NOT be treated
        # as a tool call — this protects against false positives from user
        # chatter about names or records.
        content = 'Here is a user record: {"name": "John"}'
        calls = adapter.extract_tool_calls_from_text(
            content,
            profile=OLLAMA_DEFAULT_PROFILE,
        )
        self.assertEqual(len(calls), 0)

    def test_bare_json_tool_call_accepts_name_plus_arguments(self) -> None:
        adapter = self._adapter()
        content = '{"name": "web_search", "arguments": {"query": "x"}}'
        calls = adapter.extract_tool_calls_from_text(
            content,
            profile=OLLAMA_DEFAULT_PROFILE,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "web_search")

    def test_bare_json_tool_call_accepts_name_plus_parameters(self) -> None:
        adapter = self._adapter()
        content = '{"name": "web_search", "parameters": {"query": "x"}}'
        calls = adapter.extract_tool_calls_from_text(
            content,
            profile=OLLAMA_DEFAULT_PROFILE,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "web_search")

    def test_bare_json_tool_call_accepts_tool_field(self) -> None:
        adapter = self._adapter()
        content = '{"tool": "web_search", "arguments": {"query": "x"}}'
        calls = adapter.extract_tool_calls_from_text(
            content,
            profile=OLLAMA_DEFAULT_PROFILE,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "web_search")

    # ---- Cross-format mix / ordering ----------------------------------

    def test_specific_patterns_run_before_json_fallback(self) -> None:
        """If a well-formed Hermes block is present, the Hermes extractor
        must win over the bare-JSON fallback (tested by ordering the profile
        patterns accordingly).  Asserting the tool name lets us detect any
        regression in the priority order.
        """
        adapter = self._adapter()
        content = (
            '<tool_call>{"name": "hermes_call", "arguments": {"q": 1}}</tool_call>\n'
            '{"name": "bare_call", "arguments": {"q": 2}}'
        )
        calls = adapter.extract_tool_calls_from_text(
            content,
            profile=LLAMA_DEFAULT_PROFILE,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "hermes_call")

    def test_all_artifact_patterns_stripped_in_combination(self) -> None:
        normalizer = HistoryNormalizer(OPENAI_DEFAULT_PROFILE)
        dirty = ChatMessage(
            role="assistant",
            content=(
                "prose before\n"
                '<tool_call>{"name":"a","arguments":{}}</tool_call>\n'
                "middle text\n"
                '<|python_tag|>{"name":"b","arguments":{}}<|eom_id|>\n'
                "tail text\n"
                '[TOOL_CALLS][{"name":"c","arguments":{}}]'
            ),
            created_at="2026-04-15T00:00:00+00:00",
            source_profile="openai:default",
        )
        payloads = normalizer.normalize([dirty])
        rendered = payloads[0]["content"]
        self.assertIn("prose before", rendered)
        self.assertIn("middle text", rendered)
        self.assertIn("tail text", rendered)
        self.assertNotIn("tool_call", rendered)
        self.assertNotIn("python_tag", rendered)
        self.assertNotIn("TOOL_CALLS", rendered)

    def test_code_fenced_tool_call_examples_are_preserved(self) -> None:
        """Models often quote tool-call syntax in fenced code blocks while
        explaining them.  Those quoted examples must NOT be stripped, or the
        assistant's educational content gets mutilated.
        """
        normalizer = HistoryNormalizer(OPENAI_DEFAULT_PROFILE)
        message = ChatMessage(
            role="assistant",
            content=(
                "Here's how you'd invoke a tool:\n"
                "```\n"
                '<tool_call>{"name":"web_search","arguments":{"q":"x"}}</tool_call>\n'
                "```\n"
                "That's the syntax."
            ),
            created_at="2026-04-15T00:00:00+00:00",
            source_profile="openai:default",
        )
        payloads = normalizer.normalize([message])
        rendered = payloads[0]["content"]
        self.assertIn("```", rendered)
        self.assertIn("tool_call", rendered)  # preserved inside fence


if __name__ == "__main__":
    unittest.main()
