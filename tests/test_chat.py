from __future__ import annotations

import tempfile
import textwrap
import unittest

from whesper.chat import ChatService, GenerationInterrupted
from whesper.client import CompletionResult, ToolCall
from whesper.config import load_config
from whesper.live_data import LiveContextService
from whesper.memory import MemoryService, MemoryStore
from whesper.session import ChatMessage, ConversationSession
from whesper.tools import (
    ToolRegistry,
    ToolSpec,
    _matches_cup_control_intent,
)
from whesper.agent_types import ToolExecutionMeta


class FakeStreamingClient:
    def __init__(self) -> None:
        self.completion_calls = 0

    def create_chat_completion_stream(self, provider, model, messages):
        yield "你好"
        yield "，世界"

    def create_chat_completion(self, provider, model, messages, *, tools=None):
        self.completion_calls += 1
        return CompletionResult(content="", raw_response={})


class InterruptingStreamingClient:
    def create_chat_completion_stream(self, provider, model, messages):
        yield "partial"
        raise KeyboardInterrupt()

    def create_chat_completion(self, provider, model, messages, *, tools=None):
        return CompletionResult(content="", raw_response={})


class CapturingClient:
    def __init__(self) -> None:
        self.last_messages = None

    def create_chat_completion_stream(self, provider, model, messages):
        self.last_messages = messages
        yield "memory-aware reply"

    def create_chat_completion(self, provider, model, messages, *, tools=None):
        self.last_messages = messages
        return CompletionResult(content="", raw_response={})


class StaticLiveContextService(LiveContextService):
    def __init__(self, content: str | None) -> None:
        self.content = content

    def build_prompt_context(self, user_text: str, *, route_mode: str = "chat") -> str | None:
        return self.content


class ToolCallingClient:
    def __init__(self) -> None:
        self.completion_requests: list[dict[str, object]] = []
        self.stream_requests: list[dict[str, object]] = []

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.completion_requests.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        if len(self.completion_requests) == 1:
            return CompletionResult(
                content="",
                raw_response={},
                tool_calls=(
                    ToolCall(
                        tool_call_id="call_1",
                        name="web_search",
                        arguments_json='{"query":"latest release notes"}',
                    ),
                ),
            )
        assistant_tool_message = messages[-2]
        tool_message = messages[-1]
        if assistant_tool_message["role"] != "assistant":
            raise AssertionError("assistant tool call message was not preserved")
        if tool_message["role"] != "tool":
            raise AssertionError("tool message missing")
        return CompletionResult(content="我查到了最新结果", raw_response={})

    def create_chat_completion_stream(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.stream_requests.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        yield "我查到了"
        yield "最新结果"


class ToolChoiceCapturingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.calls.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        return CompletionResult(content="direct answer", raw_response={})

    def create_chat_completion_stream(self, provider, model, messages, *, tools=None, tool_choice=None):
        raise AssertionError("streaming should not be called in this test")


class ToolUnsupportedClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.calls.append({"messages": messages, "tools": tools})
        if tools is not None:
            raise RuntimeError("unexpected path")
        return CompletionResult(content="fallback answer", raw_response={})

    def create_chat_completion_stream(self, provider, model, messages):
        raise AssertionError("streaming should not be called in this test")


class ProviderErrorToolClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        from whesper.client import ProviderError

        self.calls.append({"messages": messages, "tools": tools})
        if tools is not None:
            raise ProviderError("Provider rejected tools payload")
        return CompletionResult(content="fallback answer", raw_response={})

    def create_chat_completion_stream(self, provider, model, messages):
        raise AssertionError("streaming should not be called in this test")


class UnreachableProviderToolClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        from whesper.client import ProviderError

        self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        raise ProviderError(
            "Provider 'kimi' is unreachable: [SSL: UNEXPECTED_EOF_WHILE_READING] "
            "EOF occurred in violation of protocol (_ssl.c:1081)"
        )

    def create_chat_completion_stream(self, provider, model, messages):
        raise AssertionError("streaming should not be called in this test")


class ToolChoiceUnsupportedClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create_chat_completion(self, provider, model, messages, *, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        return CompletionResult(content="fallback without tool_choice", raw_response={})

    def create_chat_completion_stream(self, provider, model, messages):
        raise AssertionError("streaming should not be called in this test")


class DirectAnswerToolClient:
    def __init__(self) -> None:
        self.completion_requests: list[dict[str, object]] = []
        self.stream_calls = 0

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.completion_requests.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        return CompletionResult(content="直接答复", raw_response={})

    def create_chat_completion_stream(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.stream_calls += 1
        yield "unexpected stream"


class AskUserToolClient:
    def __init__(self) -> None:
        self.completion_requests: list[dict[str, object]] = []

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.completion_requests.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        return CompletionResult(
            content=(
                '<ask_user>{"prompt":"你想查哪个城市？","options":'
                '[{"label":"上海","value":"上海"},{"label":"东京","value":"东京"}],'
                '"allow_free_text":true,"field_name":"location"}</ask_user>'
            ),
            raw_response={},
        )

    def create_chat_completion_stream(self, provider, model, messages, *, tools=None, tool_choice=None):
        raise AssertionError("streaming should not be called in this test")


class AskUserFollowupClient:
    def __init__(self) -> None:
        self.completion_requests: list[dict[str, object]] = []

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.completion_requests.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        last_user_message = next(
            message["content"]
            for message in reversed(messages)
            if message["role"] == "user"
        )
        if len(self.completion_requests) == 1:
            return CompletionResult(
                content=(
                    '<ask_user>{"prompt":"你想查哪个城市？","options":'
                    '[{"label":"上海","value":"上海"},{"label":"东京","value":"东京"}],'
                    '"allow_free_text":true,"field_name":"location"}</ask_user>'
                ),
                raw_response={},
            )
        return CompletionResult(content=f"收到地点：{last_user_message}", raw_response={})

    def create_chat_completion_stream(self, provider, model, messages, *, tools=None, tool_choice=None):
        raise AssertionError("streaming should not be called in this test")


class KimiBuiltinWebSearchClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create_chat_completion(
        self,
        provider,
        model,
        messages,
        *,
        tools=None,
        tool_choice=None,
        disable_thinking=False,
    ):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "disable_thinking": disable_thinking,
            }
        )
        if len(self.calls) == 1:
            builtin_tool = next(
                tool
                for tool in tools or ()
                if tool.get("type") == "builtin_function"
            )
            if builtin_tool["function"]["name"] != "$web_search":
                raise AssertionError("missing kimi builtin web search tool declaration")
            return CompletionResult(
                content="",
                raw_response={},
                tool_calls=(
                    ToolCall(
                        tool_call_id="call_web_1",
                        name="$web_search",
                        arguments_json='{"query":"最新 AI 新闻"}',
                        tool_type="builtin_function",
                    ),
                ),
            )

        assistant_tool_message = messages[-2]
        tool_message = messages[-1]
        self.last_followup_messages = messages
        if assistant_tool_message["tool_calls"][0]["type"] != "builtin_function":
            raise AssertionError("builtin function tool type should be preserved")
        if tool_message["role"] != "tool":
            raise AssertionError("tool message missing")
        if tool_message["content"] != '{"query": "最新 AI 新闻"}':
            raise AssertionError("builtin web search tool result should echo arguments")
        return CompletionResult(content="这是联网搜索结果摘要。", raw_response={})

    def create_chat_completion_stream(self, provider, model, messages, *, tools=None, tool_choice=None):
        raise AssertionError("streaming should not be called in this test")


class StrictToolPayloadClient:
    def __init__(self) -> None:
        self.completion_requests: list[dict[str, object]] = []

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.completion_requests.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        if len(self.completion_requests) == 1:
            return CompletionResult(
                content="",
                raw_response={},
                tool_calls=(
                    ToolCall(
                        tool_call_id="call_strict_1",
                        name="web_search",
                        arguments_json='{"query":"crude oil price today"}',
                    ),
                ),
            )
        assistant_tool_message = messages[-2]
        tool_message = messages[-1]
        if assistant_tool_message["role"] != "assistant":
            raise AssertionError("assistant tool call message was not preserved")
        if assistant_tool_message.get("content", "unexpected") is not None:
            raise AssertionError("assistant tool call content should be null")
        if tool_message["role"] != "tool":
            raise AssertionError("tool message missing")
        if tool_message.get("tool_call_id") != "call_strict_1":
            raise AssertionError("tool_call_id was not forwarded")
        if "name" in tool_message:
            raise AssertionError("tool role message should not include name")
        return CompletionResult(content="严格 provider 通过", raw_response={})

    def create_chat_completion_stream(self, provider, model, messages, *, tools=None, tool_choice=None):
        raise AssertionError("streaming should not be called in this test")


class ToolCallingClientWithLeakyContent:
    def __init__(self) -> None:
        self.completion_requests: list[dict[str, object]] = []
        self.stream_requests: list[dict[str, object]] = []

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.completion_requests.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        if len(self.completion_requests) == 1:
            return CompletionResult(
                content=(
                    "我来帮你查一下。\n\n"
                    "<function_calls>\n"
                    "<invoke name=\"web_search\">\n"
                    "<parameter name=\"query\">今日原油价格走势 最新行情 Brent WTI 2025</parameter>\n"
                    "</invoke>\n"
                    "</function_calls>"
                ),
                raw_response={},
                tool_calls=(
                    ToolCall(
                        tool_call_id="call_leaky_1",
                        name="web_search",
                        arguments_json='{"query":"今日原油价格走势 最新行情 Brent WTI 2025"}',
                    ),
                ),
            )
        assistant_tool_message = messages[-2]
        if assistant_tool_message["role"] != "assistant":
            raise AssertionError("assistant tool call message missing")
        if assistant_tool_message.get("content", "unexpected") is not None:
            raise AssertionError("tool call assistant content should not leak to follow-up")
        return CompletionResult(content="最终答复", raw_response={})

    def create_chat_completion_stream(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.stream_requests.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        assistant_tool_message = messages[-2]
        if assistant_tool_message["role"] != "assistant":
            raise AssertionError("assistant tool call message missing in stream request")
        if assistant_tool_message.get("content", "unexpected") is not None:
            raise AssertionError("tool call assistant content should stay hidden in stream request")
        yield "最终"
        yield "答复"


class MultiRoundPlanningClient:
    def __init__(self) -> None:
        self.completion_requests: list[dict[str, object]] = []
        self.stream_calls = 0

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.completion_requests.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        if len(self.completion_requests) == 1:
            return CompletionResult(
                content="我来帮你查一下今天原油市场的最新动态。让我看看有没有更新的实时数据。",
                raw_response={},
            )
        if len(self.completion_requests) == 2:
            if messages[-1]["role"] != "user":
                raise AssertionError("continuation prompt should be appended")
            return CompletionResult(
                content="",
                raw_response={},
                tool_calls=(
                    ToolCall(
                        tool_call_id="call_multi_1",
                        name="web_search",
                        arguments_json='{"query":"今日原油价格走势 最新行情 Brent WTI 2025"}',
                    ),
                ),
            )
        return CompletionResult(content="今天原油价格小幅上涨，主要受供应收紧预期影响。", raw_response={})

    def create_chat_completion_stream(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.stream_calls += 1
        yield "今天原油价格小幅上涨，主要受供应收紧预期影响。"


class OllamaStructuredArgumentsClient:
    def __init__(self) -> None:
        self.completion_requests: list[dict[str, object]] = []

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        from whesper.client import ProviderError

        self.completion_requests.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        if len(self.completion_requests) == 1:
            return CompletionResult(
                content="",
                raw_response={},
                tool_calls=(
                    ToolCall(
                        tool_call_id="call_ollama_1",
                        name="web_search",
                        arguments_json='{"query":"oil price today"}',
                    ),
                ),
            )
        assistant_tool_message = messages[-2]
        arguments = assistant_tool_message["tool_calls"][0]["function"]["arguments"]
        if isinstance(arguments, str):
            raise ProviderError(
                "Provider 'ollama_local' returned HTTP 400: "
                "{\"error\":\"Value looks like object, but can't find closing '}' symbol\"}"
            )
        if not isinstance(arguments, dict):
            raise AssertionError("tool arguments should be normalized to an object")
        return CompletionResult(content="ollama follow-up ok", raw_response={})

    def create_chat_completion_stream(self, provider, model, messages, *, tools=None, tool_choice=None):
        raise AssertionError("streaming should not be called in this test")


class KimiThinkingToolClient:
    def __init__(self) -> None:
        self.completion_requests: list[dict[str, object]] = []

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.completion_requests.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        if len(self.completion_requests) == 1:
            return CompletionResult(
                content="",
                raw_response={},
                tool_calls=(
                    ToolCall(
                        tool_call_id="call_kimi_1",
                        name="web_search",
                        arguments_json='{"query":"latest release notes"}',
                    ),
                ),
                reasoning_content="先搜索最新 release notes，再整理成简短答案。",
            )
        assistant_tool_message = messages[-2]
        if assistant_tool_message["role"] != "assistant":
            raise AssertionError("assistant tool call message missing")
        if "reasoning_content" not in assistant_tool_message:
            raise AssertionError("reasoning_content should be preserved for Kimi thinking mode")
        return CompletionResult(content="Kimi thinking tool follow-up ok", raw_response={})

    def create_chat_completion_stream(self, provider, model, messages, *, tools=None, tool_choice=None):
        raise AssertionError("streaming should not be called in this test")


class KimiToolChoiceDisableThinkingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create_chat_completion(
        self,
        provider,
        model,
        messages,
        *,
        tools=None,
        tool_choice=None,
        disable_thinking=False,
    ):
        from whesper.client import ProviderError

        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "disable_thinking": disable_thinking,
            }
        )
        if len(self.calls) in {1, 3}:
            if tool_choice != "required" or disable_thinking:
                raise AssertionError("expected required tool_choice with thinking enabled first")
            raise ProviderError(
                "Provider 'kimi' returned HTTP 400: "
                "{\"error\":{\"message\":\"tool_choice 'required' is incompatible "
                "with thinking enabled\",\"type\":\"invalid_request_error\"}}"
            )
        if len(self.calls) == 2:
            if tool_choice != "required" or not disable_thinking:
                raise AssertionError("expected retry with disable_thinking while keeping required")
            return CompletionResult(
                content="",
                raw_response={},
                tool_calls=(
                    ToolCall(
                        tool_call_id="call_cup_1",
                        name="control_cup",
                        arguments_json='{"action":"set_led","color":"暖白"}',
                    ),
                ),
            )
        if len(self.calls) == 4:
            if tool_choice != "required" or not disable_thinking:
                raise AssertionError("expected follow-up retry with disable_thinking")
            return CompletionResult(content="已经帮你调成更温馨的颜色。", raw_response={})
        raise AssertionError("unexpected extra completion call")

    def create_chat_completion_stream(self, provider, model, messages, *, tools=None, tool_choice=None):
        raise AssertionError("streaming should not be called in this test")


class KimiLegacyToolCallIdClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        for message in messages:
            if message.get("role") == "tool":
                raise AssertionError("legacy tool trace should not be replayed from transcript history")
        return CompletionResult(content="legacy tool trace omitted from transcript replay", raw_response={})

    def create_chat_completion_stream(self, provider, model, messages, *, tools=None, tool_choice=None):
        raise AssertionError("streaming should not be called in this test")


class KimiOrphanToolMessageClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        for message in messages:
            if message.get("role") == "tool":
                raise AssertionError("orphan tool message should be removed before request")
        return CompletionResult(content="orphan tool message dropped", raw_response={})

    def create_chat_completion_stream(self, provider, model, messages, *, tools=None, tool_choice=None):
        raise AssertionError("streaming should not be called in this test")


class MultiStepToolLoopClient:
    def __init__(self) -> None:
        self.completion_requests: list[dict[str, object]] = []
        self.stream_calls = 0

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.completion_requests.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        if len(self.completion_requests) == 1:
            return CompletionResult(
                content="",
                raw_response={},
                tool_calls=(
                    ToolCall(
                        tool_call_id="call_1",
                        name="web_search",
                        arguments_json='{"query":"oil market latest news"}',
                    ),
                ),
            )
        if len(self.completion_requests) == 2:
            if messages[-1]["role"] != "tool":
                raise AssertionError("first tool result should be present")
            return CompletionResult(
                content="",
                raw_response={},
                tool_calls=(
                    ToolCall(
                        tool_call_id="call_2",
                        name="summarize_findings",
                        arguments_json='{"topic":"oil market"}',
                    ),
                ),
            )
        return CompletionResult(content="综合来看，油价受供应和库存预期共同影响。", raw_response={})

    def create_chat_completion_stream(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.stream_calls += 1
        yield "综合来看，油价受供应和库存预期共同影响。"


class WeatherToolChainClient:
    def __init__(self) -> None:
        self.completion_requests: list[dict[str, object]] = []

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.completion_requests.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        if len(self.completion_requests) == 1:
            return CompletionResult(
                content="",
                raw_response={},
                tool_calls=(
                    ToolCall(
                        tool_call_id="call_ip_1",
                        name="get_public_ip",
                        arguments_json="{}",
                    ),
                ),
            )
        if len(self.completion_requests) == 2:
            return CompletionResult(
                content="",
                raw_response={},
                tool_calls=(
                    ToolCall(
                        tool_call_id="call_geo_1",
                        name="get_ip_location",
                        arguments_json='{"ip":"203.0.113.10"}',
                    ),
                ),
            )
        if len(self.completion_requests) == 3:
            return CompletionResult(
                content="",
                raw_response={},
                tool_calls=(
                    ToolCall(
                        tool_call_id="call_weather_1",
                        name="get_weather_by_location",
                        arguments_json='{"location":"Shanghai, Shanghai, China"}',
                    ),
                ),
            )
        return CompletionResult(content="上海当前多云，今天最高 27.2C。", raw_response={})

    def create_chat_completion_stream(self, provider, model, messages, *, tools=None, tool_choice=None):
        raise AssertionError("streaming should not be called in this test")


def make_config():
    content = textwrap.dedent(
        """
        [scheduler]
        chat_model = "chat"
        search_model = "chat"

        [providers.local]
        base_url = "http://localhost:11434/v1"

        [models.chat]
        provider = "local"
        model = "chat-model"
        """
    ).strip()

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write(content)
        temp_path = fh.name
    return load_config(temp_path)


def make_ollama_native_config():
    content = textwrap.dedent(
        """
        [scheduler]
        chat_model = "chat"
        search_model = "chat"

        [providers.ollama_local]
        kind = "ollama_native"
        base_url = "http://localhost:11434"

        [models.chat]
        provider = "ollama_local"
        model = "qwen"
        """
    ).strip()

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write(content)
        temp_path = fh.name
    return load_config(temp_path)


def make_cup_registry() -> ToolRegistry:
    return ToolRegistry(
        specs=(
            ToolSpec(
                name="control_cup",
                description="Control the CUP hardware",
                parameters_schema={
                    "type": "object",
                    "properties": {"action": {"type": "string"}},
                    "required": ["action"],
                    "additionalProperties": False,
                },
                handler=lambda arguments: {"ok": True},
                should_offer=lambda user_text, route_mode: _matches_cup_control_intent(
                    user_text
                ),
                execution_meta=ToolExecutionMeta(side_effectful=True),
            ),
        )
    )


def make_kimi_thinking_config():
    content = textwrap.dedent(
        """
        [scheduler]
        chat_model = "kimi"
        search_model = "kimi"

        [providers.kimi]
        kind = "openai_compatible"
        base_url = "https://api.moonshot.cn/v1"

        [models.kimi]
        provider = "kimi"
        model = "kimi-k2.5"
        think = true
        """
    ).strip()

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write(content)
        temp_path = fh.name
    return load_config(temp_path)


class ChatTests(unittest.TestCase):
    def test_send_stream_builds_assistant_message(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        chunks: list[str] = []
        client = FakeStreamingClient()
        service = ChatService(config, client=client, tool_registry=ToolRegistry(specs=()))

        result = service.send_stream(
            session,
            "你好",
            on_chunk=chunks.append,
        )

        self.assertEqual(chunks, ["你好", "，世界"])
        self.assertEqual(result.assistant_message.content, "你好，世界")
        self.assertEqual(session.messages[-1].content, "你好，世界")
        self.assertEqual(client.completion_calls, 0)

    def test_retry_stream_replaces_last_assistant_message(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        service = ChatService(config, client=FakeStreamingClient(), tool_registry=ToolRegistry(specs=()))

        service.send_stream(session, "你好")
        session.messages[-1].content = "旧回复"

        chunks: list[str] = []
        result = service.retry_stream(
            session,
            on_chunk=chunks.append,
        )

        self.assertEqual(chunks, ["你好", "，世界"])
        self.assertEqual(result.assistant_message.content, "你好，世界")
        self.assertEqual(len(session.messages), 2)
        self.assertEqual(session.messages[0].role, "user")
        self.assertEqual(session.messages[1].role, "assistant")
        self.assertEqual(session.messages[1].content, "你好，世界")

    def test_send_stream_preserves_partial_reply_on_interrupt(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        service = ChatService(config, client=InterruptingStreamingClient(), tool_registry=ToolRegistry(specs=()))
        chunks: list[str] = []

        with self.assertRaises(GenerationInterrupted) as context:
            service.send_stream(
                session,
                "你好",
                on_chunk=chunks.append,
            )

        self.assertEqual(chunks, ["partial"])
        self.assertIsNotNone(context.exception.partial_result)
        assert context.exception.partial_result is not None
        self.assertEqual(context.exception.partial_result.assistant_message.content, "partial")
        self.assertEqual(len(session.messages), 2)
        self.assertEqual(session.messages[-1].content, "partial")

    def test_send_stream_injects_relevant_memory_into_system_prompt(self) -> None:
        config = make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_store = MemoryStore(tmpdir)
            memory_service = MemoryService(memory_store)
            client = CapturingClient()
            session = ConversationSession(
                session_id="demo",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
            )
            service = ChatService(
                config,
                client=client,
                memory_service=memory_service,
                tool_registry=ToolRegistry(specs=()),
            )

            service.send_stream(session, "I like jasmine tea.")
            service.send_stream(session, "What tea should I drink tonight?")

        assert client.last_messages is not None
        system_prompt = client.last_messages[0]["content"]
        self.assertIn("Known memory about the user", system_prompt)
        self.assertIn("The user likes jasmine tea.", system_prompt)

    def test_send_stream_injects_live_context_into_system_prompt(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = CapturingClient()
        service = ChatService(
            config,
            client=client,
            live_context_service=StaticLiveContextService(
                "Live weather data for the user's current local area:\n- location: Shanghai"
            ),
            tool_registry=ToolRegistry(specs=()),
        )

        service.send_stream(session, "今天天气怎么样？")

        assert client.last_messages is not None
        system_prompt = client.last_messages[0]["content"]
        self.assertIn("Live weather data for the user's current local area", system_prompt)
        self.assertIn("- location: Shanghai", system_prompt)

    def test_send_stream_skips_live_context_in_system_prompt_when_tools_are_available(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = ToolChoiceCapturingClient()
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="web_search",
                    description="Search the web",
                    parameters_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {"result": "ok"},
                    should_offer=lambda user_text, route_mode: route_mode == "search",
                ),
            )
        )
        service = ChatService(
            config,
            client=client,
            live_context_service=StaticLiveContextService(
                "Live weather data for the user's current local area:\n- location: Shanghai"
            ),
            tool_registry=registry,
        )

        service.send_stream(session, "帮我查一下最新 AI 新闻", mode_override="search")

        system_prompt = client.calls[0]["messages"][0]["content"]
        self.assertNotIn(
            "Live weather data for the user's current local area",
            system_prompt,
        )
        self.assertNotIn("- location: Shanghai", system_prompt)

    def test_send_uses_text_transcript_for_prior_turns_not_old_tool_trace(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        session.messages.extend(
            (
                ChatMessage(
                    role="user",
                    content="帮我查一下油价",
                    created_at="2026-01-01T00:00:00+00:00",
                ),
                ChatMessage(
                    role="assistant",
                    content="",
                    created_at="2026-01-01T00:00:01+00:00",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": '{"query":"oil price"}',
                            },
                        }
                    ],
                ),
                ChatMessage(
                    role="tool",
                    content='{"result":"old search result"}',
                    created_at="2026-01-01T00:00:02+00:00",
                    tool_call_id="call_1",
                ),
                ChatMessage(
                    role="assistant",
                    content="之前我查到油价有波动。",
                    created_at="2026-01-01T00:00:03+00:00",
                    model_alias="chat",
                ),
            )
        )
        session.transcript_messages.extend(
            (
                ChatMessage(
                    role="user",
                    content="帮我查一下油价",
                    created_at="2026-01-01T00:00:00+00:00",
                ),
                ChatMessage(
                    role="assistant",
                    content="之前我查到油价有波动。",
                    created_at="2026-01-01T00:00:03+00:00",
                    model_alias="chat",
                ),
            )
        )
        client = CapturingClient()
        service = ChatService(config, client=client, tool_registry=ToolRegistry(specs=()))

        service.send_stream(session, "再总结一下", on_chunk=lambda chunk: None)

        assert client.last_messages is not None
        non_system_messages = client.last_messages[1:]
        self.assertEqual(
            [message["role"] for message in non_system_messages],
            ["user", "assistant", "user"],
        )
        self.assertNotIn("tool_calls", non_system_messages[1])

    def test_send_stream_executes_tool_calls_and_streams_final_answer(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = ToolCallingClient()
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="web_search",
                    description="Search the web",
                    parameters_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {
                        "query": arguments["query"],
                        "result": "Found release notes",
                    },
                ),
            )
        )
        service = ChatService(
            config,
            client=client,
            tool_registry=registry,
        )

        chunks: list[str] = []
        result = service.send_stream(
            session,
            "帮我查一下最新 release notes",
            mode_override="search",
            on_chunk=chunks.append,
        )

        self.assertEqual(chunks, ["我查到了", "最新结果"])
        self.assertEqual(result.assistant_message.content, "我查到了最新结果")
        self.assertEqual(len(client.completion_requests), 2)
        self.assertTrue(client.completion_requests[0]["tools"])
        self.assertEqual(client.completion_requests[0]["tool_choice"], "required")
        self.assertEqual(len(client.stream_requests), 1)
        final_messages = client.stream_requests[0]["messages"]
        self.assertEqual(final_messages[-2]["role"], "assistant")
        self.assertEqual(final_messages[-2]["tool_calls"][0]["function"]["name"], "web_search")
        self.assertEqual(final_messages[-1]["role"], "tool")
        self.assertIn("Found release notes", final_messages[-1]["content"])
        self.assertEqual(session.messages[1].role, "assistant")
        self.assertIsNotNone(session.messages[1].tool_calls)
        self.assertEqual(session.messages[2].role, "tool")
        self.assertEqual(session.messages[3].content, "我查到了最新结果")

    def test_send_stream_reuses_direct_answer_from_initial_tool_probe(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = DirectAnswerToolClient()
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="web_search",
                    description="Search the web",
                    parameters_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {"result": "ok"},
                ),
            )
        )
        service = ChatService(config, client=client, tool_registry=registry)

        chunks: list[str] = []
        result = service.send_stream(
            session,
            "帮我查一下今天原油价格走势",
            mode_override="search",
            on_chunk=chunks.append,
        )

        self.assertEqual(chunks, ["直接答复"])
        self.assertEqual(result.assistant_message.content, "直接答复")
        self.assertEqual(len(client.completion_requests), 1)
        self.assertTrue(client.completion_requests[0]["tools"])
        self.assertEqual(client.stream_calls, 0)
        self.assertEqual(session.messages[-1].content, "直接答复")

    def test_send_requires_tool_choice_for_explicit_cup_request(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = ToolChoiceCapturingClient()
        service = ChatService(config, client=client, tool_registry=make_cup_registry())

        service.send(session, "帮我把飞机杯调刺激一点")

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["tool_choice"], "required")

    def test_tools_for_request_omits_side_effectful_tools_for_unrelated_input(self) -> None:
        config = make_config()
        service = ChatService(config, client=ToolChoiceCapturingClient(), tool_registry=make_cup_registry())
        model_config = config.get_model("chat")
        provider_config = config.get_provider(model_config.provider)

        tools = service._tools_for_request(
            provider=provider_config,
            model=model_config,
            user_text="你好，最近怎么样",
            route_mode="chat",
        )

        self.assertIsNone(tools)

    def test_tools_for_request_only_returns_matching_tool_specs(self) -> None:
        config = make_config()
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="control_cup",
                    description="Control the CUP hardware",
                    parameters_schema={
                        "type": "object",
                        "properties": {"action": {"type": "string"}},
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {"ok": True},
                    should_offer=lambda user_text, route_mode: _matches_cup_control_intent(
                        user_text
                    ),
                    execution_meta=ToolExecutionMeta(side_effectful=True),
                ),
                ToolSpec(
                    name="web_search",
                    description="Search the web",
                    parameters_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {"ok": True},
                    should_offer=lambda user_text, route_mode: route_mode == "search",
                ),
            )
        )
        service = ChatService(config, client=ToolChoiceCapturingClient(), tool_registry=registry)
        model_config = config.get_model("chat")
        provider_config = config.get_provider(model_config.provider)

        tools = service._tools_for_request(
            provider=provider_config,
            model=model_config,
            user_text="帮我把飞机杯转快一点",
            route_mode="chat",
        )

        self.assertIsNotNone(tools)
        assert tools is not None
        self.assertEqual(
            [tool["function"]["name"] for tool in tools],
            ["control_cup"],
        )

    def test_send_requires_tool_choice_for_cup_scene_followup_with_recent_context(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            messages=[
                ChatMessage(
                    role="user",
                    content="我想爽一点，帮我调一下飞机杯",
                    created_at="2026-01-01T00:00:00+00:00",
                ),
                ChatMessage(
                    role="assistant",
                    content="我可以帮你把速度调柔和、稳定或者刺激一点。",
                    created_at="2026-01-01T00:00:01+00:00",
                    model_alias="chat",
                ),
            ],
        )
        client = ToolChoiceCapturingClient()
        service = ChatService(config, client=client, tool_registry=make_cup_registry())

        service.send(session, "再刺激一点")

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["tool_choice"], "required")

    def test_send_requires_tool_choice_for_explicit_cup_request(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = ToolChoiceCapturingClient()
        service = ChatService(config, client=client, tool_registry=make_cup_registry())

        service.send(session, "把飞机杯转快一点")

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["tool_choice"], "required")

    def test_send_requires_tool_choice_for_cup_followup_with_recent_context(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            messages=[
                ChatMessage(
                    role="user",
                    content="把飞机杯先调到正常速度",
                    created_at="2026-01-01T00:00:00+00:00",
                ),
                ChatMessage(
                    role="assistant",
                    content="现在已经接入 CUP 控制，我们可以继续微调转速和灯光。",
                    created_at="2026-01-01T00:00:01+00:00",
                    model_alias="chat",
                ),
            ],
        )
        client = ToolChoiceCapturingClient()
        service = ChatService(config, client=client, tool_registry=make_cup_registry())

        service.send(session, "再刺激一点")

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["tool_choice"], "required")

    def test_send_falls_back_when_provider_rejects_tools_payload(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = ProviderErrorToolClient()
        service = ChatService(config, client=client)

        result = service.send(session, "帮我查一下最新 release notes", mode_override="search")

        self.assertEqual(result.assistant_message.content, "fallback answer")
        self.assertEqual(len(client.calls), 2)
        self.assertIsNotNone(client.calls[0]["tools"])
        self.assertIsNone(client.calls[1]["tools"])

    def test_send_skips_tools_after_provider_rejection_is_cached(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = ProviderErrorToolClient()
        service = ChatService(config, client=client)

        service.send(session, "帮我查一下最新 release notes", mode_override="search")
        service.send(session, "再查一下最新新闻", mode_override="search")

        self.assertEqual(len(client.calls), 3)
        self.assertIsNotNone(client.calls[0]["tools"])
        self.assertIsNone(client.calls[1]["tools"])
        self.assertIsNone(client.calls[2]["tools"])

    def test_send_does_not_disable_tools_for_unreachable_provider_errors(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = UnreachableProviderToolClient()
        service = ChatService(config, client=client)

        with self.assertRaisesRegex(Exception, "unreachable"):
            service.send(session, "帮我查一下最新 release notes", mode_override="search")

        self.assertEqual(len(client.calls), 1)
        self.assertIsNotNone(client.calls[0]["tools"])

    def test_send_retries_without_tool_choice_when_client_signature_rejects_it(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = ToolChoiceUnsupportedClient()
        service = ChatService(config, client=client)

        result = service.send(session, "帮我查一下最新 release notes", mode_override="search")

        self.assertEqual(result.assistant_message.content, "fallback without tool_choice")
        self.assertEqual(len(client.calls), 1)
        self.assertIsNotNone(client.calls[0]["tools"])

    def test_send_normalizes_tool_followup_messages_for_strict_providers(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = StrictToolPayloadClient()
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="web_search",
                    description="Search the web",
                    parameters_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {
                        "query": arguments["query"],
                        "result": "WTI crude moved higher",
                    },
                ),
            )
        )
        service = ChatService(
            config,
            client=client,
            tool_registry=registry,
        )

        result = service.send(session, "帮我查一下今天原油价格走势", mode_override="search")

        self.assertEqual(result.assistant_message.content, "严格 provider 通过")
        self.assertEqual(len(client.completion_requests), 2)

    def test_send_retries_tool_followup_with_structured_arguments_for_ollama(self) -> None:
        config = make_ollama_native_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = OllamaStructuredArgumentsClient()
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="web_search",
                    description="Search the web",
                    parameters_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {
                        "query": arguments["query"],
                        "result": "Brent was volatile",
                    },
                ),
            )
        )
        service = ChatService(
            config,
            client=client,
            tool_registry=registry,
        )

        result = service.send(session, "帮我查一下今天原油价格走势", mode_override="search")

        self.assertEqual(result.assistant_message.content, "ollama follow-up ok")
        self.assertEqual(len(client.completion_requests), 2)

    def test_send_preserves_reasoning_content_for_kimi_tool_followup(self) -> None:
        config = make_kimi_thinking_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = KimiThinkingToolClient()
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="web_search",
                    description="Search the web",
                    parameters_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {
                        "query": arguments["query"],
                        "result": "Found release notes",
                    },
                ),
            )
        )
        service = ChatService(
            config,
            client=client,
            tool_registry=registry,
        )

        result = service.send(session, "帮我查一下最新 release notes", mode_override="search")

        self.assertEqual(result.assistant_message.content, "Kimi thinking tool follow-up ok")
        self.assertEqual(len(client.completion_requests), 2)
        self.assertEqual(
            session.messages[1].reasoning_content,
            "先搜索最新 release notes，再整理成简短答案。",
        )

    def test_send_retries_kimi_required_tool_choice_with_disable_thinking(self) -> None:
        config = make_kimi_thinking_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            messages=[
                ChatMessage(
                    role="user",
                    content="把灯变蓝",
                    created_at="2026-01-01T00:00:00+00:00",
                ),
                ChatMessage(
                    role="assistant",
                    content="已经帮你把灯调成蓝色。",
                    created_at="2026-01-01T00:00:01+00:00",
                    model_alias="chat",
                ),
                ChatMessage(
                    role="tool",
                    name="control_cup",
                    content='{"ok": true, "action": "set_led"}',
                    created_at="2026-01-01T00:00:02+00:00",
                ),
            ],
        )
        client = KimiToolChoiceDisableThinkingClient()
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="control_cup",
                    description="Control the CUP hardware",
                    parameters_schema={
                        "type": "object",
                        "properties": {"action": {"type": "string"}},
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {"ok": True, "requested": arguments},
                    should_offer=lambda user_text, route_mode: _matches_cup_control_intent(
                        user_text
                    ),
                    execution_meta=ToolExecutionMeta(side_effectful=True),
                ),
            )
        )
        service = ChatService(
            config,
            client=client,
            tool_registry=registry,
        )

        result = service.send(session, "我想要温馨一点的颜色")

        self.assertEqual(result.assistant_message.content, "已经帮你调成更温馨的颜色。")
        self.assertEqual(len(client.calls), 4)
        self.assertFalse(client.calls[0]["disable_thinking"])
        self.assertEqual(client.calls[0]["tool_choice"], "required")
        self.assertTrue(client.calls[1]["disable_thinking"])
        self.assertEqual(client.calls[1]["tool_choice"], "required")
        self.assertFalse(client.calls[2]["disable_thinking"])
        self.assertEqual(client.calls[2]["tool_choice"], "required")
        self.assertTrue(client.calls[3]["disable_thinking"])
        self.assertEqual(client.calls[3]["tool_choice"], "required")
        self.assertEqual(session.messages[-2].role, "tool")

    def test_send_enables_kimi_builtin_web_search(self) -> None:
        config = make_kimi_thinking_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = KimiBuiltinWebSearchClient()
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="web_search",
                    description="Search the web",
                    parameters_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {"result": "unused local search"},
                ),
            )
        )
        service = ChatService(
            config,
            client=client,
            tool_registry=registry,
        )

        result = service.send(session, "查一下最新 AI 新闻", mode_override="search")

        self.assertEqual(result.assistant_message.content, "这是联网搜索结果摘要。")
        self.assertEqual(len(client.calls), 2)
        self.assertFalse(client.calls[0].get("disable_thinking", False))
        builtin_tools = [
            tool for tool in client.calls[0]["tools"] or ()
            if tool.get("type") == "builtin_function"
        ]
        local_search_tools = [
            tool for tool in client.calls[0]["tools"] or ()
            if tool.get("type") == "function"
            and isinstance(tool.get("function"), dict)
            and tool["function"].get("name") == "web_search"
        ]
        self.assertEqual(len(builtin_tools), 1)
        self.assertEqual(builtin_tools[0]["function"]["name"], "$web_search")
        self.assertEqual(local_search_tools, [])

    def test_send_omits_legacy_blank_tool_call_ids_from_transcript_replay(self) -> None:
        config = make_kimi_thinking_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        session.messages.extend(
            (
                ChatMessage(
                    role="assistant",
                    content="",
                    created_at="2026-01-01T00:00:01+00:00",
                    tool_calls=[
                        {
                            "id": "",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": '{"query":"old query"}',
                            },
                        }
                    ],
                ),
                ChatMessage(
                    role="tool",
                    content='{"result":"old result"}',
                    created_at="2026-01-01T00:00:02+00:00",
                    tool_call_id="",
                ),
            )
        )
        session.transcript_messages.append(
            ChatMessage(
                role="user",
                content="旧问题",
                created_at="2026-01-01T00:00:00+00:00",
            )
        )
        client = KimiLegacyToolCallIdClient()
        service = ChatService(config, client=client)

        result = service.send(session, "继续", mode_override="chat")

        self.assertEqual(result.assistant_message.content, "legacy tool trace omitted from transcript replay")
        self.assertEqual(len(client.calls), 1)

    def test_send_drops_orphan_tool_messages_before_request(self) -> None:
        config = make_kimi_thinking_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        session.messages.append(
            ChatMessage(
                role="tool",
                content='{"result":"orphan"}',
                created_at="2026-01-01T00:00:01+00:00",
                tool_call_id="",
            )
        )
        client = KimiOrphanToolMessageClient()
        service = ChatService(config, client=client)

        result = service.send(session, "继续", mode_override="chat")

        self.assertEqual(result.assistant_message.content, "orphan tool message dropped")
        self.assertEqual(len(client.calls), 1)

    def test_send_stream_hides_tool_call_content_from_user_and_followup(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = ToolCallingClientWithLeakyContent()
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="web_search",
                    description="Search the web",
                    parameters_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {
                        "query": arguments["query"],
                        "result": "WTI moved",
                    },
                ),
            )
        )
        service = ChatService(
            config,
            client=client,
            tool_registry=registry,
        )

        chunks: list[str] = []
        result = service.send_stream(
            session,
            "帮我查一下今天原油价格走势",
            mode_override="search",
            on_chunk=chunks.append,
        )

        self.assertEqual(chunks, ["最终", "答复"])
        self.assertEqual(result.assistant_message.content, "最终答复")
        self.assertEqual(session.messages[1].content, "")
        self.assertEqual(len(client.stream_requests), 1)

    def test_send_stream_continues_past_interim_planning_text_before_final_answer(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = MultiRoundPlanningClient()
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="web_search",
                    description="Search the web",
                    parameters_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {
                        "query": arguments["query"],
                        "result": "WTI rose on supply concerns",
                    },
                ),
            )
        )
        service = ChatService(
            config,
            client=client,
            tool_registry=registry,
        )

        chunks: list[str] = []
        result = service.send_stream(
            session,
            "今天原油价格走势怎么样？最近什么事情影响了原油价格",
            mode_override="search",
            on_chunk=chunks.append,
        )

        self.assertEqual(
            chunks,
            ["今天原油价格小幅上涨，主要受供应收紧预期影响。"],
        )
        self.assertEqual(
            result.assistant_message.content,
            "今天原油价格小幅上涨，主要受供应收紧预期影响。",
        )
        self.assertEqual(len(client.completion_requests), 3)
        self.assertEqual(client.stream_calls, 1)
        self.assertEqual(session.messages[1].role, "assistant")
        self.assertEqual(session.messages[1].content, "")
        self.assertEqual(session.messages[2].role, "tool")

    def test_send_supports_multiple_post_tool_rounds(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = MultiStepToolLoopClient()
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="web_search",
                    description="Search the web",
                    parameters_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {
                        "query": arguments["query"],
                        "result": "Oil prices rose after new supply headlines",
                    },
                ),
                ToolSpec(
                    name="summarize_findings",
                    description="Summarize search results",
                    parameters_schema={
                        "type": "object",
                        "properties": {"topic": {"type": "string"}},
                        "required": ["topic"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {
                        "topic": arguments["topic"],
                        "summary": "Supply concerns and inventory expectations both mattered",
                    },
                ),
            )
        )
        service = ChatService(
            config,
            client=client,
            tool_registry=registry,
        )

        result = service.send(session, "分析一下最近油价变化原因", mode_override="search")

        self.assertEqual(result.assistant_message.content, "综合来看，油价受供应和库存预期共同影响。")
        self.assertEqual(len(client.completion_requests), 3)
        self.assertEqual(session.messages[1].role, "assistant")
        self.assertEqual(session.messages[2].role, "tool")
        self.assertEqual(session.messages[3].role, "assistant")
        self.assertEqual(session.messages[4].role, "tool")
        self.assertEqual(session.messages[5].role, "assistant")

    def test_send_returns_structured_ask_user_prompt(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = AskUserToolClient()
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="get_weather_by_location",
                    description="Get weather",
                    parameters_schema={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {"location": arguments["location"], "ok": True},
                ),
            )
        )
        service = ChatService(config, client=client, tool_registry=registry)

        result = service.send(session, "帮我查天气", mode_override="chat")

        self.assertEqual(result.assistant_message.content, "你想查哪个城市？")
        self.assertIsNotNone(result.ask_user)
        assert result.ask_user is not None
        self.assertEqual(result.ask_user.options[0].label, "上海")
        self.assertIsNotNone(session.pending_ask_user)
        self.assertEqual(session.transcript_messages[-1].content, "你想查哪个城市？")

    def test_send_uses_pending_ask_option_value_for_numeric_reply(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = AskUserFollowupClient()
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="get_weather_by_location",
                    description="Get weather",
                    parameters_schema={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {"location": arguments["location"], "ok": True},
                ),
            )
        )
        service = ChatService(config, client=client, tool_registry=registry)

        first = service.send(session, "帮我查天气", mode_override="chat")
        second = service.send(session, "2", mode_override="chat")

        self.assertIsNotNone(first.ask_user)
        self.assertEqual(session.messages[-2].content, "东京")
        self.assertEqual(session.transcript_messages[-2].content, "东京")
        self.assertEqual(second.assistant_message.content, "收到地点：东京")
        self.assertIsNone(session.pending_ask_user)

    def test_send_stream_emits_agent_steps_for_multi_tool_loop(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = MultiStepToolLoopClient()
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="web_search",
                    description="Search the web",
                    parameters_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {
                        "query": arguments["query"],
                        "result": "Oil prices rose after new supply headlines",
                    },
                ),
                ToolSpec(
                    name="summarize_findings",
                    description="Summarize search results",
                    parameters_schema={
                        "type": "object",
                        "properties": {"topic": {"type": "string"}},
                        "required": ["topic"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {
                        "topic": arguments["topic"],
                        "summary": "Supply concerns and inventory expectations both mattered",
                    },
                ),
            )
        )
        service = ChatService(config, client=client, tool_registry=registry)

        steps: list[tuple[str, str | None]] = []
        result = service.send_stream(
            session,
            "分析一下最近油价变化原因",
            mode_override="search",
            on_chunk=lambda chunk: None,
            on_step=lambda step: steps.append((step.kind, step.tool_name)),
        )

        self.assertEqual(result.assistant_message.content, "综合来看，油价受供应和库存预期共同影响。")
        self.assertEqual(
            steps,
            [
                ("tool_call", "web_search"),
                ("tool_result", "web_search"),
                ("tool_call", "summarize_findings"),
                ("tool_result", "summarize_findings"),
                ("final", None),
            ],
        )
        self.assertEqual(client.stream_calls, 1)

    def test_send_supports_multi_step_weather_tool_chain(self) -> None:
        config = make_config()
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        client = WeatherToolChainClient()
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="get_public_ip",
                    description="Get public IP",
                    parameters_schema={"type": "object", "properties": {}, "additionalProperties": False},
                    handler=lambda arguments: {"public_ip": "203.0.113.10"},
                ),
                ToolSpec(
                    name="get_ip_location",
                    description="Get IP location",
                    parameters_schema={
                        "type": "object",
                        "properties": {"ip": {"type": "string"}},
                        "required": ["ip"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {
                        "public_ip": arguments["ip"],
                        "location": "Shanghai, Shanghai, China",
                        "timezone": "Asia/Shanghai",
                    },
                ),
                ToolSpec(
                    name="get_weather_by_location",
                    description="Get weather by location",
                    parameters_schema={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {
                        "location": arguments["location"],
                        "current_condition": "Partly cloudy",
                        "today_high_c": 27.2,
                    },
                ),
            )
        )
        service = ChatService(
            config,
            client=client,
            tool_registry=registry,
        )

        result = service.send(session, "我这里今天天气怎么样？", mode_override="search")

        self.assertEqual(result.assistant_message.content, "上海当前多云，今天最高 27.2C。")
        self.assertEqual(len(client.completion_requests), 4)
        self.assertEqual(session.messages[1].tool_calls[0]["function"]["name"], "get_public_ip")
        self.assertEqual(session.messages[3].tool_calls[0]["function"]["name"], "get_ip_location")
        self.assertEqual(session.messages[5].tool_calls[0]["function"]["name"], "get_weather_by_location")


if __name__ == "__main__":
    unittest.main()
