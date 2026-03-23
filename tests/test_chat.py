from __future__ import annotations

import tempfile
import textwrap
import unittest

from whesper.chat import ChatService, GenerationInterrupted
from whesper.client import CompletionResult, ToolCall
from whesper.config import load_config
from whesper.live_data import LiveContextService
from whesper.memory import MemoryService, MemoryStore
from whesper.session import ConversationSession
from whesper.tools import ToolRegistry, ToolSpec


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
        raise AssertionError("unexpected extra non-stream completion")

    def create_chat_completion_stream(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.stream_requests.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        yield "我查到了"
        yield "最新结果"


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

    def create_chat_completion_stream(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.stream_requests.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        assistant_tool_message = messages[-2]
        if assistant_tool_message["role"] != "assistant":
            raise AssertionError("assistant tool call message missing")
        if assistant_tool_message.get("content", "unexpected") is not None:
            raise AssertionError("tool call assistant content should not leak to follow-up")
        yield "最终答复"


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
        service = ChatService(config, client=client)

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
        service = ChatService(config, client=FakeStreamingClient())

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
        service = ChatService(config, client=InterruptingStreamingClient())
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
                "Live weather data for the user's current IP-based location:\n- location: Shanghai"
            ),
        )

        service.send_stream(session, "今天天气怎么样？")

        assert client.last_messages is not None
        system_prompt = client.last_messages[0]["content"]
        self.assertIn("Live weather data for the user's current IP-based location", system_prompt)
        self.assertIn("- location: Shanghai", system_prompt)

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
        self.assertEqual(len(client.completion_requests), 1)
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
        service = ChatService(config, client=client)

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

        self.assertEqual(chunks, ["最终答复"])
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
        self.assertEqual(len(client.completion_requests), 2)
        self.assertEqual(client.stream_calls, 1)
        self.assertEqual(session.messages[1].role, "assistant")
        self.assertEqual(session.messages[1].content, "")
        self.assertEqual(session.messages[2].role, "tool")


if __name__ == "__main__":
    unittest.main()
