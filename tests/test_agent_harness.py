from __future__ import annotations

from types import SimpleNamespace
import unittest

from whesper.agent_harness import AgentHarness
from whesper.agent_types import AgentCompletion, ToolInvocation
from whesper.router import RouteDecision
from whesper.session import ChatMessage, ConversationSession
from whesper.tool_executor import ToolExecutor
from whesper.provider_profile import OPENAI_DEFAULT_PROFILE, ProviderProfile
from whesper.tool_protocol import DefaultToolProtocolAdapter
from whesper.tools import ToolRegistry, ToolSpec


class StubMessageBuilder:
    def build_messages(
        self,
        session: ConversationSession,
        model_system_prompt: str | None,
        *,
        target_profile: ProviderProfile,
        user_text: str,
        route_mode: str,
        planning_prompt: str | None = None,
        preflight_prompt: str | None = None,
        include_live_context: bool = True,
        ensure_reasoning_content: bool = False,
    ) -> list[dict[str, object]]:
        system_parts = [model_system_prompt or "system"]
        if preflight_prompt:
            system_parts.append(preflight_prompt)
        messages: list[dict[str, object]] = [
            {"role": "system", "content": "\n\n".join(system_parts)}
        ]
        for message in session.messages:
            payload: dict[str, object] = {
                "role": message.role,
                "content": message.content,
            }
            if message.tool_calls is not None:
                payload["tool_calls"] = message.tool_calls
            if message.tool_call_id is not None:
                payload["tool_call_id"] = message.tool_call_id
            if message.name is not None:
                payload["name"] = message.name
            if message.reasoning_content is not None:
                payload["reasoning_content"] = message.reasoning_content
            messages.append(payload)
        return messages


class StubCompletionRequester:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        session_id: str,
        decision: RouteDecision,
        provider,
        model,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        streamed: bool,
        structured_tool_arguments: bool = False,
    ) -> AgentCompletion:
        self.calls.append(
            {
                "session_id": session_id,
                "decision": decision,
                "provider": provider,
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "streamed": streamed,
            }
        )
        if len(self.calls) == 1:
            return AgentCompletion(
                content="",
                raw_response={},
                tool_calls=(
                    ToolInvocation(
                        tool_call_id="call_fact_1",
                        name="lookup_fact",
                        arguments_json='{"topic":"release notes"}',
                    ),
                ),
            )

        assistant_tool_message = messages[-2]
        tool_message = messages[-1]
        if assistant_tool_message["role"] != "assistant":
            raise AssertionError("assistant tool call message missing from follow-up")
        if tool_message["role"] != "tool":
            raise AssertionError("tool result missing from follow-up")
        return AgentCompletion(content="final answer", raw_response={})


class AskUserCompletionRequester:
    def __call__(
        self,
        *,
        session_id: str,
        decision: RouteDecision,
        provider,
        model,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        streamed: bool,
        structured_tool_arguments: bool = False,
    ) -> AgentCompletion:
        return AgentCompletion(
            content=(
                '<ask_user>{"prompt":"你想查哪个城市？","options":'
                '[{"label":"上海","value":"上海"},{"label":"东京","value":"东京"}],'
                '"allow_free_text":true,"field_name":"location"}</ask_user>'
            ),
            raw_response={},
        )


class AgentHarnessTests(unittest.TestCase):
    def test_harness_runs_multi_step_loop_without_chat_service(self) -> None:
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            messages=[
                ChatMessage(
                    role="user",
                    content="帮我查一下最新 release notes",
                    created_at="2026-01-01T00:00:00+00:00",
                )
            ],
        )
        tool_registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="lookup_fact",
                    description="Look up a fact",
                    parameters_schema={
                        "type": "object",
                        "properties": {"topic": {"type": "string"}},
                        "required": ["topic"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: {
                        "topic": arguments["topic"],
                        "result": "Found the latest release notes",
                    },
                ),
            )
        )
        requester = StubCompletionRequester()
        protocol_adapter = DefaultToolProtocolAdapter()
        trace_events: list[dict[str, object]] = []

        harness = AgentHarness(
            message_builder=StubMessageBuilder(),
            tool_executor=ToolExecutor(tool_registry),
            completion_requester=requester,
            tool_call_payload_builder=protocol_adapter.tool_call_payload,
            tool_choice_builder=(
                lambda session, user_text, tools, *, route_mode: "required" if tools else None
            ),
            reasoning_content_resolver=lambda decision, completion: completion.reasoning_content,
            trace_emitter=lambda **kwargs: trace_events.append(kwargs),
            max_rounds=4,
        )

        result = harness.run_until_final(
            session,
            RouteDecision(model_alias="chat", mode="search", reason="explicit search mode"),
            provider=SimpleNamespace(name="stub-provider"),
            model=SimpleNamespace(name="stub-model"),
            model_system_prompt="system",
            user_text="帮我查一下最新 release notes",
            route_mode="search",
            target_profile=OPENAI_DEFAULT_PROFILE,
            tools=tool_registry.openai_tools(),
        )

        self.assertEqual(result.completion.content, "final answer")
        self.assertEqual(
            [step.kind for step in result.steps],
            ["tool_call", "tool_result", "final"],
        )
        self.assertEqual(session.messages[1].role, "assistant")
        self.assertEqual(session.messages[2].role, "tool")
        self.assertEqual(len(requester.calls), 2)
        self.assertTrue(any(event["kind"] == "tool_calls" for event in trace_events))
        self.assertTrue(any(event["kind"] == "tool_result" for event in trace_events))

    def test_harness_returns_ask_user_action(self) -> None:
        session = ConversationSession(
            session_id="demo",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            messages=[
                ChatMessage(
                    role="user",
                    content="帮我查天气",
                    created_at="2026-01-01T00:00:00+00:00",
                )
            ],
        )
        tool_registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="get_weather_by_location",
                    description="Get weather",
                    parameters_schema={"type": "object"},
                    handler=lambda arguments: {"ok": True},
                ),
            )
        )
        protocol_adapter = DefaultToolProtocolAdapter()
        harness = AgentHarness(
            message_builder=StubMessageBuilder(),
            tool_executor=ToolExecutor(tool_registry),
            completion_requester=AskUserCompletionRequester(),
            tool_call_payload_builder=protocol_adapter.tool_call_payload,
            tool_choice_builder=lambda session, user_text, tools, *, route_mode: None,
            reasoning_content_resolver=lambda decision, completion: completion.reasoning_content,
            max_rounds=2,
        )

        result = harness.run_until_final(
            session,
            RouteDecision(model_alias="chat", mode="chat", reason="default chat model"),
            provider=SimpleNamespace(name="stub-provider"),
            model=SimpleNamespace(name="stub-model"),
            model_system_prompt="system",
            user_text="帮我查天气",
            route_mode="chat",
            target_profile=OPENAI_DEFAULT_PROFILE,
            tools=tool_registry.openai_tools(),
        )

        self.assertIsNotNone(result.ask_user)
        assert result.ask_user is not None
        self.assertEqual(result.ask_user.prompt, "你想查哪个城市？")
        self.assertEqual(result.ask_user.options[0].label, "上海")
        self.assertEqual([step.kind for step in result.steps], ["ask_user"])
