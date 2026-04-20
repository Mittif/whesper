from __future__ import annotations

import io
import tempfile
import unittest

from whesper.cli import (
    CupHeartbeatMonitor,
    CupHeartbeatSnapshot,
    apply_pinned_model_override,
    build_footer_meta,
    ensure_valid_session_model,
    handle_command,
    run_streaming_turn,
)
from whesper.chat import ChatService, ChatTurnResult
from whesper.client import CompletionResult, ToolCall
from whesper.commands import ParsedCommand
from whesper.tools import ToolRegistry, ToolSpec
from whesper.config import (
    AppConfig,
    AppSettings,
    CUP_DEFAULT_BASE_URL,
    LiveContextSettings,
    ModelConfig,
    PersonaConfig,
    ProviderConfig,
    SchedulerConfig,
    ShellSandboxSettings,
)
from whesper.memory import MemoryStore
from whesper.router import RouteDecision
from whesper.session import ChatMessage, ConversationSession, SessionStore
from whesper.trace import TraceStore, make_trace_event
from whesper.agent_types import ToolExecutionMeta

CUP_DEFAULT_API_BASE_URL = f"{CUP_DEFAULT_BASE_URL}/api"


class FakeStreamingClient:
    def create_chat_completion_stream(self, provider, model, messages):
        yield "new"
        yield " reply"

    def create_chat_completion(self, provider, model, messages, *, tools=None):
        return CompletionResult(content="", raw_response={})


class InterruptingStreamingClient:
    def create_chat_completion_stream(self, provider, model, messages):
        yield "partial"
        raise KeyboardInterrupt()

    def create_chat_completion(self, provider, model, messages, *, tools=None):
        return CompletionResult(content="", raw_response={})


class DebugClient:
    def create_chat_completion_stream(self, provider, model, messages):
        raise AssertionError("streaming should not be called in this test")

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        return CompletionResult(content="debug answer", raw_response={})


class ToolTraceClient:
    def __init__(self) -> None:
        self.calls = 0

    def create_chat_completion_stream(self, provider, model, messages):
        raise AssertionError("streaming should not be called in this test")

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return CompletionResult(
                content="",
                raw_response={},
                tool_calls=(
                    ToolCall(
                        tool_call_id="call_cup_1",
                        name="control_cup",
                        arguments_json='{"action":"nudge_intensity","direction":"up"}',
                    ),
                ),
            )
        return CompletionResult(content="已经继续增强了一点。", raw_response={})


class InterruptingToolProbeClient:
    def create_chat_completion_stream(self, provider, model, messages):
        raise AssertionError("streaming should not be called in this test")

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        raise KeyboardInterrupt()


class AskUserCliClient:
    def create_chat_completion_stream(self, provider, model, messages):
        raise AssertionError("streaming should not be called in this test")

    def create_chat_completion(self, provider, model, messages, *, tools=None, tool_choice=None):
        return CompletionResult(
            content=(
                '<ask_user>{"prompt":"你想查哪个城市？","options":'
                '[{"label":"上海","value":"上海"},{"label":"东京","value":"东京","description":"更快一点"}],'
                '"allow_free_text":true,"field_name":"location"}</ask_user>'
            ),
            raw_response={},
        )


def build_config() -> AppConfig:
    return AppConfig(
        app=AppSettings(),
        persona=PersonaConfig(),
        scheduler=SchedulerConfig(chat_model="local_chat"),
        live_context=LiveContextSettings(),
        shell_sandbox=ShellSandboxSettings(),
        providers={
            "local": ProviderConfig(
                name="local",
                kind="ollama_native",
                base_url="http://localhost:11434",
            )
        },
        models={
            "local_chat": ModelConfig(
                name="local_chat",
                provider="local",
                model="qwen",
            )
        },
        source_path=None,  # type: ignore[arg-type]
    )


class CliTests(unittest.TestCase):
    def build_chat_service(self, config: AppConfig) -> ChatService:
        return ChatService(config, client=FakeStreamingClient(), tool_registry=ToolRegistry(specs=()))

    def build_interrupting_chat_service(self, config: AppConfig) -> ChatService:
        return ChatService(config, client=InterruptingStreamingClient(), tool_registry=ToolRegistry(specs=()))

    def build_debug_chat_service(
        self,
        config: AppConfig,
        *,
        trace_store: TraceStore | None = None,
    ) -> ChatService:
        return ChatService(config, client=DebugClient(), trace_store=trace_store, tool_registry=ToolRegistry(specs=()))

    def build_interrupting_tool_probe_service(self, config: AppConfig) -> ChatService:
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
        return ChatService(config, client=InterruptingToolProbeClient(), tool_registry=registry)

    def build_cup_chat_service(
        self,
        config: AppConfig,
        *,
        client=None,
        trace_store: TraceStore | None = None,
        seen_arguments: list[dict[str, object]] | None = None,
    ) -> ChatService:
        def handler(arguments: dict[str, object]) -> dict[str, object]:
            if seen_arguments is not None:
                seen_arguments.append(arguments)
            return {
                "ok": True,
                "action": arguments.get("action"),
                "echo": arguments,
                "request_trace": [
                    {"method": "GET", "url": f"{CUP_DEFAULT_API_BASE_URL}/motor"},
                    {
                        "method": "POST",
                        "url": f"{CUP_DEFAULT_API_BASE_URL}/motor",
                        "payload": arguments,
                    },
                ],
            }

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
                    handler=handler,
                    execution_meta=ToolExecutionMeta(side_effectful=True),
                ),
            )
        )
        return ChatService(
            config,
            client=client or DebugClient(),
            trace_store=trace_store,
            tool_registry=registry,
        )

    def test_in_session_model_switch_command_is_disabled(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/model", arg="missing-model"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertFalse(outcome.should_exit)
        self.assertEqual(outcome.session.pinned_model, "auto")
        self.assertIn("In-session model switching is disabled.", rendered)
        self.assertIn("whesper chat --model", rendered)

    def test_invalid_mode_is_handled_without_crash(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/mode", arg="12345"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertFalse(outcome.should_exit)
        self.assertEqual(outcome.mode_override, "auto")
        self.assertIn("Invalid mode: 12345", rendered)
        self.assertIn("available modes: auto, chat, reasoning, search", rendered)

    def test_invalid_saved_session_model_is_reset_to_auto(self) -> None:
        config = build_config()
        session = ConversationSession(
            session_id="main",
            created_at="2026-03-18T00:00:00+00:00",
            updated_at="2026-03-18T00:00:00+00:00",
            pinned_model="missing-model",
        )
        output = io.StringIO()

        changed = ensure_valid_session_model(
            config,
            session,
            output_stream=output,
        )

        rendered = output.getvalue()
        self.assertTrue(changed)
        self.assertEqual(session.pinned_model, "auto")
        self.assertIn("Session-scoped model switching is disabled", rendered)
        self.assertIn("reset model override to auto", rendered)

    def test_invalid_startup_model_override_keeps_auto(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            apply_pinned_model_override(
                config,
                store,
                session,
                "missing-model",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertEqual(session.pinned_model, "auto")
        self.assertIn("missing-model", rendered)
        self.assertIn("Starting with scheduler.chat_model", rendered)

    def test_status_command_renders_current_state(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/status"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("session: main", rendered)
        self.assertIn("effective model: local_chat", rendered)

    def test_info_command_renders_model_details(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/info"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("alias: local_chat", rendered)
        self.assertIn("model id: qwen", rendered)

    def test_cup_status_command_executes_control_tool(self) -> None:
        config = build_config()
        seen_arguments: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/cup-status"),
                config=config,
                store=store,
                chat_service=self.build_cup_chat_service(config, seen_arguments=seen_arguments),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertEqual(seen_arguments, [{"action": "status"}])
        self.assertIn("CUP Status", rendered)
        self.assertIn('"action": "status"', rendered)

    def test_cup_speed_command_sets_motor_velocity(self) -> None:
        config = build_config()
        seen_arguments: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/cup-speed", arg="88"),
                config=config,
                store=store,
                chat_service=self.build_cup_chat_service(config, seen_arguments=seen_arguments),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertEqual(
            seen_arguments,
            [{"action": "set_motor", "target_velocity": 88.0, "enabled": True}],
        )
        self.assertIn("CUP Speed", rendered)
        self.assertIn('"target_velocity": 88.0', rendered)

    def test_cup_led_command_sets_color_and_blink(self) -> None:
        config = build_config()
        seen_arguments: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/cup-led", arg="#ff69b4 1.5 3"),
                config=config,
                store=store,
                chat_service=self.build_cup_chat_service(config, seen_arguments=seen_arguments),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertEqual(
            seen_arguments,
            [{"action": "set_led", "color": "#ff69b4", "blink_hz": 1.5, "blink_mode": 3}],
        )
        self.assertIn("CUP LED", rendered)
        self.assertIn('"blink_mode": 3', rendered)

    def test_cup_scene_command_validates_scene_name(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/cup-scene", arg="turbo"),
                config=config,
                store=store,
                chat_service=self.build_cup_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("scene must be one of", rendered)
        self.assertIn("Usage: /cup-scene <gentle|steady|intense|cooldown>", rendered)

    def test_cup_command_reports_missing_tool(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/cup-status"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("control_cup tool is not available", rendered)
        self.assertIn("esp32-cup.local", rendered)

    def test_cup_heartbeat_toolbar_infers_running_from_target_velocity_without_enabled(self) -> None:
        config = build_config()
        monitor = CupHeartbeatMonitor(self.build_chat_service(config))
        monitor._set_snapshot(  # noqa: SLF001 - test-only direct state injection
            CupHeartbeatSnapshot(
                connected=True,
                motor_target=40.0,
                motor_enabled=None,
            )
        )

        _, label = monitor.toolbar_fragment()

        self.assertIn("cup online", label)
        self.assertIn("v40", label)

    def test_history_command_renders_colored_cards_for_roles(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            session.transcript_messages.extend(
                [
                    ChatMessage(
                        role="user",
                        content="hello there",
                        created_at="2026-03-18T00:00:00+00:00",
                    ),
                    ChatMessage(
                        role="assistant",
                        content="hi back",
                        created_at="2026-03-18T00:00:01+00:00",
                        model_alias="local_chat",
                        route_reason="default chat model",
                    ),
                ]
            )
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/history"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("History (2)", rendered)
        self.assertIn("You", rendered)
        self.assertIn("Whesper", rendered)
        self.assertIn("model local_chat", rendered)
        self.assertIn("default chat model", rendered)
        self.assertIn("hello there", rendered)
        self.assertIn("hi back", rendered)

    def test_history_command_prefers_transcript_over_raw_tool_trace(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            session.messages.extend(
                [
                    ChatMessage(
                        role="user",
                        content="爱丁堡最近天气怎么样？",
                        created_at="2026-03-27T00:00:00+00:00",
                    ),
                    ChatMessage(
                        role="assistant",
                        content="",
                        created_at="2026-03-27T00:00:01+00:00",
                        tool_calls=[
                            {
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "get_weather_by_location",
                                    "arguments": '{"location":"爱丁堡"}',
                                },
                            }
                        ],
                    ),
                    ChatMessage(
                        role="tool",
                        content='{"ok": true, "location": "Edinburgh"}',
                        created_at="2026-03-27T00:00:02+00:00",
                        tool_call_id="call_weather",
                    ),
                    ChatMessage(
                        role="assistant",
                        content="爱丁堡今天多云，体感偏凉。",
                        created_at="2026-03-27T00:00:03+00:00",
                        model_alias="local_chat",
                        route_reason="default chat model",
                    ),
                ]
            )
            session.transcript_messages.extend(
                [
                    ChatMessage(
                        role="user",
                        content="爱丁堡最近天气怎么样？",
                        created_at="2026-03-27T00:00:00+00:00",
                    ),
                    ChatMessage(
                        role="assistant",
                        content="爱丁堡今天多云，体感偏凉。",
                        created_at="2026-03-27T00:00:03+00:00",
                        model_alias="local_chat",
                        route_reason="default chat model",
                    ),
                ]
            )
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/history"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("History (2)", rendered)
        self.assertIn("爱丁堡最近天气怎么样？", rendered)
        self.assertIn("爱丁堡今天多云，体感偏凉。", rendered)
        self.assertNotIn("Tool", rendered)
        self.assertNotIn("call_weather", rendered)
        self.assertNotIn('{"ok": true, "location": "Edinburgh"}', rendered)

    def test_history_command_handles_empty_session(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/history"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("Session is empty.", rendered)

    def test_trace_command_handles_empty_trace_store(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            trace_store = TraceStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/trace"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                trace_store=trace_store,
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("No trace events yet.", rendered)

    def test_trace_command_renders_recent_trace_events(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            trace_store = TraceStore(tmpdir)
            trace_store.append(
                make_trace_event(
                    kind="completion_request",
                    session_id="main",
                    model_alias="local_chat",
                    provider_name="local",
                    route_mode="search",
                    streamed=False,
                    tools_enabled=True,
                    tool_choice="required",
                    note="testing trace",
                    preview="user: 今天原油价格走势怎么样",
                )
            )
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/trace"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                trace_store=trace_store,
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("Trace (1)", rendered)
        self.assertIn("completion_request", rendered)
        self.assertIn("tool_choice required", rendered)
        self.assertIn("今天原油价格走势怎么样", rendered)

    def test_trace_command_with_message_runs_shadow_debug_turn(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            trace_store = TraceStore(tmpdir)
            session = store.load("main")
            session.messages.append(
                ChatMessage(
                    role="assistant",
                    content="existing reply",
                    created_at="2026-03-18T00:00:00+00:00",
                    model_alias="local_chat",
                )
            )
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/trace", arg="帮我看看今天原油价格走势"),
                config=config,
                store=store,
                chat_service=self.build_debug_chat_service(config, trace_store=trace_store),
                trace_store=trace_store,
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertEqual(len(session.messages), 1)
        self.assertEqual(session.messages[0].content, "existing reply")
        self.assertIn("Trace Debug", rendered)
        self.assertIn("Payload Debug", rendered)
        self.assertIn("\"messages\"", rendered)
        self.assertIn("Execution Timeline", rendered)
        self.assertIn("completion_request", rendered)
        self.assertIn("Final Reply", rendered)
        self.assertIn("debug answer", rendered)

    def test_trace_command_with_tool_call_renders_request_trace(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            trace_store = TraceStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/trace", arg="我想要再刺激一点"),
                config=config,
                store=store,
                chat_service=self.build_cup_chat_service(
                    config,
                    client=ToolTraceClient(),
                    trace_store=trace_store,
                ),
                trace_store=trace_store,
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("Execution Timeline", rendered)
        self.assertIn("Tool Request Trace", rendered)
        self.assertIn("control_cup", rendered)
        self.assertIn(f"{CUP_DEFAULT_API_BASE_URL}/motor", rendered)
        self.assertIn("已经继续增强了一点", rendered)

    def test_retry_command_replaces_last_assistant_reply(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            session.messages.extend(
                [
                    ChatMessage(
                        role="user",
                        content="hello",
                        created_at="2026-03-18T00:00:00+00:00",
                    ),
                    ChatMessage(
                        role="assistant",
                        content="old reply",
                        created_at="2026-03-18T00:00:01+00:00",
                        model_alias="local_chat",
                    ),
                ]
            )
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/retry"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertEqual(session.messages[-1].content, "new reply")
        self.assertEqual(len(session.messages), 2)
        self.assertIn("Retrying the last user message...", rendered)
        self.assertIn("new", rendered)

    def test_retry_command_handles_keyboard_interrupt_without_crash(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            session.messages.extend(
                [
                    ChatMessage(
                        role="user",
                        content="hello",
                        created_at="2026-03-18T00:00:00+00:00",
                    ),
                    ChatMessage(
                        role="assistant",
                        content="old reply",
                        created_at="2026-03-18T00:00:01+00:00",
                        model_alias="local_chat",
                    ),
                ]
            )
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/retry"),
                config=config,
                store=store,
                chat_service=self.build_interrupting_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("Generation interrupted. Partial reply saved.", rendered)
        self.assertEqual(session.messages[-1].content, "partial")

    def test_run_streaming_turn_handles_keyboard_interrupt_during_tool_probe(self) -> None:
        config = build_config()
        config.scheduler.search_model = "local_chat"
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            run_streaming_turn(
                self.build_interrupting_tool_probe_service(config),
                store,
                session,
                user_text="帮我查一下最新 release notes",
                mode_override="search",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertIn("Request interrupted before the provider finished responding.", rendered)
        self.assertEqual(session.messages[-1].role, "user")

    def test_run_streaming_turn_renders_ask_user_quick_options(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()
            from whesper.tools import ToolSpec

            service = ChatService(
                config,
                client=AskUserCliClient(),
                tool_registry=ToolRegistry(
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
                            handler=lambda arguments: {"ok": True},
                        ),
                    )
                ),
            )

            run_streaming_turn(
                service,
                store,
                session,
                user_text="帮我查天气",
                mode_override="chat",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertIn("你想查哪个城市？", rendered)
        self.assertIn("Quick options:", rendered)
        self.assertIn("1. 上海", rendered)
        self.assertIn("2. 东京", rendered)
        self.assertIn("Enter a number or type your own answer to continue.", rendered)
        self.assertIsNotNone(session.pending_ask_user)

    def test_copy_last_prints_last_assistant_reply(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            session.messages.append(
                ChatMessage(
                    role="assistant",
                    content="copied text",
                    created_at="2026-03-18T00:00:00+00:00",
                )
            )
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/copy-last"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("copied text", rendered)

    def test_rename_command_updates_session(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/rename", arg="renamed"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

            sessions = store.list_sessions()

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertEqual(outcome.session.session_id, "renamed")
        self.assertIn("Renamed session to: renamed", rendered)
        self.assertIn("renamed", sessions)

    def test_session_no_arg_prints_current_session_id(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/session", arg=None),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertFalse(outcome.should_exit)
        self.assertEqual(outcome.session.session_id, "main")
        self.assertIn("Current session: main", rendered)
        self.assertIn("messages:", rendered)
        self.assertIn("/session <session_id>", rendered)

    def test_session_command_switches_to_existing_session(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            store.load("other")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/session", arg="other"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertEqual(outcome.session.session_id, "other")
        self.assertIn("Switched to session: other", rendered)

    def test_session_command_errors_when_session_not_found(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/session", arg="nonexistent"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertEqual(outcome.session.session_id, "main")
        self.assertIn("Session not found: nonexistent", rendered)
        self.assertIn("available sessions:", rendered)
        self.assertIn("/new", rendered)

    def test_delete_session_removes_named_session(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            session = store.load("main")
            store.load("throwaway")
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/delete-session", arg="throwaway"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                session=session,
                mode_override="auto",
                output_stream=output,
            )

            sessions = store.list_sessions()

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn("Deleted session: throwaway", rendered)
        self.assertNotIn("throwaway", sessions)

    def test_build_footer_meta_includes_development_details(self) -> None:
        config = build_config()
        session = ConversationSession(
            session_id="main",
            created_at="2026-03-18T00:00:00+00:00",
            updated_at="2026-03-18T00:00:01+00:00",
            messages=[
                ChatMessage(
                    role="user",
                    content="hello",
                    created_at="2026-03-18T00:00:00+00:00",
                ),
                ChatMessage(
                    role="assistant",
                    content="hello back",
                    created_at="2026-03-18T00:00:01+00:00",
                    model_alias="local_chat",
                ),
            ],
        )
        result = ChatTurnResult(
            decision=RouteDecision(
                model_alias="local_chat",
                mode="chat",
                reason="default chat model",
            ),
            assistant_message=session.messages[-1],
        )

        footer = build_footer_meta(
            config,
            result,
            total_seconds=4.8,
            first_token_seconds=1.2,
            session=session,
        )

        self.assertIn("done in 4.8s", footer)
        self.assertIn("first token 1.2s", footer)
        self.assertIn("model local_chat -> qwen", footer)
        self.assertIn("provider local/ollama_native", footer)
        self.assertIn("route chat", footer)
        self.assertIn("reason default chat model", footer)
        self.assertIn("10 chars", footer)
        self.assertIn("2 msgs", footer)

    def test_remember_and_memory_commands_round_trip_saved_memory(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            memory_store = MemoryStore(tmpdir)
            session = store.load("main")
            output = io.StringIO()

            remember_outcome = handle_command(
                ParsedCommand(name="/remember", arg="I like jasmine tea."),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                memory_store=memory_store,
                session=session,
                mode_override="auto",
                output_stream=output,
            )

            memory_outcome = handle_command(
                ParsedCommand(name="/memory"),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                memory_store=memory_store,
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(remember_outcome.handled)
        self.assertTrue(memory_outcome.handled)
        self.assertIn("Saved memory:", rendered)
        self.assertIn("I like jasmine tea.", rendered)
        self.assertIn("profile_memory", rendered)

    def test_forget_command_removes_saved_memory(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            memory_store = MemoryStore(tmpdir)
            session = store.load("main")
            saved = memory_store.remember(
                memory_type="profile_memory",
                title="Saved Note",
                content="The user likes jasmine tea.",
                source="manual_command",
                confidence=1.0,
                session_id=session.session_id,
            )
            output = io.StringIO()

            outcome = handle_command(
                ParsedCommand(name="/forget", arg=saved.memory_id),
                config=config,
                store=store,
                chat_service=self.build_chat_service(config),
                memory_store=memory_store,
                session=session,
                mode_override="auto",
                output_stream=output,
            )

        rendered = output.getvalue()
        self.assertTrue(outcome.handled)
        self.assertIn(f"Forgot memory: {saved.memory_id}", rendered)
        self.assertEqual(memory_store.list_memories(include_expired=True), [])


if __name__ == "__main__":
    unittest.main()
