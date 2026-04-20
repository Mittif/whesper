from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Callable

from whesper.agent_harness import (
    AGENTIC_PLANNING_PROMPT,
    AgentHarness,
    StepCallback,
    TOOLLESS_CONTINUE_PROMPT,
    parse_ask_user_action,
)
from whesper.agent_types import AskUserAction, AskUserOption
from whesper.agent_types import ToolInvocation
from whesper.client import CompletionResult, OpenAICompatibleClient, ProviderError, ToolCall
from whesper.config import AppConfig
from whesper.live_data import LiveContextService
from whesper.message_builder import SessionMessageBuilder
from whesper.memory import MemoryService
from whesper.provider_profile import ProviderProfile, resolve_profile
from whesper.router import RouteDecision, select_model
from whesper.session import ChatMessage, ConversationSession, utc_now_iso
from whesper.tool_executor import ToolExecutor
from whesper.tool_protocol import DefaultToolProtocolAdapter, sanitize_tool_call_artifacts
from whesper.trace import TraceStore, make_trace_event
from whesper.tools import (
    ToolRegistry,
    ToolExecutionError,
    _matches_cup_control_intent,
    _matches_cup_scene_followup,
)


@dataclass(slots=True)
class ChatTurnResult:
    decision: RouteDecision
    assistant_message: ChatMessage
    ask_user: AskUserAction | None = None


@dataclass(slots=True)
class CupPreflightResult:
    prompt: str | None = None
    status_payload: dict[str, object] | None = None


@dataclass(slots=True)
class CupStatusSnapshot:
    connected: bool
    motor_target: float | None
    motor_enabled: bool | None
    led_color: str | None
    led_blink_hz: float | None
    led_blink_mode: int | None


class GenerationInterrupted(RuntimeError):
    def __init__(self, partial_result: ChatTurnResult | None = None) -> None:
        super().__init__("Generation interrupted.")
        self.partial_result = partial_result


class ChatService:
    MAX_AGENT_ROUNDS = 10

    def __init__(
        self,
        config: AppConfig,
        client: OpenAICompatibleClient | None = None,
        memory_service: MemoryService | None = None,
        live_context_service: LiveContextService | None = None,
        tool_registry: ToolRegistry | None = None,
        trace_store: TraceStore | None = None,
    ) -> None:
        self.config = config
        self.client = client or OpenAICompatibleClient()
        self.memory_service = memory_service
        self.live_context_service = live_context_service or LiveContextService.from_config(config)
        self.tool_registry = tool_registry or ToolRegistry.default(config)
        self.tool_protocol_adapter = DefaultToolProtocolAdapter()
        self.message_builder = SessionMessageBuilder(
            config,
            memory_service=self.memory_service,
            live_context_service=self.live_context_service,
            tool_registry=self.tool_registry,
        )
        self.tool_executor = ToolExecutor(self.tool_registry)
        self.trace_store = trace_store
        self._tool_unsupported_providers: set[str] = set()
        self.harness = AgentHarness(
            message_builder=self.message_builder,
            tool_executor=self.tool_executor,
            completion_requester=self._client_create_chat_completion,
            tool_call_payload_builder=self._tool_call_payload,
            tool_choice_builder=self._tool_choice_for_request,
            reasoning_content_resolver=self._tool_message_reasoning_content,
            trace_emitter=self._append_trace,
            tool_result_observer=self._capture_tool_result_memory,
            tool_call_guard=self._guard_tool_call,
            max_rounds=self.MAX_AGENT_ROUNDS,
        )

    def send(
        self,
        session: ConversationSession,
        user_text: str,
        *,
        mode_override: str = "auto",
        on_step: StepCallback | None = None,
    ) -> ChatTurnResult:
        resolved_user_text = self._resolve_pending_ask_response(session, user_text)
        self._append_user_message(session, resolved_user_text)
        session.pending_ask_user = None
        if self.memory_service is not None:
            self.memory_service.capture_user_message(session.session_id, resolved_user_text)
        return self._complete_turn(
            session,
            resolved_user_text,
            mode_override=mode_override,
            on_step=on_step,
        )

    def send_stream(
        self,
        session: ConversationSession,
        user_text: str,
        *,
        mode_override: str = "auto",
        on_chunk: Callable[[str], None] | None = None,
        on_step: StepCallback | None = None,
    ) -> ChatTurnResult:
        resolved_user_text = self._resolve_pending_ask_response(session, user_text)
        self._append_user_message(session, resolved_user_text)
        session.pending_ask_user = None
        if self.memory_service is not None:
            self.memory_service.capture_user_message(session.session_id, resolved_user_text)
        return self._complete_turn_stream(
            session,
            resolved_user_text,
            mode_override=mode_override,
            on_chunk=on_chunk,
            on_step=on_step,
        )

    def retry_stream(
        self,
        session: ConversationSession,
        *,
        mode_override: str = "auto",
        on_chunk: Callable[[str], None] | None = None,
        on_step: StepCallback | None = None,
    ) -> ChatTurnResult:
        last_user_index = self._last_user_index(session)
        if last_user_index is None:
            raise ValueError("No user message is available to retry yet.")

        last_user_message = session.messages[last_user_index]
        del session.messages[last_user_index + 1 :]
        transcript_last_user_index = self._last_user_index_in_messages(session.transcript_messages)
        if transcript_last_user_index is not None:
            del session.transcript_messages[transcript_last_user_index + 1 :]
        self._sync_session_updated_at(session)
        return self._complete_turn_stream(
            session,
            last_user_message.content,
            mode_override=mode_override,
            on_chunk=on_chunk,
            on_step=on_step,
        )

    def _complete_turn(
        self,
        session: ConversationSession,
        user_text: str,
        *,
        mode_override: str = "auto",
        on_step: StepCallback | None = None,
    ) -> ChatTurnResult:
        turn_start_index = len(session.messages)
        decision = select_model(
            self.config,
            user_text,
            pinned_model=session.pinned_model,
            mode_override=mode_override,
        )

        model_config = self.config.get_model(decision.model_alias)
        provider_config = self.config.get_provider(model_config.provider)
        target_profile = resolve_profile(provider_config, model_config)
        cup_preflight = self._cup_preflight(
            session_id=session.session_id,
            decision=decision,
            provider_name=provider_config.name,
        )
        cup_preflight_prompt = cup_preflight.prompt
        tools = self._tools_for_request(
            session=session,
            provider=provider_config,
            model=model_config,
            user_text=user_text,
            route_mode=decision.mode,
        )

        needs_reasoning = self._requires_reasoning_content_replay(target_profile, model_config)
        messages = self._build_messages(
            session,
            model_config.system_prompt,
            target_profile=target_profile,
            user_text=user_text,
            route_mode=decision.mode,
            preflight_prompt=cup_preflight_prompt,
            include_live_context=tools is None,
            include_tool_prompt=tools is not None,
            ensure_reasoning_content=needs_reasoning,
        )
        if tools is not None:
            run_result = self.harness.run_until_final(
                session,
                decision,
                provider=provider_config,
                model=model_config,
                model_system_prompt=model_config.system_prompt,
                user_text=user_text,
                route_mode=decision.mode,
                tools=tools,
                target_profile=target_profile,
                preflight_prompt=cup_preflight_prompt,
                on_step=on_step,
                ensure_reasoning_content=needs_reasoning,
            )
            completion = run_result.completion
            ask_user = run_result.ask_user
        else:
            completion = self._client_create_chat_completion(
                session_id=session.session_id,
                decision=decision,
                provider=provider_config,
                model=model_config,
                messages=messages,
                streamed=False,
            )
            if self.harness.should_continue_tool_loop(
                completion.content, route_mode=decision.mode, round_index=0,
            ):
                messages = [
                    *messages,
                    {"role": "assistant", "content": completion.content},
                    {"role": "user", "content": TOOLLESS_CONTINUE_PROMPT},
                ]
                completion = self._client_create_chat_completion(
                    session_id=session.session_id,
                    decision=decision,
                    provider=provider_config,
                    model=model_config,
                    messages=messages,
                    streamed=False,
                )
            ask_user = parse_ask_user_action(completion.content)
        turn_result = self._append_assistant_message(
            session,
            decision,
            ask_user.prompt if ask_user is not None else completion.content,
            ask_user=ask_user,
            target_profile=target_profile,
        )
        return self._finalize_cup_turn_feedback(
            session,
            user_text=user_text,
            turn_start_index=turn_start_index,
            preflight_status_payload=cup_preflight.status_payload,
            turn_result=turn_result,
        )

    def _complete_turn_stream(
        self,
        session: ConversationSession,
        user_text: str,
        *,
        mode_override: str = "auto",
        on_chunk: Callable[[str], None] | None = None,
        on_step: StepCallback | None = None,
    ) -> ChatTurnResult:
        turn_start_index = len(session.messages)
        decision = select_model(
            self.config,
            user_text,
            pinned_model=session.pinned_model,
            mode_override=mode_override,
        )

        model_config = self.config.get_model(decision.model_alias)
        provider_config = self.config.get_provider(model_config.provider)
        target_profile = resolve_profile(provider_config, model_config)
        cup_preflight = self._cup_preflight(
            session_id=session.session_id,
            decision=decision,
            provider_name=provider_config.name,
        )
        cup_preflight_prompt = cup_preflight.prompt
        tools = self._tools_for_request(
            session=session,
            provider=provider_config,
            model=model_config,
            user_text=user_text,
            route_mode=decision.mode,
        )
        needs_reasoning = self._requires_reasoning_content_replay(target_profile, model_config)
        messages = self._build_messages(
            session,
            model_config.system_prompt,
            target_profile=target_profile,
            user_text=user_text,
            route_mode=decision.mode,
            preflight_prompt=cup_preflight_prompt,
            include_live_context=tools is None,
            include_tool_prompt=tools is not None,
            ensure_reasoning_content=needs_reasoning,
        )
        if tools is None:
            result = self._stream_final_answer(
                session,
                decision,
                provider=provider_config,
                model=model_config,
                messages=messages,
                on_chunk=on_chunk,
                target_profile=target_profile,
            )
            if self.harness.should_continue_tool_loop(
                result.assistant_message.content, route_mode=decision.mode, round_index=0,
            ):
                messages = [
                    *messages,
                    {"role": "assistant", "content": result.assistant_message.content},
                    {"role": "user", "content": TOOLLESS_CONTINUE_PROMPT},
                ]
                if on_chunk is not None:
                    on_chunk("\n\n")
                result = self._stream_final_answer(
                    session,
                    decision,
                    provider=provider_config,
                    model=model_config,
                    messages=messages,
                    on_chunk=on_chunk,
                    target_profile=target_profile,
                )
            return self._finalize_cup_turn_feedback(
                session,
                user_text=user_text,
                turn_start_index=turn_start_index,
                preflight_status_payload=cup_preflight.status_payload,
                turn_result=result,
                on_chunk=on_chunk,
            )
        run_result = self.harness.run_until_final(
            session,
            decision,
            provider=provider_config,
            model=model_config,
            model_system_prompt=model_config.system_prompt,
            user_text=user_text,
            route_mode=decision.mode,
            tools=tools,
            target_profile=target_profile,
            preflight_prompt=cup_preflight_prompt,
            on_step=on_step,
            ensure_reasoning_content=needs_reasoning,
        )
        if run_result.final_messages is not None:
            result = self._stream_final_answer(
                session,
                decision,
                provider=provider_config,
                model=model_config,
                messages=run_result.final_messages,
                on_chunk=on_chunk,
                target_profile=target_profile,
            )
            return self._finalize_cup_turn_feedback(
                session,
                user_text=user_text,
                turn_start_index=turn_start_index,
                preflight_status_payload=cup_preflight.status_payload,
                turn_result=result,
                on_chunk=on_chunk,
            )
        completion = run_result.completion
        ask_user = run_result.ask_user
        if ask_user is not None:
            if on_chunk is not None and ask_user.prompt:
                on_chunk(ask_user.prompt)
            result = self._append_assistant_message(
                session,
                decision,
                ask_user.prompt,
                ask_user=ask_user,
                target_profile=target_profile,
            )
            return self._finalize_cup_turn_feedback(
                session,
                user_text=user_text,
                turn_start_index=turn_start_index,
                preflight_status_payload=cup_preflight.status_payload,
                turn_result=result,
                on_chunk=on_chunk,
            )
        if on_chunk is not None and completion.content:
            on_chunk(completion.content)
        result = self._append_assistant_message(
            session,
            decision,
            completion.content,
            target_profile=target_profile,
        )
        return self._finalize_cup_turn_feedback(
            session,
            user_text=user_text,
            turn_start_index=turn_start_index,
            preflight_status_payload=cup_preflight.status_payload,
            turn_result=result,
            on_chunk=on_chunk,
        )

    def _last_user_index(self, session: ConversationSession) -> int | None:
        return self._last_user_index_in_messages(session.messages)

    def _last_user_index_in_messages(self, messages: list[ChatMessage]) -> int | None:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == "user":
                return index
        return None

    def _sync_session_updated_at(self, session: ConversationSession) -> None:
        if session.messages:
            session.updated_at = session.messages[-1].created_at
            return
        session.updated_at = session.created_at

    def _resolve_pending_ask_response(
        self,
        session: ConversationSession,
        user_text: str,
    ) -> str:
        pending = session.pending_ask_user
        if pending is None:
            return user_text
        normalized = user_text.strip()
        if normalized.isdigit():
            index = int(normalized) - 1
            if 0 <= index < len(pending.options):
                return pending.options[index].value
        return user_text

    def _append_user_message(self, session: ConversationSession, user_text: str) -> ChatMessage:
        user_message = ChatMessage(
            role="user",
            content=user_text,
            created_at=utc_now_iso(),
        )
        session.append(user_message)
        session.append_transcript(
            ChatMessage(
                role="user",
                content=user_text,
                created_at=user_message.created_at,
            )
        )
        return user_message

    def _build_messages(
        self,
        session: ConversationSession,
        model_system_prompt: str | None,
        *,
        target_profile: ProviderProfile,
        user_text: str,
        route_mode: str,
        preflight_prompt: str | None = None,
        include_planning_prompt: bool = True,
        include_live_context: bool = True,
        include_tool_prompt: bool = True,
        ensure_reasoning_content: bool = False,
    ) -> list[dict[str, object]]:
        planning_prompt = AGENTIC_PLANNING_PROMPT if include_planning_prompt else None
        return self.message_builder.build_messages(
            session,
            model_system_prompt,
            target_profile=target_profile,
            user_text=user_text,
            route_mode=route_mode,
            planning_prompt=planning_prompt,
            preflight_prompt=preflight_prompt,
            include_live_context=include_live_context,
            include_tool_prompt=include_tool_prompt,
            ensure_reasoning_content=ensure_reasoning_content,
        )

    def _cup_preflight(
        self,
        *,
        session_id: str,
        decision: RouteDecision,
        provider_name: str,
    ) -> CupPreflightResult:
        if self.tool_registry.spec_for_name("control_cup") is None:
            return CupPreflightResult()
        try:
            result = self.tool_executor.execute(
                ToolInvocation(
                    tool_call_id=f"cup-preflight-{time.time_ns()}",
                    name="control_cup",
                    arguments_json='{"action":"status"}',
                )
            )
        except ToolExecutionError as exc:
            self._append_trace(
                kind="cup_preflight",
                session_id=session_id,
                decision=decision,
                provider_name=provider_name,
                streamed=False,
                tools_enabled=True,
                note="disconnected",
                preview=str(exc),
            )
            return CupPreflightResult()

        try:
            payload = json.loads(result.content)
        except json.JSONDecodeError:
            self._append_trace(
                kind="cup_preflight",
                session_id=session_id,
                decision=decision,
                provider_name=provider_name,
                streamed=False,
                tools_enabled=True,
                note="invalid_payload",
                preview=result.content,
            )
            return CupPreflightResult()
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            self._append_trace(
                kind="cup_preflight",
                session_id=session_id,
                decision=decision,
                provider_name=provider_name,
                streamed=False,
                tools_enabled=True,
                note="not_connected",
                preview=result.content,
            )
            return CupPreflightResult()

        preflight_prompt = self._cup_connected_preflight_prompt(payload)
        self._append_trace(
            kind="cup_preflight",
            session_id=session_id,
            decision=decision,
            provider_name=provider_name,
            streamed=False,
            tools_enabled=True,
            note="connected",
            preview=preflight_prompt,
        )
        return CupPreflightResult(
            prompt=preflight_prompt,
            status_payload=payload,
        )

    def _cup_preflight_prompt(
        self,
        *,
        session_id: str,
        decision: RouteDecision,
        provider_name: str,
    ) -> str | None:
        return self._cup_preflight(
            session_id=session_id,
            decision=decision,
            provider_name=provider_name,
        ).prompt

    def _cup_connected_preflight_prompt(self, payload: dict[str, object]) -> str:
        device = payload.get("device")
        motor_snapshot = payload.get("motor")
        motor_target = self._first_numeric_value(
            motor_snapshot,
            keys=("target", "target_velocity", "vel", "velocity"),
        )
        motor_enabled = self._first_bool_value(
            motor_snapshot,
            keys=("enabled", "running", "on"),
        )

        lines = [
            "CUP preflight: device connection check succeeded via status API right before this turn.",
            "CUP is currently reachable, so map semantically similar ambiguous intent to possible CUP control actions when context fits.",
            "Semantic mapping hints: '刺激一点/更猛/再快一点' -> nudge_intensity up; '温柔点/慢一点/缓一缓' -> nudge_intensity down; mood/light color requests -> set_led or apply_scene.",
            "If the user appears to only chat (not control hardware), keep it conversational and do not force tool calls.",
        ]
        if isinstance(device, str) and device.strip():
            lines.append(f"- connected_device: {device.strip()}")
        if motor_target is not None:
            lines.append(f"- current_motor_target_velocity: {motor_target:g}")
        if motor_enabled is not None:
            lines.append(
                f"- current_motor_enabled: {'true' if motor_enabled else 'false'}"
            )
        return "\n".join(lines)

    @staticmethod
    def _first_numeric_value(
        payload: object,
        *,
        keys: tuple[str, ...],
    ) -> float | None:
        if not isinstance(payload, dict):
            return None
        for key in keys:
            value = payload.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                normalized = value.strip()
                if not normalized:
                    continue
                try:
                    return float(normalized)
                except ValueError:
                    continue
        return None

    @staticmethod
    def _first_bool_value(
        payload: object,
        *,
        keys: tuple[str, ...],
    ) -> bool | None:
        if not isinstance(payload, dict):
            return None
        for key in keys:
            value = payload.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, int) and value in {0, 1}:
                return bool(value)
            if isinstance(value, str):
                normalized = value.strip().casefold()
                if normalized in {"true", "1", "on", "enabled", "running"}:
                    return True
                if normalized in {"false", "0", "off", "disabled", "stopped"}:
                    return False
        return None

    def _capture_tool_result_memory(
        self,
        session_id: str,
        tool_name: str,
        content: str,
    ) -> None:
        if self.memory_service is None:
            return
        self.memory_service.capture_tool_result(session_id, tool_name, content)

    def _guard_tool_call(
        self,
        session: ConversationSession,
        tool_call: ToolCall,
        user_text: str,
        route_mode: str,
    ) -> AskUserAction | None:
        spec = self.tool_registry.spec_for_name(tool_call.name)
        if spec is None or not spec.execution_meta.side_effectful:
            return None
        if tool_call.name == "control_cup":
            if self._should_require_cup_followup_tool_choice(session, user_text):
                return None
            return self._confirm_cup_control_action(tool_call, user_text)
        if spec.should_offer is None or spec.should_offer(user_text, route_mode):
            return None
        return self._confirm_side_effectful_action(tool_call.name, user_text)

    def _confirm_cup_control_action(
        self,
        tool_call: ToolCall,
        user_text: str,
    ) -> AskUserAction:
        normalized_request = " ".join(user_text.split()) or "刚才那句请求"
        confirm_value = self._cup_confirmation_value(tool_call, normalized_request)
        return AskUserAction(
            prompt="你是想让我直接调节 CUP 设备吗？如果确认，我就按你刚才的意思执行。",
            options=(
                AskUserOption(
                    label="确认执行",
                    value=confirm_value,
                    description="立即按刚才的意思控制设备",
                ),
                AskUserOption(
                    label="先看状态",
                    value="先查看 CUP 设备当前状态，再告诉我。",
                    description="先查状态，不直接改动设备",
                ),
                AskUserOption(
                    label="先别执行",
                    value="先不要控制 CUP 设备，只用文字回复我。",
                    description="不改设备，只继续聊天",
                ),
            ),
            allow_free_text=True,
            field_name="cup_control_confirmation",
        )

    def _cup_confirmation_value(
        self,
        tool_call: ToolCall,
        normalized_request: str,
    ) -> str:
        try:
            arguments = tool_call.arguments()
        except Exception:
            arguments = {}
        action = str(arguments.get("action", "")).strip().casefold()
        if action == "status":
            return f"请直接查看 CUP 设备状态，按我刚才这句理解：{normalized_request}"
        if action == "set_led":
            return f"请直接调整 CUP 设备灯光，按我刚才这句执行：{normalized_request}"
        if action == "set_motor":
            target = arguments.get("target_velocity")
            if isinstance(target, (int, float)):
                return f"请直接把 CUP 速度设到 {target:g}，并执行。"
            return f"请直接设置 CUP 速度，按我刚才这句执行：{normalized_request}"
        if action == "stop":
            return "请直接停止 CUP 设备。"
        if action == "apply_scene":
            scene = str(arguments.get("scene", "")).strip()
            if scene:
                return f"请直接把 CUP 切到 {scene} 模式。"
            return f"请直接切换 CUP 模式，按我刚才这句执行：{normalized_request}"
        if action == "nudge_intensity":
            direction = str(arguments.get("direction", "")).strip().casefold()
            if direction == "up":
                return f"请直接把 CUP 调快一点、更刺激一点，按我刚才这句执行：{normalized_request}"
            if direction == "down":
                return f"请直接把 CUP 放缓一点、轻一点，按我刚才这句执行：{normalized_request}"
            return f"请直接调整 CUP 强度，按我刚才这句执行：{normalized_request}"
        return f"请直接控制 CUP 设备，按我刚才这句执行：{normalized_request}"

    def _confirm_side_effectful_action(
        self,
        tool_name: str,
        user_text: str,
    ) -> AskUserAction:
        normalized_request = " ".join(user_text.split()) or "刚才那句请求"
        return AskUserAction(
            prompt=f"你是想让我直接执行 {tool_name} 这个操作吗？",
            options=(
                AskUserOption(
                    label="确认执行",
                    value=f"请直接执行 {tool_name}，按我刚才这句执行：{normalized_request}",
                    description="继续执行这个操作",
                ),
                AskUserOption(
                    label="先别执行",
                    value=f"先不要执行 {tool_name}，只用文字回复我。",
                    description="先不触发外部操作",
                ),
            ),
            allow_free_text=True,
            field_name="side_effect_confirmation",
        )

    def _tool_call_payload(self, tool_call: ToolCall) -> dict[str, object]:
        return self.tool_protocol_adapter.tool_call_payload(tool_call)

    def _client_create_chat_completion(
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
        allow_disable_thinking: bool = True,
        force_disable_thinking: bool = False,
    ) -> CompletionResult:
        messages = self.tool_protocol_adapter.messages_with_normalized_tool_call_ids(messages)
        self._append_trace(
            kind="completion_request",
            session_id=session_id,
            decision=decision,
            provider_name=provider.name,
            streamed=streamed,
            tools_enabled=tools is not None,
            tool_choice=self._stringify_tool_choice(tool_choice),
            preview=self._trace_preview_from_messages(messages),
        )
        try:
            request_kwargs: dict[str, object] = {}
            if tools is not None:
                request_kwargs["tools"] = tools
            if tool_choice is not None:
                request_kwargs["tool_choice"] = tool_choice
            if force_disable_thinking:
                request_kwargs["disable_thinking"] = True
            elif allow_disable_thinking and self._should_disable_thinking_for_request(
                provider, model, tools
            ):
                request_kwargs["disable_thinking"] = True
            completion = self.client.create_chat_completion(
                provider=provider,
                model=model,
                messages=messages,
                **request_kwargs,
            )
            self._append_trace(
                kind="completion_response",
                session_id=session_id,
                decision=decision,
                provider_name=provider.name,
                streamed=streamed,
                tools_enabled=tools is not None,
                tool_choice=self._stringify_tool_choice(tool_choice),
                note=f"tool_calls={len(completion.tool_calls)}",
                preview=completion.content or self._tool_call_names_preview(completion.tool_calls),
            )
            return completion
        except TypeError as exc:
            message = str(exc)
            if (
                request_kwargs.get("disable_thinking")
                and "unexpected keyword argument 'disable_thinking'" in message
            ):
                self._append_trace(
                    kind="fallback",
                    session_id=session_id,
                    decision=decision,
                    provider_name=provider.name,
                    streamed=streamed,
                    tools_enabled=tools is not None,
                    tool_choice=self._stringify_tool_choice(tool_choice),
                    note="client does not accept disable_thinking; retrying without it",
                )
                return self._client_create_chat_completion(
                    session_id=session_id,
                    decision=decision,
                    provider=provider,
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    streamed=streamed,
                    structured_tool_arguments=structured_tool_arguments,
                    allow_disable_thinking=False,
                )
            if tool_choice is not None and "unexpected keyword argument 'tool_choice'" in message:
                self._append_trace(
                    kind="fallback",
                    session_id=session_id,
                    decision=decision,
                    provider_name=provider.name,
                    streamed=streamed,
                    tools_enabled=tools is not None,
                    tool_choice=self._stringify_tool_choice(tool_choice),
                    note="client does not accept tool_choice; retrying without it",
                )
                return self._client_create_chat_completion(
                    session_id=session_id,
                    decision=decision,
                    provider=provider,
                    model=model,
                    messages=messages,
                    tools=tools,
                    streamed=streamed,
                    structured_tool_arguments=structured_tool_arguments,
                )
            if "unexpected keyword argument 'tools'" not in message:
                raise
            self._tool_unsupported_providers.add(provider.name)
            self._append_trace(
                kind="fallback",
                session_id=session_id,
                decision=decision,
                provider_name=provider.name,
                streamed=streamed,
                tools_enabled=tools is not None,
                tool_choice=self._stringify_tool_choice(tool_choice),
                note="client does not accept tools; retrying without tools",
            )
            return self._client_create_chat_completion(
                session_id=session_id,
                decision=decision,
                provider=provider,
                model=model,
                messages=messages,
                tools=tools,
                streamed=streamed,
                structured_tool_arguments=structured_tool_arguments,
            )
        except ProviderError as exc:
            if (
                tool_choice is not None
                and tools is not None
                and "tool_choice" in str(exc).casefold()
            ):
                error_text = str(exc).casefold()
                if (
                    not request_kwargs.get("disable_thinking")
                    and "thinking enabled" in error_text
                ):
                    self._append_trace(
                        kind="fallback",
                        session_id=session_id,
                        decision=decision,
                        provider_name=provider.name,
                        streamed=streamed,
                        tools_enabled=True,
                        tool_choice=self._stringify_tool_choice(tool_choice),
                        note=(
                            "provider rejected tool_choice with thinking enabled; "
                            f"retrying with disable_thinking ({exc})"
                        ),
                    )
                    return self._client_create_chat_completion(
                        session_id=session_id,
                        decision=decision,
                        provider=provider,
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        streamed=streamed,
                        structured_tool_arguments=structured_tool_arguments,
                        allow_disable_thinking=False,
                        force_disable_thinking=True,
                    )
                self._append_trace(
                    kind="fallback",
                    session_id=session_id,
                    decision=decision,
                    provider_name=provider.name,
                    streamed=streamed,
                    tools_enabled=True,
                    tool_choice=self._stringify_tool_choice(tool_choice),
                    note=f"provider rejected tool_choice; retrying without it ({exc})",
                )
                return self._client_create_chat_completion(
                    session_id=session_id,
                    decision=decision,
                    provider=provider,
                    model=model,
                    messages=messages,
                    tools=tools,
                    streamed=streamed,
                    structured_tool_arguments=structured_tool_arguments,
                )
            if (
                not structured_tool_arguments
                and not tools
                and self._should_retry_with_structured_tool_arguments(provider, messages, exc)
            ):
                normalized_messages = self._messages_with_structured_tool_arguments(messages)
                self._append_trace(
                    kind="fallback",
                    session_id=session_id,
                    decision=decision,
                    provider_name=provider.name,
                    streamed=streamed,
                    tools_enabled=False,
                    note="provider rejected string tool arguments; retrying with structured arguments",
                )
                return self._client_create_chat_completion(
                    session_id=session_id,
                    decision=decision,
                    provider=provider,
                    model=model,
                    messages=normalized_messages,
                    streamed=streamed,
                    structured_tool_arguments=True,
                )
            if not tools:
                raise
            if not self._should_retry_without_tools(provider, exc):
                raise
            self._tool_unsupported_providers.add(provider.name)
            self._append_trace(
                kind="fallback",
                session_id=session_id,
                decision=decision,
                provider_name=provider.name,
                streamed=streamed,
                tools_enabled=True,
                tool_choice=self._stringify_tool_choice(tool_choice),
                note=f"provider rejected tools; retrying without tools ({exc})",
            )
            return self._client_create_chat_completion(
                session_id=session_id,
                decision=decision,
                provider=provider,
                model=model,
                messages=messages,
                streamed=streamed,
                structured_tool_arguments=structured_tool_arguments,
            )

    def _tools_for_request(
        self,
        *,
        session: ConversationSession | None = None,
        provider,
        model,
        user_text: str,
        route_mode: str,
    ) -> list[dict[str, object]] | None:
        if provider.name in self._tool_unsupported_providers:
            return None
        tools = list(
            self.tool_registry.openai_tools_for_request(
                user_text,
                route_mode=route_mode,
            )
        )
        if not tools and session is not None:
            tools = self._hardware_followup_tools_for_request(session, user_text)
        profile = resolve_profile(provider, model)
        if "$web_search" in profile.builtin_tools:
            tools = [
                tool
                for tool in tools
                if not (
                    isinstance(tool, dict)
                    and isinstance(tool.get("function"), dict)
                    and tool["function"].get("name") == "web_search"
                )
            ]
            tools.append(self._kimi_web_search_tool())
        if not tools:
            return None
        return tools

    def _hardware_followup_tools_for_request(
        self,
        session: ConversationSession,
        user_text: str,
    ) -> list[dict[str, object]]:
        tool_names: list[str] = []
        if self._should_require_cup_followup_tool_choice(session, user_text):
            tool_names.append("control_cup")
        if not tool_names:
            return []
        allowed_names = set(tool_names)
        return [
            tool
            for tool in self.tool_registry.openai_tools()
            if isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and tool["function"].get("name") in allowed_names
        ]

    def _tool_choice_for_request(
        self,
        session: ConversationSession,
        user_text: str,
        tools: list[dict[str, object]] | None,
        *,
        route_mode: str,
    ) -> str | dict[str, object] | None:
        if tools is None:
            return None
        default_choice = self.tool_registry.default_tool_choice(
            user_text=user_text,
            route_mode=route_mode,
        )
        if default_choice is not None:
            return default_choice
        if self._should_require_hardware_followup_tool_choice(session, user_text):
            return "required"
        return None

    def _should_require_hardware_followup_tool_choice(
        self,
        session: ConversationSession,
        user_text: str,
    ) -> bool:
        return self._should_require_cup_followup_tool_choice(session, user_text)

    def _should_require_cup_followup_tool_choice(
        self,
        session: ConversationSession,
        user_text: str,
    ) -> bool:
        if _matches_cup_control_intent(user_text):
            return True
        if not _matches_cup_scene_followup(user_text):
            return False
        recent_messages = session.messages[-6:]
        context_keywords = (
            "cup", "飞机杯", "motor", "转速", "震动", "振动", "马达", "电机",
            "灯", "led", "灯光", "颜色", "灯色",
        )
        for message in reversed(recent_messages):
            if message.role == "tool" and message.name == "control_cup":
                return True
            if message.role not in {"user", "assistant"}:
                continue
            normalized = (message.content or "").casefold()
            if not normalized:
                continue
            if _matches_cup_control_intent(message.content):
                return True
            if any(keyword in normalized for keyword in context_keywords):
                return True
        return False

    def _finalize_cup_turn_feedback(
        self,
        session: ConversationSession,
        *,
        user_text: str,
        turn_start_index: int,
        preflight_status_payload: dict[str, object] | None,
        turn_result: ChatTurnResult,
        on_chunk: Callable[[str], None] | None = None,
    ) -> ChatTurnResult:
        del user_text
        feedback = self._cup_turn_feedback_text(
            session,
            turn_start_index=turn_start_index,
            preflight_status_payload=preflight_status_payload,
        )
        if feedback is None:
            return turn_result
        current_content = turn_result.assistant_message.content.strip()
        next_content = feedback if not current_content else f"{current_content}\n\n{feedback}"
        turn_result.assistant_message.content = next_content
        self._update_transcript_assistant_content(
            session,
            assistant_message=turn_result.assistant_message,
            content=next_content,
        )
        if on_chunk is not None:
            on_chunk(f"\n\n{feedback}")
        model_config = self.config.get_model(turn_result.decision.model_alias)
        provider_config = self.config.get_provider(model_config.provider)
        self._append_trace(
            kind="cup_verification",
            session_id=session.session_id,
            decision=turn_result.decision,
            provider_name=provider_config.name,
            streamed=False,
            tools_enabled=True,
            preview=feedback,
        )
        return turn_result

    def _cup_turn_feedback_text(
        self,
        session: ConversationSession,
        *,
        turn_start_index: int,
        preflight_status_payload: dict[str, object] | None,
    ) -> str | None:
        control_tool_messages = [
            message
            for message in session.messages[turn_start_index:]
            if message.role == "tool" and message.name == "control_cup"
        ]
        if not control_tool_messages:
            return None

        parsed_payloads: list[dict[str, object]] = []
        raw_failures: list[str] = []
        for message in control_tool_messages:
            content = message.content.strip()
            if not content:
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                raw_failures.append(self._short_preview(content))
                continue
            if not isinstance(payload, dict):
                raw_failures.append(self._short_preview(content))
                continue
            parsed_payloads.append(payload)
            if payload.get("ok") is not True:
                raw_failures.append(
                    self._short_preview(
                        str(payload.get("error") or payload.get("message") or content)
                    )
                )

        side_effect_payloads = [
            payload
            for payload in parsed_payloads
            if isinstance(payload.get("action"), str)
            and str(payload.get("action")).strip().casefold() != "status"
        ]
        if not side_effect_payloads:
            if raw_failures:
                first_error = raw_failures[0]
                return (
                    "设备执行结果：执行失败。"
                    f"工具返回错误：{first_error}"
                )
            return None

        pre_snapshot = self._cup_status_snapshot_from_payload(preflight_status_payload)
        post_status_payload, post_error = self._cup_fetch_status_payload()
        post_snapshot = self._cup_status_snapshot_from_payload(post_status_payload)

        evaluations: list[tuple[str, str]] = []
        if raw_failures:
            evaluations.append(
                (
                    "failure",
                    f"工具返回错误：{raw_failures[0]}",
                )
            )
        for payload in side_effect_payloads:
            evaluations.append(
                self._evaluate_cup_action_payload(
                    payload,
                    pre_snapshot=pre_snapshot,
                    post_snapshot=post_snapshot,
                )
            )

        if post_error is not None:
            evaluations.append(("unknown", f"执行后状态读取失败：{post_error}"))

        statuses = {status for status, _ in evaluations}
        details = "; ".join(detail for _, detail in evaluations if detail)[:360]
        if "failure" in statuses:
            if details:
                return f"设备执行结果：未达预期。{details}"
            return "设备执行结果：未达预期。"
        if statuses == {"success"}:
            if details:
                return f"设备执行结果：执行前后状态校验通过。{details}"
            return "设备执行结果：执行前后状态校验通过。"
        if details:
            return f"设备执行结果：已发送指令，但状态校验信息不足。{details}"
        return "设备执行结果：已发送指令，但状态校验信息不足。"

    def _cup_fetch_status_payload(self) -> tuple[dict[str, object] | None, str | None]:
        if self.tool_registry.spec_for_name("control_cup") is None:
            return None, "control_cup tool unavailable"
        try:
            result = self.tool_executor.execute(
                ToolInvocation(
                    tool_call_id=f"cup-verify-{time.time_ns()}",
                    name="control_cup",
                    arguments_json='{"action":"status"}',
                )
            )
        except ToolExecutionError as exc:
            return None, self._short_preview(str(exc), limit=160)
        try:
            payload = json.loads(result.content)
        except json.JSONDecodeError:
            return None, "status payload is not valid JSON"
        if not isinstance(payload, dict):
            return None, "status payload is not an object"
        if payload.get("ok") is not True:
            return None, self._short_preview(str(payload), limit=160)
        return payload, None

    def _cup_status_snapshot_from_payload(
        self,
        payload: dict[str, object] | None,
    ) -> CupStatusSnapshot | None:
        if not isinstance(payload, dict):
            return None
        if payload.get("ok") is not True:
            return None
        system = payload.get("system")
        motor = payload.get("motor")
        connected = True
        if isinstance(system, dict):
            connected_value = self._first_bool_value(
                system,
                keys=("connected", "online", "reachable"),
            )
            if connected_value is not None:
                connected = connected_value
        motor_target = self._first_numeric_value(
            motor,
            keys=("target", "target_velocity", "vel", "velocity"),
        )
        motor_enabled = self._first_bool_value(
            motor,
            keys=("enabled", "running", "on"),
        )
        led_color = None
        led_blink_hz = None
        led_blink_mode = None
        if isinstance(system, dict):
            led_color = self._first_hex_color_value(
                system,
                keys=("led_color", "color"),
            )
            led_blink_hz = self._first_numeric_value(
                system,
                keys=("led_blink_hz", "blink_hz", "blinkHz"),
            )
            blink_mode_value = self._first_numeric_value(
                system,
                keys=("led_blink_mode", "blink_mode", "blinkMode"),
            )
            if blink_mode_value is not None:
                led_blink_mode = int(round(blink_mode_value))
        return CupStatusSnapshot(
            connected=connected,
            motor_target=motor_target,
            motor_enabled=motor_enabled,
            led_color=led_color,
            led_blink_hz=led_blink_hz,
            led_blink_mode=led_blink_mode,
        )

    def _evaluate_cup_action_payload(
        self,
        payload: dict[str, object],
        *,
        pre_snapshot: CupStatusSnapshot | None,
        post_snapshot: CupStatusSnapshot | None,
    ) -> tuple[str, str]:
        action = str(payload.get("action", "")).strip().casefold()
        if payload.get("ok") is not True:
            return "failure", f"{action or 'control_cup'} 返回 ok=false"
        if post_snapshot is None:
            return "unknown", "执行后状态不可用，无法校验"
        if not post_snapshot.connected:
            return "failure", "执行后设备离线"

        if action in {"set_motor", "stop", "nudge_intensity"}:
            return self._evaluate_cup_motor_outcome(
                payload,
                action=action,
                pre_snapshot=pre_snapshot,
                post_snapshot=post_snapshot,
            )
        if action == "apply_scene":
            motor_status, motor_detail = self._evaluate_cup_motor_outcome(
                payload,
                action=action,
                pre_snapshot=pre_snapshot,
                post_snapshot=post_snapshot,
            )
            scene_led_payload = {"requested": payload.get("led")}
            led_status, led_detail = self._evaluate_cup_led_outcome(
                scene_led_payload,
                post_snapshot=post_snapshot,
            )
            statuses = {motor_status, led_status}
            if "failure" in statuses:
                if motor_status == "failure":
                    return "failure", motor_detail
                return "failure", led_detail
            if statuses == {"success"}:
                return "success", f"{motor_detail}; {led_detail}"
            if motor_status == "unknown" and led_status == "success":
                return "unknown", motor_detail
            if led_status == "unknown" and motor_status == "success":
                return "unknown", led_detail
            return "unknown", f"{motor_detail}; {led_detail}"
        if action == "set_led":
            return self._evaluate_cup_led_outcome(payload, post_snapshot=post_snapshot)
        return "unknown", f"{action or 'control_cup'} 已执行，但暂无匹配校验规则"

    def _evaluate_cup_motor_outcome(
        self,
        payload: dict[str, object],
        *,
        action: str,
        pre_snapshot: CupStatusSnapshot | None,
        post_snapshot: CupStatusSnapshot,
    ) -> tuple[str, str]:
        if action == "stop":
            stopped = (
                post_snapshot.motor_enabled is False
                or (
                    post_snapshot.motor_target is not None
                    and post_snapshot.motor_target <= 1.0
                )
            )
            if stopped:
                return "success", "电机已停下"
            return "failure", "电机仍在运行"

        requested = payload.get("requested")
        if not isinstance(requested, dict):
            requested = payload.get("motor") if isinstance(payload.get("motor"), dict) else {}
        expected_target = self._first_numeric_value(
            requested,
            keys=("target_velocity", "target", "vel"),
        )
        expected_enabled = self._first_bool_value(
            requested,
            keys=("enabled", "running", "on"),
        )
        if action == "nudge_intensity":
            expected_target = self._first_numeric_value(
                payload,
                keys=("target_velocity",),
            ) or expected_target

        if expected_enabled is not None:
            if post_snapshot.motor_enabled is None:
                return "unknown", "电机开关状态不可用"
            if post_snapshot.motor_enabled != expected_enabled:
                return (
                    "failure",
                    f"电机开关不符合预期（expected={expected_enabled}, actual={post_snapshot.motor_enabled}）",
                )
        if expected_target is not None:
            if post_snapshot.motor_target is None:
                return "unknown", "电机目标转速不可用"
            if abs(post_snapshot.motor_target - expected_target) > 3.5:
                return (
                    "failure",
                    f"电机目标转速不符合预期（expected≈{expected_target:g}, actual={post_snapshot.motor_target:g}）",
                )
            return "success", f"电机目标转速已到 {post_snapshot.motor_target:g}"

        if (
            action == "nudge_intensity"
            and pre_snapshot is not None
            and pre_snapshot.motor_target is not None
            and post_snapshot.motor_target is not None
        ):
            direction = str(payload.get("direction", "")).strip().casefold()
            delta = post_snapshot.motor_target - pre_snapshot.motor_target
            if direction == "up":
                if delta >= 0.8:
                    return "success", f"电机转速已上调到 {post_snapshot.motor_target:g}"
                return (
                    "failure",
                    f"电机转速未明显上调（before={pre_snapshot.motor_target:g}, after={post_snapshot.motor_target:g}）",
                )
            if direction == "down":
                if delta <= -0.8:
                    return "success", f"电机转速已下调到 {post_snapshot.motor_target:g}"
                return (
                    "failure",
                    f"电机转速未明显下调（before={pre_snapshot.motor_target:g}, after={post_snapshot.motor_target:g}）",
                )
        return "unknown", f"{action} 已执行，但电机参数不足以校验"

    def _evaluate_cup_led_outcome(
        self,
        payload: dict[str, object],
        *,
        post_snapshot: CupStatusSnapshot,
    ) -> tuple[str, str]:
        requested = payload.get("requested")
        if not isinstance(requested, dict):
            requested = {}
        expected_color = self._first_hex_color_value(requested, keys=("color", "led_color"))
        expected_blink_hz = self._first_numeric_value(
            requested,
            keys=("blink_hz", "blinkHz", "led_blink_hz"),
        )
        expected_blink_mode = self._first_numeric_value(
            requested,
            keys=("blink_mode", "blinkMode", "led_blink_mode"),
        )

        checks: list[tuple[bool, str]] = []
        if expected_color is not None:
            if post_snapshot.led_color is None:
                return "unknown", "LED 颜色状态不可用"
            checks.append(
                (
                    post_snapshot.led_color.casefold() == expected_color.casefold(),
                    f"LED 颜色 expected={expected_color}, actual={post_snapshot.led_color}",
                )
            )
        if expected_blink_hz is not None:
            if post_snapshot.led_blink_hz is None:
                return "unknown", "LED 闪烁频率状态不可用"
            checks.append(
                (
                    abs(post_snapshot.led_blink_hz - expected_blink_hz) <= 0.15,
                    f"LED 闪烁频率 expected≈{expected_blink_hz:g}, actual={post_snapshot.led_blink_hz:g}",
                )
            )
        if expected_blink_mode is not None:
            expected_mode = int(round(expected_blink_mode))
            if post_snapshot.led_blink_mode is None:
                return "unknown", "LED 闪烁模式状态不可用"
            checks.append(
                (
                    post_snapshot.led_blink_mode == expected_mode,
                    f"LED 模式 expected={expected_mode}, actual={post_snapshot.led_blink_mode}",
                )
            )

        if not checks:
            return "unknown", "set_led 已执行，但缺少可校验参数"
        failures = [detail for ok, detail in checks if not ok]
        if failures:
            return "failure", failures[0]
        return "success", "LED 状态已按预期更新"

    def _update_transcript_assistant_content(
        self,
        session: ConversationSession,
        *,
        assistant_message: ChatMessage,
        content: str,
    ) -> None:
        for message in reversed(session.transcript_messages):
            if message.role != "assistant":
                continue
            if message.created_at != assistant_message.created_at:
                continue
            message.content = content
            return

    @staticmethod
    def _first_hex_color_value(
        payload: object,
        *,
        keys: tuple[str, ...],
    ) -> str | None:
        if not isinstance(payload, dict):
            return None
        for key in keys:
            value = payload.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                if 0 <= value <= 0xFFFFFF:
                    return f"#{value:06x}"
            if not isinstance(value, str):
                continue
            normalized = value.strip().casefold()
            if not normalized:
                continue
            if normalized.startswith("0x"):
                try:
                    numeric = int(normalized, 16)
                except ValueError:
                    continue
                if 0 <= numeric <= 0xFFFFFF:
                    return f"#{numeric:06x}"
                continue
            if normalized.startswith("#"):
                normalized = normalized[1:]
            if len(normalized) == 6 and all(ch in "0123456789abcdef" for ch in normalized):
                return f"#{normalized}"
        return None

    @staticmethod
    def _kimi_web_search_tool() -> dict[str, object]:
        return {
            "type": "builtin_function",
            "function": {
                "name": "$web_search",
            },
        }

    def _should_disable_thinking_for_request(
        self,
        provider,
        model,
        tools: list[dict[str, object]] | None,
    ) -> bool:
        # Kimi k2.5 with $web_search should keep thinking enabled.
        # Disabling thinking mid-conversation causes reasoning_content
        # mismatches that trigger HTTP 400 from the Kimi API.
        return False

    def _append_trace(
        self,
        *,
        kind: str,
        session_id: str,
        decision: RouteDecision,
        provider_name: str,
        streamed: bool,
        tools_enabled: bool,
        tool_choice: str | None = None,
        note: str | None = None,
        preview: str | None = None,
    ) -> None:
        if self.trace_store is None:
            return
        self.trace_store.append(
            make_trace_event(
                kind=kind,
                session_id=session_id,
                model_alias=decision.model_alias,
                provider_name=provider_name,
                route_mode=decision.mode,
                streamed=streamed,
                tools_enabled=tools_enabled,
                tool_choice=tool_choice,
                note=note,
                preview=self._short_preview(preview),
            )
        )

    def _short_preview(self, value: str | None, *, limit: int = 220) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if not normalized:
            return None
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."

    def _trace_preview_from_messages(self, messages: list[dict[str, object]]) -> str | None:
        for message in reversed(messages):
            content = message.get("content")
            role = message.get("role")
            if isinstance(content, str) and content.strip():
                prefix = str(role) if role is not None else "message"
                return f"{prefix}: {content}"
        return None

    def _tool_call_names_preview(self, tool_calls: tuple[ToolCall, ...]) -> str | None:
        if not tool_calls:
            return None
        return ", ".join(tool_call.name for tool_call in tool_calls)

    def _stringify_tool_choice(self, tool_choice: str | dict[str, object] | None) -> str | None:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            return tool_choice
        return str(tool_choice)

    def _should_continue_tool_loop(
        self,
        content: str,
        *,
        route_mode: str,
        round_index: int,
    ) -> bool:
        return self.harness.should_continue_tool_loop(
            content,
            route_mode=route_mode,
            round_index=round_index,
        )

    def _should_retry_with_structured_tool_arguments(
        self,
        provider,
        messages: list[dict[str, object]],
        exc: ProviderError,
    ) -> bool:
        if not self._has_assistant_tool_calls(messages):
            return False
        error_text = str(exc).casefold()
        return (
            provider.kind == "ollama_native"
            or "ollama" in provider.name.casefold()
            or "can't find closing '}' symbol" in error_text
            or "value looks like object" in error_text
        )

    def _has_assistant_tool_calls(self, messages: list[dict[str, object]]) -> bool:
        return self.tool_protocol_adapter.has_assistant_tool_calls(messages)

    def _messages_with_structured_tool_arguments(
        self,
        messages: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        return self.tool_protocol_adapter.messages_with_structured_tool_arguments(messages)

    def _should_retry_without_tools(self, provider, exc: ProviderError) -> bool:
        error_text = str(exc).casefold()
        if any(
            marker in error_text
            for marker in (
                "unreachable",
                "ssl",
                "eof occurred in violation of protocol",
                "closed the connection unexpectedly",
                "timed out",
                "timeout",
                "connection reset",
                "remote end closed connection",
            )
        ):
            return False
        if "tool" in error_text or "function" in error_text:
            return True
        provider_name = str(getattr(provider, "name", "")).casefold()
        if "ollama" in provider_name and "400" in error_text:
            return True
        return False

    def _tool_message_reasoning_content(
        self,
        decision: RouteDecision,
        completion: CompletionResult,
    ) -> str | None:
        if completion.reasoning_content is not None:
            return completion.reasoning_content
        model_config = self.config.get_model(decision.model_alias)
        provider_config = self.config.get_provider(model_config.provider)
        profile = resolve_profile(provider_config, model_config)
        if not self._requires_reasoning_content_replay(profile, model_config):
            return None
        return profile.reasoning_content_empty_placeholder

    def _requires_reasoning_content_replay(
        self,
        profile: ProviderProfile,
        model,
    ) -> bool:
        if not profile.reasoning_content_required_when_thinking:
            return False
        if model.think is False:
            return False
        return True

    def _stream_final_answer(
        self,
        session: ConversationSession,
        decision: RouteDecision,
        *,
        provider,
        model,
        messages: list[dict[str, object]],
        on_chunk: Callable[[str], None] | None,
        target_profile: ProviderProfile | None = None,
    ) -> ChatTurnResult:
        messages = self.tool_protocol_adapter.messages_with_normalized_tool_call_ids(messages)
        content_parts: list[str] = []
        self._append_trace(
            kind="stream_request",
            session_id=session.session_id,
            decision=decision,
            provider_name=provider.name,
            streamed=True,
            tools_enabled=False,
            preview=self._trace_preview_from_messages(messages),
        )
        try:
            for chunk in self.client.create_chat_completion_stream(
                provider=provider,
                model=model,
                messages=messages,
            ):
                content_parts.append(chunk)
                if on_chunk is not None:
                    on_chunk(chunk)
        except KeyboardInterrupt as exc:
            partial_result = None
            partial_content = "".join(content_parts).strip()
            if partial_content:
                partial_result = self._append_assistant_message(
                    session,
                    decision,
                    partial_content,
                    target_profile=target_profile,
                )
            raise GenerationInterrupted(partial_result=partial_result) from exc
        except ProviderError:
            if content_parts:
                raise
            completion = self._client_create_chat_completion(
                session_id=session.session_id,
                decision=decision,
                provider=provider,
                model=model,
                messages=messages,
                streamed=True,
            )
            content_parts = [completion.content]
            if on_chunk is not None:
                on_chunk(completion.content)
        final_content = "".join(content_parts).strip()
        self._append_trace(
            kind="stream_response",
            session_id=session.session_id,
            decision=decision,
            provider_name=provider.name,
            streamed=True,
            tools_enabled=False,
            preview=final_content,
        )
        return self._append_assistant_message(
            session,
            decision,
            final_content,
            target_profile=target_profile,
        )

    def _append_assistant_message(
        self,
        session: ConversationSession,
        decision: RouteDecision,
        content: str,
        *,
        ask_user: AskUserAction | None = None,
        target_profile: ProviderProfile | None = None,
    ) -> ChatTurnResult:
        # Sanitise any model-specific tool-call text artifacts that may have
        # leaked through when text-based extraction failed.  This prevents
        # stored content from confusing a different model on subsequent turns.
        content = sanitize_tool_call_artifacts(content)
        source_profile_id = target_profile.profile_id if target_profile is not None else None
        assistant_message = ChatMessage(
            role="assistant",
            content=content,
            created_at=utc_now_iso(),
            model_alias=decision.model_alias,
            route_reason=decision.reason,
            source_profile=source_profile_id,
        )
        session.append(assistant_message)
        session.append_transcript(
            ChatMessage(
                role="assistant",
                content=content,
                created_at=assistant_message.created_at,
                model_alias=decision.model_alias,
                route_reason=decision.reason,
                source_profile=source_profile_id,
            )
        )
        session.pending_ask_user = ask_user

        return ChatTurnResult(
            decision=decision,
            assistant_message=assistant_message,
            ask_user=ask_user,
        )
