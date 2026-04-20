from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import threading
import time
import sys
from typing import Callable

from whesper.agent_types import AgentStep
from whesper.agent_types import AskUserAction
from whesper.agent_types import ToolInvocation
from whesper.chat import ChatService, ChatTurnResult, GenerationInterrupted
from whesper.client import ProviderError, list_provider_models
from whesper.commands import (
    VALID_MODES,
    ParsedCommand,
    command_completions,
    help_text,
    parse_command,
)
from whesper.config import AppConfig, ConfigError, VALID_SEARCH_PROVIDERS, load_config
from whesper.memory import MemoryService, MemoryStore
from whesper.provider_profile import resolve_profile
from whesper.router import select_model
from whesper.session import (
    ChatMessage,
    ConversationSession,
    SessionStore,
    is_transcript_message,
    utc_now_iso,
)
from whesper.trace import TraceStore
from whesper.tools import ToolExecutionError


DEFAULT_CONFIG_PATH = "whesper.toml"
CUP_DEBUG_SCENES = ("gentle", "steady", "intense", "cooldown")
CUP_DEBUG_DIRECTIONS = ("up", "down")
CUP_COMMAND_USAGE: dict[str, str] = {
    "/cup-status": "/cup-status",
    "/cup-speed": "/cup-speed <velocity>",
    "/cup-stop": "/cup-stop",
    "/cup-scene": "/cup-scene <gentle|steady|intense|cooldown>",
    "/cup-intensity": "/cup-intensity <up|down> [step]",
    "/cup-led": "/cup-led <color> [blink_hz] [blink_mode]",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whesper")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the TOML config file (default: whesper.toml)",
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session")
    chat_parser.add_argument(
        "--session",
        default=None,
        help="Session name. Defaults to app.default_session from config.",
    )
    chat_parser.add_argument(
        "--model",
        default=None,
        help="Use a specific model alias for this CLI run (no in-session switching).",
    )

    subparsers.add_parser("models", help="List configured models")
    subparsers.add_parser("sessions", help="List local saved sessions")

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the HTTP/SSE backend server (requires the `server` extra).",
    )
    serve_parser.add_argument(
        "--host",
        default=None,
        help="Override [server].host from config.",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override [server].port from config.",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload (development only).",
    )

    return parser


def load_runtime(args: argparse.Namespace) -> tuple[AppConfig, SessionStore]:
    config = load_config(args.config)
    storage_dir = Path(config.app.storage_dir)
    store = SessionStore(storage_dir)
    return config, store


def print_models(config: AppConfig) -> None:
    print("Configured models:")
    for alias, model in config.models.items():
        provider = config.get_provider(model.provider)
        print(
            f"  - {alias}: model={model.model} provider={provider.name} base_url={provider.base_url}"
        )


def print_sessions(store: SessionStore) -> None:
    sessions = store.list_sessions()
    if not sessions:
        print("No saved sessions yet.")
        return
    print("Saved sessions:")
    for item in sessions:
        print(f"  - {item}")


def print_history(
    session: ConversationSession,
    limit: int = 12,
    *,
    output_stream: object = sys.stdout,
) -> None:
    history_source = session.transcript_messages
    if not history_source:
        history_source = [
            message for message in session.messages if is_transcript_message(message)
        ]
    history = history_source[-limit:]
    if not history:
        print("Session is empty.", file=output_stream)
        return
    print(divider(f"History ({len(history)})", color=SLATE), file=output_stream)
    print(file=output_stream)
    for index, message in enumerate(history, start=1):
        if message.role == "user":
            title = "You"
            accent = CYAN
            body_style = CYAN
        elif message.role == "assistant":
            title = "Whesper"
            accent = GOLD
            body_style = MINT
        else:
            title = message.role.title()
            accent = ROSE
            body_style = ROSE

        meta_parts = [f"#{index}", message.created_at]
        if message.model_alias:
            meta_parts.append(f"model {message.model_alias}")
        if message.route_reason:
            meta_parts.append(message.route_reason)

        print(divider(title, color=accent), file=output_stream)
        print(
            colorize(title.lower(), accent, BOLD)
            + colorize(f"  {' | '.join(meta_parts)}", SLATE, DIM),
            file=output_stream,
        )
        print(file=output_stream)
        print(colorize(message.content, body_style), file=output_stream)
        print(file=output_stream)


def print_memory(
    memory_store: MemoryStore,
    *,
    limit: int = 12,
    output_stream: object = sys.stdout,
) -> None:
    memories = memory_store.list_memories()[:limit]
    if not memories:
        print("No saved memories yet.", file=output_stream)
        return

    print(divider(f"Memory ({len(memories)})", color=SLATE), file=output_stream)
    print(file=output_stream)
    for memory in memories:
        meta_parts = [memory.memory_id, memory.memory_type, memory.source]
        if memory.last_confirmed_at:
            meta_parts.append(f"confirmed {memory.last_confirmed_at}")
        print(divider(memory.title, color=GOLD), file=output_stream)
        print(
            colorize(memory.title.lower(), GOLD, BOLD)
            + colorize(f"  {' | '.join(meta_parts)}", SLATE, DIM),
            file=output_stream,
        )
        print(file=output_stream)
        print(colorize(memory.content, MINT), file=output_stream)
        print(file=output_stream)


def print_trace(
    trace_store: TraceStore,
    *,
    limit: int = 20,
    session_id: str | None = None,
    title: str = "Trace",
    output_stream: object = sys.stdout,
) -> None:
    events = trace_store.tail(limit=limit, session_id=session_id)
    if not events:
        print("No trace events yet.", file=output_stream)
        return

    print(divider(f"{title} ({len(events)})", color=SLATE), file=output_stream)
    print(file=output_stream)
    for event in events:
        meta_parts = [
            event.timestamp,
            event.kind,
            f"session {event.session_id}",
            f"model {event.model_alias}",
            f"provider {event.provider_name}",
            f"route {event.route_mode}",
            "stream" if event.streamed else "sync",
            "tools on" if event.tools_enabled else "tools off",
        ]
        if event.tool_choice:
            meta_parts.append(f"tool_choice {event.tool_choice}")
        print(colorize(" | ".join(meta_parts), GOLD, BOLD), file=output_stream)
        if event.note:
            print(colorize(f"note: {event.note}", SLATE, DIM), file=output_stream)
        if event.preview:
            print(colorize(f"preview: {event.preview}", MINT), file=output_stream)
        print(file=output_stream)


def _parse_tool_message_payload(content: str) -> dict[str, object] | None:
    stripped = content.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def collect_tool_debug_entries(messages: list[ChatMessage]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for message in messages:
        if message.role != "tool":
            continue
        entry: dict[str, object] = {
            "tool": message.name or "unknown",
        }
        payload = _parse_tool_message_payload(message.content)
        if payload is not None:
            action = payload.get("action")
            if isinstance(action, str) and action:
                entry["action"] = action
            request_trace = payload.get("request_trace")
            if isinstance(request_trace, list) and request_trace:
                entry["request_trace"] = request_trace
            else:
                entry["result"] = payload
        elif message.content.strip():
            entry["raw_result"] = message.content.strip()
        if len(entry) > 1:
            entries.append(entry)
    return entries


def clone_session(
    session: ConversationSession,
    *,
    session_id: str,
) -> ConversationSession:
    return ConversationSession(
        session_id=session_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        pinned_model=session.pinned_model,
        messages=[
            ChatMessage(
                role=message.role,
                content=message.content,
                created_at=message.created_at,
                model_alias=message.model_alias,
                route_reason=message.route_reason,
                reasoning_content=message.reasoning_content,
                name=message.name,
                tool_call_id=message.tool_call_id,
                tool_calls=copy.deepcopy(message.tool_calls),
            )
            for message in session.messages
        ],
        transcript_messages=[
            ChatMessage(
                role=message.role,
                content=message.content,
                created_at=message.created_at,
                model_alias=message.model_alias,
                route_reason=message.route_reason,
                reasoning_content=message.reasoning_content,
                name=message.name,
                tool_call_id=message.tool_call_id,
                tool_calls=copy.deepcopy(message.tool_calls),
            )
            for message in session.transcript_messages
        ],
    )


def trim_debug_value(value: object, *, limit: int = 800) -> object:
    if isinstance(value, str):
        normalized = value.strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."
    if isinstance(value, list):
        return [trim_debug_value(item, limit=limit) for item in value]
    if isinstance(value, dict):
        return {
            str(key): trim_debug_value(item, limit=limit)
            for key, item in value.items()
        }
    return value


def print_debug_payload(
    payload: dict[str, object],
    *,
    output_stream: object = sys.stdout,
) -> None:
    print(divider("Payload Debug", color=SLATE), file=output_stream)
    print(file=output_stream)
    print(
        colorize(
            json.dumps(trim_debug_value(payload), ensure_ascii=False, indent=2),
            MINT,
        ),
        file=output_stream,
    )
    print(file=output_stream)


def run_trace_debug_turn(
    chat_service: ChatService,
    session: ConversationSession,
    *,
    user_text: str,
    mode_override: str,
    trace_store: TraceStore,
    output_stream: object = sys.stdout,
) -> None:
    shadow_session_id = f"{session.session_id}__trace_debug__{time.time_ns()}"
    shadow_session = clone_session(session, session_id=shadow_session_id)
    shadow_session.append(
        ChatMessage(
            role="user",
            content=user_text,
            created_at=utc_now_iso(),
        )
    )
    decision = select_model(
        chat_service.config,
        user_text,
        pinned_model=shadow_session.pinned_model,
        mode_override=mode_override,
    )
    model_config = chat_service.config.get_model(decision.model_alias)
    provider_config = chat_service.config.get_provider(model_config.provider)
    target_profile = resolve_profile(provider_config, model_config)
    tools = chat_service._tools_for_request(
        session=shadow_session,
        provider=provider_config,
        model=model_config,
        user_text=user_text,
        route_mode=decision.mode,
    )
    needs_reasoning = chat_service._requires_reasoning_content_replay(target_profile, model_config)
    messages = chat_service._build_messages(
        shadow_session,
        model_config.system_prompt,
        target_profile=target_profile,
        user_text=user_text,
        route_mode=decision.mode,
        include_live_context=tools is None,
        include_tool_prompt=tools is not None,
        ensure_reasoning_content=needs_reasoning,
    )
    tool_choice = chat_service._tool_choice_for_request(
        shadow_session,
        user_text,
        tools,
        route_mode=decision.mode,
    )

    print(divider("Trace Debug", color=SLATE), file=output_stream)
    print(file=output_stream)
    print(f"shadow session: {shadow_session.session_id}", file=output_stream)
    print(f"user input: {user_text}", file=output_stream)
    print(f"route mode: {decision.mode}", file=output_stream)
    print(f"route reason: {decision.reason}", file=output_stream)
    print(f"model alias: {decision.model_alias}", file=output_stream)
    print(f"provider: {provider_config.name} ({provider_config.kind})", file=output_stream)
    print(file=output_stream)

    print_debug_payload(
        {
            "provider": {
                "name": provider_config.name,
                "kind": provider_config.kind,
                "base_url": provider_config.base_url,
            },
            "request": {
                "model_alias": model_config.name,
                "model_id": model_config.model,
                "stream": False,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
            },
        },
        output_stream=output_stream,
    )

    print(divider("Live Trace", color=SLATE), file=output_stream)
    print(file=output_stream)

    llm_round = 0
    original_harness_completion_requester = chat_service.harness.completion_requester
    original_tool_execute = chat_service.tool_executor.execute
    had_client_create_override = "_client_create_chat_completion" in chat_service.__dict__
    previous_client_create_override = chat_service.__dict__.get("_client_create_chat_completion")

    def traced_completion_requester(**kwargs):
        nonlocal llm_round
        llm_round += 1

        model = kwargs.get("model")
        provider = kwargs.get("provider")
        messages_payload = kwargs.get("messages")
        tools_payload = kwargs.get("tools")
        tool_choice_payload = kwargs.get("tool_choice")
        streamed_payload = kwargs.get("streamed")

        print_command_payload(
            f"LLM Round {llm_round} Request",
            {
                "provider": getattr(provider, "name", None),
                "model_alias": getattr(model, "name", None),
                "model_id": getattr(model, "model", None),
                "streamed": streamed_payload,
                "tool_choice": tool_choice_payload,
                "tools": tools_payload,
                "messages": messages_payload,
            },
            output_stream=output_stream,
        )

        completion = original_harness_completion_requester(**kwargs)
        tool_calls_payload: list[dict[str, object]] = []
        for tool_call in completion.tool_calls:
            arguments: object
            try:
                arguments = tool_call.arguments()
            except Exception:
                arguments = tool_call.arguments_json
            tool_calls_payload.append(
                {
                    "tool_call_id": tool_call.tool_call_id,
                    "name": tool_call.name,
                    "arguments": arguments,
                }
            )
        print_command_payload(
            f"LLM Round {llm_round} Response",
            {
                "content": completion.content,
                "reasoning_content": completion.reasoning_content,
                "tool_calls": tool_calls_payload,
            },
            output_stream=output_stream,
        )
        return completion

    def traced_tool_execute(tool_call: ToolInvocation):
        tool_arguments: object
        try:
            tool_arguments = tool_call.arguments()
        except Exception:
            tool_arguments = tool_call.arguments_json
        print_command_payload(
            "Tool Execute",
            {
                "tool_call_id": tool_call.tool_call_id,
                "name": tool_call.name,
                "arguments": tool_arguments,
            },
            output_stream=output_stream,
        )
        try:
            result = original_tool_execute(tool_call)
        except Exception as exc:
            print_command_payload(
                "Tool Error",
                {
                    "tool_call_id": tool_call.tool_call_id,
                    "name": tool_call.name,
                    "error": f"{exc.__class__.__name__}: {exc}",
                },
                output_stream=output_stream,
            )
            raise

        tool_result_payload: object
        try:
            tool_result_payload = json.loads(result.content)
        except json.JSONDecodeError:
            tool_result_payload = result.content
        print_command_payload(
            "Tool Result",
            {
                "tool_call_id": result.tool_call_id,
                "name": result.name,
                "result": tool_result_payload,
            },
            output_stream=output_stream,
        )
        return result

    def handle_step(step: AgentStep) -> None:
        summary = step.summary.strip()
        pieces = [f"round={step.round_index}", f"kind={step.kind}"]
        if step.tool_name:
            pieces.append(f"tool={step.tool_name}")
        if summary:
            pieces.append(f"summary={summary}")
        if step.is_error:
            pieces.append("error=true")
        print(
            colorize("step> " + " | ".join(pieces), SLATE, DIM),
            file=output_stream,
        )

    chat_service.harness.completion_requester = traced_completion_requester
    chat_service.tool_executor.execute = traced_tool_execute  # type: ignore[assignment]
    chat_service._client_create_chat_completion = traced_completion_requester  # type: ignore[assignment]

    try:
        result = chat_service._complete_turn(
            shadow_session,
            user_text,
            mode_override=mode_override,
            on_step=handle_step,
        )
    finally:
        chat_service.harness.completion_requester = original_harness_completion_requester
        chat_service.tool_executor.execute = original_tool_execute  # type: ignore[assignment]
        if had_client_create_override:
            chat_service._client_create_chat_completion = previous_client_create_override  # type: ignore[assignment]
        else:
            try:
                delattr(chat_service, "_client_create_chat_completion")
            except AttributeError:
                pass

    print_trace(
        trace_store,
        session_id=shadow_session.session_id,
        title="Execution Timeline",
        output_stream=output_stream,
    )
    tool_debug_entries = collect_tool_debug_entries(shadow_session.messages)
    if tool_debug_entries:
        print_command_payload(
            "Tool Request Trace",
            {"entries": tool_debug_entries},
            output_stream=output_stream,
        )
    print(divider("Final Reply", color=SLATE), file=output_stream)
    print(file=output_stream)
    print(colorize(result.assistant_message.content or "(empty reply)", MINT), file=output_stream)
    print(file=output_stream)


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[38;5;81m"
GOLD = "\033[38;5;221m"
SLATE = "\033[38;5;110m"
MINT = "\033[38;5;151m"
ROSE = "\033[38;5;210m"
AVAILABLE_MODES_TEXT = ", ".join(VALID_MODES)


def colorize(text: str, *styles: str) -> str:
    return f"{''.join(styles)}{text}{RESET}"


def divider(label: str = "", *, color: str = SLATE) -> str:
    line = "─" * 18
    if label:
        return colorize(f"{line} {label} {line}", color, DIM)
    return colorize("─" * 44, color, DIM)


def format_elapsed(seconds: float) -> str:
    return f"{seconds:.1f}s"


def render_user_message(user_text: str) -> None:
    print()
    print(divider("You", color=CYAN))
    print(colorize(user_text, CYAN))
    print(divider(color=SLATE))
    print()


def render_assistant_header(
    *,
    model_alias: str,
    first_token_seconds: float | None = None,
    output_stream: object = sys.stdout,
) -> None:
    meta = f"model {model_alias}"
    if first_token_seconds is not None:
        meta = f"{meta} | first token {format_elapsed(first_token_seconds)}"
    print(divider("Whesper", color=GOLD), file=output_stream)
    print(
        colorize(f"whesper", GOLD, BOLD) + colorize(f"  {meta}", SLATE, DIM),
        file=output_stream,
    )
    print(file=output_stream)


def stream_printer(chunk: str, *, output_stream: object) -> None:
    print(colorize(chunk, MINT), end="", flush=True, file=output_stream)


def print_cli_error(message: str, *, output_stream: object = sys.stdout) -> None:
    print(colorize(f"error> {message}", ROSE, BOLD), file=output_stream)


def print_cli_hint(message: str, *, output_stream: object = sys.stdout) -> None:
    print(colorize(message, SLATE, DIM), file=output_stream)


def describe_agent_step(step: AgentStep) -> str:
    if step.kind == "tool_call" and step.tool_name:
        return f"whesper> running {step.tool_name}"
    if step.kind == "tool_result" and step.tool_name:
        if step.is_error:
            return f"whesper> {step.tool_name} failed"
        return f"whesper> received {step.tool_name}"
    if step.kind == "planning_retry":
        return "whesper> replanning"
    if step.kind == "error_recovery":
        return "whesper> recovering"
    if step.kind == "ask_user":
        return "whesper> waiting for your input"
    if step.kind == "final":
        return "whesper> answering"
    return "whesper> thinking"


def render_ask_user_prompt(
    ask_user: AskUserAction,
    *,
    output_stream: object = sys.stdout,
) -> None:
    if ask_user.options:
        print_cli_hint("Quick options:", output_stream=output_stream)
        for index, option in enumerate(ask_user.options, start=1):
            print(
                colorize(f"  {index}. {option.label}", CYAN),
                file=output_stream,
            )
            if option.description:
                print_cli_hint(
                    f"     {option.description}",
                    output_stream=output_stream,
                )
    if ask_user.allow_free_text:
        print_cli_hint(
            "Enter a number or type your own answer to continue.",
            output_stream=output_stream,
        )
    elif ask_user.options:
        print_cli_hint(
            "Enter one of the option numbers to continue.",
            output_stream=output_stream,
        )


def clear_screen(*, output_stream: object = sys.stdout) -> None:
    print("\033[2J\033[H", end="", flush=True, file=output_stream)


def find_last_assistant_message(session: ConversationSession) -> str | None:
    for message in reversed(session.messages):
        if message.role == "assistant":
            return message.content
    return None


def find_last_user_message(session: ConversationSession) -> str | None:
    for message in reversed(session.messages):
        if message.role == "user":
            return message.content
    return None


def current_route_decision(
    config: AppConfig,
    session: ConversationSession,
    *,
    mode_override: str,
):
    return select_model(
        config,
        "",
        pinned_model=session.pinned_model,
        mode_override=mode_override,
    )


def print_status(
    config: AppConfig,
    session: ConversationSession,
    *,
    mode_override: str,
    output_stream: object = sys.stdout,
) -> None:
    decision = current_route_decision(config, session, mode_override=mode_override)
    model_config = config.get_model(decision.model_alias)
    provider_config = config.get_provider(model_config.provider)

    print(divider("Status", color=SLATE), file=output_stream)
    print(f"session: {session.session_id}", file=output_stream)
    print(f"messages: {len(session.messages)}", file=output_stream)
    print(f"generation model override: {session.pinned_model}", file=output_stream)
    print(f"route mode: {mode_override}", file=output_stream)
    print(f"effective model: {decision.model_alias}", file=output_stream)
    print(f"provider: {provider_config.name} ({provider_config.kind})", file=output_stream)
    print(f"config: {config.source_path}", file=output_stream)


def print_model_info(
    config: AppConfig,
    session: ConversationSession,
    *,
    mode_override: str,
    output_stream: object = sys.stdout,
) -> None:
    decision = current_route_decision(config, session, mode_override=mode_override)
    model_config = config.get_model(decision.model_alias)
    provider_config = config.get_provider(model_config.provider)
    tags = ", ".join(model_config.tags) if model_config.tags else "none"

    print(divider("Model Info", color=SLATE), file=output_stream)
    print(f"alias: {model_config.name}", file=output_stream)
    print(f"model id: {model_config.model}", file=output_stream)
    print(f"provider: {provider_config.name}", file=output_stream)
    print(f"provider kind: {provider_config.kind}", file=output_stream)
    print(f"base_url: {provider_config.base_url}", file=output_stream)
    print(f"temperature: {model_config.temperature}", file=output_stream)
    print(f"max_tokens: {model_config.max_tokens if model_config.max_tokens is not None else 'default'}", file=output_stream)
    print(f"top_p: {model_config.top_p if model_config.top_p is not None else 'default'}", file=output_stream)
    print(f"think: {model_config.think if model_config.think is not None else 'default'}", file=output_stream)
    print(f"tags: {tags}", file=output_stream)
    print(f"route reason: {decision.reason}", file=output_stream)


def print_command_payload(
    title: str,
    payload: dict[str, object],
    *,
    output_stream: object = sys.stdout,
) -> None:
    print(divider(title, color=SLATE), file=output_stream)
    print(file=output_stream)
    print(
        colorize(
            json.dumps(trim_debug_value(payload), ensure_ascii=False, indent=2),
            MINT,
        ),
        file=output_stream,
    )
    print(file=output_stream)


def parse_slash_arg_tokens(arg: str | None) -> list[str]:
    if not arg:
        return []
    try:
        return shlex.split(arg)
    except ValueError as exc:
        raise ValueError(f"Invalid command arguments: {exc}") from exc


def control_cup_tool_available(chat_service: ChatService) -> bool:
    return any(spec.name == "control_cup" for spec in chat_service.tool_registry.specs)


def execute_control_cup_tool(
    chat_service: ChatService,
    arguments: dict[str, object],
) -> dict[str, object]:
    if not control_cup_tool_available(chat_service):
        raise ToolExecutionError(
            "control_cup tool is not available. Enable [hardware.cup] in your config first."
        )
    result = chat_service.tool_executor.execute(
        ToolInvocation(
            tool_call_id=f"cli-control-cup-{time.time_ns()}",
            name="control_cup",
            arguments_json=json.dumps(arguments, ensure_ascii=False),
        )
    )
    try:
        payload = json.loads(result.content)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError("control_cup returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ToolExecutionError("control_cup returned a non-object payload.")
    return payload


def _coerce_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        try:
            return float(normalized)
        except ValueError:
            return None
    return None


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "on", "enabled", "running", "online"}:
            return True
        if normalized in {"false", "0", "off", "disabled", "stopped", "offline"}:
            return False
    return None


@dataclass(slots=True)
class CupHeartbeatSnapshot:
    connected: bool | None
    checked_at: float | None = None
    device: str | None = None
    motor_target: float | None = None
    motor_enabled: bool | None = None
    error: str | None = None


class CupHeartbeatMonitor:
    def __init__(
        self,
        chat_service: ChatService,
        *,
        interval_seconds: float = 2.0,
    ) -> None:
        self.chat_service = chat_service
        self.interval_seconds = max(0.5, interval_seconds)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._paused = False
        self._thread: threading.Thread | None = None
        self._snapshot = CupHeartbeatSnapshot(connected=None)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="whesper-cup-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def snapshot(self) -> CupHeartbeatSnapshot:
        with self._lock:
            return CupHeartbeatSnapshot(
                connected=self._snapshot.connected,
                checked_at=self._snapshot.checked_at,
                device=self._snapshot.device,
                motor_target=self._snapshot.motor_target,
                motor_enabled=self._snapshot.motor_enabled,
                error=self._snapshot.error,
            )

    def toolbar_fragment(self) -> tuple[str, str]:
        snapshot = self.snapshot()
        if snapshot.connected is True:
            motor_label = "unknown"
            if snapshot.motor_enabled is True:
                if snapshot.motor_target is not None and snapshot.motor_target > 0.8:
                    motor_label = f"v{snapshot.motor_target:g}"
                else:
                    motor_label = "running"
            elif snapshot.motor_enabled is False:
                motor_label = "stopped"
            elif snapshot.motor_target is not None:
                if snapshot.motor_target > 0.8:
                    motor_label = f"v{snapshot.motor_target:g}"
                else:
                    motor_label = "stopped"
            return "class:cup_online", f" cup online {motor_label} "
        if snapshot.connected is False:
            return "class:cup_offline", " cup offline "
        if snapshot.error:
            return "class:cup_unknown", " cup unknown "
        return "class:cup_unknown", " cup checking "

    def _run(self) -> None:
        self._poll_once()
        while not self._stop_event.wait(self.interval_seconds):
            with self._lock:
                paused = self._paused
            if paused:
                continue
            self._poll_once()

    def _poll_once(self) -> None:
        try:
            payload = execute_control_cup_tool(self.chat_service, {"action": "status"})
        except ToolExecutionError as exc:
            self._set_snapshot(
                CupHeartbeatSnapshot(
                    connected=False,
                    checked_at=time.time(),
                    error=str(exc),
                )
            )
            return

        system = payload.get("system")
        motor = payload.get("motor")
        connected = bool(payload.get("ok") is True)
        if isinstance(system, dict):
            connected_value = _coerce_bool(system.get("connected"))
            if connected_value is not None:
                connected = connected_value
        motor_target = None
        motor_enabled = None
        if isinstance(motor, dict):
            for key in ("target", "target_velocity", "vel", "velocity"):
                motor_target = _coerce_number(motor.get(key))
                if motor_target is not None:
                    break
            for key in ("enabled", "running", "on"):
                motor_enabled = _coerce_bool(motor.get(key))
                if motor_enabled is not None:
                    break
        self._set_snapshot(
            CupHeartbeatSnapshot(
                connected=connected,
                checked_at=time.time(),
                device=(
                    payload.get("device")
                    if isinstance(payload.get("device"), str)
                    else None
                ),
                motor_target=motor_target,
                motor_enabled=motor_enabled,
                error=None,
            )
        )

    def _set_snapshot(self, snapshot: CupHeartbeatSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot


def _parse_required_float(token: str, *, field_name: str) -> float:
    try:
        return float(token)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number.") from exc


def _parse_required_int(token: str, *, field_name: str) -> int:
    try:
        return int(token)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc


def handle_cup_debug_command(
    parsed: ParsedCommand,
    *,
    chat_service: ChatService,
    output_stream: object = sys.stdout,
) -> bool:
    if parsed.name not in CUP_COMMAND_USAGE:
        return False

    def usage() -> None:
        print_cli_hint(f"Usage: {CUP_COMMAND_USAGE[parsed.name]}", output_stream=output_stream)

    try:
        tokens = parse_slash_arg_tokens(parsed.arg)
        if parsed.name == "/cup-status":
            if tokens:
                raise ValueError("/cup-status does not accept extra arguments.")
            payload = execute_control_cup_tool(chat_service, {"action": "status"})
            print_command_payload("CUP Status", payload, output_stream=output_stream)
            return True

        if parsed.name == "/cup-stop":
            if tokens:
                raise ValueError("/cup-stop does not accept extra arguments.")
            payload = execute_control_cup_tool(chat_service, {"action": "stop"})
            print_command_payload("CUP Stop", payload, output_stream=output_stream)
            return True

        if parsed.name == "/cup-speed":
            if len(tokens) != 1:
                raise ValueError("Missing CUP velocity.")
            velocity = _parse_required_float(tokens[0], field_name="velocity")
            if velocity < 0:
                raise ValueError("velocity must be >= 0.")
            if velocity == 0:
                payload = execute_control_cup_tool(chat_service, {"action": "stop"})
                print_command_payload("CUP Stop", payload, output_stream=output_stream)
                return True
            payload = execute_control_cup_tool(
                chat_service,
                {
                    "action": "set_motor",
                    "target_velocity": velocity,
                    "enabled": True,
                },
            )
            print_command_payload("CUP Speed", payload, output_stream=output_stream)
            return True

        if parsed.name == "/cup-scene":
            if len(tokens) != 1:
                raise ValueError("Missing CUP scene.")
            scene = tokens[0].strip()
            if scene not in CUP_DEBUG_SCENES:
                raise ValueError(
                    f"scene must be one of: {', '.join(CUP_DEBUG_SCENES)}."
                )
            payload = execute_control_cup_tool(
                chat_service,
                {"action": "apply_scene", "scene": scene},
            )
            print_command_payload("CUP Scene", payload, output_stream=output_stream)
            return True

        if parsed.name == "/cup-intensity":
            if not tokens or len(tokens) > 2:
                raise ValueError("Missing CUP intensity direction.")
            direction = tokens[0].strip()
            if direction not in CUP_DEBUG_DIRECTIONS:
                raise ValueError(
                    f"direction must be one of: {', '.join(CUP_DEBUG_DIRECTIONS)}."
                )
            arguments: dict[str, object] = {
                "action": "nudge_intensity",
                "direction": direction,
            }
            if len(tokens) == 2:
                step = _parse_required_float(tokens[1], field_name="step")
                if step <= 0:
                    raise ValueError("step must be > 0.")
                arguments["step"] = step
            payload = execute_control_cup_tool(chat_service, arguments)
            print_command_payload("CUP Intensity", payload, output_stream=output_stream)
            return True

        if parsed.name == "/cup-led":
            if not tokens or len(tokens) > 3:
                raise ValueError("Missing CUP LED color.")
            arguments = {
                "action": "set_led",
                "color": tokens[0].strip(),
            }
            if len(tokens) >= 2:
                blink_hz = _parse_required_float(tokens[1], field_name="blink_hz")
                if blink_hz < 0:
                    raise ValueError("blink_hz must be >= 0.")
                arguments["blink_hz"] = blink_hz
            if len(tokens) == 3:
                blink_mode = _parse_required_int(tokens[2], field_name="blink_mode")
                if blink_mode not in {0, 1, 2, 3}:
                    raise ValueError("blink_mode must be one of: 0, 1, 2, 3.")
                arguments["blink_mode"] = blink_mode
            payload = execute_control_cup_tool(chat_service, arguments)
            print_command_payload("CUP LED", payload, output_stream=output_stream)
            return True
    except ValueError as exc:
        print_cli_error(str(exc), output_stream=output_stream)
        usage()
        return True
    except ToolExecutionError as exc:
        print_cli_error(str(exc), output_stream=output_stream)
        print_cli_hint(
            "Tip: check [hardware.cup] base_url/token and make sure the device is reachable (default: http://esp32-cup.local).",
            output_stream=output_stream,
        )
        return True

    return False


def build_footer_meta(
    config: AppConfig,
    result: ChatTurnResult,
    *,
    total_seconds: float,
    first_token_seconds: float | None,
    session: ConversationSession,
) -> str:
    model_config = config.get_model(result.decision.model_alias)
    provider_config = config.get_provider(model_config.provider)

    parts = [f"done in {format_elapsed(total_seconds)}"]
    if first_token_seconds is not None:
        parts.append(f"first token {format_elapsed(first_token_seconds)}")
    parts.append(f"model {model_config.name} -> {model_config.model}")
    parts.append(f"provider {provider_config.name}/{provider_config.kind}")
    parts.append(f"route {result.decision.mode}")
    parts.append(f"reason {result.decision.reason}")
    parts.append(f"{len(result.assistant_message.content)} chars")
    parts.append(f"{len(session.messages)} msgs")
    return " | ".join(parts)


def render_footer_meta(
    config: AppConfig,
    result: ChatTurnResult,
    *,
    total_seconds: float,
    first_token_seconds: float | None,
    session: ConversationSession,
    output_stream: object = sys.stdout,
) -> None:
    print(
        colorize(
            build_footer_meta(
                config,
                result,
                total_seconds=total_seconds,
                first_token_seconds=first_token_seconds,
                session=session,
            ),
            SLATE,
            DIM,
        ),
        file=output_stream,
    )


def run_streaming_turn(
    chat_service: ChatService,
    store: SessionStore,
    session: ConversationSession,
    *,
    user_text: str | None = None,
    mode_override: str,
    output_stream: object = sys.stdout,
    retry: bool = False,
) -> None:
    indicator: StreamingIndicator | None = None
    result: ChatTurnResult | None = None
    try:
        route_text = user_text or ""
        if retry:
            last_user_message = find_last_user_message(session)
            if last_user_message is None:
                raise ValueError("No user message is available to retry yet.")
            route_text = last_user_message
        decision = select_model(
            chat_service.config,
            route_text,
            pinned_model=session.pinned_model,
            mode_override=mode_override,
        )
        indicator = StreamingIndicator(output_stream=output_stream)
        indicator.start()

        def handle_chunk_with_meta(chunk: str) -> None:
            indicator.begin_stream(model_alias=decision.model_alias)
            stream_printer(chunk, output_stream=output_stream)

        def handle_step(step: AgentStep) -> None:
            indicator.set_label(describe_agent_step(step))

        if retry:
            result = chat_service.retry_stream(
                session,
                mode_override=mode_override,
                on_chunk=handle_chunk_with_meta,
                on_step=handle_step,
            )
        else:
            if user_text is None:
                raise ValueError("A user message is required for a normal chat turn.")
            result = chat_service.send_stream(
                session,
                user_text,
                mode_override=mode_override,
                on_chunk=handle_chunk_with_meta,
                on_step=handle_step,
            )
        indicator.stop()
        store.save(session)
    except ValueError as exc:
        if indicator is not None:
            indicator.stop()
        print()
        print_cli_error(str(exc), output_stream=output_stream)
    except GenerationInterrupted as exc:
        if indicator is not None:
            indicator.stop()
        store.save(session)
        print(file=output_stream)
        if exc.partial_result is not None:
            print_cli_hint(
                "Generation interrupted. Partial reply saved.",
                output_stream=output_stream,
            )
            print(file=output_stream)
            print(divider(color=SLATE), file=output_stream)
            print(file=output_stream)
        else:
            print_cli_hint(
                "Generation interrupted before any reply was received.",
                output_stream=output_stream,
            )
    except KeyboardInterrupt:
        if indicator is not None:
            indicator.stop()
        store.save(session)
        print(file=output_stream)
        print_cli_hint(
            "Request interrupted before the provider finished responding.",
            output_stream=output_stream,
        )
    except (ConfigError, ProviderError) as exc:
        if indicator is not None:
            indicator.stop()
        print()
        print_cli_error(str(exc), output_stream=output_stream)
    else:
        total_seconds = 0.0
        first_token_seconds: float | None = None
        if indicator is not None:
            total_seconds = time.perf_counter() - indicator.started_at
            if indicator.first_token_at is not None:
                first_token_seconds = indicator.first_token_at - indicator.started_at
        print(file=output_stream)
        if result is not None:
            if result.ask_user is not None:
                render_ask_user_prompt(result.ask_user, output_stream=output_stream)
                print(file=output_stream)
            render_footer_meta(
                chat_service.config,
                result,
                total_seconds=total_seconds,
                first_token_seconds=first_token_seconds,
                session=session,
                output_stream=output_stream,
            )
        print(file=output_stream)
        print(divider(color=SLATE), file=output_stream)
        print(file=output_stream)


@dataclass(slots=True)
class CommandOutcome:
    handled: bool
    session: ConversationSession
    mode_override: str
    should_exit: bool = False


def ensure_valid_session_model(
    config: AppConfig,
    session: ConversationSession,
    *,
    output_stream: object = sys.stdout,
) -> bool:
    del config
    pinned_model = (session.pinned_model or "auto").strip() or "auto"
    if pinned_model == "auto":
        if session.pinned_model != "auto":
            session.pinned_model = "auto"
            return True
        return False

    print_cli_hint(
        "Session-scoped model switching is disabled; reset model override to auto.",
        output_stream=output_stream,
    )
    print_cli_hint(
        "Use `whesper chat --model <alias|provider/model>` to configure model for this run.",
        output_stream=output_stream,
    )
    session.pinned_model = "auto"
    return True


def apply_pinned_model_override(
    config: AppConfig,
    store: SessionStore,
    session: ConversationSession,
    pinned_model: str,
    *,
    output_stream: object = sys.stdout,
) -> None:
    normalized = pinned_model.strip()
    if not normalized or normalized == "auto":
        session.pinned_model = "auto"
        store.save(session)
        return

    try:
        config.get_model(normalized)
    except ConfigError:
        if "/" in normalized:
            provider_name = normalized.split("/", 1)[0]
            model_id = normalized.split("/", 1)[1]
            try:
                config.register_model(normalized, provider_name, model_id)
                session.pinned_model = normalized
                store.save(session)
                return
            except ConfigError:
                pass
        print_cli_error(
            f"Model '{normalized}' is not available.",
            output_stream=output_stream,
        )
        print_cli_hint(
            "Starting with scheduler.chat_model (session override: auto).",
            output_stream=output_stream,
        )
    else:
        session.pinned_model = normalized
        store.save(session)


def handle_command(
    parsed: ParsedCommand,
    *,
    config: AppConfig,
    store: SessionStore,
    chat_service: ChatService,
    memory_store: MemoryStore | None = None,
    trace_store: TraceStore | None = None,
    session: ConversationSession,
    mode_override: str,
    output_stream: object = sys.stdout,
    refresh_completions: Callable[[], None] | None = None,
) -> CommandOutcome:
    if memory_store is None:
        memory_store = MemoryStore(store.base_dir)
    if trace_store is None:
        trace_store = TraceStore(store.base_dir)
    try:
        if parsed.name == "/help":
            print(help_text(), file=output_stream)
            return CommandOutcome(True, session, mode_override)
        if parsed.name == "/exit":
            store.save(session)
            print("Bye.", file=output_stream)
            return CommandOutcome(True, session, mode_override, should_exit=True)
        if parsed.name == "/clear":
            clear_screen(output_stream=output_stream)
            return CommandOutcome(True, session, mode_override)
        if parsed.name == "/search":
            print_cli_error("Missing query for /search.", output_stream=output_stream)
            print_cli_hint("Usage: /search <query>", output_stream=output_stream)
            return CommandOutcome(True, session, mode_override)
        if parsed.name == "/trace":
            if parsed.arg:
                run_trace_debug_turn(
                    chat_service,
                    session,
                    user_text=parsed.arg,
                    mode_override=mode_override,
                    trace_store=trace_store,
                    output_stream=output_stream,
                )
                return CommandOutcome(True, session, mode_override)
            print_trace(trace_store, output_stream=output_stream)
            return CommandOutcome(True, session, mode_override)
        if parsed.name == "/models":
            if parsed.arg:
                provider_name = parsed.arg
                try:
                    provider = config.get_provider(provider_name)
                except ConfigError:
                    print_cli_error(
                        f"Unknown provider: {provider_name}",
                        output_stream=output_stream,
                    )
                    print_cli_hint(
                        f"available providers: {', '.join(sorted(config.providers.keys()))}",
                        output_stream=output_stream,
                    )
                    return CommandOutcome(True, session, mode_override)
                print(f"Fetching models from {provider_name}...", file=output_stream)
                try:
                    model_ids = list_provider_models(provider)
                except ProviderError as exc:
                    print_cli_error(str(exc), output_stream=output_stream)
                    return CommandOutcome(True, session, mode_override)
                if not model_ids:
                    print("  (no models returned)", file=output_stream)
                else:
                    print(
                        f"Available models on {provider_name} ({len(model_ids)}):",
                        file=output_stream,
                    )
                    for model_id in model_ids:
                        alias = f"{provider_name}/{model_id}"
                        configured = any(
                            m.provider == provider_name and m.model == model_id
                            for m in config.models.values()
                        )
                        marker = " [configured]" if configured else ""
                        print(f"  - {alias}{marker}", file=output_stream)
                    print(
                        f"\nTip: restart with `whesper chat --model {provider_name}/<model_id>` to test one model for this run",
                        file=output_stream,
                    )
            else:
                print_models(config)
            return CommandOutcome(True, session, mode_override)
        if parsed.name == "/sessions":
            print_sessions(store)
            return CommandOutcome(True, session, mode_override)
        if parsed.name == "/status":
            print_status(
                config,
                session,
                mode_override=mode_override,
                output_stream=output_stream,
            )
            return CommandOutcome(True, session, mode_override)
        if parsed.name == "/info":
            print_model_info(
                config,
                session,
                mode_override=mode_override,
                output_stream=output_stream,
            )
            return CommandOutcome(True, session, mode_override)
        if parsed.name == "/history":
            print_history(session, output_stream=output_stream)
            return CommandOutcome(True, session, mode_override)
        if parsed.name == "/memory":
            print_memory(memory_store, output_stream=output_stream)
            return CommandOutcome(True, session, mode_override)
        if parsed.name == "/remember":
            content = parsed.arg
            if not content:
                print_cli_error("Missing text for /remember.", output_stream=output_stream)
                print_cli_hint("Usage: /remember <text>", output_stream=output_stream)
                return CommandOutcome(True, session, mode_override)
            memory = MemoryService(memory_store).remember_manual(session.session_id, content)
            if refresh_completions is not None:
                refresh_completions()
            print(f"Saved memory: {memory.memory_id}", file=output_stream)
            return CommandOutcome(True, session, mode_override)
        if parsed.name == "/forget":
            memory_id = parsed.arg
            if not memory_id:
                print_cli_error("Missing memory id for /forget.", output_stream=output_stream)
                print_cli_hint("Usage: /forget <memory_id>", output_stream=output_stream)
                return CommandOutcome(True, session, mode_override)
            if not memory_store.delete(memory_id):
                print_cli_error(f"Memory not found: {memory_id}", output_stream=output_stream)
                available_memories = ", ".join(memory_store.list_memory_ids()) or "none"
                print_cli_hint(
                    f"available memories: {available_memories}",
                    output_stream=output_stream,
                )
                return CommandOutcome(True, session, mode_override)
            if refresh_completions is not None:
                refresh_completions()
            print(f"Forgot memory: {memory_id}", file=output_stream)
            return CommandOutcome(True, session, mode_override)
        if parsed.name == "/retry":
            print_cli_hint("Retrying the last user message...", output_stream=output_stream)
            run_streaming_turn(
                chat_service,
                store,
                session,
                mode_override=mode_override,
                output_stream=output_stream,
                retry=True,
            )
            return CommandOutcome(True, session, mode_override)
        if parsed.name == "/copy-last":
            last_assistant = find_last_assistant_message(session)
            if last_assistant is None:
                print_cli_error("No assistant reply is available yet.", output_stream=output_stream)
                return CommandOutcome(True, session, mode_override)
            print(last_assistant, file=output_stream)
            return CommandOutcome(True, session, mode_override)
        if handle_cup_debug_command(
            parsed,
            chat_service=chat_service,
            output_stream=output_stream,
        ):
            return CommandOutcome(True, session, mode_override)
        if parsed.name == "/websearch":
            search_service = chat_service.tool_registry.search_service
            if parsed.arg is None:
                current = search_service.provider if search_service else "duckduckgo"
                available = ", ".join(VALID_SEARCH_PROVIDERS)
                print(
                    f"Web search provider: {current}\n"
                    f"Available: {available}",
                    file=output_stream,
                )
                return CommandOutcome(True, session, mode_override)
            provider = parsed.arg.strip().casefold()
            if provider not in VALID_SEARCH_PROVIDERS:
                available = ", ".join(VALID_SEARCH_PROVIDERS)
                print_cli_error(f"Unknown provider: {parsed.arg}", output_stream=output_stream)
                print_cli_hint(f"available: {available}", output_stream=output_stream)
                return CommandOutcome(True, session, mode_override)
            if search_service is not None:
                search_service.provider = provider
                if provider == "brave":
                    import os
                    search_service.api_key = os.getenv("WHESPER_BRAVE_API_KEY")
                elif provider == "serpapi":
                    import os
                    search_service.api_key = os.getenv("WHESPER_SERPAPI_API_KEY")
                else:
                    search_service.api_key = None
            print(f"Web search provider set to: {provider}", file=output_stream)
            return CommandOutcome(True, session, mode_override)
        if parsed.name in {"/model", "/use"}:
            print_cli_error(
                "In-session model switching is disabled.",
                output_stream=output_stream,
            )
            print_cli_hint(
                "Use `whesper chat --model <alias|provider/model>` before generation.",
                output_stream=output_stream,
            )
            print_cli_hint(
                f"Current session override: {session.pinned_model or 'auto'}",
                output_stream=output_stream,
            )
            return CommandOutcome(True, session, mode_override)
        if parsed.name == "/mode":
            if not parsed.arg:
                print(
                    f"Current routing mode: {mode_override}",
                    file=output_stream,
                )
                print(
                    f"Tip: /mode <{AVAILABLE_MODES_TEXT}> to switch",
                    file=output_stream,
                )
                return CommandOutcome(True, session, mode_override)
            next_mode = parsed.arg
            if next_mode not in VALID_MODES:
                print_cli_error(f"Invalid mode: {next_mode}", output_stream=output_stream)
                print_cli_hint(
                    f"available modes: {AVAILABLE_MODES_TEXT}",
                    output_stream=output_stream,
                )
                return CommandOutcome(True, session, mode_override)
            print(f"Routing mode set to: {next_mode}", file=output_stream)
            return CommandOutcome(True, session, next_mode)
        if parsed.name == "/session":
            if not parsed.arg:
                print(f"Current session: {session.session_id}", file=output_stream)
                print(
                    f"  messages: {len(session.messages)}  "
                    f"created: {session.created_at}",
                    file=output_stream,
                )
                print(
                    "Tip: /session <session_id> to switch, /sessions to list all",
                    file=output_stream,
                )
                return CommandOutcome(True, session, mode_override)
            target_id = parsed.arg.strip()
            if not target_id:
                print_cli_error("Session ID cannot be empty.", output_stream=output_stream)
                print_cli_hint("Usage: /session <session_id>", output_stream=output_stream)
                return CommandOutcome(True, session, mode_override)
            available = store.list_sessions()
            if target_id not in available:
                print_cli_error(
                    f"Session not found: {target_id}",
                    output_stream=output_stream,
                )
                hint = ", ".join(available) if available else "none"
                print_cli_hint(
                    f"available sessions: {hint}\n"
                    "Tip: use /new <session_id> to create a new session",
                    output_stream=output_stream,
                )
                return CommandOutcome(True, session, mode_override)
            store.save(session)
            next_session = store.load(target_id)
            if ensure_valid_session_model(
                config,
                next_session,
                output_stream=output_stream,
            ):
                store.save(next_session)
            if refresh_completions is not None:
                refresh_completions()
            print(
                f"Switched to session: {next_session.session_id} "
                f"(model_override={next_session.pinned_model})",
                file=output_stream,
            )
            return CommandOutcome(True, next_session, mode_override)
        if parsed.name == "/new":
            next_session_id = parsed.arg or config.app.default_session
            store.save(session)
            next_session = store.load(next_session_id)
            if ensure_valid_session_model(
                config,
                next_session,
                output_stream=output_stream,
            ):
                store.save(next_session)
            if refresh_completions is not None:
                refresh_completions()
            print(
                f"Switched session to: {next_session.session_id} "
                f"(model_override={next_session.pinned_model})",
                file=output_stream,
            )
            return CommandOutcome(True, next_session, mode_override)
        if parsed.name == "/rename":
            next_session_id = parsed.arg
            if not next_session_id:
                print_cli_error("Missing session id for /rename.", output_stream=output_stream)
                print_cli_hint("Usage: /rename <session_id>", output_stream=output_stream)
                return CommandOutcome(True, session, mode_override)
            if next_session_id == session.session_id:
                print_cli_hint("Session name is unchanged.", output_stream=output_stream)
                return CommandOutcome(True, session, mode_override)
            session = store.rename_session(session.session_id, next_session_id)
            if refresh_completions is not None:
                refresh_completions()
            print(f"Renamed session to: {session.session_id}", file=output_stream)
            return CommandOutcome(True, session, mode_override)
        if parsed.name == "/delete-session":
            target_session_id = parsed.arg
            if not target_session_id:
                print_cli_error("Missing session id for /delete-session.", output_stream=output_stream)
                print_cli_hint("Usage: /delete-session <session_id>", output_stream=output_stream)
                return CommandOutcome(True, session, mode_override)
            if not store.delete_session(target_session_id):
                print_cli_error(f"Session not found: {target_session_id}", output_stream=output_stream)
                available_sessions = ", ".join(store.list_sessions()) or "none"
                print_cli_hint(
                    f"available sessions: {available_sessions}",
                    output_stream=output_stream,
                )
                return CommandOutcome(True, session, mode_override)
            if target_session_id == session.session_id:
                next_session = store.load(config.app.default_session)
                if ensure_valid_session_model(
                    config,
                    next_session,
                    output_stream=output_stream,
                ):
                    store.save(next_session)
                session = next_session
                print(
                    f"Deleted current session. Switched to: {session.session_id}",
                    file=output_stream,
                )
            else:
                print(f"Deleted session: {target_session_id}", file=output_stream)
            if refresh_completions is not None:
                refresh_completions()
            return CommandOutcome(True, session, mode_override)

        print(f"Unknown command: {parsed.name}. Type /help.", file=output_stream)
        return CommandOutcome(True, session, mode_override)
    except ConfigError as exc:
        print_cli_error(str(exc), output_stream=output_stream)
        if parsed.name in {"/status", "/info"}:
            print_cli_hint("Tip: check /models and /mode for the current routing state.", output_stream=output_stream)
        return CommandOutcome(True, session, mode_override)
    except (FileExistsError, FileNotFoundError) as exc:
        print_cli_error(str(exc), output_stream=output_stream)
        if parsed.name in {"/session", "/rename", "/delete-session"}:
            available_sessions = ", ".join(store.list_sessions()) or "none"
            print_cli_hint(
                f"available sessions: {available_sessions}",
                output_stream=output_stream,
            )
        return CommandOutcome(True, session, mode_override)
    except Exception as exc:  # pragma: no cover - defensive guard for interactive CLI
        print_cli_error(
            f"Unexpected command error: {exc.__class__.__name__}: {exc}",
            output_stream=output_stream,
        )
        print_cli_hint("The CLI is still running. Please try again.", output_stream=output_stream)
        return CommandOutcome(True, session, mode_override)


class StreamingIndicator:
    def __init__(
        self,
        *,
        output_stream: object,
        label: str = "whesper> thinking",
        interval_seconds: float = 0.08,
    ) -> None:
        self.output_stream = output_stream
        self.label = label
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._started_stream = False
        self.started_at = time.perf_counter()
        self.first_token_at: float | None = None
        self._label_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def begin_stream(self, *, model_alias: str) -> None:
        if self._started_stream:
            return
        self._started_stream = True
        self.first_token_at = time.perf_counter()
        self._stop_event.set()
        self._thread.join(timeout=0.3)
        print("\r\033[2K", end="", flush=True, file=self.output_stream)
        print()
        render_assistant_header(
            model_alias=model_alias,
            first_token_seconds=self.first_token_at - self.started_at,
            output_stream=self.output_stream,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.3)
        if not self._started_stream:
            print("\r\033[2K", end="", flush=True, file=self.output_stream)

    def set_label(self, label: str) -> None:
        with self._label_lock:
            self.label = label

    def _run(self) -> None:
        frames = ("   ", ".  ", ".. ", "...")
        index = 0
        while True:
            if self._stop_event.is_set():
                break
            with self._label_lock:
                label = self.label
            print(
                colorize(
                    f"\r\033[2K{label}{frames[index % len(frames)]}",
                    GOLD,
                    DIM,
                ),
                end="",
                flush=True,
                file=self.output_stream,
            )
            index += 1
            time.sleep(self.interval_seconds)


def interactive_chat(
    config: AppConfig,
    store: SessionStore,
    *,
    session_id: str | None,
    pinned_model: str | None,
) -> int:
    try:
        from prompt_toolkit import HTML, PromptSession
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import NestedCompleter
        from prompt_toolkit.formatted_text import FormattedText
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.styles import Style
    except ImportError as exc:
        print(
            "Missing dependency: prompt_toolkit. Install dependencies with "
            "`python3 -m pip install -e .` first.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    active_session_id = session_id or config.app.default_session
    session = store.load(active_session_id)
    if ensure_valid_session_model(config, session):
        store.save(session)
    if pinned_model:
        apply_pinned_model_override(config, store, session, pinned_model)

    memory_store = MemoryStore(store.base_dir)
    memory_service = MemoryService(memory_store)
    trace_store = TraceStore(store.base_dir)
    chat_service = ChatService(
        config,
        memory_service=memory_service,
        trace_store=trace_store,
    )
    mode_override = "auto"
    history_path = store.base_dir / "prompt_history.txt"

    style = Style.from_dict(
        {
            "prompt": "bold #7cc7ff",
            "brand": "bold #f5d76e",
            "status": "fg:#b8c7d9 bg:#1c2433",
            "cup_online": "fg:#d6ffd1 bg:#1c2433",
            "cup_offline": "fg:#ffb3b3 bg:#1c2433",
            "cup_unknown": "fg:#e6d08c bg:#1c2433",
            "hint": "fg:#8fa1b8 bg:#1c2433",
            "assistant": "#dce7f7",
            "meta": "#7f91aa",
            "error": "bold #ff8f8f",
        }
    )

    prompt_session = PromptSession(
        history=FileHistory(str(history_path)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=NestedCompleter.from_nested_dict(
            command_completions(config, store, memory_store)
        ),
        complete_while_typing=True,
        complete_in_thread=True,
        reserve_space_for_menu=8,
    )
    cup_heartbeat_monitor: CupHeartbeatMonitor | None = None
    if control_cup_tool_available(chat_service):
        cup_heartbeat_monitor = CupHeartbeatMonitor(chat_service)
        cup_heartbeat_monitor.start()

    def toolbar() -> FormattedText:
        awaiting_followup = " yes " if session.pending_ask_user is not None else " no "
        cup_style = "class:cup_unknown"
        cup_label = " cup n/a "
        if cup_heartbeat_monitor is not None:
            cup_style, cup_label = cup_heartbeat_monitor.toolbar_fragment()
        return FormattedText(
            [
                ("class:status", f" session {session.session_id} "),
                ("class:hint", "  "),
                ("class:status", f" model_override {session.pinned_model} "),
                ("class:hint", "  "),
                ("class:status", f" route {mode_override} "),
                ("class:hint", "  "),
                (cup_style, cup_label),
                ("class:hint", "  "),
                ("class:status", f" follow-up{awaiting_followup}"),
                ("class:hint", "  Tab complete  Ctrl+R history  / commands "),
            ]
        )

    print()
    print("Whesper CLI")
    print(
        f"Session: {session.session_id} | model_override: {session.pinned_model} | "
        "Type /help for commands"
    )
    print()
    if session.pending_ask_user is not None:
        print_cli_hint("Pending follow-up:", output_stream=sys.stdout)
        print(colorize(session.pending_ask_user.prompt, MINT), file=sys.stdout)
        render_ask_user_prompt(session.pending_ask_user, output_stream=sys.stdout)
        print()

    try:
        while True:
            try:
                prompt_label = "answer" if session.pending_ask_user is not None else "you"
                user_text = prompt_session.prompt(
                    HTML(f"<prompt>{prompt_label}</prompt> <brand>></brand> "),
                    bottom_toolbar=toolbar,
                    style=style,
                    refresh_interval=0.5,
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                store.save(session)
                return 0

            if not user_text:
                continue

            parsed = parse_command(user_text)
            if parsed is not None:
                if cup_heartbeat_monitor is not None:
                    cup_heartbeat_monitor.pause()
                try:
                    outcome = handle_command(
                        parsed,
                        config=config,
                        store=store,
                        chat_service=chat_service,
                        memory_store=memory_store,
                        trace_store=trace_store,
                        session=session,
                        mode_override=mode_override,
                        refresh_completions=lambda: setattr(
                            prompt_session,
                            "completer",
                            NestedCompleter.from_nested_dict(
                                command_completions(config, store, memory_store)
                            ),
                        ),
                    )
                finally:
                    if cup_heartbeat_monitor is not None:
                        cup_heartbeat_monitor.resume()
                session = outcome.session
                mode_override = outcome.mode_override
                if outcome.should_exit:
                    return 0
                if outcome.handled:
                    continue

            render_user_message(user_text)
            if cup_heartbeat_monitor is not None:
                cup_heartbeat_monitor.pause()
            try:
                run_streaming_turn(
                    chat_service,
                    store,
                    session,
                    user_text=user_text,
                    mode_override=mode_override,
                    output_stream=sys.stdout,
                )
            finally:
                if cup_heartbeat_monitor is not None:
                    cup_heartbeat_monitor.resume()
    finally:
        if cup_heartbeat_monitor is not None:
            cup_heartbeat_monitor.stop()


def run_server(
    config: AppConfig,
    *,
    host_override: str | None = None,
    port_override: int | None = None,
    reload: bool = False,
) -> int:
    server_settings = config.server
    if not server_settings.enabled:
        print(
            "Server is disabled. Set [server].enabled = true in your config to enable it.",
            file=sys.stderr,
        )
        return 1

    if server_settings.require_auth and not server_settings.resolved_api_token():
        print(
            "Server requires auth but no token is configured. Set [server].api_token "
            "or export the env var referenced by [server].api_token_env.",
            file=sys.stderr,
        )
        return 1

    try:
        import uvicorn  # noqa: PLC0415
        from whesper.api.application import AgentApplication  # noqa: PLC0415
        from whesper.server.http import create_app  # noqa: PLC0415
    except ImportError as exc:
        print(
            "Missing server extra. Install with `python3 -m pip install -e \".[server]\"`.",
            file=sys.stderr,
        )
        print(f"underlying error: {exc}", file=sys.stderr)
        return 1

    host = host_override or server_settings.host
    port = port_override or server_settings.port
    application = AgentApplication(config)
    app = create_app(application)
    print(
        f"Starting Whesper server on http://{host}:{port} "
        f"(auth {'on' if server_settings.require_auth else 'off'})"
    )
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        timeout_keep_alive=server_settings.request_timeout_seconds,
    )
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "chat"

    try:
        config, store = load_runtime(args)
    except (FileNotFoundError, ConfigError) as exc:
        print(colorize("Configuration error", ROSE, BOLD), file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    if command == "models":
        print_models(config)
        raise SystemExit(0)

    if command == "sessions":
        print_sessions(store)
        raise SystemExit(0)

    if command == "serve":
        exit_code = run_server(
            config,
            host_override=getattr(args, "host", None),
            port_override=getattr(args, "port", None),
            reload=getattr(args, "reload", False),
        )
        raise SystemExit(exit_code)

    exit_code = interactive_chat(
        config,
        store,
        session_id=getattr(args, "session", None),
        pinned_model=getattr(args, "model", None),
    )
    raise SystemExit(exit_code)
