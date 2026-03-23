from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
import sys
from typing import Callable

from whesper.chat import ChatService, ChatTurnResult, GenerationInterrupted
from whesper.client import ProviderError
from whesper.commands import (
    VALID_MODES,
    ParsedCommand,
    command_completions,
    help_text,
    parse_command,
)
from whesper.config import AppConfig, ConfigError, load_config
from whesper.memory import MemoryService, MemoryStore
from whesper.router import select_model
from whesper.session import ChatMessage, ConversationSession, SessionStore, utc_now_iso
from whesper.trace import TraceStore


DEFAULT_CONFIG_PATH = "whesper.toml"


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
        help="Pin the session to a model alias immediately.",
    )

    subparsers.add_parser("models", help="List configured models")
    subparsers.add_parser("sessions", help="List local saved sessions")

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
    history = session.messages[-limit:]
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
                name=message.name,
                tool_call_id=message.tool_call_id,
                tool_calls=copy.deepcopy(message.tool_calls),
            )
            for message in session.messages
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
    tool_message_format = chat_service._tool_message_format(model_config, provider_config)
    messages = chat_service._build_messages(
        shadow_session,
        model_config.system_prompt,
        tool_message_format=tool_message_format,
        user_text=user_text,
        route_mode=decision.mode,
    )
    tools = chat_service._tools_for_request(
        provider_name=provider_config.name,
        user_text=user_text,
        route_mode=decision.mode,
    )
    tool_choice = chat_service._tool_choice_for_request(
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

    result = chat_service._complete_turn(
        shadow_session,
        user_text,
        mode_override=mode_override,
    )

    print_trace(
        trace_store,
        session_id=shadow_session.session_id,
        title="Execution Timeline",
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
    print(f"pinned model: {session.pinned_model}", file=output_stream)
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

        if retry:
            result = chat_service.retry_stream(
                session,
                mode_override=mode_override,
                on_chunk=handle_chunk_with_meta,
            )
        else:
            if user_text is None:
                raise ValueError("A user message is required for a normal chat turn.")
            result = chat_service.send_stream(
                session,
                user_text,
                mode_override=mode_override,
                on_chunk=handle_chunk_with_meta,
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
    pinned_model = session.pinned_model or "auto"
    if pinned_model == "auto":
        if session.pinned_model != "auto":
            session.pinned_model = "auto"
            return True
        return False

    try:
        config.get_model(pinned_model)
    except ConfigError as exc:
        print_cli_error(str(exc), output_stream=output_stream)
        print_cli_hint("Session model was reset to auto.", output_stream=output_stream)
        print_cli_hint("Tip: run /models to inspect configured aliases.", output_stream=output_stream)
        session.pinned_model = "auto"
        return True

    return False


def apply_pinned_model_override(
    config: AppConfig,
    store: SessionStore,
    session: ConversationSession,
    pinned_model: str,
    *,
    output_stream: object = sys.stdout,
) -> None:
    try:
        config.get_model(pinned_model)
    except ConfigError as exc:
        print_cli_error(str(exc), output_stream=output_stream)
        print_cli_hint("Starting with session model: auto.", output_stream=output_stream)
        print_cli_hint("Tip: run /models to inspect configured aliases.", output_stream=output_stream)
    else:
        session.pinned_model = pinned_model
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
        if parsed.name in {"/model", "/use"}:
            alias = parsed.arg or "auto"
            if alias != "auto":
                config.get_model(alias)
            session.pinned_model = alias
            store.save(session)
            print(f"Session model set to: {alias}", file=output_stream)
            return CommandOutcome(True, session, mode_override)
        if parsed.name == "/mode":
            next_mode = parsed.arg or "auto"
            if next_mode not in VALID_MODES:
                print_cli_error(f"Invalid mode: {next_mode}", output_stream=output_stream)
                print_cli_hint(
                    f"available modes: {AVAILABLE_MODES_TEXT}",
                    output_stream=output_stream,
                )
                return CommandOutcome(True, session, mode_override)
            print(f"Routing mode set to: {next_mode}", file=output_stream)
            return CommandOutcome(True, session, next_mode)
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
                f"(pinned_model={next_session.pinned_model})",
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
        if parsed.name in {"/model", "/use"}:
            print_cli_hint("Tip: run /models to inspect configured aliases.", output_stream=output_stream)
        if parsed.name in {"/status", "/info"}:
            print_cli_hint("Tip: check /models and /mode for the current routing state.", output_stream=output_stream)
        return CommandOutcome(True, session, mode_override)
    except (FileExistsError, FileNotFoundError) as exc:
        print_cli_error(str(exc), output_stream=output_stream)
        if parsed.name in {"/rename", "/delete-session"}:
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

    def _run(self) -> None:
        frames = ("   ", ".  ", ".. ", "...")
        index = 0
        while True:
            if self._stop_event.is_set():
                break
            print(
                colorize(
                    f"\r\033[2K{self.label}{frames[index % len(frames)]}",
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

    def toolbar() -> FormattedText:
        return FormattedText(
            [
                ("class:status", f" session {session.session_id} "),
                ("class:hint", "  "),
                ("class:status", f" model {session.pinned_model} "),
                ("class:hint", "  "),
                ("class:status", f" route {mode_override} "),
                ("class:hint", "  Tab complete  Ctrl+R history  / commands "),
            ]
        )

    print()
    print("Whesper CLI")
    print(
        f"Session: {session.session_id} | pinned_model: {session.pinned_model} | "
        "Type /help for commands"
    )
    print()

    while True:
        try:
            user_text = prompt_session.prompt(
                HTML("<prompt>you</prompt> <brand>></brand> "),
                bottom_toolbar=toolbar,
                style=style,
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            store.save(session)
            return 0

        if not user_text:
            continue

        parsed = parse_command(user_text)
        if parsed is not None:
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
            session = outcome.session
            mode_override = outcome.mode_override
            if outcome.should_exit:
                return 0
            if outcome.handled:
                continue

        render_user_message(user_text)
        run_streaming_turn(
            chat_service,
            store,
            session,
            user_text=user_text,
            mode_override=mode_override,
            output_stream=sys.stdout,
        )


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

    exit_code = interactive_chat(
        config,
        store,
        session_id=getattr(args, "session", None),
        pinned_model=getattr(args, "model", None),
    )
    raise SystemExit(exit_code)
