from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import threading
import time
import sys

from whesper.chat import ChatService
from whesper.client import ProviderError
from whesper.commands import VALID_MODES, command_completions, help_text, parse_command
from whesper.config import AppConfig, ConfigError, load_config
from whesper.session import ConversationSession, SessionStore


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


def print_history(session: ConversationSession, limit: int = 12) -> None:
    history = session.messages[-limit:]
    if not history:
        print("Session is empty.")
        return
    for message in history:
        prefix = "you" if message.role == "user" else "whesper"
        meta = f" [{message.model_alias}]" if message.model_alias else ""
        print(f"{prefix}{meta}> {message.content}")


def stream_printer(chunk: str, *, output_stream: object) -> None:
    print(chunk, end="", flush=True, file=output_stream)


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
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def begin_stream(self) -> None:
        if self._started_stream:
            return
        self._started_stream = True
        self._stop_event.set()
        self._thread.join(timeout=0.3)
        print("\r\033[2Kwhesper> ", end="", flush=True, file=self.output_stream)

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
                f"\r\033[2K{self.label}{frames[index % len(frames)]}",
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
    if pinned_model:
        config.get_model(pinned_model)
        session.pinned_model = pinned_model
        store.save(session)

    chat_service = ChatService(config)
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
        completer=NestedCompleter.from_nested_dict(command_completions(config, store)),
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
            if parsed.name == "/help":
                print(help_text())
                continue
            if parsed.name == "/exit":
                store.save(session)
                print("Bye.")
                return 0
            if parsed.name == "/models":
                print_models(config)
                continue
            if parsed.name == "/sessions":
                print_sessions(store)
                continue
            if parsed.name == "/history":
                print_history(session)
                continue
            if parsed.name == "/use":
                alias = parsed.arg or "auto"
                if alias != "auto":
                    config.get_model(alias)
                session.pinned_model = alias
                store.save(session)
                print(f"Session pinned model set to: {alias}")
                continue
            if parsed.name == "/mode":
                next_mode = parsed.arg or "auto"
                if next_mode not in VALID_MODES:
                    print("Invalid mode. Use auto, chat, reasoning, or search.")
                    continue
                mode_override = next_mode
                print(f"Routing mode set to: {mode_override}")
                continue
            if parsed.name == "/new":
                next_session_id = parsed.arg or config.app.default_session
                store.save(session)
                session = store.load(next_session_id)
                prompt_session.completer = NestedCompleter.from_nested_dict(
                    command_completions(config, store)
                )
                print(
                    f"Switched session to: {session.session_id} "
                    f"(pinned_model={session.pinned_model})"
                )
                continue

            print(f"Unknown command: {parsed.name}. Type /help.")
            continue

        try:
            print()
            indicator = StreamingIndicator(output_stream=sys.stdout)
            indicator.start()

            def handle_chunk(chunk: str) -> None:
                indicator.begin_stream()
                stream_printer(chunk, output_stream=sys.stdout)

            result = chat_service.send_stream(
                session,
                user_text,
                mode_override=mode_override,
                on_chunk=handle_chunk,
            )
            indicator.stop()
            store.save(session)
        except (ConfigError, ProviderError) as exc:
            indicator.stop()
            print()
            print(f"error> {exc}")
            continue

        print()
        print()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "chat"

    try:
        config, store = load_runtime(args)
    except (FileNotFoundError, ConfigError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
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
