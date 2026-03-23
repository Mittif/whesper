from __future__ import annotations

from dataclasses import dataclass

from whesper.config import AppConfig
from whesper.memory import MemoryStore
from whesper.session import SessionStore


VALID_MODES = ("auto", "chat", "reasoning", "search")


@dataclass(slots=True)
class CommandSpec:
    name: str
    args: str = ""
    description: str = ""


COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec("/help", description="Show available commands"),
    CommandSpec("/exit", description="Exit the CLI"),
    CommandSpec("/clear", description="Clear the current terminal screen"),
    CommandSpec("/search", "<query>", "Run a web-backed search turn"),
    CommandSpec("/trace", "[message]", "Show recent trace or debug one message step-by-step"),
    CommandSpec("/models", description="List configured models"),
    CommandSpec("/sessions", description="List saved sessions"),
    CommandSpec("/status", description="Show current session and routing state"),
    CommandSpec("/info", description="Show the active model configuration"),
    CommandSpec("/history", description="Show recent transcript"),
    CommandSpec("/memory", description="Show saved user memory"),
    CommandSpec("/remember", "<text>", "Save a memory note for future chats"),
    CommandSpec("/forget", "<memory_id>", "Delete a saved memory item"),
    CommandSpec("/retry", description="Regenerate the last assistant reply"),
    CommandSpec("/copy-last", description="Print the last assistant reply as plain text"),
    CommandSpec("/model", "<alias|auto>", "Switch the current session model"),
    CommandSpec("/use", "<alias|auto>", "Pin the current session model"),
    CommandSpec("/mode", "<auto|chat|reasoning|search>", "Override routing mode"),
    CommandSpec("/new", "[session_id]", "Create or switch session"),
    CommandSpec("/rename", "<session_id>", "Rename the current session"),
    CommandSpec("/delete-session", "<session_id>", "Delete a saved session"),
)


@dataclass(slots=True)
class ParsedCommand:
    name: str
    arg: str | None = None


def help_text() -> str:
    lines = ["Commands:"]
    for spec in COMMAND_SPECS:
        label = spec.name if not spec.args else f"{spec.name} {spec.args}"
        lines.append(f"  {label:<28} {spec.description}")
    lines.append("")
    lines.append("Tips:")
    lines.append("  Up/Down history, Ctrl+R search, Tab completion, / for commands")
    return "\n".join(lines)


def parse_command(text: str) -> ParsedCommand | None:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None

    parts = stripped.split(maxsplit=1)
    name = parts[0]
    arg = parts[1].strip() if len(parts) > 1 else None
    if name == "/search" and arg:
        return None
    return ParsedCommand(name=name, arg=arg or None)


def command_completions(
    config: AppConfig,
    store: SessionStore,
    memory_store: MemoryStore | None = None,
) -> dict[str, object]:
    memory_ids = memory_store.list_memory_ids() if memory_store is not None else []
    return {
        "/help": None,
        "/exit": None,
        "/clear": None,
        "/search": None,
        "/trace": None,
        "/models": None,
        "/sessions": None,
        "/status": None,
        "/info": None,
        "/history": None,
        "/memory": None,
        "/remember": None,
        "/forget": {memory_id: None for memory_id in memory_ids},
        "/retry": None,
        "/copy-last": None,
        "/model": {alias: None for alias in ("auto", *config.models.keys())},
        "/use": {alias: None for alias in ("auto", *config.models.keys())},
        "/mode": {mode: None for mode in VALID_MODES},
        "/new": {session_id: None for session_id in store.list_sessions()},
        "/rename": {session_id: None for session_id in store.list_sessions()},
        "/delete-session": {session_id: None for session_id in store.list_sessions()},
    }
