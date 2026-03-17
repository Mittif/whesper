from __future__ import annotations

from dataclasses import dataclass

from whesper.config import AppConfig
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
    CommandSpec("/models", description="List configured models"),
    CommandSpec("/sessions", description="List saved sessions"),
    CommandSpec("/history", description="Show recent transcript"),
    CommandSpec("/use", "<alias|auto>", "Pin the current session model"),
    CommandSpec("/mode", "<auto|chat|reasoning|search>", "Override routing mode"),
    CommandSpec("/new", "[session_id]", "Create or switch session"),
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
    return ParsedCommand(name=name, arg=arg or None)


def command_completions(config: AppConfig, store: SessionStore) -> dict[str, object]:
    return {
        "/help": None,
        "/exit": None,
        "/models": None,
        "/sessions": None,
        "/history": None,
        "/use": {alias: None for alias in ("auto", *config.models.keys())},
        "/mode": {mode: None for mode in VALID_MODES},
        "/new": {session_id: None for session_id in store.list_sessions()},
    }

