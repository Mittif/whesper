from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from whesper.session import utc_now_iso


@dataclass(slots=True)
class TraceEvent:
    timestamp: str
    kind: str
    session_id: str
    model_alias: str
    provider_name: str
    route_mode: str
    streamed: bool
    tools_enabled: bool
    tool_choice: str | None = None
    note: str | None = None
    preview: str | None = None


class TraceStore:
    def __init__(self, storage_dir: str | Path) -> None:
        self.base_dir = Path(storage_dir).expanduser().resolve()
        self.trace_dir = self.base_dir / "traces"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.trace_dir / "model_trace.jsonl"

    def append(self, event: TraceEvent) -> None:
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

    def tail(
        self,
        limit: int = 20,
        *,
        session_id: str | None = None,
    ) -> list[TraceEvent]:
        if not self.trace_path.exists():
            return []
        lines = self.trace_path.read_text(encoding="utf-8").splitlines()
        events: list[TraceEvent] = []
        for line in lines:
            raw = json.loads(line)
            event = TraceEvent(
                timestamp=str(raw["timestamp"]),
                kind=str(raw["kind"]),
                session_id=str(raw["session_id"]),
                model_alias=str(raw["model_alias"]),
                provider_name=str(raw["provider_name"]),
                route_mode=str(raw["route_mode"]),
                streamed=bool(raw["streamed"]),
                tools_enabled=bool(raw["tools_enabled"]),
                tool_choice=(
                    str(raw["tool_choice"])
                    if raw.get("tool_choice") is not None
                    else None
                ),
                note=str(raw["note"]) if raw.get("note") is not None else None,
                preview=str(raw["preview"]) if raw.get("preview") is not None else None,
            )
            if session_id is not None and event.session_id != session_id:
                continue
            events.append(event)
        return events[-limit:]


def make_trace_event(
    *,
    kind: str,
    session_id: str,
    model_alias: str,
    provider_name: str,
    route_mode: str,
    streamed: bool,
    tools_enabled: bool,
    tool_choice: str | None = None,
    note: str | None = None,
    preview: str | None = None,
) -> TraceEvent:
    return TraceEvent(
        timestamp=utc_now_iso(),
        kind=kind,
        session_id=session_id,
        model_alias=model_alias,
        provider_name=provider_name,
        route_mode=route_mode,
        streamed=streamed,
        tools_enabled=tools_enabled,
        tool_choice=tool_choice,
        note=note,
        preview=preview,
    )
