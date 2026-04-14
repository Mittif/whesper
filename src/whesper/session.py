from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path

from whesper.agent_types import AskUserAction, AskUserOption


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str
    created_at: str
    model_alias: str | None = None
    route_reason: str | None = None
    reasoning_content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, object]] | None = None
    source_profile: str | None = None


def is_transcript_message(message: ChatMessage) -> bool:
    if message.role == "user":
        return True
    if message.role != "assistant":
        return False
    if message.tool_calls is not None:
        return False
    return bool(message.content.strip())


def _message_from_dict(item: dict[str, object]) -> ChatMessage:
    return ChatMessage(
        role=str(item["role"]),
        content=str(item["content"]),
        created_at=str(item["created_at"]),
        model_alias=(
            str(item["model_alias"])
            if item.get("model_alias") is not None
            else None
        ),
        route_reason=(
            str(item["route_reason"])
            if item.get("route_reason") is not None
            else None
        ),
        reasoning_content=(
            str(item["reasoning_content"])
            if item.get("reasoning_content") is not None
            else None
        ),
        name=(
            str(item["name"])
            if item.get("name") is not None
            else None
        ),
        tool_call_id=(
            str(item["tool_call_id"])
            if item.get("tool_call_id") is not None
            else None
        ),
        tool_calls=(
            item["tool_calls"]
            if isinstance(item.get("tool_calls"), list)
            else None
        ),
        source_profile=(
            str(item["source_profile"])
            if item.get("source_profile") is not None
            else None
        ),
    )


@dataclass(slots=True)
class ConversationSession:
    session_id: str
    created_at: str
    updated_at: str
    pinned_model: str = "auto"
    messages: list[ChatMessage] = field(default_factory=list)
    transcript_messages: list[ChatMessage] = field(default_factory=list)
    pending_ask_user: AskUserAction | None = None

    def append(self, message: ChatMessage) -> None:
        self.messages.append(message)
        self.updated_at = message.created_at

    def append_transcript(self, message: ChatMessage) -> None:
        self.transcript_messages.append(message)
        self.updated_at = message.created_at


def _ask_user_from_dict(item: dict[str, object] | None) -> AskUserAction | None:
    if not isinstance(item, dict):
        return None
    prompt = item.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    options_raw = item.get("options")
    options: list[AskUserOption] = []
    if isinstance(options_raw, list):
        for option_item in options_raw:
            if not isinstance(option_item, dict):
                continue
            label = option_item.get("label")
            value = option_item.get("value")
            if not isinstance(label, str) or not label.strip():
                continue
            if not isinstance(value, str) or not value.strip():
                value = label
            description = option_item.get("description")
            options.append(
                AskUserOption(
                    label=label.strip(),
                    value=value.strip(),
                    description=(
                        str(description).strip()
                        if description is not None and str(description).strip()
                        else None
                    ),
                )
            )
    allow_free_text = item.get("allow_free_text", True)
    field_name = item.get("field_name")
    return AskUserAction(
        prompt=prompt.strip(),
        options=tuple(options),
        allow_free_text=bool(allow_free_text),
        field_name=str(field_name).strip() if isinstance(field_name, str) and field_name.strip() else None,
    )


class SessionStore:
    def __init__(self, storage_dir: str | Path) -> None:
        self.base_dir = Path(storage_dir).expanduser().resolve()
        self.sessions_dir = self.base_dir / "sessions"
        self.transcripts_dir = self.base_dir / "transcripts"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, session_id: str) -> Path:
        safe_session = self._safe_session_id(session_id)
        return self.sessions_dir / f"{safe_session}.json"

    def transcript_path_for(self, session_id: str) -> Path:
        safe_session = self._safe_session_id(session_id)
        return self.transcripts_dir / f"{safe_session}.json"

    def _safe_session_id(self, session_id: str) -> str:
        safe_session = "".join(
            char for char in session_id if char.isalnum() or char in {"-", "_"}
        )
        if not safe_session:
            safe_session = "main"
        return safe_session

    def load(self, session_id: str) -> ConversationSession:
        path = self.path_for(session_id)
        transcript_path = self.transcript_path_for(session_id)
        if not path.exists():
            timestamp = utc_now_iso()
            session = ConversationSession(
                session_id=session_id,
                created_at=timestamp,
                updated_at=timestamp,
            )
            self.save(session)
            return session

        raw = json.loads(path.read_text(encoding="utf-8"))
        messages = [
            _message_from_dict(item)
            for item in raw.get("messages", [])
        ]
        if transcript_path.exists():
            transcript_raw = json.loads(transcript_path.read_text(encoding="utf-8"))
            transcript_messages = [
                _message_from_dict(item)
                for item in transcript_raw.get("messages", [])
            ]
        else:
            transcript_messages = [
                ChatMessage(
                    role=message.role,
                    content=message.content,
                    created_at=message.created_at,
                    model_alias=message.model_alias,
                    route_reason=message.route_reason,
                    reasoning_content=message.reasoning_content,
                )
                for message in messages
                if is_transcript_message(message)
            ]
        return ConversationSession(
            session_id=str(raw["session_id"]),
            created_at=str(raw["created_at"]),
            updated_at=str(raw["updated_at"]),
            pinned_model=str(raw.get("pinned_model", "auto")),
            messages=messages,
            transcript_messages=transcript_messages,
            pending_ask_user=_ask_user_from_dict(raw.get("pending_ask_user")),
        )

    def save(self, session: ConversationSession) -> None:
        path = self.path_for(session.session_id)
        transcript_path = self.transcript_path_for(session.session_id)
        payload = {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "pinned_model": session.pinned_model,
            "messages": [asdict(message) for message in session.messages],
            "pending_ask_user": (
                asdict(session.pending_ask_user)
                if session.pending_ask_user is not None
                else None
            ),
        }
        transcript_payload = {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "messages": [asdict(message) for message in session.transcript_messages],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        transcript_path.write_text(
            json.dumps(transcript_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_sessions(self) -> list[str]:
        return sorted(path.stem for path in self.sessions_dir.glob("*.json"))

    def rename_session(self, current_session_id: str, next_session_id: str) -> ConversationSession:
        current_path = self.path_for(current_session_id)
        next_path = self.path_for(next_session_id)
        current_transcript_path = self.transcript_path_for(current_session_id)
        next_transcript_path = self.transcript_path_for(next_session_id)
        if not current_path.exists():
            raise FileNotFoundError(f"Session not found: {current_session_id}")
        if next_path.exists() and next_path != current_path:
            raise FileExistsError(f"Session already exists: {next_session_id}")

        session = self.load(current_session_id)
        session.session_id = next_session_id
        self.save(session)
        if next_path != current_path and current_path.exists():
            current_path.unlink()
        if (
            next_transcript_path != current_transcript_path
            and current_transcript_path.exists()
        ):
            current_transcript_path.unlink()
        return session

    def delete_session(self, session_id: str) -> bool:
        path = self.path_for(session_id)
        transcript_path = self.transcript_path_for(session_id)
        deleted = False
        if path.exists():
            path.unlink()
            deleted = True
        if transcript_path.exists():
            transcript_path.unlink()
            deleted = True
        if not deleted:
            return False
        return True
