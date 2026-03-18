from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str
    created_at: str
    model_alias: str | None = None
    route_reason: str | None = None


@dataclass(slots=True)
class ConversationSession:
    session_id: str
    created_at: str
    updated_at: str
    pinned_model: str = "auto"
    messages: list[ChatMessage] = field(default_factory=list)

    def append(self, message: ChatMessage) -> None:
        self.messages.append(message)
        self.updated_at = message.created_at


class SessionStore:
    def __init__(self, storage_dir: str | Path) -> None:
        self.base_dir = Path(storage_dir).expanduser().resolve()
        self.sessions_dir = self.base_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, session_id: str) -> Path:
        safe_session = "".join(
            char for char in session_id if char.isalnum() or char in {"-", "_"}
        )
        if not safe_session:
            safe_session = "main"
        return self.sessions_dir / f"{safe_session}.json"

    def load(self, session_id: str) -> ConversationSession:
        path = self.path_for(session_id)
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
        return ConversationSession(
            session_id=str(raw["session_id"]),
            created_at=str(raw["created_at"]),
            updated_at=str(raw["updated_at"]),
            pinned_model=str(raw.get("pinned_model", "auto")),
            messages=[
                ChatMessage(
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
                )
                for item in raw.get("messages", [])
            ],
        )

    def save(self, session: ConversationSession) -> None:
        path = self.path_for(session.session_id)
        payload = {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "pinned_model": session.pinned_model,
            "messages": [asdict(message) for message in session.messages],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_sessions(self) -> list[str]:
        return sorted(path.stem for path in self.sessions_dir.glob("*.json"))

    def rename_session(self, current_session_id: str, next_session_id: str) -> ConversationSession:
        current_path = self.path_for(current_session_id)
        next_path = self.path_for(next_session_id)
        if not current_path.exists():
            raise FileNotFoundError(f"Session not found: {current_session_id}")
        if next_path.exists() and next_path != current_path:
            raise FileExistsError(f"Session already exists: {next_session_id}")

        session = self.load(current_session_id)
        session.session_id = next_session_id
        self.save(session)
        if next_path != current_path and current_path.exists():
            current_path.unlink()
        return session

    def delete_session(self, session_id: str) -> bool:
        path = self.path_for(session_id)
        if not path.exists():
            return False
        path.unlink()
        return True
