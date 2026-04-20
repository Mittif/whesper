from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import re

from whesper.session import utc_now_iso


WORD_RE = re.compile(r"[A-Za-z0-9']+|[\u4e00-\u9fff]{2,}")


def _parse_iso(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _slug(value: str) -> str:
    collapsed = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return collapsed or "memory"


def _memory_id(memory_type: str, content: str) -> str:
    digest = hashlib.sha1(f"{memory_type}:{content.strip().lower()}".encode("utf-8")).hexdigest()
    return f"{_slug(memory_type)}-{digest[:10]}"


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in WORD_RE.findall(text):
        lowered = token.lower()
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
            tokens.add(token)
            continue
        if len(lowered) >= 3:
            tokens.add(lowered)
    return tokens


def _truncate_fact(value: str, *, limit: int = 120) -> str:
    cleaned = " ".join(value.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _future_iso(*, hours: int = 0, days: int = 0) -> str:
    return (datetime.now(UTC) + timedelta(hours=hours, days=days)).replace(
        microsecond=0
    ).isoformat()


def _as_mapping(value: object) -> dict[str, object] | None:
    return value if isinstance(value, dict) else None


def _coerce_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "on", "enabled", "enable", "yes"}:
            return True
        if normalized in {"false", "0", "off", "disabled", "disable", "no"}:
            return False
    return None


def _normalize_hex_color(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if re.fullmatch(r"#[0-9a-f]{6}", normalized):
        return normalized
    return None


@dataclass(slots=True)
class MemoryItem:
    memory_id: str
    memory_type: str
    title: str
    content: str
    source: str
    confidence: float
    created_at: str
    updated_at: str
    last_confirmed_at: str | None = None
    expires_at: str | None = None
    session_id: str | None = None

    def is_active(self, *, now: datetime | None = None) -> bool:
        expires_at = _parse_iso(self.expires_at)
        if expires_at is None:
            return True
        reference = now or datetime.now(UTC)
        return expires_at >= reference


@dataclass(slots=True)
class MemoryCandidate:
    memory_type: str
    title: str
    content: str
    confidence: float
    expires_at: str | None = None
    memory_key: str | None = None


class MemoryStore:
    def __init__(self, storage_dir: str | Path) -> None:
        self.base_dir = Path(storage_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.base_dir / "memory.json"

    def list_memories(self, *, include_expired: bool = False) -> list[MemoryItem]:
        memories = self._load_all()
        if not include_expired:
            memories = [memory for memory in memories if memory.is_active()]
        memories.sort(
            key=lambda item: (
                _parse_iso(item.updated_at) or datetime.min.replace(tzinfo=UTC),
                item.confidence,
            ),
            reverse=True,
        )
        return memories

    def list_memory_ids(self) -> list[str]:
        return [memory.memory_id for memory in self.list_memories(include_expired=True)]

    def upsert(self, memory: MemoryItem) -> MemoryItem:
        memories = self._load_all()
        for index, existing in enumerate(memories):
            if existing.memory_id == memory.memory_id:
                memories[index] = memory
                self._save_all(memories)
                return memory
        memories.append(memory)
        self._save_all(memories)
        return memory

    def remember(
        self,
        *,
        memory_type: str,
        title: str,
        content: str,
        source: str,
        confidence: float,
        session_id: str | None = None,
        expires_at: str | None = None,
        memory_key: str | None = None,
    ) -> MemoryItem:
        now = utc_now_iso()
        memory_id = _memory_id(memory_type, memory_key or content)

        for existing in self._load_all():
            if existing.memory_id != memory_id:
                continue
            existing.title = title
            existing.content = content
            existing.source = source
            existing.confidence = confidence
            existing.updated_at = now
            existing.last_confirmed_at = now
            existing.expires_at = expires_at
            existing.session_id = session_id
            return self.upsert(existing)

        memory = MemoryItem(
            memory_id=memory_id,
            memory_type=memory_type,
            title=title,
            content=content,
            source=source,
            confidence=confidence,
            created_at=now,
            updated_at=now,
            last_confirmed_at=now,
            expires_at=expires_at,
            session_id=session_id,
        )
        return self.upsert(memory)

    def delete(self, memory_id: str) -> bool:
        memories = self._load_all()
        next_memories = [memory for memory in memories if memory.memory_id != memory_id]
        if len(next_memories) == len(memories):
            return False
        self._save_all(next_memories)
        return True

    def _load_all(self) -> list[MemoryItem]:
        if not self.path.exists():
            return []

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        items = raw.get("memories", []) if isinstance(raw, dict) else raw
        result: list[MemoryItem] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            result.append(
                MemoryItem(
                    memory_id=str(item["memory_id"]),
                    memory_type=str(item["memory_type"]),
                    title=str(item["title"]),
                    content=str(item["content"]),
                    source=str(item["source"]),
                    confidence=float(item["confidence"]),
                    created_at=str(item["created_at"]),
                    updated_at=str(item["updated_at"]),
                    last_confirmed_at=(
                        str(item["last_confirmed_at"])
                        if item.get("last_confirmed_at") is not None
                        else None
                    ),
                    expires_at=(
                        str(item["expires_at"]) if item.get("expires_at") is not None else None
                    ),
                    session_id=(
                        str(item["session_id"]) if item.get("session_id") is not None else None
                    ),
                )
            )
        return result

    def _save_all(self, memories: list[MemoryItem]) -> None:
        payload = {"memories": [asdict(memory) for memory in memories]}
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class MemoryService:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def remember_manual(self, session_id: str, content: str) -> MemoryItem:
        text = _truncate_fact(content, limit=240)
        return self.store.remember(
            memory_type="profile_memory",
            title="Saved Note",
            content=text,
            source="manual_command",
            confidence=1.0,
            session_id=session_id,
        )

    def capture_user_message(self, session_id: str, user_text: str) -> list[MemoryItem]:
        saved: list[MemoryItem] = []
        seen_ids: set[str] = set()
        for candidate in extract_memory_candidates(user_text):
            memory = self.store.remember(
                memory_type=candidate.memory_type,
                title=candidate.title,
                content=candidate.content,
                source="user_message",
                confidence=candidate.confidence,
                session_id=session_id,
                expires_at=candidate.expires_at,
                memory_key=candidate.memory_key,
            )
            if memory.memory_id in seen_ids:
                continue
            seen_ids.add(memory.memory_id)
            saved.append(memory)
        return saved

    def capture_tool_result(
        self,
        session_id: str,
        tool_name: str,
        content: str,
    ) -> list[MemoryItem]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict):
            return []

        saved: list[MemoryItem] = []
        seen_ids: set[str] = set()
        for candidate in extract_tool_memory_candidates(
            tool_name,
            payload,
            session_id=session_id,
        ):
            memory = self.store.remember(
                memory_type=candidate.memory_type,
                title=candidate.title,
                content=candidate.content,
                source="tool_result",
                confidence=candidate.confidence,
                session_id=session_id,
                expires_at=candidate.expires_at,
                memory_key=candidate.memory_key,
            )
            if memory.memory_id in seen_ids:
                continue
            seen_ids.add(memory.memory_id)
            saved.append(memory)
        return saved

    def relevant_memories(
        self,
        query: str,
        *,
        limit: int = 6,
        session_id: str | None = None,
    ) -> list[MemoryItem]:
        memories = self.store.list_memories()
        if not memories:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return self._fallback_memories(memories, limit=limit, session_id=session_id)

        scored: list[tuple[int, int, float, datetime, MemoryItem]] = []
        for memory in memories:
            haystack = f"{memory.title} {memory.content}"
            overlap = len(query_tokens & _tokenize(haystack))
            same_session = 1 if session_id is not None and memory.session_id == session_id else 0
            updated_at = _parse_iso(memory.updated_at) or datetime.min.replace(tzinfo=UTC)
            scored.append((overlap, same_session, memory.confidence, updated_at, memory))

        scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
        matched = [memory for overlap, _, _, _, memory in scored if overlap > 0]
        if matched:
            return matched[:limit]
        return self._fallback_memories(memories, limit=limit, session_id=session_id)

    def _fallback_memories(
        self,
        memories: list[MemoryItem],
        *,
        limit: int,
        session_id: str | None,
    ) -> list[MemoryItem]:
        if session_id is not None:
            same_session = [memory for memory in memories if memory.session_id == session_id]
            if same_session:
                return same_session[: min(limit, 3)]
        return memories[: min(limit, 3)]

    def build_prompt_context(
        self,
        query: str,
        *,
        limit: int = 6,
        session_id: str | None = None,
    ) -> str | None:
        memories = self.relevant_memories(query, limit=limit, session_id=session_id)
        if not memories:
            return None

        lines = [
            "Known memory about the user and recent session context. Use it when relevant, and ask a quick follow-up if it may be outdated:",
        ]
        for memory in memories:
            lines.append(f"- [{memory.memory_type}] {memory.content}")
        return "\n".join(lines)


def extract_memory_candidates(user_text: str) -> list[MemoryCandidate]:
    text = " ".join(user_text.strip().split())
    if not text:
        return []

    candidates: list[MemoryCandidate] = []
    lower = text.lower()

    name_match = re.search(r"\bmy name is (?P<fact>[^.!?\n]+)", text, re.IGNORECASE)
    if name_match:
        name = _truncate_fact(name_match.group("fact"))
        candidates.append(
            MemoryCandidate(
                memory_type="profile_memory",
                title="Name",
                content=f"The user's name is {name}.",
                confidence=0.98,
            )
        )

    location_match = re.search(r"\bi live in (?P<fact>[^.!?\n]+)", text, re.IGNORECASE)
    if location_match:
        location = _truncate_fact(location_match.group("fact"))
        candidates.append(
            MemoryCandidate(
                memory_type="profile_memory",
                title="Location",
                content=f"The user lives in {location}.",
                confidence=0.86,
            )
        )

    work_match = re.search(r"\bi work (?P<prep>as|at) (?P<fact>[^.!?\n]+)", text, re.IGNORECASE)
    if work_match:
        prep = work_match.group("prep").lower()
        work_fact = _truncate_fact(work_match.group("fact"))
        candidates.append(
            MemoryCandidate(
                memory_type="profile_memory",
                title="Work",
                content=f"The user works {prep} {work_fact}.",
                confidence=0.82,
            )
        )

    preference_match = re.search(
        r"\b(?:i like|i love|i enjoy|i prefer) (?P<fact>[^.!?\n]+)",
        text,
        re.IGNORECASE,
    )
    if preference_match:
        preference = _truncate_fact(preference_match.group("fact"))
        candidates.append(
            MemoryCandidate(
                memory_type="preference_memory",
                title="Preference",
                content=f"The user likes {preference}.",
                confidence=0.78,
            )
        )

    if any(marker in lower for marker in ("today ", "recently ", "lately ", "this week", "yesterday ")):
        candidates.append(
            MemoryCandidate(
                memory_type="episodic_memory",
                title="Recent Update",
                content=_truncate_fact(text, limit=180),
                confidence=0.68,
                expires_at=(
                    datetime.now(UTC) + timedelta(days=14)
                ).replace(microsecond=0).isoformat(),
            )
        )

    unique: dict[str, MemoryCandidate] = {}
    for candidate in candidates:
        unique[_memory_id(candidate.memory_type, candidate.memory_key or candidate.content)] = candidate
    return list(unique.values())


_COLOR_LABELS: dict[str, str] = {
    "#0000ff": "blue (蓝色)",
    "#ff0000": "red (红色)",
    "#00ff00": "green (绿色)",
    "#ff69b4": "pink (粉色)",
    "#ffb460": "warm white (暖白)",
    "#ffb36b": "warm amber (暖橙)",
    "#40c4ff": "bright cyan (亮青蓝)",
    "#ff3b30": "intense red (强烈红)",
    "#7fd8ff": "cool blue (冷蓝)",
}


def _describe_color(hex_color: str | None) -> str | None:
    if hex_color is None:
        return None
    label = _COLOR_LABELS.get(hex_color)
    return f"{label} ({hex_color})" if label is not None else hex_color


def _cup_led_color_from_payload(payload: dict[str, object]) -> str | None:
    for container in (
        _as_mapping(payload.get("requested")),
        _as_mapping(payload.get("led")),
        _as_mapping(payload.get("result")),
        _as_mapping(payload.get("led_result")),
    ):
        if container is None:
            continue
        for key in ("color", "led_color"):
            color = _normalize_hex_color(container.get(key))
            if color is not None:
                return color
    return None


def _cup_motor_target_from_payload(payload: dict[str, object]) -> float | None:
    for container in (
        _as_mapping(payload.get("motor")),
        _as_mapping(payload.get("requested")),
        _as_mapping(payload.get("motor_result")),
        _as_mapping(payload.get("result")),
    ):
        if container is None:
            continue
        for key in ("target_velocity", "target", "vel"):
            target = _coerce_number(container.get(key))
            if target is not None:
                return round(target, 3)
    target = _coerce_number(payload.get("target_velocity"))
    return round(target, 3) if target is not None else None


def _cup_motor_enabled_from_payload(payload: dict[str, object]) -> bool | None:
    if payload.get("action") == "stop":
        return False
    for container in (
        _as_mapping(payload.get("motor")),
        _as_mapping(payload.get("requested")),
        _as_mapping(payload.get("motor_result")),
        _as_mapping(payload.get("result")),
    ):
        if container is None:
            continue
        enabled = _coerce_bool(container.get("enabled"))
        if enabled is not None:
            return enabled
    return None


def extract_tool_memory_candidates(
    tool_name: str,
    payload: dict[str, object],
    *,
    session_id: str,
) -> list[MemoryCandidate]:
    if tool_name != "control_cup" or payload.get("ok") is not True:
        return []
    action = payload.get("action")
    if not isinstance(action, str):
        return []

    candidates: list[MemoryCandidate] = []
    session_prefix = f"{session_id}:control_cup"
    color_description = _describe_color(_cup_led_color_from_payload(payload))
    motor_target = _cup_motor_target_from_payload(payload)
    motor_enabled = _cup_motor_enabled_from_payload(payload)

    if color_description is not None:
        candidates.append(
            MemoryCandidate(
                memory_type="device_state_memory",
                title="CUP LED State",
                content=f"The CUP LED is currently set to {color_description}.",
                confidence=0.9,
                expires_at=_future_iso(hours=12),
                memory_key=f"{session_prefix}:led_state",
            )
        )

    if motor_enabled is not None or motor_target is not None:
        if motor_enabled is False or (motor_target is not None and motor_target <= 0):
            motor_content = "The CUP motor is currently stopped."
        elif motor_target is not None:
            motor_content = (
                f"The CUP motor is currently enabled with target velocity {motor_target:g}."
            )
        else:
            motor_content = "The CUP motor is currently enabled."
        candidates.append(
            MemoryCandidate(
                memory_type="device_state_memory",
                title="CUP Motor State",
                content=motor_content,
                confidence=0.9,
                expires_at=_future_iso(hours=12),
                memory_key=f"{session_prefix}:motor_state",
            )
        )

    if action == "set_led" and color_description is not None:
        candidates.append(
            MemoryCandidate(
                memory_type="device_preference_memory",
                title="CUP Lighting Preference",
                content=f"The user recently chose {color_description} lighting for the CUP.",
                confidence=0.78,
                expires_at=_future_iso(days=7),
                memory_key=f"{session_prefix}:led_preference",
            )
        )

    if action == "apply_scene":
        scene = payload.get("scene")
        if isinstance(scene, str) and scene.strip():
            scene_name = scene.strip()
            candidates.extend(
                (
                    MemoryCandidate(
                        memory_type="device_state_memory",
                        title="CUP Scene State",
                        content=f"The CUP is currently using the {scene_name} scene.",
                        confidence=0.92,
                        expires_at=_future_iso(hours=12),
                        memory_key=f"{session_prefix}:scene_state",
                    ),
                    MemoryCandidate(
                        memory_type="device_preference_memory",
                        title="CUP Scene Preference",
                        content=f"The user recently chose the {scene_name} scene for the CUP.",
                        confidence=0.8,
                        expires_at=_future_iso(days=7),
                        memory_key=f"{session_prefix}:scene_preference",
                    ),
                )
            )

    if action == "nudge_intensity":
        direction = payload.get("direction")
        if direction == "up":
            preference_text = "The user recently asked for stronger CUP stimulation."
        elif direction == "down":
            preference_text = "The user recently asked for gentler CUP stimulation."
        else:
            preference_text = None
        if preference_text is not None:
            candidates.append(
                MemoryCandidate(
                    memory_type="device_preference_memory",
                    title="CUP Intensity Preference",
                    content=preference_text,
                    confidence=0.76,
                    expires_at=_future_iso(days=3),
                    memory_key=f"{session_prefix}:intensity_preference",
                )
            )

    unique: dict[str, MemoryCandidate] = {}
    for candidate in candidates:
        unique[_memory_id(candidate.memory_type, candidate.memory_key or candidate.content)] = candidate
    return list(unique.values())
