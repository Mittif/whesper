from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tomllib
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass(slots=True)
class ProviderConfig:
    name: str
    kind: str
    base_url: str
    api_key: str | None = None
    api_key_env: str | None = None
    timeout_seconds: int = 120
    extra_headers: dict[str, str] = field(default_factory=dict)

    def resolved_api_key(self) -> str | None:
        if self.api_key is not None:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None


@dataclass(slots=True)
class ModelConfig:
    name: str
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None
    think: bool | str | None = None
    system_prompt: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(slots=True)
class SchedulerConfig:
    chat_model: str
    reasoning_model: str | None = None
    search_model: str | None = None
    long_message_chars: int = 800
    reasoning_keywords: tuple[str, ...] = ()


@dataclass(slots=True)
class PersonaConfig:
    name: str = "Whesper"
    system_prompt: str = (
        "You are Whesper, a warm, attentive, emotionally aware AI companion. "
        "Keep replies natural, supportive, and conversational."
    )


@dataclass(slots=True)
class AppSettings:
    storage_dir: str = ".whesper"
    default_session: str = "main"
    history_limit: int = 16


@dataclass(slots=True)
class AppConfig:
    app: AppSettings
    persona: PersonaConfig
    scheduler: SchedulerConfig
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    source_path: Path

    def get_provider(self, name: str) -> ProviderConfig:
        try:
            return self.providers[name]
        except KeyError as exc:
            raise ConfigError(f"Unknown provider: {name}") from exc

    def get_model(self, name: str) -> ModelConfig:
        try:
            return self.models[name]
        except KeyError as exc:
            raise ConfigError(f"Unknown model alias: {name}") from exc


def _require_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid [{key}] section in config.")
    return value


def _to_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError("Expected a list of strings in config.")
    return tuple(value)


def _optional_think(value: Any) -> bool | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    raise ConfigError("Expected 'think' to be a boolean or string.")


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)

    app_raw = raw.get("app", {})
    persona_raw = raw.get("persona", {})
    scheduler_raw = _require_table(raw, "scheduler")
    providers_raw = _require_table(raw, "providers")
    models_raw = _require_table(raw, "models")

    app = AppSettings(
        storage_dir=str(app_raw.get("storage_dir", ".whesper")),
        default_session=str(app_raw.get("default_session", "main")),
        history_limit=int(app_raw.get("history_limit", 16)),
    )
    persona = PersonaConfig(
        name=str(persona_raw.get("name", "Whesper")),
        system_prompt=str(persona_raw.get("system_prompt", PersonaConfig.system_prompt)),
    )
    scheduler = SchedulerConfig(
        chat_model=str(scheduler_raw["chat_model"]),
        reasoning_model=(
            str(scheduler_raw["reasoning_model"])
            if scheduler_raw.get("reasoning_model")
            else None
        ),
        search_model=(
            str(scheduler_raw["search_model"])
            if scheduler_raw.get("search_model")
            else None
        ),
        long_message_chars=int(scheduler_raw.get("long_message_chars", 800)),
        reasoning_keywords=_to_tuple(scheduler_raw.get("reasoning_keywords")),
    )

    providers: dict[str, ProviderConfig] = {}
    for name, item in providers_raw.items():
        if not isinstance(item, dict):
            raise ConfigError(f"Provider config for {name} must be a table.")
        providers[name] = ProviderConfig(
            name=name,
            kind=str(item.get("kind", "openai_compatible")),
            base_url=str(item["base_url"]).rstrip("/"),
            api_key=str(item["api_key"]) if item.get("api_key") is not None else None,
            api_key_env=(
                str(item["api_key_env"]) if item.get("api_key_env") is not None else None
            ),
            timeout_seconds=int(item.get("timeout_seconds", 120)),
            extra_headers={
                str(key): str(value)
                for key, value in item.get("extra_headers", {}).items()
            },
        )

    models: dict[str, ModelConfig] = {}
    for name, item in models_raw.items():
        if not isinstance(item, dict):
            raise ConfigError(f"Model config for {name} must be a table.")
        models[name] = ModelConfig(
            name=name,
            provider=str(item["provider"]),
            model=str(item["model"]),
            temperature=float(item.get("temperature", 0.7)),
            max_tokens=int(item["max_tokens"]) if item.get("max_tokens") is not None else None,
            top_p=float(item["top_p"]) if item.get("top_p") is not None else None,
            think=_optional_think(item.get("think")),
            system_prompt=str(item["system_prompt"]) if item.get("system_prompt") else None,
            tags=_to_tuple(item.get("tags")),
        )

    config = AppConfig(
        app=app,
        persona=persona,
        scheduler=scheduler,
        providers=providers,
        models=models,
        source_path=config_path,
    )

    if scheduler.chat_model not in models:
        raise ConfigError("scheduler.chat_model must refer to a defined model.")
    if scheduler.reasoning_model and scheduler.reasoning_model not in models:
        raise ConfigError("scheduler.reasoning_model must refer to a defined model.")
    if scheduler.search_model and scheduler.search_model not in models:
        raise ConfigError("scheduler.search_model must refer to a defined model.")

    for model in models.values():
        if model.provider not in providers:
            raise ConfigError(
                f"Model alias '{model.name}' refers to unknown provider '{model.provider}'."
            )

    return config
