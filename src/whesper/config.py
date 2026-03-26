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
class LiveContextEndpointConfig:
    name: str
    url_template: str
    trigger_keywords: tuple[str, ...] = ()
    timeout_seconds: int = 15
    response_format: str = "json"
    query_param: str | None = None
    instruction: str | None = None
    api_key_env: str | None = None
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class LiveContextSettings:
    custom_api: dict[str, LiveContextEndpointConfig] = field(default_factory=dict)
    status_api: dict[str, LiveContextEndpointConfig] = field(default_factory=dict)


@dataclass(slots=True)
class ShellSandboxSettings:
    enabled: bool = False
    timeout_seconds: int = 10
    max_output_chars: int = 8000
    allowed_roots: tuple[str, ...] = (".",)
    allowed_command_prefixes: tuple[tuple[str, ...], ...] = ()


@dataclass(slots=True)
class AppConfig:
    app: AppSettings
    persona: PersonaConfig
    scheduler: SchedulerConfig
    live_context: LiveContextSettings
    shell_sandbox: ShellSandboxSettings
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    source_path: Path

    def get_provider(self, name: str) -> ProviderConfig:
        try:
            return self.providers[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.providers.keys()))
            if available:
                raise ConfigError(
                    f"Unknown provider: {name}. available providers: {available}"
                ) from exc
            raise ConfigError(
                f"Unknown provider: {name}. no providers are configured."
            ) from exc

    def get_model(self, name: str) -> ModelConfig:
        try:
            return self.models[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.models.keys()))
            if available:
                raise ConfigError(
                    f"Unknown model alias: {name}. available models: {available}"
                ) from exc
            raise ConfigError(
                f"Unknown model alias: {name}. no models are configured."
            ) from exc


def _available_items_text(items: dict[str, object]) -> str:
    names = ", ".join(sorted(items.keys()))
    return names or "none"


def _scheduler_model_error(field_name: str, alias: str, models: dict[str, ModelConfig]) -> ConfigError:
    available = _available_items_text(models)
    if models:
        return ConfigError(
            f"{field_name} refers to unknown model alias '{alias}'. "
            f"available models: {available}. "
            f"Update [scheduler] {field_name.split('.')[-1]} or add [models.\"{alias}\"]."
        )
    return ConfigError(
        f"{field_name} refers to unknown model alias '{alias}', but no models are configured. "
        "Add at least one [models.<alias>] section and point scheduler.chat_model to it."
    )


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


def _parse_shell_command_prefixes(raw: Any) -> tuple[tuple[str, ...], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ConfigError(
            "shell_sandbox.allowed_command_prefixes must be a list of string lists."
        )

    prefixes: list[tuple[str, ...]] = []
    for item in raw:
        if not isinstance(item, list) or not item or not all(
            isinstance(part, str) and part.strip() for part in item
        ):
            raise ConfigError(
                "shell_sandbox.allowed_command_prefixes entries must be non-empty string lists."
            )
        prefixes.append(tuple(part.strip() for part in item))
    return tuple(prefixes)


def _parse_live_context_endpoints(raw: Any, section_name: str) -> dict[str, LiveContextEndpointConfig]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Invalid [live_context.{section_name}] section in config.")

    endpoints: dict[str, LiveContextEndpointConfig] = {}
    for name, item in raw.items():
        if not isinstance(item, dict):
            raise ConfigError(f"live_context.{section_name}.{name} must be a table.")
        url_template = item.get("url_template")
        if not isinstance(url_template, str) or not url_template.strip():
            raise ConfigError(
                f"live_context.{section_name}.{name} must define a non-empty url_template."
            )
        response_format = str(item.get("response_format", "json"))
        if response_format not in {"json", "text"}:
            raise ConfigError(
                f"live_context.{section_name}.{name} response_format must be 'json' or 'text'."
            )
        endpoints[name] = LiveContextEndpointConfig(
            name=name,
            url_template=url_template.strip(),
            trigger_keywords=_to_tuple(item.get("trigger_keywords")),
            timeout_seconds=int(item.get("timeout_seconds", 15)),
            response_format=response_format,
            query_param=str(item["query_param"]) if item.get("query_param") else None,
            instruction=str(item["instruction"]) if item.get("instruction") else None,
            api_key_env=str(item["api_key_env"]) if item.get("api_key_env") else None,
            api_key_header=str(item.get("api_key_header", "Authorization")),
            api_key_prefix=str(item.get("api_key_prefix", "Bearer ")),
            extra_headers={
                str(key): str(value)
                for key, value in item.get("extra_headers", {}).items()
            },
        )
    return endpoints


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)

    app_raw = raw.get("app", {})
    persona_raw = raw.get("persona", {})
    live_context_raw = raw.get("live_context", {})
    shell_sandbox_raw = raw.get("shell_sandbox", {})
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
    if live_context_raw and not isinstance(live_context_raw, dict):
        raise ConfigError("Invalid [live_context] section in config.")
    if shell_sandbox_raw and not isinstance(shell_sandbox_raw, dict):
        raise ConfigError("Invalid [shell_sandbox] section in config.")
    live_context = LiveContextSettings(
        custom_api=_parse_live_context_endpoints(
            live_context_raw.get("custom_api") if isinstance(live_context_raw, dict) else None,
            "custom_api",
        ),
        status_api=_parse_live_context_endpoints(
            live_context_raw.get("status_api") if isinstance(live_context_raw, dict) else None,
            "status_api",
        ),
    )
    shell_sandbox = ShellSandboxSettings(
        enabled=bool(shell_sandbox_raw.get("enabled", False))
        if isinstance(shell_sandbox_raw, dict)
        else False,
        timeout_seconds=int(shell_sandbox_raw.get("timeout_seconds", 10))
        if isinstance(shell_sandbox_raw, dict)
        else 10,
        max_output_chars=int(shell_sandbox_raw.get("max_output_chars", 8000))
        if isinstance(shell_sandbox_raw, dict)
        else 8000,
        allowed_roots=_to_tuple(shell_sandbox_raw.get("allowed_roots"))
        if isinstance(shell_sandbox_raw, dict)
        else (".",),
        allowed_command_prefixes=_parse_shell_command_prefixes(
            shell_sandbox_raw.get("allowed_command_prefixes")
            if isinstance(shell_sandbox_raw, dict)
            else None
        ),
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

    if not models:
        raise ConfigError(
            "No models are configured. Add at least one [models.<alias>] section "
            "and set scheduler.chat_model to that alias."
        )

    config = AppConfig(
        app=app,
        persona=persona,
        scheduler=scheduler,
        live_context=live_context,
        shell_sandbox=shell_sandbox,
        providers=providers,
        models=models,
        source_path=config_path,
    )

    if scheduler.chat_model not in models:
        raise _scheduler_model_error("scheduler.chat_model", scheduler.chat_model, models)
    if scheduler.reasoning_model and scheduler.reasoning_model not in models:
        raise _scheduler_model_error(
            "scheduler.reasoning_model", scheduler.reasoning_model, models
        )
    if scheduler.search_model and scheduler.search_model not in models:
        raise _scheduler_model_error("scheduler.search_model", scheduler.search_model, models)

    for model in models.values():
        if model.provider not in providers:
            raise ConfigError(
                f"Model alias '{model.name}' refers to unknown provider '{model.provider}'. "
                f"available providers: {_available_items_text(providers)}. "
                f"Update [models.\"{model.name}\"] provider or add [providers.{model.provider}]."
            )

    return config
