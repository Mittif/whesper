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
class SearchApiSettings:
    provider: str = "serpapi"
    api_key: str | None = None
    api_key_env: str | None = "WHESPER_SERPAPI_API_KEY"
    engine: str = "google"
    timeout_seconds: int = 10

    def resolved_api_key(self) -> str | None:
        if self.api_key is not None:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None


@dataclass(slots=True)
class LiveContextSettings:
    search_api: SearchApiSettings = field(default_factory=SearchApiSettings)
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
class LedHardwareSettings:
    enabled: bool = False
    base_url: str = "http://led.local"
    timeout_seconds: int = 5


@dataclass(slots=True)
class CupHardwareSettings:
    enabled: bool = False
    base_url: str = "http://localhost:3001"
    api_token: str | None = None
    api_token_env: str | None = "WHESPER_CUP_API_TOKEN"
    timeout_seconds: int = 5

    def resolved_api_token(self) -> str | None:
        if self.api_token is not None:
            token = self.api_token.strip()
            return token or None
        if self.api_token_env:
            token = os.getenv(self.api_token_env)
            if token is not None:
                normalized = token.strip()
                return normalized or None
        return None


@dataclass(slots=True)
class HardwareSettings:
    led: LedHardwareSettings = field(default_factory=LedHardwareSettings)
    cup: CupHardwareSettings = field(default_factory=CupHardwareSettings)


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
    hardware: HardwareSettings = field(default_factory=HardwareSettings)

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


def _parse_search_api_settings(raw: Any) -> SearchApiSettings:
    if raw is None:
        return SearchApiSettings()
    if not isinstance(raw, dict):
        raise ConfigError("Invalid [live_context.search_api] section in config.")

    provider = str(raw.get("provider", "serpapi")).strip()
    if provider != "serpapi":
        raise ConfigError("live_context.search_api.provider currently only supports 'serpapi'.")

    engine = str(raw.get("engine", "google")).strip()
    if not engine:
        raise ConfigError("live_context.search_api.engine must be a non-empty string.")

    api_key = raw.get("api_key")
    api_key_env = raw.get("api_key_env", "WHESPER_SERPAPI_API_KEY")
    return SearchApiSettings(
        provider=provider,
        api_key=str(api_key).strip() if api_key is not None else None,
        api_key_env=str(api_key_env).strip() if api_key_env is not None else None,
        engine=engine,
        timeout_seconds=int(raw.get("timeout_seconds", 10)),
    )


def _parse_led_hardware_settings(raw: Any) -> LedHardwareSettings:
    if raw is None:
        return LedHardwareSettings()
    if not isinstance(raw, dict):
        raise ConfigError("Invalid [hardware.led] section in config.")

    base_url = str(raw.get("base_url", "http://led.local")).strip().rstrip("/")
    if not base_url:
        raise ConfigError("hardware.led.base_url must be a non-empty string.")

    return LedHardwareSettings(
        enabled=bool(raw.get("enabled", False)),
        base_url=base_url,
        timeout_seconds=int(raw.get("timeout_seconds", 5)),
    )


def _parse_cup_hardware_settings(raw: Any) -> CupHardwareSettings:
    if raw is None:
        return CupHardwareSettings()
    if not isinstance(raw, dict):
        raise ConfigError("Invalid [hardware.cup] section in config.")

    base_url = str(raw.get("base_url", "http://localhost:3001")).strip().rstrip("/")
    if not base_url:
        raise ConfigError("hardware.cup.base_url must be a non-empty string.")

    api_token = raw.get("api_token")
    api_token_env = raw.get("api_token_env", "WHESPER_CUP_API_TOKEN")
    normalized_api_token = str(api_token).strip() if api_token is not None else None
    normalized_api_token_env = (
        str(api_token_env).strip() if api_token_env is not None else None
    )

    return CupHardwareSettings(
        enabled=bool(raw.get("enabled", False)),
        base_url=base_url,
        api_token=normalized_api_token or None,
        api_token_env=normalized_api_token_env or None,
        timeout_seconds=int(raw.get("timeout_seconds", 5)),
    )


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)

    app_raw = raw.get("app", {})
    persona_raw = raw.get("persona", {})
    live_context_raw = raw.get("live_context", {})
    shell_sandbox_raw = raw.get("shell_sandbox", {})
    hardware_raw = raw.get("hardware", {})
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
    if hardware_raw and not isinstance(hardware_raw, dict):
        raise ConfigError("Invalid [hardware] section in config.")
    live_context = LiveContextSettings(
        search_api=_parse_search_api_settings(
            live_context_raw.get("search_api") if isinstance(live_context_raw, dict) else None
        ),
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
    hardware = HardwareSettings(
        led=_parse_led_hardware_settings(
            hardware_raw.get("led") if isinstance(hardware_raw, dict) else None
        ),
        cup=_parse_cup_hardware_settings(
            hardware_raw.get("cup") if isinstance(hardware_raw, dict) else None
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
        hardware=hardware,
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
