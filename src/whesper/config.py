from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import tomllib
from typing import Any


class ConfigError(ValueError):
    pass


CUP_DEFAULT_BASE_URL = "http://esp32-cup.local"
_CONFIG_VARIABLE_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)"
)
@dataclass(slots=True)
class ProviderConfig:
    name: str
    kind: str
    base_url: str
    api_key: str | None = None
    api_key_env: str | None = None
    timeout_seconds: int = 120
    extra_headers: dict[str, str] = field(default_factory=dict)
    profile_override: str | None = None

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
    profile_override: str | None = None


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
    context_strategy: str = "memory_first"
    recent_history_limit: int = 4
    tool_exposure_strategy: str = "matched_only"


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
    provider: str = "duckduckgo"
    api_key: str | None = None
    api_key_env: str | None = None
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
class CupHardwareSettings:
    enabled: bool = False
    base_url: str = CUP_DEFAULT_BASE_URL
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
    cup: CupHardwareSettings = field(default_factory=CupHardwareSettings)


@dataclass(slots=True)
class ServerSettings:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    api_token: str | None = None
    api_token_env: str | None = "WHESPER_SERVER_TOKEN"
    require_auth: bool = True
    cors_origins: tuple[str, ...] = ()
    request_timeout_seconds: int = 300
    max_concurrent_turns: int = 4

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
    server: ServerSettings = field(default_factory=ServerSettings)

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

    def register_model(self, alias: str, provider_name: str, model_id: str) -> ModelConfig:
        """Create a model config at runtime and add it to the config."""
        if provider_name not in self.providers:
            raise ConfigError(
                f"Unknown provider: {provider_name}. "
                f"available providers: {', '.join(sorted(self.providers.keys()))}"
            )
        model = ModelConfig(
            name=alias,
            provider=provider_name,
            model=model_id,
            temperature=0.7,
            max_tokens=1600,
        )
        self.models[alias] = model
        return model

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


def _parse_extra_headers(raw: Any, *, field_name: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{field_name} must be a table of header values.")
    return {
        str(key): str(value)
        for key, value in raw.items()
    }


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


def _parse_context_strategy(raw: Any) -> str:
    if raw is None:
        return "memory_first"
    value = str(raw).strip().lower()
    if value not in {"memory_first", "full_transcript"}:
        raise ConfigError(
            "app.context_strategy must be either 'memory_first' or 'full_transcript'."
        )
    return value


def _parse_tool_exposure_strategy(raw: Any) -> str:
    if raw is None:
        return "matched_only"
    value = str(raw).strip().lower()
    if value not in {"matched_only", "model_first"}:
        raise ConfigError(
            "app.tool_exposure_strategy must be either 'matched_only' or 'model_first'."
        )
    return value


def _parse_config_variables(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("Invalid [variables] section in config.")

    variables: dict[str, str] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigError("variables keys must be non-empty strings.")
        if not isinstance(value, str):
            raise ConfigError(f"variables.{name} must be a string.")
        variables[name.strip()] = value.strip()
    return variables


def _resolve_config_variables(
    value: str,
    *,
    field_name: str,
    variables: dict[str, str],
) -> str:
    def replace(match: re.Match[str]) -> str:
        variable_name = match.group(1) or match.group(3)
        default_value = match.group(2)
        assert variable_name is not None
        if variable_name in variables:
            return variables[variable_name]
        environment_value = os.getenv(variable_name)
        if environment_value is not None:
            return environment_value

        if default_value is not None:
            return default_value
        raise ConfigError(
            f"{field_name} references unknown variable '{variable_name}'. "
            "Define it under [variables] or export it in the environment."
        )

    return _CONFIG_VARIABLE_PATTERN.sub(replace, value)


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
            extra_headers=_parse_extra_headers(
                item.get("extra_headers"),
                field_name=f"live_context.{section_name}.{name}.extra_headers",
            ),
        )
    return endpoints


VALID_SEARCH_PROVIDERS = ("duckduckgo", "brave", "serpapi")

_SEARCH_API_KEY_ENV_DEFAULTS: dict[str, str] = {
    "serpapi": "WHESPER_SERPAPI_API_KEY",
    "brave": "WHESPER_BRAVE_API_KEY",
}


def _parse_search_api_settings(raw: Any) -> SearchApiSettings:
    if raw is None:
        return SearchApiSettings()
    if not isinstance(raw, dict):
        raise ConfigError("Invalid [live_context.search_api] section in config.")

    provider = str(raw.get("provider", "duckduckgo")).strip()
    if provider not in VALID_SEARCH_PROVIDERS:
        raise ConfigError(
            f"live_context.search_api.provider must be one of: "
            f"{', '.join(VALID_SEARCH_PROVIDERS)}."
        )

    engine = str(raw.get("engine", "google")).strip()
    if provider == "serpapi" and not engine:
        raise ConfigError("live_context.search_api.engine must be a non-empty string.")

    api_key = raw.get("api_key")
    api_key_env = raw.get("api_key_env", _SEARCH_API_KEY_ENV_DEFAULTS.get(provider))
    return SearchApiSettings(
        provider=provider,
        api_key=str(api_key).strip() if api_key is not None else None,
        api_key_env=str(api_key_env).strip() if api_key_env is not None else None,
        engine=engine,
        timeout_seconds=int(raw.get("timeout_seconds", 10)),
    )


def _parse_cup_hardware_settings(
    raw: Any,
    *,
    variables: dict[str, str],
) -> CupHardwareSettings:
    if raw is None:
        return CupHardwareSettings()
    if not isinstance(raw, dict):
        raise ConfigError("Invalid [hardware.cup] section in config.")

    base_url = _resolve_config_variables(
        str(raw.get("base_url", CUP_DEFAULT_BASE_URL)),
        field_name="hardware.cup.base_url",
        variables=variables,
    ).strip().rstrip("/")
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


def _parse_server_settings(raw: Any) -> ServerSettings:
    if raw is None:
        return ServerSettings()
    if not isinstance(raw, dict):
        raise ConfigError("Invalid [server] section in config.")

    host = str(raw.get("host", "127.0.0.1")).strip()
    if not host:
        raise ConfigError("server.host must be a non-empty string.")

    port_raw = raw.get("port", 8765)
    try:
        port = int(port_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("server.port must be an integer.") from exc
    if not (1 <= port <= 65535):
        raise ConfigError("server.port must be between 1 and 65535.")

    api_token = raw.get("api_token")
    api_token_env = raw.get("api_token_env", "WHESPER_SERVER_TOKEN")
    normalized_api_token = (
        str(api_token).strip() if api_token is not None else None
    ) or None
    normalized_api_token_env = (
        str(api_token_env).strip() if api_token_env is not None else None
    ) or None

    cors_origins_raw = raw.get("cors_origins")
    if cors_origins_raw is None:
        cors_origins: tuple[str, ...] = ()
    elif isinstance(cors_origins_raw, list) and all(
        isinstance(item, str) for item in cors_origins_raw
    ):
        cors_origins = tuple(item.strip() for item in cors_origins_raw if item.strip())
    else:
        raise ConfigError("server.cors_origins must be a list of strings.")

    request_timeout_seconds = int(raw.get("request_timeout_seconds", 300))
    if request_timeout_seconds <= 0:
        raise ConfigError("server.request_timeout_seconds must be > 0.")

    max_concurrent_turns = int(raw.get("max_concurrent_turns", 4))
    if max_concurrent_turns <= 0:
        raise ConfigError("server.max_concurrent_turns must be > 0.")

    return ServerSettings(
        enabled=bool(raw.get("enabled", False)),
        host=host,
        port=port,
        api_token=normalized_api_token,
        api_token_env=normalized_api_token_env,
        require_auth=bool(raw.get("require_auth", True)),
        cors_origins=cors_origins,
        request_timeout_seconds=request_timeout_seconds,
        max_concurrent_turns=max_concurrent_turns,
    )


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)

    variables = _parse_config_variables(raw.get("variables"))
    app_raw = raw.get("app", {})
    persona_raw = raw.get("persona", {})
    live_context_raw = raw.get("live_context", {})
    shell_sandbox_raw = raw.get("shell_sandbox", {})
    hardware_raw = raw.get("hardware", {})
    server_raw = raw.get("server", {})
    scheduler_raw = _require_table(raw, "scheduler")
    providers_raw = _require_table(raw, "providers")
    models_raw = _require_table(raw, "models")

    app = AppSettings(
        storage_dir=str(app_raw.get("storage_dir", ".whesper")),
        default_session=str(app_raw.get("default_session", "main")),
        history_limit=int(app_raw.get("history_limit", 16)),
        context_strategy=_parse_context_strategy(app_raw.get("context_strategy")),
        recent_history_limit=int(app_raw.get("recent_history_limit", 4)),
        tool_exposure_strategy=_parse_tool_exposure_strategy(
            app_raw.get("tool_exposure_strategy")
        ),
    )
    if app.history_limit < 0:
        raise ConfigError("app.history_limit must be >= 0.")
    if app.recent_history_limit < 0:
        raise ConfigError("app.recent_history_limit must be >= 0.")
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
    if server_raw and not isinstance(server_raw, dict):
        raise ConfigError("Invalid [server] section in config.")
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
        cup=_parse_cup_hardware_settings(
            hardware_raw.get("cup") if isinstance(hardware_raw, dict) else None,
            variables=variables,
        ),
    )
    server = _parse_server_settings(server_raw if isinstance(server_raw, dict) else None)

    providers: dict[str, ProviderConfig] = {}
    for name, item in providers_raw.items():
        if not isinstance(item, dict):
            raise ConfigError(f"Provider config for {name} must be a table.")
        providers[name] = ProviderConfig(
            name=name,
            kind=str(item.get("kind", "openai_compatible")),
            base_url=_resolve_config_variables(
                str(item["base_url"]),
                field_name=f"providers.{name}.base_url",
                variables=variables,
            ).rstrip("/"),
            api_key=str(item["api_key"]) if item.get("api_key") is not None else None,
            api_key_env=(
                str(item["api_key_env"]) if item.get("api_key_env") is not None else None
            ),
            timeout_seconds=int(item.get("timeout_seconds", 120)),
            extra_headers=_parse_extra_headers(
                item.get("extra_headers"),
                field_name=f"providers.{name}.extra_headers",
            ),
            profile_override=(
                str(item["profile"]).strip()
                if isinstance(item.get("profile"), str) and item["profile"].strip()
                else None
            ),
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
            profile_override=(
                str(item["profile"]).strip()
                if isinstance(item.get("profile"), str) and item["profile"].strip()
                else None
            ),
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
        server=server,
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
