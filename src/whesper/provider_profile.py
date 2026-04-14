from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from whesper.config import ModelConfig, ProviderConfig


_MISSING = object()


@dataclass(slots=True, frozen=True)
class ProviderProfile:
    profile_id: str

    thinking_enable_field: str = "think"
    thinking_disable_field: str | None = "thinking"
    thinking_disable_value: object = None
    thinking_accepts_string_value: bool = True

    tool_arguments_mode: Literal["string", "object"] = "string"
    assistant_tool_content_null: bool = True
    include_tool_name_in_tool_message: bool = False

    reasoning_content_required_when_thinking: bool = False
    reasoning_content_empty_placeholder: str = " "
    extract_reasoning_from_response: bool = True

    text_tool_call_patterns: tuple[str, ...] = ("function_calls",)

    temperature_override: Callable[[ModelConfig, bool], float] | None = None

    builtin_tools: tuple[str, ...] = ()

    def effective_temperature(
        self,
        model: ModelConfig,
        *,
        thinking_enabled: bool,
    ) -> float:
        if self.temperature_override is not None:
            return self.temperature_override(model, thinking_enabled)
        return model.temperature

    def apply_thinking(
        self,
        payload: dict[str, object],
        *,
        model_think: bool | str | None,
        disable_thinking: bool,
    ) -> None:
        if disable_thinking:
            if self.thinking_disable_field is None:
                return
            payload[self.thinking_disable_field] = self.thinking_disable_value
            return
        if model_think is None:
            return
        value = self._coerce_enable_value(model_think)
        if value is _MISSING:
            return
        payload[self.thinking_enable_field] = value

    def _coerce_enable_value(self, raw: bool | str) -> object:
        if isinstance(raw, bool):
            return raw
        if not isinstance(raw, str):
            return _MISSING
        if self.thinking_accepts_string_value:
            return raw
        normalized = raw.strip().casefold()
        if normalized in {"1", "true", "on", "enable", "enabled", "thinking"}:
            return True
        if normalized in {"0", "false", "off", "disable", "disabled", "no_think"}:
            return False
        return _MISSING


def _kimi_k25_temperature(_: ModelConfig, thinking_enabled: bool) -> float:
    return 1.0 if thinking_enabled else 0.6


OPENAI_DEFAULT_PROFILE = ProviderProfile(
    profile_id="openai:default",
    thinking_disable_value={"type": "disabled"},
)

KIMI_K25_PROFILE = ProviderProfile(
    profile_id="kimi:k2-5",
    reasoning_content_required_when_thinking=True,
    reasoning_content_empty_placeholder=" ",
    tool_arguments_mode="string",
    temperature_override=_kimi_k25_temperature,
    builtin_tools=("$web_search",),
    thinking_disable_value={"type": "disabled"},
)

KIMI_DEFAULT_PROFILE = ProviderProfile(
    profile_id="kimi:default",
    reasoning_content_required_when_thinking=True,
    reasoning_content_empty_placeholder=" ",
    tool_arguments_mode="string",
    thinking_disable_value={"type": "disabled"},
)

SILICONFLOW_QWEN_PROFILE = ProviderProfile(
    profile_id="siliconflow:qwen",
    thinking_enable_field="enable_thinking",
    thinking_disable_field="enable_thinking",
    thinking_disable_value=False,
    thinking_accepts_string_value=False,
    tool_arguments_mode="object",
    text_tool_call_patterns=("function_calls", "tool_arg"),
)

SILICONFLOW_DEEPSEEK_PROFILE = ProviderProfile(
    profile_id="siliconflow:deepseek",
    thinking_enable_field="enable_thinking",
    thinking_disable_field="enable_thinking",
    thinking_disable_value=False,
    thinking_accepts_string_value=False,
    text_tool_call_patterns=("function_calls", "tool_arg"),
)

SILICONFLOW_DEFAULT_PROFILE = ProviderProfile(
    profile_id="siliconflow:default",
    thinking_enable_field="enable_thinking",
    thinking_disable_field="enable_thinking",
    thinking_disable_value=False,
    thinking_accepts_string_value=False,
)

OLLAMA_QWEN_PROFILE = ProviderProfile(
    profile_id="ollama:qwen",
    thinking_disable_field=None,
    tool_arguments_mode="object",
    text_tool_call_patterns=("function_calls", "tool_arg"),
)

OLLAMA_DEFAULT_PROFILE = ProviderProfile(
    profile_id="ollama:default",
    thinking_disable_field=None,
    tool_arguments_mode="object",
)

# ---------------------------------------------------------------------------
# Grok (xAI)  —  OpenAI-compatible; no explicit thinking-disable field needed
# ---------------------------------------------------------------------------
GROK_DEFAULT_PROFILE = ProviderProfile(
    profile_id="grok:default",
    thinking_disable_field=None,
)

# ---------------------------------------------------------------------------
# Gemini (Google)  —  `content` must be "" not null; no thinking-disable field
# ---------------------------------------------------------------------------
GEMINI_DEFAULT_PROFILE = ProviderProfile(
    profile_id="gemini:default",
    assistant_tool_content_null=False,
    thinking_disable_field=None,
)

# ---------------------------------------------------------------------------
# GLM / Zhipu AI  —  GLM-4 rejects null content; no thinking-disable field
# ---------------------------------------------------------------------------
GLM_DEFAULT_PROFILE = ProviderProfile(
    profile_id="glm:default",
    assistant_tool_content_null=False,
    thinking_disable_field=None,
)

# ---------------------------------------------------------------------------
# DeepSeek (native api.deepseek.com)
# R1 / reasoner models surface reasoning_content but don't require it in input
# ---------------------------------------------------------------------------
DEEPSEEK_DEFAULT_PROFILE = ProviderProfile(
    profile_id="deepseek:default",
    thinking_disable_field=None,
    text_tool_call_patterns=("function_calls", "tool_arg"),
)

DEEPSEEK_R1_PROFILE = ProviderProfile(
    profile_id="deepseek:r1",
    thinking_disable_field=None,
    text_tool_call_patterns=("function_calls", "tool_arg"),
)

# ---------------------------------------------------------------------------
# Claude (Anthropic, accessed via an OpenAI-compatible proxy such as LiteLLM)
# ---------------------------------------------------------------------------
CLAUDE_DEFAULT_PROFILE = ProviderProfile(
    profile_id="claude:default",
    thinking_disable_field=None,
)

# ---------------------------------------------------------------------------
# Llama (Meta, served locally or via inference providers)
# Safer to send "" rather than null for content; no thinking-disable field
# ---------------------------------------------------------------------------
LLAMA_DEFAULT_PROFILE = ProviderProfile(
    profile_id="llama:default",
    assistant_tool_content_null=False,
    thinking_disable_field=None,
)

# ---------------------------------------------------------------------------
# Qwen (Alibaba / DashScope native)  —  same thinking toggle as SiliconFlow Qwen
# ---------------------------------------------------------------------------
QWEN_DEFAULT_PROFILE = ProviderProfile(
    profile_id="qwen:default",
    thinking_enable_field="enable_thinking",
    thinking_disable_field="enable_thinking",
    thinking_disable_value=False,
    thinking_accepts_string_value=False,
    text_tool_call_patterns=("function_calls", "tool_arg"),
)


PROFILES: dict[tuple[str, str], ProviderProfile] = {
    ("openai", "default"): OPENAI_DEFAULT_PROFILE,
    ("kimi", "k2-5"): KIMI_K25_PROFILE,
    ("kimi", "default"): KIMI_DEFAULT_PROFILE,
    ("siliconflow", "qwen"): SILICONFLOW_QWEN_PROFILE,
    ("siliconflow", "deepseek"): SILICONFLOW_DEEPSEEK_PROFILE,
    ("siliconflow", "default"): SILICONFLOW_DEFAULT_PROFILE,
    ("ollama", "qwen"): OLLAMA_QWEN_PROFILE,
    ("ollama", "default"): OLLAMA_DEFAULT_PROFILE,
    ("grok", "default"): GROK_DEFAULT_PROFILE,
    ("gemini", "default"): GEMINI_DEFAULT_PROFILE,
    ("glm", "default"): GLM_DEFAULT_PROFILE,
    ("deepseek", "default"): DEEPSEEK_DEFAULT_PROFILE,
    ("deepseek", "r1"): DEEPSEEK_R1_PROFILE,
    ("claude", "default"): CLAUDE_DEFAULT_PROFILE,
    ("llama", "default"): LLAMA_DEFAULT_PROFILE,
    ("qwen", "default"): QWEN_DEFAULT_PROFILE,
}


PROFILES_BY_ID: dict[str, ProviderProfile] = {
    profile.profile_id: profile for profile in PROFILES.values()
}


def _detect_kimi_family(model: ModelConfig) -> str:
    name = f"{model.model} {model.name}".casefold()
    if "k2.5" in name or "k2-5" in name:
        return "k2-5"
    return "default"


def _detect_siliconflow_family(model: ModelConfig) -> str:
    name = model.model.casefold()
    if name.startswith("qwen/"):
        return "qwen"
    if name.startswith("deepseek-ai/") or name.startswith("deepseek/"):
        return "deepseek"
    return "default"


def _detect_ollama_family(model: ModelConfig) -> str:
    name = model.model.casefold()
    if "qwen" in name:
        return "qwen"
    if name.startswith("deepseek"):
        return "deepseek"
    return "default"


def _detect_openai_family(_: ModelConfig) -> str:
    return "default"


def _detect_deepseek_family(model: ModelConfig) -> str:
    name = model.model.casefold()
    if "r1" in name or "reasoner" in name:
        return "r1"
    return "default"


def _detect_single_family(_: ModelConfig) -> str:
    """Fallback for providers that have only one profile family."""
    return "default"


FAMILY_DETECTORS: dict[str, Callable[[ModelConfig], str]] = {
    "openai": _detect_openai_family,
    "kimi": _detect_kimi_family,
    "siliconflow": _detect_siliconflow_family,
    "ollama": _detect_ollama_family,
    "grok": _detect_single_family,
    "gemini": _detect_single_family,
    "glm": _detect_single_family,
    "deepseek": _detect_deepseek_family,
    "claude": _detect_single_family,
    "llama": _detect_single_family,
    "qwen": _detect_single_family,
}


def _detect_provider_key(provider: ProviderConfig) -> str:
    override = getattr(provider, "profile_override", None)
    if isinstance(override, str) and override.strip():
        return override.strip()
    if provider.kind == "ollama_native":
        return "ollama"
    signature = f"{provider.name} {provider.base_url}".casefold()
    # Check more-specific signatures first to avoid false positives.
    if "moonshot" in signature or "kimi" in signature:
        return "kimi"
    if "siliconflow" in signature:
        return "siliconflow"
    if "ollama" in signature or ":11434" in signature:
        return "ollama"
    if "x.ai" in signature or "grok" in signature:
        return "grok"
    if "googleapis.com" in signature or "gemini" in signature:
        return "gemini"
    if "bigmodel.cn" in signature or "zhipu" in signature or "chatglm" in signature or "glm" in signature:
        return "glm"
    if "deepseek.com" in signature or "deepseek" in signature:
        return "deepseek"
    if "anthropic.com" in signature or "claude" in signature:
        return "claude"
    if "dashscope" in signature or "aliyun" in signature or "qwen" in signature:
        return "qwen"
    if "meta.ai" in signature or "llama" in signature:
        return "llama"
    return "openai"


def resolve_profile(provider: ProviderConfig, model: ModelConfig) -> ProviderProfile:
    model_override = getattr(model, "profile_override", None)
    if isinstance(model_override, str) and model_override.strip():
        explicit = PROFILES_BY_ID.get(model_override.strip())
        if explicit is not None:
            return explicit

    provider_key = _detect_provider_key(provider)
    detector = FAMILY_DETECTORS.get(provider_key, _detect_openai_family)
    family = detector(model)

    profile = PROFILES.get((provider_key, family))
    if profile is not None:
        return profile
    fallback = PROFILES.get((provider_key, "default"))
    if fallback is not None:
        return fallback
    return OPENAI_DEFAULT_PROFILE


def profile_by_id(profile_id: str | None) -> ProviderProfile | None:
    if not isinstance(profile_id, str) or not profile_id:
        return None
    return PROFILES_BY_ID.get(profile_id)
