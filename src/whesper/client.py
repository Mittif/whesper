from __future__ import annotations

import http.client
import json
from typing import Iterable
from urllib import error, request

from whesper.agent_types import AgentCompletion, ToolInvocation
from whesper.config import ModelConfig, ProviderConfig
from whesper.provider_profile import ProviderProfile, resolve_profile
from whesper.tool_protocol import DefaultToolProtocolAdapter


class ProviderError(RuntimeError):
    pass


ToolCall = ToolInvocation
CompletionResult = AgentCompletion
DEFAULT_TOOL_PROTOCOL_ADAPTER = DefaultToolProtocolAdapter()


def _effective_temperature(
    provider: ProviderConfig,
    model: ModelConfig,
    *,
    disable_thinking: bool,
) -> float:
    profile = resolve_profile(provider, model)
    return profile.effective_temperature(model, thinking_enabled=not disable_thinking)


def _build_openai_payload(
    provider: ProviderConfig,
    model: ModelConfig,
    messages: list[dict[str, object]],
    *,
    stream: bool,
    tools: list[dict[str, object]] | None = None,
    tool_choice: str | dict[str, object] | None = None,
    disable_thinking: bool = False,
) -> bytes:
    profile: ProviderProfile = resolve_profile(provider, model)
    payload: dict[str, object] = {
        "model": model.model,
        "messages": messages,
        "temperature": profile.effective_temperature(
            model,
            thinking_enabled=not disable_thinking,
        ),
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if model.max_tokens is not None:
        payload["max_tokens"] = model.max_tokens
    if model.top_p is not None:
        payload["top_p"] = model.top_p
    profile.apply_thinking(
        payload,
        model_think=model.think,
        disable_thinking=disable_thinking,
    )
    return json.dumps(payload).encode("utf-8")


def _build_ollama_payload(
    model: ModelConfig,
    messages: list[dict[str, object]],
    *,
    stream: bool,
    tools: list[dict[str, object]] | None = None,
    tool_choice: str | dict[str, object] | None = None,
) -> bytes:
    payload: dict[str, object] = {
        "model": model.model,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": model.temperature,
        },
    }
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if model.top_p is not None:
        payload["options"]["top_p"] = model.top_p
    if model.max_tokens is not None:
        payload["options"]["num_predict"] = model.max_tokens
    if model.think is not None:
        payload["think"] = model.think
    return json.dumps(payload).encode("utf-8")


def _request_headers(provider: ProviderConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    headers.update(provider.extra_headers)

    api_key = provider.resolved_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _request_payload(
    provider: ProviderConfig,
    model: ModelConfig,
    messages: list[dict[str, object]],
    *,
    stream: bool,
    tools: list[dict[str, object]] | None = None,
    tool_choice: str | dict[str, object] | None = None,
    disable_thinking: bool = False,
) -> bytes:
    if provider.kind == "ollama_native":
        return _build_ollama_payload(
            model,
            messages,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
        )
    return _build_openai_payload(
        provider,
        model,
        messages,
        stream=stream,
        tools=tools,
        tool_choice=tool_choice,
        disable_thinking=disable_thinking,
    )


def _completion_endpoint(provider: ProviderConfig) -> str:
    if provider.kind == "ollama_native":
        return f"{provider.base_url.rstrip('/')}/api/chat"
    return f"{provider.base_url}/chat/completions"


def _normalize_content(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(value)


def _extract_stream_text(raw: dict) -> str:
    try:
        choice = raw["choices"][0]
    except (KeyError, IndexError, TypeError):
        return ""

    if not isinstance(choice, dict):
        return ""

    delta = choice.get("delta")
    if isinstance(delta, dict) and delta.get("content") is not None:
        return _normalize_content(delta["content"])

    message = choice.get("message")
    if isinstance(message, dict) and message.get("content") is not None:
        return _normalize_content(message["content"])

    if choice.get("text") is not None:
        return _normalize_content(choice["text"])

    return ""


def _extract_tool_calls(provider: ProviderConfig, raw: dict) -> tuple[ToolCall, ...]:
    return DEFAULT_TOOL_PROTOCOL_ADAPTER.extract_tool_calls(provider, raw)


def _extract_tool_calls_from_text(
    content: str,
    *,
    profile: ProviderProfile | None = None,
    allowed_tool_names: Iterable[str] | None = None,
) -> tuple[ToolCall, ...]:
    return DEFAULT_TOOL_PROTOCOL_ADAPTER.extract_tool_calls_from_text(
        content,
        profile=profile,
        allowed_tool_names=allowed_tool_names,
    )


def _tool_names_from_tools(
    tools: list[dict[str, object]] | None,
) -> frozenset[str]:
    if not tools:
        return frozenset()
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return frozenset(names)


def _extract_reasoning_content(provider: ProviderConfig, raw: dict) -> str | None:
    try:
        if provider.kind == "ollama_native":
            message = raw["message"]
            if isinstance(message, dict):
                value = message.get("reasoning_content")
                if isinstance(value, str):
                    return value
                return None
            return None
        choice = raw["choices"][0]
        if not isinstance(choice, dict):
            return None
        message = choice.get("message")
        if isinstance(message, dict):
            value = message.get("reasoning_content")
            if isinstance(value, str):
                return value
    except (KeyError, IndexError, TypeError):
        return None
    return None


def _iter_sse_events(lines: Iterable[bytes]) -> Iterable[str]:
    buffer: list[str] = []
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if buffer:
                yield "\n".join(buffer)
                buffer.clear()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            buffer.append(line[5:].lstrip())

    if buffer:
        yield "\n".join(buffer)


def _iter_json_lines(lines: Iterable[bytes]) -> Iterable[dict]:
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProviderError("Ollama returned invalid JSON in the stream.") from exc


def list_provider_models(provider: ProviderConfig) -> list[str]:
    """Query a provider's API for available model IDs.

    OpenAI-compatible: GET {base_url}/models
    Ollama:            GET {base_url}/api/tags
    """
    if provider.kind == "ollama_native":
        url = f"{provider.base_url.rstrip('/')}/api/tags"
    else:
        url = f"{provider.base_url.rstrip('/')}/models"

    req = request.Request(url, headers=_request_headers(provider), method="GET")
    try:
        with request.urlopen(req, timeout=provider.timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ProviderError(
            f"Provider '{provider.name}' returned HTTP {exc.code}: {detail}"
        ) from exc
    except error.URLError as exc:
        raise ProviderError(
            f"Provider '{provider.name}' is unreachable: {exc.reason}"
        ) from exc

    if provider.kind == "ollama_native":
        models = raw.get("models", [])
        return sorted(m["name"] for m in models if isinstance(m, dict) and "name" in m)

    data = raw.get("data", [])
    return sorted(m["id"] for m in data if isinstance(m, dict) and "id" in m)


class OpenAICompatibleClient:
    def create_chat_completion(
        self,
        provider: ProviderConfig,
        model: ModelConfig,
        messages: list[dict[str, object]],
        *,
        tools: list[dict[str, object]] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        disable_thinking: bool = False,
    ) -> CompletionResult:
        profile = resolve_profile(provider, model)
        req = request.Request(
            _completion_endpoint(provider),
            data=_request_payload(
                provider,
                model,
                messages,
                stream=False,
                tools=tools,
                tool_choice=tool_choice,
                disable_thinking=disable_thinking,
            ),
            headers=_request_headers(provider),
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=provider.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(
                f"Provider '{provider.name}' returned HTTP {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise ProviderError(
                f"Provider '{provider.name}' is unreachable: {exc.reason}"
            ) from exc
        except http.client.RemoteDisconnected as exc:
            raise ProviderError(
                f"Provider '{provider.name}' closed the connection unexpectedly. "
                "Please verify the base_url and whether the endpoint supports "
                "OpenAI-compatible /chat/completions requests."
            ) from exc

        tool_calls = _extract_tool_calls(provider, raw)

        try:
            if provider.kind == "ollama_native":
                content = _normalize_content(raw["message"]["content"])
            else:
                choice = raw["choices"][0]
                message = choice["message"]
                content = _normalize_content(message["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"Provider '{provider.name}' returned an unexpected response shape."
            ) from exc

        if not tool_calls and content and tools:
            allowed = _tool_names_from_tools(tools)
            tool_calls = _extract_tool_calls_from_text(
                content,
                profile=profile,
                allowed_tool_names=allowed,
            )
            if tool_calls:
                content = ""

        return CompletionResult(
            content=content.strip(),
            raw_response=raw,
            tool_calls=tool_calls,
            reasoning_content=_extract_reasoning_content(provider, raw),
        )

    def create_chat_completion_stream(
        self,
        provider: ProviderConfig,
        model: ModelConfig,
        messages: list[dict[str, object]],
        *,
        tools: list[dict[str, object]] | None = None,
        tool_choice: str | dict[str, object] | None = None,
    ) -> Iterable[str]:
        req = request.Request(
            _completion_endpoint(provider),
            data=_request_payload(
                provider,
                model,
                messages,
                stream=True,
                tools=tools,
                tool_choice=tool_choice,
            ),
            headers=_request_headers(provider),
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=provider.timeout_seconds) as response:
                if provider.kind == "ollama_native":
                    for raw in _iter_json_lines(response):
                        message = raw.get("message")
                        if isinstance(message, dict):
                            chunk = message.get("content")
                            if isinstance(chunk, str) and chunk:
                                yield chunk
                    return
                for event in _iter_sse_events(response):
                    if event == "[DONE]":
                        break
                    try:
                        raw = json.loads(event)
                    except json.JSONDecodeError as exc:
                        raise ProviderError(
                            f"Provider '{provider.name}' returned invalid stream data."
                        ) from exc
                    chunk = _extract_stream_text(raw)
                    if chunk:
                        yield chunk
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(
                f"Provider '{provider.name}' returned HTTP {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise ProviderError(
                f"Provider '{provider.name}' is unreachable: {exc.reason}"
            ) from exc
        except http.client.RemoteDisconnected as exc:
            raise ProviderError(
                f"Provider '{provider.name}' closed the connection unexpectedly. "
                "Please verify the base_url and whether the endpoint supports "
                "OpenAI-compatible /chat/completions requests."
            ) from exc
