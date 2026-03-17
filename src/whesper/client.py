from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
from typing import Iterable
from urllib import error, request

from whesper.config import ModelConfig, ProviderConfig


class ProviderError(RuntimeError):
    pass


@dataclass(slots=True)
class CompletionResult:
    content: str
    raw_response: dict


def _build_openai_payload(
    model: ModelConfig,
    messages: list[dict[str, str]],
    *,
    stream: bool,
) -> bytes:
    payload: dict[str, object] = {
        "model": model.model,
        "messages": messages,
        "temperature": model.temperature,
        "stream": stream,
    }
    if model.max_tokens is not None:
        payload["max_tokens"] = model.max_tokens
    if model.top_p is not None:
        payload["top_p"] = model.top_p
    if model.think is not None:
        payload["think"] = model.think
    return json.dumps(payload).encode("utf-8")


def _build_ollama_payload(
    model: ModelConfig,
    messages: list[dict[str, str]],
    *,
    stream: bool,
) -> bytes:
    payload: dict[str, object] = {
        "model": model.model,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": model.temperature,
        },
    }
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
    messages: list[dict[str, str]],
    *,
    stream: bool,
) -> bytes:
    if provider.kind == "ollama_native":
        return _build_ollama_payload(model, messages, stream=stream)
    return _build_openai_payload(model, messages, stream=stream)


def _completion_endpoint(provider: ProviderConfig) -> str:
    if provider.kind == "ollama_native":
        return f"{provider.base_url.rstrip('/')}/api/chat"
    return f"{provider.base_url}/chat/completions"


def _normalize_content(value: object) -> str:
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


class OpenAICompatibleClient:
    def create_chat_completion(
        self,
        provider: ProviderConfig,
        model: ModelConfig,
        messages: list[dict[str, str]],
    ) -> CompletionResult:
        req = request.Request(
            _completion_endpoint(provider),
            data=_request_payload(provider, model, messages, stream=False),
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

        return CompletionResult(content=content.strip(), raw_response=raw)

    def create_chat_completion_stream(
        self,
        provider: ProviderConfig,
        model: ModelConfig,
        messages: list[dict[str, str]],
    ) -> Iterable[str]:
        req = request.Request(
            _completion_endpoint(provider),
            data=_request_payload(provider, model, messages, stream=True),
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
