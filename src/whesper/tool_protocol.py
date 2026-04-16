from __future__ import annotations

import json
import re
from typing import Iterable
from xml.etree import ElementTree

from whesper.agent_types import ToolInvocation
from whesper.provider_profile import ProviderProfile


FUNCTION_CALLS_BLOCK_PATTERN = re.compile(
    r"<function_calls>\s*.*?</function_calls>",
    re.DOTALL,
)

# DeepSeek-style: <tool>name</tool>\n<arg>{"key": "value"}</arg>
_TOOL_ARG_PATTERN = re.compile(
    r"<tool>\s*(\S+?)\s*</tool>\s*<arg>\s*(.*?)\s*</arg>",
    re.DOTALL,
)

_BOX_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*([A-Za-z_][\w-]*)\s*\(\s*query\s*=\s*([\"'])(.*?)\2\s*\)\s*<\|end_of_box\|>",
    re.DOTALL,
)

# MiniMax models emit <minimax:tool_call>…<invoke>…</invoke></minimax:tool_call>.
# Normalise to <function_calls> before parsing so the same extraction path applies.
_MINIMAX_OPEN_TAG = re.compile(r"<minimax:tool_call>", re.IGNORECASE)
_MINIMAX_CLOSE_TAG = re.compile(r"</minimax:tool_call>", re.IGNORECASE)

# ChatGLM ✿FUNCTION✿name✿ARGS✿json✿RESULT✿ / ✿FUNCTION✿{json}✿RESULT✿
_GLM_FUNCTION_PATTERN = re.compile(
    r"✿FUNCTION✿\s*(\{.*?\})\s*(?:✿RESULT✿|✿END✿|$)",
    re.DOTALL,
)

# Qwen <|plugin|> format (older Qwen-Agent models)
_QWEN_PLUGIN_PATTERN = re.compile(
    r"<\|plugin\|>\s*(\{.*?\})\s*(?:<\|endoftext\|>|\n\n|$)",
    re.DOTALL,
)

# Hermes / NousResearch / OpenChat: <tool_call>{json}</tool_call>
# Distinct from box-style <tool_call>name(args)<|end_of_box|> because the body
# here is a JSON object and the closer is </tool_call> rather than <|end_of_box|>.
# Capture the raw body (non-greedy outer) and parse JSON with raw_decode to
# correctly handle nested arguments objects.
_HERMES_OUTER_PATTERN = re.compile(
    r"<tool_call>(.*?)</tool_call>",
    re.DOTALL | re.IGNORECASE,
)

# Llama 3.x: <|python_tag|>{json}(<|eom_id|>|<|eot_id|>|EOF)
_LLAMA_OUTER_PATTERN = re.compile(
    r"<\|python_tag\|>(.*?)(?:<\|eom_id\|>|<\|eot_id\|>|$)",
    re.DOTALL,
)

# Mistral: [TOOL_CALLS][{...}, ...]  —  capture the list body, parse with json.
_MISTRAL_OUTER_PATTERN = re.compile(
    r"\[TOOL_CALLS\](.*?)(?:\[/TOOL_CALLS\]|\Z)",
    re.DOTALL,
)

# ---------------------------------------------------------------------------
# Comprehensive list of all known model-specific tool-call text artifacts.
# Used both for sanitising assistant content before storage and for cleaning
# dirty history before replaying to a different model.
# ---------------------------------------------------------------------------
TOOL_CALL_ARTIFACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # OpenAI / Claude-proxy function_calls XML
    re.compile(r"<function_calls>.*?</function_calls>", re.DOTALL | re.IGNORECASE),
    # MiniMax wrapper
    re.compile(r"<minimax:tool_call>.*?</minimax:tool_call>", re.DOTALL | re.IGNORECASE),
    # DeepSeek <tool>name</tool>\n<arg>json</arg>
    re.compile(r"<tool>\s*\S+?\s*</tool>\s*<arg>\s*.*?\s*</arg>", re.DOTALL),
    # Qwen / Box <tool_call>name(args)<|end_of_box|>
    re.compile(r"<tool_call>.*?<\|end_of_box\|>", re.DOTALL),
    # Hermes / NousResearch / OpenChat <tool_call>{json}</tool_call>
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE),
    # ChatGLM ✿FUNCTION✿…✿RESULT✿
    re.compile(r"✿FUNCTION✿.*?(?:✿RESULT✿|✿END✿|$)", re.DOTALL),
    # Qwen <|plugin|>…<|endoftext|>
    re.compile(r"<\|plugin\|>.*?(?:<\|endoftext\|>|\n\n|$)", re.DOTALL),
    # Llama 3 <|python_tag|>…<|eom_id|>
    re.compile(r"<\|python_tag\|>.*?(?:<\|eom_id\|>|<\|eot_id\|>|$)", re.DOTALL),
    # Mistral [TOOL_CALLS][...]
    re.compile(r"\[TOOL_CALLS\]\s*\[.*?\]", re.DOTALL),
)

# Backward-compatible alias kept for any external references.
TOOL_CALL_XML_BLOCK_PATTERNS = TOOL_CALL_ARTIFACT_PATTERNS

# Fenced (```...```) and inline (`...`) code segments where tool-call-like
# syntax is almost always a quoted example rather than a real invocation.
_CODE_FENCE_PATTERN = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_PATTERN = re.compile(r"`[^`\n]*`")


def _strip_code_segments(content: str) -> str:
    without_fences = _CODE_FENCE_PATTERN.sub(" ", content)
    return _INLINE_CODE_PATTERN.sub(" ", without_fences)


_NAME_FIELDS: tuple[str, ...] = ("name", "tool", "function", "tool_name", "function_name")
_ARGS_FIELDS: tuple[str, ...] = ("arguments", "parameters", "args", "params", "input")


def _tool_call_from_dict(
    payload: dict[str, object],
    *,
    index: int,
) -> ToolInvocation | None:
    """Build a ToolInvocation from a loose dict payload emitted in text.

    Accepts several synonymous field names to tolerate the many JSON tool-call
    dialects used by open-source models (Hermes, Llama, Mistral, etc.).
    Returns None when the payload lacks a usable name AND arguments container.
    """
    name: str | None = None
    for key in _NAME_FIELDS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            name = value.strip()
            break
    if name is None:
        return None

    arguments: object = None
    for key in _ARGS_FIELDS:
        if key in payload:
            arguments = payload[key]
            break
    # Require the payload to look like a tool call — reject pure `{"name": X}`
    # with no arguments field, since that catches plain JSON mentions.
    if arguments is None:
        return None

    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            parsed = {"input": arguments}
        arguments = parsed if isinstance(parsed, dict) else {"input": parsed}
    if not isinstance(arguments, dict):
        arguments = {"input": arguments}

    raw_id = payload.get("id")
    call_id = (
        raw_id.strip()
        if isinstance(raw_id, str) and raw_id.strip()
        else f"text-tool-call-{index}"
    )
    return ToolInvocation(
        tool_call_id=call_id,
        name=name,
        arguments_json=json.dumps(arguments, ensure_ascii=False),
        tool_type="function",
    )


def sanitize_tool_call_artifacts(content: str) -> str:
    """Strip all known model-specific tool-call text artifacts from *content*.

    Different models emit tool calls in vendor-specific plain-text formats
    (XML blocks, special tokens, etc.).  When these are not successfully
    extracted as structured ``tool_calls``, the raw text leaks into the stored
    assistant message and may confuse a different model on subsequent turns.

    This function removes every known pattern while preserving surrounding
    prose.  Code-fenced blocks are protected so that quoted examples of tool
    call syntax are not accidentally stripped.
    """
    # Protect fenced code blocks from stripping.
    placeholders: dict[str, str] = {}

    def _protect(match: re.Match[str]) -> str:
        key = f"\x00CODEBLOCK{len(placeholders)}\x00"
        placeholders[key] = match.group(0)
        return key

    protected = _CODE_FENCE_PATTERN.sub(_protect, content)

    for pattern in TOOL_CALL_ARTIFACT_PATTERNS:
        protected = pattern.sub("", protected)

    # Restore fenced code blocks.
    for key, value in placeholders.items():
        protected = protected.replace(key, value)

    # Collapse excessive blank lines left by removals.
    protected = re.sub(r"\n{3,}", "\n\n", protected)
    return protected.strip()


class DefaultToolProtocolAdapter:
    def tool_call_payload(self, tool_call: ToolInvocation) -> dict[str, object]:
        return {
            "id": tool_call.tool_call_id,
            "type": tool_call.tool_type,
            "function": {
                "name": tool_call.name,
                "arguments": tool_call.arguments_json,
            },
        }

    def extract_tool_calls(self, provider, raw: dict) -> tuple[ToolInvocation, ...]:
        try:
            if provider.kind == "ollama_native":
                message = raw["message"]
                if isinstance(message, dict):
                    return self._tool_calls_from_items(message.get("tool_calls"))
                return ()
            choice = raw["choices"][0]
            if not isinstance(choice, dict):
                return ()
            message = choice.get("message")
            if isinstance(message, dict):
                return self._tool_calls_from_items(message.get("tool_calls"))
        except (KeyError, IndexError, TypeError):
            return ()
        return ()

    def extract_tool_calls_from_text(
        self,
        content: str,
        *,
        profile: ProviderProfile | None = None,
        allowed_tool_names: Iterable[str] | None = None,
    ) -> tuple[ToolInvocation, ...]:
        patterns = (
            profile.text_tool_call_patterns
            if profile is not None
            else ("function_calls", "tool_arg")
        )
        allow = (
            frozenset(allowed_tool_names)
            if allowed_tool_names is not None
            else None
        )
        cleaned = _strip_code_segments(content)
        for pattern_name in patterns:
            tool_calls = self._extract_by_pattern(pattern_name, cleaned)
            if allow is not None:
                tool_calls = tuple(tc for tc in tool_calls if tc.name in allow)
            if tool_calls:
                return tool_calls
        return ()

    def _extract_by_pattern(
        self,
        pattern_name: str,
        content: str,
    ) -> tuple[ToolInvocation, ...]:
        if pattern_name == "function_calls":
            return self._extract_function_calls_block(content)
        if pattern_name == "tool_arg":
            return self._extract_tool_arg_pairs(content)
        if pattern_name == "action_json":
            return self._extract_action_json(content)
        if pattern_name == "box_tool_call":
            return self._extract_box_tool_calls(content)
        if pattern_name == "glm_function":
            return self._extract_glm_function_calls(content)
        if pattern_name == "plugin":
            return self._extract_plugin_calls(content)
        if pattern_name == "hermes_tool_call":
            return self._extract_hermes_tool_calls(content)
        if pattern_name == "llama_python_tag":
            return self._extract_llama_python_tag_calls(content)
        if pattern_name == "mistral_tool_calls":
            return self._extract_mistral_tool_calls(content)
        if pattern_name == "json_tool_call":
            return self._extract_json_tool_calls(content)
        return ()

    def _extract_function_calls_block(self, content: str) -> tuple[ToolInvocation, ...]:
        # Normalise MiniMax-style wrapper tags so the same extraction path applies.
        # <minimax:tool_call>…</minimax:tool_call>  →  <function_calls>…</function_calls>
        content = _MINIMAX_OPEN_TAG.sub("<function_calls>", content)
        content = _MINIMAX_CLOSE_TAG.sub("</function_calls>", content)
        match = FUNCTION_CALLS_BLOCK_PATTERN.search(content)
        if match is None:
            return ()
        try:
            root = ElementTree.fromstring(match.group(0))
        except ElementTree.ParseError:
            return ()

        tool_calls: list[ToolInvocation] = []
        for index, invoke in enumerate(root.findall("invoke")):
            name = invoke.attrib.get("name")
            if not name:
                continue
            arguments: dict[str, object] = {}
            for parameter in invoke.findall("parameter"):
                parameter_name = parameter.attrib.get("name")
                if not parameter_name:
                    continue
                arguments[parameter_name] = "".join(parameter.itertext()).strip()
            tool_calls.append(
                ToolInvocation(
                    tool_call_id=f"text-tool-call-{index}",
                    name=name,
                    arguments_json=json.dumps(arguments, ensure_ascii=False),
                    tool_type="function",
                )
            )
        return tuple(tool_calls)

    def _extract_tool_arg_pairs(self, content: str) -> tuple[ToolInvocation, ...]:
        matches = _TOOL_ARG_PATTERN.findall(content)
        if not matches:
            return ()
        tool_calls: list[ToolInvocation] = []
        for index, (name, args_text) in enumerate(matches):
            try:
                arguments = json.loads(args_text)
            except json.JSONDecodeError:
                arguments = {"input": args_text}
            tool_calls.append(
                ToolInvocation(
                    tool_call_id=f"text-tool-call-{index}",
                    name=name,
                    arguments_json=json.dumps(arguments, ensure_ascii=False),
                    tool_type="function",
                )
            )
        return tuple(tool_calls)

    def _extract_action_json(self, content: str) -> tuple[ToolInvocation, ...]:
        decoder = json.JSONDecoder()
        tool_calls: list[ToolInvocation] = []
        search_actions = {"search", "web_search"}

        for start_index, char in enumerate(content):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(content[start_index:])
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            action = payload.get("action")
            if not isinstance(action, str):
                continue
            normalized_action = action.strip().casefold()
            if normalized_action not in search_actions:
                continue
            query = payload.get("query")
            if not isinstance(query, str) or not query.strip():
                continue
            tool_calls.append(
                ToolInvocation(
                    tool_call_id=f"text-tool-call-{len(tool_calls)}",
                    name="web_search",
                    arguments_json=json.dumps(
                        {"query": query.strip()},
                        ensure_ascii=False,
                    ),
                    tool_type="function",
                )
            )
        return tuple(tool_calls)

    def _extract_box_tool_calls(self, content: str) -> tuple[ToolInvocation, ...]:
        matches = _BOX_TOOL_CALL_PATTERN.findall(content)
        if not matches:
            return ()
        tool_calls: list[ToolInvocation] = []
        for index, (name, _quote, query) in enumerate(matches):
            normalized_name = name.strip()
            normalized_query = query.strip()
            if normalized_name != "web_search" or not normalized_query:
                continue
            tool_calls.append(
                ToolInvocation(
                    tool_call_id=f"text-tool-call-{index}",
                    name="web_search",
                    arguments_json=json.dumps(
                        {"query": normalized_query},
                        ensure_ascii=False,
                    ),
                    tool_type="function",
                )
            )
        return tuple(tool_calls)

    def _extract_glm_function_calls(self, content: str) -> tuple[ToolInvocation, ...]:
        """Extract ChatGLM ✿FUNCTION✿{json}✿RESULT✿ tool calls."""
        matches = _GLM_FUNCTION_PATTERN.findall(content)
        if not matches:
            return ()
        tool_calls: list[ToolInvocation] = []
        for index, json_text in enumerate(matches):
            try:
                payload = json.loads(json_text)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            name = payload.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            arguments = payload.get("arguments") or payload.get("parameters") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"input": arguments}
            tool_calls.append(
                ToolInvocation(
                    tool_call_id=f"text-tool-call-{index}",
                    name=name.strip(),
                    arguments_json=json.dumps(
                        arguments if isinstance(arguments, dict) else {},
                        ensure_ascii=False,
                    ),
                    tool_type="function",
                )
            )
        return tuple(tool_calls)

    def _extract_plugin_calls(self, content: str) -> tuple[ToolInvocation, ...]:
        """Extract older Qwen <|plugin|>{json}<|endoftext|> tool calls."""
        matches = _QWEN_PLUGIN_PATTERN.findall(content)
        if not matches:
            return ()
        tool_calls: list[ToolInvocation] = []
        for index, json_text in enumerate(matches):
            try:
                payload = json.loads(json_text)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            name = payload.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            arguments = payload.get("parameters") or payload.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"input": arguments}
            tool_calls.append(
                ToolInvocation(
                    tool_call_id=f"text-tool-call-{index}",
                    name=name.strip(),
                    arguments_json=json.dumps(
                        arguments if isinstance(arguments, dict) else {},
                        ensure_ascii=False,
                    ),
                    tool_type="function",
                )
            )
        return tuple(tool_calls)

    def _extract_hermes_tool_calls(self, content: str) -> tuple[ToolInvocation, ...]:
        """Extract Hermes / NousResearch <tool_call>{json}</tool_call> calls."""
        bodies = _HERMES_OUTER_PATTERN.findall(content)
        return self._tool_calls_from_body_blobs(bodies)

    def _extract_llama_python_tag_calls(
        self,
        content: str,
    ) -> tuple[ToolInvocation, ...]:
        """Extract Llama 3.x <|python_tag|>{json}<|eom_id|> calls."""
        bodies = _LLAMA_OUTER_PATTERN.findall(content)
        return self._tool_calls_from_body_blobs(bodies)

    def _extract_mistral_tool_calls(self, content: str) -> tuple[ToolInvocation, ...]:
        """Extract Mistral [TOOL_CALLS][{...}, ...] calls."""
        bodies = _MISTRAL_OUTER_PATTERN.findall(content)
        return self._tool_calls_from_body_blobs(bodies)

    def _extract_json_tool_calls(self, content: str) -> tuple[ToolInvocation, ...]:
        """Extract bare `{"name": ..., "arguments": ...}` / `{"tool": ...}` blobs.

        Used as a last-resort fallback for models that emit a raw JSON tool-call
        object outside any wrapper.  Requires both a name-ish field AND an
        arguments-ish field to reduce false positives from incidental JSON.
        """
        decoder = json.JSONDecoder()
        tool_calls: list[ToolInvocation] = []

        for start_index, char in enumerate(content):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(content[start_index:])
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            tool_call = _tool_call_from_dict(payload, index=len(tool_calls))
            if tool_call is not None:
                tool_calls.append(tool_call)
        return tuple(tool_calls)

    def _tool_calls_from_body_blobs(
        self,
        bodies: list[str],
    ) -> tuple[ToolInvocation, ...]:
        """Parse JSON object(s) / array out of arbitrary wrapper bodies.

        Uses ``json.JSONDecoder.raw_decode`` so nested arguments objects are
        matched correctly and trailing prose is tolerated.  Each body may hold
        either a single object or a list of objects.
        """
        decoder = json.JSONDecoder()
        tool_calls: list[ToolInvocation] = []

        for body in bodies:
            cursor = 0
            stripped_body = body
            # Scan through the body looking for successive JSON values.
            while cursor < len(stripped_body):
                while cursor < len(stripped_body) and stripped_body[cursor] not in "{[":
                    cursor += 1
                if cursor >= len(stripped_body):
                    break
                try:
                    payload, offset = decoder.raw_decode(stripped_body[cursor:])
                except json.JSONDecodeError:
                    cursor += 1
                    continue
                cursor += offset
                if isinstance(payload, dict):
                    tool_call = _tool_call_from_dict(payload, index=len(tool_calls))
                    if tool_call is not None:
                        tool_calls.append(tool_call)
                elif isinstance(payload, list):
                    for item in payload:
                        if not isinstance(item, dict):
                            continue
                        tool_call = _tool_call_from_dict(item, index=len(tool_calls))
                        if tool_call is not None:
                            tool_calls.append(tool_call)
        return tuple(tool_calls)

    def has_assistant_tool_calls(self, messages: list[dict[str, object]]) -> bool:
        for message in messages:
            if message.get("role") != "assistant":
                continue
            if isinstance(message.get("tool_calls"), list):
                return True
        return False

    def messages_with_structured_tool_arguments(
        self,
        messages: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        normalized_messages: list[dict[str, object]] = []
        for message in messages:
            normalized_message = dict(message)
            raw_tool_calls = normalized_message.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                normalized_tool_calls: list[dict[str, object]] = []
                for tool_call in raw_tool_calls:
                    if not isinstance(tool_call, dict):
                        normalized_tool_calls.append(tool_call)
                        continue
                    normalized_call = dict(tool_call)
                    function = normalized_call.get("function")
                    if isinstance(function, dict):
                        normalized_function = dict(function)
                        arguments = normalized_function.get("arguments")
                        if isinstance(arguments, str):
                            try:
                                parsed_arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                parsed_arguments = arguments
                            normalized_function["arguments"] = parsed_arguments
                        normalized_call["function"] = normalized_function
                    normalized_tool_calls.append(normalized_call)
                normalized_message["tool_calls"] = normalized_tool_calls
            normalized_messages.append(normalized_message)
        return normalized_messages

    def messages_with_normalized_tool_call_ids(
        self,
        messages: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        normalized_messages: list[dict[str, object]] = []
        pending_tool_call_ids: list[str] = []

        for message_index, message in enumerate(messages):
            normalized_message = dict(message)
            role = normalized_message.get("role")

            raw_tool_calls = normalized_message.get("tool_calls")
            if role == "assistant" and isinstance(raw_tool_calls, list):
                normalized_tool_calls: list[dict[str, object]] = []
                assistant_call_ids: list[str] = []
                for call_index, tool_call in enumerate(raw_tool_calls):
                    if not isinstance(tool_call, dict):
                        normalized_tool_calls.append(tool_call)
                        continue
                    normalized_call = dict(tool_call)
                    raw_call_id = normalized_call.get("id")
                    if isinstance(raw_call_id, str) and raw_call_id.strip():
                        call_id = raw_call_id.strip()
                    else:
                        call_id = f"tool-call-{message_index}-{call_index}"
                    normalized_call["id"] = call_id
                    normalized_tool_calls.append(normalized_call)
                    assistant_call_ids.append(call_id)
                normalized_message["tool_calls"] = normalized_tool_calls
                pending_tool_call_ids.extend(assistant_call_ids)
                normalized_messages.append(normalized_message)
                continue

            if role == "tool":
                raw_tool_call_id = normalized_message.get("tool_call_id")
                tool_call_id = (
                    raw_tool_call_id.strip()
                    if isinstance(raw_tool_call_id, str) and raw_tool_call_id.strip()
                    else None
                )
                if tool_call_id is None and pending_tool_call_ids:
                    tool_call_id = pending_tool_call_ids.pop(0)
                elif tool_call_id is not None and tool_call_id in pending_tool_call_ids:
                    pending_tool_call_ids.remove(tool_call_id)
                elif tool_call_id is not None and pending_tool_call_ids:
                    # Drop orphan tool messages that cannot be matched to any pending
                    # assistant tool call. Strict providers such as Kimi reject them.
                    continue
                elif tool_call_id is None:
                    # Drop tool messages that have no resolvable tool_call_id.
                    continue
                normalized_message["tool_call_id"] = tool_call_id
                normalized_messages.append(normalized_message)
                continue

            normalized_messages.append(normalized_message)

        return normalized_messages

    def _tool_calls_from_items(self, raw_calls: object) -> tuple[ToolInvocation, ...]:
        if not isinstance(raw_calls, list):
            return ()

        tool_calls: list[ToolInvocation] = []
        for index, item in enumerate(raw_calls):
            if not isinstance(item, dict):
                continue
            function = item.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            arguments = function.get("arguments", "{}")
            tool_type = item.get("type", "function")
            if not isinstance(name, str):
                continue
            if isinstance(arguments, dict):
                arguments_json = json.dumps(arguments, ensure_ascii=False)
            else:
                arguments_json = str(arguments)
            tool_calls.append(
                ToolInvocation(
                    tool_call_id=str(item.get("id") or f"tool-call-{index}"),
                    name=name,
                    arguments_json=arguments_json,
                    tool_type=str(tool_type) if tool_type is not None else "function",
                )
            )
        return tuple(tool_calls)
