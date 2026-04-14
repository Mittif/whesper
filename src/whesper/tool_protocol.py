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

# Fenced (```...```) and inline (`...`) code segments where tool-call-like
# syntax is almost always a quoted example rather than a real invocation.
_CODE_FENCE_PATTERN = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_PATTERN = re.compile(r"`[^`\n]*`")


def _strip_code_segments(content: str) -> str:
    without_fences = _CODE_FENCE_PATTERN.sub(" ", content)
    return _INLINE_CODE_PATTERN.sub(" ", without_fences)


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
        return ()

    def _extract_function_calls_block(self, content: str) -> tuple[ToolInvocation, ...]:
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
