from __future__ import annotations

from dataclasses import dataclass
import json
import re
from xml.etree import ElementTree

from whesper.agent_types import ToolInvocation


@dataclass(slots=True, frozen=True)
class ToolMessageFormat:
    assistant_tool_content_null: bool = True
    tool_arguments_mode: str = "string"
    include_tool_name: bool = False


FUNCTION_CALLS_BLOCK_PATTERN = re.compile(
    r"<function_calls>\s*.*?</function_calls>",
    re.DOTALL,
)


class DefaultToolProtocolAdapter:
    DEFAULT_TOOL_MESSAGE_FORMAT = ToolMessageFormat()
    KIMI_TOOL_MESSAGE_FORMAT = ToolMessageFormat(
        assistant_tool_content_null=True,
        tool_arguments_mode="string",
        include_tool_name=False,
    )
    QWEN_TOOL_MESSAGE_FORMAT = ToolMessageFormat(
        assistant_tool_content_null=True,
        tool_arguments_mode="object",
        include_tool_name=False,
    )

    def tool_message_format(self, model, provider) -> ToolMessageFormat:
        candidates = (
            getattr(model, "name", ""),
            getattr(model, "model", ""),
            getattr(provider, "name", ""),
        )
        lowered = " ".join(str(item).casefold() for item in candidates if item)
        if "kimi" in lowered or "moonshot" in lowered:
            return self.KIMI_TOOL_MESSAGE_FORMAT
        if "qwen" in lowered:
            return self.QWEN_TOOL_MESSAGE_FORMAT
        if getattr(provider, "kind", "") == "ollama_native":
            return self.QWEN_TOOL_MESSAGE_FORMAT
        return self.DEFAULT_TOOL_MESSAGE_FORMAT

    def tool_call_payload(self, tool_call: ToolInvocation) -> dict[str, object]:
        return {
            "id": tool_call.tool_call_id,
            "type": "function",
            "function": {
                "name": tool_call.name,
                "arguments": tool_call.arguments_json,
            },
        }

    def serialize_tool_calls(
        self,
        tool_calls: list[dict[str, object]],
        *,
        tool_message_format: ToolMessageFormat,
    ) -> list[dict[str, object]]:
        serialized_calls: list[dict[str, object]] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                serialized_calls.append(tool_call)
                continue
            serialized_call = dict(tool_call)
            function = serialized_call.get("function")
            if isinstance(function, dict):
                serialized_function = dict(function)
                arguments = serialized_function.get("arguments")
                if (
                    tool_message_format.tool_arguments_mode == "object"
                    and isinstance(arguments, str)
                ):
                    try:
                        serialized_function["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError:
                        serialized_function["arguments"] = arguments
                elif (
                    tool_message_format.tool_arguments_mode == "string"
                    and isinstance(arguments, dict)
                ):
                    serialized_function["arguments"] = json.dumps(
                        arguments,
                        ensure_ascii=False,
                    )
                serialized_call["function"] = serialized_function
            serialized_calls.append(serialized_call)
        return serialized_calls

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

    def extract_tool_calls_from_text(self, content: str) -> tuple[ToolInvocation, ...]:
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
                )
            )
        return tuple(tool_calls)
