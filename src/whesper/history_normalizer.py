from __future__ import annotations

import json

from whesper.provider_profile import ProviderProfile
from whesper.session import ChatMessage


class HistoryNormalizer:
    def __init__(self, target_profile: ProviderProfile) -> None:
        self.target_profile = target_profile

    def normalize(
        self,
        messages: list[ChatMessage],
        *,
        inject_reasoning_placeholder: bool = False,
    ) -> list[dict[str, object]]:
        adapted_messages = self._degrade_unsupported_tool_calls(messages)

        payloads: list[dict[str, object]] = []
        for message in adapted_messages:
            payload = self._message_to_payload(
                message,
                inject_reasoning_placeholder=inject_reasoning_placeholder,
            )
            payloads.append(payload)
        return payloads

    def _message_to_payload(
        self,
        message: ChatMessage,
        *,
        inject_reasoning_placeholder: bool,
    ) -> dict[str, object]:
        profile = self.target_profile
        payload: dict[str, object] = {"role": message.role}

        if (
            message.role == "assistant"
            and message.tool_calls is not None
            and not message.content.strip()
            and profile.assistant_tool_content_null
        ):
            payload["content"] = None
        else:
            payload["content"] = message.content

        reasoning_value = self._resolve_reasoning_content(
            message,
            inject_reasoning_placeholder=inject_reasoning_placeholder,
        )
        if reasoning_value is not None:
            payload["reasoning_content"] = reasoning_value

        if message.name is not None and (
            message.role != "tool" or profile.include_tool_name_in_tool_message
        ):
            payload["name"] = message.name

        if message.tool_call_id is not None:
            payload["tool_call_id"] = message.tool_call_id

        if message.tool_calls is not None:
            payload["tool_calls"] = self._serialize_tool_calls(message.tool_calls)

        return payload

    def _resolve_reasoning_content(
        self,
        message: ChatMessage,
        *,
        inject_reasoning_placeholder: bool,
    ) -> str | None:
        profile = self.target_profile
        same_profile = (
            message.source_profile is None
            or message.source_profile == profile.profile_id
        )

        existing = message.reasoning_content or ""
        if same_profile and existing:
            return existing

        if inject_reasoning_placeholder and message.role == "assistant":
            if existing:
                return existing
            return profile.reasoning_content_empty_placeholder

        return None

    def _serialize_tool_calls(
        self,
        tool_calls: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        profile = self.target_profile
        serialized: list[dict[str, object]] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                serialized.append(tool_call)
                continue
            call = dict(tool_call)
            function = call.get("function")
            if isinstance(function, dict):
                function_copy = dict(function)
                arguments = function_copy.get("arguments")
                if profile.tool_arguments_mode == "object" and isinstance(arguments, str):
                    try:
                        function_copy["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError:
                        function_copy["arguments"] = arguments
                elif profile.tool_arguments_mode == "string" and isinstance(arguments, dict):
                    function_copy["arguments"] = json.dumps(arguments, ensure_ascii=False)
                call["function"] = function_copy
            serialized.append(call)
        return serialized

    def _degrade_unsupported_tool_calls(
        self,
        messages: list[ChatMessage],
    ) -> list[ChatMessage]:
        """Fold unsupported builtin tool sequences into text-only assistant messages.

        Example: Kimi's `$web_search` becomes a plain assistant summary when the
        target profile does not advertise that builtin.
        """
        profile = self.target_profile
        output: list[ChatMessage] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if message.role != "assistant" or not message.tool_calls:
                output.append(message)
                index += 1
                continue

            unsupported = [
                call
                for call in message.tool_calls
                if self._is_unsupported_builtin(call, profile)
            ]
            if not unsupported:
                output.append(message)
                index += 1
                continue

            unsupported_ids = {
                str(call.get("id"))
                for call in unsupported
                if isinstance(call.get("id"), str)
            }
            summary_parts: list[str] = []
            if message.content.strip():
                summary_parts.append(message.content.strip())
            for call in unsupported:
                summary_parts.append(_format_builtin_call_summary(call))

            look_ahead = index + 1
            matched_tool_messages: list[ChatMessage] = []
            while (
                look_ahead < len(messages)
                and messages[look_ahead].role == "tool"
                and (
                    messages[look_ahead].tool_call_id is None
                    or messages[look_ahead].tool_call_id in unsupported_ids
                )
            ):
                matched_tool_messages.append(messages[look_ahead])
                look_ahead += 1

            for tool_message in matched_tool_messages:
                trimmed = (tool_message.content or "").strip()
                if trimmed:
                    summary_parts.append(trimmed)

            remaining_calls = [
                call for call in message.tool_calls if call not in unsupported
            ]
            degraded_content = "\n".join(part for part in summary_parts if part)

            if remaining_calls:
                output.append(
                    ChatMessage(
                        role="assistant",
                        content=degraded_content,
                        created_at=message.created_at,
                        model_alias=message.model_alias,
                        route_reason=message.route_reason,
                        reasoning_content=message.reasoning_content,
                        tool_calls=remaining_calls,
                        source_profile=message.source_profile,
                    )
                )
            else:
                output.append(
                    ChatMessage(
                        role="assistant",
                        content=degraded_content or message.content,
                        created_at=message.created_at,
                        model_alias=message.model_alias,
                        route_reason=message.route_reason,
                        reasoning_content=message.reasoning_content,
                        source_profile=message.source_profile,
                    )
                )

            index = look_ahead
        return output

    @staticmethod
    def _is_unsupported_builtin(
        tool_call: dict[str, object],
        profile: ProviderProfile,
    ) -> bool:
        if not isinstance(tool_call, dict):
            return False
        call_type = tool_call.get("type")
        function = tool_call.get("function")
        name = ""
        if isinstance(function, dict):
            raw_name = function.get("name")
            if isinstance(raw_name, str):
                name = raw_name
        if call_type == "builtin_function":
            return name not in profile.builtin_tools
        if isinstance(name, str) and name.startswith("$"):
            return name not in profile.builtin_tools
        return False


def _format_builtin_call_summary(tool_call: dict[str, object]) -> str:
    function = tool_call.get("function")
    name = "builtin_tool"
    arguments_text = ""
    if isinstance(function, dict):
        raw_name = function.get("name")
        if isinstance(raw_name, str) and raw_name:
            name = raw_name
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            arguments_text = arguments
        elif isinstance(arguments, dict):
            try:
                arguments_text = json.dumps(arguments, ensure_ascii=False)
            except (TypeError, ValueError):
                arguments_text = str(arguments)
    if arguments_text:
        return f"[previous {name} call: {arguments_text}]"
    return f"[previous {name} call]"
