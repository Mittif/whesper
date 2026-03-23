from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable

from whesper.client import ToolCall
from whesper.live_data import LocalWeatherContextService, SearchContextService, should_fetch_local_weather


class ToolExecutionError(RuntimeError):
    pass


ToolHandler = Callable[[dict[str, object]], dict[str, object]]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters_schema: dict[str, object]
    handler: ToolHandler

    def as_openai_tool(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


@dataclass(slots=True)
class ToolExecutionResult:
    tool_call_id: str
    name: str
    content: str


class ToolRegistry:
    def __init__(self, specs: tuple[ToolSpec, ...]) -> None:
        self.specs = specs
        self._by_name = {spec.name: spec for spec in specs}

    @classmethod
    def default(cls) -> "ToolRegistry":
        search_service = SearchContextService()
        weather_service = LocalWeatherContextService()

        return cls(
            specs=(
                ToolSpec(
                    name="web_search",
                    description=(
                        "Search the web for recent or factual information when you are unsure "
                        "or need up-to-date context."
                    ),
                    parameters_schema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to look up on the web.",
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: _handle_web_search(search_service, arguments),
                ),
                ToolSpec(
                    name="get_local_weather",
                    description=(
                        "Get current weather for the user's current IP-based location. "
                        "Use when asked about the user's local weather right now or today."
                    ),
                    parameters_schema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: _handle_local_weather(weather_service, arguments),
                ),
            )
        )

    def openai_tools(self) -> list[dict[str, object]]:
        return [spec.as_openai_tool() for spec in self.specs]

    def tool_prompt(self) -> str:
        tool_names = ", ".join(spec.name for spec in self.specs)
        return (
            "Tool use policy:\n"
            f"- available_tools: {tool_names}\n"
            "- Use a tool only when you need fresh factual data or are unsure.\n"
            "- Prefer staying conversational and avoid unnecessary tool use.\n"
            "- After receiving tool results, answer naturally and briefly."
        )

    def should_offer_tools(self, user_text: str, *, route_mode: str) -> bool:
        text = user_text.strip()
        if route_mode == "search":
            return True
        if text.startswith("/search "):
            return True
        return should_fetch_local_weather(text)

    def default_tool_choice(self, *, route_mode: str) -> str | None:
        if route_mode == "search":
            return "required"
        return None

    def execute(self, tool_call: ToolCall) -> ToolExecutionResult:
        try:
            spec = self._by_name[tool_call.name]
        except KeyError as exc:
            raise ToolExecutionError(f"Unknown tool: {tool_call.name}") from exc

        try:
            arguments = json.loads(tool_call.arguments_json) if tool_call.arguments_json else {}
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(
                f"Tool '{tool_call.name}' received invalid JSON arguments."
            ) from exc

        if not isinstance(arguments, dict):
            raise ToolExecutionError(
                f"Tool '{tool_call.name}' expects a JSON object as arguments."
            )

        try:
            payload = spec.handler(arguments)
        except Exception as exc:
            raise ToolExecutionError(
                f"Tool '{tool_call.name}' failed: {exc.__class__.__name__}: {exc}"
            ) from exc

        return ToolExecutionResult(
            tool_call_id=tool_call.tool_call_id,
            name=tool_call.name,
            content=json.dumps(payload, ensure_ascii=False),
        )


def _handle_web_search(
    search_service: SearchContextService,
    arguments: dict[str, object],
) -> dict[str, object]:
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolExecutionError("web_search requires a non-empty 'query' string.")

    context = search_service.build_prompt_context(query.strip(), route_mode="search")
    return {
        "query": query.strip(),
        "result": context or "No concise search result was available.",
    }


def _handle_local_weather(
    weather_service: LocalWeatherContextService,
    arguments: dict[str, object],
) -> dict[str, object]:
    if arguments:
        raise ToolExecutionError("get_local_weather does not accept arguments.")

    report = weather_service._lookup_weather_report()
    return {
        "location": report.location_label,
        "timezone": report.timezone,
        "public_ip": report.public_ip,
        "current_condition": report.current_condition,
        "current_temperature_c": report.current_temperature_c,
        "apparent_temperature_c": report.apparent_temperature_c,
        "relative_humidity_percent": report.relative_humidity_percent,
        "wind_speed_kmh": report.wind_speed_kmh,
        "today_condition": report.today_condition,
        "today_high_c": report.today_high_c,
        "today_low_c": report.today_low_c,
        "today_precipitation_probability_max_percent": (
            report.today_precipitation_probability_max_percent
        ),
    }
