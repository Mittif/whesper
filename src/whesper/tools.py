from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Callable

from whesper.config import AppConfig
from whesper.agent_types import ToolExecutionMeta, ToolInvocation
from whesper.shell_sandbox import ShellSandboxError, ShellSandboxExecutor
from whesper.live_data import (
    CustomApiContextService,
    DOC_KEYWORDS,
    EXCHANGE_KEYWORDS,
    ExchangeRateContextService,
    FLIGHT_KEYWORDS,
    GeocodingContextService,
    LOCAL_TIME_KEYWORDS,
    LocalWeatherContextService,
    MAP_KEYWORDS,
    MARKET_KEYWORDS,
    MarketPriceContextService,
    NEWS_KEYWORDS,
    NewsContextService,
    PARCEL_KEYWORDS,
    SearchContextService,
    SEARCH_KEYWORDS,
    TIME_KEYWORDS,
    TRAIN_KEYWORDS,
    TechDocsContextService,
    TimeContextService,
    TransportStatusContextService,
    UrlSummaryContextService,
    _contains_any,
    _extract_urls,
    _extract_weather_request_window,
    _looks_like_docs_url,
    _normalize_place_candidate,
    _weather_reference_date,
    lookup_ip_location,
    lookup_public_ip,
    lookup_weather_for_ip,
    lookup_weather_for_place,
    should_fetch_weather,
)


class ToolExecutionError(RuntimeError):
    pass


ToolHandler = Callable[[dict[str, object]], dict[str, object]]
RequestMatcher = Callable[[str, str], bool]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters_schema: dict[str, object]
    handler: ToolHandler
    usage_guidance: str | None = None
    examples: tuple[tuple[str, str], ...] = ()
    should_offer: RequestMatcher | None = None
    execution_meta: ToolExecutionMeta = ToolExecutionMeta()

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
    def default(cls, config: AppConfig | None = None) -> "ToolRegistry":
        search_service = SearchContextService()
        url_summary_service = UrlSummaryContextService(fetch_text=_tool_fetch_text)
        tech_docs_service = TechDocsContextService(fetch_text=_tool_fetch_text)
        exchange_service = ExchangeRateContextService(fetch_json=_tool_fetch_json)
        market_service = MarketPriceContextService(fetch_json=_tool_fetch_json)
        geocoding_service = GeocodingContextService(fetch_json=_tool_fetch_json)
        time_service = TimeContextService(fetch_json=_tool_fetch_json)
        news_service = NewsContextService(fetch_text=_tool_fetch_text)

        specs: list[ToolSpec] = [
            ToolSpec(
                name="get_public_ip",
                description=(
                    "Get the user's current public IP address. Useful as a first step "
                    "when you need to determine the user's approximate location."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=_handle_get_public_ip,
                usage_guidance=(
                    "Use as the first step when you need the user's location but they "
                    "haven't specified one. Follow up with get_ip_location to resolve "
                    "the IP to a city/region."
                ),
                examples=(
                    ('用户说: "我这里天气怎么样"（未指定地点）', "{}"),
                ),
                execution_meta=ToolExecutionMeta(timeout_seconds=5.0),
            ),
            ToolSpec(
                name="get_ip_location",
                description=(
                    "Resolve a public IP address to a geographic location including "
                    "city, region, country, coordinates, and timezone."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "ip": {
                            "type": "string",
                            "description": "The public IP address to geolocate.",
                        }
                    },
                    "required": ["ip"],
                    "additionalProperties": False,
                },
                handler=_handle_get_ip_location,
                usage_guidance=(
                    "Use after get_public_ip to resolve an IP to a location. The result "
                    "includes city, region, country, and timezone which can then be used "
                    "with location-based tools like get_weather_by_location."
                ),
                examples=(
                    ('已获取 IP "203.0.113.42"', '{"ip":"203.0.113.42"}'),
                ),
                execution_meta=ToolExecutionMeta(timeout_seconds=5.0),
            ),
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
                usage_guidance=(
                    "Use when the user explicitly asks to search, wants recent information, "
                    "or the answer is likely to have changed recently."
                ),
                examples=(
                    ('用户说: "/search openai release notes"', '{"query":"openai release notes"}'),
                    ('用户说: "最新 AI 新闻"', '{"query":"最新 AI 新闻"}'),
                ),
                should_offer=lambda user_text, route_mode: (
                    route_mode == "search"
                    or user_text.strip().startswith("/search ")
                    or _contains_any(user_text, SEARCH_KEYWORDS)
                ),
            ),
            ToolSpec(
                name="summarize_webpage",
                description=(
                    "Open a user-provided URL and return a lightweight page snapshot."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The exact webpage URL provided by the user.",
                        }
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
                handler=lambda arguments: _handle_url_context_tool(
                    "summarize_webpage",
                    arguments,
                    url_summary_service,
                ),
                usage_guidance=(
                    "Use when the user includes a URL and wants you to inspect the page "
                    "instead of guessing what it contains."
                ),
                examples=(
                    (
                        '用户说: "帮我看看这个页面 https://example.com/docs"',
                        '{"url":"https://example.com/docs"}',
                    ),
                ),
                should_offer=lambda user_text, route_mode: bool(_extract_urls(user_text)),
            ),
            ToolSpec(
                name="read_tech_docs",
                description=(
                    "Read a technical documentation URL and return a concise doc snapshot."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Documentation or API reference URL from the user.",
                        }
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
                handler=lambda arguments: _handle_url_context_tool(
                    "read_tech_docs",
                    arguments,
                    tech_docs_service,
                ),
                usage_guidance=(
                    "Use when the user asks about documentation, API reference pages, guides, "
                    "or README-like URLs."
                ),
                examples=(
                    (
                        '用户说: "请看这个 API 文档 https://example.com/docs/sdk"',
                        '{"url":"https://example.com/docs/sdk"}',
                    ),
                ),
                should_offer=lambda user_text, route_mode: bool(_extract_urls(user_text))
                and (
                    _contains_any(user_text, DOC_KEYWORDS)
                    or any(_looks_like_docs_url(url) for url in _extract_urls(user_text))
                ),
            ),
            ToolSpec(
                name="get_local_weather",
                description=(
                    "Get live weather for the user's current area. Accept a raw time "
                    "expression such as 今天, 明天, 后天, 周末, or next Monday."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "time_expression": {
                            "type": "string",
                            "description": (
                                "Optional raw time phrase from the user, such as 明天, 周末, "
                                "today, tomorrow, or next Monday."
                            ),
                        }
                    },
                    "additionalProperties": False,
                },
                handler=_handle_get_local_weather,
                usage_guidance=(
                    "Use for the user's current local weather. Keep the user's time phrase raw; "
                    "the code will resolve dates and forecast windows."
                ),
                examples=(
                    ('用户说: "我这里明天的天气"', '{"time_expression":"明天"}'),
                    ('用户说: "this weekend weather"', '{"time_expression":"this weekend"}'),
                ),
                should_offer=lambda user_text, route_mode: _matches_local_weather_intent(user_text),
            ),
            ToolSpec(
                name="get_weather_by_location",
                description=(
                    "Get live weather for a named place. Pass the location separately and keep "
                    "the user's time phrase raw when present."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": (
                                "Named place from the user, such as 嘉定区, Shanghai, or Tokyo."
                            ),
                        },
                        "time_expression": {
                            "type": "string",
                            "description": (
                                "Optional raw time phrase from the user, such as 明天, 周末, "
                                "today, tomorrow, or next Monday."
                            ),
                        },
                    },
                    "required": ["location"],
                    "additionalProperties": False,
                },
                handler=_handle_get_weather_by_location,
                usage_guidance=(
                    "Use for weather in a named place. Put only the place in location, and put "
                    "the time phrase in time_expression instead of mixing them together."
                ),
                examples=(
                    ('用户说: "嘉定区明天的天气"', '{"location":"嘉定区","time_expression":"明天"}'),
                    (
                        '用户说: "Tokyo weather next Monday"',
                        '{"location":"Tokyo","time_expression":"next Monday"}',
                    ),
                ),
                should_offer=lambda user_text, route_mode: (
                    should_fetch_weather(user_text) and not _matches_local_weather_intent(user_text)
                ),
            ),
            ToolSpec(
                name="lookup_exchange_rate",
                description=(
                    "Get live exchange-rate data from a raw currency-conversion query."
                ),
                parameters_schema=_query_tool_schema(
                    "Original user request about exchange rates or currency conversion."
                ),
                handler=lambda arguments: _handle_query_context_tool(
                    "lookup_exchange_rate",
                    arguments,
                    exchange_service,
                ),
                usage_guidance=(
                    "Use for currency conversion and FX questions. Pass the original request "
                    "text so the code can extract the currency pair."
                ),
                examples=(
                    (
                        '用户说: "美元兑人民币汇率现在是多少？"',
                        '{"query":"美元兑人民币汇率现在是多少？"}',
                    ),
                ),
                should_offer=lambda user_text, route_mode: _contains_any(user_text, EXCHANGE_KEYWORDS),
            ),
            ToolSpec(
                name="lookup_market_price",
                description=(
                    "Get live market prices from a raw stock or crypto query."
                ),
                parameters_schema=_query_tool_schema(
                    "Original user request about stocks, crypto, market cap, or price."
                ),
                handler=lambda arguments: _handle_query_context_tool(
                    "lookup_market_price",
                    arguments,
                    market_service,
                ),
                usage_guidance=(
                    "Use for stock and crypto price questions. Pass the original request text "
                    "so the code can extract symbols deterministically."
                ),
                examples=(
                    ('用户说: "Tesla and bitcoin price today"', '{"query":"Tesla and bitcoin price today"}'),
                ),
                should_offer=lambda user_text, route_mode: _contains_any(user_text, MARKET_KEYWORDS),
            ),
            ToolSpec(
                name="lookup_place",
                description=(
                    "Resolve a place query into coordinates, country, and timezone."
                ),
                parameters_schema=_query_tool_schema(
                    "Original user request asking where a place is, or asking for coordinates."
                ),
                handler=lambda arguments: _handle_query_context_tool(
                    "lookup_place",
                    arguments,
                    geocoding_service,
                ),
                usage_guidance=(
                    "Use for map, coordinate, latitude, longitude, or address-style questions."
                ),
                examples=(
                    ('用户说: "Hangzhou coordinates"', '{"query":"Hangzhou coordinates"}'),
                ),
                should_offer=lambda user_text, route_mode: _contains_any(user_text, MAP_KEYWORDS),
            ),
            ToolSpec(
                name="lookup_time",
                description=(
                    "Get live local time or timezone information from a raw location query."
                ),
                parameters_schema=_query_tool_schema(
                    "Original user request about local time, timezone, time difference, or holidays."
                ),
                handler=lambda arguments: _handle_query_context_tool(
                    "lookup_time",
                    arguments,
                    time_service,
                ),
                usage_guidance=(
                    "Use for local time, timezone, time difference, and holiday calendar questions. "
                    "Pass the original request text."
                ),
                examples=(
                    ('用户说: "current time in Singapore"', '{"query":"current time in Singapore"}'),
                    ('用户说: "2026 Japan holidays"', '{"query":"2026 Japan holidays"}'),
                ),
                should_offer=lambda user_text, route_mode: _contains_any(user_text, TIME_KEYWORDS),
            ),
            ToolSpec(
                name="get_news",
                description=(
                    "Get a lightweight live news snapshot from a raw current-events query."
                ),
                parameters_schema=_query_tool_schema(
                    "Original user request about news or headlines."
                ),
                handler=lambda arguments: _handle_query_context_tool(
                    "get_news",
                    arguments,
                    news_service,
                ),
                usage_guidance=(
                    "Use for headlines and current-news requests. Pass the user's original news topic."
                ),
                examples=(
                    ('用户说: "AI 新闻"', '{"query":"AI 新闻"}'),
                ),
                should_offer=lambda user_text, route_mode: _contains_any(user_text, NEWS_KEYWORDS),
            ),
        ]

        if config is not None and config.shell_sandbox.enabled:
            shell_executor = ShellSandboxExecutor(
                config.shell_sandbox,
                workspace_root=config.source_path.parent,
            )
            supported_commands = shell_executor.describe_supported_commands()
            specs.append(
                ToolSpec(
                    name="run_shell_command",
                    description=(
                        "Execute a sandboxed workspace shell command from a small allowlist. "
                        "This tool is for repository inspection only and does not allow pipes, "
                        "redirection, or command chaining."
                    ),
                    parameters_schema={
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": (
                                    "The exact command to run. Use only a supported prefix and "
                                    "do not include shell operators like |, >, &&, or ;."
                                ),
                            },
                            "cwd": {
                                "type": "string",
                                "description": (
                                    "Optional working directory relative to the workspace root."
                                ),
                            },
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: _handle_run_shell_command(
                        shell_executor,
                        arguments,
                    ),
                    usage_guidance=(
                        "Use only when repository inspection or safe local command output is needed. "
                        f"Supported prefixes: {supported_commands}. Prefer direct file-reading tools when possible."
                    ),
                    examples=(
                        ('用户说: "帮我看下仓库根目录有哪些文件"', '{"command":"ls","cwd":"."}'),
                        ('用户说: "在 src 里搜 AgentHarness"', '{"command":"rg AgentHarness src","cwd":"."}'),
                        ('用户说: "看一下 git 状态"', '{"command":"git status","cwd":"."}'),
                    ),
                    should_offer=lambda user_text, route_mode: _matches_shell_command_intent(
                        user_text
                    ),
                    execution_meta=ToolExecutionMeta(
                        timeout_seconds=float(config.shell_sandbox.timeout_seconds),
                        side_effectful=False,
                    ),
                )
            )

        if config is not None and config.live_context.status_api:
            transport_service = TransportStatusContextService(
                config.live_context.status_api,
                fetch_json=_tool_fetch_json_with_headers,
                fetch_text=_tool_fetch_text_with_headers,
            )
            endpoint_names = ", ".join(sorted(config.live_context.status_api))
            specs.append(
                ToolSpec(
                    name="lookup_transport_status",
                    description=(
                        "Query configured transport or tracking status endpoints using the raw user request."
                    ),
                    parameters_schema=_query_tool_schema(
                        "Original user request containing a flight number, train number, or tracking code."
                    ),
                    handler=lambda arguments: _handle_query_context_tool(
                        "lookup_transport_status",
                        arguments,
                        transport_service,
                    ),
                    usage_guidance=(
                        "Use for flight, train, parcel, or tracking requests when matching endpoints are configured. "
                        f"Configured endpoint keys: {endpoint_names}."
                    ),
                    examples=(
                        ('用户说: "Flight MU5123 status"', '{"query":"Flight MU5123 status"}'),
                    ),
                    should_offer=lambda user_text, route_mode: _contains_any(
                        user_text,
                        FLIGHT_KEYWORDS + TRAIN_KEYWORDS + PARCEL_KEYWORDS,
                    ),
                )
            )

        if config is not None and config.live_context.custom_api:
            custom_service = CustomApiContextService(
                config.live_context.custom_api,
                fetch_json=_tool_fetch_json_with_headers,
                fetch_text=_tool_fetch_text_with_headers,
            )
            trigger_keywords = tuple(
                keyword
                for endpoint in config.live_context.custom_api.values()
                for keyword in endpoint.trigger_keywords
            )
            endpoint_names = ", ".join(sorted(config.live_context.custom_api))
            specs.append(
                ToolSpec(
                    name="lookup_custom_live_data",
                    description=(
                        "Query configured custom live-data endpoints using the raw user request."
                    ),
                    parameters_schema=_query_tool_schema(
                        "Original user request that matches a configured custom live-data endpoint."
                    ),
                    handler=lambda arguments: _handle_query_context_tool(
                        "lookup_custom_live_data",
                        arguments,
                        custom_service,
                    ),
                    usage_guidance=(
                        "Use for project-specific live-data integrations whose trigger keywords match the user's request. "
                        f"Configured endpoint keys: {endpoint_names}."
                    ),
                    should_offer=lambda user_text, route_mode: bool(trigger_keywords)
                    and _contains_any(user_text, trigger_keywords),
                )
            )

        return cls(specs=tuple(specs))

    def openai_tools(self) -> list[dict[str, object]]:
        return [
            spec.as_openai_tool()
            for spec in self.specs
            if spec.execution_meta.visible_to_model
        ]

    def tool_prompt(self) -> str:
        visible_specs = [
            spec for spec in self.specs if spec.execution_meta.visible_to_model
        ]
        tool_names = ", ".join(spec.name for spec in visible_specs)
        lines = [
            "Tool skills:",
            f"- available_tools: {tool_names}",
            "- Use tools when you need fresh factual data, external pages, or project-specific live endpoints.",
            "- Prefer high-level tools that accept raw user phrases. Keep time expressions and user wording natural unless a tool explicitly needs normalization.",
            "- Do not invent arguments that the user did not imply. If a location, URL, or code is missing, ask naturally instead of guessing.",
            "- After receiving tool results, answer naturally and briefly.",
        ]
        for spec in visible_specs:
            lines.append(f"- tool: {spec.name}")
            lines.append(f"  description: {spec.description}")
            if spec.usage_guidance:
                lines.append(f"  use_when: {spec.usage_guidance}")
            for argument_line in _tool_argument_lines(spec):
                lines.append(f"  {argument_line}")
            for user_example, argument_example in spec.examples:
                lines.append(f"  example: {user_example} -> {argument_example}")
        return "\n".join(lines)

    def should_offer_tools(self, user_text: str, *, route_mode: str) -> bool:
        visible_specs = [
            spec for spec in self.specs if spec.execution_meta.visible_to_model
        ]
        if any(
            spec.should_offer is not None and spec.should_offer(user_text, route_mode)
            for spec in visible_specs
        ):
            return True
        if not any(spec.should_offer is not None for spec in visible_specs):
            text = user_text.strip()
            if route_mode == "search":
                return True
            if text.startswith("/search "):
                return True
            return should_fetch_weather(text)
        return False

    def default_tool_choice(self, *, route_mode: str) -> str | None:
        if route_mode == "search":
            return "required"
        return None

    def execute(self, tool_call: ToolInvocation) -> ToolExecutionResult:
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
    query = _require_string_argument(arguments, "query", tool_name="web_search")
    context = search_service.build_prompt_context(query, route_mode="search")
    return {
        "query": query,
        "result": context or "No concise search result was available.",
    }


def _handle_get_public_ip(arguments: dict[str, object]) -> dict[str, object]:
    if arguments:
        raise ToolExecutionError("get_public_ip does not accept arguments.")

    public_ip = lookup_public_ip(_tool_fetch_json, _TOOL_TIMEOUT_SECONDS)
    return {"public_ip": public_ip}


def _handle_get_ip_location(arguments: dict[str, object]) -> dict[str, object]:
    public_ip = arguments.get("ip")
    if not isinstance(public_ip, str) or not public_ip.strip():
        raise ToolExecutionError("get_ip_location requires a non-empty 'ip' string.")

    location = lookup_ip_location(_tool_fetch_json, _TOOL_TIMEOUT_SECONDS, public_ip.strip())
    return {
        "public_ip": location.public_ip,
        "location": location.label,
        "city": location.city,
        "region": location.region,
        "country": location.country,
        "latitude": location.latitude,
        "longitude": location.longitude,
        "timezone": location.timezone,
    }


def _handle_get_weather_by_location(arguments: dict[str, object]) -> dict[str, object]:
    location = _require_string_argument(
        arguments,
        "location",
        tool_name="get_weather_by_location",
    )
    normalized_location = _normalize_place_candidate(location)
    if normalized_location is None:
        raise ToolExecutionError(
            "get_weather_by_location requires 'location' to identify a concrete place."
        )
    time_expression = _optional_string_argument(arguments, "time_expression")
    report = lookup_weather_for_place(
        _tool_fetch_json, _TOOL_TIMEOUT_SECONDS, normalized_location
    )
    query = _weather_query_for_location(normalized_location, time_expression)
    return _weather_payload_from_report(report, request_query=query)


def _handle_get_local_weather(arguments: dict[str, object]) -> dict[str, object]:
    time_expression = _optional_string_argument(arguments, "time_expression")
    report = lookup_weather_for_ip(_tool_fetch_json, _TOOL_TIMEOUT_SECONDS)
    query = _weather_query_for_local(time_expression)
    return _weather_payload_from_report(report, request_query=query)


def _handle_run_shell_command(
    executor: ShellSandboxExecutor,
    arguments: dict[str, object],
) -> dict[str, object]:
    command = _require_string_argument(arguments, "command", tool_name="run_shell_command")
    cwd = arguments.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        raise ToolExecutionError("run_shell_command expects 'cwd' to be a string when provided.")
    try:
        result = executor.execute(command, cwd=cwd)
    except ShellSandboxError as exc:
        return {
            "ok": False,
            "command": command,
            "cwd": cwd or ".",
            "error": str(exc),
        }
    return result.as_payload()


def _handle_query_context_tool(
    tool_name: str,
    arguments: dict[str, object],
    service,
) -> dict[str, object]:
    query = _require_string_argument(arguments, "query", tool_name=tool_name)
    context = service.build_prompt_context(query, route_mode="chat")
    if context is None:
        raise ToolExecutionError(
            f"{tool_name} could not derive live context from the provided query."
        )
    return {
        "query": query,
        "context": context,
    }


def _handle_url_context_tool(
    tool_name: str,
    arguments: dict[str, object],
    service,
) -> dict[str, object]:
    url = _require_string_argument(arguments, "url", tool_name=tool_name)
    context = service.build_prompt_context(url, route_mode="chat")
    if context is None:
        raise ToolExecutionError(f"{tool_name} could not inspect the provided URL.")
    return {
        "url": url,
        "context": context,
    }


def _matches_local_weather_intent(user_text: str) -> bool:
    lowered = user_text.casefold()
    return should_fetch_weather(user_text) and any(
        keyword.casefold() in lowered for keyword in LOCAL_TIME_KEYWORDS
    )


def _matches_shell_command_intent(user_text: str) -> bool:
    lowered = user_text.casefold()
    keywords = (
        "shell",
        "terminal",
        "command",
        "run command",
        "git status",
        "git diff",
        "ls ",
        "rg ",
        "pwd",
        "终端",
        "命令行",
        "shell命令",
        "运行命令",
        "执行命令",
    )
    return any(keyword in lowered for keyword in keywords)


def _query_tool_schema(description: str) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": description,
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    }


def _tool_argument_lines(spec: ToolSpec) -> list[str]:
    properties = spec.parameters_schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return ["arguments: none"]
    required = spec.parameters_schema.get("required", [])
    required_set = {item for item in required if isinstance(item, str)}
    lines: list[str] = []
    for name, payload in properties.items():
        if not isinstance(payload, dict):
            continue
        arg_type = str(payload.get("type", "object"))
        qualifier = "required" if name in required_set else "optional"
        description = str(payload.get("description", "")).strip()
        if description:
            lines.append(f"argument {name} ({arg_type}, {qualifier}): {description}")
        else:
            lines.append(f"argument {name} ({arg_type}, {qualifier})")
    return lines


def _require_string_argument(
    arguments: dict[str, object],
    key: str,
    *,
    tool_name: str,
) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolExecutionError(f"{tool_name} requires a non-empty '{key}' string.")
    return value.strip()


def _optional_string_argument(arguments: dict[str, object], key: str) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolExecutionError(f"'{key}' must be a string when provided.")
    normalized = value.strip()
    return normalized or None


def _weather_query_for_local(time_expression: str | None) -> str:
    if time_expression is None:
        return "我这里今天的天气"
    if _contains_cjk(time_expression):
        return f"我这里{time_expression}的天气"
    return f"weather in my area {time_expression}"


def _weather_query_for_location(location: str, time_expression: str | None) -> str:
    if time_expression is None:
        if _contains_cjk(location):
            return f"{location}的天气"
        return f"{location} weather"
    if _contains_cjk(location) or _contains_cjk(time_expression):
        return f"{location}{time_expression}的天气"
    return f"{location} weather {time_expression}"


def _contains_cjk(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", value))


def _weather_payload_from_report(report, *, request_query: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "location": report.location_label,
        "latitude": report.latitude,
        "longitude": report.longitude,
        "timezone": report.timezone,
        "current_condition": report.current_condition,
        "current_temperature_c": report.current_temperature_c,
        "apparent_temperature_c": report.apparent_temperature_c,
        "relative_humidity_percent": report.relative_humidity_percent,
        "wind_speed_kmh": report.wind_speed_kmh,
        "today": _forecast_payload(
            report.daily_forecasts[0]
            if report.daily_forecasts
            else None,
            fallback_condition=report.today_condition,
            fallback_high_c=report.today_high_c,
            fallback_low_c=report.today_low_c,
            fallback_precipitation_probability_max_percent=(
                report.today_precipitation_probability_max_percent
            ),
        ),
    }
    if report.tomorrow_condition is not None:
        payload["tomorrow"] = {
            "condition": report.tomorrow_condition,
            "high_c": report.tomorrow_high_c,
            "low_c": report.tomorrow_low_c,
            "precipitation_probability_max_percent": (
                report.tomorrow_precipitation_probability_max_percent
            ),
        }

    request_window = _extract_weather_request_window(
        request_query,
        reference_date=_weather_reference_date(report),
    )
    if request_window is not None:
        start_index, end_index, label = request_window
        selected = report.daily_forecasts[start_index : end_index + 1]
        if selected:
            payload["requested_period"] = label
            if len(selected) == 1:
                payload["requested_forecast"] = _forecast_payload(selected[0])
            else:
                payload["requested_forecasts"] = [
                    _forecast_payload(item) for item in selected
                ]
    return payload


def _forecast_payload(
    forecast,
    *,
    fallback_condition: str | None = None,
    fallback_high_c: float | None = None,
    fallback_low_c: float | None = None,
    fallback_precipitation_probability_max_percent: float | None = None,
) -> dict[str, object]:
    if forecast is not None:
        return {
            "date": forecast.date,
            "condition": forecast.condition,
            "high_c": forecast.high_c,
            "low_c": forecast.low_c,
            "precipitation_probability_max_percent": (
                forecast.precipitation_probability_max_percent
            ),
        }
    return {
        "condition": fallback_condition,
        "high_c": fallback_high_c,
        "low_c": fallback_low_c,
        "precipitation_probability_max_percent": (
            fallback_precipitation_probability_max_percent
        ),
    }


_TOOL_TIMEOUT_SECONDS = 10


def _tool_fetch_json(url: str, timeout_seconds: int) -> dict:
    from whesper.live_data import _default_fetch_json

    return _default_fetch_json(url, timeout_seconds)


def _tool_fetch_text(url: str, timeout_seconds: int):
    from whesper.live_data import _default_fetch_text

    return _default_fetch_text(url, timeout_seconds)


def _tool_fetch_json_with_headers(
    url: str,
    timeout_seconds: int,
    headers: dict[str, str],
) -> dict:
    from whesper.live_data import _default_fetch_json_with_headers

    return _default_fetch_json_with_headers(url, timeout_seconds, headers)


def _tool_fetch_text_with_headers(
    url: str,
    timeout_seconds: int,
    headers: dict[str, str],
):
    from whesper.live_data import _default_fetch_text_with_headers

    return _default_fetch_text_with_headers(url, timeout_seconds, headers)
