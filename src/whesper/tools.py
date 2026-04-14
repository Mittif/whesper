from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import re
from typing import Callable
from urllib import request as urlrequest
from zoneinfo import ZoneInfo

from whesper.config import AppConfig, CUP_DEFAULT_BASE_URL
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
    _extract_map_place,
    _extract_city_weather_place,
    _extract_time_place,
    _extract_urls,
    _extract_weather_request_window,
    _looks_like_docs_url,
    _normalize_place_candidate,
    _weather_reference_date,
    lookup_ip_location,
    lookup_local_area,
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
    related_tools: tuple[str, ...] = ()
    related_tools_when: RequestMatcher | None = None
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
    def __init__(
        self,
        specs: tuple[ToolSpec, ...],
        *,
        search_service: SearchContextService | None = None,
    ) -> None:
        self.specs = specs
        self._by_name = {spec.name: spec for spec in specs}
        self.search_service = search_service

    @classmethod
    def default(cls, config: AppConfig | None = None) -> "ToolRegistry":
        search_settings = config.live_context.search_api if config is not None else None
        search_service = SearchContextService(
            provider=search_settings.provider if search_settings is not None else "duckduckgo",
            api_key=search_settings.resolved_api_key() if search_settings is not None else None,
            engine=search_settings.engine if search_settings is not None else "google",
            timeout_seconds=(
                int(search_settings.timeout_seconds) if search_settings is not None else 10
            ),
        )
        url_summary_service = UrlSummaryContextService(fetch_text=_tool_fetch_text)
        tech_docs_service = TechDocsContextService(fetch_text=_tool_fetch_text)
        exchange_service = ExchangeRateContextService(fetch_json=_tool_fetch_json)
        market_service = MarketPriceContextService(fetch_json=_tool_fetch_json)
        geocoding_service = GeocodingContextService(fetch_json=_tool_fetch_json)
        time_service = TimeContextService(fetch_json=_tool_fetch_json)
        news_service = NewsContextService(fetch_text=_tool_fetch_text)

        specs: list[ToolSpec] = [
            ToolSpec(
                name="web_search",
                description=(
                    "Search the web for recent or factual information when you are unsure "
                    "or need up-to-date context. Returns a lightweight search snapshot plus "
                    "structured source URLs."
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
                    "or needs broad web research beyond a single page or headline snapshot. "
                    "If the request is specifically for concise current-news headlines on a topic, "
                    "get_news is usually a better fit unless the user explicitly asked to search. "
                    "Use the returned source titles or domains when you cite what you relied on."
                ),
                examples=(
                    ('用户说: "/search openai release notes"', '{"query":"openai release notes"}'),
                    ('用户说: "最新 AI 新闻"', '{"query":"最新 AI 新闻"}'),
                ),
                should_offer=lambda user_text, route_mode: (
                    route_mode == "search"
                    or user_text.strip().startswith("/search ")
                    or (
                        _contains_any(user_text, SEARCH_KEYWORDS)
                        and not _contains_any(user_text, NEWS_KEYWORDS)
                    )
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
                    "Use when the user includes a URL and wants you to inspect a general webpage, "
                    "article, landing page, or non-doc page. If the URL or surrounding words indicate "
                    "API docs, technical reference, guides, or README-like documentation, prefer "
                    "read_tech_docs."
                ),
                examples=(
                    (
                        '用户说: "帮我看看这个页面 https://example.com/docs"',
                        '{"url":"https://example.com/docs"}',
                    ),
                ),
                should_offer=lambda user_text, route_mode: bool(_extract_urls(user_text))
                and not (
                    _contains_any(user_text, DOC_KEYWORDS)
                    or any(_looks_like_docs_url(url) for url in _extract_urls(user_text))
                ),
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
                    "Use when the user asks about documentation, API reference pages, SDK docs, "
                    "guides, or README-like URLs. Prefer this over summarize_webpage when the URL "
                    "or nearby words suggest docs or reference material."
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
                related_tools=("summarize_webpage",),
            ),
            ToolSpec(
                name="get_local_area",
                description=(
                    "Get the user's current local area details, including location label, "
                    "timezone, and coordinates, without exposing the raw IP address."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=_handle_get_local_area,
                usage_guidance=(
                    "Use when the user asks about their own current area, timezone, or coordinates. "
                    "If the user is more likely referring to a previously mentioned non-local city "
                    "or country, prefer lookup_place or lookup_time instead."
                ),
                examples=(
                    ('用户说: "我这里是哪个时区？"', "{}"),
                    ('用户说: "我这里的经纬度是多少"', "{}"),
                    ('用户说: "Where am I roughly located right now?"', "{}"),
                ),
                should_offer=lambda user_text, route_mode: _matches_local_area_intent(user_text),
                related_tools=("lookup_place", "lookup_time"),
                related_tools_when=lambda user_text, route_mode: _has_contextual_place_reference(
                    user_text
                ),
                execution_meta=ToolExecutionMeta(timeout_seconds=5.0),
            ),
            ToolSpec(
                name="get_local_time",
                description=(
                    "Get the user's current local time and timezone without exposing the raw "
                    "IP address."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=_handle_get_local_time,
                usage_guidance=(
                    "Use when the user asks for the time in their own current area. If the user "
                    "names another place, or if words like 当地/here likely refer to a previously "
                    "mentioned non-local place, use lookup_time instead."
                ),
                examples=(
                    ('用户说: "我这里几点了？"', "{}"),
                    ('用户说: "我这边现在时间"', "{}"),
                    ('用户说: "what time is it here right now?"', "{}"),
                ),
                should_offer=lambda user_text, route_mode: _matches_local_time_intent(user_text),
                related_tools=("lookup_time",),
                related_tools_when=lambda user_text, route_mode: _has_contextual_place_reference(
                    user_text
                ),
                execution_meta=ToolExecutionMeta(timeout_seconds=5.0),
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
                    "Use for the user's own current-area weather. If the user names another place, "
                    "or if words like 当地/here likely refer to a previously mentioned non-local "
                    "place, prefer get_weather_by_location instead. Keep the user's time phrase raw."
                ),
                examples=(
                    ('用户说: "我这里明天的天气"', '{"time_expression":"明天"}'),
                    ('用户说: "this weekend weather"', '{"time_expression":"this weekend"}'),
                ),
                should_offer=lambda user_text, route_mode: _matches_local_weather_intent(user_text),
                related_tools=("get_weather_by_location",),
                related_tools_when=lambda user_text, route_mode: _has_contextual_place_reference(
                    user_text
                ),
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
                    "Use for weather in a named place, or when the current turn likely refers back "
                    "to a previously mentioned non-local place. Put only the place in location, and "
                    "put the time phrase in time_expression instead of mixing them together."
                ),
                examples=(
                    ('用户说: "嘉定区明天的天气"', '{"location":"嘉定区","time_expression":"明天"}'),
                    (
                        '用户说: "Tokyo weather next Monday"',
                        '{"location":"Tokyo","time_expression":"next Monday"}',
                    ),
                ),
                should_offer=lambda user_text, route_mode: (
                    should_fetch_weather(user_text)
                    and _matches_weather_by_location_intent(user_text)
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
                parameters_schema=_location_or_query_tool_schema(
                    query_description=(
                        "Original user request asking where a place is, or asking for coordinates."
                    ),
                    location_description=(
                        "Concrete place name resolved from the current turn or prior context, "
                        "such as Singapore, Tokyo, or 嘉定区."
                    ),
                ),
                handler=lambda arguments: _handle_lookup_place(
                    arguments,
                    geocoding_service,
                ),
                usage_guidance=(
                    "Use for map, coordinate, latitude, longitude, or address-style questions about "
                    "a named or previously mentioned non-local place. If the current user text says "
                    "那里/那个城市/that city/there, resolve that reference from conversation "
                    "context and pass the concrete place in location."
                ),
                examples=(
                    ('用户说: "Hangzhou coordinates"', '{"query":"Hangzhou coordinates"}'),
                    (
                        '用户说: "那个城市的经纬度"',
                        '{"query":"那个城市的经纬度","location":"Singapore"}',
                    ),
                ),
                should_offer=lambda user_text, route_mode: (
                    _contains_any(user_text, MAP_KEYWORDS)
                    and (
                        _extract_map_place(user_text) is not None
                        or _has_contextual_place_reference(user_text)
                    )
                    and not _has_explicit_local_reference(user_text)
                ),
            ),
            ToolSpec(
                name="lookup_time",
                description=(
                    "Get live local time or timezone information from a raw location query."
                ),
                parameters_schema=_location_or_query_tool_schema(
                    query_description=(
                        "Original user request about local time, timezone, time difference, or holidays."
                    ),
                    location_description=(
                        "Concrete place name resolved from the current turn or prior context, "
                        "such as Singapore, Tokyo, or 嘉定区."
                    ),
                ),
                handler=lambda arguments: _handle_lookup_time(
                    arguments,
                    time_service,
                ),
                usage_guidance=(
                    "Use for local time, timezone, time difference, and holiday calendar questions "
                    "about a named place, or when context suggests words like 当地/here refer to a "
                    "previously mentioned non-local place. If the user says 那个城市/there/that city, "
                    "resolve it from conversation context and pass the concrete place in location."
                ),
                examples=(
                    ('用户说: "current time in Singapore"', '{"query":"current time in Singapore"}'),
                    ('用户说: "2026 Japan holidays"', '{"query":"2026 Japan holidays"}'),
                    (
                        '用户说: "那个城市的时区"',
                        '{"query":"那个城市的时区","location":"Singapore"}',
                    ),
                ),
                should_offer=lambda user_text, route_mode: (
                    (
                        _contains_any(user_text, ("holiday", "holidays", "节假日", "假期"))
                        or _extract_time_place(user_text) is not None
                        or _has_contextual_place_reference(user_text)
                    )
                    and _contains_any(user_text, TIME_KEYWORDS)
                    and not (
                        _has_explicit_local_reference(user_text)
                        and not _has_contextual_place_reference(user_text)
                    )
                ),
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
                    "Use for headlines and current-news requests. Prefer this for concise news "
                    "snapshots on a topic; use web_search for broader research or when the user "
                    "explicitly asks to search."
                ),
                examples=(
                    ('用户说: "AI 新闻"', '{"query":"AI 新闻"}'),
                ),
                should_offer=lambda user_text, route_mode: _contains_any(user_text, NEWS_KEYWORDS),
                related_tools=("web_search",),
            ),
        ]

        if config is not None and config.hardware.cup.enabled:
            cup_settings = config.hardware.cup
            specs.append(
                ToolSpec(
                    name="control_cup",
                    description=(
                        "Control the configured CUP client API over HTTP. Use it to inspect "
                        "status, change motor speed, stop the motor, adjust LED mood cues, "
                        "or apply higher-level intensity scenes."
                    ),
                    parameters_schema={
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "One of: status, set_motor, stop, set_led, "
                                    "apply_scene, or nudge_intensity."
                                ),
                            },
                            "target_velocity": {
                                "type": "number",
                                "description": (
                                    "Target motor velocity for set_motor in the device's "
                                    "native units."
                                ),
                            },
                            "enabled": {
                                "type": "boolean",
                                "description": (
                                    "Whether the motor should be enabled for set_motor."
                                ),
                            },
                            "color": {
                                "type": "string",
                                "description": (
                                    "Optional LED color for set_led, such as warmwhite, "
                                    "red, 粉色, or #ff6600."
                                ),
                            },
                            "blink_hz": {
                                "type": "number",
                                "description": (
                                    "Optional LED blink frequency in Hz for set_led."
                                ),
                            },
                            "blink_mode": {
                                "type": "integer",
                                "description": (
                                    "Optional LED blink mode for set_led: 0 steady, "
                                    "1 slow square, 2 fast square, 3 breathe."
                                ),
                            },
                            "scene": {
                                "type": "string",
                                "description": (
                                    "Scene name for apply_scene, such as gentle, steady, "
                                    "intense, or cooldown."
                                ),
                            },
                            "direction": {
                                "type": "string",
                                "description": (
                                    "Direction for nudge_intensity: up to intensify or "
                                    "down to soften."
                                ),
                            },
                            "step": {
                                "type": "number",
                                "description": (
                                    "Optional intensity delta for nudge_intensity."
                                ),
                            },
                        },
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                    handler=lambda arguments: _handle_control_cup(
                        arguments,
                        base_url=cup_settings.base_url,
                        api_token=cup_settings.resolved_api_token(),
                        timeout_seconds=cup_settings.timeout_seconds,
                    ),
                    usage_guidance=(
                        "Use only when the user clearly asks to inspect or change the CUP "
                        "hardware, or when there is obvious ongoing CUP-control context. "
                        "Prefer nudge_intensity for requests like '再刺激一点' or '温柔一点', "
                        "apply_scene for broad mode changes, and stop for any pause or "
                        "safety-oriented request."
                    ),
                    examples=(
                        ('用户说: "看看飞机杯现在的状态"', '{"action":"status"}'),
                        (
                            '用户说: "把 CUP 转快一点"',
                            '{"action":"nudge_intensity","direction":"up"}',
                        ),
                        (
                            '用户说: "切到温柔一点的模式"',
                            '{"action":"apply_scene","scene":"gentle"}',
                        ),
                    ),
                    should_offer=lambda user_text, route_mode: _matches_cup_control_intent(
                        user_text
                    ),
                    execution_meta=ToolExecutionMeta(
                        parallel_safe=False,
                        side_effectful=True,
                        retryable=False,
                        timeout_seconds=float(cup_settings.timeout_seconds),
                    ),
                )
            )

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

        return cls(specs=tuple(specs), search_service=search_service)

    def openai_tools(self) -> list[dict[str, object]]:
        return [
            spec.as_openai_tool()
            for spec in self.specs
            if spec.execution_meta.visible_to_model
        ]

    def openai_tools_for_request(
        self,
        user_text: str,
        *,
        route_mode: str,
    ) -> list[dict[str, object]]:
        return [
            spec.as_openai_tool()
            for spec in self._visible_specs_for_request(
                user_text,
                route_mode=route_mode,
            )
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
            "- Use conversation context to resolve references like '当地', '这里', 'here', 'that city', or a previously shared URL/topic. If two related tools are available, choose the one whose description best matches the user's intent and referenced entity.",
            "- Do not invent arguments that the user did not imply. If a location, URL, or code is missing, ask naturally instead of guessing.",
            "- Only use side-effectful tools when the user clearly asked you to inspect or change the external device or system.",
            "- After receiving tool results, answer naturally and briefly.",
        ]
        for spec in visible_specs:
            lines.append(f"- tool: {spec.name}")
            lines.append(f"  description: {spec.description}")
            if spec.execution_meta.side_effectful:
                lines.append("  requires_clear_user_intent: yes")
            if spec.usage_guidance:
                lines.append(f"  use_when: {spec.usage_guidance}")
            for argument_line in _tool_argument_lines(spec):
                lines.append(f"  {argument_line}")
            for user_example, argument_example in spec.examples:
                lines.append(f"  example: {user_example} -> {argument_example}")
        return "\n".join(lines)

    def should_offer_tools(self, user_text: str, *, route_mode: str) -> bool:
        return bool(
            self._visible_specs_for_request(user_text, route_mode=route_mode)
        )

    def default_tool_choice(self, *, user_text: str, route_mode: str) -> str | None:
        if route_mode == "search":
            return "required"
        for spec in self.specs:
            if not spec.execution_meta.visible_to_model or not spec.execution_meta.side_effectful:
                continue
            if spec.should_offer is not None and spec.should_offer(user_text, route_mode):
                return "required"
        return None

    def _visible_specs_for_request(
        self,
        user_text: str,
        *,
        route_mode: str,
    ) -> tuple[ToolSpec, ...]:
        visible_specs = tuple(
            spec for spec in self.specs if spec.execution_meta.visible_to_model
        )
        matched_specs = tuple(
            spec
            for spec in visible_specs
            if spec.should_offer is not None and spec.should_offer(user_text, route_mode)
        )
        if matched_specs:
            selected_names = {spec.name for spec in matched_specs}
            for spec in matched_specs:
                if (
                    spec.related_tools_when is None
                    or spec.related_tools_when(user_text, route_mode)
                ):
                    selected_names.update(spec.related_tools)
            return tuple(
                spec for spec in visible_specs if spec.name in selected_names
            )
        if not any(spec.should_offer is not None for spec in visible_specs):
            return visible_specs
        return ()

    def execute(self, tool_call: ToolInvocation) -> ToolExecutionResult:
        if tool_call.name == "$web_search":
            try:
                parsed_arguments = tool_call.arguments()
            except Exception as exc:
                raise ToolExecutionError(
                    "Tool '$web_search' received invalid JSON arguments."
                ) from exc
            return ToolExecutionResult(
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                content=json.dumps(parsed_arguments, ensure_ascii=False),
            )
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
    return search_service.search_query(query).to_tool_payload()


def _handle_get_public_ip(arguments: dict[str, object]) -> dict[str, object]:
    if arguments:
        raise ToolExecutionError("get_public_ip does not accept arguments.")

    public_ip = lookup_public_ip(_tool_fetch_json, _TOOL_TIMEOUT_SECONDS)
    return {"public_ip": public_ip}


def _handle_get_local_area(arguments: dict[str, object]) -> dict[str, object]:
    if arguments:
        raise ToolExecutionError("get_local_area does not accept arguments.")

    location = lookup_local_area(_tool_fetch_json, _TOOL_TIMEOUT_SECONDS)
    return {
        "location": location.label,
        "city": location.city,
        "region": location.region,
        "country": location.country,
        "latitude": location.latitude,
        "longitude": location.longitude,
        "timezone": location.timezone,
    }


def _handle_get_local_time(arguments: dict[str, object]) -> dict[str, object]:
    if arguments:
        raise ToolExecutionError("get_local_time does not accept arguments.")

    location = lookup_local_area(_tool_fetch_json, _TOOL_TIMEOUT_SECONDS)
    try:
        now = datetime.now(ZoneInfo(location.timezone))
    except Exception as exc:
        raise ToolExecutionError(
            f"get_local_time could not resolve timezone {location.timezone!r}."
        ) from exc
    return {
        "location": location.label,
        "timezone": location.timezone,
        "local_datetime": now.isoformat(),
        "local_date": now.date().isoformat(),
        "local_time": now.strftime("%H:%M:%S"),
    }


def _handle_lookup_place(
    arguments: dict[str, object],
    service: GeocodingContextService,
) -> dict[str, object]:
    query = _optional_string_argument(arguments, "query")
    location = _optional_string_argument(arguments, "location")
    if location is not None:
        normalized_location = _normalize_place_candidate(location)
        if normalized_location is None:
            raise ToolExecutionError(
                "lookup_place requires 'location' to identify a concrete place."
            )
        context = service.build_place_context(normalized_location)
        if context is None:
            raise ToolExecutionError(
                "lookup_place could not derive live context from the provided location."
            )
        payload: dict[str, object] = {
            "location": normalized_location,
            "context": context,
        }
        if query is not None:
            payload["query"] = query
        return payload
    if query is None:
        raise ToolExecutionError("lookup_place requires a non-empty 'query' string.")
    context = service.build_prompt_context(query, route_mode="chat")
    if context is None:
        raise ToolExecutionError(
            "lookup_place could not derive live context from the provided query."
        )
    return {
        "query": query,
        "context": context,
    }


def _handle_lookup_time(
    arguments: dict[str, object],
    service: TimeContextService,
) -> dict[str, object]:
    query = _optional_string_argument(arguments, "query")
    location = _optional_string_argument(arguments, "location")
    if location is not None:
        normalized_location = _normalize_place_candidate(location)
        if normalized_location is None:
            raise ToolExecutionError(
                "lookup_time requires 'location' to identify a concrete place."
            )
        context = service.build_time_context(normalized_location)
        if context is None:
            raise ToolExecutionError(
                "lookup_time could not derive live context from the provided location."
            )
        payload: dict[str, object] = {
            "location": normalized_location,
            "context": context,
        }
        if query is not None:
            payload["query"] = query
        return payload
    if query is None:
        raise ToolExecutionError("lookup_time requires a non-empty 'query' string.")
    context = service.build_prompt_context(query, route_mode="chat")
    if context is None:
        raise ToolExecutionError(
            "lookup_time could not derive live context from the provided query."
        )
    return {
        "query": query,
        "context": context,
    }


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


def _handle_control_cup(
    arguments: dict[str, object],
    *,
    base_url: str,
    api_token: str | None,
    timeout_seconds: int,
) -> dict[str, object]:
    request_trace: list[dict[str, object]] = []

    def cup_request(
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        url = _cup_api_url(base_url, path)
        trace_item: dict[str, object] = {
            "method": method.upper(),
            "url": url,
        }
        if payload is not None:
            trace_item["payload"] = payload
        request_trace.append(trace_item)
        try:
            return _tool_http_json_request(
                url,
                timeout_seconds,
                method=method,
                payload=payload,
                headers=_cup_api_headers(api_token),
            )
        except Exception as exc:
            raise ToolExecutionError(
                f"HTTP {method.upper()} {url} failed: {exc.__class__.__name__}: {exc}"
            ) from exc

    action = _resolve_cup_action(
        _require_string_argument(arguments, "action", tool_name="control_cup")
    )
    if action == "status":
        system_response = cup_request(_CUP_STATUS_ENDPOINT)
        motor_response = cup_request(_CUP_MOTOR_ENDPOINT)
        return {
            "ok": True,
            "device": base_url,
            "action": "status",
            "system": _unwrap_device_response(system_response),
            "motor": _unwrap_device_response(motor_response),
            "request_trace": request_trace,
        }

    if action == "set_motor":
        body: dict[str, object] = {}
        target_velocity = _optional_number_argument(arguments, "target_velocity")
        enabled = _optional_boolish_argument(arguments, "enabled")
        if target_velocity is not None:
            body["target_velocity"] = round(target_velocity, 3)
        if enabled is not None:
            body["enabled"] = enabled
        if not body:
            raise ToolExecutionError(
                "control_cup set_motor requires 'target_velocity', 'enabled', or both."
            )
        response = cup_request(_CUP_MOTOR_ENDPOINT, method="POST", payload=body)
        payload = _device_action_payload(
            base_url,
            action="set_motor",
            requested=body,
            response=response,
        )
        payload["request_trace"] = request_trace
        return payload

    if action == "stop":
        response = cup_request(_CUP_STOP_ENDPOINT, method="POST", payload={})
        payload = _device_action_payload(
            base_url,
            action="stop",
            requested={"target_velocity": 0.0, "enabled": False},
            response=response,
        )
        payload["stopped"] = True
        payload["request_trace"] = request_trace
        return payload

    if action == "set_led":
        body = _cup_led_body_from_arguments(arguments)
        response = cup_request(_CUP_LED_ENDPOINT, method="POST", payload=body)
        payload = _device_action_payload(
            base_url,
            action="set_led",
            requested=body,
            response=response,
        )
        payload["request_trace"] = request_trace
        return payload

    if action == "apply_scene":
        scene_name, scene_payload = _resolve_cup_scene(
            _require_string_argument(arguments, "scene", tool_name="control_cup")
        )
        led_body = dict(scene_payload["led"])
        motor_body = dict(scene_payload["motor"])
        led_response = cup_request(_CUP_LED_ENDPOINT, method="POST", payload=led_body)
        if not motor_body.get("enabled", True):
            motor_response = cup_request(_CUP_STOP_ENDPOINT, method="POST", payload={})
        else:
            motor_response = cup_request(_CUP_MOTOR_ENDPOINT, method="POST", payload=motor_body)
        return {
            "ok": True,
            "device": base_url,
            "action": "apply_scene",
            "scene": scene_name,
            "motor": motor_body,
            "led": led_body,
            "motor_result": _unwrap_device_response(motor_response),
            "led_result": _unwrap_device_response(led_response),
            "request_trace": request_trace,
        }

    direction = _resolve_cup_direction(
        _require_string_argument(arguments, "direction", tool_name="control_cup")
    )
    current_response = cup_request(_CUP_MOTOR_ENDPOINT)
    current_target = _cup_current_target_velocity(_unwrap_device_response(current_response))
    requested_step = _optional_number_argument(arguments, "step")
    step = abs(requested_step) if requested_step is not None else 15.0
    next_target = current_target + step if direction == "up" else current_target - step
    next_target = max(0.0, min(_CUP_MAX_TARGET_VELOCITY, next_target))
    led_body = _cup_led_profile_for_velocity(next_target)
    led_response = cup_request(_CUP_LED_ENDPOINT, method="POST", payload=led_body)
    if next_target <= 0.0:
        motor_response = cup_request(_CUP_STOP_ENDPOINT, method="POST", payload={})
        motor_body: dict[str, object] = {"target_velocity": 0.0, "enabled": False}
    else:
        motor_body = {
            "target_velocity": round(next_target, 3),
            "enabled": True,
        }
        motor_response = cup_request(_CUP_MOTOR_ENDPOINT, method="POST", payload=motor_body)
    return {
        "ok": True,
        "device": base_url,
        "action": "nudge_intensity",
        "direction": direction,
        "previous_target_velocity": round(current_target, 3),
        "target_velocity": round(next_target, 3),
        "motor": motor_body,
        "led": led_body,
        "motor_result": _unwrap_device_response(motor_response),
        "led_result": _unwrap_device_response(led_response),
        "request_trace": request_trace,
    }


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
    if not should_fetch_weather(user_text):
        return False
    place = _extract_city_weather_place(user_text)
    return place is None or _is_local_reference_phrase(place)


def _matches_weather_by_location_intent(user_text: str) -> bool:
    if not should_fetch_weather(user_text):
        return False
    place = _extract_city_weather_place(user_text)
    return place is not None and not _is_local_reference_phrase(place)


def _matches_local_area_intent(user_text: str) -> bool:
    lowered = user_text.casefold()
    if should_fetch_weather(user_text):
        return False
    explicit_phrases = (
        "where am i",
        "where i am",
        "where i'm",
        "my location",
        "my coordinates",
        "my coordinate",
        "my latitude",
        "my longitude",
        "my timezone",
        "local timezone",
        "当前位置",
        "我在哪",
        "我在哪里",
        "我这里是哪里",
        "我的位置",
        "我的坐标",
        "我的经纬度",
        "我的时区",
        "本地时区",
        "当地时区",
        "我这里的坐标",
        "我这里的经纬度",
        "我这里的时区",
    )
    if any(phrase in lowered for phrase in explicit_phrases):
        return True
    local_reference = any(keyword.casefold() in lowered for keyword in LOCAL_TIME_KEYWORDS)
    location_request = any(
        keyword in lowered
        for keyword in (
            "location",
            "coordinates",
            "coordinate",
            "latitude",
            "longitude",
            "timezone",
            "时区",
            "坐标",
            "经纬度",
            "位置",
            "在哪",
            "哪里",
        )
    )
    return local_reference and location_request


def _matches_local_time_intent(user_text: str) -> bool:
    lowered = user_text.casefold()
    if should_fetch_weather(user_text):
        return False
    if _contains_any(user_text, ("holiday", "holidays", "节假日", "假期")):
        return False
    extracted_place = _extract_time_place(user_text)
    if extracted_place is not None:
        return _is_local_reference_phrase(extracted_place)
    explicit_phrases = (
        "what time is it here",
        "what time is it where i am",
        "time here",
        "我这里几点",
        "我这里现在几点",
        "我这边几点",
        "我这边现在几点",
        "我这里现在时间",
        "我这边现在时间",
    )
    if any(phrase in lowered for phrase in explicit_phrases):
        return True
    if any(
        phrase in lowered
        for phrase in (
            "here",
            "where i am",
            "where i'm",
            "my location",
            "my area",
            "我这里",
            "我这边",
            "这里",
            "这边",
            "当前位置",
        )
    ):
        return any(
            keyword in lowered
            for keyword in (
                "current time",
                "what time",
                "local time",
                "time now",
                "几点",
                "时间",
            )
        )
    stripped = re.sub(r"\s+", " ", lowered).strip(" ?？!！.,，。")
    return stripped in {
        "what time is it",
        "current time",
        "local time",
        "time now",
        "现在几点",
        "几点了",
        "本地时间",
        "当地时间",
        "当前时间",
        "现在时间",
    }


def _has_contextual_place_reference(user_text: str) -> bool:
    lowered = user_text.casefold()
    return any(
        phrase in lowered
        for phrase in (
            "当地",
            "那里",
            "那个城市",
            "那个地方",
            "there",
            "that city",
            "that place",
        )
    )


def _has_explicit_local_reference(user_text: str) -> bool:
    lowered = user_text.casefold()
    return any(
        phrase in lowered
        for phrase in (
            "我这里",
            "我这边",
            "where i am",
            "where i'm",
            "my location",
            "my area",
            "当前位置",
        )
    )


def _is_local_reference_phrase(value: str) -> bool:
    normalized = re.sub(r"\s+", " ", value.casefold()).strip(" ?？!！.,，。")
    return normalized in {
        "here",
        "where i am",
        "where i'm",
        "my location",
        "my area",
        "local",
        "local area",
        "local timezone",
        "current location",
        "current area",
        "我这里",
        "我这边",
        "这里",
        "这边",
        "当前位置",
        "本地",
        "当地",
    }


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


_CONTROL_VERBS = (
    "set",
    "switch",
    "change",
    "adjust",
    "tune",
    "make",
    "turn",
    "调",
    "调整",
    "切换",
    "换",
    "改",
    "设置",
    "设成",
    "调成",
    "改成",
    "变成",
    "弄成",
    "开",
    "关",
)


_CUP_DEVICE_KEYWORDS = (
    "cup",
    "飞机杯",
    "飞机杯硬件",
    "masturbator",
    "马达",
    "电机",
    "motor",
    "转速",
    "震动",
    "振动",
)
_CUP_CONTROL_KEYWORDS = (
    "status",
    "状态",
    "启动",
    "开始",
    "停止",
    "停下",
    "停一下",
    "暂停",
    "恢复",
    "快一点",
    "慢一点",
    "强一点",
    "弱一点",
    "刺激一点",
    "温柔一点",
    "柔和一点",
    "加速",
    "减速",
    "intensity",
    "speed",
    "faster",
    "slower",
    "motor",
    "led",
    "灯光",
    "颜色",
)
_CUP_SCENE_KEYWORDS = (
    "gentle",
    "steady",
    "intense",
    "cooldown",
    "轻柔",
    "温柔",
    "稳一点",
    "刺激",
    "猛烈",
    "缓一缓",
    "冷静一下",
)


def _matches_cup_control_intent(user_text: str) -> bool:
    lowered = user_text.casefold()
    return any(keyword in lowered for keyword in _CUP_DEVICE_KEYWORDS) and any(
        keyword in lowered
        for keyword in (
            *_CUP_CONTROL_KEYWORDS,
            *_CUP_SCENE_KEYWORDS,
            *_CONTROL_VERBS,
        )
    )


def _matches_cup_scene_followup(user_text: str) -> bool:
    lowered = user_text.casefold()
    return any(
        keyword in lowered
        for keyword in (
            *_CUP_CONTROL_KEYWORDS,
            *_CUP_SCENE_KEYWORDS,
        )
    )


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


def _location_or_query_tool_schema(
    *,
    query_description: str,
    location_description: str,
) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": query_description,
            },
            "location": {
                "type": "string",
                "description": location_description,
            },
        },
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


def _optional_number_argument(arguments: dict[str, object], key: str) -> float | None:
    value = arguments.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolExecutionError(f"'{key}' must be a number when provided.")
    return float(value)


def _optional_integer_argument(arguments: dict[str, object], key: str) -> int | None:
    value = arguments.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolExecutionError(f"'{key}' must be an integer when provided.")
    return value


def _optional_boolish_argument(arguments: dict[str, object], key: str) -> bool | None:
    value = arguments.get(key)
    if value is None:
        return None
    coerced = _coerce_boolish(value)
    if coerced is None:
        raise ToolExecutionError(f"'{key}' must be a boolean when provided.")
    return coerced


_COLOR_ALIASES: dict[str, tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "红": (255, 0, 0),
    "红色": (255, 0, 0),
    "green": (0, 255, 0),
    "lime": (0, 255, 0),
    "绿": (0, 255, 0),
    "绿色": (0, 255, 0),
    "blue": (0, 0, 255),
    "蓝": (0, 0, 255),
    "蓝色": (0, 0, 255),
    "yellow": (255, 255, 0),
    "黄": (255, 255, 0),
    "黄色": (255, 255, 0),
    "orange": (255, 165, 0),
    "橙": (255, 165, 0),
    "橙色": (255, 165, 0),
    "purple": (128, 0, 128),
    "violet": (128, 0, 128),
    "紫": (128, 0, 128),
    "紫色": (128, 0, 128),
    "pink": (255, 105, 180),
    "粉": (255, 105, 180),
    "粉色": (255, 105, 180),
    "cyan": (0, 255, 255),
    "青": (0, 255, 255),
    "青色": (0, 255, 255),
    "white": (255, 255, 255),
    "白": (255, 255, 255),
    "白色": (255, 255, 255),
    "warmwhite": (255, 180, 96),
    "暖白": (255, 180, 96),
    "暖白色": (255, 180, 96),
    "black": (0, 0, 0),
    "off": (0, 0, 0),
    "关闭": (0, 0, 0),
    "关灯": (0, 0, 0),
    "熄灭": (0, 0, 0),
    "黑": (0, 0, 0),
    "黑色": (0, 0, 0),
}

_CUP_API_ROOT_PATH = "/api"
_CUP_STATUS_ENDPOINT = "/status"
_CUP_MOTOR_ENDPOINT = "/motor"
_CUP_STOP_ENDPOINT = "/motor/stop"
_CUP_LED_ENDPOINT = "/led"
_CUP_MAX_TARGET_VELOCITY = 140.0
_CUP_SCENES: dict[str, dict[str, object]] = {
    "gentle": {
        "aliases": ("gentle", "soft", "轻柔", "温柔", "柔和"),
        "motor": {"target_velocity": 40.0, "enabled": True},
        "led": {"color": "#ffb36b", "blink_hz": 0.45, "blink_mode": 3},
    },
    "steady": {
        "aliases": ("steady", "normal", "稳一点", "平稳", "常规"),
        "motor": {"target_velocity": 70.0, "enabled": True},
        "led": {"color": "#40c4ff", "blink_hz": 0.9, "blink_mode": 1},
    },
    "intense": {
        "aliases": ("intense", "strong", "刺激", "猛烈", "更猛"),
        "motor": {"target_velocity": 110.0, "enabled": True},
        "led": {"color": "#ff3b30", "blink_hz": 2.0, "blink_mode": 2},
    },
    "cooldown": {
        "aliases": ("cooldown", "calm", "缓一缓", "冷静一下", "放松"),
        "motor": {"target_velocity": 20.0, "enabled": True},
        "led": {"color": "#7fd8ff", "blink_hz": 0.35, "blink_mode": 3},
    },
}


def _resolve_cup_action(action: str) -> str:
    key = _normalize_control_token(action)
    if key in {"status", "getstatus", "readstatus", "querystatus", "状态"}:
        return "status"
    if key in {"setmotor", "motor", "speed", "velocity", "setspeed", "setvelocity"}:
        return "set_motor"
    if key in {"stop", "pause", "停止", "停下", "停一下", "暂停"}:
        return "stop"
    if key in {"setled", "led", "light", "灯光", "颜色"}:
        return "set_led"
    if key in {"applyscene", "scene", "mode", "preset", "场景", "模式"}:
        return "apply_scene"
    if key in {"nudgeintensity", "intensity", "adjust", "tune", "调整强度", "强弱"}:
        return "nudge_intensity"
    raise ToolExecutionError(
        "control_cup action must be one of status, set_motor, stop, set_led, "
        "apply_scene, or nudge_intensity."
    )


def _resolve_cup_scene(scene: str) -> tuple[str, dict[str, dict[str, object]]]:
    key = _normalize_control_token(scene)
    for scene_name, payload in _CUP_SCENES.items():
        aliases = payload.get("aliases", ())
        if not isinstance(aliases, tuple):
            continue
        if any(_normalize_control_token(alias) == key for alias in aliases):
            return (
                scene_name,
                {
                    "motor": dict(payload["motor"]),
                    "led": dict(payload["led"]),
                },
            )
    available = ", ".join(sorted(_CUP_SCENES.keys()))
    raise ToolExecutionError(
        f"control_cup apply_scene scene must be one of: {available}."
    )


def _resolve_cup_direction(direction: str) -> str:
    key = _normalize_control_token(direction)
    if key in {"up", "increase", "more", "higher", "faster", "stronger", "更强", "更快", "增加"}:
        return "up"
    if key in {"down", "decrease", "less", "lower", "slower", "softer", "更慢", "减弱", "降低"}:
        return "down"
    raise ToolExecutionError(
        "control_cup nudge_intensity direction must be 'up' or 'down'."
    )


def _parse_led_color(color: str) -> tuple[int, int, int] | None:
    normalized = color.strip()
    if not normalized:
        return None
    hex_match = re.fullmatch(r"#?([0-9a-fA-F]{6})", normalized)
    if hex_match:
        value = hex_match.group(1)
        return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    rgb_match = re.fullmatch(
        r"rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)",
        normalized,
        re.IGNORECASE,
    )
    if rgb_match:
        red, green, blue = (int(part) for part in rgb_match.groups())
        if all(0 <= value <= 255 for value in (red, green, blue)):
            return red, green, blue
        return None
    return _COLOR_ALIASES.get(_normalize_control_token(normalized))


def _normalize_control_token(value: str) -> str:
    return (
        value.strip()
        .casefold()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )


def _coerce_boolish(value: object, *, fallback: bool | None = None) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        key = _normalize_control_token(value)
        if key in {"1", "true", "on", "enable", "enabled", "yes", "start"}:
            return True
        if key in {"0", "false", "off", "disable", "disabled", "no", "stop"}:
            return False
    return fallback


def _rgb_to_hex(red: int, green: int, blue: int) -> str:
    return f"#{red:02x}{green:02x}{blue:02x}"


def _cup_led_body_from_arguments(arguments: dict[str, object]) -> dict[str, object]:
    body: dict[str, object] = {}
    color = _optional_string_argument(arguments, "color")
    if color is not None:
        parsed = _parse_led_color(color)
        if parsed is None:
            raise ToolExecutionError(
                "control_cup set_led could not parse the requested color."
            )
        body["color"] = _rgb_to_hex(*parsed)
    blink_hz = _optional_number_argument(arguments, "blink_hz")
    if blink_hz is not None:
        if blink_hz < 0:
            raise ToolExecutionError("control_cup set_led requires 'blink_hz' >= 0.")
        body["blink_hz"] = round(blink_hz, 3)
    blink_mode = _optional_integer_argument(arguments, "blink_mode")
    if blink_mode is not None:
        if blink_mode not in {0, 1, 2, 3}:
            raise ToolExecutionError("control_cup set_led requires blink_mode between 0 and 3.")
        body["blink_mode"] = blink_mode
    if not body:
        raise ToolExecutionError(
            "control_cup set_led requires at least one of 'color', 'blink_hz', or 'blink_mode'."
        )
    return body


def _cup_led_profile_for_velocity(target_velocity: float) -> dict[str, object]:
    if target_velocity <= 0:
        return {"color": "#7fd8ff", "blink_hz": 0.25, "blink_mode": 3}
    if target_velocity < 50:
        return {"color": "#ffb36b", "blink_hz": 0.45, "blink_mode": 3}
    if target_velocity < 90:
        return {"color": "#40c4ff", "blink_hz": 0.9, "blink_mode": 1}
    return {"color": "#ff3b30", "blink_hz": 2.0, "blink_mode": 2}


def _cup_current_target_velocity(payload: object) -> float:
    if not isinstance(payload, dict):
        return 0.0
    target = _coerce_number(payload.get("target"))
    if target is not None:
        return target
    velocity = _coerce_number(payload.get("vel"))
    return velocity or 0.0


def _cup_api_base_url(base_url: str) -> str:
    normalized_base = base_url.strip().rstrip("/") or CUP_DEFAULT_BASE_URL
    if normalized_base.endswith(_CUP_API_ROOT_PATH):
        return normalized_base
    return f"{normalized_base}{_CUP_API_ROOT_PATH}"


def _cup_api_url(base_url: str, endpoint: str) -> str:
    normalized_endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    return f"{_cup_api_base_url(base_url)}{normalized_endpoint}"


def _cup_api_headers(api_token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
    }
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    return headers


def _unwrap_device_response(response: dict[str, object]) -> dict[str, object]:
    if not isinstance(response, dict):
        raise ToolExecutionError("Device API returned a non-object payload.")
    ok = response.get("ok")
    if ok is False:
        error = response.get("error")
        raise ToolExecutionError(
            f"Device API error: {error}" if isinstance(error, str) and error else "Device API error."
        )
    data = response.get("data")
    if isinstance(data, dict):
        return data
    return response


def _device_action_payload(
    base_url: str,
    *,
    action: str,
    requested: dict[str, object],
    response: dict[str, object],
) -> dict[str, object]:
    return {
        "ok": True,
        "device": base_url,
        "action": action,
        "requested": requested,
        "result": _unwrap_device_response(response),
    }


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


def _coerce_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _tool_http_json_request(
    url: str,
    timeout_seconds: int,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, object]:
    merged_headers = {
        "User-Agent": "Whesper/1.0",
        "Accept": "application/json",
    }
    if payload is not None:
        merged_headers["Content-Type"] = "application/json"
    if headers:
        merged_headers.update(headers)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urlrequest.Request(
        url,
        data=body,
        headers=merged_headers,
        method=method.upper(),
    )
    with urlrequest.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


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
