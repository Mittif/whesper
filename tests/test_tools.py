from __future__ import annotations

import json
import unittest
from pathlib import Path

from whesper.client import ToolCall
from whesper.config import (
    AppConfig,
    AppSettings,
    CupHardwareSettings,
    HardwareSettings,
    LedHardwareSettings,
    LiveContextSettings,
    ModelConfig,
    PersonaConfig,
    ProviderConfig,
    SchedulerConfig,
    ShellSandboxSettings,
)
from whesper.tools import (
    ToolRegistry,
    ToolSpec,
    _handle_control_cup,
    _handle_control_led,
    _handle_get_local_area,
    _handle_get_local_time,
    _handle_get_local_weather,
    _handle_get_public_ip,
    _handle_get_weather_by_location,
    _matches_cup_control_intent,
    _matches_cup_scene_followup,
    _matches_led_control_intent,
    _matches_led_scene_followup,
)
from whesper.agent_types import ToolExecutionMeta


class ToolRegistryTests(unittest.TestCase):
    def test_execute_runs_registered_tool(self) -> None:
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="web_search",
                    description="Search the web",
                    parameters_schema={"type": "object"},
                    handler=lambda arguments: {"ok": True, "query": arguments["query"]},
                ),
            )
        )

        result = registry.execute(
            ToolCall(
                tool_call_id="call_1",
                name="web_search",
                arguments_json='{"query":"hello"}',
            )
        )

        self.assertEqual(result.tool_call_id, "call_1")
        self.assertEqual(result.name, "web_search")
        self.assertIn('"query": "hello"', result.content)

    def test_default_registry_exposes_high_level_live_tools(self) -> None:
        registry = ToolRegistry.default()

        tool_names = {spec.name for spec in registry.specs}

        self.assertIn("web_search", tool_names)
        self.assertIn("summarize_webpage", tool_names)
        self.assertIn("read_tech_docs", tool_names)
        self.assertIn("get_local_area", tool_names)
        self.assertIn("get_local_time", tool_names)
        self.assertIn("get_local_weather", tool_names)
        self.assertIn("get_weather_by_location", tool_names)
        self.assertIn("lookup_exchange_rate", tool_names)
        self.assertIn("lookup_market_price", tool_names)
        self.assertIn("lookup_place", tool_names)
        self.assertIn("lookup_time", tool_names)
        self.assertIn("get_news", tool_names)
        self.assertNotIn("get_public_ip", tool_names)
        self.assertNotIn("get_ip_location", tool_names)

    def test_default_registry_does_not_expose_raw_ip_tools_to_model(self) -> None:
        registry = ToolRegistry.default()

        openai_tool_names = {
            tool["function"]["name"]
            for tool in registry.openai_tools()
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        }

        self.assertNotIn("get_public_ip", openai_tool_names)
        self.assertNotIn("get_ip_location", openai_tool_names)

    def test_default_registry_registers_led_tool_when_hardware_enabled(self) -> None:
        config = AppConfig(
            app=AppSettings(),
            persona=PersonaConfig(),
            scheduler=SchedulerConfig(chat_model="local_chat"),
            live_context=LiveContextSettings(),
            shell_sandbox=ShellSandboxSettings(),
            providers={
                "local": ProviderConfig(
                    name="local",
                    kind="ollama_native",
                    base_url="http://localhost:11434",
                )
            },
            models={
                "local_chat": ModelConfig(
                    name="local_chat",
                    provider="local",
                    model="qwen",
                )
            },
            source_path=Path("whesper.toml"),
            hardware=HardwareSettings(
                led=LedHardwareSettings(
                    enabled=True,
                    base_url="http://led.local",
                    timeout_seconds=5,
                )
            ),
        )

        registry = ToolRegistry.default(config)

        tool_names = {spec.name for spec in registry.specs}
        self.assertIn("control_led", tool_names)

    def test_led_control_intent_matches_ambience_request(self) -> None:
        self.assertTrue(_matches_led_control_intent("帮我调整成浪漫一点的氛围灯"))

    def test_led_scene_followup_matches_romantic_mode(self) -> None:
        self.assertTrue(_matches_led_scene_followup("浪漫模式"))

    def test_default_registry_registers_cup_tool_when_hardware_enabled(self) -> None:
        config = AppConfig(
            app=AppSettings(),
            persona=PersonaConfig(),
            scheduler=SchedulerConfig(chat_model="local_chat"),
            live_context=LiveContextSettings(),
            shell_sandbox=ShellSandboxSettings(),
            providers={
                "local": ProviderConfig(
                    name="local",
                    kind="ollama_native",
                    base_url="http://localhost:11434",
                )
            },
            models={
                "local_chat": ModelConfig(
                    name="local_chat",
                    provider="local",
                    model="qwen",
                )
            },
            source_path=Path("whesper.toml"),
            hardware=HardwareSettings(
                cup=CupHardwareSettings(
                    enabled=True,
                    base_url="http://localhost:3001",
                    timeout_seconds=5,
                )
            ),
        )

        registry = ToolRegistry.default(config)

        tool_names = {spec.name for spec in registry.specs}
        self.assertIn("control_cup", tool_names)

    def test_cup_control_intent_matches_explicit_request(self) -> None:
        self.assertTrue(_matches_cup_control_intent("把飞机杯转快一点"))

    def test_cup_scene_followup_matches_intensity_request(self) -> None:
        self.assertTrue(_matches_cup_scene_followup("再刺激一点"))

    def test_default_tool_choice_requires_side_effectful_led_request(self) -> None:
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="control_led",
                    description="Control the LED",
                    parameters_schema={"type": "object"},
                    handler=lambda arguments: {"ok": True},
                    should_offer=lambda user_text, route_mode: _matches_led_control_intent(
                        user_text
                    ),
                    execution_meta=ToolExecutionMeta(side_effectful=True),
                ),
            )
        )

        self.assertEqual(
            registry.default_tool_choice(
                user_text="帮我调整成浪漫一点的氛围灯",
                route_mode="manual",
            ),
            "required",
        )

    def test_openai_tools_for_request_returns_only_matching_visible_tools(self) -> None:
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="control_led",
                    description="Control the LED",
                    parameters_schema={"type": "object"},
                    handler=lambda arguments: {"ok": True},
                    should_offer=lambda user_text, route_mode: _matches_led_control_intent(
                        user_text
                    ),
                    execution_meta=ToolExecutionMeta(side_effectful=True),
                ),
                ToolSpec(
                    name="web_search",
                    description="Search the web",
                    parameters_schema={"type": "object"},
                    handler=lambda arguments: {"ok": True},
                    should_offer=lambda user_text, route_mode: "搜索" in user_text,
                ),
            )
        )

        tools = registry.openai_tools_for_request(
            "帮我调整成浪漫一点的氛围灯",
            route_mode="chat",
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in tools],
            ["control_led"],
        )

    def test_openai_tools_for_request_returns_no_tools_for_unmatched_request(self) -> None:
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="control_led",
                    description="Control the LED",
                    parameters_schema={"type": "object"},
                    handler=lambda arguments: {"ok": True},
                    should_offer=lambda user_text, route_mode: _matches_led_control_intent(
                        user_text
                    ),
                    execution_meta=ToolExecutionMeta(side_effectful=True),
                ),
            )
        )

        tools = registry.openai_tools_for_request(
            "今天天气怎么样",
            route_mode="chat",
        )

        self.assertEqual(tools, [])

    def test_handle_get_public_ip_returns_ip_payload(self) -> None:
        from whesper import tools as tools_module

        original_fetch = tools_module._tool_fetch_json
        try:
            tools_module._tool_fetch_json = lambda url, timeout: {"ip": "203.0.113.10"}
            payload = _handle_get_public_ip({})
        finally:
            tools_module._tool_fetch_json = original_fetch

        self.assertEqual(payload["public_ip"], "203.0.113.10")

    def test_handle_get_local_area_returns_location_payload_without_ip(self) -> None:
        from whesper import tools as tools_module

        original_fetch = tools_module._tool_fetch_json
        try:
            def fetch(url, timeout):
                if "api.ipify.org" in url:
                    return {"ip": "203.0.113.10"}
                if "ipwho.is" in url:
                    return {
                        "success": True,
                        "city": "Shanghai",
                        "region": "Shanghai",
                        "country": "China",
                        "latitude": 31.2304,
                        "longitude": 121.4737,
                        "timezone": {"id": "Asia/Shanghai"},
                    }
                raise AssertionError(f"unexpected URL: {url}")

            tools_module._tool_fetch_json = fetch
            payload = _handle_get_local_area({})
        finally:
            tools_module._tool_fetch_json = original_fetch

        self.assertEqual(payload["location"], "Shanghai, Shanghai, China")
        self.assertEqual(payload["timezone"], "Asia/Shanghai")
        self.assertEqual(payload["latitude"], 31.2304)
        self.assertEqual(payload["longitude"], 121.4737)
        self.assertNotIn("public_ip", payload)

    def test_handle_get_local_time_returns_time_payload_without_ip(self) -> None:
        from whesper import tools as tools_module

        original_fetch = tools_module._tool_fetch_json
        try:
            def fetch(url, timeout):
                if "api.ipify.org" in url:
                    return {"ip": "203.0.113.10"}
                if "ipwho.is" in url:
                    return {
                        "success": True,
                        "city": "Shanghai",
                        "region": "Shanghai",
                        "country": "China",
                        "latitude": 31.2304,
                        "longitude": 121.4737,
                        "timezone": {"id": "Asia/Shanghai"},
                    }
                raise AssertionError(f"unexpected URL: {url}")

            tools_module._tool_fetch_json = fetch
            payload = _handle_get_local_time({})
        finally:
            tools_module._tool_fetch_json = original_fetch

        self.assertEqual(payload["location"], "Shanghai, Shanghai, China")
        self.assertEqual(payload["timezone"], "Asia/Shanghai")
        self.assertRegex(payload["local_datetime"], r"^\d{4}-\d{2}-\d{2}T")
        self.assertRegex(payload["local_date"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertRegex(payload["local_time"], r"^\d{2}:\d{2}:\d{2}$")
        self.assertNotIn("public_ip", payload)

    def test_handle_get_local_weather_returns_structured_payload_without_ip(self) -> None:
        from whesper import tools as tools_module

        original_fetch = tools_module._tool_fetch_json
        try:
            def fetch(url, timeout):
                if "api.ipify.org" in url:
                    return {"ip": "203.0.113.10"}
                if "ipwho.is" in url:
                    return {
                        "success": True,
                        "city": "Shanghai",
                        "region": "Shanghai",
                        "country": "China",
                        "latitude": 31.2304,
                        "longitude": 121.4737,
                        "timezone": {"id": "Asia/Shanghai"},
                    }
                return {
                    "current_condition": [
                        {
                            "temp_C": "23.4",
                            "FeelsLikeC": "24.0",
                            "humidity": "61",
                            "windspeedKmph": "12.1",
                            "weatherDesc": [{"value": "Partly cloudy"}],
                        }
                    ],
                    "weather": [
                        {
                            "date": "2026-03-25",
                            "maxtempC": "27.2",
                            "mintempC": "19.8",
                            "hourly": [
                                {
                                    "time": "1200",
                                    "weatherDesc": [{"value": "Overcast"}],
                                    "chanceofrain": "30",
                                    "chanceofsnow": "0",
                                    "chanceofthunder": "0",
                                }
                            ],
                        },
                        {
                            "date": "2026-03-26",
                            "maxtempC": "25.1",
                            "mintempC": "18.3",
                            "hourly": [
                                {
                                    "time": "1200",
                                    "weatherDesc": [{"value": "Slight rain"}],
                                    "chanceofrain": "70",
                                    "chanceofsnow": "0",
                                    "chanceofthunder": "0",
                                }
                            ],
                        },
                    ],
                }

            tools_module._tool_fetch_json = fetch
            payload = _handle_get_local_weather({"time_expression": "明天"})
        finally:
            tools_module._tool_fetch_json = original_fetch

        self.assertEqual(payload["location"], "Shanghai, Shanghai, China")
        self.assertNotIn("public_ip", payload)
        self.assertEqual(payload["requested_period"], "tomorrow")
        self.assertEqual(payload["requested_forecast"]["date"], "2026-03-26")

    def test_openai_tools_for_request_matches_local_area_intent(self) -> None:
        registry = ToolRegistry.default()

        tools = registry.openai_tools_for_request(
            "我这里是哪个时区？",
            route_mode="chat",
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in tools],
            ["get_local_area", "lookup_place", "lookup_time"],
        )

    def test_openai_tools_for_request_matches_local_time_intent(self) -> None:
        registry = ToolRegistry.default()

        tools = registry.openai_tools_for_request(
            "我这里几点了？",
            route_mode="chat",
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in tools],
            ["get_local_time", "lookup_time"],
        )

    def test_openai_tools_for_request_prefers_lookup_time_for_remote_city_query(self) -> None:
        registry = ToolRegistry.default()

        tools = registry.openai_tools_for_request(
            "current time in Singapore",
            route_mode="chat",
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in tools],
            ["lookup_time"],
        )

    def test_openai_tools_for_request_prefers_lookup_time_for_place_prefixed_current_time(self) -> None:
        registry = ToolRegistry.default()

        tools = registry.openai_tools_for_request(
            "Tokyo current time",
            route_mode="chat",
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in tools],
            ["lookup_time"],
        )

    def test_openai_tools_for_request_prefers_lookup_time_for_chinese_remote_current_time(self) -> None:
        registry = ToolRegistry.default()

        tools = registry.openai_tools_for_request(
            "新加坡当前时间",
            route_mode="chat",
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in tools],
            ["lookup_time"],
        )

    def test_openai_tools_for_request_prefers_weather_by_location_for_named_place(self) -> None:
        registry = ToolRegistry.default()

        tools = registry.openai_tools_for_request(
            "Tokyo weather today",
            route_mode="chat",
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in tools],
            ["get_weather_by_location"],
        )

    def test_openai_tools_for_request_exposes_both_weather_tools_for_ambiguous_local_phrase(self) -> None:
        registry = ToolRegistry.default()

        tools = registry.openai_tools_for_request(
            "当地明天天气",
            route_mode="chat",
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in tools],
            ["get_local_weather", "get_weather_by_location"],
        )

    def test_openai_tools_for_request_exposes_related_weather_tool_for_explicit_local_reference(self) -> None:
        registry = ToolRegistry.default()

        tools = registry.openai_tools_for_request(
            "我这里明天的天气",
            route_mode="chat",
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in tools],
            ["get_local_weather", "get_weather_by_location"],
        )

    def test_openai_tools_for_request_keeps_general_web_summary_for_non_doc_url(self) -> None:
        registry = ToolRegistry.default()

        tools = registry.openai_tools_for_request(
            "帮我看看这个页面 https://example.com/blog/post",
            route_mode="chat",
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in tools],
            ["summarize_webpage"],
        )

    def test_openai_tools_for_request_exposes_docs_and_related_summary_tools(self) -> None:
        registry = ToolRegistry.default()

        tools = registry.openai_tools_for_request(
            "请看这个 API 文档 https://example.com/docs/sdk",
            route_mode="chat",
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in tools],
            ["summarize_webpage", "read_tech_docs"],
        )

    def test_openai_tools_for_request_exposes_news_and_related_search_tools(self) -> None:
        registry = ToolRegistry.default()

        tools = registry.openai_tools_for_request(
            "latest news about OpenAI",
            route_mode="chat",
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in tools],
            ["web_search", "get_news"],
        )

    def test_openai_tools_for_request_keeps_generic_search_without_news_tool(self) -> None:
        registry = ToolRegistry.default()

        tools = registry.openai_tools_for_request(
            "search OpenAI release notes",
            route_mode="chat",
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in tools],
            ["web_search"],
        )

    def test_handle_get_weather_by_location_returns_weather_payload(self) -> None:
        from whesper import tools as tools_module

        geocode_response = {
            "results": [
                {
                    "name": "Tokyo",
                    "admin1": "Tokyo",
                    "country": "Japan",
                    "latitude": 35.6895,
                    "longitude": 139.6917,
                    "timezone": "Asia/Tokyo",
                }
            ]
        }
        weather_response = {
            "current_condition": [
                {
                    "temp_C": "18.2",
                    "FeelsLikeC": "18.0",
                    "humidity": "70",
                    "windspeedKmph": "8.4",
                    "weatherDesc": [{"value": "Mainly clear"}],
                }
            ],
            "weather": [
                {
                    "date": "2026-03-25",
                    "maxtempC": "22.5",
                    "mintempC": "14.3",
                    "hourly": [
                        {
                            "time": "1200",
                            "weatherDesc": [{"value": "Partly cloudy"}],
                            "chanceofrain": "20",
                            "chanceofsnow": "0",
                            "chanceofthunder": "0",
                        }
                    ],
                }
            ],
        }
        original_fetch = tools_module._tool_fetch_json
        try:
            tools_module._tool_fetch_json = (
                lambda url, timeout: geocode_response
                if "geocoding-api.open-meteo.com" in url
                else weather_response
            )
            payload = _handle_get_weather_by_location(
                {"location": "Tokyo", "time_expression": "next Monday"}
            )
        finally:
            tools_module._tool_fetch_json = original_fetch

        self.assertEqual(payload["location"], "Tokyo, Tokyo, Japan")
        self.assertEqual(payload["current_condition"], "Mainly clear")
        self.assertEqual(payload["today"]["condition"], "Partly cloudy")

    def test_handle_get_weather_by_location_normalizes_search_prefixed_place(self) -> None:
        from whesper import tools as tools_module

        seen_urls: list[str] = []
        geocode_response = {
            "results": [
                {
                    "name": "Shanghai",
                    "admin1": "Shanghai",
                    "country": "China",
                    "latitude": 31.2304,
                    "longitude": 121.4737,
                    "timezone": "Asia/Shanghai",
                }
            ]
        }
        weather_response = {
            "current_condition": [
                {
                    "temp_C": "23.4",
                    "FeelsLikeC": "24.0",
                    "humidity": "61",
                    "windspeedKmph": "12.1",
                    "weatherDesc": [{"value": "Partly cloudy"}],
                }
            ],
            "weather": [
                {
                    "date": "2026-03-25",
                    "maxtempC": "27.2",
                    "mintempC": "19.8",
                    "hourly": [
                        {
                            "time": "1200",
                            "weatherDesc": [{"value": "Overcast"}],
                            "chanceofrain": "30",
                            "chanceofsnow": "0",
                            "chanceofthunder": "0",
                        }
                    ],
                }
            ],
        }
        original_fetch = tools_module._tool_fetch_json
        try:
            def fetch(url, timeout):
                seen_urls.append(url)
                if "geocoding-api.open-meteo.com" in url:
                    return geocode_response
                return weather_response

            tools_module._tool_fetch_json = fetch
            payload = _handle_get_weather_by_location({"location": "搜索上海"})
        finally:
            tools_module._tool_fetch_json = original_fetch

        self.assertEqual(payload["location"], "Shanghai, Shanghai, China")
        self.assertIn("name=%E4%B8%8A%E6%B5%B7", seen_urls[0])

    def test_registry_executes_weather_by_location_tool(self) -> None:
        registry = ToolRegistry(
            specs=(
                ToolSpec(
                    name="get_weather_by_location",
                    description="Weather lookup",
                    parameters_schema={"type": "object"},
                    handler=lambda arguments: {"location": arguments["location"], "ok": True},
                ),
            )
        )

        result = registry.execute(
            ToolCall(
                tool_call_id="call_weather_1",
                name="get_weather_by_location",
                arguments_json=json.dumps({"location": "Tokyo"}),
            )
        )

        self.assertEqual(result.name, "get_weather_by_location")
        self.assertIn('"location": "Tokyo"', result.content)

    def test_handle_control_led_sets_color_from_named_color(self) -> None:
        from whesper import tools as tools_module

        seen_urls: list[str] = []
        original_fetch = tools_module._tool_fetch_json
        try:
            def fetch(url, timeout):
                seen_urls.append(url)
                return {"r": 255, "g": 0, "b": 0}

            tools_module._tool_fetch_json = fetch
            payload = _handle_control_led(
                {"action": "set_color", "color": "红色"},
                base_url="http://led.local",
                timeout_seconds=5,
            )
        finally:
            tools_module._tool_fetch_json = original_fetch

        self.assertEqual(
            seen_urls,
            ["http://led.local/led?r=255&g=0&b=0"],
        )
        self.assertEqual(payload["action"], "set_color")
        self.assertEqual(payload["color_hex"], "#ff0000")
        self.assertEqual(payload["requested_color_hex"], "#ff0000")

    def test_handle_control_led_toggles_blink(self) -> None:
        from whesper import tools as tools_module

        seen_urls: list[str] = []
        original_fetch = tools_module._tool_fetch_json
        try:
            def fetch(url, timeout):
                seen_urls.append(url)
                return {"blink": 1}

            tools_module._tool_fetch_json = fetch
            payload = _handle_control_led(
                {"action": "set_blink", "blink": True},
                base_url="http://led.local",
                timeout_seconds=5,
            )
        finally:
            tools_module._tool_fetch_json = original_fetch

        self.assertEqual(seen_urls, ["http://led.local/blink?v=1"])
        self.assertEqual(payload["action"], "set_blink")
        self.assertTrue(payload["blink"])

    def test_handle_control_led_reads_status(self) -> None:
        from whesper import tools as tools_module

        seen_urls: list[str] = []
        original_fetch = tools_module._tool_fetch_json
        try:
            def fetch(url, timeout):
                seen_urls.append(url)
                return {"r": 16, "g": 16, "b": 16}

            tools_module._tool_fetch_json = fetch
            payload = _handle_control_led(
                {"action": "status"},
                base_url="http://led.local",
                timeout_seconds=5,
            )
        finally:
            tools_module._tool_fetch_json = original_fetch

        self.assertEqual(seen_urls, ["http://led.local/status"])
        self.assertEqual(payload["action"], "status")
        self.assertEqual(payload["color_hex"], "#101010")

    def test_handle_control_cup_sets_led(self) -> None:
        from whesper import tools as tools_module

        seen_requests: list[tuple[str, str, dict[str, object] | None, dict[str, str] | None]] = []
        original_request = tools_module._tool_http_json_request
        try:
            def request(url, timeout, *, method="GET", payload=None, headers=None):
                seen_requests.append((method, url, payload, headers))
                return {"ok": True, "data": {"color": "#ff69b4", "blink_mode": 3}}

            tools_module._tool_http_json_request = request
            payload = _handle_control_cup(
                {"action": "set_led", "color": "粉色", "blink_mode": 3},
                base_url="http://localhost:3001",
                api_token="token-1",
                timeout_seconds=5,
            )
        finally:
            tools_module._tool_http_json_request = original_request

        self.assertEqual(
            seen_requests,
            [
                (
                    "POST",
                    "http://localhost:3001/api/led",
                    {"color": "#ff69b4", "blink_mode": 3},
                    {
                        "Accept": "application/json",
                        "Authorization": "Bearer token-1",
                    },
                )
            ],
        )
        self.assertEqual(payload["action"], "set_led")
        self.assertEqual(payload["requested"]["color"], "#ff69b4")

    def test_handle_control_cup_nudges_intensity_up(self) -> None:
        from whesper import tools as tools_module

        seen_requests: list[tuple[str, str, dict[str, object] | None]] = []
        original_request = tools_module._tool_http_json_request
        try:
            def request(url, timeout, *, method="GET", payload=None, headers=None):
                seen_requests.append((method, url, payload))
                if method == "GET":
                    return {"ok": True, "data": {"target": 80.0, "vel": 79.5}}
                if url.endswith("/api/led"):
                    return {"ok": True, "data": {"color": "#ff3b30"}}
                return {"ok": True, "data": {"target": 95.0, "enabled": True}}

            tools_module._tool_http_json_request = request
            payload = _handle_control_cup(
                {"action": "nudge_intensity", "direction": "up"},
                base_url="http://localhost:3001",
                api_token=None,
                timeout_seconds=5,
            )
        finally:
            tools_module._tool_http_json_request = original_request

        self.assertEqual(seen_requests[0], ("GET", "http://localhost:3001/api/motor", None))
        self.assertEqual(seen_requests[1][0], "POST")
        self.assertEqual(seen_requests[1][1], "http://localhost:3001/api/led")
        self.assertEqual(seen_requests[2][0], "POST")
        self.assertEqual(seen_requests[2][1], "http://localhost:3001/api/motor")
        self.assertEqual(
            seen_requests[2][2],
            {"target_velocity": 95.0, "enabled": True},
        )
        self.assertEqual(payload["direction"], "up")
        self.assertEqual(payload["target_velocity"], 95.0)
        self.assertEqual(
            payload["request_trace"],
            [
                {"method": "GET", "url": "http://localhost:3001/api/motor"},
                {
                    "method": "POST",
                    "url": "http://localhost:3001/api/led",
                    "payload": {"color": "#ff3b30", "blink_hz": 2.0, "blink_mode": 2},
                },
                {
                    "method": "POST",
                    "url": "http://localhost:3001/api/motor",
                    "payload": {"target_velocity": 95.0, "enabled": True},
                },
            ],
        )

    def test_handle_control_cup_reads_status(self) -> None:
        from whesper import tools as tools_module

        seen_requests: list[str] = []
        original_request = tools_module._tool_http_json_request
        try:
            def request(url, timeout, *, method="GET", payload=None, headers=None):
                seen_requests.append(url)
                if url.endswith("/api/status"):
                    return {"ok": True, "data": {"ip": "192.168.1.42", "wifi_state": "connected"}}
                return {"ok": True, "data": {"target": 70.0, "vel": 68.5}}

            tools_module._tool_http_json_request = request
            payload = _handle_control_cup(
                {"action": "status"},
                base_url="http://localhost:3001",
                api_token=None,
                timeout_seconds=5,
            )
        finally:
            tools_module._tool_http_json_request = original_request

        self.assertEqual(
            seen_requests,
            [
                "http://localhost:3001/api/status",
                "http://localhost:3001/api/motor",
            ],
        )
        self.assertEqual(payload["action"], "status")
        self.assertEqual(payload["system"]["ip"], "192.168.1.42")
        self.assertEqual(payload["motor"]["target"], 70.0)

    def test_handle_control_cup_accepts_base_url_with_api_suffix(self) -> None:
        from whesper import tools as tools_module

        seen_requests: list[str] = []
        original_request = tools_module._tool_http_json_request
        try:
            def request(url, timeout, *, method="GET", payload=None, headers=None):
                seen_requests.append(url)
                return {"ok": True, "data": {"target": 70.0, "vel": 68.5}}

            tools_module._tool_http_json_request = request
            _handle_control_cup(
                {"action": "status"},
                base_url="http://localhost:3001/api",
                api_token=None,
                timeout_seconds=5,
            )
        finally:
            tools_module._tool_http_json_request = original_request

        self.assertEqual(
            seen_requests,
            [
                "http://localhost:3001/api/status",
                "http://localhost:3001/api/motor",
            ],
        )

    def test_handle_control_cup_error_mentions_http_method_and_url(self) -> None:
        from whesper import tools as tools_module

        original_request = tools_module._tool_http_json_request
        try:
            def request(url, timeout, *, method="GET", payload=None, headers=None):
                raise ConnectionRefusedError("refused")

            tools_module._tool_http_json_request = request
            with self.assertRaisesRegex(
                Exception,
                "HTTP GET http://localhost:3001/api/status failed",
            ):
                _handle_control_cup(
                    {"action": "status"},
                    base_url="http://localhost:3001",
                    api_token=None,
                    timeout_seconds=5,
                )
        finally:
            tools_module._tool_http_json_request = original_request

    def test_tool_prompt_renders_schema_guidance(self) -> None:
        registry = ToolRegistry.default()

        prompt = registry.tool_prompt()

        self.assertIn("Tool skills:", prompt)
        self.assertIn("tool: get_weather_by_location", prompt)
        self.assertIn("argument location", prompt)
        self.assertIn('"location":"嘉定区","time_expression":"明天"}', prompt)


if __name__ == "__main__":
    unittest.main()
