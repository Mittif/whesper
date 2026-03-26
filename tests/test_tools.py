from __future__ import annotations

import json
import unittest

from whesper.client import ToolCall
from whesper.tools import (
    ToolRegistry,
    ToolSpec,
    _handle_get_local_weather,
    _handle_get_public_ip,
    _handle_get_weather_by_location,
)


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
        self.assertIn("get_local_weather", tool_names)
        self.assertIn("get_weather_by_location", tool_names)
        self.assertIn("lookup_exchange_rate", tool_names)
        self.assertIn("lookup_market_price", tool_names)
        self.assertIn("lookup_place", tool_names)
        self.assertIn("lookup_time", tool_names)
        self.assertIn("get_news", tool_names)

    def test_handle_get_public_ip_returns_ip_payload(self) -> None:
        from whesper import tools as tools_module

        original_fetch = tools_module._tool_fetch_json
        try:
            tools_module._tool_fetch_json = lambda url, timeout: {"ip": "203.0.113.10"}
            payload = _handle_get_public_ip({})
        finally:
            tools_module._tool_fetch_json = original_fetch

        self.assertEqual(payload["public_ip"], "203.0.113.10")

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

    def test_tool_prompt_renders_schema_guidance(self) -> None:
        registry = ToolRegistry.default()

        prompt = registry.tool_prompt()

        self.assertIn("Tool skills:", prompt)
        self.assertIn("tool: get_weather_by_location", prompt)
        self.assertIn("argument location", prompt)
        self.assertIn('"location":"嘉定区","time_expression":"明天"}', prompt)


if __name__ == "__main__":
    unittest.main()
