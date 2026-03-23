from __future__ import annotations

import unittest
from urllib import parse

from whesper.config import LiveContextEndpointConfig
from whesper.live_data import (
    CityWeatherContextService,
    CustomApiContextService,
    ExchangeRateContextService,
    FetchedTextResponse,
    GeocodingContextService,
    LiveContextService,
    LocalWeatherContextService,
    MarketPriceContextService,
    NewsContextService,
    SearchContextService,
    TechDocsContextService,
    TimeContextService,
    TransportStatusContextService,
    UrlSummaryContextService,
    should_fetch_local_weather,
)


class LiveDataTests(unittest.TestCase):
    def test_should_fetch_local_weather_requires_weather_and_local_time_intent(self) -> None:
        self.assertTrue(should_fetch_local_weather("今天天气怎么样？"))
        self.assertTrue(should_fetch_local_weather("What's the weather like right now?"))
        self.assertFalse(should_fetch_local_weather("Explain how weather systems form."))
        self.assertFalse(should_fetch_local_weather("今天心情怎么样？"))

    def test_build_prompt_context_formats_live_weather_report(self) -> None:
        weather_url = (
            "https://api.open-meteo.com/v1/forecast?"
            + parse.urlencode(
                {
                    "latitude": 31.2304,
                    "longitude": 121.4737,
                    "current": ",".join(
                        (
                            "temperature_2m",
                            "relative_humidity_2m",
                            "apparent_temperature",
                            "weather_code",
                            "wind_speed_10m",
                        )
                    ),
                    "daily": ",".join(
                        (
                            "weather_code",
                            "temperature_2m_max",
                            "temperature_2m_min",
                            "precipitation_probability_max",
                        )
                    ),
                    "forecast_days": 1,
                    "timezone": "Asia/Shanghai",
                }
            )
        )

        responses = {
            "https://api.ipify.org?format=json": {"ip": "203.0.113.10"},
            "https://ipwho.is/203.0.113.10": {
                "success": True,
                "city": "Shanghai",
                "region": "Shanghai",
                "country": "China",
                "latitude": 31.2304,
                "longitude": 121.4737,
                "timezone": {"id": "Asia/Shanghai"},
            },
            weather_url: {
                "timezone": "Asia/Shanghai",
                "current": {
                    "temperature_2m": 23.4,
                    "relative_humidity_2m": 61,
                    "apparent_temperature": 24.0,
                    "weather_code": 2,
                    "wind_speed_10m": 12.1,
                },
                "daily": {
                    "weather_code": [3],
                    "temperature_2m_max": [27.2],
                    "temperature_2m_min": [19.8],
                    "precipitation_probability_max": [30],
                },
            },
        }

        service = LocalWeatherContextService(
            fetch_json=lambda url, timeout: responses[url],
        )

        context = service.build_prompt_context("今天天气怎么样？")

        assert context is not None
        self.assertIn("public_ip: 203.0.113.10", context)
        self.assertIn("location: Shanghai, Shanghai, China", context)
        self.assertIn("current_condition: Partly cloudy", context)
        self.assertIn("today_condition: Overcast", context)
        self.assertIn("today_high_c: 27.2", context)
        self.assertIn("today_low_c: 19.8", context)

    def test_build_prompt_context_returns_failure_instruction_when_lookup_breaks(self) -> None:
        service = LocalWeatherContextService(
            fetch_json=lambda url, timeout: (_ for _ in ()).throw(TimeoutError("network timeout")),
        )

        context = service.build_prompt_context("What's the weather like today?")

        assert context is not None
        self.assertIn("failed: TimeoutError: network timeout", context)
        self.assertIn("Do not invent the user's current weather", context)

    def test_exchange_rate_context_formats_live_rate(self) -> None:
        service = ExchangeRateContextService(
            fetch_json=lambda url, timeout: {
                "rates": {"CNY": 7.2345},
                "time_last_update_utc": "Sun, 23 Mar 2026 00:02:31 +0000",
            }
        )

        context = service.build_prompt_context("美元兑人民币汇率现在是多少？")

        assert context is not None
        self.assertIn("base_currency: USD", context)
        self.assertIn("quote_currency: CNY", context)
        self.assertIn("exchange_rate: 7.2345", context)

    def test_city_weather_context_formats_requested_location_weather(self) -> None:
        geocode_url = (
            "https://geocoding-api.open-meteo.com/v1/search?"
            + parse.urlencode({"name": "Tokyo", "count": 1, "language": "en", "format": "json"})
        )
        weather_url = (
            "https://api.open-meteo.com/v1/forecast?"
            + parse.urlencode(
                {
                    "latitude": 35.6895,
                    "longitude": 139.6917,
                    "current": ",".join(
                        (
                            "temperature_2m",
                            "relative_humidity_2m",
                            "apparent_temperature",
                            "weather_code",
                            "wind_speed_10m",
                        )
                    ),
                    "daily": ",".join(
                        (
                            "weather_code",
                            "temperature_2m_max",
                            "temperature_2m_min",
                            "precipitation_probability_max",
                        )
                    ),
                    "forecast_days": 1,
                    "timezone": "Asia/Tokyo",
                }
            )
        )
        responses = {
            geocode_url: {
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
            },
            weather_url: {
                "timezone": "Asia/Tokyo",
                "current": {
                    "temperature_2m": 18.2,
                    "relative_humidity_2m": 70,
                    "apparent_temperature": 18.0,
                    "weather_code": 1,
                    "wind_speed_10m": 8.4,
                },
                "daily": {
                    "weather_code": [2],
                    "temperature_2m_max": [22.5],
                    "temperature_2m_min": [14.3],
                    "precipitation_probability_max": [20],
                },
            },
        }
        service = CityWeatherContextService(fetch_json=lambda url, timeout: responses[url])

        context = service.build_prompt_context("Tokyo weather today")

        assert context is not None
        self.assertIn("location: Tokyo, Tokyo, Japan", context)
        self.assertIn("current_condition: Mainly clear", context)

    def test_market_price_context_formats_live_quote_data(self) -> None:
        expected_url = (
            "https://query1.finance.yahoo.com/v7/finance/quote?"
            + parse.urlencode({"symbols": "TSLA,BTC-USD"})
        )
        service = MarketPriceContextService(
            fetch_json=lambda url, timeout: {
                "quoteResponse": {
                    "result": [
                        {
                            "symbol": "TSLA",
                            "shortName": "Tesla, Inc.",
                            "regularMarketPrice": 241.23,
                            "regularMarketChangePercent": 1.8,
                            "currency": "USD",
                        },
                        {
                            "symbol": "BTC-USD",
                            "shortName": "Bitcoin USD",
                            "regularMarketPrice": 81234.5,
                            "regularMarketChangePercent": -0.4,
                            "currency": "USD",
                        },
                    ]
                }
            }
            if url == expected_url
            else (_ for _ in ()).throw(AssertionError(f"unexpected url: {url}")),
        )

        context = service.build_prompt_context("Tesla and bitcoin price today")

        assert context is not None
        self.assertIn("asset_1_symbol: TSLA", context)
        self.assertIn("asset_2_symbol: BTC-USD", context)

    def test_geocoding_context_formats_coordinates(self) -> None:
        expected_url = (
            "https://geocoding-api.open-meteo.com/v1/search?"
            + parse.urlencode({"name": "Hangzhou", "count": 1, "language": "en", "format": "json"})
        )
        service = GeocodingContextService(
            fetch_json=lambda url, timeout: {
                "results": [
                    {
                        "name": "Hangzhou",
                        "admin1": "Zhejiang",
                        "country": "China",
                        "latitude": 30.2741,
                        "longitude": 120.1551,
                        "timezone": "Asia/Shanghai",
                    }
                ]
            }
            if url == expected_url
            else (_ for _ in ()).throw(AssertionError(f"unexpected url: {url}")),
        )

        context = service.build_prompt_context("Hangzhou coordinates")

        assert context is not None
        self.assertIn("location: Hangzhou, Zhejiang, China", context)
        self.assertIn("latitude: 30.2741", context)

    def test_time_context_formats_local_time_for_city(self) -> None:
        expected_url = (
            "https://geocoding-api.open-meteo.com/v1/search?"
            + parse.urlencode({"name": "Singapore", "count": 1, "language": "en", "format": "json"})
        )
        service = TimeContextService(
            fetch_json=lambda url, timeout: {
                "results": [
                    {
                        "name": "Singapore",
                        "admin1": "Singapore",
                        "country": "Singapore",
                        "latitude": 1.2897,
                        "longitude": 103.8501,
                        "timezone": "Asia/Singapore",
                    }
                ]
            }
            if url == expected_url
            else (_ for _ in ()).throw(AssertionError(f"unexpected url: {url}")),
        )

        context = service.build_prompt_context("current time in Singapore")

        assert context is not None
        self.assertIn("timezone: Asia/Singapore", context)
        self.assertIn("local_time:", context)

    def test_time_context_formats_holiday_snapshot(self) -> None:
        expected_url = "https://date.nager.at/api/v3/PublicHolidays/2026/JP"
        service = TimeContextService(
            fetch_json=lambda url, timeout: [
                {"date": "2026-01-01", "localName": "元日"},
                {"date": "2026-01-12", "localName": "成人の日"},
            ]
            if url == expected_url
            else (_ for _ in ()).throw(AssertionError(f"unexpected url: {url}")),
        )

        context = service.build_prompt_context("2026 Japan holidays")

        assert context is not None
        self.assertIn("country_code: JP", context)
        self.assertIn("holiday_1_name: 元日", context)

    def test_news_context_formats_rss_items(self) -> None:
        expected_url = (
            "https://news.google.com/rss/search?"
            + parse.urlencode(
                {
                    "q": "AI",
                    "hl": "en-US",
                    "gl": "US",
                    "ceid": "US:en",
                }
            )
        )
        service = NewsContextService(
            fetch_text=lambda url, timeout: FetchedTextResponse(
                text=(
                    "<rss><channel>"
                    "<item><title>AI headline one</title><link>https://example.com/1</link>"
                    "<pubDate>Mon, 23 Mar 2026 00:00:00 GMT</pubDate></item>"
                    "<item><title>AI headline two</title><link>https://example.com/2</link></item>"
                    "</channel></rss>"
                ),
                content_type="application/xml",
            )
            if url == expected_url
            else (_ for _ in ()).throw(AssertionError(f"unexpected url: {url}")),
        )

        context = service.build_prompt_context("AI 新闻")

        assert context is not None
        self.assertIn("article_1_title: AI headline one", context)
        self.assertIn("article_2_link: https://example.com/2", context)

    def test_url_summary_context_extracts_title_and_description(self) -> None:
        service = UrlSummaryContextService(
            fetch_text=lambda url, timeout: FetchedTextResponse(
                text=(
                    "<html><head><title>Example Page</title>"
                    "<meta name=\"description\" content=\"A useful example page.\" />"
                    "</head><body></body></html>"
                ),
                content_type="text/html",
            )
        )

        context = service.build_prompt_context("帮我看看这个页面 https://example.com/docs")

        assert context is not None
        self.assertIn("page_1_url: https://example.com/docs", context)
        self.assertIn("page_1_title: Example Page", context)
        self.assertIn("page_1_description: A useful example page.", context)

    def test_search_context_uses_duckduckgo_snapshot_in_search_mode(self) -> None:
        expected_url = (
            "https://api.duckduckgo.com/?"
            + parse.urlencode(
                {
                    "q": "openai release notes",
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                }
            )
        )
        service = SearchContextService(
            fetch_json=lambda url, timeout: {
                "Heading": "OpenAI",
                "AbstractText": "OpenAI develops artificial intelligence systems.",
                "AbstractURL": "https://openai.com/",
                "RelatedTopics": [
                    {"Text": "OpenAI API documentation"},
                    {"Topics": [{"Text": "OpenAI release history"}]},
                ],
            }
            if url == expected_url
            else (_ for _ in ()).throw(AssertionError(f"unexpected url: {url}")),
        )

        context = service.build_prompt_context("openai release notes", route_mode="search")

        assert context is not None
        self.assertIn("heading: OpenAI", context)
        self.assertIn("source_url: https://openai.com/", context)
        self.assertIn("related_1: OpenAI API documentation", context)

    def test_tech_docs_context_extracts_headings(self) -> None:
        service = TechDocsContextService(
            fetch_text=lambda url, timeout: FetchedTextResponse(
                text=(
                    "<html><head><title>SDK Docs</title>"
                    "<meta name=\"description\" content=\"SDK reference docs.\" />"
                    "</head><body><h1>Authentication</h1><h2>API Keys</h2></body></html>"
                ),
                content_type="text/html",
            )
        )

        context = service.build_prompt_context("请看这个 API 文档 https://example.com/docs/sdk")

        assert context is not None
        self.assertIn("title: SDK Docs", context)
        self.assertIn("heading_1: Authentication", context)

    def test_transport_status_context_uses_configured_endpoint(self) -> None:
        service = TransportStatusContextService(
            endpoint_configs={
                "flight": LiveContextEndpointConfig(
                    name="flight",
                    url_template="https://status.example.com/flight/{query}",
                )
            },
            fetch_json=lambda url, timeout, headers: {"status": "On Time", "gate": "A12"}
            if url == "https://status.example.com/flight/MU5123"
            else (_ for _ in ()).throw(AssertionError(f"unexpected url: {url}")),
        )

        context = service.build_prompt_context("Flight MU5123 status")

        assert context is not None
        self.assertIn("endpoint_name: flight", context)
        self.assertIn("\"status\": \"On Time\"", context)

    def test_custom_api_context_uses_trigger_keywords(self) -> None:
        service = CustomApiContextService(
            endpoint_configs={
                "device": LiveContextEndpointConfig(
                    name="device",
                    url_template="https://internal.example.com/device",
                    trigger_keywords=("device status",),
                    query_param="q",
                    instruction="Use this internal device payload when needed.",
                )
            },
            fetch_json=lambda url, timeout, headers: {"online": True, "battery": 78}
            if url == "https://internal.example.com/device?q=device+status+for+my+companion"
            else (_ for _ in ()).throw(AssertionError(f"unexpected url: {url}")),
        )

        context = service.build_prompt_context("device status for my companion")

        assert context is not None
        self.assertIn("endpoint_name: device", context)
        self.assertIn("\"battery\": 78", context)

    def test_live_context_service_combines_multiple_sources(self) -> None:
        class StaticSource:
            def __init__(self, content: str | None) -> None:
                self.content = content

            def build_prompt_context(self, user_text: str, *, route_mode: str = "chat") -> str | None:
                return self.content

        service = LiveContextService(
            sources=(
                StaticSource("section one"),
                StaticSource(None),
                StaticSource("section two"),
            )
        )

        context = service.build_prompt_context("test")

        self.assertEqual(context, "section one\n\nsection two")


if __name__ == "__main__":
    unittest.main()
