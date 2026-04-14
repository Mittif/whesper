from __future__ import annotations

import unittest
from urllib import parse

from whesper.config import LiveContextEndpointConfig
from whesper.live_data import (
    _extract_map_place,
    _extract_time_place,
    _extract_city_weather_place,
    CityWeatherContextService,
    CustomApiContextService,
    ExchangeRateContextService,
    FetchedTextResponse,
    GeocodingContextService,
    lookup_ip_location,
    lookup_public_ip,
    lookup_weather_for_ip,
    lookup_weather_for_place,
    LiveContextService,
    LocalWeatherContextService,
    MarketPriceContextService,
    NewsContextService,
    SearchContextService,
    should_fetch_weather,
    TechDocsContextService,
    TimeContextService,
    TransportStatusContextService,
    UrlSummaryContextService,
    should_fetch_local_weather,
)


def wttr_weather_url(latitude: float, longitude: float) -> str:
    return f"https://wttr.in/{latitude},{longitude}?format=j1&m"


def open_meteo_geocode_url(name: str, *, language: str = "en") -> str:
    return (
        "https://geocoding-api.open-meteo.com/v1/search?"
        + parse.urlencode({"name": name, "count": 1, "language": language, "format": "json"})
    )


def open_meteo_forecast_url(latitude: float, longitude: float, *, timezone: str) -> str:
    return (
        "https://api.open-meteo.com/v1/forecast?"
        + parse.urlencode(
            {
                "latitude": str(latitude),
                "longitude": str(longitude),
                "timezone": timezone,
                "forecast_days": 8,
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "wind_speed_10m,weather_code"
                ),
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max"
                ),
            }
        )
    )


def wttr_day(
    date_value: str,
    *,
    condition: str,
    high_c: float,
    low_c: float,
    precipitation_probability_max_percent: float,
) -> dict[str, object]:
    return {
        "date": date_value,
        "maxtempC": str(high_c),
        "mintempC": str(low_c),
        "hourly": [
            {
                "time": "1200",
                "weatherDesc": [{"value": condition}],
                "chanceofrain": str(precipitation_probability_max_percent),
                "chanceofsnow": "0",
                "chanceofthunder": "0",
            }
        ],
    }


def wttr_response(
    *,
    current_condition: str,
    current_temperature_c: float,
    apparent_temperature_c: float,
    relative_humidity_percent: float,
    wind_speed_kmh: float,
    daily_forecasts: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "current_condition": [
            {
                "temp_C": str(current_temperature_c),
                "FeelsLikeC": str(apparent_temperature_c),
                "humidity": str(relative_humidity_percent),
                "windspeedKmph": str(wind_speed_kmh),
                "weatherDesc": [{"value": current_condition}],
            }
        ],
        "weather": daily_forecasts,
    }


class LiveDataTests(unittest.TestCase):
    def test_should_fetch_local_weather_requires_weather_and_local_time_intent(self) -> None:
        self.assertTrue(should_fetch_local_weather("今天天气怎么样？"))
        self.assertTrue(should_fetch_local_weather("What's the weather like right now?"))
        self.assertFalse(should_fetch_local_weather("Explain how weather systems form."))
        self.assertFalse(should_fetch_local_weather("今天心情怎么样？"))

    def test_should_fetch_weather_includes_named_location_queries(self) -> None:
        self.assertTrue(should_fetch_weather("Tokyo weather today"))
        self.assertTrue(should_fetch_weather("上海天气怎么样"))
        self.assertFalse(should_fetch_weather("今天心情怎么样？"))

    def test_extract_city_weather_place_strips_search_prefixes(self) -> None:
        self.assertEqual(_extract_city_weather_place("搜索上海天气"), "上海")
        self.assertEqual(_extract_city_weather_place("查一下上海天气"), "上海")
        self.assertEqual(_extract_city_weather_place("/search 上海天气"), "上海")

    def test_build_prompt_context_formats_live_weather_report(self) -> None:
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
            wttr_weather_url(31.2304, 121.4737): wttr_response(
                current_condition="Partly cloudy",
                current_temperature_c=23.4,
                apparent_temperature_c=24.0,
                relative_humidity_percent=61,
                wind_speed_kmh=12.1,
                daily_forecasts=[
                    wttr_day(
                        "2026-03-25",
                        condition="Overcast",
                        high_c=27.2,
                        low_c=19.8,
                        precipitation_probability_max_percent=30,
                    ),
                    wttr_day(
                        "2026-03-26",
                        condition="Slight rain",
                        high_c=25.1,
                        low_c=18.3,
                        precipitation_probability_max_percent=70,
                    ),
                ],
            ),
        }

        service = LocalWeatherContextService(
            fetch_json=lambda url, timeout: responses[url],
        )

        context = service.build_prompt_context("今天天气怎么样？")

        assert context is not None
        self.assertNotIn("public_ip:", context)
        self.assertIn("Live weather data for the user's current local area:", context)
        self.assertIn("location: Shanghai, Shanghai, China", context)
        self.assertIn("current_condition: Partly cloudy", context)
        self.assertIn("today_condition: Overcast", context)
        self.assertIn("today_high_c: 27.2", context)
        self.assertIn("today_low_c: 19.8", context)
        self.assertIn("tomorrow_condition: Slight rain", context)
        self.assertIn("tomorrow_high_c: 25.1", context)

    def test_build_prompt_context_returns_failure_instruction_when_lookup_breaks(self) -> None:
        service = LocalWeatherContextService(
            fetch_json=lambda url, timeout: (_ for _ in ()).throw(TimeoutError("network timeout")),
        )

        context = service.build_prompt_context("What's the weather like today?")

        assert context is not None
        self.assertIn("failed: TimeoutError: network timeout", context)
        self.assertIn("Do not invent the user's current weather", context)

    def test_lookup_weather_helpers_split_ip_geo_and_weather_steps(self) -> None:
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
            wttr_weather_url(31.2304, 121.4737): wttr_response(
                current_condition="Partly cloudy",
                current_temperature_c=23.4,
                apparent_temperature_c=24.0,
                relative_humidity_percent=61,
                wind_speed_kmh=12.1,
                daily_forecasts=[
                    wttr_day(
                        "2026-03-25",
                        condition="Overcast",
                        high_c=27.2,
                        low_c=19.8,
                        precipitation_probability_max_percent=30,
                    ),
                    wttr_day(
                        "2026-03-26",
                        condition="Slight rain",
                        high_c=25.1,
                        low_c=18.3,
                        precipitation_probability_max_percent=70,
                    ),
                ],
            ),
        }
        fetch_json = lambda url, timeout: responses[url]

        public_ip = lookup_public_ip(fetch_json, 10)
        location = lookup_ip_location(fetch_json, 10, public_ip)
        report = lookup_weather_for_ip(fetch_json, 10)

        self.assertEqual(public_ip, "203.0.113.10")
        self.assertEqual(location.label, "Shanghai, Shanghai, China")
        self.assertEqual(report.public_ip, "203.0.113.10")
        self.assertEqual(report.location_label, "Shanghai, Shanghai, China")
        self.assertEqual(report.tomorrow_condition, "Slight rain")

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
        geocode_url = open_meteo_geocode_url("Tokyo")
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
            wttr_weather_url(35.6895, 139.6917): wttr_response(
                current_condition="Mainly clear",
                current_temperature_c=18.2,
                apparent_temperature_c=18.0,
                relative_humidity_percent=70,
                wind_speed_kmh=8.4,
                daily_forecasts=[
                    wttr_day(
                        "2026-03-25",
                        condition="Partly cloudy",
                        high_c=22.5,
                        low_c=14.3,
                        precipitation_probability_max_percent=20,
                    ),
                    wttr_day(
                        "2026-03-26",
                        condition="Overcast",
                        high_c=21.4,
                        low_c=15.0,
                        precipitation_probability_max_percent=35,
                    ),
                ],
            ),
        }
        service = CityWeatherContextService(fetch_json=lambda url, timeout: responses[url])

        context = service.build_prompt_context("Tokyo weather today")

        assert context is not None
        self.assertIn("location: Tokyo, Tokyo, Japan", context)
        self.assertIn("current_condition: Mainly clear", context)
        self.assertIn("tomorrow_condition: Overcast", context)

    def test_lookup_weather_for_place_returns_report(self) -> None:
        geocode_url = open_meteo_geocode_url("Tokyo")
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
            wttr_weather_url(35.6895, 139.6917): wttr_response(
                current_condition="Mainly clear",
                current_temperature_c=18.2,
                apparent_temperature_c=18.0,
                relative_humidity_percent=70,
                wind_speed_kmh=8.4,
                daily_forecasts=[
                    wttr_day(
                        "2026-03-25",
                        condition="Partly cloudy",
                        high_c=22.5,
                        low_c=14.3,
                        precipitation_probability_max_percent=20,
                    ),
                    wttr_day(
                        "2026-03-26",
                        condition="Overcast",
                        high_c=21.4,
                        low_c=15.0,
                        precipitation_probability_max_percent=35,
                    ),
                ],
            ),
        }

        report = lookup_weather_for_place(lambda url, timeout: responses[url], 10, "Tokyo")

        self.assertEqual(report.location_label, "Tokyo, Tokyo, Japan")
        self.assertEqual(report.timezone, "Asia/Tokyo")
        self.assertEqual(report.tomorrow_condition, "Overcast")

    def test_lookup_weather_for_place_uses_language_specific_geocoding(self) -> None:
        zh_geocode_url = open_meteo_geocode_url("東京", language="zh")
        geocode_url = open_meteo_geocode_url("東京", language="ja")
        responses = {
            zh_geocode_url: {"generationtime_ms": 0.12},
            geocode_url: {
                "results": [
                    {
                        "name": "東京",
                        "admin1": "東京都",
                        "country": "日本",
                        "latitude": 35.6895,
                        "longitude": 139.6917,
                        "timezone": "Asia/Tokyo",
                    }
                ]
            },
            wttr_weather_url(35.6895, 139.6917): wttr_response(
                current_condition="Mainly clear",
                current_temperature_c=18.2,
                apparent_temperature_c=18.0,
                relative_humidity_percent=70,
                wind_speed_kmh=8.4,
                daily_forecasts=[
                    wttr_day(
                        "2026-03-25",
                        condition="Partly cloudy",
                        high_c=22.5,
                        low_c=14.3,
                        precipitation_probability_max_percent=20,
                    )
                ],
            ),
        }

        report = lookup_weather_for_place(lambda url, timeout: responses[url], 10, "東京")

        self.assertEqual(report.location_label, "東京, 東京都, 日本")
        self.assertEqual(report.timezone, "Asia/Tokyo")

    def test_lookup_weather_for_place_falls_back_to_open_meteo_forecast_when_wttr_fails(self) -> None:
        geocode_url = open_meteo_geocode_url("Shanghai")
        forecast_url = open_meteo_forecast_url(31.2304, 121.4737, timezone="Asia/Shanghai")
        responses = {
            geocode_url: {
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
            },
            wttr_weather_url(31.2304, 121.4737): None,
            "https://wttr.in/31.2304,121.4737?format=j1": None,
            "https://wttr.in/Shanghai?format=j1&m": None,
            "https://wttr.in/Shanghai?format=j1": None,
            forecast_url: {
                "current": {
                    "temperature_2m": 23.4,
                    "apparent_temperature": 24.0,
                    "relative_humidity_2m": 61,
                    "wind_speed_10m": 12.1,
                    "weather_code": 2,
                },
                "daily": {
                    "time": ["2026-03-25", "2026-03-26"],
                    "weather_code": [3, 61],
                    "temperature_2m_max": [27.2, 25.1],
                    "temperature_2m_min": [19.8, 18.3],
                    "precipitation_probability_max": [30, 70],
                },
            },
        }

        report = lookup_weather_for_place(lambda url, timeout: responses[url], 10, "Shanghai")

        self.assertEqual(report.current_condition, "Partly cloudy")
        self.assertEqual(report.today_condition, "Overcast")
        self.assertEqual(report.tomorrow_condition, "Slight rain")

    def test_city_weather_context_handles_chinese_district_tomorrow_query(self) -> None:
        geocode_url = open_meteo_geocode_url("嘉定区", language="zh")
        responses = {
            geocode_url: {
                "results": [
                    {
                        "name": "Jiading District",
                        "admin1": "Shanghai",
                        "country": "China",
                        "latitude": 31.3835,
                        "longitude": 121.2503,
                        "timezone": "Asia/Shanghai",
                    }
                ]
            },
            wttr_weather_url(31.3835, 121.2503): wttr_response(
                current_condition="Partly cloudy",
                current_temperature_c=22.1,
                apparent_temperature_c=22.0,
                relative_humidity_percent=65,
                wind_speed_kmh=10.2,
                daily_forecasts=[
                    wttr_day(
                        "2026-03-25",
                        condition="Partly cloudy",
                        high_c=25.0,
                        low_c=18.1,
                        precipitation_probability_max_percent=20,
                    ),
                    wttr_day(
                        "2026-03-26",
                        condition="Slight rain",
                        high_c=23.4,
                        low_c=17.2,
                        precipitation_probability_max_percent=80,
                    ),
                ],
            ),
        }
        service = CityWeatherContextService(fetch_json=lambda url, timeout: responses[url])

        context = service.build_prompt_context("嘉定区明天的天气")

        assert context is not None
        self.assertIn("location: Jiading District, Shanghai, China", context)
        self.assertIn("tomorrow_condition: Slight rain", context)
        self.assertIn("tomorrow_high_c: 23.4", context)
        self.assertIn("requested_period: tomorrow", context)
        self.assertIn("requested_date: 2026-03-26", context)

    def test_city_weather_context_handles_next_monday_query(self) -> None:
        geocode_url = open_meteo_geocode_url("嘉定区", language="zh")
        responses = {
            geocode_url: {
                "results": [
                    {
                        "name": "Jiading District",
                        "admin1": "Shanghai",
                        "country": "China",
                        "latitude": 31.3835,
                        "longitude": 121.2503,
                        "timezone": "Asia/Shanghai",
                    }
                ]
            },
            wttr_weather_url(31.3835, 121.2503): wttr_response(
                current_condition="Partly cloudy",
                current_temperature_c=22.1,
                apparent_temperature_c=22.0,
                relative_humidity_percent=65,
                wind_speed_kmh=10.2,
                daily_forecasts=[
                    wttr_day("2026-03-25", condition="Partly cloudy", high_c=25.0, low_c=18.1, precipitation_probability_max_percent=20),
                    wttr_day("2026-03-26", condition="Slight rain", high_c=23.4, low_c=17.2, precipitation_probability_max_percent=80),
                    wttr_day("2026-03-27", condition="Overcast", high_c=24.2, low_c=16.4, precipitation_probability_max_percent=40),
                    wttr_day("2026-03-28", condition="Partly cloudy", high_c=26.1, low_c=18.8, precipitation_probability_max_percent=10),
                    wttr_day("2026-03-29", condition="Fog", high_c=21.0, low_c=15.0, precipitation_probability_max_percent=15),
                    wttr_day("2026-03-30", condition="Moderate rain", high_c=19.8, low_c=13.7, precipitation_probability_max_percent=75),
                ],
            ),
        }
        service = CityWeatherContextService(fetch_json=lambda url, timeout: responses[url])

        context = service.build_prompt_context("嘉定区下周一的天气")

        assert context is not None
        self.assertIn("requested_period: next_monday", context)
        self.assertIn("requested_date: 2026-03-30", context)
        self.assertIn("requested_condition: Moderate rain", context)

    def test_city_weather_context_handles_weekend_query(self) -> None:
        geocode_url = open_meteo_geocode_url("嘉定区", language="zh")
        responses = {
            geocode_url: {
                "results": [
                    {
                        "name": "Jiading District",
                        "admin1": "Shanghai",
                        "country": "China",
                        "latitude": 31.3835,
                        "longitude": 121.2503,
                        "timezone": "Asia/Shanghai",
                    }
                ]
            },
            wttr_weather_url(31.3835, 121.2503): wttr_response(
                current_condition="Partly cloudy",
                current_temperature_c=22.1,
                apparent_temperature_c=22.0,
                relative_humidity_percent=65,
                wind_speed_kmh=10.2,
                daily_forecasts=[
                    wttr_day("2026-03-25", condition="Partly cloudy", high_c=25.0, low_c=18.1, precipitation_probability_max_percent=20),
                    wttr_day("2026-03-26", condition="Slight rain", high_c=23.4, low_c=17.2, precipitation_probability_max_percent=80),
                    wttr_day("2026-03-27", condition="Overcast", high_c=24.2, low_c=16.4, precipitation_probability_max_percent=40),
                    wttr_day("2026-03-28", condition="Partly cloudy", high_c=26.1, low_c=18.8, precipitation_probability_max_percent=10),
                    wttr_day("2026-03-29", condition="Fog", high_c=21.0, low_c=15.0, precipitation_probability_max_percent=15),
                    wttr_day("2026-03-30", condition="Moderate rain", high_c=19.8, low_c=13.7, precipitation_probability_max_percent=75),
                ],
            ),
        }
        service = CityWeatherContextService(fetch_json=lambda url, timeout: responses[url])

        context = service.build_prompt_context("嘉定区周末天气")

        assert context is not None
        self.assertIn("requested_period: weekend", context)
        self.assertIn("requested_day_1_date: 2026-03-28", context)
        self.assertIn("requested_day_2_date: 2026-03-29", context)
        self.assertIn("requested_day_2_condition: Fog", context)

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

    def test_time_context_formats_local_time_for_place_prefixed_current_time_query(self) -> None:
        expected_url = (
            "https://geocoding-api.open-meteo.com/v1/search?"
            + parse.urlencode({"name": "Tokyo", "count": 1, "language": "en", "format": "json"})
        )
        service = TimeContextService(
            fetch_json=lambda url, timeout: {
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
            if url == expected_url
            else (_ for _ in ()).throw(AssertionError(f"unexpected url: {url}")),
        )

        context = service.build_prompt_context("Tokyo current time")

        assert context is not None
        self.assertIn("location: Tokyo, Tokyo, Japan", context)
        self.assertIn("timezone: Asia/Tokyo", context)

    def test_time_context_formats_local_time_for_chinese_remote_current_time_query(self) -> None:
        expected_url = (
            "https://geocoding-api.open-meteo.com/v1/search?"
            + parse.urlencode({"name": "新加坡", "count": 1, "language": "zh", "format": "json"})
        )
        service = TimeContextService(
            fetch_json=lambda url, timeout: {
                "results": [
                    {
                        "name": "新加坡",
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

        context = service.build_prompt_context("新加坡当前时间")

        assert context is not None
        self.assertIn("location: 新加坡, Singapore", context)
        self.assertIn("timezone: Asia/Singapore", context)

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

    def test_extract_time_place_ignores_contextual_placeholder_city(self) -> None:
        self.assertIsNone(_extract_time_place("那个城市的时区"))

    def test_extract_map_place_ignores_contextual_placeholder_location(self) -> None:
        self.assertIsNone(_extract_map_place("那里经纬度"))

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

    def test_search_context_uses_serpapi_snapshot_in_search_mode(self) -> None:
        expected_url = (
            "https://serpapi.com/search?"
            + parse.urlencode(
                {
                    "engine": "google",
                    "q": "openai release notes",
                    "api_key": "serp-key",
                    "num": 3,
                }
            )
        )
        service = SearchContextService(
            provider="serpapi",
            api_key="serp-key",
            fetch_json=lambda url, timeout: {
                "answer_box": {
                    "answer": "OpenAI 发布了新的 release notes。",
                    "link": "https://openai.com/",
                },
                "organic_results": [
                    {
                        "title": "OpenAI API release notes",
                        "snippet": "Latest platform changes and release notes.",
                        "link": "https://platform.openai.com/docs/release-notes",
                    },
                    {
                        "title": "OpenAI changelog",
                        "snippet": "Model updates and API changes.",
                        "link": "https://openai.com/changelog",
                    },
                ],
            }
            if url == expected_url
            else (_ for _ in ()).throw(AssertionError(f"unexpected url: {url}")),
        )

        context = service.build_prompt_context("openai release notes", route_mode="search")

        assert context is not None
        self.assertIn("provider: serpapi", context)
        self.assertIn("summary: OpenAI 发布了新的 release notes。", context)
        self.assertIn("summary_url: https://openai.com/", context)
        self.assertIn("result_1_title: OpenAI API release notes", context)
        self.assertIn("result_2_link: https://openai.com/changelog", context)

    def test_search_context_brave_requires_api_key(self) -> None:
        service = SearchContextService(provider="brave")
        snapshot = service.search_query("openai release notes")
        self.assertEqual(snapshot.status, "error")
        self.assertEqual(snapshot.error_code, "missing_api_key")

    def test_search_context_duckduckgo_is_default(self) -> None:
        service = SearchContextService()
        self.assertEqual(service.provider, "duckduckgo")

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
