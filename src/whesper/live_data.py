from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from html import unescape as html_unescape
from html.parser import HTMLParser
import json
import os
import re
import sys
from zoneinfo import ZoneInfo
from typing import Callable, Protocol
from urllib import parse, request
from xml.etree import ElementTree

from whesper.config import AppConfig, LiveContextEndpointConfig


USER_AGENT = "Whesper/0.1"
WEATHER_FORECAST_DAYS = 8
WTTR_BASE_URL = "https://wttr.in"
WTTR_FETCH_ATTEMPTS = 2
URL_PATTERN = re.compile(r"https?://[^\s<>\"]+")
CITY_PATTERN = re.compile(
    r"(?P<place>[A-Za-z][A-Za-z .'-]{1,40}|[\u4e00-\u9fff][\u4e00-\u9fffA-Za-z·\-\s]{0,20})"
)
FLIGHT_CODE_PATTERN = re.compile(r"\b([A-Z]{2,3}\s?\d{2,4})\b")
TRAIN_CODE_PATTERN = re.compile(r"\b([GDCZKTLSY]\d{1,4})\b", re.IGNORECASE)
TRACKING_CODE_PATTERN = re.compile(r"\b([A-Z0-9]{8,22})\b")
_ALLOW_DDGS_ON_PY314_ENV = "WHESPER_ENABLE_DDGS_PY314"


WEATHER_CODE_LABELS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

WEATHER_KEYWORDS = (
    "weather",
    "forecast",
    "temperature",
    "rain",
    "snow",
    "storm",
    "天气",
    "气温",
    "温度",
    "下雨",
    "下雪",
    "降雨",
    "降雪",
    "晴",
    "阴",
    "多云",
)

WEATHER_DAY_KEYWORDS = (
    "today",
    "tomorrow",
    "day after tomorrow",
    "tonight",
    "今天",
    "明天",
    "后天",
    "今晚",
)

WEEKDAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

WEEKEND_KEYWORDS = (
    "weekend",
    "this weekend",
    "next weekend",
    "周末",
    "本周末",
    "这周末",
    "下周末",
)

WEEKDAY_KEYWORDS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (0, ("monday", "mon", "周一", "星期一", "礼拜一", "本周一", "这周一", "下周一")),
    (1, ("tuesday", "tue", "周二", "星期二", "礼拜二", "本周二", "这周二", "下周二")),
    (2, ("wednesday", "wed", "周三", "星期三", "礼拜三", "本周三", "这周三", "下周三")),
    (3, ("thursday", "thu", "周四", "星期四", "礼拜四", "本周四", "这周四", "下周四")),
    (4, ("friday", "fri", "周五", "星期五", "礼拜五", "本周五", "这周五", "下周五")),
    (5, ("saturday", "sat", "周六", "星期六", "礼拜六", "本周六", "这周六", "下周六")),
    (6, ("sunday", "sun", "周日", "周天", "星期日", "星期天", "礼拜日", "礼拜天", "本周日", "这周日", "下周日")),
)

LOCAL_TIME_KEYWORDS = (
    "today",
    "tonight",
    "now",
    "current",
    "local",
    "where i am",
    "where i'm",
    "my area",
    "今天",
    "现在",
    "当前",
    "本地",
    "当地",
    "我这里",
    "这边",
)

EXCHANGE_KEYWORDS = (
    "exchange rate",
    "fx",
    "forex",
    "rate",
    "convert",
    "conversion",
    "兑换",
    "汇率",
    "汇价",
    "换算",
    "兑",
)

SEARCH_KEYWORDS = (
    "search",
    "look up",
    "find",
    "news",
    "latest",
    "recent",
    "搜索",
    "查一下",
    "查查",
    "最新",
    "最近",
    "新闻",
)

PLACE_QUERY_PREFIXES = (
    "/search ",
    "search ",
    "look up ",
    "find ",
    "搜索",
    "查一下",
    "查查",
    "搜一下",
    "搜搜",
    "帮我搜索",
    "帮我查一下",
    "帮我查",
    "帮我搜一下",
    "请搜索",
    "请查一下",
    "请查",
    "请搜一下",
    "麻烦搜索",
    "麻烦查一下",
    "麻烦查",
    "麻烦搜一下",
)

NEWS_KEYWORDS = (
    "news",
    "headline",
    "headlines",
    "breaking",
    "latest news",
    "recent news",
    "新闻",
    "头条",
    "最新消息",
    "最近新闻",
)

MARKET_KEYWORDS = (
    "stock",
    "stocks",
    "share price",
    "price",
    "market cap",
    "crypto",
    "coin",
    "token",
    "股票",
    "股价",
    "币价",
    "加密货币",
    "市值",
)

MAP_KEYWORDS = (
    "where is",
    "locate",
    "coordinates",
    "latitude",
    "longitude",
    "map",
    "在哪",
    "哪里",
    "坐标",
    "经纬度",
    "地址",
)

TIME_KEYWORDS = (
    "time in",
    "current time",
    "local time",
    "what time",
    "timezone",
    "time difference",
    "holiday",
    "holidays",
    "几点",
    "当前时间",
    "本地时间",
    "当地时间",
    "现在时间",
    "时区",
    "时差",
    "节假日",
    "假期",
)

DOC_KEYWORDS = (
    "docs",
    "documentation",
    "api reference",
    "reference",
    "manual",
    "guide",
    "readme",
    "文档",
    "接口文档",
    "参考",
    "说明书",
)

FLIGHT_KEYWORDS = ("flight", "航班")
TRAIN_KEYWORDS = ("train", "高铁", "列车", "火车")
PARCEL_KEYWORDS = ("tracking", "package", "parcel", "快递", "物流", "运单")

CURRENCY_ALIASES = {
    "usd": "USD",
    "美元": "USD",
    "us dollar": "USD",
    "cny": "CNY",
    "rmb": "CNY",
    "人民币": "CNY",
    "eur": "EUR",
    "欧元": "EUR",
    "jpy": "JPY",
    "日元": "JPY",
    "gbp": "GBP",
    "英镑": "GBP",
    "hkd": "HKD",
    "港币": "HKD",
    "aud": "AUD",
    "澳元": "AUD",
    "cad": "CAD",
    "加元": "CAD",
    "sgd": "SGD",
    "新加坡元": "SGD",
    "chf": "CHF",
    "瑞士法郎": "CHF",
    "krw": "KRW",
    "韩元": "KRW",
}

MARKET_SYMBOL_ALIASES = {
    "bitcoin": "BTC-USD",
    "比特币": "BTC-USD",
    "btc": "BTC-USD",
    "ethereum": "ETH-USD",
    "以太坊": "ETH-USD",
    "eth": "ETH-USD",
    "solana": "SOL-USD",
    "sol": "SOL-USD",
    "dogecoin": "DOGE-USD",
    "doge": "DOGE-USD",
    "apple": "AAPL",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "microsoft": "MSFT",
    "google": "GOOGL",
}

COUNTRY_ALIASES = {
    "china": "CN",
    "中国": "CN",
    "united states": "US",
    "usa": "US",
    "美国": "US",
    "japan": "JP",
    "日本": "JP",
    "united kingdom": "GB",
    "uk": "GB",
    "britain": "GB",
    "英国": "GB",
    "singapore": "SG",
    "新加坡": "SG",
    "germany": "DE",
    "德国": "DE",
    "france": "FR",
    "法国": "FR",
}

STOPWORD_LOCATIONS = {
    "today",
    "tonight",
    "now",
    "current",
    "local",
    "weather",
    "forecast",
    "time",
    "news",
    "holiday",
    "holidays",
    "今天",
    "现在",
    "当前",
    "本地",
    "当地",
    "那里",
    "那个城市",
    "那个地方",
    "there",
    "that city",
    "that place",
    "天气",
    "新闻",
    "节假日",
}
STOPWORD_LOCATION_PREFIXES = ("what", "which", "how", "my", "the", "this", "that")


JsonFetcher = Callable[[str, int], dict]
TextFetcher = Callable[[str, int], "FetchedTextResponse"]
HeaderJsonFetcher = Callable[[str, int, dict[str, str]], dict]
HeaderTextFetcher = Callable[[str, int, dict[str, str]], "FetchedTextResponse"]


class LiveContextSource(Protocol):
    def build_prompt_context(self, user_text: str, *, route_mode: str = "chat") -> str | None:
        ...


@dataclass(slots=True)
class FetchedTextResponse:
    text: str
    content_type: str


@dataclass(slots=True)
class HtmlSnapshot:
    title: str
    description: str
    headings: tuple[str, ...] = ()


@dataclass(slots=True)
class SearchResultItem:
    title: str
    url: str
    snippet: str
    source: str
    page_age: str | None = None


@dataclass(slots=True)
class SearchSnapshot:
    query: str
    provider: str
    status: str = "ok"
    summary: str | None = None
    summary_url: str | None = None
    results: tuple[SearchResultItem, ...] = ()
    error_code: str | None = None
    error_message: str | None = None

    def to_prompt_context(self) -> str:
        if self.status == "error":
            lines = [
                "Live search lookup status:",
                f"- query: {self.query}",
                f"- provider: {self.provider}",
            ]
            if self.error_code:
                lines.append(f"- error_code: {self.error_code}")
            if self.error_message:
                lines.append(f"- failed: {self.error_message}")
            lines.append(
                "- instruction: If live search is unavailable, say so briefly instead of pretending you searched."
            )
            return "\n".join(lines)

        lines = [
            "Live search snapshot:",
            f"- query: {self.query}",
            f"- provider: {self.provider}",
            f"- status: {self.status}",
        ]
        if self.summary:
            lines.append(f"- summary: {self.summary}")
        if self.summary_url:
            lines.append(f"- summary_url: {self.summary_url}")
        for index, item in enumerate(self.results, start=1):
            lines.append(f"- result_{index}_title: {item.title}")
            if item.source:
                lines.append(f"- result_{index}_source: {item.source}")
            if item.page_age:
                lines.append(f"- result_{index}_page_age: {item.page_age}")
            if item.snippet:
                lines.append(f"- result_{index}_snippet: {item.snippet}")
            lines.append(f"- result_{index}_link: {item.url}")
        if not self.results and not self.summary:
            lines.append("- note: No concise web search results were available for this query.")
        lines.append(
            "- instruction: Treat this as a lightweight web search snapshot. Mention the source title or domain you relied on, and be careful with freshness claims."
        )
        return "\n".join(lines)

    def to_tool_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "query": self.query,
            "provider": self.provider,
            "status": self.status,
            "sources": [
                {
                    "title": item.title,
                    "url": item.url,
                    "snippet": item.snippet,
                    "source": item.source,
                    "page_age": item.page_age,
                }
                for item in self.results
            ],
            "result": self.to_prompt_context(),
        }
        if self.summary:
            payload["summary"] = self.summary
        if self.summary_url:
            payload["summary_url"] = self.summary_url
        if self.error_code:
            payload["error_code"] = self.error_code
        if self.error_message:
            payload["error"] = self.error_message
        return payload


@dataclass(slots=True)
class WeatherReport:
    fetched_at_utc: str
    location_label: str
    latitude: float
    longitude: float
    timezone: str
    current_condition: str
    current_temperature_c: float
    apparent_temperature_c: float
    relative_humidity_percent: float
    wind_speed_kmh: float
    today_condition: str
    today_high_c: float
    today_low_c: float
    today_precipitation_probability_max_percent: float
    tomorrow_condition: str | None = None
    tomorrow_high_c: float | None = None
    tomorrow_low_c: float | None = None
    tomorrow_precipitation_probability_max_percent: float | None = None
    daily_forecasts: tuple["WeatherDailyForecast", ...] = ()
    public_ip: str | None = None


@dataclass(slots=True)
class WeatherDailyForecast:
    date: str
    condition: str
    high_c: float
    low_c: float
    precipitation_probability_max_percent: float


@dataclass(slots=True)
class IpLocation:
    public_ip: str
    city: str
    region: str
    country: str
    latitude: float
    longitude: float
    timezone: str

    @property
    def label(self) -> str:
        return ", ".join(part for part in (self.city, self.region, self.country) if part) or "Unknown"


def _http_request(url: str, headers: dict[str, str] | None = None) -> request.Request:
    merged_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json,text/html,application/xml,text/plain,*/*",
    }
    if headers:
        merged_headers.update(headers)
    return request.Request(url, headers=merged_headers)


def _default_fetch_json(url: str, timeout_seconds: int) -> dict:
    with request.urlopen(_http_request(url), timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _default_fetch_text(url: str, timeout_seconds: int) -> FetchedTextResponse:
    with request.urlopen(_http_request(url), timeout=timeout_seconds) as response:
        raw = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
        content_type = response.headers.get_content_type() or "text/plain"
    return FetchedTextResponse(
        text=raw.decode(charset, errors="replace"),
        content_type=content_type,
    )


def _default_fetch_json_with_headers(
    url: str,
    timeout_seconds: int,
    headers: dict[str, str],
) -> dict:
    with request.urlopen(_http_request(url, headers), timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _default_fetch_text_with_headers(
    url: str,
    timeout_seconds: int,
    headers: dict[str, str],
) -> FetchedTextResponse:
    with request.urlopen(_http_request(url, headers), timeout=timeout_seconds) as response:
        raw = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
        content_type = response.headers.get_content_type() or "text/plain"
    return FetchedTextResponse(
        text=raw.decode(charset, errors="replace"),
        content_type=content_type,
    )


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def _truncate_text(value: str, limit: int = 320) -> str:
    collapsed = _collapse_whitespace(value)
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def _format_number(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _weather_label(code: object) -> str:
    if not isinstance(code, int):
        return "Unknown"
    return WEATHER_CODE_LABELS.get(code, f"Unknown ({code})")


def _extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for match in URL_PATTERN.findall(text):
        url = match.rstrip(".,!?)]}")
        if url not in seen:
            urls.append(url)
            seen.add(url)
    return urls


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(keyword.casefold() in lowered for keyword in keywords)


def _looks_like_docs_url(url: str) -> bool:
    lowered = url.casefold()
    return any(
        marker in lowered
        for marker in ("/docs", "/api", "/reference", "/manual", "/guide", "readme", "docs.")
    )


def _normalize_place_candidate(candidate: str) -> str | None:
    cleaned = candidate.strip(" ,.?，。！？")
    cleaned = _strip_place_query_prefixes(cleaned)
    cleaned = re.sub(r"^(in|for|at)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.removeprefix("在").removeprefix("去")
    cleaned = _strip_weather_time_words(cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return None
    lowered = cleaned.casefold()
    if lowered.startswith(STOPWORD_LOCATION_PREFIXES):
        return None
    if cleaned.casefold() in STOPWORD_LOCATIONS:
        return None
    return cleaned


def _strip_place_query_prefixes(value: str) -> str:
    cleaned = value.strip()
    while cleaned:
        lowered = cleaned.casefold()
        matched_prefix = next(
            (
                prefix
                for prefix in sorted(PLACE_QUERY_PREFIXES, key=len, reverse=True)
                if lowered.startswith(prefix.casefold())
            ),
            None,
        )
        if matched_prefix is None:
            break
        cleaned = cleaned[len(matched_prefix) :].lstrip(" ,.?，。！？")
    return cleaned


def _strip_weather_time_words(value: str) -> str:
    cleaned = value
    phrases = list(WEATHER_DAY_KEYWORDS) + list(WEEKEND_KEYWORDS)
    for weekday, aliases in WEEKDAY_KEYWORDS:
        phrases.extend(aliases)
        phrases.append(f"next {WEEKDAY_NAMES[weekday]}")
    for keyword in sorted(set(phrases), key=len, reverse=True):
        cleaned = re.sub(re.escape(keyword), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*的\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" ,.?，。！？")


def should_fetch_local_weather(user_text: str) -> bool:
    lowered = user_text.casefold()
    has_weather_keyword = any(keyword in lowered for keyword in WEATHER_KEYWORDS)
    has_local_time_keyword = any(keyword in lowered for keyword in LOCAL_TIME_KEYWORDS)
    if not has_local_time_keyword:
        has_local_time_keyword = _extract_weather_request_window(user_text, reference_date=None) is not None
    return has_weather_keyword and has_local_time_keyword


def should_fetch_weather(user_text: str) -> bool:
    return should_fetch_local_weather(user_text) or _extract_city_weather_place(user_text) is not None


def _extract_currency_pair(user_text: str) -> tuple[str, str] | None:
    matches: list[tuple[int, str]] = []
    lowered = user_text.casefold()

    for alias, code in sorted(CURRENCY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        alias_lower = alias.casefold()
        start = 0
        while True:
            index = lowered.find(alias_lower, start)
            if index == -1:
                break
            matches.append((index, code))
            start = index + len(alias_lower)

    for match in re.finditer(r"\b([A-Za-z]{3})\b", user_text):
        code = match.group(1).upper()
        if code in CURRENCY_ALIASES.values():
            matches.append((match.start(), code))

    matches.sort()
    pair: list[str] = []
    for _, code in matches:
        if not pair or pair[-1] != code:
            pair.append(code)
        if len(pair) == 2:
            return pair[0], pair[1]
    return None


def _extract_market_symbols(user_text: str) -> tuple[str, ...]:
    indexed_symbols: list[tuple[int, str]] = []
    lowered = user_text.casefold()
    for alias, symbol in MARKET_SYMBOL_ALIASES.items():
        index = lowered.find(alias.casefold())
        if index != -1:
            indexed_symbols.append((index, symbol))

    for match in re.finditer(r"\$([A-Za-z]{1,10})\b", user_text):
        symbol = match.group(1).upper()
        indexed_symbols.append((match.start(), symbol))

    for match in re.finditer(r"\b([A-Z]{2,5}(?:-[A-Z]{3})?)\b", user_text):
        symbol = match.group(1).upper()
        indexed_symbols.append((match.start(), symbol))

    indexed_symbols.sort()

    normalized: list[str] = []
    seen: set[str] = set()
    for _, symbol in indexed_symbols:
        if symbol in {"BTC", "ETH", "SOL", "DOGE"}:
            symbol = f"{symbol}-USD"
        if symbol in seen:
            continue
        normalized.append(symbol)
        seen.add(symbol)
        if len(normalized) >= 3:
            break
    return tuple(normalized)


def _extract_place_near_keywords(user_text: str, keywords: tuple[str, ...]) -> str | None:
    for keyword in keywords:
        pattern = re.compile(
            rf"{re.escape(keyword)}\s+(?P<place>[A-Za-z][A-Za-z .'-]{{1,40}}|[\u4e00-\u9fff][\u4e00-\u9fffA-Za-z·\-\s]{{0,20}})",
            re.IGNORECASE,
        )
        match = pattern.search(user_text)
        if match:
            candidate = _normalize_place_candidate(match.group("place"))
            if candidate:
                return candidate
    return None


def _extract_city_weather_place(user_text: str) -> str | None:
    if not _contains_any(user_text, WEATHER_KEYWORDS):
        return None

    direct_patterns = (
        r"(?P<place>[\u4e00-\u9fffA-Za-z·\-\s]{1,20})的?(?:天气|气温|温度|降雨|降雪)",
        r"(?P<place>[A-Za-z][A-Za-z .'-]{1,40})\s+(?:weather|forecast|temperature)",
        r"(?:weather|forecast|temperature)\s+(?:in|for)\s+(?P<place>[A-Za-z][A-Za-z .'-]{1,40})",
    )
    for pattern in direct_patterns:
        match = re.search(pattern, user_text, re.IGNORECASE)
        if match:
            candidate = _normalize_place_candidate(match.group("place"))
            if candidate:
                return candidate
    return None


def _extract_map_place(user_text: str) -> str | None:
    if not _contains_any(user_text, MAP_KEYWORDS):
        return None
    for pattern in (
        r"(?:where is|locate|coordinates for|latitude of|longitude of)\s+(?P<place>[A-Za-z][A-Za-z .'-]{1,40})",
        r"(?P<place>[A-Za-z][A-Za-z .'-]{1,40})\s+(?:coordinates|latitude|longitude|map)",
        r"(?P<place>[\u4e00-\u9fffA-Za-z·\-\s]{1,20})(?:在哪|在哪里|坐标|经纬度|地址)",
    ):
        match = re.search(pattern, user_text, re.IGNORECASE)
        if match:
            candidate = _normalize_place_candidate(match.group("place"))
            if candidate:
                return candidate
    return None


def _extract_time_place(user_text: str) -> str | None:
    for pattern in (
        r"(?:time in|what time is it in|timezone of|time difference between)\s+(?P<place>[A-Za-z][A-Za-z .'-]{1,40})",
        r"(?P<place>[A-Za-z][A-Za-z .'-]{1,40})\s+(?:current time|local time|time now|timezone)",
        r"(?P<place>[\u4e00-\u9fffA-Za-z·\-\s]{1,20})(?:当前时间|本地时间|当地时间|几点|时间|时区|时差)",
    ):
        match = re.search(pattern, user_text, re.IGNORECASE)
        if match:
            candidate = _normalize_time_place_candidate(match.group("place"))
            if candidate:
                return candidate
    return None


def _normalize_time_place_candidate(candidate: str) -> str | None:
    cleaned = candidate.strip(" ,.?，。！？")
    cleaned = _strip_place_query_prefixes(cleaned)
    cleaned = re.sub(r"^(in|for|at)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.removeprefix("在").removeprefix("去")
    for keyword in (
        "current",
        "local",
        "now",
        "right now",
        "当前",
        "本地",
        "当地",
        "现在",
    ):
        cleaned = re.sub(re.escape(keyword), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*的\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.?，。！？")
    if not cleaned:
        return None
    lowered = cleaned.casefold()
    if lowered.startswith(STOPWORD_LOCATION_PREFIXES):
        return None
    if lowered in STOPWORD_LOCATIONS:
        return None
    return cleaned


def _extract_country_code(user_text: str) -> str | None:
    lowered = user_text.casefold()
    for alias, code in sorted(COUNTRY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias.casefold() in lowered:
            return code
    match = re.search(r"\b([A-Z]{2})\b", user_text)
    if match:
        return match.group(1).upper()
    return None


def _extract_year(user_text: str) -> int | None:
    match = re.search(r"\b(20\d{2})\b", user_text)
    if match:
        return int(match.group(1))
    return None


def _extract_news_query(user_text: str) -> str:
    text = user_text.strip()
    text = re.sub(r"^/search\s+", "", text, flags=re.IGNORECASE)
    prefixes = (
        "latest news about",
        "news about",
        "search news about",
        "最新新闻",
        "最近新闻",
        "查一下新闻",
        "新闻",
    )
    lowered = text.casefold()
    for prefix in prefixes:
        if lowered.startswith(prefix.casefold()):
            stripped = text[len(prefix) :].strip(" :：")
            return stripped or "latest headlines"
    text = re.sub(r"\s*(新闻|news)\s*$", "", text, flags=re.IGNORECASE).strip()
    return text or "latest headlines"


def _extract_status_code(user_text: str, patterns: tuple[re.Pattern[str], ...]) -> str | None:
    for pattern in patterns:
        match = pattern.search(user_text)
        if match:
            return match.group(1).replace(" ", "").upper()
    return None


class _MetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_title = False
        self._in_heading = False
        self._title_parts: list[str] = []
        self._heading_parts: list[str] = []
        self._meta_title: str | None = None
        self._description: str | None = None
        self._headings: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered == "title":
            self._in_title = True
            return
        if lowered in {"h1", "h2"}:
            self._in_heading = True
            self._heading_parts = []
            return
        if lowered != "meta":
            return

        normalized = {key.lower(): value or "" for key, value in attrs}
        name = normalized.get("name", "").casefold()
        prop = normalized.get("property", "").casefold()
        content = _collapse_whitespace(normalized.get("content", ""))
        if not content:
            return
        if self._meta_title is None and prop == "og:title":
            self._meta_title = content
        if self._description is None and name == "description":
            self._description = content
        if self._description is None and prop == "og:description":
            self._description = content

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered == "title":
            self._in_title = False
            return
        if lowered in {"h1", "h2"} and self._in_heading:
            heading = _collapse_whitespace("".join(self._heading_parts))
            if heading and heading not in self._headings:
                self._headings.append(heading)
            self._in_heading = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
        if self._in_heading:
            self._heading_parts.append(data)

    def snapshot(self) -> HtmlSnapshot:
        title = self._meta_title or _collapse_whitespace("".join(self._title_parts))
        description = self._description or ""
        return HtmlSnapshot(
            title=title,
            description=description,
            headings=tuple(self._headings[:3]),
        )


@dataclass(slots=True)
class LiveContextService:
    sources: tuple[LiveContextSource, ...] = field(default_factory=tuple)

    @classmethod
    def from_config(cls, config: AppConfig) -> "LiveContextService":
        return cls(
            sources=(
                UrlSummaryContextService(),
                TechDocsContextService(),
                LocalWeatherContextService(),
                CityWeatherContextService(),
                ExchangeRateContextService(),
                MarketPriceContextService(),
                GeocodingContextService(),
                TimeContextService(),
                NewsContextService(),
                SearchContextService(
                    provider=config.live_context.search_api.provider,
                    api_key=config.live_context.search_api.resolved_api_key(),
                    engine=config.live_context.search_api.engine,
                    timeout_seconds=config.live_context.search_api.timeout_seconds,
                ),
                TransportStatusContextService(config.live_context.status_api),
                CustomApiContextService(config.live_context.custom_api),
            )
        )

    def __post_init__(self) -> None:
        if not self.sources:
            self.sources = LiveContextService.from_config(_default_config()).sources

    def build_prompt_context(self, user_text: str, *, route_mode: str = "chat") -> str | None:
        contexts: list[str] = []
        for source in self.sources:
            context = source.build_prompt_context(user_text, route_mode=route_mode)
            if context:
                contexts.append(context)
        if not contexts:
            return None
        return "\n\n".join(contexts)


def _default_config() -> AppConfig:
    from pathlib import Path
    from whesper.config import (
        AppConfig as _AppConfig,
        AppSettings,
        LiveContextSettings,
        PersonaConfig,
        SchedulerConfig,
        ShellSandboxSettings,
    )

    return _AppConfig(
        app=AppSettings(),
        persona=PersonaConfig(),
        scheduler=SchedulerConfig(chat_model="chat"),
        live_context=LiveContextSettings(),
        shell_sandbox=ShellSandboxSettings(),
        providers={},
        models={},
        source_path=Path("whesper.toml"),
    )


@dataclass(slots=True)
class UrlSummaryContextService:
    fetch_text: TextFetcher = _default_fetch_text
    timeout_seconds: int = 10
    max_urls: int = 2

    def build_prompt_context(self, user_text: str, *, route_mode: str = "chat") -> str | None:
        urls = _extract_urls(user_text)
        if not urls:
            return None

        lines = ["Live webpage snapshot data:"]
        for index, url in enumerate(urls[: self.max_urls], start=1):
            try:
                response = self.fetch_text(url, self.timeout_seconds)
                snapshot = self._snapshot_from_response(response)
                lines.extend(
                    (
                        f"- page_{index}_url: {url}",
                        f"- page_{index}_content_type: {response.content_type}",
                        f"- page_{index}_title: {snapshot.title or 'Unknown'}",
                        f"- page_{index}_description: {snapshot.description or 'Unavailable'}",
                    )
                )
            except Exception as exc:
                lines.extend(
                    (
                        f"- page_{index}_url: {url}",
                        f"- page_{index}_error: {exc.__class__.__name__}: {exc}",
                    )
                )
        lines.append(
            "- instruction: Use these fetched webpage details instead of guessing what the URL contains."
        )
        return "\n".join(lines)

    def _snapshot_from_response(self, response: FetchedTextResponse) -> HtmlSnapshot:
        if "html" not in response.content_type:
            plain_text = _truncate_text(response.text, limit=280)
            return HtmlSnapshot(title=plain_text or "Plain text response", description="")

        parser = _MetadataParser()
        parser.feed(response.text)
        parser.close()
        return parser.snapshot()


@dataclass(slots=True)
class TechDocsContextService:
    fetch_text: TextFetcher = _default_fetch_text
    timeout_seconds: int = 10

    def build_prompt_context(self, user_text: str, *, route_mode: str = "chat") -> str | None:
        urls = _extract_urls(user_text)
        if not urls:
            return None
        if not (_contains_any(user_text, DOC_KEYWORDS) or any(_looks_like_docs_url(url) for url in urls)):
            return None

        url = urls[0]
        try:
            response = self.fetch_text(url, self.timeout_seconds)
        except Exception as exc:
            return (
                "Live technical-doc lookup status:\n"
                f"- url: {url}\n"
                f"- failed: {exc.__class__.__name__}: {exc}\n"
                "- instruction: Do not pretend you successfully opened the documentation page."
            )

        snapshot = UrlSummaryContextService(fetch_text=self.fetch_text)._snapshot_from_response(response)
        lines = [
            "Live technical documentation snapshot:",
            f"- url: {url}",
            f"- title: {snapshot.title or 'Unknown'}",
            f"- description: {snapshot.description or 'Unavailable'}",
        ]
        for index, heading in enumerate(snapshot.headings[:3], start=1):
            lines.append(f"- heading_{index}: {heading}")
        lines.append(
            "- instruction: Use this documentation snapshot to ground your answer, and avoid inventing undocumented API details."
        )
        return "\n".join(lines)


@dataclass(slots=True)
class LocalWeatherContextService:
    fetch_json: JsonFetcher = _default_fetch_json
    timeout_seconds: int = 10

    def build_prompt_context(self, user_text: str, *, route_mode: str = "chat") -> str | None:
        if _extract_city_weather_place(user_text) is not None:
            return None
        if not should_fetch_local_weather(user_text):
            return None

        try:
            report = self._lookup_weather_report()
        except Exception as exc:
            return (
                "Live weather lookup status:\n"
                f"- failed: {exc.__class__.__name__}: {exc}\n"
                "- instruction: Do not invent the user's current weather. Explain briefly that live weather lookup was unavailable."
            )

        return _weather_context_lines(
            "Live weather data for the user's current local area:",
            report,
            include_ip=False,
            user_text=user_text,
        )

    def _lookup_weather_report(self) -> WeatherReport:
        return lookup_weather_for_ip(self.fetch_json, self.timeout_seconds)


@dataclass(slots=True)
class CityWeatherContextService:
    fetch_json: JsonFetcher = _default_fetch_json
    timeout_seconds: int = 10

    def build_prompt_context(self, user_text: str, *, route_mode: str = "chat") -> str | None:
        place = _extract_city_weather_place(user_text)
        if place is None:
            return None

        try:
            report = lookup_weather_for_place(self.fetch_json, self.timeout_seconds, place)
        except Exception as exc:
            return (
                "Live weather lookup status:\n"
                f"- place: {place}\n"
                f"- failed: {exc.__class__.__name__}: {exc}\n"
                "- instruction: Do not invent the requested city's live weather."
            )

        return _weather_context_lines(
            "Live weather data for a requested location:",
            report,
            include_ip=False,
            user_text=user_text,
        )


@dataclass(slots=True)
class ExchangeRateContextService:
    fetch_json: JsonFetcher = _default_fetch_json
    timeout_seconds: int = 10

    def build_prompt_context(self, user_text: str, *, route_mode: str = "chat") -> str | None:
        has_exchange_intent = _contains_any(user_text, EXCHANGE_KEYWORDS) or bool(
            re.search(r"\b[A-Za-z]{3}\s*(?:/|to|against)\s*[A-Za-z]{3}\b", user_text)
        )
        if not has_exchange_intent:
            return None

        pair = _extract_currency_pair(user_text)
        if pair is None:
            return None
        base_currency, quote_currency = pair

        try:
            response = self.fetch_json(
                f"https://open.er-api.com/v6/latest/{base_currency}",
                self.timeout_seconds,
            )
            rates = response["rates"]
            if quote_currency not in rates:
                raise KeyError(quote_currency)
        except Exception as exc:
            return (
                "Live exchange-rate lookup status:\n"
                f"- pair: {base_currency}/{quote_currency}\n"
                f"- failed: {exc.__class__.__name__}: {exc}\n"
                "- instruction: Do not invent the latest exchange rate."
            )

        rate = float(rates[quote_currency])
        lines = [
            "Live exchange-rate data:",
            f"- base_currency: {base_currency}",
            f"- quote_currency: {quote_currency}",
            f"- exchange_rate: {_format_number(rate)}",
            "- fetched_from: open.er-api.com",
        ]
        if response.get("time_last_update_utc"):
            lines.append(f"- last_update_utc: {response['time_last_update_utc']}")
        lines.append(
            "- instruction: Use this live exchange rate instead of model priors when answering conversion questions."
        )
        return "\n".join(lines)


@dataclass(slots=True)
class MarketPriceContextService:
    fetch_json: JsonFetcher = _default_fetch_json
    timeout_seconds: int = 10

    def build_prompt_context(self, user_text: str, *, route_mode: str = "chat") -> str | None:
        if not _contains_any(user_text, MARKET_KEYWORDS):
            return None

        symbols = _extract_market_symbols(user_text)
        if not symbols:
            return None

        try:
            response = self.fetch_json(
                "https://query1.finance.yahoo.com/v7/finance/quote?"
                + parse.urlencode({"symbols": ",".join(symbols)}),
                self.timeout_seconds,
            )
            results = response["quoteResponse"]["result"]
        except Exception as exc:
            return (
                "Live market lookup status:\n"
                f"- symbols: {', '.join(symbols)}\n"
                f"- failed: {exc.__class__.__name__}: {exc}\n"
                "- instruction: Do not invent the latest stock or crypto price."
            )

        if not results:
            return None

        lines = [
            "Live market price data:",
            f"- requested_symbols: {', '.join(symbols)}",
        ]
        for index, item in enumerate(results[:3], start=1):
            lines.extend(
                (
                    f"- asset_{index}_symbol: {item.get('symbol', 'Unknown')}",
                    f"- asset_{index}_name: {_collapse_whitespace(str(item.get('shortName', item.get('longName', 'Unknown'))))}",
                    f"- asset_{index}_price: {_format_number(float(item.get('regularMarketPrice', 0.0)))}",
                    f"- asset_{index}_currency: {item.get('currency', 'Unknown')}",
                )
            )
            if item.get("regularMarketChangePercent") is not None:
                lines.append(
                    f"- asset_{index}_change_percent: {_format_number(float(item['regularMarketChangePercent']))}"
                )
        lines.append(
            "- instruction: Use these live prices instead of model priors, and mention they may update throughout the day."
        )
        return "\n".join(lines)


@dataclass(slots=True)
class GeocodingContextService:
    fetch_json: JsonFetcher = _default_fetch_json
    timeout_seconds: int = 10

    def build_prompt_context(self, user_text: str, *, route_mode: str = "chat") -> str | None:
        if _contains_any(user_text, WEATHER_KEYWORDS):
            return None
        place = _extract_map_place(user_text)
        if place is None:
            return None

        return self.build_place_context(place)

    def build_place_context(self, place: str) -> str | None:
        normalized_place = _normalize_place_candidate(place)
        if normalized_place is None:
            return None

        try:
            geocode = _geocode_place(self.fetch_json, self.timeout_seconds, normalized_place)
        except Exception as exc:
            return (
                "Live geocoding lookup status:\n"
                f"- place: {normalized_place}\n"
                f"- failed: {exc.__class__.__name__}: {exc}\n"
                "- instruction: Do not invent coordinates or addresses."
            )

        return "\n".join(
            (
                "Live geocoding data:",
                f"- query: {normalized_place}",
                f"- location: {geocode['label']}",
                f"- latitude: {_format_number(float(geocode['latitude']))}",
                f"- longitude: {_format_number(float(geocode['longitude']))}",
                f"- country: {geocode['country']}",
                f"- timezone: {geocode['timezone']}",
                "- instruction: Use this geocoding data instead of guessing where the place is.",
            )
        )


@dataclass(slots=True)
class TimeContextService:
    fetch_json: JsonFetcher = _default_fetch_json
    timeout_seconds: int = 10

    def build_prompt_context(self, user_text: str, *, route_mode: str = "chat") -> str | None:
        if not _contains_any(user_text, TIME_KEYWORDS):
            return None

        if _contains_any(user_text, ("holiday", "holidays", "节假日", "假期")):
            country_code = _extract_country_code(user_text)
            if country_code is None:
                return None
            year = _extract_year(user_text) or datetime.now(UTC).year
            return self._build_holiday_context(country_code, year)

        place = _extract_time_place(user_text)
        if place is None:
            return None
        return self.build_time_context(place)

    def build_time_context(self, place: str) -> str | None:
        normalized_place = _normalize_place_candidate(place)
        if normalized_place is None:
            return None
        return self._build_time_context(normalized_place)

    def _build_time_context(self, place: str) -> str | None:
        try:
            geocode = _geocode_place(self.fetch_json, self.timeout_seconds, place)
            timezone = str(geocode["timezone"])
            now = datetime.now(ZoneInfo(timezone))
        except Exception as exc:
            return (
                "Live time lookup status:\n"
                f"- place: {place}\n"
                f"- failed: {exc.__class__.__name__}: {exc}\n"
                "- instruction: Do not invent the local time or timezone."
            )

        return "\n".join(
            (
                "Live time data:",
                f"- location: {geocode['label']}",
                f"- timezone: {timezone}",
                f"- local_datetime: {now.isoformat()}",
                f"- local_date: {now.date().isoformat()}",
                f"- local_time: {now.strftime('%H:%M:%S')}",
                "- instruction: Use this local time information instead of estimating from your priors.",
            )
        )

    def _build_holiday_context(self, country_code: str, year: int) -> str | None:
        try:
            response = self.fetch_json(
                f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}",
                self.timeout_seconds,
            )
        except Exception as exc:
            return (
                "Live holiday lookup status:\n"
                f"- country_code: {country_code}\n"
                f"- year: {year}\n"
                f"- failed: {exc.__class__.__name__}: {exc}\n"
                "- instruction: Do not invent public holiday calendars."
            )

        if not isinstance(response, list) or not response:
            return None

        lines = [
            "Live holiday calendar snapshot:",
            f"- country_code: {country_code}",
            f"- year: {year}",
        ]
        for index, item in enumerate(response[:3], start=1):
            lines.extend(
                (
                    f"- holiday_{index}_date: {item.get('date', 'Unknown')}",
                    f"- holiday_{index}_name: {_collapse_whitespace(str(item.get('localName', item.get('name', 'Unknown'))))}",
                )
            )
        lines.append(
            "- instruction: Present these as a quick holiday snapshot rather than a complete legal calendar."
        )
        return "\n".join(lines)


@dataclass(slots=True)
class NewsContextService:
    fetch_text: TextFetcher = _default_fetch_text
    timeout_seconds: int = 10
    max_items: int = 3

    def build_prompt_context(self, user_text: str, *, route_mode: str = "chat") -> str | None:
        if not _contains_any(user_text, NEWS_KEYWORDS):
            return None

        query = _extract_news_query(user_text)
        url = (
            "https://news.google.com/rss/search?"
            + parse.urlencode(
                {
                    "q": query,
                    "hl": "en-US",
                    "gl": "US",
                    "ceid": "US:en",
                }
            )
        )
        try:
            response = self.fetch_text(url, self.timeout_seconds)
            root = ElementTree.fromstring(response.text)
        except Exception as exc:
            return (
                "Live news lookup status:\n"
                f"- query: {query}\n"
                f"- failed: {exc.__class__.__name__}: {exc}\n"
                "- instruction: Do not pretend you have current news if the live lookup failed."
            )

        items = root.findall(".//channel/item")
        if not items:
            return None

        lines = [
            "Live news snapshot:",
            f"- query: {query}",
        ]
        for index, item in enumerate(items[: self.max_items], start=1):
            title = _collapse_whitespace(item.findtext("title", default="Unknown"))
            link = _collapse_whitespace(item.findtext("link", default=""))
            pub_date = _collapse_whitespace(item.findtext("pubDate", default=""))
            lines.append(f"- article_{index}_title: {title}")
            if pub_date:
                lines.append(f"- article_{index}_published: {pub_date}")
            if link:
                lines.append(f"- article_{index}_link: {link}")
        lines.append(
            "- instruction: Treat this as a lightweight current-news snapshot and mention uncertainty if needed."
        )
        return "\n".join(lines)


@dataclass(slots=True)
class SearchContextService:
    provider: str = "duckduckgo"
    fetch_json: JsonFetcher = _default_fetch_json
    fetch_text: TextFetcher = _default_fetch_text
    timeout_seconds: int = 10
    api_key: str | None = None
    engine: str = "google"
    max_results: int = 3

    def build_prompt_context(self, user_text: str, *, route_mode: str = "chat") -> str | None:
        normalized_query = user_text.strip()
        if normalized_query.startswith("/search "):
            normalized_query = normalized_query[len("/search ") :].strip()

        should_search = route_mode == "search" or _contains_any(normalized_query, SEARCH_KEYWORDS)
        if not should_search or not normalized_query or _extract_urls(normalized_query):
            return None

        return self.search_query(normalized_query).to_prompt_context()

    def search_query(self, query: str) -> SearchSnapshot:
        normalized_query = query.strip()
        if not normalized_query:
            return SearchSnapshot(
                query="",
                provider=self.provider,
                status="error",
                error_code="invalid_input",
                error_message="missing search query",
            )
        dispatch = {
            "duckduckgo": self._search_with_duckduckgo,
            "brave": self._search_with_brave,
            "serpapi": self._search_with_serpapi,
        }
        handler = dispatch.get(self.provider)
        if handler is None:
            return SearchSnapshot(
                query=normalized_query,
                provider=self.provider,
                status="error",
                error_code="unsupported_provider",
                error_message=f"unsupported search provider: {self.provider}",
            )
        return handler(normalized_query)

    def _search_with_serpapi(self, query: str) -> SearchSnapshot:
        if not self.api_key:
            return SearchSnapshot(
                query=query,
                provider="serpapi",
                status="error",
                error_code="missing_api_key",
                error_message="missing SerpAPI API key",
            )

        search_url = (
            "https://serpapi.com/search?"
            + parse.urlencode(
                {
                    "engine": self.engine,
                    "q": query,
                    "api_key": self.api_key,
                    "num": self.max_results,
                }
            )
        )
        try:
            response = self.fetch_json(search_url, self.timeout_seconds)
        except Exception as exc:
            return SearchSnapshot(
                query=query,
                provider="serpapi",
                status="error",
                error_code="unavailable",
                error_message=f"{exc.__class__.__name__}: {exc}",
            )

        summary, summary_url = self._extract_serpapi_summary(response)
        results = tuple(self._extract_organic_results(response.get("organic_results")))
        status = "ok" if results or summary else "no_results"
        return SearchSnapshot(
            query=query,
            provider="serpapi",
            status=status,
            summary=summary,
            summary_url=summary_url,
            results=results,
        )

    def _search_with_duckduckgo(self, query: str) -> SearchSnapshot:
        # ddgs currently depends on the native `primp` extension, which can abort
        # the interpreter on some Python 3.14 builds. Guard this path to avoid
        # crashing the whole CLI process.
        if sys.version_info >= (3, 14):
            allow_unsafe_runtime = os.getenv(_ALLOW_DDGS_ON_PY314_ENV, "").strip().lower()
            if allow_unsafe_runtime not in {"1", "true", "yes", "on"}:
                return SearchSnapshot(
                    query=query,
                    provider="duckduckgo",
                    status="error",
                    error_code="unsupported_runtime",
                    error_message=(
                        "duckduckgo search is disabled on Python 3.14+ to avoid a known "
                        "native extension crash (primp). Use /websearch brave or /websearch "
                        "serpapi, or run with Python 3.13. Set "
                        f"{_ALLOW_DDGS_ON_PY314_ENV}=1 to force-enable at your own risk."
                    ),
                )
        try:
            from ddgs import DDGS  # type: ignore[import-untyped]
        except ImportError:
            return SearchSnapshot(
                query=query,
                provider="duckduckgo",
                status="error",
                error_code="missing_dependency",
                error_message=(
                    "ddgs package is not installed; "
                    "run: pip install ddgs"
                ),
            )
        try:
            ddgs = DDGS()
            raw_results = list(ddgs.text(query, max_results=self.max_results))
        except Exception as exc:
            return SearchSnapshot(
                query=query,
                provider="duckduckgo",
                status="error",
                error_code="unavailable",
                error_message=f"{exc.__class__.__name__}: {exc}",
            )

        results = tuple(
            SearchResultItem(
                title=_collapse_whitespace(str(item.get("title", "Untitled"))),
                url=str(item.get("href", "")),
                snippet=_collapse_whitespace(str(item.get("body", ""))),
                source=_search_result_source(str(item.get("href", ""))),
            )
            for item in raw_results
            if item.get("title") or item.get("href")
        )
        status = "ok" if results else "no_results"
        return SearchSnapshot(
            query=query,
            provider="duckduckgo",
            status=status,
            results=results,
        )

    def _search_with_brave(self, query: str) -> SearchSnapshot:
        if not self.api_key:
            return SearchSnapshot(
                query=query,
                provider="brave",
                status="error",
                error_code="missing_api_key",
                error_message="missing Brave Search API key",
            )

        search_url = (
            "https://api.search.brave.com/res/v1/web/search?"
            + parse.urlencode({"q": query, "count": self.max_results})
        )
        req = request.Request(
            search_url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return SearchSnapshot(
                query=query,
                provider="brave",
                status="error",
                error_code="unavailable",
                error_message=f"{exc.__class__.__name__}: {exc}",
            )

        results = tuple(self._extract_brave_api_results(response))
        status = "ok" if results else "no_results"
        return SearchSnapshot(
            query=query,
            provider="brave",
            status=status,
            results=results,
        )

    def _extract_serpapi_summary(self, response: dict[str, object]) -> tuple[str | None, str | None]:
        answer_box = response.get("answer_box")
        if isinstance(answer_box, dict):
            answer = _collapse_whitespace(
                str(
                    answer_box.get("answer")
                    or answer_box.get("snippet")
                    or answer_box.get("title")
                    or ""
                )
            )
            answer_link = _collapse_whitespace(str(answer_box.get("link", ""))) or None
            if answer:
                return answer, answer_link

        knowledge_graph = response.get("knowledge_graph")
        if isinstance(knowledge_graph, dict):
            title = _collapse_whitespace(str(knowledge_graph.get("title", "")))
            description = _collapse_whitespace(str(knowledge_graph.get("description", "")))
            website = _collapse_whitespace(str(knowledge_graph.get("website", ""))) or None
            summary = _collapse_whitespace(" - ".join(part for part in (title, description) if part))
            if summary:
                return summary, website
        return None, None

    def _extract_organic_results(self, raw_results: object) -> list[SearchResultItem]:
        if not isinstance(raw_results, list):
            return []
        results: list[SearchResultItem] = []
        for item in raw_results:
            if len(results) >= self.max_results:
                break
            if not isinstance(item, dict):
                continue
            title = _collapse_whitespace(str(item.get("title", "")))
            snippet = _collapse_whitespace(str(item.get("snippet", "")))
            link = _collapse_whitespace(str(item.get("link", "")))
            if not title and not snippet and not link:
                continue
            results.append(
                SearchResultItem(
                    title=title or "Untitled result",
                    url=link,
                    snippet=snippet,
                    source=_search_result_source(link),
                )
            )
        return results

    def _extract_brave_api_results(self, response: dict[str, object]) -> list[SearchResultItem]:
        web = response.get("web")
        if not isinstance(web, dict):
            return []
        raw_results = web.get("results")
        if not isinstance(raw_results, list):
            return []

        results: list[SearchResultItem] = []
        for item in raw_results:
            if len(results) >= self.max_results:
                break
            if not isinstance(item, dict):
                continue
            title = _collapse_whitespace(str(item.get("title", "")))
            url = str(item.get("url", ""))
            snippet = _collapse_whitespace(str(item.get("description", "")))
            page_age = str(item.get("page_age", "")).strip() or None
            if not title or not url:
                continue
            results.append(
                SearchResultItem(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source=_search_result_source(url),
                    page_age=page_age,
                )
            )
        return results

def _search_result_source(url: str) -> str:
    try:
        return parse.urlparse(url).netloc
    except ValueError:
        return ""


def _split_top_level_js_objects(raw: str) -> list[str]:
    items: list[str] = []
    depth = 0
    start_index: int | None = None
    in_string = False
    escaped = False
    for index, char in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0 and start_index is not None:
                items.append(raw[start_index : index + 1])
                start_index = None
    return items


def _extract_js_string_field(
    raw: str,
    field_name: str,
    *,
    strip_tags: bool = True,
) -> str | None:
    match = re.search(
        rf'{re.escape(field_name)}:"((?:[^"\\]|\\.)*)"',
        raw,
    )
    if match is None:
        return None
    value = _decode_js_string(match.group(1))
    value = html_unescape(value)
    if strip_tags:
        value = re.sub(r"<[^>]+>", "", value)
    value = _collapse_whitespace(value)
    return value or None


def _decode_js_string(raw: str) -> str:
    try:
        return str(json.loads(f'"{raw}"'))
    except json.JSONDecodeError:
        return raw


@dataclass(slots=True)
class TransportStatusContextService:
    endpoint_configs: dict[str, LiveContextEndpointConfig]
    fetch_json: HeaderJsonFetcher = _default_fetch_json_with_headers
    fetch_text: HeaderTextFetcher = _default_fetch_text_with_headers

    def build_prompt_context(self, user_text: str, *, route_mode: str = "chat") -> str | None:
        transport_targets = (
            ("flight", FLIGHT_KEYWORDS, (FLIGHT_CODE_PATTERN,)),
            ("train", TRAIN_KEYWORDS, (TRAIN_CODE_PATTERN,)),
            ("parcel", PARCEL_KEYWORDS, (TRACKING_CODE_PATTERN,)),
        )
        for name, keywords, patterns in transport_targets:
            if not _contains_any(user_text, keywords):
                continue
            endpoint = self.endpoint_configs.get(name)
            if endpoint is None:
                return None
            code = _extract_status_code(user_text, patterns)
            if code is None:
                return None
            return _build_endpoint_context(
                endpoint,
                query=code,
                fetch_json=self.fetch_json,
                fetch_text=self.fetch_text,
                title=f"Live {name} status data:",
            )
        return None


@dataclass(slots=True)
class CustomApiContextService:
    endpoint_configs: dict[str, LiveContextEndpointConfig]
    fetch_json: HeaderJsonFetcher = _default_fetch_json_with_headers
    fetch_text: HeaderTextFetcher = _default_fetch_text_with_headers

    def build_prompt_context(self, user_text: str, *, route_mode: str = "chat") -> str | None:
        lowered = user_text.casefold()
        for endpoint in self.endpoint_configs.values():
            if endpoint.trigger_keywords and not any(
                keyword.casefold() in lowered for keyword in endpoint.trigger_keywords
            ):
                continue
            return _build_endpoint_context(
                endpoint,
                query=user_text.strip(),
                fetch_json=self.fetch_json,
                fetch_text=self.fetch_text,
                title=f"Live custom API data from {endpoint.name}:",
            )
        return None


def _weather_context_lines(
    title: str,
    report: WeatherReport,
    *,
    include_ip: bool,
    user_text: str,
) -> str:
    lines = [
        title,
        f"- fetched_at_utc: {report.fetched_at_utc}",
        f"- location: {report.location_label}",
        f"- latitude: {_format_number(report.latitude)}",
        f"- longitude: {_format_number(report.longitude)}",
        f"- timezone: {report.timezone}",
    ]
    if include_ip and report.public_ip:
        lines.append(f"- public_ip: {report.public_ip}")
    lines.extend(
        (
            f"- current_condition: {report.current_condition}",
            f"- current_temperature_c: {_format_number(report.current_temperature_c)}",
            f"- apparent_temperature_c: {_format_number(report.apparent_temperature_c)}",
            f"- relative_humidity_percent: {_format_number(report.relative_humidity_percent)}",
            f"- wind_speed_kmh: {_format_number(report.wind_speed_kmh)}",
            f"- today_condition: {report.today_condition}",
            f"- today_high_c: {_format_number(report.today_high_c)}",
            f"- today_low_c: {_format_number(report.today_low_c)}",
            f"- today_precipitation_probability_max_percent: {_format_number(report.today_precipitation_probability_max_percent)}",
        )
    )
    if report.tomorrow_condition is not None:
        lines.extend(
            (
                f"- tomorrow_condition: {report.tomorrow_condition}",
                f"- tomorrow_high_c: {_format_number(report.tomorrow_high_c)}",
                f"- tomorrow_low_c: {_format_number(report.tomorrow_low_c)}",
                f"- tomorrow_precipitation_probability_max_percent: {_format_number(report.tomorrow_precipitation_probability_max_percent)}",
            )
        )
    request_window = _extract_weather_request_window(
        user_text,
        reference_date=_weather_reference_date(report),
    )
    if request_window is not None:
        start_index, end_index, label = request_window
        selected = report.daily_forecasts[start_index : end_index + 1]
        if selected:
            lines.append(f"- requested_period: {label}")
            if len(selected) == 1:
                forecast = selected[0]
                lines.extend(
                    (
                        f"- requested_date: {forecast.date}",
                        f"- requested_condition: {forecast.condition}",
                        f"- requested_high_c: {_format_number(forecast.high_c)}",
                        f"- requested_low_c: {_format_number(forecast.low_c)}",
                        f"- requested_precipitation_probability_max_percent: {_format_number(forecast.precipitation_probability_max_percent)}",
                    )
                )
            else:
                for index, forecast in enumerate(selected, start=1):
                    lines.extend(
                        (
                            f"- requested_day_{index}_date: {forecast.date}",
                            f"- requested_day_{index}_condition: {forecast.condition}",
                            f"- requested_day_{index}_high_c: {_format_number(forecast.high_c)}",
                            f"- requested_day_{index}_low_c: {_format_number(forecast.low_c)}",
                            f"- requested_day_{index}_precipitation_probability_max_percent: {_format_number(forecast.precipitation_probability_max_percent)}",
                        )
                    )
    lines.append(
        "- instruction: Use this live weather data instead of model priors when answering weather questions. If requested_* fields are present, prefer them because they match the user's requested day or period."
    )
    return "\n".join(lines)


def _weather_reference_date(report: WeatherReport) -> date | None:
    if report.daily_forecasts:
        try:
            return date.fromisoformat(report.daily_forecasts[0].date)
        except ValueError:
            return None
    return None


def _extract_weather_request_window(
    user_text: str,
    *,
    reference_date: date | None,
) -> tuple[int, int, str] | None:
    lowered = user_text.casefold()

    if "day after tomorrow" in lowered or "后天" in user_text:
        return (2, 2, "day_after_tomorrow")
    if "tomorrow" in lowered or "明天" in user_text:
        return (1, 1, "tomorrow")
    if "today" in lowered or "今晚" in user_text or "今天" in user_text:
        return (0, 0, "today")

    if reference_date is None:
        return None

    weekend = _extract_weekend_window(user_text, reference_date)
    if weekend is not None:
        return weekend

    weekday = _extract_weekday_window(user_text, reference_date)
    if weekday is not None:
        return weekday

    return None


def _extract_weekend_window(user_text: str, reference_date: date) -> tuple[int, int, str] | None:
    lowered = user_text.casefold()
    has_weekend = any(keyword.casefold() in lowered for keyword in WEEKEND_KEYWORDS)
    if not has_weekend:
        return None

    weekday = reference_date.weekday()
    saturday_offset = (5 - weekday) % 7
    if "next weekend" in lowered or "下周末" in user_text:
        saturday_offset += 7
        label = "next_weekend"
    else:
        label = "weekend"
    sunday_offset = saturday_offset + 1
    return (saturday_offset, sunday_offset, label)


def _extract_weekday_window(user_text: str, reference_date: date) -> tuple[int, int, str] | None:
    lowered = user_text.casefold()
    for target_weekday, aliases in WEEKDAY_KEYWORDS:
        matched_alias = next(
            (alias for alias in sorted(aliases, key=len, reverse=True) if alias.casefold() in lowered),
            None,
        )
        if matched_alias is None:
            continue

        offset = (target_weekday - reference_date.weekday()) % 7
        english_name = _weekday_label(target_weekday)
        is_explicit_next = (
            matched_alias.startswith("下周")
            or f"next {english_name}" in lowered
        )
        if is_explicit_next and offset == 0:
            offset = 7
        label = _weekday_label(target_weekday, prefix="next_" if is_explicit_next else "")
        return (offset, offset, label)
    return None


def _weekday_label(weekday: int, *, prefix: str = "") -> str:
    return f"{prefix}{WEEKDAY_NAMES[weekday]}"


def lookup_public_ip(fetch_json: JsonFetcher, timeout_seconds: int) -> str:
    ip_response = fetch_json("https://api.ipify.org?format=json", timeout_seconds)
    public_ip = ip_response.get("ip")
    if not isinstance(public_ip, str) or not public_ip.strip():
        raise RuntimeError("IP lookup returned no usable IP address")
    return public_ip.strip()


def lookup_ip_location(
    fetch_json: JsonFetcher,
    timeout_seconds: int,
    public_ip: str,
) -> IpLocation:
    geo_response = fetch_json(f"https://ipwho.is/{parse.quote(public_ip)}", timeout_seconds)
    if geo_response.get("success") is False:
        detail = str(geo_response.get("message", "unknown geolocation error"))
        raise RuntimeError(f"IP geolocation failed: {detail}")

    return IpLocation(
        public_ip=public_ip,
        city=str(geo_response.get("city", "")),
        region=str(geo_response.get("region", "")),
        country=str(geo_response.get("country", "")),
        latitude=float(geo_response["latitude"]),
        longitude=float(geo_response["longitude"]),
        timezone=str(geo_response.get("timezone", {}).get("id", "auto")),
    )


def lookup_local_area(fetch_json: JsonFetcher, timeout_seconds: int) -> IpLocation:
    public_ip = lookup_public_ip(fetch_json, timeout_seconds)
    return lookup_ip_location(fetch_json, timeout_seconds, public_ip)


def lookup_weather_for_ip(fetch_json: JsonFetcher, timeout_seconds: int) -> WeatherReport:
    location = lookup_local_area(fetch_json, timeout_seconds)
    return _fetch_weather_report(
        fetch_json,
        timeout_seconds,
        latitude=location.latitude,
        longitude=location.longitude,
        timezone=location.timezone,
        location_label=location.label,
        public_ip=location.public_ip,
    )


def lookup_weather_for_place(fetch_json: JsonFetcher, timeout_seconds: int, place: str) -> WeatherReport:
    geocode = _geocode_place(fetch_json, timeout_seconds, place)
    return _fetch_weather_report(
        fetch_json,
        timeout_seconds,
        latitude=float(geocode["latitude"]),
        longitude=float(geocode["longitude"]),
        timezone=str(geocode["timezone"]),
        location_label=str(geocode["label"]),
    )


def _geocode_place(fetch_json: JsonFetcher, timeout_seconds: int, place: str) -> dict[str, object]:
    attempted_languages: list[str] = []
    for language in _geocoding_language_candidates(place):
        response = fetch_json(
            "https://geocoding-api.open-meteo.com/v1/search?"
            + parse.urlencode(
                {"name": place, "count": 1, "language": language, "format": "json"}
            ),
            timeout_seconds,
        )
        results = response.get("results")
        attempted_languages.append(language)
        if not isinstance(results, list) or not results:
            continue
        first = results[0]
        name = str(first.get("name", place))
        admin1 = str(first.get("admin1", ""))
        country = str(first.get("country", ""))
        label = ", ".join(part for part in (name, admin1, country) if part)
        return {
            "label": label or name,
            "country": country or "Unknown",
            "latitude": float(first["latitude"]),
            "longitude": float(first["longitude"]),
            "timezone": str(first.get("timezone", "UTC")),
        }
    tried = ", ".join(attempted_languages) or "none"
    raise RuntimeError(f"No geocoding result for {place!r} across languages: {tried}")


def _fetch_weather_report(
    fetch_json: JsonFetcher,
    timeout_seconds: int,
    *,
    latitude: float,
    longitude: float,
    timezone: str,
    location_label: str,
    public_ip: str | None = None,
) -> WeatherReport:
    wttr_error: Exception | None = None
    try:
        weather_response = _fetch_wttr_weather_response(
            fetch_json,
            timeout_seconds,
            latitude=latitude,
            longitude=longitude,
            location_label=location_label,
        )
        return _weather_report_from_wttr_response(
            weather_response,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone,
            location_label=location_label,
            public_ip=public_ip,
        )
    except Exception as exc:
        wttr_error = exc

    try:
        return _fetch_open_meteo_weather_report(
            fetch_json,
            timeout_seconds,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone,
            location_label=location_label,
            public_ip=public_ip,
        )
    except Exception as exc:
        raise RuntimeError(
            "Weather lookup failed. "
            f"wttr.in error: {wttr_error.__class__.__name__}: {wttr_error}. "
            f"open-meteo fallback error: {exc.__class__.__name__}: {exc}"
        ) from exc


def _weather_report_from_wttr_response(
    weather_response: dict[str, object],
    *,
    latitude: float,
    longitude: float,
    timezone: str,
    location_label: str,
    public_ip: str | None = None,
) -> WeatherReport:
    current = _wttr_current_condition(weather_response)
    daily_forecasts = _wttr_daily_forecasts(weather_response)
    return _weather_report_from_values(
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        location_label=location_label,
        current_condition=_wttr_desc(current.get("weatherDesc")) or "Unknown",
        current_temperature_c=_wttr_float(current.get("temp_C")),
        apparent_temperature_c=_wttr_float(current.get("FeelsLikeC")),
        relative_humidity_percent=_wttr_float(current.get("humidity")),
        wind_speed_kmh=_wttr_float(current.get("windspeedKmph")),
        daily_forecasts=daily_forecasts,
        public_ip=public_ip,
    )


def _fetch_open_meteo_weather_report(
    fetch_json: JsonFetcher,
    timeout_seconds: int,
    *,
    latitude: float,
    longitude: float,
    timezone: str,
    location_label: str,
    public_ip: str | None = None,
) -> WeatherReport:
    response = fetch_json(
        _open_meteo_forecast_url(
            latitude=latitude,
            longitude=longitude,
            timezone=timezone,
        ),
        timeout_seconds,
    )
    if not isinstance(response, dict):
        raise RuntimeError("open-meteo forecast returned non-object JSON payload.")
    current = _open_meteo_current(response)
    daily_forecasts = _open_meteo_daily_forecasts(response)
    return _weather_report_from_values(
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        location_label=location_label,
        current_condition=_weather_label(current["weather_code"]),
        current_temperature_c=float(current["temperature_2m"]),
        apparent_temperature_c=float(current["apparent_temperature"]),
        relative_humidity_percent=float(current["relative_humidity_2m"]),
        wind_speed_kmh=float(current["wind_speed_10m"]),
        daily_forecasts=daily_forecasts,
        public_ip=public_ip,
    )


def _weather_report_from_values(
    *,
    latitude: float,
    longitude: float,
    timezone: str,
    location_label: str,
    current_condition: str,
    current_temperature_c: float,
    apparent_temperature_c: float,
    relative_humidity_percent: float,
    wind_speed_kmh: float,
    daily_forecasts: tuple[WeatherDailyForecast, ...],
    public_ip: str | None = None,
) -> WeatherReport:
    today_forecast = daily_forecasts[0] if daily_forecasts else None
    tomorrow_forecast = daily_forecasts[1] if len(daily_forecasts) > 1 else None
    return WeatherReport(
        fetched_at_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
        location_label=location_label,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        current_condition=current_condition,
        current_temperature_c=current_temperature_c,
        apparent_temperature_c=apparent_temperature_c,
        relative_humidity_percent=relative_humidity_percent,
        wind_speed_kmh=wind_speed_kmh,
        today_condition=(
            today_forecast.condition
            if today_forecast is not None
            else "Unknown"
        ),
        today_high_c=(
            today_forecast.high_c
            if today_forecast is not None
            else 0.0
        ),
        today_low_c=(
            today_forecast.low_c
            if today_forecast is not None
            else 0.0
        ),
        today_precipitation_probability_max_percent=(
            today_forecast.precipitation_probability_max_percent
            if today_forecast is not None
            else 0.0
        ),
        tomorrow_condition=(
            tomorrow_forecast.condition if tomorrow_forecast is not None else None
        ),
        tomorrow_high_c=(
            tomorrow_forecast.high_c if tomorrow_forecast is not None else None
        ),
        tomorrow_low_c=(
            tomorrow_forecast.low_c if tomorrow_forecast is not None else None
        ),
        tomorrow_precipitation_probability_max_percent=(
            tomorrow_forecast.precipitation_probability_max_percent
            if tomorrow_forecast is not None
            else None
        ),
        daily_forecasts=daily_forecasts,
        public_ip=public_ip,
    )


def _fetch_wttr_weather_response(
    fetch_json: JsonFetcher,
    timeout_seconds: int,
    *,
    latitude: float,
    longitude: float,
    location_label: str,
) -> dict[str, object]:
    errors: list[str] = []
    for _ in range(WTTR_FETCH_ATTEMPTS):
        for url in _wttr_weather_urls(
            latitude=latitude,
            longitude=longitude,
            location_label=location_label,
        ):
            try:
                weather_response = fetch_json(url, timeout_seconds)
            except Exception as exc:
                errors.append(f"{url}: {exc.__class__.__name__}: {exc}")
                continue
            if not isinstance(weather_response, dict):
                errors.append(f"{url}: returned non-object JSON payload")
                continue
            try:
                _wttr_current_condition(weather_response)
                _wttr_daily_forecasts(weather_response)
            except Exception as exc:
                errors.append(f"{url}: {exc}")
                continue
            return weather_response
    detail = errors[-1] if errors else "unknown wttr.in error"
    raise RuntimeError(
        f"wttr.in weather lookup failed after {WTTR_FETCH_ATTEMPTS} attempts: {detail}"
    )


def _wttr_weather_urls(
    *,
    latitude: float,
    longitude: float,
    location_label: str,
) -> tuple[str, ...]:
    place = location_label.split(",", 1)[0].strip()
    urls = [
        _wttr_weather_url(latitude=latitude, longitude=longitude, metric=True),
        _wttr_weather_url(latitude=latitude, longitude=longitude, metric=False),
    ]
    if place:
        urls.extend(
            (
                _wttr_place_weather_url(place, metric=True),
                _wttr_place_weather_url(place, metric=False),
            )
        )
    return tuple(dict.fromkeys(urls))


def _wttr_weather_url(*, latitude: float, longitude: float, metric: bool = True) -> str:
    coordinates = f"{_format_number(latitude)},{_format_number(longitude)}"
    suffix = "?format=j1&m" if metric else "?format=j1"
    return f"{WTTR_BASE_URL}/{parse.quote(coordinates, safe=',')}{suffix}"


def _wttr_place_weather_url(place: str, *, metric: bool = True) -> str:
    suffix = "?format=j1&m" if metric else "?format=j1"
    return f"{WTTR_BASE_URL}/{parse.quote(place)}{suffix}"


def _open_meteo_forecast_url(*, latitude: float, longitude: float, timezone: str) -> str:
    return "https://api.open-meteo.com/v1/forecast?" + parse.urlencode(
        {
            "latitude": _format_number(latitude),
            "longitude": _format_number(longitude),
            "timezone": timezone,
            "forecast_days": WEATHER_FORECAST_DAYS,
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "current": ",".join(
                (
                    "temperature_2m",
                    "apparent_temperature",
                    "relative_humidity_2m",
                    "wind_speed_10m",
                    "weather_code",
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
        }
    )


def _open_meteo_current(response: dict[str, object]) -> dict[str, float | int]:
    current = response.get("current")
    if not isinstance(current, dict):
        raise RuntimeError("open-meteo forecast did not include current weather data.")
    parsed: dict[str, float | int] = {}
    for key in (
        "temperature_2m",
        "apparent_temperature",
        "relative_humidity_2m",
        "wind_speed_10m",
    ):
        value = current.get(key)
        if not isinstance(value, (int, float)):
            raise RuntimeError(f"open-meteo forecast did not include {key}.")
        parsed[key] = float(value)
    weather_code = current.get("weather_code")
    if not isinstance(weather_code, (int, float)):
        raise RuntimeError("open-meteo forecast did not include weather_code.")
    parsed["weather_code"] = int(weather_code)
    return parsed


def _open_meteo_daily_forecasts(response: dict[str, object]) -> tuple[WeatherDailyForecast, ...]:
    daily = response.get("daily")
    if not isinstance(daily, dict):
        raise RuntimeError("open-meteo forecast did not include daily forecast data.")
    dates = daily.get("time")
    weather_codes = daily.get("weather_code")
    highs = daily.get("temperature_2m_max")
    lows = daily.get("temperature_2m_min")
    precipitations = daily.get("precipitation_probability_max")
    if not all(isinstance(values, list) for values in (dates, weather_codes, highs, lows, precipitations)):
        raise RuntimeError("open-meteo forecast daily payload was incomplete.")

    forecasts: list[WeatherDailyForecast] = []
    forecast_count = min(len(dates), len(weather_codes), len(highs), len(lows), len(precipitations))
    for index in range(forecast_count):
        date_value = dates[index]
        weather_code = weather_codes[index]
        high = highs[index]
        low = lows[index]
        precipitation = precipitations[index]
        if not isinstance(date_value, str):
            continue
        if not isinstance(weather_code, (int, float)):
            continue
        if not isinstance(high, (int, float)) or not isinstance(low, (int, float)):
            continue
        if not isinstance(precipitation, (int, float)):
            continue
        forecasts.append(
            WeatherDailyForecast(
                date=date_value,
                condition=_weather_label(int(weather_code)),
                high_c=float(high),
                low_c=float(low),
                precipitation_probability_max_percent=float(precipitation),
            )
        )
    if not forecasts:
        raise RuntimeError("open-meteo forecast daily payload did not contain usable forecast rows.")
    return tuple(forecasts)


def _geocoding_language_candidates(place: str) -> tuple[str, ...]:
    candidates: list[str] = []
    if re.search(r"[\u3040-\u30ff]", place):
        candidates.append("ja")
    if re.search(r"[\uac00-\ud7af]", place):
        candidates.append("ko")
    if re.search(r"[\u0600-\u06ff]", place):
        candidates.append("ar")
    if re.search(r"[\u0400-\u04ff]", place):
        candidates.append("ru")
    if re.search(r"[\u4e00-\u9fff]", place):
        candidates.extend(("zh", "ja"))
    if re.search(r"[A-Za-z]", place):
        candidates.append("en")
    if not candidates:
        candidates.append("en")

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates + ["en"]:
        if candidate in seen:
            continue
        deduped.append(candidate)
        seen.add(candidate)
    return tuple(deduped)


def _wttr_current_condition(weather_response: dict[str, object]) -> dict[str, object]:
    current = weather_response.get("current_condition")
    if isinstance(current, list) and current and isinstance(current[0], dict):
        return current[0]
    raise RuntimeError("wttr.in response did not include current_condition data.")


def _wttr_daily_forecasts(weather_response: dict[str, object]) -> tuple[WeatherDailyForecast, ...]:
    raw_weather = weather_response.get("weather")
    if not isinstance(raw_weather, list) or not raw_weather:
        raise RuntimeError("wttr.in response did not include daily forecast data.")
    forecasts: list[WeatherDailyForecast] = []
    for item in raw_weather:
        if not isinstance(item, dict):
            continue
        hourly = item.get("hourly")
        forecasts.append(
            WeatherDailyForecast(
                date=str(item.get("date", "")),
                condition=_wttr_daily_condition(hourly),
                high_c=_wttr_float(item.get("maxtempC")),
                low_c=_wttr_float(item.get("mintempC")),
                precipitation_probability_max_percent=_wttr_daily_precipitation_probability(hourly),
            )
        )
    if not forecasts:
        raise RuntimeError("wttr.in response did not include any usable daily forecasts.")
    return tuple(forecasts)


def _wttr_daily_condition(hourly: object) -> str:
    selected = _wttr_hourly_snapshot(hourly)
    return _wttr_desc(selected.get("weatherDesc")) or "Unknown"


def _wttr_daily_precipitation_probability(hourly: object) -> float:
    if not isinstance(hourly, list) or not hourly:
        return 0.0
    probabilities = [
        _wttr_hourly_precipitation_probability(item)
        for item in hourly
        if isinstance(item, dict)
    ]
    if not probabilities:
        return 0.0
    return max(probabilities)


def _wttr_hourly_snapshot(hourly: object) -> dict[str, object]:
    if not isinstance(hourly, list) or not hourly:
        return {}
    snapshots = [item for item in hourly if isinstance(item, dict)]
    if not snapshots:
        return {}
    selected = snapshots[0]
    best_distance = abs(_wttr_hour_time(selected) - 1200)
    for item in snapshots[1:]:
        distance = abs(_wttr_hour_time(item) - 1200)
        if distance < best_distance:
            selected = item
            best_distance = distance
    return selected


def _wttr_hour_time(snapshot: dict[str, object]) -> int:
    value = snapshot.get("time")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _wttr_hourly_precipitation_probability(snapshot: dict[str, object]) -> float:
    return max(
        _wttr_float(snapshot.get("chanceofrain")),
        _wttr_float(snapshot.get("chanceofsnow")),
        _wttr_float(snapshot.get("chanceofthunder")),
    )


def _wttr_desc(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            nested = first.get("value")
            if isinstance(nested, str):
                return nested.strip()
        if isinstance(first, str):
            return first.strip()
    return ""


def _wttr_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _build_endpoint_context(
    endpoint: LiveContextEndpointConfig,
    *,
    query: str,
    fetch_json: HeaderJsonFetcher,
    fetch_text: HeaderTextFetcher,
    title: str,
) -> str:
    url = _resolve_endpoint_url(endpoint, query)
    headers = _resolve_endpoint_headers(endpoint)
    try:
        if endpoint.response_format == "json":
            response = fetch_json(url, endpoint.timeout_seconds, headers)
            payload_text = _truncate_text(json.dumps(response, ensure_ascii=False), limit=480)
        else:
            response = fetch_text(url, endpoint.timeout_seconds, headers)
            payload_text = _truncate_text(response.text, limit=480)
    except Exception as exc:
        return (
            f"{title}\n"
            f"- query: {query}\n"
            f"- failed: {exc.__class__.__name__}: {exc}\n"
            "- instruction: Do not fabricate data from this endpoint."
        )

    instruction = endpoint.instruction or "Use this endpoint payload only when it directly answers the user's request."
    return "\n".join(
        (
            title,
            f"- endpoint_name: {endpoint.name}",
            f"- query: {query}",
            f"- response_format: {endpoint.response_format}",
            f"- payload: {payload_text}",
            f"- instruction: {instruction}",
        )
    )


def _resolve_endpoint_url(endpoint: LiveContextEndpointConfig, query: str) -> str:
    if "{query}" in endpoint.url_template:
        return endpoint.url_template.replace("{query}", parse.quote(query))
    if endpoint.query_param:
        parts = parse.urlsplit(endpoint.url_template)
        existing = parse.parse_qsl(parts.query, keep_blank_values=True)
        existing.append((endpoint.query_param, query))
        return parse.urlunsplit(
            (
                parts.scheme,
                parts.netloc,
                parts.path,
                parse.urlencode(existing),
                parts.fragment,
            )
        )
    return endpoint.url_template


def _resolve_endpoint_headers(endpoint: LiveContextEndpointConfig) -> dict[str, str]:
    headers = dict(endpoint.extra_headers)
    if endpoint.api_key_env:
        from os import getenv

        api_key = getenv(endpoint.api_key_env)
        if api_key:
            headers[endpoint.api_key_header] = f"{endpoint.api_key_prefix}{api_key}"
    return headers
