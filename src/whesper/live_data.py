from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
import json
import re
from zoneinfo import ZoneInfo
from typing import Callable, Protocol
from urllib import parse, request
from xml.etree import ElementTree

from whesper.config import AppConfig, LiveContextEndpointConfig


USER_AGENT = "Whesper/0.1"
URL_PATTERN = re.compile(r"https?://[^\s<>\"]+")
CITY_PATTERN = re.compile(
    r"(?P<place>[A-Za-z][A-Za-z .'-]{1,40}|[\u4e00-\u9fff][\u4e00-\u9fffA-Za-z·\-\s]{0,20})"
)
FLIGHT_CODE_PATTERN = re.compile(r"\b([A-Z]{2,3}\s?\d{2,4})\b")
TRAIN_CODE_PATTERN = re.compile(r"\b([GDCZKTLSY]\d{1,4})\b", re.IGNORECASE)
TRACKING_CODE_PATTERN = re.compile(r"\b([A-Z0-9]{8,22})\b")


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
    "what time",
    "timezone",
    "time difference",
    "holiday",
    "holidays",
    "几点",
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
    public_ip: str | None = None


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
    cleaned = re.sub(r"^(in|for|at)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.removeprefix("在").removeprefix("去")
    cleaned = cleaned.strip()
    if not cleaned:
        return None
    lowered = cleaned.casefold()
    if lowered.startswith(STOPWORD_LOCATION_PREFIXES):
        return None
    if cleaned.casefold() in STOPWORD_LOCATIONS:
        return None
    return cleaned


def should_fetch_local_weather(user_text: str) -> bool:
    lowered = user_text.casefold()
    has_weather_keyword = any(keyword in lowered for keyword in WEATHER_KEYWORDS)
    has_local_time_keyword = any(keyword in lowered for keyword in LOCAL_TIME_KEYWORDS)
    return has_weather_keyword and has_local_time_keyword


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
        r"(?P<place>[\u4e00-\u9fffA-Za-z·\-\s]{1,20})(?:几点|时间|时区|时差)",
    ):
        match = re.search(pattern, user_text, re.IGNORECASE)
        if match:
            candidate = _normalize_place_candidate(match.group("place"))
            if candidate:
                return candidate
    return None


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
                SearchContextService(),
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
    )

    return _AppConfig(
        app=AppSettings(),
        persona=PersonaConfig(),
        scheduler=SchedulerConfig(chat_model="chat"),
        live_context=LiveContextSettings(),
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
            "Live weather data for the user's current IP-based location:",
            report,
            include_ip=True,
        )

    def _lookup_weather_report(self) -> WeatherReport:
        ip_response = self.fetch_json("https://api.ipify.org?format=json", self.timeout_seconds)
        public_ip = str(ip_response["ip"])
        geo_response = self.fetch_json(f"https://ipwho.is/{parse.quote(public_ip)}", self.timeout_seconds)
        if geo_response.get("success") is False:
            detail = str(geo_response.get("message", "unknown geolocation error"))
            raise RuntimeError(f"IP geolocation failed: {detail}")

        location = ", ".join(
            part
            for part in (
                str(geo_response.get("city", "")),
                str(geo_response.get("region", "")),
                str(geo_response.get("country", "")),
            )
            if part
        )
        return _fetch_weather_report(
            self.fetch_json,
            self.timeout_seconds,
            latitude=float(geo_response["latitude"]),
            longitude=float(geo_response["longitude"]),
            timezone=str(geo_response.get("timezone", {}).get("id", "auto")),
            location_label=location or "Unknown",
            public_ip=public_ip,
        )


@dataclass(slots=True)
class CityWeatherContextService:
    fetch_json: JsonFetcher = _default_fetch_json
    timeout_seconds: int = 10

    def build_prompt_context(self, user_text: str, *, route_mode: str = "chat") -> str | None:
        place = _extract_city_weather_place(user_text)
        if place is None:
            return None

        try:
            geocode = _geocode_place(self.fetch_json, self.timeout_seconds, place)
            report = _fetch_weather_report(
                self.fetch_json,
                self.timeout_seconds,
                latitude=geocode["latitude"],
                longitude=geocode["longitude"],
                timezone=geocode["timezone"],
                location_label=str(geocode["label"]),
            )
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

        try:
            geocode = _geocode_place(self.fetch_json, self.timeout_seconds, place)
        except Exception as exc:
            return (
                "Live geocoding lookup status:\n"
                f"- place: {place}\n"
                f"- failed: {exc.__class__.__name__}: {exc}\n"
                "- instruction: Do not invent coordinates or addresses."
            )

        return "\n".join(
            (
                "Live geocoding data:",
                f"- query: {place}",
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
        return self._build_time_context(place)

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
    fetch_json: JsonFetcher = _default_fetch_json
    timeout_seconds: int = 10
    max_related_topics: int = 3

    def build_prompt_context(self, user_text: str, *, route_mode: str = "chat") -> str | None:
        normalized_query = user_text.strip()
        if normalized_query.startswith("/search "):
            normalized_query = normalized_query[len("/search ") :].strip()

        should_search = route_mode == "search" or _contains_any(normalized_query, SEARCH_KEYWORDS)
        if not should_search or not normalized_query or _extract_urls(normalized_query):
            return None

        search_url = (
            "https://api.duckduckgo.com/?"
            + parse.urlencode(
                {
                    "q": normalized_query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                }
            )
        )
        try:
            response = self.fetch_json(search_url, self.timeout_seconds)
        except Exception as exc:
            return (
                "Live search lookup status:\n"
                f"- query: {normalized_query}\n"
                f"- failed: {exc.__class__.__name__}: {exc}\n"
                "- instruction: If live search is unavailable, say so briefly instead of pretending you searched."
            )

        lines = [
            "Live search snapshot:",
            f"- query: {normalized_query}",
        ]
        heading = _collapse_whitespace(str(response.get("Heading", "")))
        abstract_text = _collapse_whitespace(str(response.get("AbstractText", "")))
        abstract_url = _collapse_whitespace(str(response.get("AbstractURL", "")))
        if heading:
            lines.append(f"- heading: {heading}")
        if abstract_text:
            lines.append(f"- abstract: {abstract_text}")
        if abstract_url:
            lines.append(f"- source_url: {abstract_url}")

        related_topics = self._extract_related_topics(response.get("RelatedTopics", []))
        for index, topic in enumerate(related_topics, start=1):
            lines.append(f"- related_{index}: {topic}")
        if len(lines) == 2:
            lines.append("- note: No concise instant-answer result was available for this query.")
        lines.append(
            "- instruction: If you use this search snapshot, make it clear it is a lightweight live lookup rather than a full web crawl."
        )
        return "\n".join(lines)

    def _extract_related_topics(self, raw_topics: object) -> list[str]:
        if not isinstance(raw_topics, list):
            return []
        results: list[str] = []

        def visit(items: list[object]) -> None:
            for item in items:
                if len(results) >= self.max_related_topics:
                    return
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("Text"), str) and item["Text"].strip():
                    results.append(_collapse_whitespace(item["Text"]))
                    continue
                nested = item.get("Topics")
                if isinstance(nested, list):
                    visit(nested)

        visit(raw_topics)
        return results


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


def _weather_context_lines(title: str, report: WeatherReport, *, include_ip: bool) -> str:
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
            "- instruction: Use this live weather data instead of model priors when answering weather questions.",
        )
    )
    return "\n".join(lines)


def _geocode_place(fetch_json: JsonFetcher, timeout_seconds: int, place: str) -> dict[str, object]:
    response = fetch_json(
        "https://geocoding-api.open-meteo.com/v1/search?"
        + parse.urlencode({"name": place, "count": 1, "language": "en", "format": "json"}),
        timeout_seconds,
    )
    results = response.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError(f"No geocoding result for {place!r}")
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
    weather_url = (
        "https://api.open-meteo.com/v1/forecast?"
        + parse.urlencode(
            {
                "latitude": latitude,
                "longitude": longitude,
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
                "timezone": timezone,
            }
        )
    )
    weather_response = fetch_json(weather_url, timeout_seconds)
    current = weather_response["current"]
    daily = weather_response["daily"]
    return WeatherReport(
        fetched_at_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
        location_label=location_label,
        latitude=latitude,
        longitude=longitude,
        timezone=str(weather_response.get("timezone", timezone)),
        current_condition=_weather_label(current.get("weather_code")),
        current_temperature_c=float(current["temperature_2m"]),
        apparent_temperature_c=float(current["apparent_temperature"]),
        relative_humidity_percent=float(current["relative_humidity_2m"]),
        wind_speed_kmh=float(current["wind_speed_10m"]),
        today_condition=_weather_label(daily["weather_code"][0]),
        today_high_c=float(daily["temperature_2m_max"][0]),
        today_low_c=float(daily["temperature_2m_min"][0]),
        today_precipitation_probability_max_percent=float(daily["precipitation_probability_max"][0]),
        public_ip=public_ip,
    )


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
