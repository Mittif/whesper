from __future__ import annotations

from dataclasses import dataclass

from whesper.config import AppConfig

AUTO_SEARCH_KEYWORDS = (
    "search",
    "look up",
    "lookup",
    "find online",
    "check online",
    "查一下",
    "查查",
    "搜一下",
    "搜索",
    "最新",
    "最近",
    "新闻",
    "headline",
    "release notes",
    "changelog",
    "现任",
    "current",
    "today",
)


@dataclass(slots=True)
class RouteDecision:
    model_alias: str
    mode: str
    reason: str


def _matched_search_keyword(text: str) -> str | None:
    lowered = text.casefold()
    for keyword in AUTO_SEARCH_KEYWORDS:
        if keyword.casefold() in lowered:
            return keyword
    return None


def select_model(
    config: AppConfig,
    user_text: str,
    *,
    pinned_model: str | None = None,
    mode_override: str = "auto",
) -> RouteDecision:
    text = user_text.strip()
    model_alias = config.scheduler.chat_model
    model_reason = "default chat model"
    if pinned_model and pinned_model != "auto":
        config.get_model(pinned_model)
        model_alias = pinned_model
        model_reason = "generation configured model"

    if mode_override == "search":
        return RouteDecision(
            model_alias=model_alias,
            mode="search",
            reason=f"explicit search mode ({model_reason})",
        )

    if mode_override == "reasoning":
        return RouteDecision(
            model_alias=model_alias,
            mode="reasoning",
            reason=f"explicit reasoning mode ({model_reason})",
        )

    if mode_override == "chat":
        return RouteDecision(
            model_alias=model_alias,
            mode="chat",
            reason=f"explicit chat mode ({model_reason})",
        )

    if text.startswith("/search "):
        return RouteDecision(
            model_alias=model_alias,
            mode="search",
            reason=f"slash search shortcut ({model_reason})",
        )

    matched_search_keyword = _matched_search_keyword(text)
    if matched_search_keyword:
        return RouteDecision(
            model_alias=model_alias,
            mode="search",
            reason=f"matched search keyword: {matched_search_keyword} ({model_reason})",
        )

    if len(text) >= config.scheduler.long_message_chars:
        return RouteDecision(
            model_alias=model_alias,
            mode="reasoning",
            reason=f"long input heuristic ({model_reason})",
        )

    lowered = text.casefold()
    for keyword in config.scheduler.reasoning_keywords:
        if keyword.casefold() in lowered:
            return RouteDecision(
                model_alias=model_alias,
                mode="reasoning",
                reason=f"matched keyword: {keyword} ({model_reason})",
            )

    return RouteDecision(
        model_alias=model_alias,
        mode="chat",
        reason=f"default chat mode ({model_reason})",
    )
