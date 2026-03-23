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

    if mode_override == "search" and config.scheduler.search_model:
        return RouteDecision(
            model_alias=config.scheduler.search_model,
            mode="search",
            reason="explicit search mode",
        )

    if mode_override == "reasoning" and config.scheduler.reasoning_model:
        return RouteDecision(
            model_alias=config.scheduler.reasoning_model,
            mode="reasoning",
            reason="explicit reasoning mode",
        )

    if mode_override == "chat":
        return RouteDecision(
            model_alias=config.scheduler.chat_model,
            mode="chat",
            reason="explicit chat mode",
        )

    if text.startswith("/search ") and config.scheduler.search_model:
        return RouteDecision(
            model_alias=config.scheduler.search_model,
            mode="search",
            reason="slash search shortcut",
        )

    if pinned_model and pinned_model != "auto":
        config.get_model(pinned_model)
        return RouteDecision(
            model_alias=pinned_model,
            mode="manual",
            reason="session pinned model",
        )

    matched_search_keyword = _matched_search_keyword(text)
    if matched_search_keyword and config.scheduler.search_model:
        return RouteDecision(
            model_alias=config.scheduler.search_model,
            mode="search",
            reason=f"matched search keyword: {matched_search_keyword}",
        )

    if len(text) >= config.scheduler.long_message_chars:
        alias = config.scheduler.reasoning_model or config.scheduler.chat_model
        return RouteDecision(
            model_alias=alias,
            mode="reasoning",
            reason="long input heuristic",
        )

    lowered = text.casefold()
    for keyword in config.scheduler.reasoning_keywords:
        if keyword.casefold() in lowered:
            alias = config.scheduler.reasoning_model or config.scheduler.chat_model
            return RouteDecision(
                model_alias=alias,
                mode="reasoning",
                reason=f"matched keyword: {keyword}",
            )

    return RouteDecision(
        model_alias=config.scheduler.chat_model,
        mode="chat",
        reason="default chat model",
    )
