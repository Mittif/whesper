from __future__ import annotations

from dataclasses import dataclass

from whesper.config import AppConfig


@dataclass(slots=True)
class RouteDecision:
    model_alias: str
    mode: str
    reason: str


def select_model(
    config: AppConfig,
    user_text: str,
    *,
    pinned_model: str | None = None,
    mode_override: str = "auto",
) -> RouteDecision:
    text = user_text.strip()

    if pinned_model and pinned_model != "auto":
        config.get_model(pinned_model)
        return RouteDecision(
            model_alias=pinned_model,
            mode="manual",
            reason="session pinned model",
        )

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

