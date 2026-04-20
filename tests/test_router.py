from __future__ import annotations

import tempfile
import textwrap
import unittest

from whesper.config import load_config
from whesper.router import select_model


def make_config():
    content = textwrap.dedent(
        """
        [scheduler]
        chat_model = "chat"
        reasoning_model = "reasoning"
        search_model = "search"
        long_message_chars = 10
        reasoning_keywords = ["分析"]

        [providers.local]
        base_url = "http://localhost:11434/v1"

        [models.chat]
        provider = "local"
        model = "chat-model"

        [models.reasoning]
        provider = "local"
        model = "reasoning-model"

        [models.search]
        provider = "local"
        model = "search-model"
        """
    ).strip()

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write(content)
        temp_path = fh.name
    return load_config(temp_path)


class RouterTests(unittest.TestCase):
    def test_default_chat_route(self) -> None:
        config = make_config()
        decision = select_model(config, "你好")
        self.assertEqual(decision.model_alias, "chat")
        self.assertEqual(decision.mode, "chat")

    def test_reasoning_keyword_route(self) -> None:
        config = make_config()
        decision = select_model(config, "帮我分析一下这段对话")
        self.assertEqual(decision.model_alias, "chat")
        self.assertEqual(decision.mode, "reasoning")

    def test_search_shortcut_route(self) -> None:
        config = make_config()
        decision = select_model(config, "/search latest release notes")
        self.assertEqual(decision.model_alias, "chat")
        self.assertEqual(decision.mode, "search")

    def test_search_shortcut_keeps_generation_configured_model(self) -> None:
        config = make_config()
        decision = select_model(
            config,
            "/search latest release notes",
            pinned_model="reasoning",
        )
        self.assertEqual(decision.model_alias, "reasoning")
        self.assertEqual(decision.mode, "search")

    def test_explicit_search_intent_routes_to_search_mode(self) -> None:
        config = make_config()
        decision = select_model(config, "帮我查一下 OpenAI release notes")
        self.assertEqual(decision.model_alias, "chat")
        self.assertEqual(decision.mode, "search")

    def test_current_info_keywords_route_to_search_mode(self) -> None:
        config = make_config()
        decision = select_model(config, "OpenAI 最新新闻")
        self.assertEqual(decision.model_alias, "chat")
        self.assertEqual(decision.mode, "search")


if __name__ == "__main__":
    unittest.main()
