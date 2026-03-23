from __future__ import annotations

import unittest

from whesper.client import ToolCall
from whesper.tools import ToolRegistry, ToolSpec


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


if __name__ == "__main__":
    unittest.main()
