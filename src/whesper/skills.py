from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from whesper.agent_types import ToolInvocation, ToolObservation


class SkillPlatformAdapter(Protocol):
    def owns_tool(self, tool_name: str) -> bool: ...

    def list_tools(self) -> tuple[object, ...]: ...

    def execute(self, invocation: ToolInvocation) -> ToolObservation: ...


@dataclass(slots=True)
class SkillPlatformAdapterRegistry:
    adapters: tuple[SkillPlatformAdapter, ...] = field(default_factory=tuple)

    def list_tools(self) -> tuple[object, ...]:
        tools: list[object] = []
        for adapter in self.adapters:
            tools.extend(adapter.list_tools())
        return tuple(tools)

    def adapter_for(self, tool_name: str) -> SkillPlatformAdapter | None:
        for adapter in self.adapters:
            if adapter.owns_tool(tool_name):
                return adapter
        return None
