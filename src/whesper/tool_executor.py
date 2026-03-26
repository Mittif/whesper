from __future__ import annotations

from whesper.agent_types import ToolInvocation
from whesper.skills import SkillPlatformAdapterRegistry
from whesper.tools import ToolExecutionError, ToolExecutionResult, ToolRegistry


class ToolExecutor:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        *,
        skill_registry: SkillPlatformAdapterRegistry | None = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.skill_registry = skill_registry or SkillPlatformAdapterRegistry()

    def execute(self, tool_call: ToolInvocation) -> ToolExecutionResult:
        adapter = self.skill_registry.adapter_for(tool_call.name)
        if adapter is None:
            return self.tool_registry.execute(tool_call)

        observation = adapter.execute(tool_call)
        if observation.is_error:
            raise ToolExecutionError(observation.content)
        return ToolExecutionResult(
            tool_call_id=observation.tool_call_id,
            name=observation.name,
            content=observation.content,
        )
