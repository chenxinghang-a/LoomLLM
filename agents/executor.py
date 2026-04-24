from __future__ import annotations
from typing import Optional

# Intra-package imports
from .base import BaseAgent
from .types import TaskState
from ..core.events import EventBus, Event, EventType, bus
from ..experts.registry import ExpertConfig


class ExecutorAgent(BaseAgent):
    """Standard execution agent. Produces the main output."""
    
    def run(self, task_state: TaskState, expert: ExpertConfig, messages: list[dict]) -> str:
        bus.publish(Event(EventType.AGENT_EXECUTE, {"task": task_state.task_id}, source="Executor"))
        
        # Build enhanced system prompt with expert config
        system_parts = [expert.system_prompt]
        if expert.style_hints:
            system_parts.append(f"\n【风格要求】{expert.style_hints}")
        
        ctx_header = task_state.plan and f"\n【已有规划】{task_state.plan}"
        if ctx_header:
            system_parts.insert(0, ctx_header)
        
        exec_messages = [
            {"role": "system", "content": "\n".join(system_parts)}
        ] + messages
        
        model = expert.model_override or ""
        response, _usage = self.llm.chat_completion(exec_messages, 
                                            temperature=expert.temperature,
                                            model=model)
        task_state.draft = response
        return response


