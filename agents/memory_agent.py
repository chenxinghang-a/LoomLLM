from __future__ import annotations
import json, re

# Intra-package imports
from .base import BaseAgent
from .types import TaskState
from ..core.events import EventBus, Event, EventType, bus
from ..experts.registry import ExpertConfig


class MemoryAgent(BaseAgent):
    """Memory management agent. Summarizes and extracts learnings."""
    
    def run(self, task_state: TaskState, expert: ExpertConfig, messages: list[dict], session_id: str) -> str:
        bus.publish(Event(EventType.AGENT_MEMORY, {"task": task_state.task_id}, source="MemoryAgent"))
        
        if len(messages) < 4:
            return "对话太短，暂不需要摘要"
        
        # Create summary of recent conversation
        summarize_prompt = f"""请将以下对话摘要为关键要点（用于长期记忆）。
保留：决策、结论、偏好、重要事实。
丢弃：寒暄、重复、中间过程。

对话记录：
{json.dumps(messages[-6:], ensure_ascii=False, indent=2)}

输出格式（纯文本）：
### 关键决策
...
### 用户偏好发现
...
### 重要事实
..."""
        
        summary_resp, _usage = self.llm.chat_completion([
            {"role": "system", "content": "你是记忆整理员。只输出结构化摘要，不要多余的话。"},
            {"role": "user", "content": summarize_prompt}
        ], temperature=0.3)
        
        # Save to memory system
        turn_range = f"{len(messages)-5}-{len(messages)}"
        key_points = re.findall(r'### (.*)', summary_resp)
        self.memory.save_summary(session_id, summary_resp, key_points, turn_range)
        
        # Also extract explicit preferences
        self.memory.extract_preferences_from_chat(messages, session_id)
        
        return summary_resp


__all__ = ['MemoryAgent']
