from __future__ import annotations
import re

# Intra-package imports
from .base import BaseAgent
from .types import TaskState
from ..core.events import EventBus, Event, EventType, bus
from ..experts.registry import ExpertConfig


class CoTAgent(BaseAgent):
    """
    Chain-of-Thought Planning Agent.
    Breaks down complex problems into structured steps before execution.
    Forces <thinking> block generation.
    """
    
    COT_SYSTEM = """你是思考链规划师。在回答前，你必须：
1. 先用 <thinking> 标签进行深度思考（分解问题、识别关键点、规划步骤）
2. 再用 <answer> 标签给出最终答案

<thinking> 中必须包含：
- 问题理解（这句话在问什么）
- 关键拆解（需要几个步骤）
- 潜在陷阱（容易出错的地方）
- 输出策略（用什么格式呈现）

思考要深入但不要啰嗦。"""
    
    # Keywords that should trigger CoT (multi-language)
    COT_TRIGGERS = re.compile(
        r'(?i)^(?=.*?(?:分析|设计|如何|怎么|为什么|what|how|why|explain|'
        r'analyze|design|compare|对比|区别|方案|plan|step|步骤|原理|'
        r'implement|实现|架构|architecture|optimize|优化|review|审查|'
        r'research|研究|deep.?dive|深入|详细|详述|总结|summarize))'
    )
    
    MIN_LENGTH_FOR_COT = 30
    
    @classmethod
    def should_trigger(cls, text: str) -> bool:
        """Determine if CoT planning should be triggered for this input."""
        if len(text) >= cls.MIN_LENGTH_FOR_COT:
            return True
        return bool(cls.COT_TRIGGERS.search(text))
    
    def run(self, task_state: TaskState, expert: ExpertConfig, messages: list[dict]) -> str:
        self.bus.publish(Event(EventType.AGENT_THINK, {"task": task_state.task_id}, source="CoTAgent"))
        
        cot_messages = [
            {"role": "system", "content": self.COT_SYSTEM},
            {"role": "system", "content": expert.system_prompt},
        ] + messages
        
        response, _usage = self.llm.chat_completion(cot_messages, temperature=0.8)
        
        # Extract thinking part for logging
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
        if thinking_match:
            task_state.plan = thinking_match.group(1).strip()
            self.bus.publish(Event(EventType.AGENT_THINK, {
                "task": task_state.task_id, "plan_saved": True
            }, source="CoTAgent"))
        
        # Extract answer
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Fallback: return full response without tags
        return re.sub(r'</?(thinking|answer)>', '', response).strip()


__all__ = ['CoTAgent']


