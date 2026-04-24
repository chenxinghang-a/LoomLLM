from __future__ import annotations
import re

# Intra-package imports
from .base import BaseAgent
from .types import TaskState
from ..core.events import EventBus, Event, EventType, bus
from ..core.validation import ValidationResult
from ..experts.registry import ExpertConfig


class ReviewAgent(BaseAgent):
    """
    Quality Review Agent.
    Critiques output and suggests improvements before final delivery.
    """
    
    REVIEW_PROMPT = """你是一个严格的输出质量审查员。请审查以下AI回复的质量。

【原始问题】
{question}

【AI回复】
{response}

请按以下格式输出审查结果：

## 质量评分: X/10
## 是否通过: 是/否
## 问题列表:
1. ...
## 改进建议:
..."""

    def run(self, task_state: TaskState, expert: ExpertConfig, messages: list[dict]) -> ValidationResult:
        bus.publish(Event(EventType.AGENT_REVIEW, {"task": task_state.task_id}, source="ReviewAgent"))
        
        last_question = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_question = m["content"][:1000]
                break
        
        review_msg = self.REVIEW_PROMPT.format(question=last_question, response=task_state.draft[:4000])
        review_messages = [
            {"role": "system", "content": "你是严格的质量审查员。只输出审查结果，不要客套话。"},
            {"role": "user", "content": review_msg}
        ]
        
        review_response, _usage = self.llm.chat_completion(review_messages, temperature=0.2)
        
        # Parse score
        score_match = re.search(r'质量评分[::\s]*(\d+)/?10?', review_response)
        score = float(score_match.group(1)) / 10 if score_match else 0.7
        
        # Parse pass/fail
        passed = bool(re.search(r'是否通过[::\s]*是', review_response))
        
        # Parse issues
        issues_section = re.search(r'问题列表[::\s]*\n(.*?)(?=##|$)', review_response, re.DOTALL)
        issues = []
        if issues_section:
            for line in issues_section.group(1).strip().split('\n'):
                line = line.strip()
                if line and line[0].isdigit():
                    issues.append(re.sub(r'^\d+[\.\)]\s*', '', line))
        
        if not issues and not passed:
            issues = ["LLM审查未通过但未说明原因"]
        
        return ValidationResult(passed=passed, score=score, issues=issues)


__all__ = ['ReviewAgent']


