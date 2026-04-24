from __future__ import annotations
import re, json
from dataclasses import dataclass, field
from typing import Optional

from .events import EventBus, Event, EventType

# Use the global bus from events module
_bus = None

def _get_bus() -> EventBus:
    global _bus
    if _bus is None:
        from .events import bus as _b
        _bus = _b
    return _bus


@dataclass
class ValidationResult:
    passed: bool
    score: float  # 0.0 - 1.0
    issues: list[str]
    improved_text: Optional[str] = None

class OutputValidator:
    """Validate LLM output quality before returning to user. Strategy pattern by output_format."""
    
    RULES = {
        "min_length": lambda text, threshold: len(text) >= threshold,
        "no_empty_lines_3plus": lambda text, _: not re.search(r'\n{4,}', text),
        "has_structure": lambda text: bool(re.search(r'(#{1,3}\s|\*\*|[-*]\s|\d+\.)', text)),
        "is_complete_sentence": lambda text: text.rstrip().endswith(('。', '.', '!', '？', '?', '"', "'", '》', '」', '`', ')')),
        "no_repetition_5char": lambda text: not any(
            text.count(text[i:i+5]) > 3 for i in range(len(text)-5)
        ) if len(text) > 20 else True,
        "contains_chinese": lambda text: bool(re.search(r'[\u4e00-\u9fff]', text)) if not text.startswith(('```', '<', '{')) else True,
        "valid_json_if_json": lambda text: (
            json.loads(text.strip().removeprefix('```json').removeprefix('```').strip())
            or True
        ) if text.strip().startswith(('{','[')) else True,
    }
    
    # Format-specific rule sets
    FORMAT_RULES = {
        "text": ["min_length", "no_empty_lines_3plus", "is_complete_sentence"],
        "markdown": ["min_length", "has_structure", "no_repetition_5char"],
        "code": [],  # Code outputs are not validated by rules (compiler is the validator)
        "json": ["valid_json_if_json"],
    }
    
    def __init__(self, rules: list[str] = None):
        self.rules = rules or ["min_length", "no_empty_lines_3plus"]
        self._thresholds = {"min_length": 50}
    
    def validate(self, text: str, fmt: str = "text") -> ValidationResult:
        # Use format-specific rules as base, merge with custom rules
        fmt_rules = set(self.FORMAT_RULES.get(fmt, self.rules))
        rules_to_check = list(fmt_rules | set(self.rules))
        
        issues = []
        score = 1.0
        deductions = 0.15  # Each failure deducts this much
        
        for rule_name in rules_to_check:
            rule_fn = self.RULES.get(rule_name)
            if not rule_fn:
                continue
            threshold = self._thresholds.get(rule_name)
            try:
                passed = rule_fn(text) if threshold is None else rule_fn(text, threshold)
                if not passed:
                    issues.append(f"规则[{rule_name}]未通过")
                    score -= deductions
            except Exception as e:
                issues.append(f"规则[{rule_name}]异常: {e}")
                score -= deductions * 0.5
        
        score = max(0.0, min(1.0, score))
        result = ValidationResult(
            passed=len(issues) == 0,
            score=score,
            issues=issues
        )
        
        event_type = EventType.VALIDATION_PASS if result.passed else EventType.VALIDATION_FAIL
        _get_bus().publish(Event(event_type, {"score": score, "issues": issues, "format": fmt}, source="Validator"))
        return result




