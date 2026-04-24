from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .events import EventBus, Event, EventType

# Lazy reference to global bus (avoids circular import issues)
_bus = None

def _get_bus() -> EventBus:
    global _bus
    if _bus is None:
        from .events import bus as _b
        _bus = _b
    return _bus


@dataclass
class BudgetConfig:
    max_tokens_per_task: int = 50000
    max_cost_usd: float = 1.0
    warn_threshold: float = 0.8
    cheap_model: str = ""
    expensive_model: str = ""

class TokenBudgetManager:
    """Track token usage and enforce budget limits."""
    
    def __init__(self, config: BudgetConfig):
        self.config = config
        self.tokens_used = 0
        self.estimated_cost = 0.0
        self.task_count = 0
    
    def record(self, input_tokens: int, output_tokens: int, model: str = ""):
        self.tokens_used += input_tokens + output_tokens
        self.task_count += 1
        # Rough cost estimation (adjust per model)
        rate = 0.00015 if "flash" in model or "lite" in model else 0.002
        self.estimated_cost += (input_tokens + output_tokens) * rate / 1000
        
        ratio = self.tokens_used / self.config.max_tokens_per_task if self.config.max_tokens_per_task else 0
        if ratio >= self.config.warn_threshold:
            _get_bus().publish(Event(EventType.BUDGET_WARNING, {
                "tokens": self.tokens_used, "max": self.config.max_tokens_per_task,
                "ratio": round(ratio, 2), "cost": round(self.estimated_cost, 4)
            }, source="BudgetManager"))
    
    @property
    def is_exhausted(self) -> bool:
        return self.tokens_used >= self.config.max_tokens_per_task
    
    def summary(self) -> dict:
        return {
            "tokens_used": self.tokens_used,
            "estimated_cost_usd": round(self.estimated_cost, 4),
            "task_count": self.task_count,
            "exhausted": self.is_exhausted
        }



