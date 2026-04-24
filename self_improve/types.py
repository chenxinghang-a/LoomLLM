from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class ImprovementRecord:
    """One self-improvement cycle record."""
    timestamp: str
    trigger: str                  # "low_score" | "failure" | "periodic" | "manual"
    target_component: str         # "expert_prompt" | "classifier_rule" | "router_strategy"
    before_hash: str = ""
    after_content: str = ""        # What changed
    reasoning: str = ""            # Why the LLM decided this
    validation_score: float = 0.0  # Post-improvement quality check
    applied: bool = False          # Whether change was applied


