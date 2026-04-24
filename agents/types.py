from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Intra-package imports
from ..core.constants import VERSION
from ..core.validation import ValidationResult

class AgentState(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    MEMORIZING = "memorizing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class TaskState:
    task_id: str
    state: AgentState = AgentState.IDLE
    plan: str = ""
    draft: str = ""
    review_result: Optional[ValidationResult] = None
    final_output: str = ""
    error: str = ""
    retry_count: int = 0
    history: list[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "state": self.state.value,
            "plan": self.plan, "draft": self.draft,
            "final_output": self.final_output, "error": self.error,
            "retry_count": self.retry_count, "history": self.history,
            "review_result": {
                "passed": self.review_result.passed, "score": self.review_result.score,
                "issues": self.review_result.issues
            } if self.review_result else None
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TaskState':
        ts = cls(task_id=d["task_id"])
        ts.state = AgentState(d.get("state", "idle"))
        ts.plan = d.get("plan", "")
        ts.draft = d.get("draft", "")
        ts.final_output = d.get("final_output", "")
        ts.error = d.get("error", "")
        ts.retry_count = d.get("retry_count", 0)
        ts.history = d.get("history", [])
        rr = d.get("review_result")
        if rr:
            ts.review_result = ValidationResult(rr["passed"], rr["score"], rr["issues"])
        return ts


# ═══════════════════════════════════════════════════════════
# COLLABORATION RESULT — Structured Output (V3)
# ═══════════════════════════════════════════════════════════

@dataclass
class CollaborationResult:
    """Structured result from a collaboration or auto_run session."""
    goal: str = ""
    status: str = "success"          # success | partial | failed
    strategy_mode: str = ""           # Which TaskClassifier mode was used
    trace_id: str = ""               # 追踪ID，贯穿整个协作过程
    
    # Core deliverables (the VALUE, not just chat log)
    deliverables: dict[str, str] = field(default_factory=dict)  # {"name": "content"}
    
    # Full process records
    transcript: str = ""
    interaction_log: list[dict] = field(default_factory=list)
    
    # Quality metrics
    quality_score: float = 0.0       # 0-10
    rounds_used: int = 0
    total_tokens: int = 0
    total_time_sec: float = 0.0
    experts_used: list[str] = field(default_factory=list)
    
    def save(self, output_dir: str):
        """Save all outputs to a directory — one command delivers everything."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        files_saved = []
        
        # 1. Save each deliverable as separate file
        for name, content in self.deliverables.items():
            fpath = out_path / name
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(content)
            files_saved.append(str(fpath))
        
        # 2. Save full report (summary)
        report = self._generate_report()
        report_path = out_path / "report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        files_saved.append(str(report_path))
        
        # 3. Save transcript
        if self.transcript:
            tx_path = out_path / "transcript.txt"
            with open(tx_path, 'w', encoding='utf-8') as f:
                f.write(self.transcript)
            files_saved.append(str(tx_path))
        
        return files_saved
    
    def _generate_report(self) -> str:
        """Generate a summary markdown report."""
        lines = [
            f"# AI-Staff Execution Report",
            f"",
            f"**Goal:** {self.goal}",
            f"**Mode:** {self.strategy_mode} | **Status:** {self.status}",
            f"**Quality:** {self.quality_score}/10",
            f"**Time:** {self.total_time_sec:.1f}s | **Rounds:** {self.rounds_used}",
            f"**Experts:** {', '.join(self.experts_used)}",
            f"",
            f"## Deliverables ({len(self.deliverables)})",
            f"",
        ]
        for name, content in self.deliverables.items():
            preview = content.replace('\n', ' ')[:120]
            lines.append(f"- **{name}**: {preview}...")
        
        if self.interaction_log:
            lines.extend(["", f"## Interaction Log", ""])
            for entry in self.interaction_log[-10:]:
                lines.append(f"- [{entry.get('expert','?')}] {entry.get('action','')} "
                            f"({entry.get('chars',0)}ch)")
        
        lines.extend(["", "---", f"*AI-Staff V{VERSION} · {datetime.now().strftime('%Y-%m-%d %H:%M')}*"])
        return "\n".join(lines)

    @property
    def final_text(self) -> str:
        """Get the main deliverable text (first or 'answer.txt')."""
        if not self.deliverables:
            return ""
        # Prefer 'answer.txt', then first value
        if "answer.txt" in self.deliverables:
            return self.deliverables["answer.txt"]
        return next(iter(self.deliverables.values()))

    def __repr__(self) -> str:
        score = f"{self.quality_score:.0f}/10" if self.quality_score else "N/A"
        n_deliverables = len(self.deliverables)
        preview = ""
        if self.final_text:
            preview = self.final_text[:60].replace('\n', ' ')
        return (f"CollabResult(mode={self.strategy_mode}, status={self.status}, "
                f"score={score}, deliverables={n_deliverables}, "
                f"preview='{preview}...')")

    def __str__(self) -> str:
        return self.final_text




