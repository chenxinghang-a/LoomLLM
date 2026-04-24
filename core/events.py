from __future__ import annotations
import json, time
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    TASK_START = "task:start"
    TASK_COMPLETE = "task:complete"
    TASK_ERROR = "task:error"
    AGENT_THINK = "agent:think"
    AGENT_EXECUTE = "agent:execute"
    AGENT_REVIEW = "agent:review"
    AGENT_MEMORY = "agent:memory"
    VALIDATION_PASS = "validation:pass"
    VALIDATION_FAIL = "validation:fail"
    BUDGET_WARNING = "budget:warning"
    CHECKPOINT_SAVE = "checkpoint:save"
    USER_FEEDBACK = "user:feedback"

@dataclass
class Event:
    type: EventType
    data: dict
    timestamp: float = field(default_factory=time.time)
    source: str = ""

class EventBus:
    """Lightweight pub/sub event bus for agent coordination."""
    
    MAX_LOG_SIZE = 2000  # Prevent unbounded memory growth
    
    def __init__(self):
        self._listeners: dict[EventType, list] = {}
        self._log: deque[Event] = deque(maxlen=self.MAX_LOG_SIZE)
    
    def subscribe(self, event_type: EventType, callback):
        self._listeners.setdefault(event_type, []).append(callback)
    
    def publish(self, event: Event):
        self._log.append(event)
        for cb in self._listeners.get(event.type, []):
            try:
                cb(event)
            except Exception as e:
                print(f"  [EventBus] Handler error: {e}")
    
    def get_log(self, event_type=None) -> list[Event]:
        if event_type:
            return [e for e in self._log if e.type == event_type]
        return list(self._log)
    
    def audit_log(self) -> str:
        """Generate human-readable audit trail."""
        lines = [f"{'='*60}", f"Audit Log @ {datetime.now().isoformat()} (showing last {len(self._log)} events)", f"{'='*60}"]
        for e in self._log:
            ts = datetime.fromtimestamp(e.timestamp).strftime("%H:%M:%S.%f")[:-3]
            lines.append(f"[{ts}] {e.source:20s} | {e.type.value:25s} | {json.dumps(e.data, ensure_ascii=False)[:80]}")
        return "\n".join(lines)


# Global event bus instance
bus = EventBus()


__all__ = ['EventType', 'Event', 'EventBus', 'bus']
