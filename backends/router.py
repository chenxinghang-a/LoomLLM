from __future__ import annotations
from typing import Optional

# Intra-package imports
from .profile import BackendProfile
from ..core.events import EventBus, Event, EventType, bus


class ModelRouter:
    """
    Intelligent model routing based on task complexity.
    
    Core idea: NOT every question needs GPT-4o.
    - Simple questions → fast/cheap model
    - Complex reasoning → powerful model
    - Creative tasks → high-temperature model
    - Code execution → code-specialized model
    
    This is THE differentiator vs CrewAI/LangGraph which use ONE model for everything.
    """

    # Complexity detection keywords (higher weight = more complex)
    COMPLEXITY_KEYWORDS = {
        "design": 3, "architecture": 3, "analyze": 2, "research": 3,
        "compare": 2, "evaluate": 2, "critique": 2, "optimize": 2,
        "explain": 1, "write": 1, "code": 2, "debug": 2,
        "translate": 0, "summarize": 1, "calculate": 0, "what is": 0,
        "hello": -1, "hi": -1, "thanks": -1, "1+1": -2, "2+2": -2,
    }

    # Length thresholds
    SHORT_QUERY_MAX = 20       # chars
    MEDIUM_QUERY_MAX = 100     # chars

    def __init__(self, profiles: dict[str, BackendProfile]):
        self.profiles = profiles
        # Categorize available backends by tier
        self._by_tier: dict[str, list[BackendProfile]] = {}
        for p in profiles.values():
            if not p.enabled:
                continue
            self._by_tier.setdefault(p.tier, []).append(p)

    def score_complexity(self, text: str) -> int:
        """Score query complexity 0-10. Higher = needs better model."""
        if not text:
            return 0
        text_lower = text.lower().strip()
        
        score = 0
        
        # Keyword matching
        for kw, weight in self.COMPLEXITY_KEYWORDS.items():
            if kw in text_lower:
                score += weight
        
        # Length factor
        length = len(text_lower)
        if length <= self.SHORT_QUERY_MAX:
            score -= 1
        elif length >= self.MEDIUM_QUERY_MAX:
            score += min(length // 100, 3)
        
        # Multi-sentence / structured queries suggest complexity
        sentence_count = text_lower.count('.') + text_lower.count('?') + text_lower.count('\n')
        if sentence_count >= 2:
            score += 1
        if sentence_count >= 4:
            score += 1
        
        # Code-like content
        if any(c in text_lower for c in ['def ', 'function', 'class ', 'import ', 'const ', 'let ']):
            score += 2
        
        return max(0, min(10, score))

    def route(self, user_input: str, expert: object = None, 
              forced_profile: str = "", forced_model: str = "") -> BackendProfile:
        """
        Select the best backend for this request.
        
        Priority order:
        1. Expert-level explicit profile override (api_profile in YAML)
        2. Caller-specified profile/model
        3. Router auto-detection by complexity + tier availability
        4. Fallback to first enabled profile
        """
        
        # Priority 1 & 2: Explicit overrides
        if forced_profile and forced_profile in self.profiles:
            return self.profiles[forced_profile]
        
        if expert and hasattr(expert, 'api_profile') and expert.api_profile:
            if expert.api_profile in self.profiles:
                return self.profiles[expert.api_profile]

        # Priority 3: Auto-routing by complexity
        complexity = self.score_complexity(user_input)
        
        if complexity <= 1:
            # Trivial query: use cheapest/fastest available
            for tier in ["free", "cheap", "local", "standard"]:
                if tier in self._by_tier and self._by_tier[tier]:
                    picked = max(self._by_tier[tier], key=lambda p: p.priority)
                    bus.publish(Event(EventType.AGENT_EXECUTE, {
                        "router": f"auto→{tier}", "complexity": complexity,
                        "profile": picked.display_name, "reason": "trivial_query"
                    }, source="ModelRouter"))
                    return picked
        
        elif complexity >= 6:
            # Complex query: use best available
            for tier in ["premium", "standard", "cheap"]:
                if tier in self._by_tier and self._by_tier[tier]:
                    picked = max(self._by_tier[tier], key=lambda p: p.priority)
                    bus.publish(Event(EventType.AGENT_EXECUTE, {
                        "router": f"auto→{tier}", "complexity": complexity,
                        "profile": picked.display_name, "reason": "complex_query"
                    }, source="ModelRouter"))
                    return picked
        
        else:
            # Medium complexity: standard tier
            for tier in ["standard", "cheap"]:
                if tier in self._by_tier and self._by_tier[tier]:
                    picked = max(self._by_tier[tier], key=lambda p: p.priority)
                    bus.publish(Event(EventType.AGENT_EXECUTE, {
                        "router": f"auto→{tier}", "complexity": complexity,
                        "profile": picked.display_name, "reason": "medium_query"
                    }, source="ModelRouter"))
                    return picked
        
        # Priority 4: Fallback to any enabled profile
        enabled = [p for p in self.profiles.values() if p.enabled]
        if enabled:
            return max(enabled, key=lambda p: p.priority)
        
        raise RuntimeError("No enabled API backend available!")


