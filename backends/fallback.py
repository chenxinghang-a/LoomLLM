from __future__ import annotations
import time
from typing import Optional

# Intra-package imports
from .profile import BackendProfile


class FallbackManager:
    """
    Cascade fallback: when primary backend fails, try alternatives.
    
    Handles: rate limits (429), timeouts, connection errors, auth failures.
    Each backend can be tried in priority order.
    """

    def __init__(self, profiles: dict[str, BackendProfile]):
        self.profiles = profiles
        # Track per-backend failure count for circuit-breaker pattern
        self._failure_counts: dict[str, int] = {}
        self._last_failure_time: dict[str, float] = {}
        self._circuit_breaker_threshold = 3   # Consecutive failures before skipping
        self._circuit_breaker_cooldown = 60   # Seconds before retrying a broken backend

    def get_fallback_chain(self, exclude: str = "") -> list[BackendProfile]:
        """Get ordered list of fallback candidates."""
        now = time.time()
        candidates = []
        
        for p in sorted(self.profiles.values(), key=lambda x: -x.priority):
            if p.name == exclude or not p.enabled:
                continue
            # Circuit breaker check
            fail_count = self._failure_counts.get(p.name, 0)
            last_fail = self._last_failure_time.get(p.name, 0)
            if fail_count >= self._circuit_breaker_threshold and (now - last_fail) < self._circuit_breaker_cooldown:
                continue  # Skip this backend, it's in cooldown
            candidates.append(p)
        
        return candidates

    def record_success(self, profile_name: str):
        """Reset failure counter on success."""
        self._failure_counts[profile_name] = 0

    def record_failure(self, profile_name: str):
        """Increment failure counter."""
        self._failure_counts[profile_name] = self._failure_counts.get(profile_name, 0) + 1
        self._last_failure_time[profile_name] = time.time()


