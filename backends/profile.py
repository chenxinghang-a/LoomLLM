from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class BackendProfile:
    """A single API backend configuration (one LLM provider)."""
    name: str                          # Unique ID: "openai", "gemini", "deepseek", etc.
    base_url: str                      # API base URL
    api_key: str                       # API key for this backend
    model: str                         # Default model for this backend
    proxy: str = ""                    # Per-backend proxy (empty = inherit)
    tier: str = "standard"             # free | cheap | standard | premium | local
    max_rpm: int = 0                  # Rate limit (0 = unknown/unlimited)
    enabled: bool = True               # Enable/disable without removing config
    priority: int = 0                 # Higher = preferred for fallback ordering
    # Cost estimation (per 1K tokens)
    input_cost_per_1k: float = 0.0003
    output_cost_per_1k: float = 0.001

    @property
    def display_name(self) -> str:
        return f"{self.name} ({self.model})"


__all__ = ['BackendProfile']
