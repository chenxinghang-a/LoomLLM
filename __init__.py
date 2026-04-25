"""
LoomLLM — The Iterative LLM Framework

Weave multiple LLM providers into a self-improving collaboration loop:
  - core/: Events, Budget, Memory, Validation
  - experts/: Expert Registry, Task Classifier
  - agents/: CoT, Executor, Reviewer, Memory, CollabLoop (V5)
  - backends/: LLM Client, Multi-Backend, SmartInit, Fallback
  - main_mod/: AIStaff orchestrator

10 providers, OpenAI-compatible, zero config.
"""

__version__ = "1.0.0"
__author__ = "LoomLLM Contributors"

# Lazy imports — only load when accessed
def __getattr__(name):
    """Lazy import for backward compatibility."""
    _imports = {
        # Main entry point
        'AIStaff': ('.main_mod.staff', 'AIStaff'),
        # Core
        'EventBus': ('.core.events', 'EventBus'),
        'Event': ('.core.events', 'Event'),
        'EventType': ('.core.events', 'EventType'),
        'TokenBudgetManager': ('.core.budget', 'TokenBudgetManager'),
        'BudgetConfig': ('.core.budget', 'BudgetConfig'),
        'MemorySystem': ('.core.memory', 'MemorySystem'),
        # Experts
        'ExpertRegistry': ('.experts.registry', 'ExpertRegistry'),
        'ExpertConfig': ('.experts.registry', 'ExpertConfig'),
        'TaskClassifier': ('.experts.classifier', 'TaskClassifier'),
        # Agents
        'CoTAgent': ('.agents.cot', 'CoTAgent'),
        'CollaborationLoop': ('.agents.collab_loop', 'CollaborationLoop'),
        'RouteContext': ('.agents.collab_loop', 'RouteContext'),
        'StructuredFeedback': ('.agents.collab_loop', 'StructuredFeedback'),
        # Backends
        'LLMClient': ('.backends.client', 'LLMClient'),
        'BackendProfile': ('.backends.profile', 'BackendProfile'),
        'MultiLLMClient': ('.backends.multi_client', 'MultiLLMClient'),
        'ModelRouter': ('.backends.router', 'ModelRouter'),
        'SmartInit': ('.backends.smart_init', 'SmartInit'),
        'FallbackManager': ('.backends.fallback', 'FallbackManager'),
    }
    
    if name in _imports:
        mod_path, attr = _imports[name]
        import importlib
        mod = importlib.import_module(mod_path, package='ai_staff_v4')
        return getattr(mod, attr)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Convenience factory functions
def from_env(**kwargs):
    """Zero-config launch: auto-detect API keys from environment."""
    from .main_mod.staff import AIStaff
    return AIStaff.from_env(**kwargs)


def quick_start(prompt: str, **kwargs):
    """One-liner: create staff and run."""
    staff = from_env(**kwargs)
    return staff.chat(prompt)


def discover_and_start(**kwargs):
    """Discover all available backends and start with best one."""
    from .main_mod.staff import AIStaff
    return AIStaff.discover_and_start(**kwargs)


__all__ = [
    '__version__', 'AIStaff', 'from_env', 'quick_start', 'discover_and_start',
    'EventBus', 'MemorySystem', 'ExpertRegistry', 'TaskClassifier',
    'MultiLLMClient', 'ModelRouter', 'SmartInit', 'FallbackManager',
    'CollaborationLoop', 'RouteContext', 'StructuredFeedback',
]
