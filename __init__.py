"""
AI-Staff V4.0 — Universal AI Staff Dispatcher

Modular architecture:
  - core/: Events, Budget, Memory, Validation, Checkpoint
  - experts/: Expert Registry, Task Classifier
  - agents/: Agent Framework (CoT, Executor, Reviewer, Memory, Workflow)
  - backends/: LLM Client, Multi-Backend Router, Fallback
  - main_mod/: AIStaff orchestrator
  - self_improve/: Self-Improvement Engine
  - workflow_v2: Dynamic DAG Workflow Engine
  - skills/: Skill Registry (dynamic discovery & hot-reload)
  - endpoints/: REST API, MCP Bridge
"""

__version__ = "4.0.0"
__author__ = "AI-Team"

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
        'BaseAgent': ('.agents.base', 'BaseAgent'),
        'CoTAgent': ('.agents.cot', 'CoTAgent'),
        # Backends
        'LLMClient': ('.backends.client', 'LLMClient'),
        'MultiLLMClient': ('.backends.multi_client', 'MultiLLMClient'),
        'ModelRouter': ('.backends.router', 'ModelRouter'),
        # V4 New
        'SelfImprovementEngine': ('.self_improve.engine', 'SelfImprovementEngine'),
        'WorkflowGeneratorV2': ('.workflow_v2.generator', 'WorkflowGeneratorV2'),
        'WorkflowExecutorV2': ('.workflow_v2.executor', 'WorkflowExecutorV2'),
        'SkillRegistry': ('.skills.registry', 'SkillRegistry'),
        'CollaborationLoop': ('.agents.collab_loop', 'CollaborationLoop'),
        'RouteContext': ('.agents.collab_loop', 'RouteContext'),
        'StructuredFeedback': ('.agents.collab_loop', 'StructuredFeedback'),
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
    return staff.auto_run(prompt)


def discover_and_start(**kwargs):
    """Discover all available backends and start with best one."""
    from .main_mod.staff import AIStaff
    return AIStaff.discover_and_start(**kwargs)


def create_skill_registry(*args, **kwargs):
    """Create a new skill registry."""
    from .skills.registry import SkillRegistry, create_builtin_registry
    if args or kwargs:
        return SkillRegistry(*args, **kwargs)
    return create_builtin_registry()


__all__ = [
    '__version__', 'AIStaff', 'from_env', 'quick_start', 'discover_and_start',
    'SkillRegistry', 'create_skill_registry',
    'SelfImprovementEngine', 'WorkflowGeneratorV2', 'WorkflowExecutorV2',
    'EventBus', 'MemorySystem', 'ExpertRegistry', 'TaskClassifier',
    'MultiLLMClient', 'ModelRouter',
]
