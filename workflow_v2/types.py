from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class WorkflowNodeV2:
    """A single step in a V2 workflow graph (DAG node)."""
    node_id: str
    expert_id: str
    action: str                     # "generate" | "review" | "refine" | "synthesize"
    prompt_template: str            # Supports {variable} substitution
    inputs: list[str]               # Upstream dependency node IDs
    condition: str = ""             # Optional Python expression for conditional exec
    parallel_group: int = 0         # Nodes in same group can run in parallel


@dataclass
class WorkflowGraph:
    """A complete workflow DAG (directed acyclic graph)."""
    workflow_id: str
    goal: str = ""
    nodes: list[WorkflowNodeV2] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)  # [(src, dst), ...]
    entry_nodes: list[str] = field(default_factory=list)
    exit_nodes: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


__all__ = ['WorkflowNodeV2', 'WorkflowGraph']
