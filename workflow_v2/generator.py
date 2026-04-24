from __future__ import annotations
import json, re, hashlib, os, time
from datetime import datetime
from typing import Any, Optional

# Intra-package imports
from ..backends.client import LLMClient
from .types import WorkflowNodeV2, WorkflowGraph
from ..experts.registry import ExpertConfig


class WorkflowGeneratorV2:
    """
    V2 Workflow Generator: LLM generates a custom DAG per task.

    Unlike V1's static 5-phase pipeline, this generates UNIQUE workflows
    based on task complexity, type, and available experts.

    Key difference:
      V1: Every task goes through plan→execute→review→revise→deliver
      V2: Simple Q&A = 1-node graph, Code task = write+review graph,
          Research = plan+dive+synth graph, Complex = multi-branch DAG
    """

    GENERATOR_SYSTEM = """你是一个高级工作流编排师。根据用户目标，设计一个最优的多步骤执行计划（有向无环图DAG）。

【可用专家】
{experts_info}

【输出规则】
1. 分析目标复杂度，决定步骤数和专家组合
2. 步骤间可以形成依赖关系(DAG边)，无依赖的节点可并行
3. 每步指派最合适的专家
4. 必须有明确的输入输出约定
5. 估计每步复杂度

【严格JSON输出格式】：
{{"goal":"目标复述","complexity":"simple|medium|complex","estimated_steps":N,
"nodes":[
  {{"node_id":"step_1","expert_id":"planner","action":"generate",
    "prompt_template":"规划: {{goal}}","inputs":["user_input"],"parallel_group":0}},
  ...
],"edges":[["step_1","step_2"],...],
"reasoning":"为什么这样设计的理由"}}"""

    def __init__(self, llm_client):
        self.llm = llm_client
        self._cache: dict[str, WorkflowGraph] = {}
        self._history: list[dict] = []

    def generate(self, goal: str, experts: list[ExpertConfig],
                 constraints: dict = None) -> WorkflowGraph:
        """
        Generate an optimal workflow DAG for the given goal.
        
        Args:
            goal: User's objective description
            experts: Available expert configs
            constraints: Hints like {"max_steps": 5, "prefer_parallel": True}
        
        Returns:
            A WorkflowGraph ready for execution
        """
        # Build expert info string
        experts_info = "\n".join(
            f"  - {e.id}: {e.name} — {e.description}" for e in experts
        )

        constraint_text = ""
        if constraints:
            constraint_text = "\n【约束条件】\n" + "\n".join(
                f"  - {k}: {v}" for k, v in constraints.items()
            )

        user_msg = (
            f"请为以下目标设计最优工作流DAG：\n\n"
            f"## 目标\n{goal}\n\n{constraint_text}\n\n"
            f"输出JSON格式的完整工作流定义。"
        )

        # Call LLM
        response, _usage = self.llm.chat_completion([
            {"role": "system", "content": self.GENERATOR_SYSTEM.format(experts_info=experts_info)},
            {"role": "user", "content": user_msg}
        ], temperature=0.5, max_tokens=3072)

        # Parse JSON response
        wf_data = self._parse_json(response)
        
        # Build graph object
        graph = self._build_graph(wf_data, goal)
        
        # Cache & record
        cache_key = hashlib.md5(goal.encode()).hexdigest()[:12]
        self._cache[cache_key] = graph
        self._history.append({
            "goal": goal[:80], "nodes": len(graph.nodes),
            "complexity": graph.metadata.get("complexity", "?"),
            "ts": datetime.now().isoformat(),
        })

        print(f"  [WF-V2] Generated DAG: {graph.metadata.get('complexity')}, "
              f"{len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        return graph

    def _parse_json(self, text: str) -> dict:
        """Extract JSON from LLM response with fallbacks."""
        # Try ```json ... ``` block
        m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        raw = m.group(1) if m else text.strip()

        # If it's a bare array, wrap it
        if raw.startswith('['):
            raw = '{"nodes":' + raw + ',"edges":[]}'

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print("  [WF-V2] JSON解析失败，使用fallback线性工作流")
            return self._fallback_workflow()

    def _fallback_workflow(self) -> dict:
        """Generate simple linear workflow when parsing fails."""
        return {
            "goal": "fallback", "complexity": "medium",
            "nodes": [
                {"node_id": "step_1", "expert_id": "generalist", "action": "generate",
                 "prompt_template": "分析并回答: {goal}", "inputs": ["user_input"],
                 "parallel_group": 0},
            ],
            "edges": [],
            "reasoning": "Fallback: JSON parse failed"
        }

    def _build_graph(self, data: dict, goal: str) -> WorkflowGraph:
        """Convert dict to WorkflowGraph object."""
        nodes = []
        for nd in data.get("nodes", []):
            nodes.append(WorkflowNodeV2(
                node_id=nd.get("node_id", f"node_{len(nodes)}"),
                expert_id=nd.get("expert_id", "generalist"),
                action=nd.get("action", "generate"),
                prompt_template=nd.get("prompt_template", "{goal}"),
                inputs=nd.get("inputs", ["user_input"]),
                condition=nd.get("condition", ""),
                parallel_group=nd.get("parallel_group", 0),
            ))

        edges = [(e[0], e[1]) for e in data.get("edges", [])]

        all_targets = set(e[1] for e in edges)
        entry_nodes = [n.node_id for n in nodes if "user_input" in n.inputs]
        exit_nodes = [n.node_id for n in nodes
                      if n.node_id not in all_targets and n.node_id not in entry_nodes]
        if not exit_nodes and nodes:
            exit_nodes = [nodes[-1].node_id]

        return WorkflowGraph(
            workflow_id=f"wf_{datetime.now().strftime('%H%M%S')}_{os.urandom(3).hex()}",
            goal=goal,
            nodes=nodes,
            edges=edges,
            entry_nodes=entry_nodes,
            exit_nodes=exit_nodes,
            metadata={
                "complexity": data.get("complexity", "unknown"),
                "estimated_steps": data.get("estimated_steps", len(nodes)),
                "reasoning": data.get("reasoning", ""),
                "generated_at": datetime.now().isoformat(),
            }
        )



__all__ = ['WorkflowGeneratorV2']


