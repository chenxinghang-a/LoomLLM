from __future__ import annotations
import json, time
from typing import Any, Optional

# Intra-package imports
from .types import WorkflowNodeV2, WorkflowGraph
from ..experts.registry import ExpertRegistry


class WorkflowExecutorV2:
    """
    V2 Executor: Runs a WorkflowGraph (DAG) with topological sort.
    
    Algorithm:
    1. Topological sort → layers (same layer = can be parallelized)
    2. Execute layer-by-layer
    3. Pass outputs between dependent nodes via template substitution
    4. Collect final output(s) from exit nodes
    """

    def __init__(self, agents: dict, multi_llm_or_client, memory=None, validator=None):
        self.agents = agents
        self.mllm = multi_llm_or_client
        self.memory = memory
        self.validator = validator
        self._results: dict[str, str] = {}   # node_id → output content

    def execute(self, graph: WorkflowGraph, user_input: str,
                session_id: str = "") -> tuple[str, dict]:
        """Execute the workflow DAG. Returns (final_output_string, execution_stats_dict)."""
        layers = self._topological_sort(graph)
        exec_log = []
        t0 = time.time()

        print(f"  [WF-V2 Exec] 开始执行: {len(layers)}层 × {len(graph.nodes)}节点")

        for li, layer in enumerate(layers):
            layer_tag = f"L{li+1}"
            node_ids = [n.node_id for n in layer]
            print(f"  [WF-V2 Exec] {layer_tag}: {node_ids}")

            for node in layer:
                if node.condition:
                    ctx = {"results": self._results, "input": user_input}
                    try:
                        if not eval(node.condition, {"__builtins__": {}}, ctx):
                            print(f"    └─ [{node.node_id}] 跳过 (condition=false)")
                            continue
                    except Exception:
                        pass

                prompt = node.prompt_template
                prompt = prompt.replace("{goal}", graph.goal)
                prompt = prompt.replace("{user_input}", user_input)
                for dep_id in node.inputs:
                    if dep_id in self._results:
                        dep_output = self._results[dep_id]
                        prompt = prompt.replace(f"{{{dep_id}_output}}", dep_output[:3000])

                expert = ExpertRegistry.get(node.expert_id) or ExpertRegistry.get("generalist")

                try:
                    t1 = time.time()
                    msgs = [
                        {"role": "system", "content": expert.system_prompt},
                        {"role": "user", "content": prompt},
                    ]

                    if hasattr(self.mllm, 'chat'):
                        content, usage = self.mllm.chat(
                            msgs, temperature=expert.temperature,
                            user_input=user_input, expert=expert, max_tokens=4096
                        )
                    else:
                        content, usage = self.mllm.chat_completion(
                            msgs, temperature=expert.temperature, max_tokens=4096
                        )

                    elapsed = time.time() - t1
                    self._results[node.node_id] = content

                    exec_log.append({
                        "layer": layer_tag, "node": node.node_id,
                        "expert": expert.name, "action": node.action,
                        "chars": len(content), "time": round(elapsed, 2),
                        "status": "ok",
                    })
                    print(f"    ✓ [{node.node_id}] {expert.name}: "
                          f"{len(content)}ch / {elapsed:.1f}s")

                except Exception as e:
                    err_str = f"[{type(e).__name__}] {str(e)[:120]}"
                    self._results[node.node_id] = f"[ERROR] {err_str}"
                    exec_log.append({
                        "layer": layer_tag, "node": node.node_id,
                        "status": "error", "error": err_str,
                    })
                    print(f"    ✗ [{node.node_id}] FAILED: {err_str}")

            time.sleep(0.25)

        # Collect final output
        final_parts = []
        for eid in graph.exit_nodes:
            if eid in self._results:
                final_parts.append(self._results[eid])
        final_output = "\n\n---\n\n".join(final_parts) if final_parts else json.dumps(
            self._results, ensure_ascii=False, indent=2
        )

        total_time = time.time() - t0
        ok_count = sum(1 for l in exec_log if l.get("status") == "ok")

        stats = {
            "total_time": round(total_time, 2),
            "total_nodes": len(graph.nodes),
            "completed_nodes": ok_count,
            "layers": len(layers),
            "graph_metadata": graph.metadata,
        }

        print(f"  [WF-V2 Exec] 完成: {ok_count}/{len(graph.nodes)} 节点成功, "
              f"{total_time:.1f}s")

        return final_output, stats

    def _topological_sort(self, graph: WorkflowGraph) -> list[list[WorkflowNodeV2]]:
        """Kahn's algorithm: partition DAG into executable layers."""
        node_map = {n.node_id: n for n in graph.nodes}
        in_deg = {n.node_id: 0 for n in graph.nodes}
        for src, dst in graph.edges:
            if dst in in_deg:
                in_deg[dst] += 1

        layers = []
        remaining = set(in_deg.keys())

        while remaining:
            current_layer_nids = [nid for nid in remaining if in_deg[nid] == 0]
            if not current_layer_nids:
                break

            layers.append([node_map[nid] for nid in current_layer_nids])

            for nid in current_layer_nids:
                remaining.remove(nid)
                for src, dst in graph.edges:
                    if src == nid and dst in in_deg:
                        in_deg[dst] -= 1

        return layers



__all__ = ['WorkflowExecutorV2']
