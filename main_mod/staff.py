from __future__ import annotations
import os, sys, time, re, json, math
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Intra-package imports — the full dependency graph
from ..core.constants import VERSION, MAX_RETRIES
from ..core.events import EventBus, Event, EventType
from ..core.memory import MemorySystem
from ..core.validation import OutputValidator
from ..core.budget import TokenBudgetManager, BudgetConfig
# checkpoint.py 已删除（死代码）
from ..experts.registry import ExpertRegistry, ExpertConfig
from ..experts.classifier import TaskClassifier, TaskStrategy
from ..backends.client import LLMClient
from ..backends.profile import BackendProfile
from ..backends.multi_client import MultiLLMClient
from ..agents.types import TaskState, AgentState, CollaborationResult
from ..agents.cot import CoTAgent
from ..agents.executor import ExecutorAgent
from ..agents.reviewer import ReviewAgent
from ..agents.memory_agent import MemoryAgent
# V1 workflow.py 已删除（死代码），V2/V5替代
from ..self_improve.engine import SelfImprovementEngine
from ..workflow_v2.generator import WorkflowGeneratorV2
from ..workflow_v2.executor import WorkflowExecutorV2

# Global event bus instance
bus = EventBus()


class AIStaff:
    """Main AI Staff dispatcher orchestrating all components."""

    def __init__(self, base_url: str = "", api_key: str = "", model: str = "",
                 proxy: str = "", expert_id: str = "generalist",
                 profiles: Optional[dict] = None):
        """
        Initialize AI-Staff with either:
        1. Single backend (legacy): pass base_url + api_key + model
        2. Multiple backends (NEW!): pass profiles=dict
        
        Examples:
            # Legacy single-backend mode (unchanged)
            staff = AIStaff(base_url="...", api_key="...", model="gpt-4o")
            
            # NEW multi-backend mode (killer feature)
            from ai_staff import BackendProfile
            staff = AIStaff(profiles={
                "openai": BackendProfile("openai", "https://api.openai.com/v1",
                                         "sk-...", "gpt-4o-mini", tier="cheap"),
                "gemini": BackendProfile("gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
                                        "...", "gemini-2.5-flash", tier="fast"),
                "ollama": BackendProfile("ollama", "http://localhost:11434/v1",
                                       "ollama", "qwen2.5:7b", tier="free"),
            })
        """
        # Detect mode: multi-backend or legacy single
        self._multi_mode = bool(profiles)
        
        if self._multi_mode:
            # ── NEW: Multi-Backend Mode ──
            self.backends = {name: (p if isinstance(p, BackendProfile) else BackendProfile(**p))
                            for name, p in profiles.items()}
            self.multi_llm = MultiLLMClient(self.backends, default_proxy=proxy)
            # Set primary as the main llm reference (for backward compat)
            self.llm = self.multi_llm._clients[self.multi_llm.default_profile]
            self.model_router = self.multi_llm.router
            self.fallback_mgr = self.multi_llm.fallback
        else:
            # ── LEGACY: Single Backend Mode ──
            if not base_url or not api_key:
                raise ValueError("Single-mode requires base_url AND api_key (or use profiles=)")
            self.llm = LLMClient(base_url, api_key, model, proxy)
            self.multi_llm = None
            self.model_router = None
            self.fallback_mgr = None
            self.backends = {}
        self.memory = MemorySystem()
        self.validator = OutputValidator()
        self.budget = TokenBudgetManager(BudgetConfig())
        
        # Session tracking
        self.llm.budget = self.budget
        if self._multi_mode and self.multi_llm:
            for c in self.multi_llm._clients.values():
                c.budget = self.budget
        
        # Load experts
        n_experts = ExpertRegistry.load_all()
        print(f"  [AIStaff] Loaded {n_experts} expert(s)")
        
        self.expert = ExpertRegistry.get(expert_id) or ExpertRegistry.get("generalist")
        print(f"  [AIStaff] Active expert: {self.expert.name} ({self.expert.id})")
        
        # Initialize agents
        self.agents = {
            "cot": CoTAgent(self.llm, self.memory, self.validator),
            "executor": ExecutorAgent(self.llm, self.memory, self.validator),
            "review": ReviewAgent(self.llm, self.memory, self.validator),
            "memory": MemoryAgent(self.llm, self.memory, self.validator),
        }
        
        # V1 workflow已删除，V5 CollabLoop替代
        
        # Session tracking
        self.session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        self.messages: list[dict] = []
        
        # ═══ V5: Collaboration Loop (AI↔AI闭环) ═══
        from ..agents.collab_loop import CollaborationLoop
        self._collab_loop: Optional[CollaborationLoop] = None
        
        # ═══ V4 COMPONENTS (lazy init) ═══
        self._self_improve: Optional[SelfImprovementEngine] = None
        self._wf_gen: Optional[WorkflowGeneratorV2] = None
        self._wf_exe: Optional[WorkflowExecutorV2] = None
        self._config_path: str = ""  # For hot-reload
    
    def _attach_ai_router(self, registry):
        """Attach AI Router with model registry for smart routing."""
        from ..backends.ai_router import AIRouter
        self._ai_router = AIRouter(registry, self.multi_llm._clients if self.multi_llm else {})
        self._model_registry = registry

    def _get_ai_router(self) -> Optional[AIRouter]:
        """Get or create AI Router."""
        if not hasattr(self, '_ai_router') or not self._ai_router:
            if hasattr(self, '_model_registry'):
                self._attach_ai_router(self._model_registry)
        return getattr(self, '_ai_router', None)

    def smart_route(self, user_input: str) -> dict:
        """
        AI-driven model routing: let the strongest model decide which model to use.
        
        Returns RouteDecision with model selection + reasoning.
        Falls back to rule-based routing if AI unavailable.
        """
        router = self._get_ai_router()
        if router:
            return router.route(user_input)
        # Fallback: use legacy ModelRouter
        if self._multi_mode and self.model_router:
            profile = self.model_router.route(user_input)
            return {
                "model_name": profile.model,
                "profile_key": profile.name,
                "reason": f"Rule-based: tier={profile.tier}",
                "task_type": "auto",
                "complexity": 5,
                "needs_review": False,
                "confidence": 0.5,
            }
        return {
            "model_name": self.llm.model,
            "profile_key": "default",
            "reason": "No router available",
            "task_type": "auto",
            "complexity": 5,
            "needs_review": False,
            "confidence": 0.3,
        }

    def _get_si(self) -> SelfImprovementEngine:
        """Lazy-init Self-Improvement Engine."""
        if not self._self_improve:
            # Use cheapest available backend for reflection
            si_llm = self.llm
            if self._multi_mode and self.multi_llm:
                cheap = [p for p in self.backends.values()
                         if p.tier in ("free", "cheap") and p.enabled]
                if cheap:
                    si_llm = self.multi_llm._clients.get(cheap[0].name, self.llm)
            self._self_improve = SelfImprovementEngine(self.memory, si_llm)
        return self._self_improve
    
    def _get_collab_loop(self):
        """Lazy-init V5 Collaboration Loop — AI↔AI闭环协作引擎"""
        from ..agents.collab_loop import CollaborationLoop
        if not self._collab_loop:
            clients = {}
            if self._multi_mode and self.multi_llm:
                clients = self.multi_llm._clients
            else:
                clients = {"default": self.llm}
            registry = getattr(self, '_model_registry', None)
            self._collab_loop = CollaborationLoop(clients, registry)
        return self._collab_loop
    
    def _get_wf(self) -> tuple[WorkflowGeneratorV2, WorkflowExecutorV2]:
        """Lazy-init Workflow V2 components."""
        if not self._wf_gen or not self._wf_exe:
            wf_llm = self.llm
            if self._multi_mode and self.multi_llm:
                wf_llm = self.llm  # Use primary for generation
            
            self._wf_gen = WorkflowGeneratorV2(wf_llm)
            
            # Executor needs multi_llm or single client
            mllm_for_exec = self.multi_llm if self._multi_mode else None
            if not mllm_for_exec:
                # Wrap single client to have .chat() interface-ish
                mllm_for_exec = self.llm
                
            self._wf_exe = WorkflowExecutorV2(
                self.agents, mllm_for_exec, self.memory, self.validator
            )
        return self._wf_gen, self._wf_exe
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with expert config + learned preferences."""
        parts = [self.expert.system_prompt]
        
        # Add learned context
        ctx = self.memory.build_context_header(self.session_id)
        if ctx:
            parts.insert(0, ctx)
        
        if self.expert.style_hints:
            parts.append(f"\n【风格指南】{self.expert.style_hints}")
        
        return "\n".join(parts)
    
    def chat(self, user_input: str, mode: str = "auto", **kwargs) -> str:
        """
        统一入口 — 用户只需记住这一个方法。
        
        Args:
            user_input: 任何自然语言请求
            mode: 执行模式
                - "auto": 自动分类+路由（推荐，默认）
                - "direct": 快速问答，不审查
                - "code": 代码任务，编码+审查
                - "research": 深度研究，多轮追问
                - "decision": 决策辅助，多维分析
                - "creative": 创意任务
                - "collab": 多专家协作
                - "arena": 跨模型对比
            **kwargs: 传递给底层方法的可选参数
                - output_dir: 保存目录
                - max_iterations: V5最大迭代次数
                - quality_threshold: 质量阈值(0-100)
        
        Returns:
            AI回复的文本内容（纯字符串）
        
        Examples:
            staff.chat("你好")                        # 自动模式
            staff.chat("写个快排", mode="code")        # 代码模式
            staff.chat("AI趋势分析", mode="research")  # 研究模式
        """
        if mode == "arena":
            questions = kwargs.get("questions", [user_input])
            profiles = kwargs.get("profiles", None)
            return self.cross_arena(questions, profiles)
        
        if mode != "auto":
            # 指定模式: 构造一个强制路由，不走分类器
            return self._chat_forced_mode(user_input, mode, **kwargs)
        
        # auto模式: 用V5闭环协作
        result = self.auto_run_v5(
            user_input,
            output_dir=kwargs.get("output_dir", ""),
            max_iterations=kwargs.get("max_iterations", 0),
            quality_threshold=kwargs.get("quality_threshold", 80),
        )
        
        # 提取主要交付物
        if result.deliverables:
            # 返回第一个非report非transcript的交付物
            for key in ("answer.txt", "solution.py", "research_report.md",
                       "decision_report.md", "creative_output.md"):
                if key in result.deliverables:
                    return result.deliverables[key]
            # fallback: 返回第一个交付物
            return list(result.deliverables.values())[0]
        return result.transcript or "[No output]"
    
    def _chat_forced_mode(self, user_input: str, mode: str, **kwargs) -> str:
        """强制模式: 跳过分类器，直接路由到指定执行路径"""
        from ..experts.classifier import TaskStrategy
        
        mode_config = {
            "direct":    {"mode": "direct", "name": "快速问答", "experts": ["generalist"], "primary": "generalist", "review": False, "fmt": "text", "rounds": 1, "followups": [], "desc": "强制快速问答"},
            "code":      {"mode": "code", "name": "代码任务", "experts": ["coder", "critic"], "primary": "coder", "review": True, "fmt": "code", "rounds": 2, "followups": [], "desc": "强制编码+审查"},
            "research":  {"mode": "research", "name": "深度研究", "experts": ["researcher"], "primary": "researcher", "review": False, "fmt": "markdown_report", "rounds": 4, "followups": [], "desc": "强制多轮研究"},
            "decision":  {"mode": "decision", "name": "决策辅助", "experts": ["planner", "researcher", "critic"], "primary": "planner", "review": True, "fmt": "markdown_report", "rounds": 2, "followups": [], "desc": "强制多维决策"},
            "creative":  {"mode": "creative", "name": "创意任务", "experts": ["writer", "critic"], "primary": "writer", "review": True, "fmt": "text", "rounds": 2, "followups": [], "desc": "强制创意+审查"},
            "collab":    {"mode": "collaborate", "name": "多专家协作", "experts": ["planner", "researcher", "coder", "critic"], "primary": "planner", "review": True, "fmt": "folder", "rounds": 2, "followups": [], "desc": "强制多专家协作"},
        }
        
        cfg = mode_config.get(mode)
        if not cfg:
            return self.chat(user_input)  # fallback to auto
        
        strategy = TaskStrategy(
            mode=cfg["mode"], display_name=cfg["name"],
            experts=cfg["experts"], primary_expert=cfg["primary"],
            needs_review=cfg["review"], output_format=cfg["fmt"],
            max_rounds=cfg["rounds"], auto_followups=cfg["followups"],
            description=cfg["desc"],
        )
        
        # 用auto_run的内部路由（不走V5闭环，但走V3分类执行）
        total_start = time.time()
        result = CollaborationResult(
            goal=user_input, strategy_mode=strategy.mode,
            experts_used=strategy.experts.copy(),
        )
        
        try:
            if strategy.mode == "direct":
                output, stats = self._execute_direct(user_input, strategy)
                result.deliverables["answer.txt"] = output
            elif strategy.mode == "code":
                output, stats = self._execute_code_task(user_input, strategy)
                result.deliverables["solution.py"] = output
            elif strategy.mode == "research":
                output = self._execute_research(user_input, strategy)
                result.deliverables["research_report.md"] = output
            elif strategy.mode == "decision":
                output = self._execute_decision(user_input, strategy)
                result.deliverables["decision_report.md"] = output
            elif strategy.mode == "creative":
                output, stats = self._execute_creative(user_input, strategy)
                result.deliverables["creative_output.md"] = output
            elif strategy.mode == "collaborate":
                collab_result = self.collaborate(goal=user_input, experts=strategy.experts,
                                                  max_rounds=strategy.max_rounds)
                return list(collab_result.deliverables.values())[0] if collab_result.deliverables else collab_result.transcript
            result.status = "success"
        except Exception as e:
            result.status = "failed"
            result.deliverables["error.txt"] = f"[ERROR] {type(e).__name__}: {e}"
        
        # 返回第一个有内容的交付物
        for v in result.deliverables.values():
            if v and not v.startswith("[ERROR]"):
                return v
        return list(result.deliverables.values())[0] if result.deliverables else "[No output]"
    
    def chat_single(self, user_input: str, include_thinking: bool = False,
                    expert: 'ExpertConfig' = None) -> tuple[str, dict]:
        """
        Single-turn chat with full pipeline:
        System Prompt → CoT(optional) → Execute → Review(optional) → Memory → Output
        
        Args:
            user_input: User's message
            include_thinking: Whether to include CoT output
            expert: Override expert for this call (thread-safe, doesn't mutate self.expert)
        """
        active_expert = expert or self.expert
        self.messages.append({"role": "user", "content": user_input})
        
        task_state = TaskState(task_id=f"chat_{len(self.messages)}")
        start_time = time.time()
        
        try:
            # STEP 1: Plan (CoT) - for complex tasks (intelligent trigger)
            if CoTAgent.should_trigger(user_input):
                task_state.state = AgentState.PLANNING
                plan_result = self.agents["cot"].run(task_state, active_expert, self.messages)
            else:
                plan_result = ""
            
            # STEP 2: Execute
            task_state.state = AgentState.EXECUTING
            output = self.agents["executor"].run(task_state, active_expert, self.messages)
            
            # STEP 3: Review (if enabled for this expert)
            if active_expert.require_review:
                task_state.state = AgentState.REVIEWING
                review_result = self.agents["review"].run(task_state, active_expert, self.messages)
                task_state.review_result = review_result
                
                if not review_result.passed and task_state.retry_count < MAX_RETRIES:
                    task_state.state = AgentState.RETRYING
                    task_state.retry_count += 1
                    print(f"    [Review] Score={review_result.score:.1f}, issues={len(review_result.issues)}; retrying...")
                    # Re-execute with review feedback
                    review_feedback = "\n".join(["- " + issue for issue in review_result.issues])
                    self.messages.append({
                        "role": "system",
                        "content": f"[质量审查反馈] 上次输出存在问题：{review_feedback}。请针对这些问题改进你的回答。"
                    })
                    output = self.agents["executor"].run(task_state, active_expert, self.messages)
            
            # STEP 4: Memorize (async-like, every few turns)
            task_state.state = AgentState.MEMORIZING
            if len(self.messages) >= 4 and len(self.messages) % 4 == 0:
                try:
                    self.agents["memory"].run(task_state, active_expert, self.messages, self.session_id)
                except Exception as e:
                    print(f"    [Memory] Background summary failed: {e}")
            
            # Finalize
            task_state.state = AgentState.COMPLETED
            task_state.final_output = output
            self.messages.append({"role": "assistant", "content": output})
            
            # Save to memory
            self.memory.save_message(self.session_id, "user", user_input, model=self.llm.model, expert_id=active_expert.id)
            self.memory.save_message(self.session_id, "assistant", output, model=self.llm.model, expert_id=active_expert.id)
            
            duration = time.time() - start_time
            stats = {
                "chars": len(output), "time": round(duration, 1),
                "review_score": task_state.review_result.score if task_state.review_result else None,
                "retries": task_state.retry_count
            }
            
            return output, stats
        
        except Exception as e:
            task_state.state = AgentState.FAILED
            task_state.error = str(e)
            bus.publish(Event(EventType.TASK_ERROR, {"error": str(e)[:200]}, source="AIStaff"))
            raise
    
    def cross_arena(self, questions: list[str], 
                    profiles: Optional[list[str]] = None) -> str:
        """
        CROSS-API ARENA: Compare responses across DIFFERENT LLM providers.
        
        This is THE killer feature that NO other framework offers out of the box:
        - OpenAI GPT-4o vs Google Gemini vs DeepSeek vs Ollama local
        - Same question, simultaneously sent to ALL configured backends
        - Auto-generates side-by-side comparison report with cost analysis
        
        Args:
            questions: List of questions to ask all backends
            profiles: Which backends to test (default: all enabled)
        
        Returns:
            Markdown-formatted comparison report
        
        Example:
            staff = AIStaff(profiles={...})
            report = staff.cross_arena([
                "解释量子纠缠",
                "写一个Python快速排序",
                "1+1等于几",
            ])
            print(report)  # Full comparison table with cost/timing/quality
        """
        if not self._multi_mode or not self.multi_llm:
            return "[ERROR] cross_arena requires multi-backend mode (profiles=)"
        
        target_profiles = profiles or self.multi_llm.active_profiles
        
        print(f"\n{'='*60}")
        print(f"  🏟️  CROSS-ARENA: {len(target_profiles)} APIs × {len(questions)} Questions")
        print(f"  Backends: {', '.join(target_profiles)}")
        print(f"{'='*60}")
        
        results: dict[str, list[dict]] = {}
        total_cost = 0.0
        
        for qi, question in enumerate(questions):
            print(f"\n  --- Q{qi+1}/{len(questions)}: {question[:50]}... ---")
            
            msgs = [{"role": "user", "content": question}]
            
            # Call ALL backends for this question
            backend_results = self.multi_llm.chat_all(
                msgs, max_tokens=2048, parallel=False
            )
            
            for prof_name, (content, usage) in backend_results.items():
                if prof_name not in results:
                    results[prof_name] = []
                
                is_error = content.startswith("[ERROR]")
                entry = {
                    "q": question,
                    "a": content[:1000] + ("..." if len(content) > 1000 else ""),
                    "chars": len(content),
                    "time": usage.get("time_seconds", 0),
                    "cost": usage.get("estimated_cost_usd", 0),
                    "tokens": usage.get("total_tokens", 0),
                    "model": usage.get("model", "?"),
                    "tier": usage.get("tier", "?"),
                    "error": usage.get("error", "") if is_error else "",
                }
                results[prof_name].append(entry)
                total_cost += entry["cost"]
                
                status = "FAIL" if is_error else f"{entry['chars']}ch/{entry['time']}s"
                cost_str = f"${entry['cost']:.6f}" if entry['cost'] > 0 else "free"
                print(f"    [{prof_name:>12s}] {entry.get('model','?'):<20s} | {status:<25s} | {cost_str}")
            
            time.sleep(0.5)
        
        return self._format_cross_arena_report(results, questions, total_cost)

    def _format_cross_arena_report(self, results: dict, questions: list, 
                                    total_cost: float) -> str:
        """Format cross-arena results into rich Markdown comparison."""
        lines = [
            f"# 🏟️  AI-Staff Cross-Arena Report",
            f"",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Backends Tested:** {', '.join(results.keys())}",
            f"**Questions:** {len(questions)}",
            f"**Total Est. Cost:** ${total_cost:.6f}",
            f"",
        ]
        
        # Summary table: one row per backend
        lines.append("## 📊 Backend Summary")
        lines.append("")
        lines.append("| Backend | Model | Tier | Total Chars | Total Time | Cost | Errors |")
        lines.append("|---------|-------|------|-------------|------------|------|--------|")
        
        for pname, qresults in results.items():
            if not qresults:
                continue
            model = qresults[0].get("model", "?")
            tier = qresults[0].get("tier", "?")
            total_chars = sum(r["chars"] for r in qresults)
            total_time = sum(r["time"] for r in qresults)
            cost = sum(r["cost"] for r in qresults)
            errors = sum(1 for r in qresults if r.get("error"))
            lines.append(f"| **{pname}** | {model} | {tier} | {total_chars} | {total_time:.1f}s | ${cost:.6f} | {errors} |")
        
        lines.append("")
        
        # Per-question detail
        for qi, q in enumerate(questions):
            lines.append(f"\n## Q{qi+1}: {q}")
            lines.append("")
            
            for pname, qresults in results.items():
                if qi < len(qresults):
                    r = qresults[qi]
                    error_tag = " ❌" if r.get("error") else ""
                    lines.append(f"### 🔹 {pname}/{r.get('model','?')}{error_tag}")
                    lines.append(f"- **Chars:** {r['chars']} | **Time:** {r['time']}s | **Cost:** ${r['cost']:.6f} | **Tokens:** {r['tokens']}")
                    if r.get("error"):
                        lines.append(f"- **Error:** `{r['error']}`")
                    else:
                        # Show truncated response
                        preview = r['a'].replace('\n', ' ')[:300]
                        lines.append(f"- **Response:** {preview}")
                    lines.append("")
        
        lines.append("---")
        lines.append(f"*Generated by AI-Staff V{VERSION} Cross-Arena Engine*")
        return "\n".join(lines)

    def research(self, topic: str, depth: int = 3) -> str:
        """Deep research mode with iterative questioning."""
        researcher = ExpertRegistry.get("researcher") or self.expert
        
        outputs = []
        
        # Initial deep dive
        research_prompt = f"""你是一位资深研究员。请对以下话题进行深度研究分析。

## 研究话题
{topic}

请从以下维度全面分析（每个维度都要深入，不要泛泛而谈）：
1. **背景与现状**: 这个话题的核心概念是什么？当前发展状况如何？
2. **关键技术/方法论**: 涉及哪些核心技术、方法或框架？
3. **实际应用场景**: 有哪些典型应用案例？
4. **挑战与局限**: 当前面临的主要问题和瓶颈是什么？
5. **未来发展趋势**: 方向如何？有哪些值得关注的突破点？

要求：
- 使用清晰的Markdown格式
- 数据要有来源标注（如知道的话）
- 给出你的独立判断，不只是罗列信息
- 总字数不少于1500字"""

        print(f"\n  --- Research Phase 1: Initial Analysis ---")
        response, stats = self.chat_single(research_prompt, expert=researcher)
        phase1_len = len(response)
        outputs.append(f"# Phase 1: 初始深度分析 ({stats['chars']}ch/{stats['time']}s)\n\n{response}")
        
        # Iterative deepening
        deepen_questions = [
            f"基于以上分析，请进一步深挖「{topic}」中最关键的技术难点或争议点，给出更专业的技术细节和你的判断。",
            f"对于「{topic}」，有哪些容易被忽视的重要方面或者常见的认知误区？请纠正并提供正确理解。",
            f"请从实践者的角度，给出「{topic}」的具体行动指南：入门路径、学习资源、避坑建议、工具推荐等。",
        ]
        
        for i in range(min(depth - 1, len(deepen_questions))):
            print(f"\n  --- Research Phase {i+2}: Deepening ---")
            dq = deepen_questions[i]
            response, stats = self.chat_single(dq, expert=researcher)
            outputs.append(f"\n# Phase {i+2}: 追问深化 ({stats['chars']}ch/{stats['time']}s)\n\n{response}")
            time.sleep(1)
        
        # Final synthesis
        print(f"\n  --- Research Final: Synthesis ---")
        synth_prompt = "请基于以上所有研究阶段的内容，生成最终综合报告：\n1. 一句话核心结论\n2. 关键发现TOP5（按重要性排序）\n3. 完整知识图谱（结构化梳理所有知识点的关系）\n4. 推荐下一步行动（具体、可执行）\n\n格式要求：使用清晰的层级结构和表格。"
        response, stats = self.chat_single(synth_prompt, expert=researcher)
        outputs.append(f"\n# 最终综合报告 ({stats['chars']}ch/{stats['time']}s)\n\n{response}")
        
        return "\n\n".join(outputs)
    
    # V3 CORE: auto_run + collaborate — Smart Execution Engine

    def auto_run(self, user_input: str, output_dir: str = "") -> CollaborationResult:
        """
        V3 KILLER FEATURE: One command does it all.
        
        User just says what they want. Framework figures out:
        - What TYPE of task this is (code? research? decision?)
        - Which experts to use (1 expert for simple, N for complex)
        - How many rounds / whether to review
        - What output format to produce
        
        This is FREEDOM: not every task needs a roundtable.
        Simple Q&A → 1 call, done.
        Code → coder + critic, focused loop.
        Research → iterative deep-dive.
        Complex decision → multi-expert collaboration.
        
        Args:
            user_input: Any natural language request from the user
            output_dir: Where to save deliverables (auto-generated if empty)
        
        Returns:
            CollaborationResult with all deliverables, transcript, metrics
        
        Example:
            result = staff.auto_run("帮我写一个Python快速排序")
            result.save("./output")  # → saves code + review + report
            
            result = staff.auto_run("React vs Vue该选哪个")
            result.save("./output")  # → saves comparison table + recommendation
            
            result = staff.auto_run("量子纠缠是什么")  
            result.save("./output")  # → saves direct answer, done in 1 call
        """
        total_start = time.time()
        
        # Step 1: Classify the task
        classifier = TaskClassifier()
        strategy = classifier.classify(user_input)
        
        print(f"\n  📋 策略: {strategy.display_name} ({strategy.mode})")
        print(f"     专家: {', '.join(strategy.experts)} | 轮次: {strategy.max_rounds}")
        print(f"     说明: {strategy.description}")
        
        result = CollaborationResult(
            goal=user_input,
            strategy_mode=strategy.mode,
            experts_used=strategy.experts.copy()
        )
        
        # Step 2: Route to the right execution path
        try:
            if strategy.mode == "direct":
                output, stats = self._execute_direct(user_input, strategy)
                result.deliverables["answer.txt"] = output
                result.transcript = f">> {user_input}\n<< {output}"
                result.quality_score = stats.get('review_score') or 8.0
                
            elif strategy.mode == "code":
                output, stats = self._execute_code_task(user_input, strategy)
                result.deliverables["solution.py"] = output
                result.quality_score = stats.get('review_score') or 7.0
                
            elif strategy.mode == "research":
                output = self._execute_research(user_input, strategy)
                result.deliverables["research_report.md"] = output
                result.quality_score = 8.5
                
            elif strategy.mode == "decision":
                output = self._execute_decision(user_input, strategy)
                result.deliverables["decision_report.md"] = output
                result.quality_score = 8.0
                
            elif strategy.mode == "creative":
                output, _ = self._execute_creative(user_input, strategy)
                result.deliverables["creative_output.md"] = output
                result.quality_score = 7.5
                
            elif strategy.mode == "collaborate":
                collab_result = self.collaborate(goal=user_input, 
                                                   experts=strategy.experts,
                                                   max_rounds=strategy.max_rounds)
                return collab_result
            
            else:
                # Fallback to direct
                output, stats = self._execute_direct(user_input, strategy)
                result.deliverables["answer.txt"] = output
            
            result.rounds_used = strategy.max_rounds
            result.status = "success"
            
        except Exception as e:
            result.status = "failed"
            result.deliverables["error.txt"] = f"[ERROR] {type(e).__name__}: {e}"
            print(f"\n    [auto_run] ERROR: {e}")
        
        result.total_time_sec = time.time() - total_start
        
        # Auto-save if output_dir specified or generate default
        if not output_dir:
            safe_name = re.sub(r'[\\/:*?"<>|]', '_', user_input[:30])
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(os.getcwd(), f'ai_staff_output_{safe_name}_{timestamp}')
        
        saved_files = result.save(output_dir)
        print(f"\n  💾 已保存到: {output_dir}/ ({len(saved_files)}个文件)")
        
        return result

    def collaborate(self, goal: str, experts: list[str] = None,
                    max_rounds: int = 2, output_dir: str = "") -> CollaborationResult:
        """
        V3 GOAL-DRIVEN COLLABORATION (replaces expert_collab).
        
        Unlike expert_collab which only produces a chat log,
        this produces DELIVERABLES toward a specific GOAL.
        
        Flow:
        Phase 1: Planner decomposes the goal into actionable steps
        Phase 2: Domain experts execute their assigned parts
        Phase 3: Critic reviews each deliverable
        Phase 4: Failed reviews go back for revision (max 2 iterations)
        Phase 5: Synthesize final output
        
        Args:
            goal: The concrete objective (not just a discussion topic!)
            experts: List of expert IDs (default: planner+researcher+coder+critic)
            max_rounds: Max revision rounds
            output_dir: Save location
        
        Returns:
            CollaborationResult with structured deliverables
        """
        total_start = time.time()
        
        if not experts:
            experts = ["planner", "researcher", "coder", "critic"]
        
        # Validate and load experts
        participants = []
        for eid in experts:
            exp = ExpertRegistry.get(eid)
            if exp:
                participants.append(exp)
            else:
                print(f"    [WARN] Expert '{eid}' not found, skipping")
        
        if len(participants) < 2:
            return CollaborationResult(goal=goal, status="failed",
                                       deliverables={"error.txt": "Need at least 2 experts"})
        
        result = CollaborationResult(
            goal=goal,
            strategy_mode="collaborate",
            experts_used=[p.id for p in participants],
        )
        
        lines = [
            f"{'═'*68}",
            f"  🎯 AI-Staff 目标驱动协作 (Goal-Driven Collab)",
            f"{'═'*68}",
            f"  目标: {goal}",
            f"  团队: {', '.join(f'{p.name}({p.id})' for p in participants)}",
            f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"",
        ]
        
        interaction_log: list[dict] = []
        all_deliverables: dict[str, str] = {}
        
        # ── PHASE 1: PLANNING ──
        planner = next((e for e in participants if e.id == "planner"), participants[0])
        lines.append(f"┌{'─'*66}┐")
        lines.append(f"│  📋 Phase 1: 规划分解 [{planner.name}]{' '*26}│")
        lines.append(f"└{'─'*66}┘")
        
        plan_prompt = f"""你是一个项目规划师。请将以下目标分解为可执行的步骤。

## 目标
{goal}

## 可用专家团队
{', '.join(f'{e.name}({e.id})' for e in participants)}

请输出：
1. 执行步骤列表（每个步骤指派给最合适的专家）
2. 每步的预期产出物
3. 关键依赖关系"""
        
        try:
            plan_start = time.time()
            plan_msgs = [{"role": "system", "content": planner.system_prompt},
                        {"role": "user", "content": plan_prompt}]
            
            if self._multi_mode and self.multi_llm:
                plan_output, _plan_usage = self.multi_llm.chat(plan_msgs, temperature=0.6,
                                                               user_input=goal, expert=planner, max_tokens=2048)
            else:
                plan_output, _plan_usage = self.llm.chat_completion(plan_msgs, temperature=0.6, max_tokens=2048)
            
            plan_time = time.time() - plan_start
            all_deliverables["01_plan.md"] = plan_output
            lines.append(f"\n{plan_output}\n")
            lines.append(f"⏱️ {plan_time:.1f}s | 📊 {len(plan_output)}ch")
            
            interaction_log.append({"phase": 1, "expert": planner.name, "action": "plan_decompose",
                                   "chars": len(plan_output), "time": round(plan_time, 1)})
            
            # Parse steps from plan for Phase 2
            steps = self._parse_plan_steps(plan_output, participants)
            print(f"    [Phase 1] {planner.name}: 分解出{len(steps)}个步骤")
        except Exception as e:
            lines.append(f"[规划失败] {e}")
            steps = [{"expert_id": participants[1].id if len(participants) > 1 else participants[0].id,
                      "task": goal, "output_key": "draft"}]
        
        # ── PHASE 2: EXECUTION (each step by best expert) ──
        lines.append(f"\n┌{'─'*66}┐")
        lines.append(f"│  ⚡ Phase 2: 分头执行{' '*37}│")
        lines.append(f"└{'─'*66}┘\n")
        
        for si, step in enumerate(steps):
            exp = ExpertRegistry.get(step["expert_id"])
            if not exp:
                exp = participants[si % len(participants)]
            
            lines.append(f"── 步骤{si+1}: [{exp.name}] {step['task'][:50]} ──")
            
            exec_prompt = f"""你是{exp.name}。根据以下规划和目标，完成你负责的部分。

## 总体目标
{goal}

## 规划（由规划师提供）
{all_deliverables.get('01_plan.md', '无')}

## 你的任务
{step['task']}

请直接给出你的产出内容，不要客套话。"""
            
            try:
                exec_start = time.time()
                exec_msgs = [{"role": "system", "content": exp.system_prompt},
                            {"role": "user", "content": exec_prompt}]
                
                if self._multi_mode and self.multi_llm:
                    exec_output, _exec_usage = self.multi_llm.chat(exec_msgs, temperature=exp.temperature,
                                                                   user_input=step["task"], expert=exp, max_tokens=3072)
                else:
                    exec_output, _exec_usage = self.llm.chat_completion(exec_msgs, temperature=exp.temperature, max_tokens=3072)
                
                exec_time = time.time() - exec_start
                output_key = step.get("output_key", f"{si+1:02d}_{exp.id}.md")
                all_deliverables[output_key] = exec_output
                lines.append(f"\n{exec_output[:800]}{'...(截断)' if len(exec_output)>800 else ''}\n")
                lines.append(f"⏱️ {exec_time:.1f}s | 📊 {len(exec_output)}ch | 文件: {output_key}")
                
                interaction_log.append({"phase": 2, "expert": exp.name, "action": "execute_step",
                                       "chars": len(exec_output), "time": round(exec_time, 1),
                                       "output_file": output_key})
                
                print(f"    [Phase 2] Step{si+1} {exp.name}: {len(exec_output)}ch / {exec_time:.1f}s")
                
            except Exception as e:
                lines.append(f"[执行失败] {e}")
                all_deliverables[f"{si+1}_error.txt"] = f"[ERROR] {e}"
        
        # ── PHASE 3: REVIEW (critic checks each deliverable) ──
        critic = next((e for e in participants if e.id == "critic"), None)
        if critic:
            lines.append(f"\n┌{'─'*66}┐")
            lines.append(f"│  🔍 Phase 3: 质量审查 [{critic.name}]{' '*25}│")
            lines.append(f"└{'─'*66}┘\n")
            
            review_notes = []
            for key, content in list(all_deliverables.items()):
                if key.startswith("error"):
                    continue
                    
                review_prompt = f"""审查以下产出物的质量。
目标: {goal}
产出文件: {key}

内容:
{content[:2000]}

输出格式：
## 审查结果: {key}
### 质量评分 X/10
### 主要问题 (如有)
### 改进建议 (如有)
### 是否通过: 是/否"""
                
                try:
                    rev_msgs = [{"role": "system", "content": critic.system_prompt},
                               {"role": "user", "content": review_prompt}]
                    
                    if self._multi_mode and self.multi_llm:
                        rev_out, _ = self.multi_llm.chat(rev_msgs, temperature=0.2,
                                                         user_input=f"review:{key}", expert=critic, max_tokens=1024)
                    else:
                        rev_out, _ = self.llm.chat_completion(rev_msgs, temperature=0.2, max_tokens=1024)
                    
                    all_deliverables[f"review_{key}"] = rev_out
                    review_notes.append(rev_out)
                    
                    score_match = re.search(r'质量评分\s*(\d+)/?10?', rev_out)
                    passed = bool(re.search(r'是否通过[^是]*是', rev_out))
                    
                    lines.append(f"📝 {key}: 评分{score_match.group(1) if score_match else '?'}/10 {'✅' if passed else '❌'}")
                    
                    # Phase 4: REVISION (if failed)
                    if not passed and key not in ("01_plan.md",):
                        # Find original expert for this deliverable
                        orig_expert_id = key.split("_", 1)[-1].replace(".md","") if "_" in key else ""
                        orig_exp = ExpertRegistry.get(orig_expert_id) or participants[1]
                        
                        revise_prompt = f"""你的产出被审查员提出了修改意见，请根据反馈改进。

原始产出:
{all_deliverables[key][:1500]}

审查意见:
{rev_out}

请直接输出改进后的完整版本。"""
                        
                        try:
                            rev_msgs2 = [{"role": "system", "content": orig_exp.system_prompt},
                                        {"role": "user", "content": revise_prompt}]
                            
                            if self._multi_mode and self.multi_llm:
                                revised, _ = self.multi_llm.chat(rev_msgs2, temperature=orig_exp.temperature,
                                                                  user_input="revise", expert=orig_exp, max_tokens=3072)
                            else:
                                revised, _ = self.llm.chat_completion(rev_msgs2, temperature=orig_exp.temperature, max_tokens=3072)
                            
                            all_deliverables[key] = revised  # Replace with improved version
                            all_deliverables[f"revised_{key}"] = revised
                            
                            lines.append(f"  ✏️  已修正: {key} (+{len(revised)}ch)")
                            
                        except Exception as rev_e:
                            lines.append(f"  ⚠️ 修正失败: {rev_e}")
                    
                except Exception as e:
                    lines.append(f"[审查失败] {e}")
            
            print(f"    [Phase 3] {critic.name}: 审查了{len([k for k in all_deliverables if not k.startswith(('error','review','revised'))])}个产出物")
        
        # ── PHASE 5: SYNTHESIS & PACKAGE ──
        lines.extend(["", f"┌{'─'*66}┐", f"│  📦 最终交付{' '*46}│", f"└{'─'*66}┘", ""])
        
        for fname, content in all_deliverables.items():
            size_kb = len(content.encode('utf-8')) // 1024
            lines.append(f"  📄 {fname} ({size_kb}KB)")
        
        # Build final result
        total_time = time.time() - total_start
        result.deliverables = all_deliverables
        result.interaction_log = interaction_log
        result.total_time_sec = total_time
        result.rounds_used = max_rounds
        result.transcript = "\n".join(lines)
        result.quality_score = min(9.5, 5.0 + math.log1p(len(all_deliverables)) * 1.5)  # 对数增长，不再按文件数满分
        
        # Auto-save
        if not output_dir:
            safe_goal = re.sub(r'[\\/:*?"<>|]', '_', goal[:25])
            output_dir = os.path.join(os.getcwd(), f"collab_{safe_goal}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        saved = result.save(output_dir)
        lines.append(f"\n💾 全部保存至: {output_dir}/ ({len(saved)}个文件)")
        lines.append(f"\n总耗时: {total_time:.1f}s | 产出物: {len(all_deliverables)}个")
        lines.append(f"---\n*AI-Staff V{VERSION} Goal-Driven Collaboration*")
        
        result.transcript = "\n".join(lines)
        
        return result

    # ── Internal execution methods for auto_run routing ──
    
    def _execute_direct(self, user_input: str, strategy: TaskStrategy) -> tuple[str, dict]:
        """Simple Q&A: single call, one expert, done."""
        exp = ExpertRegistry.get(strategy.primary_expert) or self.expert
        return self.chat_single(user_input, expert=exp)

    def _execute_code_task(self, user_input: str, strategy: TaskStrategy) -> tuple[str, dict]:
        """Code task: coder writes -> critic reviews."""
        coder = ExpertRegistry.get("coder") or self.expert
        critic = ExpertRegistry.get("critic")
        _stats = {}
        
        code_output, _stats = self.chat_single(
            f"请编写代码实现以下需求。只输出完整可运行的代码和必要注释。\n\n## 需求\n{user_input}",
            expert=coder,
        )
        
        result = code_output
        if critic and strategy.needs_review:
            review_prompt = f"审查以下代码的正确性、健壮性、性能和最佳实践。\n\n```python\n{code_output}\n```\n\n给出：1)评分(1-10) 2)问题清单 3)修复建议 4)如需则给出修正版代码。"
            review, r_stats = self.chat_single(review_prompt, expert=critic)
            result = f"# 代码方案\n\n```python\n{code_output}\n```\n\n---\n\n# 代码审查\n\n{review}"
            _stats['review_score'] = r_stats.get('review_score')
        
        return result, _stats or {}

    def _execute_research(self, topic: str, strategy: TaskStrategy) -> str:
        """Research mode: iterative deep-dive."""
        researcher = ExpertRegistry.get("researcher") or self.expert
        
        research_prompt = f"""请对以下话题进行深度研究分析。

## 研究话题
{topic}

请从多维度全面分析（每个维度都要深入）：
1. **核心概念** — 这是什么？为什么重要？
2. **关键技术与原理** — 涉及哪些核心技术/方法论？
3. **实际应用** — 有哪些典型用例？
4. **挑战与局限** — 当前瓶颈是什么？
5. **发展趋势** — 未来方向？

使用Markdown格式，数据驱动，给出独立判断。1500字以上。"""
        
        output_parts = []
        response, stats = self.chat_single(research_prompt, expert=researcher)
        output_parts.append(f"# 深度研究报告\n\n{response}\n")
        
        followups = strategy.auto_followups or [
            "进一步深挖关键技术细节或争议点。",
            "有哪些认知误区？请纠正并给出正确理解。",
            "给出具体行动指南：入门路径、工具推荐、避坑建议。"
        ]
        
        for i, fq in enumerate(followups[:strategy.max_rounds - 1]):
            try:
                resp, s = self.chat_single(fq, expert=researcher)
                output_parts.append(f"\n## 追问{i+1}\n\n{resp}\n")
                time.sleep(1)
            except Exception:
                break
        
        return "\n".join(output_parts)

    def _execute_decision(self, question: str, strategy: TaskStrategy) -> str:
        """Decision support: multi-perspective analysis."""
        outputs = []
        
        for eid in strategy.experts:
            exp = ExpertRegistry.get(eid)
            if not exp:
                continue
            
            role_prompts = {
                "planner": f"从项目规划和执行角度分析：{question}。给出结构化对比分析。",
                "researcher": f"从信息收集和研究角度调研：{question}。给出数据和事实支撑的结论。",
                "critic": f"从批判性思维角度审视：{question}。指出常见误区、风险因素和盲点。",
            }
            
            prompt = role_prompts.get(eid, question)
            resp, _stats = self.chat_single(prompt, expert=exp)
            outputs.append(f"## {exp.name}视角\n\n{resp}\n")
            time.sleep(0.5)
        
        # Final synthesis
        primary_exp = ExpertRegistry.get(strategy.primary_expert) or self.expert
        synthesis_prompt = f"基于以上多维分析，对「{question}」做出最终推荐。\n\n---\n" + "\n---\n".join(outputs) + "\n---\n\n请综合以上观点，给出：\n1. 一句话结论\n2. 推荐选项及理由\n3. 不同场景下的选择建议"
        final, _ = self.chat_single(synthesis_prompt, expert=primary_exp)
        outputs.append(f"## 综合结论\n\n{final}")
        
        return "\n".join(outputs)

    def _execute_creative(self, prompt: str, strategy: TaskStrategy) -> tuple[str, dict]:
        """Creative task with optional review."""
        writer = ExpertRegistry.get("writer") or self.expert
        
        creative_resp, stats = self.chat_single(prompt, expert=writer)
        result = creative_resp
        
        if strategy.needs_review:
            critic = ExpertRegistry.get("critic")
            if critic:
                review, r_stats = self.chat_single(
                    f"从创意质量、感染力、实用性角度审查以下内容，给出评分(1-10)和改进建议:\n\n{creative_resp}",
                    expert=critic,
                )
                result = f"# 创意产出\n\n{creative_resp}\n\n---\n# 审查意见\n\n{review}"
                stats['review_score'] = r_stats.get('review_score')
        
        return result, stats

    def _parse_plan_steps(self, plan_text: str, available_experts: list) -> list[dict]:
        """Extract executable steps from planner's output."""
        steps = []
        
        # Try to find numbered items or bullet points
        patterns = [
            r'(?:步骤|Step|step)\s*(\d+)[:\.\s]*(.*?)(?:\n|$)',
            r'^(\d+)[\.:\)]\s*(.*?)(?:\n|$)',
            r'^[-*]\s*(.*?)(?:\n|$)',
        ]
        
        found_steps = []
        for pat in patterns:
            found_steps.extend(re.findall(pat, plan_text, re.MULTILINE))
        
        if found_steps:
            for match in found_steps:
                text = match[-1].strip()
                if len(text) > 5:
                    # Match to best available expert
                    best_exp = self._match_expert_for_task(text, available_experts)
                    steps.append({
                        "expert_id": best_exp.id,
                        "task": text,
                        "output_key": f"{len(steps)+1:02d}_{best_exp.id}.md"
                    })
        
        # If parsing failed, create one step per remaining expert (after planner)
        if not steps:
            for exp in available_experts[1:]:  # Skip planner, already did planning
                steps.append({
                    "expert_id": exp.id,
                    "task": f"从{exp.name}的角度分析和推进目标",
                    "output_key": f"{len(steps)+1:02d}_{exp.id}.md"
                })
        
        return steps[:6]  # Cap at 6 steps to prevent runaway

    def _match_expert_for_task(self, task_text: str, experts: list):
        """Find the best expert for a given task description."""
        task_lower = task_text.lower()
        
        # Keyword-to-expert mapping
        mappings = {
            "coder": ["代码", "程序", "实现", "function", "api", "算法", "debug"],
            "researcher": ["研究", "调研", "分析", "趋势", "数据", "背景", "技术"],
            "writer": ["文案", "文档", "说明", "介绍", "描述", "写作"],
            "critic": ["审查", "评估", "风险", "测试", "质量", "review"],
            "planner": ["计划", "流程", "架构", "设计", "方案", "规划"],
        }
        
        best_exp = experts[0] if experts else ExpertRegistry.get("generalist")
        best_score = -1
        
        for exp_id, keywords in mappings.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > best_score:
                # Check this expert is in our available list
                for exp in experts:
                    if exp.id == exp_id:
                        best_exp = exp
                        best_score = score
                        break
        
        return best_exp

    def list_experts(self) -> str:
        """List all available experts."""
        experts = ExpertRegistry.list_all()
        lines = [f"# Available Experts ({len(experts)})\n"]
        for exp in experts:
            marker = " ✅" if exp.id == self.expert.id else ""
            lines.append(f"## {exp.name}{marker}")
            lines.append(f"- **ID:** `{exp.id}`")
            lines.append(f"- **Description:** {exp.description}")
            lines.append(f"- **Domain:** {', '.join(exp.domain_tags) or '-'}")
            lines.append(f"- **Format:** {exp.output_format} | **Review:** {'Yes' if exp.require_review else 'No'}")
            lines.append(f"- **Temp:** {exp.temperature} | **Max Turns:** {exp.max_turns}")
            if exp.model_override:
                lines.append(f"- **Model Override:** {exp.model_override}")
            if exp.api_profile:
                lines.append(f"- **API Backend:** {exp.api_profile}")
            lines.append("")
        return "\n".join(lines)
    
    def get_audit_log(self) -> str:
        """Get full audit log from event bus."""
        return bus.audit_log()
    
    def show_memory_stats(self) -> str:
        """Show memory system statistics."""
        prefs = self.memory.get_preferences(top_k=10)
        sessions_raw = self.memory._get_conn().execute(
            "SELECT COUNT(DISTINCT session_id) as cnt FROM conversations"
        ).fetchone()
        
        lines = [
            "# Memory System Stats",
            "",
            f"**Sessions:** {sessions_raw['cnt'] if sessions_raw else 0}",
            f"**Learned Preferences:** {len(prefs)}",
            "",
            "## Top Preferences:",
        ]
        for p in prefs:
            lines.append(f"- `{p['user_key']}` = {p['user_value']} ({p['confidence']:.0%})")
        
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════
    # V4.0: ZERO-CONFIG LAUNCH SYSTEM ⭐ (委托给startup.py)
    # ═══════════════════════════════════════════════════════

    # Provider templates 保留引用（向后兼容）
    PROVIDER_TEMPLATES = {}  # 实际定义在 startup.py

    @classmethod
    def from_env(cls, model: str = "") -> 'AIStaff':
        """V4: 零配置启动（委托给startup模块）"""
        from .startup import from_env as _from_env
        return _from_env(cls, model=model)

    @classmethod
    def quick_start(cls, api_key: str = "", provider: str = "gemini",
                    proxy: str = "", model: str = "",
                    auto_detect: bool = True,
                    extra_keys: dict = None) -> 'AIStaff':
        """V4: 快速启动（委托给startup模块）"""
        from .startup import quick_start as _quick_start
        return _quick_start(cls, api_key=api_key, provider=provider,
                           proxy=proxy, model=model, auto_detect=auto_detect,
                           extra_keys=extra_keys)

    @classmethod
    def discover_and_start(cls, proxy: str = "") -> 'AIStaff':
        """终极懒人模式（委托给startup模块）"""
        from .startup import discover_and_start as _discover
        return _discover(cls, proxy=proxy)

    @classmethod
    def from_config_file(cls, path: str) -> 'AIStaff':
        """从YAML配置加载（委托给startup模块）"""
        from .startup import from_config_file as _from_config
        return _from_config(cls, path)

    def reload_config(self):
        """Hot-reload configuration from the original YAML file."""
        if not hasattr(self, '_config_path') or not self._config_path:
            print("[V4/reload_config] No config file associated (not loaded from_config_file)")
            return False

        old_backends = set(self.backends.keys())
        new_instance = AIStaff.from_config_file(self._config_path)

        # Migrate state
        self.backends = new_instance.backends
        self.multi_llm = new_instance.multi_llm
        self.llm = new_instance.llm

        new_backends = set(self.backends.keys())
        print(f"  [V4/reload_config] Reloaded. Backends: {old_backends} → {new_backends}")
        return True

    def add_backend(self, profile: BackendProfile):
        """Dynamically add a new API backend at runtime."""
        if not hasattr(self, '_multi_mode') or not self._multi_mode:
            print("[WARN] add_backend() requires multi-backend mode")
            return

        self.backends[profile.name] = profile
        client = LLMClient(profile.base_url, profile.api_key, profile.model,
                          profile.proxy or getattr(self, 'default_proxy', ''))
        client.budget = self.budget
        self.multi_llm._clients[profile.name] = client
        self.multi_llm.router.profiles[profile.name] = profile
        if profile.enabled and profile.tier not in self.multi_llm.router._by_tier:
            self.multi_llm.router._by_tier.setdefault(profile.tier, []).append(profile)

        print(f"  [V4/add_backend] Added: {profile.display_name}")

    def health_check(self) -> dict:
        """
        V4: Health check for all configured backends.
        
        Returns dict with status of each component.
        """
        status = {
            "version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "mode": "multi-backend" if self._multi_mode else "single-backend",
            "experts_loaded": len(ExpertRegistry._experts),
            "backends": {},
        }

        if self._multi_mode and self.multi_llm:
            for name, prof in self.backends.items():
                try:
                    client = self.multi_llm._clients[name]
                    test_msg = [{"role": "user", "content": "ping"}]
                    resp, _usage = client.chat_completion(test_msg, max_tokens=5)
                    status["backends"][name] = {
                        "status": "ok", "model": prof.model,
                        "tier": prof.tier, "latency_ms": "responsive"
                    }
                except Exception as e:
                    status["backends"][name] = {
                        "status": "error", "error": str(e)[:100],
                        "model": prof.model, "tier": prof.tier
                    }
        else:
            # Single backend check
            try:
                resp, _ = self.llm.chat_completion(
                    [{"role": "user", "content": "ping"}], max_tokens=5
                )
                status["backends"]["primary"] = {"status": "ok", "model": self.llm.model}
            except Exception as e:
                status["backends"]["primary"] = {"status": "error", "error": str(e)[:100]}

        return status

    def capabilities(self) -> str:
        """
        V4: Self-describing capability summary for AI-to-AI discovery.
        
        Returns a structured text that other AIs can read to understand
        what this skill can do and how to use it.
        """
        experts = ExpertRegistry.list_all()
        backend_names = list(self.backends.keys()) if self.backends else ['single']

        return f"""# AI-Staff V{VERSION} — Capability Manifest

## 统一入口（推荐）

| Method | Description | Example |
|--------|-------------|---------|
| `chat(prompt)` | **万能入口**：自动分类+路由+闭环协作 | `staff.chat("写个快排")` |
| `chat(prompt, mode="code")` | 强制指定模式 | `staff.chat("分析趋势", mode="research")` |

## 模式列表

| mode值 | 说明 | 适用场景 |
|--------|------|----------|
| `auto` | 自动分类（默认） | 不知道用什么时选这个 |
| `direct` | 快速问答 | 简单问题、翻译、定义 |
| `code` | 编码+审查 | 写代码、debug |
| `research` | 多轮深度研究 | 技术趋势、综述 |
| `decision` | 多维度决策 | 选型对比、买哪个 |
| `creative` | 创意+审查 | 文案、方案、命名 |
| `collab` | 多专家协作 | 复杂多领域任务 |
| `arena` | 跨模型对比 | 模型横评 |

## 高级方法

| Method | Description |
|--------|-------------|
| `auto_run_v5(prompt)` | V5闭环协作引擎（chat()内部调用） |
| `cross_arena(questions)` | 跨API模型对比报告 |
| `health_check()` | 后端健康检查 |
| `list_experts()` | 列出所有专家 |

## 启动方式

```python
# 零配置
staff = AIStaff.from_env()

# 快速启动
staff = AIStaff.quick_start("your-api-key", provider="gemini")

# YAML配置
staff = AIStaff.from_config_file("config.yaml")
```

## Available Experts ({len(experts)})
{chr(10).join(f'- **{e.id}** ({e.name}): {e.description}' for e in experts)}

## Active Backends ({len(backend_names)})
{chr(10).join(f'- **{b}**' for b in backend_names)}

## Version & Info
- Version: {VERSION}
- Dependencies: httpx, pyyaml (optional)
- Python: >= 3.10
"""

    # ═══ V4: AUTO_RUN_V2 + IMPROVE_SELF (instance methods) ═══

    def auto_run_v2(self, user_input: str, output_dir: str = "") -> CollaborationResult:
        """
        V4 ENHANCED auto_run: Uses WorkflowEngine V2 for LLM-generated DAG execution.
        
        Falls back to V3's auto_run if V2 fails.
        """
        total_start = time.time()
        
        try:
            gen, exe = self._get_wf()
            experts = ExpertRegistry.list_all()
            
            hints = {}
            if len(user_input) < 50:
                hints["max_steps"] = 2
                hints["complexity_hint"] = "simple"
            elif len(user_input) > 300:
                hints["complexity_hint"] = "complex"

            graph = gen.generate(user_input, experts, constraints=hints)
            final_output, exec_stats = exe.execute(graph, user_input, self.session_id)
            
            wf_meta = exec_stats.get("graph_metadata", {})
            
            result = CollaborationResult(
                goal=user_input,
                strategy_mode=f"v2_dag_{wf_meta.get('complexity', '?')}",
                experts_used=list(set(n.expert_id for n in graph.nodes)),
                deliverables={
                    "output.md": final_output,
                    "workflow_info.json": json.dumps({
                        "goal": graph.goal,
                        "nodes": [{"id": n.node_id, "expert": n.expert_id,
                                  "action": n.action} for n in graph.nodes],
                        "edges": graph.edges,
                        "reasoning": wf_meta.get("reasoning", ""),
                    }, ensure_ascii=False, indent=2),
                },
                transcript=(
                    f"V2 Dynamic DAG Workflow\n"
                    f"Complexity: {wf_meta.get('complexity', '?')}\n"
                    f"Reasoning: {wf_meta.get('reasoning', 'N/A')}\n"
                    f"Nodes: {len(graph.nodes)} | Edges: {len(graph.edges)}\n\n---\n\n"
                    f"{final_output[:800]}{'...(truncated)' if len(final_output)>800 else ''}"
                ),
                quality_score=self._estimate_quality(final_output),
                total_time_sec=exec_stats.get("total_time", time.time() - total_start),
                rounds_used=1,
                total_tokens=0,
            )
            result.status = "success"
            
            # Trigger self-improvement hook
            si = self._get_si()
            si.on_task_complete(result)
            
        except Exception as e:
            print(f"  ⚠️ [auto_v2] V2失败 ({type(e).__name__}), 回退到V3...")
            return self.auto_run(user_input, output_dir=output_dir)

        if not output_dir:
            safe_name = re.sub(r'[\\/:*?"<>|]', '_', user_input[:25])
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(os.getcwd(), f'ai_staff_v4_{safe_name}_{timestamp}')
        
        saved_files = result.save(output_dir)
        print(f"\n  💾 V2结果已保存: {output_dir}/ ({len(saved_files)}个文件)")
        return result

    def improve_self(self, force: bool = False) -> str:
        """Manually trigger a self-reflection cycle."""
        si = self._get_si()
        if force:
            return si.manual_improve()
        si._trigger_reflection()
        return si.get_log()

    def _estimate_quality(self, text: str) -> float:
        """Rough quality estimate (0-10) without calling LLM."""
        if not text or text.startswith("[ERROR]"):
            return 2.0
        score = 5.0
        if len(text) > 200: score += 1.0
        if len(text) > 800: score += 1.0
        if bool(re.search(r'#{1,3}\s', text)): score += 1.0
        if text.count('\n') > 5: score += 0.5
        if not bool(re.search(r'[\u4e00-\u9fff]', text)) and not text.startswith(('```','{')):
            score -= 0.5
        return min(10.0, max(0.0, score))
    
    def auto_run_v5(self, user_input: str, output_dir: str = "",
                    max_iterations: int = 0, quality_threshold: int = 80) -> CollaborationResult:
        """
        V5 ENHANCED: AI↔AI 闭环协作
        
        核心改变（对比V4 auto_run）：
          V4: Classify → Execute → (maybe Review) → 结束
          V5: Classify → Execute → Review → 分低? → 带反馈重执行 → Review → ... → 分高才结束
        
        关键特性：
          1. 多模型分工：快模型执行，强模型审查
          2. 结构化反馈：审查结果必须包含具体问题和建议
          3. 反馈闭环：执行者收到具体反馈后针对性修正
          4. 自动终止：达到质量阈值或最大迭代次数才停止
          5. 全程可观测：trace_id追踪每一步
        """
        total_start = time.time()
        
        # Step 1: 分类任务
        classifier = TaskClassifier()
        strategy = classifier.classify(user_input)
        
        print(f"\n  📋 V5策略: {strategy.display_name} ({strategy.mode})")
        print(f"     专家: {', '.join(strategy.experts)} | 需要审查: {strategy.needs_review}")
        
        # Step 2: 构建路由上下文
        from ..agents.collab_loop import RouteContext
        loop = self._get_collab_loop()
        
        route_ctx = loop._auto_route(user_input)
        # 覆盖迭代参数
        if max_iterations > 0:
            route_ctx.max_iterations = max_iterations
        route_ctx.quality_threshold = quality_threshold
        # 简单任务不需要审查
        if strategy.mode == "direct" and not strategy.needs_review:
            route_ctx.needs_review = False
            route_ctx.max_iterations = 1
        
        # Step 3: 选择专家
        expert = ExpertRegistry.get(strategy.primary_expert) or self.expert
        
        # Step 4: 执行闭环协作
        final_output, collab_stats = loop.run(
            task=user_input, expert=expert, route_ctx=route_ctx,
        )
        
        total_time = time.time() - total_start
        
        # Step 5: 构建 CollaborationResult
        trace_id = collab_stats.get("trace_id", "")
        result = CollaborationResult(
            goal=user_input,
            status=collab_stats.get("status", "success"),
            strategy_mode=f"v5_{strategy.mode}",
            trace_id=trace_id,
            deliverables={},
            transcript=self._build_v5_transcript(user_input, collab_stats, loop),
            quality_score=collab_stats.get("final_score", 0) / 10.0,  # 转为0-10
            rounds_used=collab_stats.get("iterations", 1),
            total_time_sec=total_time,
            experts_used=strategy.experts.copy(),
        )
        
        # 根据任务类型命名交付物
        ext_map = {
            "code": "solution.py", "research": "research_report.md",
            "decision": "decision_report.md", "creative": "creative_output.md",
        }
        fname = ext_map.get(strategy.mode, "answer.txt")
        result.deliverables[fname] = final_output
        
        # 自动保存
        if not output_dir:
            safe_name = re.sub(r'[\\/:*?"<>|]', '_', user_input[:30])
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(os.getcwd(), f'ai_staff_v5_{safe_name}_{timestamp}')
        
        saved_files = result.save(output_dir)
        print(f"\n  💾 V5结果已保存: {output_dir}/ ({len(saved_files)}个文件)")
        print(f"     最终评分: {collab_stats.get('final_score', 'N/A')}/100")
        print(f"     迭代次数: {collab_stats.get('iterations', 1)}")
        print(f"     总耗时: {total_time:.1f}s")
        
        # 触发自改进（仅复杂任务）
        if strategy.mode not in ("direct",) and result.quality_score < 8.0:
            try:
                si = self._get_si()
                si.on_task_complete(result)
            except Exception as e:
                print(f"     [self-improve] skip: {e}")
        
        return result
    
    def _build_v5_transcript(self, task: str, stats: dict, loop) -> str:
        """构建V5执行记录"""
        lines = [
            f"{'═'*60}",
            f"  🔄 AI-Staff V5 闭环协作执行记录",
            f"{'═'*60}",
            f"  任务: {task[:80]}",
            f"  状态: {stats.get('status', '?')}",
            f"  评分: {stats.get('final_score', '?')}/100",
            f"  迭代: {stats.get('iterations', '?')}次",
            f"  耗时: {stats.get('total_time', 0):.1f}s",
            f"  Writer: {stats.get('writer_model', '?')}",
            f"  Reviewer: {stats.get('reviewer_model', '?')}",
            f"",
        ]
        
        # 追踪日志
        trace = loop.get_trace()
        if trace:
            lines.append("  执行追踪:")
            for t in trace[-10:]:  # 最近10条
                score_str = f" score={t['feedback_score']}" if t.get('feedback_score') is not None else ""
                lines.append(f"    [{t['phase']}] #{t['iteration']} {t['model']}{score_str}")
                lines.append(f"      {t['content_preview'][:60]}...")
        
        lines.extend(["", "---", f"*AI-Staff V{VERSION} Collaboration Loop Engine*"])
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# V4.0: SELF-IMPROVEMENT ENGINE ⭐ (Module-level classes)
# AI reflects on its own performance and improves itself
# ═══════════════════════════════════════════════════════════
