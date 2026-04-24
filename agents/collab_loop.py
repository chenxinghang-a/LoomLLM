"""
AI Collaboration Loop — 真正的AI↔AI闭环协作

核心改变：
  V4: Executor → Reviewer → 结束（不管分低分高）
  V5: Executor → Reviewer → 分低? → Executor(带反馈) → Reviewer → ... → 分高才结束

设计原则：
  1. 反馈必须结构化（JSON），程序才能自动判定
  2. 修正必须带具体问题点，不是笼统的"请改进"
  3. 循环必须有上限，防止无限循环
  4. 多模型分工：强模型审查，快模型执行
  5. 整个过程可观测（trace_id + 状态机）
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..backends.client import LLMClient
from ..backends.smart_init import ModelRegistry, ModelInfo
from ..core.events import EventBus, Event, EventType, bus
from ..experts.registry import ExpertRegistry, ExpertConfig
from ..core.validation import ValidationResult


class CollabError(Exception):
    """协作循环异常"""
    pass


# ═══════════════════════════════════════════════════════════
#  协作状态机
# ═══════════════════════════════════════════════════════════

class CollabPhase(str, Enum):
    PLANNING = "planning"      # 规划：怎么拆任务
    EXECUTING = "executing"    # 执行：写初稿
    REVIEWING = "reviewing"    # 审查：结构化评审
    REBUTTAL = "rebuttal"      # 辩论：Executor对Review的批评进行辩解
    REVISING = "revising"      # 修正：根据反馈改
    COMPLETED = "completed"    # 完成：质量达标
    FAILED = "failed"          # 失败：超过重试上限


# ═══════════════════════════════════════════════════════════
#  结构化反馈协议
# ═══════════════════════════════════════════════════════════

@dataclass
class StructuredFeedback:
    """结构化审查反馈 — 这是闭环的关键！"""
    score: int                      # 0-100
    passed: bool                    # 是否通过（>= threshold）
    issues: list[str]               # 具体问题列表
    suggestions: list[str]          # 具体改进建议
    strengths: list[str]            # 做得好的地方（给AI正向反馈）
    discussion: str = ""            # 讨论模式：Reviewer的补充观点
    rebuttal: str = ""              # 辩论模式：Executor的辩解
    
    def to_revision_prompt(self) -> str:
        """生成修正指令 — 告诉执行者具体要改什么"""
        parts = [
            "## 审查反馈（请针对以下问题逐一修正）\n",
        ]
        if self.discussion:
            parts.append(f"### 合作者的讨论\n{self.discussion}\n")
        if self.rebuttal:
            parts.append(f"### 辩解记录\n{self.rebuttal}\n")
        if self.strengths:
            parts.append("### ✅ 做得好的部分（保持）")
            for s in self.strengths[:3]:
                parts.append(f"- {s}")
            parts.append("")
        if self.issues:
            parts.append("### ❌ 需要修正的问题")
            for i, issue in enumerate(self.issues, 1):
                parts.append(f"{i}. {issue}")
            parts.append("")
        if self.suggestions:
            parts.append("### 📝 改进建议")
            for s in self.suggestions:
                parts.append(f"- {s}")
        parts.append(f"\n**当前评分: {self.score}/100** — 目标: >= 80分")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════
#  协作消息包
# ═══════════════════════════════════════════════════════════

@dataclass
class CollabPacket:
    """协作消息包 — AI之间传递的结构化信息"""
    trace_id: str                   # 追踪ID
    phase: CollabPhase              # 当前阶段
    content: str                    # 内容
    feedback: Optional[StructuredFeedback] = None  # 审查反馈
    iteration: int = 0              # 第几轮迭代
    model_used: str = ""            # 使用的模型
    expert_used: str = ""           # 使用的专家
    timestamp: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════
#  路由上下文 — 让路由决策真正传到执行层
# ═══════════════════════════════════════════════════════════

@dataclass
class RouteContext:
    """路由上下文 — 携带AIRouter的决策结果到执行层"""
    task_type: str                  # direct/code/research/decision/creative
    complexity: int                 # 0-10
    writer_model: str               # 执行用的模型（快模型）
    reviewer_model: str             # 审查用的模型（强模型）
    max_iterations: int = 3         # 最大迭代次数
    quality_threshold: int = 80     # 质量阈值（0-100）
    needs_review: bool = True       # 是否需要审查
    reason: str = ""                # 为什么这样路由


# ═══════════════════════════════════════════════════════════
#  AI 协作循环引擎 — 核心！
# ═══════════════════════════════════════════════════════════

class CollaborationLoop:
    """
    真正的AI↔AI闭环协作引擎
    
    流程：
      1. Dispatch: 根据任务分派Writer(快模型) + Reviewer(强模型)
      2. Execute: Writer生成初稿
      3. Review: Reviewer给出结构化反馈（JSON）
      4. Judge: 分数>=阈值? → 完成 : → 进入Revising
      5. Revise: Writer收到具体反馈，针对性修正
      6. 重复3-5，直到达标或超过最大迭代次数
    
    这不是"排队说话"，而是"主编指挥作者和校对在协作文档上迭代"。
    """
    
    # 审查提示词 — 强制JSON输出
    REVIEW_PROMPT = """你是严格的质量审查员。请审查以下AI生成的内容。

【原始任务】
{task}

【AI生成的内容】
{content}

请严格按以下JSON格式输出审查结果（不要输出其他内容）：
```json
{{
  "score": <0-100的整数评分>,
  "passed": <true如果score>=80>,
  "issues": ["具体问题1", "具体问题2"],
  "suggestions": ["具体改进建议1", "具体改进建议2"],
  "strengths": ["做得好的地方1", "做得好的地方2"]
}}
```

评分标准：
- 90-100: 优秀，无需修改
- 80-89: 良好，小修即可
- 60-79: 需要较大修改
- 40-59: 有明显问题
- 0-39: 严重不符合要求

重要：只输出JSON，不要输出任何其他文字！"""
    
    # 讨论模式提示词 — Reviewer不只是打分，而是参与讨论
    DISCUSS_REVIEW_PROMPT = """你是一位资深合作者，正在与另一位AI协作者一起完成一个任务。

【原始任务】
{task}

【协作者的输出】
{content}

请你从以下角度参与讨论：
1. **补充观点**：协作者遗漏了什么重要内容？请补充你的见解。
2. **建设性批评**：有什么可以改进的地方？给出具体的修改建议。
3. **正面反馈**：哪些部分做得好，应该保持？
4. **综合评分**：给整体质量打分。

请严格按以下JSON格式输出（不要输出其他内容）：
```json
{{
  "score": <0-100的整数评分>,
  "passed": <true如果score>=80>,
  "issues": ["需要改进的问题1", "需要改进的问题2"],
  "suggestions": ["具体补充/改进建议1", "具体补充/改进建议2"],
  "strengths": ["做得好的地方1", "做得好的地方2"],
  "discussion": "你的补充观点和讨论内容（2-4句话，这是你与协作者的对话）"
}}
```

重要：只输出JSON！discussion字段是你"说话"的地方。"""
    
    # 修正提示词 — 带具体反馈
    REVISION_PROMPT = """你是一个正在与审查员协作的作者。请根据审查反馈修正你的作品。

【原始任务】
{task}

【你的初稿】
{previous_draft}

【审查反馈】
{feedback_prompt}

请针对反馈中提到的每个问题逐一修正。保持原有优点（strengths中提到的），只修改有问题的部分。输出修正后的完整内容（不是diff）。"""
    
    @staticmethod
    def _safe_truncate(text: str, limit: int = 4000) -> str:
        """按段落边界截断，避免破坏Markdown结构"""
        if len(text) <= limit:
            return text
        # 找到最后一个完整段落
        cut = text[:limit]
        last_nl = cut.rfind('\n\n')
        if last_nl > limit // 2:
            return cut[:last_nl]
        last_line = cut.rfind('\n')
        if last_line > limit // 2:
            return cut[:last_line]
        return cut
    
    def __init__(self, clients: dict[str, LLMClient], registry: ModelRegistry = None):
        """
        Args:
            clients: {profile_key: LLMClient} 多模型客户端
            registry: ModelRegistry，用于选择快模型/强模型
        """
        self._clients = clients
        self._registry = registry
        self._trace_log: list[CollabPacket] = []
    
    def run(self, task: str, expert: ExpertConfig = None,
            route_ctx: RouteContext = None) -> tuple[str, dict]:
        """
        执行闭环协作
        
        Args:
            task: 用户任务描述
            expert: 专家配置（默认generalist）
            route_ctx: 路由上下文（决定用哪个模型、迭代几次等）
        
        Returns:
            (final_output, stats_dict)
        """
        trace_id = uuid.uuid4().hex[:12]
        # 确保expert registry已加载
        if expert is None:
            expert = ExpertRegistry.get("generalist")
        if expert is None:
            # 终极fallback：内联创建
            expert = ExpertConfig(
                id="generalist", name="通用助手",
                description="Fallback expert",
                system_prompt="你是一个专业、高效的AI助手。请用中文回答，结构清晰。",
            )
        
        # 如果没有路由上下文，自动生成
        if not route_ctx:
            route_ctx = self._auto_route(task)
        
        # 选择客户端
        writer_client = self._get_client(route_ctx.writer_model)
        reviewer_client = self._get_client(route_ctx.reviewer_model)
        
        # 如果没有指定的客户端，用默认的
        if not writer_client:
            writer_client = list(self._clients.values())[0] if self._clients else None
        if not reviewer_client:
            reviewer_client = writer_client
        
        if not writer_client:
            bus.publish(Event(EventType.TASK_ERROR, {
                "trace_id": trace_id, "error": "no_client",
            }, source="CollabLoop"))
            raise CollabError("No LLM client available")
        
        start_time = time.time()
        current_draft = ""
        last_feedback: Optional[StructuredFeedback] = None
        
        bus.publish(Event(EventType.TASK_START, {
            "trace_id": trace_id, "task": task[:100],
            "writer": route_ctx.writer_model,
            "reviewer": route_ctx.reviewer_model,
            "max_iterations": route_ctx.max_iterations,
        }, source="CollabLoop"))
        
        print(f"\n  [CollabLoop] trace={trace_id}")
        print(f"     Task: {task[:60]}...")
        print(f"     Writer: {route_ctx.writer_model} | Reviewer: {route_ctx.reviewer_model}")
        print(f"     MaxIter: {route_ctx.max_iterations} | Threshold: {route_ctx.quality_threshold}")
        
        for iteration in range(route_ctx.max_iterations):
            phase = CollabPhase.EXECUTING if iteration == 0 else CollabPhase.REVISING
            
            # ── Phase: Execute / Revise ──
            if phase == CollabPhase.EXECUTING:
                exec_msgs = [
                    {"role": "system", "content": expert.system_prompt},
                    {"role": "user", "content": task},
                ]
                exec_model = route_ctx.writer_model
            else:
                feedback_prompt = last_feedback.to_revision_prompt() if last_feedback else ""
                exec_msgs = [
                    {"role": "system", "content": expert.system_prompt},
                    {"role": "user", "content": self.REVISION_PROMPT.format(
                        task=task,
                        previous_draft=self._safe_truncate(current_draft),
                        feedback_prompt=feedback_prompt,
                    )},
                ]
                exec_model = route_ctx.writer_model
            
            # ── Call LLM with 429 fallback ──
            current_draft, exec_model_used = self._call_with_fallback(
                writer_client, exec_msgs, exec_model,
                temperature=expert.temperature, max_tokens=4096,
                fallback_model=route_ctx.reviewer_model,
                phase_label=f"{phase.value} #{iteration+1}",
            )
            
            if not current_draft:
                print(f"     [{phase.value} #{iteration+1}] EMPTY output, skipping review")
                bus.publish(Event(EventType.TASK_ERROR, {
                    "trace_id": trace_id, "phase": phase.value, "error": "empty_output",
                }, source="CollabLoop"))
                continue
            
            self._trace_log.append(CollabPacket(
                trace_id=trace_id, phase=phase, content=current_draft[:200],
                iteration=iteration, model_used=exec_model_used, expert_used=expert.id,
            ))
            
            # ── 不需要审查的简单任务 → 直接返回 ──
            if not route_ctx.needs_review:
                print(f"     >> Simple task, skip review")
                total_time = time.time() - start_time
                return current_draft, {
                    "status": "success", "iterations": iteration + 1,
                    "final_score": -1, "total_time": round(total_time, 2),
                    "writer_model": exec_model_used, "review_needed": False,
                    "trace_id": trace_id,
                }
            
            # ── Phase: Review ──
            # writer→review之间加短间隔，避免连续调用同一模型429
            if exec_model_used == route_ctx.reviewer_model:
                time.sleep(1)
            
            # 复杂任务(research/decision/creative/collaborate)用讨论模式，简单任务用标准模式
            use_discuss = route_ctx.task_type in ("research", "decision", "creative", "collaborate")
            review_template = self.DISCUSS_REVIEW_PROMPT if use_discuss else self.REVIEW_PROMPT
            review_system = "你是一位资深合作者，正在与另一位AI一起完成任务。请给出结构化JSON审查+讨论。" if use_discuss else "你是严格的质量审查员。只输出JSON格式的审查结果，不要输出其他任何文字。"
            
            review_msgs = [
                {"role": "system", "content": review_system},
                {"role": "user", "content": review_template.format(
                    task=task, content=self._safe_truncate(current_draft),
                )},
            ]
            
            review_raw, review_model_used = self._call_with_fallback(
                reviewer_client, review_msgs, route_ctx.reviewer_model,
                temperature=0.2, max_tokens=1024,
                fallback_model=route_ctx.writer_model,
                phase_label=f"review #{iteration+1}",
            )
            
            if not review_raw:
                # review完全失败，用writer自审
                print(f"     [review #{iteration+1}] Review failed, using self-assessment")
                last_feedback = StructuredFeedback(
                    score=65, passed=False,
                    issues=["审查模型不可用，自动降分"],
                    suggestions=["检查reviewer模型可用性"],
                    strengths=[],
                )
                continue
            
            feedback = self._parse_review(review_raw)
            last_feedback = feedback
            
            self._trace_log.append(CollabPacket(
                trace_id=trace_id, phase=CollabPhase.REVIEWING,
                content=f"score={feedback.score}, issues={len(feedback.issues)}",
                feedback=feedback, iteration=iteration,
                model_used=review_model_used, expert_used="critic",
            ))
            
            print(f"     [review #{iteration+1}] score={feedback.score}/100 "
                  f"issues={len(feedback.issues)} / {review_model_used}")
            
            # ── Phase: Judge ──
            if feedback.score >= route_ctx.quality_threshold:
                total_time = time.time() - start_time
                print(f"     >> PASSED (score={feedback.score} >= {route_ctx.quality_threshold})")
                
                bus.publish(Event(EventType.TASK_COMPLETE, {
                    "trace_id": trace_id, "score": feedback.score,
                    "iterations": iteration + 1, "total_time": round(total_time, 2),
                }, source="CollabLoop"))
                
                return current_draft, {
                    "status": "success",
                    "iterations": iteration + 1,
                    "final_score": feedback.score,
                    "issues_remaining": feedback.issues,
                    "total_time": round(total_time, 2),
                    "writer_model": exec_model_used,
                    "reviewer_model": review_model_used,
                    "trace_id": trace_id,
                }
            else:
                # ── Phase: Rebuttal (辩论协议) ──
                # 低分时：先让Executor辩解，不是盲目改
                if feedback.score >= 50 and iteration < route_ctx.max_iterations - 1:
                    print(f"     >> REBUTTAL (score={feedback.score}, letting writer defend)")
                    
                    rebuttal_msgs = [
                        {"role": "system", "content": expert.system_prompt},
                        {"role": "user", "content": f"""你刚才完成的任务：{task}

审查员给了 {feedback.score}/100 分，指出了以下问题：
{chr(10).join(f'- {i}' for i in feedback.issues)}

请简要辩解（2-4句话）：
1. 你是否同意审查员的判断？
2. 如果不同意，你的理由是什么？
3. 如果同意，你打算怎么改？

注意：诚实回答。如果你的输出确实有问题，直接承认并提出修改方案。只有在你确实有合理理由时才辩解。"""},
                    ]
                    
                    rebuttal_text, _ = self._call_with_fallback(
                        writer_client, rebuttal_msgs, exec_model_used,
                        temperature=0.3, max_tokens=512,
                        fallback_model=route_ctx.reviewer_model,
                        phase_label=f"rebuttal #{iteration+1}",
                    )
                    
                    if rebuttal_text:
                        feedback.rebuttal = rebuttal_text
                        self._trace_log.append(CollabPacket(
                            trace_id=trace_id, phase=CollabPhase.REBUTTAL,
                            content=rebuttal_text[:200],
                            iteration=iteration, model_used=exec_model_used,
                            expert_used=expert.id,
                        ))
                        
                        # Reviewer看到辩解后重新评估
                        rejudge_msgs = [
                            {"role": "system", "content": "你是审查员。执行者对你的批评进行了辩解。请重新评估。"},
                            {"role": "user", "content": f"""你之前给了 {feedback.score}/100 分，问题：
{chr(10).join(f'- {i}' for i in feedback.issues)}

执行者的辩解：
{rebuttal_text}

请重新评估：
1. 执行者的辩解是否合理？
2. 是否需要调整评分？如果辩解有理，可上调5-15分。
3. 哪些问题确实需要改，哪些可以接受？

只输出JSON：{{"new_score": <int>, "accepted_rebuttals": ["接受的辩解1"], "remaining_issues": ["仍需修改1"]}}"""},
                        ]
                        
                        rejudge_raw, _ = self._call_with_fallback(
                            reviewer_client, rejudge_msgs, route_ctx.reviewer_model,
                            temperature=0.2, max_tokens=256,
                            fallback_model=route_ctx.writer_model,
                            phase_label=f"rejudge #{iteration+1}",
                        )
                        
                        if rejudge_raw:
                            try:
                                # 解析rejudge结果
                                import re
                                json_match = re.search(r'\{[^{}]+\}', rejudge_raw, re.DOTALL)
                                if json_match:
                                    rj = json.loads(json_match.group())
                                    new_score = rj.get("new_score", feedback.score)
                                    if new_score > feedback.score:
                                        print(f"     >> REJUDGE: {feedback.score} -> {new_score} (rebuttal accepted)")
                                        feedback.score = min(100, new_score)
                                        feedback.passed = new_score >= route_ctx.quality_threshold
                                        remaining = rj.get("remaining_issues", feedback.issues)
                                        if remaining:
                                            feedback.issues = remaining
                                        # 如果rejudge后通过了，直接返回
                                        if feedback.passed:
                                            total_time = time.time() - start_time
                                            return current_draft, {
                                                "status": "success",
                                                "iterations": iteration + 1,
                                                "final_score": feedback.score,
                                                "issues_remaining": feedback.issues,
                                                "total_time": round(total_time, 2),
                                                "writer_model": exec_model_used,
                                                "reviewer_model": review_model_used,
                                                "rebuttal_accepted": True,
                                                "trace_id": trace_id,
                                            }
                            except (json.JSONDecodeError, Exception):
                                pass  # rejudge失败，继续正常revising
                
                print(f"     >> RETRY (score={feedback.score} < {route_ctx.quality_threshold})")
        
        # 超过最大迭代次数
        total_time = time.time() - start_time
        final_score = last_feedback.score if last_feedback else 0
        print(f"     ⚠️ 超过最大迭代次数 ({route_ctx.max_iterations}), 当前分数={final_score}")
        
        bus.publish(Event(EventType.TASK_COMPLETE, {
            "trace_id": trace_id, "score": final_score,
            "iterations": route_ctx.max_iterations,
            "status": "max_iterations_reached",
        }, source="CollabLoop"))
        
        return current_draft, {
            "status": "max_iterations_reached",
            "iterations": route_ctx.max_iterations,
            "final_score": final_score,
            "issues_remaining": last_feedback.issues if last_feedback else [],
            "total_time": round(total_time, 2),
            "writer_model": route_ctx.writer_model,
            "reviewer_model": route_ctx.reviewer_model,
            "trace_id": trace_id,
        }
    
    def _call_with_fallback(self, primary_client: LLMClient,
                            messages: list[dict], model: str,
                            temperature: float = 0.7, max_tokens: int = 4096,
                            fallback_model: str = "",
                            phase_label: str = "call",
                            max_retries: int = 1) -> tuple[str, str]:
        """
        带模型降级的LLM调用
        
        策略（简化版，LLMClient内部已有3次重试）:
        1. 用primary_client + model调用（LLMClient内部会重试3次）
        2. 失败 → 直接换fallback_model
        3. 还失败 → 轮转所有可用客户端
        
        Returns: (content, actual_model_used)
        """
        actual_model = model
        
        # 尝试1: 用primary client（LLMClient内部已有重试机制）
        try:
            content, _ = primary_client.chat_completion(
                messages, temperature=temperature,
                model=actual_model, max_tokens=max_tokens,
            )
            if content and content.strip():
                return content, actual_model
        except Exception as e:
            err_str = str(e)
            print(f"     [{phase_label}] Primary failed: {err_str[:60]}")
        
        # 尝试2: 降级到fallback模型
        if fallback_model and fallback_model != actual_model:
            fallback_client = self._get_client(fallback_model)
            if fallback_client:
                print(f"     [{phase_label}] Fallback -> {fallback_model}")
                try:
                    content, _ = fallback_client.chat_completion(
                        messages, temperature=temperature,
                        model=fallback_model, max_tokens=max_tokens,
                    )
                    if content and content.strip():
                        return content, fallback_model
                except Exception as e:
                    print(f"     [{phase_label}] Fallback also failed: {str(e)[:60]}")
        
        # 尝试3: 轮转所有可用客户端
        for client_key, client in self._clients.items():
            client_model = getattr(client, 'model', '')
            if client_model in (actual_model, fallback_model):
                continue
            print(f"     [{phase_label}] Last resort: {client_model}")
            try:
                content, _ = client.chat_completion(
                    messages, temperature=temperature,
                    model=client_model, max_tokens=max_tokens,
                )
                if content and content.strip():
                    return content, client_model
            except Exception:
                continue
        
        print(f"     [{phase_label}] ALL attempts exhausted")
        return "", actual_model
    
    def _auto_route(self, task: str) -> RouteContext:
        """自动路由：复用TaskClassifier统一分类，不再重复维护关键词表"""
        from ..experts.classifier import TaskClassifier
        
        classifier = TaskClassifier()
        strategy = classifier.classify(task)
        
        # 简单任务（direct模式且不需要审查）
        if strategy.mode == "direct" and not strategy.needs_review:
            return RouteContext(
                task_type="direct", complexity=0,
                writer_model=self._pick_model("fast"),
                reviewer_model=self._pick_model("strong"),
                max_iterations=1, needs_review=False,
                reason="Simple query, no review needed",
            )
        
        # 根据分类结果映射路由参数
        mode_config = {
            "code":       {"complexity": 7, "max_iter": 2, "threshold": 70, "writer": "fast",   "review": True},
            "research":   {"complexity": 8, "max_iter": 2, "threshold": 70, "writer": "fast",   "review": True},
            "decision":   {"complexity": 6, "max_iter": 2, "threshold": 70, "writer": "fast",   "review": True},
            "creative":   {"complexity": 6, "max_iter": 2, "threshold": 70, "writer": "fast",   "review": True},
            "collaborate":{"complexity": 8, "max_iter": 2, "threshold": 70, "writer": "fast",   "review": True},
        }
        
        cfg = mode_config.get(strategy.mode, {"complexity": 5, "max_iter": 2, "threshold": 70, "writer": "fast", "review": True})
        
        return RouteContext(
            task_type=strategy.mode,
            complexity=cfg["complexity"],
            writer_model=self._pick_model(cfg["writer"]),
            reviewer_model=self._pick_model("strong"),
            max_iterations=cfg["max_iter"],
            quality_threshold=cfg["threshold"],
            needs_review=cfg["review"],
            reason=f"TaskClassifier: {strategy.display_name} ({strategy.mode})",
        )
    
    def _pick_model(self, role: str) -> str:
        """根据角色选择模型：fast(快模型) / strong(强模型)
        
        策略：
          fast: 选最快且免费的模型做Writer（2.5-flash-lite优先）
          strong: 选能力最强的可用模型做Reviewer（3.x优先，更强判断力）
        """
        if not self._registry:
            return "gemini-2.5-flash-lite" if role == "fast" else "gemini-3-flash-preview"
        
        usable = self._registry.usable_models
        if not usable:
            return "gemini-2.5-flash-lite"
        
        if role == "fast":
            # 选最快的免费模型
            free = [m for m in usable if m.is_free]
            if free:
                return min(free, key=lambda m: m.latency_ms).name
            return min(usable, key=lambda m: m.latency_ms).name
        
        else:  # strong/reviewer
            # 优先选3.x系列（更强判断力），其次选最强模型
            v3_models = [m for m in usable if m.name.startswith("gemini-3")]
            if v3_models:
                return max(v3_models, key=lambda m: m.strength_score).name
            return max(usable, key=lambda m: m.strength_score).name
    
    def _get_client(self, model_name: str) -> Optional[LLMClient]:
        """根据模型名找到对应的LLMClient"""
        # 先精确匹配profile key
        key = model_name.replace('-', '_').replace('.', '')
        if key in self._clients:
            return self._clients[key]
        # 加provider前缀
        for k in self._clients:
            if key in k or model_name.replace('-', '_') in k:
                return self._clients[k]
        # 模糊匹配
        for k, client in self._clients.items():
            if hasattr(client, 'model') and model_name in client.model:
                return client
        return None
    
    def _parse_review(self, raw: str) -> StructuredFeedback:
        """解析审查结果为结构化反馈（增强容错）"""
        # 去markdown包裹
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r'^```\w*\n?', '', text)
            text = re.sub(r'\n?```\s*$', '', text)
            text = text.strip()
        
        # 尝试解析JSON
        try:
            data = json.loads(text)
            return StructuredFeedback(
                score=int(data.get("score", 50)),
                passed=bool(data.get("passed", data.get("score", 50) >= 80)),
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                strengths=data.get("strengths", []),
                discussion=data.get("discussion", ""),
            )
        except (json.JSONDecodeError, ValueError):
            pass
        
        # JSON解析失败 — 尝试修复截断的JSON
        # 场景：429重试后返回不完整的JSON，如 {"score": 85, "passed": true, "issues": ["xx
        score = 50
        try:
            # 尝试提取score值
            score_match = re.search(r'"score"\s*:\s*(\d{1,3})', text)
            if score_match:
                score = int(score_match.group(1))
            
            # 尝试提取passed值
            passed_match = re.search(r'"passed"\s*:\s*(true|false)', text)
            passed = passed_match.group(1) == "true" if passed_match else score >= 80
            
            # 尝试提取issues列表（即使被截断）
            issues = re.findall(r'"issues"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            issue_list = []
            if issues:
                issue_list = re.findall(r'"([^"]*)"', issues[0])
            
            # 尝试提取suggestions
            suggestions = re.findall(r'"suggestions"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            sug_list = []
            if suggestions:
                sug_list = re.findall(r'"([^"]*)"', suggestions[0])
            
            if score_match:  # 至少找到了score，算部分解析成功
                return StructuredFeedback(
                    score=min(100, max(0, score)),
                    passed=passed,
                    issues=issue_list or ["JSON解析不完整，请人工检查"],
                    suggestions=sug_list,
                    strengths=[],
                    discussion="",
                )
        except Exception:
            pass
        
        # 完全无法解析
        score_match = re.search(r'(\d{1,3})\s*/\s*100|score\s*[=:]\s*(\d{1,3})', text)
        if score_match:
            score = int(score_match.group(1) or score_match.group(2))
        
        issues = []
        for m in re.finditer(r'(?:问题|issue|缺点|不足)\s*\d*[.::]\s*(.+)', text, re.IGNORECASE):
            issues.append(m.group(1).strip())
        
        suggestions = []
        for m in re.finditer(r'(?:建议|suggest|改进|优化)\s*\d*[.::]\s*(.+)', text, re.IGNORECASE):
            suggestions.append(m.group(1).strip())
        
        return StructuredFeedback(
            score=min(100, max(0, score)),
            passed=score >= 80,
            issues=issues or ["审查结果解析失败，请人工检查"],
            suggestions=suggestions,
            strengths=[],
        )
    
    def get_trace(self) -> list[dict]:
        """获取执行追踪"""
        return [
            {
                "trace_id": p.trace_id,
                "phase": p.phase.value,
                "content_preview": p.content[:100],
                "feedback_score": p.feedback.score if p.feedback else None,
                "iteration": p.iteration,
                "model": p.model_used,
            }
            for p in self._trace_log
        ]


__all__ = [
    'CollaborationLoop', 'RouteContext', 'CollabPhase',
    'StructuredFeedback', 'CollabPacket',
]
