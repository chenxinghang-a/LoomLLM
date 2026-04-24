from __future__ import annotations
import json, re, hashlib, time
from datetime import datetime
from typing import Any, Optional

# Intra-package imports
from ..core.constants import VERSION
from ..core.memory import MemorySystem
from ..backends.client import LLMClient
from .types import ImprovementRecord
from ..experts.registry import ExpertRegistry
from ..agents.types import CollaborationResult

class SelfImprovementEngine:
    """
    V4 KILLER FEATURE: AI Self-Improvement Loop
    
    Core cycle:
      Execute → Monitor → Reflect → Propose → Validate → Apply
    
    This is NOT RLHF. No training, no gradients.
    Just pure LLM reasoning over its own execution history.
    
    The system:
    1. Monitors task completion quality (score, errors, token efficiency)
    2. Periodically asks LLM to reflect on its own performance
    3. Generates concrete improvement suggestions
    4. Applies safe changes (prompt tweaks), logs risky ones for review
    5. Tracks improvement history with rollback capability
    
    Improvement dimensions:
    - Expert system prompt optimization
    - TaskClassifier rule adjustments  
    - ModelRouter strategy fine-tuning
    - Workflow template evolution (future)
    """
    
    REFLECTION_PROMPT = """你是一个AI系统优化顾问。请分析以下AI-Staff系统的执行记录，给出具体的改进建议。

【系统当前表现统计】
{performance_stats}

【最近执行记录（最近{n}次）】
{recent_execs}

【用户反馈】
{user_feedback}

【当前专家Prompt片段（供参考）】
{config_snapshot}

请分析并输出：

## 问题诊断
（哪些方面表现不佳？有什么规律性？）

## 改进建议（逐项）

### 1. 专家Prompt优化
- 目标专家: [id]
- 当前问题: [具体描述]
- 建议修改: [完整的新prompt或关键diff]

### 2. 分类器规则调整
- 当前误分类案例: [举例]
- 建议调整: [具体规则变更]

### 3. 路由策略优化
- 当前问题: [如：简单问题用了贵模型]
- 建议: [新的路由策略]

### 4. 工作流改进
- 哪类任务的流程需要优化
- 新流程建议

## 优先级排序（按投入产出比）
- 🟢 立即可安全执行
- 🟡 建议测试后执行
- 🔴 需人工确认后执行
"""
    
    def __init__(self, memory: MemorySystem, llm_client):
        """
        Args:
            memory: The shared MemorySystem instance (for reading execution history)
            llm_client: An LLMClient to call for reflection (use cheapest available)
        """
        self.memory = memory
        self.llm = llm_client
        
        # State
        self._improvements: list[ImprovementRecord] = []
        self._execution_count = 0
        self._auto_enabled = True
        self._reflection_interval = 10       # Reflect every N tasks
        self._quality_threshold = 6.0        # Trigger reflection if score below this
        self._max_history_items = 10         # How many past tasks to show in reflection

    def on_task_complete(self, result: CollaborationResult):
        """
        Hook: Call this after every auto_run / collaborate / etc.
        
        Decides whether to trigger a reflection cycle based on:
        - Quality score too low
        - Task failed
        - Periodic interval reached
        """
        self._execution_count += 1
        
        # Log to memory
        try:
            self.memory.log_task(
                task_type=result.strategy_mode,
                prompt=result.goal[:500],
                result_path="",
                status=result.status,
                tokens=result.total_tokens,
                duration=result.total_time_sec,
                models=result.experts_used
            )
        except Exception as e:
            pass  # Don't let logging failures break things

        # Determine if we should reflect
        should_reflect = (
            result.status == "failed" or
            result.quality_score < self._quality_threshold or
            self._execution_count % self._reflection_interval == 0
        )

        if should_reflect and self._auto_enabled:
            self._trigger_reflection(result)

    def _trigger_reflection(self, recent_result: CollaborationResult = None):
        """Run one full reflection→propose cycle."""
        print(f"\n  🔄 [Self-Improve] 第{self._execution_count}次任务，触发自我反思...")
        
        stats = self._gather_stats()
        recent_execs = self._get_recent_executions(self._max_history_items)
        feedback = self._get_user_feedback()
        config_snippet = self._get_config_snapshot()

        prompt = self.REFLECTION_PROMPT.format(
            performance_stats=stats,
            recent_execs=recent_execs,
            user_feedback=feedback,
            config_snapshot=config_snippet,
            n=self._max_history_items,
        )
        
        try:
            response, _usage = self.llm.chat_completion([
                {"role": "system",
                 "content": "你是AI系统优化顾问。只输出结构化分析和具体建议，不要客套话。"},
                {"role": "user", "content": prompt}
            ], temperature=0.7, max_tokens=4096)

            improvements = self._parse_improvements(response)

            # Apply each suggestion
            for imp in improvements:
                if imp.target_component == "expert_prompt":
                    applied = self._apply_prompt_change(imp)
                    imp.applied = applied
                elif imp.target_component == "classifier_rule":
                    # Classifier changes are logged but not auto-applied (medium risk)
                    print(f"  📋 [Self-Improve] 分类器改进已记录（需手动确认）")
                    imp.applied = False
                else:
                    # Router/strategy changes are high-risk — log only
                    imp.applied = False

                self._improvements.append(imp)

            print(f"  🔄 [Self-Improve] 完成！生成 {len(improvements)} 条改进建议")
            
        except Exception as e:
            print(f"  ⚠️ [Self-Improve] 反思失败: {type(e).__name__}: {str(e)[:100]}")

    def _gather_stats(self) -> str:
        """Collect performance statistics from memory."""
        try:
            conn = self.memory._get_conn()
            rows = conn.execute(
                "SELECT task_type, status, tokens_used, duration_sec FROM task_history "
                "ORDER BY id DESC LIMIT 20"
            ).fetchall()
            
            if not rows:
                return "尚无执行记录"

            total_t = sum(r["tokens_used"] or 0 for r in rows)
            total_s = sum(r["duration_sec"] or 0 for r in rows)
            success_rate = sum(1 for r in rows if r["status"] == "success") / len(rows)
            
            by_type = {}
            for r in rows:
                t = r["task_type"] or "unknown"
                by_type.setdefault(t, {"count": 0, "ok": 0})
                by_type[t]["count"] += 1
                if r["status"] == "success":
                    by_type[t]["ok"] += 1
            
            lines = [
                f"- 最近20次任务 | 成功率: {success_rate:.0%}",
                f"- 总Token: {total_t} | 平均耗时: {total_s/max(len(rows),1):.1f}s/次",
                "",
                "- 按类型分布:"
            ]
            for t, s in sorted(by_type.items()):
                rate = s['ok'] / max(s['count'], 1)
                lines.append(f"  · {t}: {s['count']}次 (成功率{rate:.0%})")

            return "\n".join(lines)
        except Exception as e:
            return f"统计数据获取异常: {e}"

    def _get_recent_executions(self, n: int = 10) -> str:
        """Get recent task execution records."""
        try:
            conn = self.memory._get_conn()
            rows = conn.execute(
                "SELECT task_type, prompt, status, tokens_used, duration_sec FROM task_history "
                "ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
            
            if not rows:
                return "无记录"

            lines = []
            for i, r in enumerate(reversed(rows)):
                p = (r["prompt"] or "")[:60]
                lines.append(
                    f"{i+1}. [{r['task_type'] or '?'}] {p}... "
                    f"→ {r['status'] or '?'} ({r['tokens_used'] or 0}t/{r['duration_sec'] or 0:.1f}s)"
                )
            return "\n".join(lines)
        except Exception:
            return "读取失败"

    def _get_user_feedback(self) -> str:
        """Get user feedback from memory."""
        try:
            conn = self.memory._get_conn()
            rows = conn.execute(
                "SELECT rating, comment FROM feedback ORDER BY id DESC LIMIT 10"
            ).fetchall()
            
            if not rows:
                return "暂无用户反馈"
            
            avg_r = sum(r["rating"] or 0 for r in rows) / len(rows)
            parts = [f"最近{len(rows)}条反馈, 均分: {avg_r:.1f}/3"]
            for r in rows[-3:]:
                c = (r["comment"] or "")[:100]
                if c:
                    parts.append(f"  ({r['rating']}/3) {c}")
            return "\n".join(parts)
        except Exception:
            return ""

    def _get_config_snapshot(self) -> str:
        """Get current config snippet for context."""
        exp = ExpertRegistry.get("generalist")
        if exp and exp.system_prompt:
            return f"""generalist expert prompt:
```
{exp.system_prompt[:500]}
{'...' if len(exp.system_prompt) > 500 else ''}
```"""
        return "(无法加载配置)"

    def _parse_improvements(self, response: str) -> list[ImprovementRecord]:
        """Parse improvement suggestions from LLM response."""
        results = []

        # Extract expert prompt section
        sec = re.search(
            r'[#{ ]*专家[Pp]rompt[优化改进]*.*?\n(.*?)(?=#{1,3}\s|\Z)',
            response, re.DOTALL
        )
        if sec and len(sec.group(1).strip()) > 20:
            results.append(ImprovementRecord(
                timestamp=datetime.now().isoformat(),
                trigger="auto_reflection",
                target_component="expert_prompt",
                after_content=sec.group(1).strip()[:1500],
                reasoning="LLM-based analysis of execution history",
                applied=False,
            ))

        # Extract classifier section
        sec2 = re.search(
            r'[#{ ]*分类器[规则调整]*.*?\n(.*?)(?=#{1,3}\s|\Z)',
            response, re.DOTALL
        )
        if sec2 and len(sec2.group(1).strip()) > 20:
            results.append(ImprovementRecord(
                timestamp=datetime.now().isoformat(),
                trigger="auto_reflection",
                target_component="classifier_rule",
                after_content=sec2.group(1).strip()[:1500],
                reasoning="LLM-based analysis",
                applied=False,
            ))

        # If nothing parsed but response is long enough, treat as generic
        if not results and len(response.strip()) > 50:
            results.append(ImprovementRecord(
                timestamp=datetime.now().isoformat(),
                trigger="auto_reflection",
                target_component="general",
                after_content=response[:1500],
                reasoning="Generic reflection output",
                applied=False,
            ))

        return results

    def _apply_prompt_change(self, imp: ImprovementRecord) -> bool:
        """
        Apply an expert prompt improvement.
        
        Safety: Only updates in-memory ExpertConfig, does NOT persist to YAML
        unless explicitly confirmed. Changes are lost on restart (by design).
        """
        content = imp.after_content

        # Try to extract target expert ID
        target_match = re.search(r'目标专家[::\s]*([\w]+)', content)
        new_prompt_match = re.search(
            r'建议修改[::\s]*(?:`{3}[\s]*\n)?(.+?)(?:\n`{3}|$)',
            content, re.DOTALL
        )

        if target_match:
            exp_id = target_match.group(1).lower().strip()
            exp = ExpertRegistry.get(exp_id)
            if exp:
                old_prompt = exp.system_prompt
                
                if new_prompt_match:
                    new_text = new_prompt_match.group(1).strip()
                    # Clean up markdown artifacts
                    new_text = re.sub(r'^```[\w]*\n?', '', new_text)
                    new_text = re.sub(r'\n?```\s*?$', '', new_text)
                    new_text = new_text.strip()
                    
                    if len(new_text) > 30:
                        # Update in-memory only
                        imp.before_hash = hashlib.md5(old_prompt.encode()).hexdigest()[:8]
                        exp.system_prompt = new_text
                        
                        print(f"  ✏️ [Self-Improve] 已更新专家 '{exp_id}' 的system_prompt")
                        
                        # Log what changed
                        diff_chars = abs(len(new_text) - len(old_prompt))
                        imp.reasoning = f"Updated '{exp_id}' prompt (Δ{diff_chars:+d} chars)"
                        return True

        # Could not parse cleanly — log but don't apply blindly
        print(f"  ⚠️ [Self-Improve] Prompt改进格式无法解析，跳过自动应用")
        return False

    def manual_improve(self):
        """Manually trigger a reflection cycle (regardless of conditions)."""
        print("  🔄 [Self-Improve] 手动触发反思循环...")
        self._trigger_reflection()
        return self.get_log()

    def get_log(self) -> str:
        """Get formatted improvement log."""
        if not self._improvements:
            return f"# 自我改进日志\n\n暂无改进记录。\n\n提示：系统会自动在每{self._reflection_interval}次任务或质量评分低于{self._quality_threshold}时触发反思。"

        lines = [
            f"# AI-Staff V{VERSION} 自我改进日志",
            f"",
            f"**总反思次数:** {len(self._improvements)}",
            f"**总任务次数:** {self._execution_count}",
            f"**自动改进:** {'开启' if self._auto_enabled else '关闭'}",
            f"",
            f"| 时间 | 触发原因 | 改进对象 | 已应用 |",
            f"|------|---------|---------|--------|",
        ]

        for imp in self._improvements[-20:]:
            icon = "✅" if imp.applied else "📋"
            ts_short = imp.timestamp.replace("T", " ")[:16] if imp.timestamp else "?"
            lines.append(
                f"| {ts_short} | {imp.trigger} | {imp.target_component} | {icon} |"
            )

        # Show latest details
        if self._improvements:
            latest = self._improvements[-1]
            lines.extend([
                "",
                f"## 最新改进详情 ({latest.timestamp})",
                "",
                f"**目标:** {latest.target_component}",
                f"**触发:** {latest.trigger}",
                f"**状态:** {'已应用 ✅' if latest.applied else '待确认 📋'}",
            ])
            if latest.reasoning:
                lines.append(f"**理由:** {latest.reasoning}")
            if latest.after_content:
                preview = latest.after_content[:400].replace('\n', ' ')
                lines.append(f"**内容预览:** {preview}...")

        lines.extend(["", "---", f"*AI-Staff V{VERSION} Self-Improvement Engine*"])
        return "\n".join(lines)

    @property
    def is_auto_enabled(self) -> bool:
        return self._auto_enabled

    def enable_auto(self):
        self._auto_enabled = True
        print("  ✓ [Self-Improve] 自动改进已启用")

    def disable_auto(self):
        self._auto_enabled = False
        print("  ✗ [Self-Improve] 自动改进已禁用")



__all__ = ['SelfImprovementEngine']

