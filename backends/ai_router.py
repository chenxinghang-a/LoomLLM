"""
AI Router — 让AI自己决定用什么模型

核心思想：
  不是关键词硬编码选模型，而是让最强可用模型分析任务特征，
  自动决定用哪个模型执行。

流程：
  1. 收到用户输入
  2. 如果是简单任务（1+1、翻译等）→ 直接用最快免费模型
  3. 如果是复杂任务 → 用最强模型做一次极短"路由决策"调用
     → 返回 JSON: {"model": "xxx", "reason": "xxx"}
  4. 用选定的模型执行实际任务
  5. 决策结果缓存（相似问题不重复问）

这样可以做到：
  - 简单问题不浪费算力
  - 复杂问题精准匹配最强模型
  - 每个provider的模型都有机会被选中
  - AI自己学习什么任务该用什么模型
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .smart_init import ModelRegistry, ModelInfo
from .client import LLMClient


# ── 路由决策结果 ──

@dataclass
class RouteDecision:
    """AI路由决策结果"""
    model_name: str          # 选定的模型
    profile_key: str         # 对应的profile key
    reason: str              # 为什么选这个
    task_type: str           # 任务分类
    complexity: int          # 复杂度 0-10
    needs_review: bool       # 是否需要审查
    confidence: float        # 决策置信度 0-1


# ── 快速规则（不需要AI决策的简单情况） ──

FAST_RULES = {
    # pattern → (task_type, max_complexity)
    r'^(1\+1|[0-9]+\s*[\+\-\*\/]\s*[0-9]+)$': ("arithmetic", 0),
    r'^(hi|hello|hey|你好|嗨|谢谢)$': ("greeting", 0),
    r'^(ok|好的|嗯|yes|no)$': ("acknowledgment", 0),
}


class AIRouter:
    """
    AI决策路由 — 用AI选AI
    
    设计原则：
    1. 简单任务不过AI决策（省token）
    2. 复杂任务用最强模型做路由决策（精准）
    3. 决策结果缓存（相似问题不重复问）
    4. 降级策略（AI决策失败时回退到规则路由）
    """

    def __init__(self, registry: ModelRegistry, llm_clients: dict[str, LLMClient] = None):
        self.registry = registry
        self._clients = llm_clients or {}
        self._decision_cache: dict[str, RouteDecision] = {}

    def register_client(self, profile_key: str, client: LLMClient):
        self._clients[profile_key] = client

    def route(self, user_input: str) -> RouteDecision:
        """
        主入口：分析任务 → 决定用什么模型
        
        三层决策：
          1. 快速规则（简单任务直接分配最快模型）
          2. AI决策（复杂任务让最强模型分析）
          3. 规则降级（AI不可用时回退到关键词路由）
        """
        # Layer 1: 快速规则
        fast = self._fast_route(user_input)
        if fast:
            return fast

        # Layer 2: AI决策
        ai = self._ai_route(user_input)
        if ai:
            return ai

        # Layer 3: 规则降级
        return self._fallback_route(user_input)

    # ── Layer 1: 快速规则 ──

    def _fast_route(self, user_input: str) -> Optional[RouteDecision]:
        """极简任务直接用最快模型，不走AI决策"""
        text = user_input.strip().lower()
        
        for pattern, (task_type, complexity) in FAST_RULES.items():
            if re.match(pattern, text):
                # 用最快的免费模型
                free_models = self.registry.free_models
                if free_models:
                    fastest = min(free_models, key=lambda m: m.latency_ms)
                    key = fastest.name.replace('-', '_').replace('.', '')
                    if key not in self._clients:
                        key = f"{fastest.provider}_{key}"
                    return RouteDecision(
                        model_name=fastest.name, profile_key=key,
                        reason=f"Fast rule: {task_type}", task_type=task_type,
                        complexity=complexity, needs_review=False, confidence=1.0,
                    )

        # 短问题（<15字符且不含复杂词）也走快速
        if len(user_input) < 15 and not any(w in text for w in 
            ["分析", "设计", "研究", "开发", "架构", "对比", "实现"]):
            free_models = self.registry.free_models
            if free_models:
                fastest = min(free_models, key=lambda m: m.latency_ms)
                key = fastest.name.replace('-', '_').replace('.', '')
                if key not in self._clients:
                    key = f"{fastest.provider}_{key}"
                return RouteDecision(
                    model_name=fastest.name, profile_key=key,
                    reason="Short simple query", task_type="simple",
                    complexity=1, needs_review=False, confidence=0.9,
                )

        return None

    # ── Layer 2: AI决策 ──

    def _ai_route(self, user_input: str) -> Optional[RouteDecision]:
        """让最强可用模型分析任务，决定用什么模型执行"""
        
        # 找到决策模型（最强的可用模型）
        decider = self.registry.get_strongest()
        if not decider:
            return None

        decider_key = decider.name.replace('-', '_').replace('.', '')
        # 也尝试带provider前缀
        client = self._clients.get(decider_key)
        if not client:
            decider_key = f"{decider.provider}_{decider_key}"
            client = self._clients.get(decider_key)
        if not client:
            return None

        # 构建可用模型菜单
        usable = self.registry.usable_models
        if not usable:
            return None

        model_menu = []
        for m in usable[:10]:  # 最多列10个，避免prompt太长
            cost_label = "free" if m.is_free else f"${m.input_cost}/1K"
            local_label = " [LOCAL]" if m.is_local else ""
            caps_label = ",".join(m.capabilities[:3]) if m.capabilities else "general"
            model_menu.append(
                f'  "{m.name}": tier={m.tier}, cost={cost_label}, '
                f'caps=[{caps_label}], latency={m.latency_ms:.0f}ms{local_label}'
            )

        prompt = f"""You are a model router. Analyze this task and pick the best model.

TASK: {user_input[:500]}

AVAILABLE MODELS:
{chr(10).join(model_menu)}

Respond with ONLY this JSON (no markdown, no explanation):
{{"model": "model-name", "reason": "brief reason", "task_type": "code|research|creative|decision|simple", "complexity": 0-10, "needs_review": true/false}}

Rules:
- Simple/greeting → cheapest free model
- Code/reasoning → model with "code" or "reasoning" capability
- Creative → model with "creative" capability
- Research/analysis → strongest model available
- Always prefer free over paid unless task complexity > 7
- Keep reason under 20 words"""

        try:
            content, _ = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # 低温度，决策要确定性
                max_tokens=200,
                model=decider.name,
            )

            # 解析JSON
            content = content.strip()
            # 去掉可能的markdown包裹
            if content.startswith("```"):
                content = re.sub(r'^```\w*\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
                content = content.strip()

            decision = json.loads(content)

            model_name = decision.get("model", "")
            # 验证模型是否在可用列表中
            target = self.registry.get_model(model_name)
            if not target:
                # 找最近的匹配
                for m in usable:
                    if model_name in m.name or m.name in model_name:
                        target = m
                        break

            if target:
                target_key = target.name.replace('-', '_').replace('.', '')
                if target_key not in self._clients:
                    target_key = f"{target.provider}_{target_key}"
                return RouteDecision(
                    model_name=target.name,
                    profile_key=target_key,
                    reason=decision.get("reason", "AI routed")[:50],
                    task_type=decision.get("task_type", "general"),
                    complexity=min(10, max(0, decision.get("complexity", 5))),
                    needs_review=decision.get("needs_review", False),
                    confidence=0.85,
                )

        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"  [AIRouter] AI decision failed: {type(e).__name__}: {str(e)[:60]}")

        return None

    # ── Layer 3: 规则降级 ──

    def _fallback_route(self, user_input: str) -> RouteDecision:
        """AI决策失败时的关键词规则路由"""
        text = user_input.lower()
        
        # 关键词 → 任务类型 → 推荐tier
        task_tier_map = [
            (["写代码", "debug", "实现", "function", "代码", "编程", "算法", "def ", "class "],
             "code", "cheap"),
            (["研究", "分析", "调研", "综述", "deep", "深入", "全面"],
             "research", "premium"),
            (["创意", "设计", "文案", "故事", "营销"],
             "creative", "standard"),
            (["应该", "选择", "推荐", "对比", "选型", "vs"],
             "decision", "standard"),
        ]

        task_type = "general"
        target_tier = "free"

        for keywords, ttype, tier in task_tier_map:
            if any(kw in text for kw in keywords):
                task_type = ttype
                target_tier = tier
                break

        # 根据目标tier选模型
        usable = self.registry.usable_models
        tier_order = {"free": ["free", "local", "cheap", "standard"],
                      "cheap": ["cheap", "free", "standard"],
                      "standard": ["standard", "premium", "cheap"],
                      "premium": ["premium", "standard", "cheap"]}

        preferred_tiers = tier_order.get(target_tier, ["free", "cheap", "standard"])

        for tier in preferred_tiers:
            candidates = [m for m in usable if m.tier == tier]
            if candidates:
                best = max(candidates, key=lambda m: m.strength_score)
                key = best.name.replace('-', '_').replace('.', '')
                if key not in self._clients:
                    key = f"{best.provider}_{key}"
                return RouteDecision(
                    model_name=best.name, profile_key=key,
                    reason=f"Rule fallback: {task_type}→{tier}",
                    task_type=task_type, complexity=5,
                    needs_review=task_type in ("code", "decision"),
                    confidence=0.6,
                )

        # 终极降级：用全局最优
        best_name = self.registry.best_overall
        target = self.registry.get_model(best_name)
        if target:
            key = target.name.replace('-', '_').replace('.', '')
            if key not in self._clients:
                key = f"{target.provider}_{key}"
            return RouteDecision(
                model_name=best_name, profile_key=key,
                reason="Fallback: best overall", task_type=task_type,
                complexity=5, needs_review=False, confidence=0.4,
            )

        # 完全没模型
        return RouteDecision(
            model_name="", profile_key="",
            reason="No models available!", task_type="error",
            complexity=0, needs_review=False, confidence=0,
        )


__all__ = ['AIRouter', 'RouteDecision']
