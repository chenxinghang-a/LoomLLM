from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaskStrategy:
    """Optimal execution strategy for a detected task type."""
    mode: str                    # direct | code | research | collaborate | analysis
    display_name: str            # Human-readable name
    experts: list[str]           # Expert IDs to use
    primary_expert: str          # Main expert for single-expert modes
    needs_review: bool           # Whether critic should check output
    output_format: str           # text | code | markdown_report | folder
    max_rounds: int              # Max conversation rounds
    auto_followups: list[str]     # Auto-generated follow-up questions
    description: str             # Why this strategy was chosen


class TaskClassifier:
    """
    V3 CORE: Automatically classify any user input into an optimal execution strategy.
    
    Key principle (from user feedback):
    - NOT everything needs a roundtable meeting!
    - Simple questions → direct answer (1 call, 1 expert)
    - Code tasks → coder + critic (2 experts, focused)
    - Research → deep iterative dive (1 expert, multiple rounds)
    - Complex decisions → multi-expert collaboration (only when needed!)
    
    This is THE differentiator: users just say what they want,
    the framework figures out HOW to do it optimally.
    """

    # Task type definitions with keyword patterns
    TASK_DEFINITIONS: dict[str, dict] = {
        "direct": {
            "name": "快速问答",
            "keywords": ["什么是", "怎么", "如何", "解释", "定义", "意思", "1+1",
                        "hello", "hi", "你好", "谢谢", "翻译", "convert",
                        "多少", "几", "谁", "哪里", "什么时候", "list"],
            "anti_keywords": ["分析", "设计", "实现", "开发", "研究", "对比", 
                            "评估", "方案", "架构", "系统"],
            "max_length": 80,  # Short queries are usually direct Q&A
            "experts": ["generalist"],
            "primary": "generalist",
            "needs_review": False,
            "output_format": "text",
            "max_rounds": 1,
            "followups": [],
            "desc": "单专家快速回答，适合简单问答和事实查询"
        },
        "code": {
            "name": "代码任务",
            "keywords": ["写代码", "实现", "function", "debug", "bug", "程序",
                        "算法", "script", "api", "接口", "爬虫", "函数",
                        "代码", "编程", "python", "javascript", "java", "c++",
                        "class ", "def ", "import ", "实现一个", "帮我写"],
            "experts": ["coder", "critic"],
            "primary": "coder",
            "needs_review": True,
            "output_format": "code",
            "max_rounds": 2,
            "followups": [
                "请检查以上代码的边界情况和潜在错误。",
                "能否优化性能或简化逻辑？给出改进版本。"
            ],
            "desc": "编码+审查双专家流程，确保代码质量"
        },
        "research": {
            "name": "深度研究",
            "keywords": ["研究", "分析", "调研", "综述", "趋势", "原理",
                        "为什么", "对比.*优缺", "发展史", "现状", "前景",
                        "deep.?dive", "深入", "详细分析", "全面", "报告"],
            "experts": ["researcher"],
            "primary": "researcher",
            "needs_review": False,
            "output_format": "markdown_report",
            "max_rounds": 4,
            "followups": [
                "基于以上内容，进一步深挖最关键的技术细节或争议点。",
                "有哪些容易被忽视的重要方面或常见认知误区？",
                "从实践者角度，给出具体行动指南：入门路径、避坑建议、工具推荐。"
            ],
            "desc": "研究员多轮迭代追问，适合复杂分析和研究报告"
        },
        "decision": {
            "name": "决策辅助",
            "keywords": ["应该", "选择", "建议", "哪个好", "买哪个",
                        "推荐", "优缺点", "比较", "对比", "选型",
                        "是否值得", "a还是b", "or", "vs", "取舍",
                        "更好用", "哪个更"],
            "experts": ["planner", "researcher", "critic"],
            "primary": "planner",
            "needs_review": True,
            "output_format": "markdown_report",
            "max_rounds": 2,
            "followups": [],
            "desc": "多维度分析+权衡建议，帮助做决策"
        },
        "creative": {
            "name": "创意任务",
            "keywords": ["创意", "设计", "文案", "故事", "脑暴", "想法",
                        "命名", "slogan", "标题", "海报", "logo",
                        "营销", "推广", "文案写作", "广告", "策划",
                        "发布会", "活动方案", "宣传", "品牌"],
            "experts": ["writer", "critic"],
            "primary": "writer",
            "needs_review": True,
            "output_format": "text",
            "max_rounds": 2,
            "followups": [
                "这个方案还有什么可以改进或更有吸引力的地方？",
                "给我3个不同风格的替代版本。"
            ],
            "desc": "创作+审美审查双重保障"
        },
        "collaborate": {
            "name": "圆桌协作",
            # This is the fallback for complex/multi-domain tasks
            # or explicitly requested via keywords
            "keywords": ["圆桌", "讨论", "辩论", "多方", "会议",
                        "综合意见", "团队", "协作", "各抒己见"],
            "experts": ["planner", "researcher", "coder", "critic"],
            "primary": "planner",
            "needs_review": True,
            "output_format": "folder",
            "max_rounds": 2,
            "followups": [],
            "desc": "多专家目标驱动协作，产出完整工作成果（仅复杂任务触发）"
        },
    }

    def __init__(self):
        self._definitions = self.TASK_DEFINITIONS
        self._sync_experts()

    def _sync_experts(self):
        """从ExpertRegistry同步可用专家，覆盖硬编码的experts列表"""
        try:
            from .registry import ExpertRegistry
            if not ExpertRegistry._experts:
                ExpertRegistry.load_all()
            available = {e.id for e in ExpertRegistry.list_all()}
        except Exception:
            return  # registry不可用时用默认值

        for task_type, cfg in self._definitions.items():
            # 过滤掉不存在的专家，fallback到generalist
            valid = [e for e in cfg["experts"] if e in available]
            if not valid:
                valid = ["generalist"] if "generalist" in available else []
            cfg["experts"] = valid
            cfg["primary"] = valid[0] if valid else "generalist"

    def classify(self, user_input: str) -> TaskStrategy:
        """
        Classify user input into optimal execution strategy.
        
        Uses: keyword matching + length heuristics + anti-keyword filtering.
        
        Returns a TaskStrategy that tells ai-staff EXACTLY what to do:
        - Which experts to use
        - How many rounds
        - What output format
        - Whether to review
        
        The caller (auto_run) just executes this strategy blindly.
        """
        text_lower = user_input.lower().strip()
        text_len = len(user_input)

        scores: dict[str, float] = {}

        for task_type, config in self._definitions.items():
            score = 0.0

            # Keyword matching (positive signals)
            kw_score = sum(2.0 for kw in config["keywords"] if re.search(kw, text_lower, re.IGNORECASE))
            score += kw_score

            # Anti-keyword penalty (this is NOT this type of task)
            anti_kws = config.get("anti_keywords", [])
            anti_penalty = sum(3.0 for akw in anti_kws if akw in text_lower)
            score -= anti_penalty

            # Length heuristic: short queries favor "direct"
            if "max_length" in config and text_len <= config["max_length"]:
                score += 1.5
            elif task_type == "direct" and text_len > 100:
                score -= 1.0  # Long queries are probably not simple Q&A

            # Multi-sentence / complex structure suggests research or decision
            sentence_count = text_lower.count('.') + text_lower.count('?') + text_lower.count('\n')
            if task_type in ("research", "collaborate") and sentence_count >= 2:
                score += 1.5

            # Question mark suggests direct Q&A or decision
            if '?' in user_input or '？' in user_input:
                if task_type == "direct":
                    score += 1.0
                elif task_type == "decision":
                    score += 0.8

            scores[task_type] = score

        # Find best match
        best_type = max(scores.keys(), key=lambda k: scores[k]) if scores else "direct"
        best_score = scores[best_type]

        # Fallback: if no clear signal, use direct for short, research for long
        if best_score <= 0:
            best_type = "direct" if text_len < 50 else "research"

        cfg = self._definitions[best_type]

        return TaskStrategy(
            mode=best_type,
            display_name=cfg["name"],
            experts=cfg["experts"],
            primary_expert=cfg["primary"],
            needs_review=cfg["needs_review"],
            output_format=cfg["output_format"],
            max_rounds=cfg["max_rounds"],
            auto_followups=cfg["followups"],
            description=cfg["desc"]
        )

    def explain(self, user_input: str, strategy: TaskStrategy = None) -> str:
        """Explain why this strategy was chosen (for transparency)."""
        if not strategy:
            strategy = self.classify(user_input)
        return (
            f"📋 任务分类: [{strategy.display_name}] ({strategy.mode})\n"
            f"   策略: {strategy.description}\n"
            f"   专家: {', '.join(strategy.experts)}\n"
            f"   轮次: {strategy.max_rounds} | 审查: {'是' if strategy.needs_review else '否'}\n"
            f"   输出: {strategy.output_format}"
        )


__all__ = ['TaskStrategy', 'TaskClassifier']
