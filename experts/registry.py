from __future__ import annotations
import json, os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional

# Intra-package imports
from ..core.constants import PACKAGE_ROOT

# Expert configs directory
EXPERTS_DIR = PACKAGE_ROOT / "experts"

@dataclass
class ExpertConfig:
    """An expert role definition."""
    id: str
    name: str
    description: str
    system_prompt: str
    style_hints: str = ""          # Tone, format, language preferences
    domain_tags: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)     # Tool names this expert can use
    max_turns: int = 5             # Max conversation turns
    temperature: float = 0.7
    model_override: str = ""       # Use specific model for this expert
    api_profile: str = ""          # Use specific API backend (multi-mode only)
    
    # Quality gates
    require_review: bool = True
    output_format: str = "text"    # text | json | markdown | code
    validation_rules: list[str] = field(default_factory=list)

class ExpertRegistry:
    """Load and manage expert configurations from YAML files."""
    
    _experts: dict[str, ExpertConfig] = {}
    
    @classmethod
    def load_all(cls) -> int:
        """Load all .yaml files from experts directory."""
        try:
            import yaml
        except ImportError:
            print("[ExpertRegistry] PyYAML not installed, using built-in experts only")
            cls._load_builtin()
            return len(cls._experts)
        
        count = 0
        for f in EXPERTS_DIR.glob("*.yaml"):
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = yaml.safe_load(fh)
                if isinstance(data, list):
                    for item in data:
                        exp = cls._parse(item)
                        cls._experts[exp.id] = exp
                        count += 1
                elif isinstance(data, dict):
                    exp = cls._parse(data)
                    cls._experts[exp.id] = exp
                    count += 1
            except Exception as e:
                print(f"  [ExpertRegistry] Failed to load {f.name}: {e}")
        
        if count == 0:
            cls._load_builtin()
            count = len(cls._experts)
        
        return count
    
    @classmethod
    def _parse(cls, data: dict) -> ExpertConfig:
        return ExpertConfig(
            id=data.get('id', 'unknown'),
            name=data.get('name', 'Unnamed'),
            description=data.get('description', ''),
            system_prompt=data.get('system_prompt', 'You are a helpful assistant.'),
            style_hints=data.get('style_hints', ''),
            domain_tags=data.get('domain_tags', []),
            tools=data.get('tools', []),
            max_turns=data.get('max_turns', 5),
            temperature=data.get('temperature', 0.7),
            model_override=data.get('model_override', ''),
        api_profile=data.get('api_profile', ''),
            require_review=data.get('require_review', True),
            output_format=data.get('output_format', 'text'),
            validation_rules=data.get('validation_rules', [])
        )
    
    @classmethod
    def _load_builtin(cls):
        """Built-in default experts when no YAML files found."""
        builtin_experts = [
            ExpertConfig(
                id="generalist", name="通用助手",
                description="全能型助手，适合大多数任务",
                system_prompt="你是一个专业、高效、友好的AI助手。请用中文回答，结构清晰，重点突出。",
                domain_tags=["general"], max_turns=10, require_review=False
            ),
            ExpertConfig(
                id="researcher", name="深度研究员",
                description="擅长信息搜集、分析和综合报告",
                system_prompt="你是一位资深研究员。你的任务是深入分析问题，提供详实的数据支撑和逻辑严密的结论。输出格式：先给摘要(3行)，再展开详细分析。使用Markdown格式。",
                style_hints="严谨、数据驱动、多角度分析",
                domain_tags=["research", "analysis", "report"],
                output_format="markdown",
                validation_rules=["contains_summary", "has_data_support"]
            ),
            ExpertConfig(
                id="coder", name="高级工程师",
                description="擅长代码编写、调试和架构设计",
                system_prompt="你是一位高级软件工程师。代码要求：1) 注释清晰 2) 错误处理完善 3) 遵循最佳实践 4) 提供使用示例。优先Python，其他语言请注明版本。",
                style_hints="代码注释详细、给出示例、考虑边界情况",
                domain_tags=["code", "programming", "debugging", "architecture"],
                output_format="code",
                tools=["code_executor", "linter"]
            ),
            ExpertConfig(
                id="writer", name="内容创作者",
                description="擅长各类文案写作、创意内容",
                system_prompt="你是一位专业内容创作者。根据需求创作高质量内容：标题吸引人、结构清晰、语言流畅有感染力。注意：不要空洞套话，要有实质内容和独特观点。",
                style_hints="生动、有观点、避免陈词滥调",
                domain_tags=["writing", "content", "creative", "copywriting"],
                output_format="markdown"
            ),
            ExpertConfig(
                id="critic", name="质量审查员",
                description="审查输出质量，提出改进建议",
                system_prompt="你是一位严格的质量审查员。审查输入内容并给出：1) 质量评分(1-10) 2) 主要问题清单 3) 具体改进建议 4) 改进后的版本(如需)。格式：先用表格总结，再逐项展开。",
                style_hints="严格但建设性、具体不模糊",
                domain_tags=["review", "quality", "critique"],
                temperature=0.3,  # Lower temp for consistent reviews
                require_review=False
            ),
            ExpertConfig(
                id="planner", name="规划师",
                description="分解复杂任务为可执行的步骤计划",
                system_prompt="你是一位资深项目规划师。将复杂任务分解为清晰、有序的执行步骤。格式：## 目标概述\n## 执行步骤(编号列表)\n## 所需资源\n## 风险与备选方案\n每个步骤要具体可执行，不要笼统描述。",
                style_hints="结构化、步骤明确、考虑依赖关系",
                domain_tags=["planning", "breakdown", "architecture"],
                require_review=True,
                output_format="markdown"
            )
        ]
        for exp in builtin_experts:
            cls._experts[exp.id] = exp
    
    @classmethod
    def get(cls, expert_id: str) -> Optional[ExpertConfig]:
        if not cls._experts:
            cls.load_all()
        return cls._experts.get(expert_id)
    
    @classmethod
    def list_all(cls) -> list[ExpertConfig]:
        return list(cls._experts.values())
    
    @classmethod
    def search(cls, query: str) -> list[ExpertConfig]:
        """Search experts by keyword matching."""
        q = query.lower()
        results = []
        for exp in cls._experts.values():
            if (q in exp.name.lower() or q in exp.description.lower() or
                any(q in tag for tag in exp.domain_tags)):
                results.append(exp)
        return results
    
    @classmethod
    def create_expert_file(cls, expert: ExpertConfig, filename: str = None):
        """Save an expert config to YAML file."""
        try:
            import yaml
        except ImportError:
            print("[ExpertRegistry] Need pyyaml to save expert files")
            return
        
        fname = filename or f"{expert.id}.yaml"
        filepath = EXPERTS_DIR / fname
        data = {
            'id': expert.id, 'name': expert.name,
            'description': expert.description,
            'system_prompt': expert.system_prompt,
            'style_hints': expert.style_hints,
            'domain_tags': expert.domain_tags,
            'tools': expert.tools, 'max_turns': expert.max_turns,
            'temperature': expert.temperature,
            'model_override': expert.model_override,
            'require_review': expert.require_review,
            'output_format': expert.output_format,
            'validation_rules': expert.validation_rules
        }
        with open(filepath, 'w', encoding='utf-8') as fh:
            yaml.dump(data, fh, allow_unicode=True, default_flow_style=False)
            print(f"  [ExpertRegistry] Saved expert '{expert.id}' to {filepath}")


__all__ = ['ExpertConfig', 'ExpertRegistry', 'EXPERTS_DIR']

