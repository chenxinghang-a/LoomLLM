"""Example 3: Expert Task — 代码生成+审查闭环，确保代码质量

Usage:
    set GEMINI_API_KEY=your-key
    python examples/expert_task.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_staff_v4 import AIStaff

staff = AIStaff.from_env()

# Code mode: Coder writes → Critic reviews → Coder revises
result = staff.chat(
    "用Python写一个线程安全的单例模式，要求：1) 支持带参数初始化 2) 线程安全 3) 有类型提示",
    mode="code",
    return_details=True
)

print(f"\n{'='*50}")
print(f"Status: {result.status}")
print(f"Quality Score: {result.quality_score}/10")
print(f"Experts Used: {', '.join(result.experts_used)}")

if result.deliverables:
    for name, content in result.deliverables.items():
        print(f"\n📄 {name}:")
        print(content[:800] + ("..." if len(content) > 800 else ""))
