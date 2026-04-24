"""Example 2: Research — V5闭环协作，Writer写+Reviewer查+低分重写

Usage:
    set GEMINI_API_KEY=your-key
    python examples/research_flow.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_staff_v4 import AIStaff

staff = AIStaff.from_env()

# Research mode: Writer drafts → Reviewer checks → Writer revises if score < 80
result = staff.chat(
    "2025年最值得关注的3个AI Agent框架，对比它们的核心差异",
    mode="research",
    return_details=True  # Get full CollaborationResult with score/trace
)

# Detailed result
print(f"\n{'='*50}")
print(f"Status: {result.status}")
print(f"Quality Score: {result.quality_score}/10")
print(f"Rounds: {result.rounds_used}")
print(f"Time: {result.total_time_sec:.1f}s")
print(f"Tokens: {result.total_tokens:,}")
print(f"\n{'='*50}")

# Extract the main output
if result.deliverables:
    for name, content in result.deliverables.items():
        print(f"\n📄 {name}:")
        print(content[:500] + ("..." if len(content) > 500 else ""))
