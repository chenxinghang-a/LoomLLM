"""Example: Code generation with AI review loop.

Usage:
    export GEMINI_API_KEY=your-key
    python examples/code_gen.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_staff_v4 import AIStaff

# Zero-config: auto-detect API keys from environment
staff = AIStaff.from_env()

# Code mode: Writer generates → Reviewer checks → Writer revises
result = staff.chat("Write a binary search function in Python that handles edge cases", mode="code")
print(result)
