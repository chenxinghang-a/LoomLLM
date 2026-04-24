"""Example: Creative writing with critique loop.

Usage:
    export GEMINI_API_KEY=your-key
    python examples/creative.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_staff_v4 import AIStaff

staff = AIStaff.from_env()

# Creative mode: Writer drafts → Critic reviews → Writer refines
result = staff.chat("Write a compelling product tagline for an AI-powered code review tool", mode="creative")
print(result)
