"""Example: Deep research with multi-round follow-ups.

Usage:
    export GEMINI_API_KEY=your-key
    python examples/research.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_staff_v4 import AIStaff

staff = AIStaff.from_env()

# Research mode: Initial analysis → Follow-up deepening → Synthesis
result = staff.chat("Current state and future of AI agent frameworks in 2025", mode="research")
print(result)
