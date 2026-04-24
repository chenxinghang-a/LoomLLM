"""Example 1: Simple — 3行代码启动，零配置

Usage:
    set GEMINI_API_KEY=your-key    (Windows)
    export GEMINI_API_KEY=your-key (Linux/Mac)
    python examples/simple.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_staff_v4 import AIStaff

# 🚀 That's it. One line to create, one line to chat.
staff = AIStaff.from_env()

# Auto mode: AI-Staff figures out the best strategy
answer = staff.chat("用3句话解释什么是量子纠缠")
print(f"\n{'='*50}")
print(answer)
