# LoomLLM

> **3 lines. 10 providers. Automatic quality loop.**

```python
from ai_staff_v4 import AIStaff

staff = AIStaff.from_env()                      # Set one API key, auto-discover everything
result = staff.chat("Write a quicksort")        # Auto-classify → draft → review → refine
print(f"Score: {result.quality_score}/100")     # Built-in quality gate
```

---

## What makes it different?

**Every LLM framework lets you call an API. LoomLLM makes the output *good*.**

| | LangChain | CrewAI | AutoGen | **LoomLLM** |
|---|-----------|--------|---------|-------------|
| Lines to first result | 50+ | 30+ | 40+ | **3** |
| Quality gate | Build it | No | No | **Auto review + rewrite** |
| Cheap model drafts, strong reviews | Manual | No | No | **Built-in** |
| 429 fallback | Build it | No | No | **Auto** |
| "Is my answer good?" | You guess | You guess | You guess | **Score out of 100** |
| Simple Q&A wastes tokens? | Yes | Yes | Yes | **No — auto fast-path** |

### The core idea

```
1. Classify the task (simple? code? research?)
2. Pick the cheapest model that can handle it
3. Draft → Review → Refine until quality passes
4. Auto-save the result
```

A simple question like `"1+1=?"` takes **1 API call, 12 tokens**.  
A code task like `"Write quicksort"` takes **3 calls, ~6K tokens** but scores **92/100**.

No wasted tokens. No manual prompt engineering. No "I hope this is good enough."

---

## Features

### 🎯 Smart Routing — Don't Burn Tokens on Easy Questions

```python
staff.chat("1+1=?")              # → direct: 1 call, 12 tokens
staff.chat("Write quicksort")    # → code: Coder + Critic loop, 92/100
staff.chat("AI trend analysis")  # → research: Multi-turn inquiry
staff.chat("React vs Vue")       # → decision: Multi-dimensional analysis
```

`TaskClassifier` scores your input against 6 task types, then routes to the cheapest pipeline that can deliver quality. Simple questions skip the review loop entirely.

### 🔄 V5 CollabLoop — The Quality Engine

```
Writer (cheap model) → Reviewer (strong model) → Score < 80? → Rewrite with feedback → Repeat
```

- **Cost-aware**: Fast/cheap model drafts, strong model only reviews (~50% token savings)
- **Structured feedback**: Reviewer returns specific issues + suggestions, not vague "make it better"
- **Debate protocol**: Writer can push back on unfair criticism, preventing over-iteration
- **Auto-terminate**: Stops when score ≥ threshold, max iterations reached, or no improvement

### 🔌 10 Providers, Zero Vendor Lock-in

| Provider | Direct | Free Tier | Proxy |
|----------|--------|-----------|-------|
| DeepSeek | ✅ | ❌ | ❌ |
| Zhipu GLM | ✅ | ✅ glm-4-flash | ❌ |
| SiliconFlow | ✅ | ✅ Qwen2.5-7B | ❌ |
| Moonshot | ✅ | ❌ | ❌ |
| Qwen | ✅ | ✅ | ❌ |
| Google AI | ❌ | ✅ flash-lite | ✅ |
| OpenAI | ❌ | ❌ | ✅ |
| Groq | ❌ | ✅ | ✅ |
| Anthropic | ❌ | ❌ | ✅ |
| Ollama | Local | ✅ | ❌ |

All providers use **OpenAI-compatible format** (`/v1/chat/completions`). No proprietary APIs, no vendor lock-in. If it speaks OpenAI, it works with LoomLLM.

### 🛡️ Cascade Fallback

```
Provider A (429) → Provider B (503) → Provider C (200 OK ✓)
```

When a provider fails (429 quota, 503 overload, timeout), LoomLLM automatically falls back across providers and models.

### 💰 Token Budget & Cost Tracking

```
💰 gemini-2.5-flash-lite | 1,234 tok | free | Total: 5,678 tokens (3 calls)
```

Per-call token counting, cost estimation, daily budget limits, session totals.

### 💾 Auto-Save

Every `chat()` call automatically saves output to a timestamped directory:

```
ai_staff_code_Write_quicksort_20260425_153000/
├── solution.py          # The code
├── report.md            # Quality report
└── transcript.txt       # Full execution log
```

### 🧠 Persistent Memory

SQLite-backed conversation history across sessions:

```python
staff = AIStaff.from_env(session_id="project-x")
staff.chat("I'm building a Flask app")   # Remembered next session
```

---

## Quick Start

### Install
```bash
pip install httpx pyyaml
```

### Set an API key (any one)
```bash
# Pick one — all work out of the box
export DEEPSEEK_API_KEY=your-key       # Best value, direct connect from China
export ZHIPU_API_KEY=your-key          # Free tier available (glm-4-flash)
export OPENAI_API_KEY=your-key         # Standard choice
export GEMINI_API_KEY=your-key         # Free tier available (flash-lite)
```

<details>
<summary>All supported keys</summary>

| Key | Provider | Get Key | Free? |
|-----|----------|---------|-------|
| `DEEPSEEK_API_KEY` | DeepSeek | [platform.deepseek.com](https://platform.deepseek.com/api_keys) | ❌ |
| `ZHIPU_API_KEY` | Zhipu GLM | [open.bigmodel.cn](https://open.bigmodel.cn/usercenter/apikeys) | ✅ |
| `SILICONFLOW_API_KEY` | SiliconFlow | [cloud.siliconflow.cn](https://cloud.siliconflow.cn/account/ak) | ✅ |
| `MOONSHOT_API_KEY` | Moonshot | [platform.moonshot.cn](https://platform.moonshot.cn/console/api-keys) | ❌ |
| `QWEN_API_KEY` | Qwen/DashScope | [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com/apiKey) | ✅ |
| `GEMINI_API_KEY` | Google AI | [aistudio.google.com](https://aistudio.google.com/apikey) | ✅ |
| `OPENAI_API_KEY` | OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | ❌ |
| `GROQ_API_KEY` | Groq | [console.groq.com](https://console.groq.com/keys) | ✅ |
| `ANTHROPIC_API_KEY` | Anthropic | [console.anthropic.com](https://console.anthropic.com/) | ❌ |

</details>

### Run
```python
from ai_staff_v4 import AIStaff

# Zero-config: detects keys, tests connectivity, picks best model
staff = AIStaff.from_env()

# That's it. Start chatting.
answer = staff.chat("Hello")
```

### Or use the setup wizard
```bash
python -m ai_staff_v4 setup
```

---

## Usage Patterns

### Simple Q&A — auto fast-path
```python
staff.chat("What is asyncio?")      # 1 call, no review, ~100 tokens
```

### Code with quality gate
```python
result = staff.chat("Write quicksort", mode="code", return_details=True)
print(result.quality_score)          # 92/100
print(result.rounds_used)            # 1 (passed on first review)
```

### Multi-model arena
```python
report = staff.chat("Explain quantum entanglement", mode="arena")
# 6 models compete, ranked by quality
```

### Deep research
```python
report = staff.chat("Compare asyncio vs threading in Python", mode="research")
# Multi-turn inquiry with follow-up questions
```

### Custom experts
```yaml
# experts/experts.yaml
- id: code_reviewer
  name: Code Reviewer
  system_prompt: "You are a senior engineer. Review for correctness and performance."
  temperature: 0.3
```

---

## Architecture

```
User Input → chat()
  │
  ├─ TaskClassifier
  │   ├─ direct    → 1 API call (12 tokens for "1+1=?")
  │   ├─ code      → Coder + Critic loop
  │   ├─ research  → Multi-turn follow-up inquiry
  │   ├─ decision  → Multi-perspective analysis
  │   └─ creative  → Writer + Reviewer loop
  │
  └─ V5 CollabLoop (for complex tasks)
      ├─ Writer drafts (fast/cheap model)
      ├─ Reviewer scores + gives feedback (strong model)
      ├─ Score < threshold? → Writer rewrites with feedback
      └─ Auto-terminate when quality passes
```

```
ai_staff_v4/
├── core/              # Infrastructure (logging, budget, events, memory)
├── experts/           # Expert roles (YAML-configurable)
├── agents/            # AI sub-agents (collab loop, reviewer, executor)
├── backends/          # 10 LLM providers (OpenAI-compatible)
├── main_mod/          # AIStaff orchestrator
├── examples/          # Working examples
└── tests/             # Unit tests (20/20 passing)
```

---

## Design Philosophy

See [DESIGN.md](DESIGN.md) for the full rationale. TL;DR:

1. **Not everything needs a roundtable** — Simple Q&A should be 1 call, not a 5-agent meeting
2. **Cost-aware by default** — Cheap model drafts, strong model reviews
3. **Quality > Speed > Cost** — But never waste tokens on trivial tasks
4. **Zero config is a feature** — Set one key, get 10 providers
5. **OpenAI format only** — No vendor lock-in, no proprietary APIs

---

## Testing

```bash
# Unit tests (no API key needed)
python -m unittest ai_staff_v4.tests.test_core -v

# Quick import check
python -c "from ai_staff_v4 import AIStaff; print('OK')"
```

---

## Requirements

- Python 3.10+
- httpx >= 0.27
- pyyaml >= 6.0

---

## Comparison

| Feature | LoomLLM | LangChain | CrewAI | AutoGen |
|---------|---------|-----------|--------|---------|
| Lines to first chat | **3** | 50+ | 30+ | 40+ |
| Iterative quality loop | **Built-in** | Manual | Manual | Manual |
| Multi-provider | **10 built-in** | Manual | Manual | Manual |
| Zero-config startup | **Yes** | No | No | No |
| Auto-fallback on 429 | **Yes** | No | No | No |
| Token cost tracking | **Real-time** | No | No | No |
| Task auto-classification | **Yes** | No | No | No |
| OpenAI-compatible only | **Yes** | Multi-format | Multi-format | Multi-format |

---

## License

[MIT](LICENSE)

---

**[中文文档](README_CN.md)** | **[设计哲学](DESIGN.md)**
