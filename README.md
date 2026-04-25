# LoomLLM — The Iterative LLM Framework

> Write → Review → Refine → Repeat. Automatically.

**One function call, 10 providers, zero config.**

```python
from ai_staff_v4 import AIStaff

staff = AIStaff.from_env()                          # Auto-detect API keys, zero config
answer = staff.chat("Write a quicksort")            # Auto-classify → select expert → iterate until quality threshold
print(f"Score: {answer.quality_score}/100")          # Built-in quality scoring
```

## Why LoomLLM?

Existing frameworks make you **assemble your own pipeline**. LoomLLM **is** the pipeline.

| | LangChain | CrewAI | **LoomLLM** |
|---|---|---|---|
| Setup | Write 50 lines of chain | Configure 4 YAML files | **3 lines of code** |
| Quality control | Build it yourself | Manual configuration | **Auto-review + rewrite loop** |
| Multi-model | Single model | Single model | **Fast model writes, strong model reviews** |
| Cost awareness | ❌ | ❌ | **Real-time token + cost tracking** |
| Graceful degradation | Write it yourself | ❌ | **Auto-fallback on 429 errors** |

**One-liner positioning**: LangChain = building blocks. CrewAI = assembly line. LoomLLM = autonomous workstation.

---

## Core Features

### 1. Zero-Config Startup

Set **one** environment variable and LoomLLM auto-discovers all available models:

```python
# Just set any provider key
# export GEMINI_API_KEY=your-key     # or DEEPSEEK_API_KEY, OPENAI_API_KEY, etc.
staff = AIStaff.from_env()
# → Auto-discovers: gemini-2.5-flash-lite(free), deepseek-chat(cheap)...
```

`SmartInit` scans environment variables, tests connectivity, lists available models, and configures the best backend — all automatically.

### 2. Iterative Refinement Loop (V5 CollabLoop)

This is LoomLLM's core innovation. Instead of a single LLM call and hoping for the best:

```
Writer (fast/cheap model) → Reviewer (strong model) → Score < 80? → Rewrite with feedback → Repeat
```

**How it saves ~50% cost:**
- The **Writer** uses a fast, cheap model (e.g. `gemini-2.5-flash-lite`, `deepseek-chat`)
- The **Reviewer** uses a stronger model (e.g. `gemini-3-flash`, `deepseek-reasoner`)
- Only the review step needs the expensive model — the draft + rewrite use the cheap one

**Structured feedback, not vague handwaving:**
```json
{
  "score": 72,
  "issues": ["Missing edge case for empty arrays", "No complexity analysis"],
  "suggestions": ["Add base case for len(arr) <= 1", "Include Big-O annotation"],
  "strengths": ["Clean recursion structure", "Good variable naming"]
}
```

**Debate protocol:** The Writer can argue against the Reviewer's criticism. If the argument is valid, the score adjusts upward. This prevents over-iteration on stylistic preferences.

**Auto-termination:** The loop stops when:
- Quality score ≥ threshold (default: 80/100)
- Max iterations reached (default: 3)
- No meaningful improvement between rounds

### 3. Smart Task Routing

You don't need to pick a mode — LoomLLM auto-classifies your input:

```python
staff.chat("1+1=?")                # → direct: single call, no loop
staff.chat("Write quicksort")      # → code: Coder + Critic loop
staff.chat("AI trend analysis")    # → research: multi-turn inquiry
staff.chat("React vs Vue")         # → decision: multi-dimensional analysis
staff.chat("Write a slogan")       # → creative: creative + review
```

**TaskClassifier** uses the LLM itself to categorize the input, then routes to the optimal pipeline:
- **direct**: Simple Q&A → 1 API call, no overhead
- **code**: Programming → Coder writes, Critic reviews, loop until quality met
- **research**: Open-ended research → Multi-turn follow-up questions to build depth
- **decision**: Comparative analysis → Multiple perspectives, structured pros/cons
- **creative**: Creative writing → Writer + Reviewer loop with creative-focused prompts

### 4. 10 Providers, One API

| Provider | Direct Connect | Free Tier | Needs Proxy |
|----------|---------------|-----------|-------------|
| DeepSeek | ✅ | ❌ | ❌ |
| Zhipu GLM | ✅ | ✅ glm-4-flash | ❌ |
| SiliconFlow | ✅ | ✅ Qwen2.5-7B | ❌ |
| Moonshot (Kimi) | ✅ | ❌ | ❌ |
| Qwen (DashScope) | ✅ | ❌ | ❌ |
| Gemini | ❌ | ✅ flash-lite | ✅ |
| OpenAI | ❌ | ❌ | ✅ |
| Groq | ❌ | ✅ | ✅ |
| Anthropic | ❌ | ❌ | ✅ |
| Ollama | Local | ✅ | ❌ |

**All providers use the OpenAI-compatible API format** (`/v1/chat/completions`). No vendor lock-in, no proprietary APIs. If a provider offers an OpenAI-compatible endpoint, it works with LoomLLM.

**Provider features:**
- **Smart proxy detection**: Automatically detects local proxy (Clash/V2Ray) via port scanning
- **needs_proxy flag**: Chinese providers (DeepSeek, Zhipu, etc.) never get proxy — avoids latency
- **Model listing**: Gemini supports live model discovery via API; others use curated model lists
- **Tier classification**: free / cheap / standard / premium — used for cost-aware routing

### 5. Cascade Fallback

When a provider fails (429 quota, 503 overload, network timeout), LoomLLM automatically falls back:

```
gemini-2.5-flash-lite (429) → gemini-3-flash-preview (503) → deepseek-chat (200 OK)
```

**Fallback strategy:**
1. Same provider, different model (tier downgrade)
2. Different provider, similar model (cross-provider)
3. Stop if all exhausted (report which providers failed)

### 6. Token Budget & Cost Tracking

Real-time cost display on every call:

```
💰 Budget  gemini-2.5-flash-lite | 1,234 tok | free  ←  Total: 5,678 tokens (3 calls) | 2,345 tok/s | free | 2.4s
```

**Features:**
- Per-call token counting (prompt + completion)
- Cost estimation based on model pricing
- Daily budget limit with warning threshold
- Accumulated session totals

### 7. Colorized Process Logging

See exactly what's happening in real-time:

```
🟢 Writer      Started #1
🔵 Reviewer    Score: 85/100 | 1.2s | 2,345ch
✅ Done        PASSED (score=85 >= 80)
```

Or when the loop iterates:

```
🟢 Writer      Started #1
🔴 Reviewer    Score: 62/100 | Issues: 3 | Suggestions: 2
🟡 Writer      Rewriting #2 with feedback...
🔵 Reviewer    Score: 84/100 | 1.1s | 2,890ch
✅ Done        PASSED (score=84 >= 80) | 2 rounds | 5,135 tokens
```

### 8. Custom Experts

Edit `experts/experts.yaml` — no code changes needed:

```yaml
- id: code_reviewer
  name: Code Reviewer
  system_prompt: "You are a senior software engineer. Review code for correctness, performance, and readability."
  temperature: 0.3
  require_review: false    # Skip the review loop for this expert
```

**Built-in experts:** generalist, coder, critic, researcher, creative_writer, analyst, reviewer

### 9. Persistent Memory

SQLite-backed conversation history that persists across sessions:

```python
staff = AIStaff.from_env(session_id="project-x")
staff.chat("I'm working on a Flask app")    # Remembered in next session
staff.chat("Add authentication to my app")  # Knows context from previous chat
```

---

## Quick Start

### Install
```bash
pip install httpx pyyaml
```

### First-Time Setup (Interactive Wizard)
```bash
python -m ai_staff_v4 setup
```
This launches the setup wizard that:
- Shows all supported providers (with key registration links)
- Lets you enter your API key interactively
- Verifies connectivity immediately
- Suggests permanent environment variable setup

**Don't have an API key yet?** Pick one:
| Provider | Get Key | Free Tier? | Proxy Needed? |
|----------|---------|------------|---------------|
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com/api_keys) | ❌ | ❌ |
| Zhipu GLM | [open.bigmodel.cn](https://open.bigmodel.cn/usercenter/apikeys) | ✅ glm-4-flash | ❌ |
| Qwen | [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com/apiKey) | ✅ | ❌ |
| Moonshot | [platform.moonshot.cn](https://platform.moonshot.cn/console/api-keys) | ❌ | ❌ |
| Gemini | [aistudio.google.com](https://aistudio.google.com/apikey) | ✅ flash-lite | ✅ |
| OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | ❌ | ✅ |

### Run
```python
from ai_staff_v4 import AIStaff

# Option 1: Environment variable (recommended)
# Set any provider key: GEMINI_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY, etc.
staff = AIStaff.from_env()

# Option 2: Direct key
staff = AIStaff.quick_start("your-api-key", provider="deepseek")  # or "gemini", "openai", "moonshot", ...

# Option 3: YAML config (multi-backend)
staff = AIStaff.from_config_file("config.yaml")

# Chat
answer = staff.chat("Hello")
```

### Specify Mode
```python
staff.chat("Write quicksort", mode="code")         # Code + review
staff.chat("AI trend analysis", mode="research")    # Multi-turn research
staff.chat("React vs Vue", mode="decision")         # Multi-dimensional analysis
staff.chat("Write a slogan", mode="creative")       # Creative + review
```

### Inspect Results
```python
result = staff.chat("Write quicksort", mode="code", return_details=True)
print(f"Quality score: {result.quality_score}/100")
print(f"Iterations: {result.rounds_used}")
print(f"Tokens used: {result.total_tokens:,}")
print(f"Experts involved: {result.experts_used}")
```

---

## Architecture

```
User Input → chat()
  │
  ├─ TaskClassifier (auto-detect task type)
  │   ├─ direct (simple)  → Single call
  │   ├─ code (coding)    → Coder + Critic loop
  │   ├─ research (study) → Multi-turn inquiry
  │   └─ complex (hard)   → Full V5 collab loop
  │
  └─ V5 CollabLoop
      ├─ Writer (fast model) drafts
      ├─ Reviewer (strong model) scores & gives feedback
      ├─ Score < 80? → Writer rewrites with feedback
      ├─ Debate protocol: Writer can argue back
      └─ Auto-terminate: Quality met or timeout
```

```
ai_staff_v4/
├── core/              # Infrastructure
│   ├── verbose.py     # Colorized logging + token cost display
│   ├── budget.py      # Token budget management
│   ├── events.py      # Event bus (pub/sub)
│   ├── memory.py      # SQLite persistent memory
│   ├── constants.py   # Global constants
│   └── validation.py  # Output validation
├── experts/           # Expert roles
│   ├── experts.yaml   # User-editable expert config
│   ├── registry.py    # Expert registry
│   └── classifier.py  # Task auto-classification
├── agents/            # AI sub-agents
│   ├── collab_loop.py # V5 iterative collaboration engine
│   ├── cot.py         # Chain-of-thought planner
│   ├── executor.py    # Execution agent
│   ├── reviewer.py    # Review agent
│   ├── memory_agent.py# Memory-aware agent
│   ├── base.py        # Base agent class
│   └── types.py       # Shared type definitions
├── backends/          # LLM backends
│   ├── client.py      # Unified API client (OpenAI-compatible)
│   ├── smart_init.py  # Zero-config auto-discovery
│   ├── multi_client.py# Multi-backend manager
│   ├── profile.py     # Backend profile dataclass
│   ├── router.py      # Smart routing
│   └── fallback.py    # Cascade fallback
├── main_mod/
│   ├── staff.py       # AIStaff orchestrator
│   └── startup.py     # Zero-config startup logic
├── examples/
│   ├── simple.py      # 3-line quickstart
│   ├── research_flow.py # V5 research loop
│   └── expert_task.py   # Code + review
├── tests/
│   └── test_core.py   # Core unit tests
├── config_template.yaml # Config template (10 providers)
├── requirements.txt
├── LICENSE            # MIT
└── README.md
```

---

## Configuration

### Single environment variable (simplest)
```bash
# Any one of these works:
export GEMINI_API_KEY=your-key        # Gemini (free flash-lite)
export DEEPSEEK_API_KEY=your-key      # DeepSeek (best value)
export ZHIPU_API_KEY=your-key         # Zhipu GLM (glm-4-flash is free)
```

### config.yaml (multi-backend)
Copy `config_template.yaml` and fill in your keys. 10 providers pre-configured — uncomment what you need.

<details>
<summary>Full config template</summary>

```yaml
default_model: ""                     # Leave empty = auto-select

settings:
  proxy: ""                           # Required for overseas providers (Gemini/OpenAI/Groq/Anthropic)
  default_expert: "generalist"
  timeout: 120
  max_retries: 3
  language: "zh-CN"

profiles:
  # ── China-direct (no proxy needed) ──
  deepseek:
    provider: "deepseek"
    base_url: "https://api.deepseek.com/v1"
    api_key: "${DEEPSEEK_API_KEY}"
    model: "deepseek-chat"
    tier: "cheap"
    priority: 9

  # ── Overseas (proxy required) ──
  gemini_flash:
    provider: "gemini"
    base_url: "https://generativelanguage.googleapis.com/v1beta/openai"
    api_key: "${GEMINI_API_KEY}"
    model: "gemini-2.5-flash-lite"
    tier: "free"
    priority: 10

budget:
  daily_limit_usd: 1.0
  warn_threshold: 0.7
  enable_tracking: true

memory:
  db_path: ".ai_staff_memory.db"
  max_history_per_session: 100
  auto_summarize_every: 20
```

</details>

---

## Testing

```bash
# Unit tests (no API key needed)
python -m unittest ai_staff_v4.tests.test_core -v

# Quick verify import
python -c "from ai_staff_v4 import AIStaff; print('OK')"

# Run examples (API key required)
python examples/simple.py
python examples/research_flow.py
python examples/expert_task.py
```

---

## Requirements

- Python 3.10+
- httpx >= 0.27
- pyyaml >= 6.0

---

## Comparison with Alternatives

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

**[中文文档](README_CN.md)**
