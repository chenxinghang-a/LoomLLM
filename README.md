# LoomLLM — The Iterative LLM Framework

> Write → Review → Refine → Repeat. Automatically.

**One function call, 10 providers, zero config.**

```python
from ai_staff_v4 import AIStaff

staff = AIStaff.from_env()          # Auto-detect API keys, zero config
answer = staff.chat("Write a quicksort")  # Auto-classify → select expert → iterate until quality threshold
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

### The Core Innovation: Iterative Refinement Loop

```
Writer (fast/cheap model) → Reviewer (strong model) → Score < 80? → Rewrite with feedback → Repeat
```

- **Cost-efficient**: Fast model drafts, strong model inspects — saves ~50% tokens
- **Structured feedback**: Specific issues, suggestions, and strengths — not vague "please improve"
- **Debate protocol**: Writer can argue against reviewer criticism; score may adjust upward
- **Auto-termination**: Stops when quality threshold met or max iterations reached

### 10 Providers, One API

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

All providers use the **OpenAI-compatible API format**. No vendor lock-in, no proprietary APIs.

## Quick Start

### Install
```bash
pip install httpx pyyaml
```

### Run
```python
from ai_staff_v4 import AIStaff

# Option 1: Environment variable (recommended)
# Set any provider key: GEMINI_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY, etc.
staff = AIStaff.from_env()

# Option 2: Direct key
staff = AIStaff.quick_start("your-api-key", provider="gemini")  # or "deepseek", "openai", ...

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

## How It Works

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

## Configuration

### Single environment variable (simplest)
```bash
# Any one of these works:
export GEMINI_API_KEY=your-key        # Gemini (free flash-lite)
export DEEPSEEK_API_KEY=your-key      # DeepSeek (best value)
export ZHIPU_API_KEY=your-key         # Zhipu GLM (glm-4-flash is free)
```

### config.yaml (multi-backend)
Copy `config_template.yaml` and fill in your keys. 10 providers pre-configured, uncomment what you need.

## Project Structure

```
ai_staff_v4/
├── core/           # Infrastructure
│   ├── verbose.py  # Colorized logging + token cost display
│   ├── budget.py   # Token budget management
│   ├── events.py   # Event bus
│   ├── memory.py   # SQLite persistent memory
│   └── validation.py # Output validation
├── experts/        # Expert roles
│   ├── experts.yaml # User-editable expert config
│   ├── registry.py  # Expert registry
│   └── classifier.py # Task auto-classification
├── agents/         # AI sub-agents
│   ├── collab_loop.py # V5 iterative collaboration engine
│   ├── cot.py       # Chain-of-thought planner
│   ├── executor.py  # Execution agent
│   └── reviewer.py  # Review agent
├── backends/       # LLM backends
│   ├── client.py    # Unified API client (OpenAI-compatible)
│   ├── smart_init.py # Zero-config auto-discovery
│   ├── multi_client.py # Multi-backend manager
│   ├── router.py    # Smart routing
│   └── fallback.py  # Cascade fallback
├── main_mod/
│   ├── staff.py     # AIStaff orchestrator
│   └── startup.py   # Zero-config startup
├── examples/
│   ├── simple.py    # 3-line quickstart
│   ├── research_flow.py # V5 research loop
│   └── expert_task.py   # Code + review
├── config_template.yaml # Config template (10 providers)
└── README.md
```

## Testing

```bash
# Quick verify
python -c "from ai_staff_v4 import AIStaff; print('OK')"

# Run examples
python examples/simple.py
python examples/research_flow.py
python examples/expert_task.py
```

## Requirements

- Python 3.10+
- httpx >= 0.27
- pyyaml >= 6.0

## License

MIT
