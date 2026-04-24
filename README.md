# AI-Staff V4 — AI Collaboration Framework

> **Multi-model AI collaboration with closed-loop quality control**
> One API key → Auto-classify → Writer+Reviewer loop → Structured output

---

## Features

| Feature | Description |
|---------|-------------|
| **Unified `chat()` API** | One method, 7 modes: auto/direct/code/research/decision/creative/collab |
| **Closed-loop collaboration** | Writer → Reviewer → Rebuttal → Revise, until quality threshold met |
| **Debate protocol** | Executor can defend its output; Reviewer re-evaluates after rebuttal |
| **Multi-model routing** | Fast model for writing, strong model (Gemini 3.x) for reviewing |
| **8 built-in experts** | Generalist, Coder, Researcher, Critic, Writer, Planner, Analyst, Engineer |
| **REST API** | `/chat` and `/run` endpoints, ready to integrate |
| **Full observability** | `trace_id` tracks every step; structured feedback with scores |

## Architecture

```
ai_staff_v4/
├── main_mod/
│   ├── staff.py         # AIStaff main controller + chat() unified entry
│   └── startup.py       # from_env / quick_start / from_config_file
├── agents/
│   ├── collab_loop.py   # V5 closed-loop collaboration engine (core!)
│   ├── cot.py           # Chain-of-Thought planning agent
│   ├── executor.py      # Task execution agent
│   ├── reviewer.py      # Quality review agent
│   └── memory_agent.py  # Session summarization agent
├── backends/
│   ├── client.py        # LLMClient (OpenAI-compatible)
│   ├── multi_client.py  # Multi-backend client manager
│   ├── smart_init.py    # SmartInit V2 (auto-discover keys/models)
│   └── ai_router.py     # AI-driven routing (experimental)
├── experts/
│   ├── classifier.py    # TaskClassifier (7 task types)
│   └── registry.py      # ExpertRegistry (8 experts)
├── core/
│   ├── events.py        # Event bus (pub/sub)
│   ├── memory.py        # SQLite session memory
│   └── constants.py     # Shared constants
├── endpoints/
│   ├── rest_api.py      # HTTP API server
│   └── mcp_bridge.py    # MCP protocol bridge
├── examples/            # Usage examples
└── config_template.yaml # Configuration template
```

## Quick Start

### Install

```bash
pip install httpx pyyaml
```

### Set API Key

```bash
# Option A: Environment variable (recommended)
export GEMINI_API_KEY=your-key-here

# Option B: Other providers
export OPENAI_API_KEY=sk-...
export DEEPSEEK_API_KEY=sk-...
```

### Use

```python
from ai_staff_v4 import AIStaff

# Zero-config launch (auto-detects keys from environment)
staff = AIStaff.from_env()

# Unified chat() — the only method you need
staff.chat("What is recursion?")                        # auto mode
staff.chat("Write quicksort in Python", mode="code")    # code mode
staff.chat("AI trends in 2025", mode="research")        # research mode
staff.chat("React vs Vue for SPA?", mode="decision")    # decision mode
staff.chat("Slogan for AI product", mode="creative")    # creative mode
```

### Modes

| Mode | Description | Use When |
|------|-------------|----------|
| `auto` | Auto-classify + route (default) | Don't know which mode to use |
| `direct` | Quick Q&A | Simple questions, translation, definitions |
| `code` | Write + review code | Coding, debugging |
| `research` | Multi-round deep research | Tech trends, overviews |
| `decision` | Multi-perspective analysis | Tech comparison, buy/decide |
| `creative` | Creative + critique | Copywriting, naming, proposals |
| `collab` | Multi-expert collaboration | Complex multi-domain tasks |
| `arena` | Cross-model comparison | Model benchmarking |

## Collaboration Loop (V5)

The core innovation: **AI ↔ AI closed-loop with debate protocol**.

```
┌─────────┐    ┌──────────┐    ┌─────────┐
│  Writer  │───▶│ Reviewer │───▶│  Judge  │
│ (fast)   │    │ (strong) │    │ score≥80│
└─────────┘    └──────────┘    └────┬────┘
     ▲                              │ No
     │         ┌──────────┐         │
     │         │ Rebuttal │◀────────┘
     │         │ (defend) │    ┌────────┐
     │         └──────────┘    │Rejudge │
     │              │          └───┬────┘
     │              ▼              │
     │         ┌──────────┐       │ Still fail
     └─────────│  Revise  │◀──────┘
               └──────────┘
```

1. **Writer** generates initial draft (fast model like `gemini-2.5-flash-lite`)
2. **Reviewer** gives structured feedback with score (strong model like `gemini-3.x`)
3. **Judge**: Score ≥ threshold → done. Otherwise →
4. **Rebuttal**: Writer can *defend* its choices (not blindly accept criticism)
5. **Rejudge**: Reviewer sees the rebuttal, may adjust score up
6. **Revise**: If still failing, Writer revises with specific feedback
7. Loop until quality threshold met or max iterations reached

## REST API

```bash
# Start server
python -c "
from ai_staff_v4.endpoints.rest_api import RestAPIServer
from ai_staff_v4 import AIStaff
staff = AIStaff.from_env()
srv = RestAPIServer(staff)
srv.start()
"
```

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/chat` | POST | `{"prompt": "...", "mode": "auto"}` | Unified chat entry |
| `/run` | POST | `{"prompt": "...", "max_iterations": 3}` | V5 closed-loop run |
| `/status` | GET | — | Runtime status |
| `/health` | GET | — | Health check |
| `/experts` | GET | — | Expert list |

## Multi-Backend Config

Edit `config.yaml`:

```yaml
profiles:
  gemini_flash:
    provider: gemini
    base_url: https://generativelanguage.googleapis.com/v1beta/openai
    api_key: ${GEMINI_API_KEY}
    model: gemini-2.5-flash
    priority: 10

  deepseek:
    provider: deepseek
    base_url: https://api.deepseek.com/v1
    api_key: ${DEEPSEEK_API_KEY}
    model: deepseek-chat
    priority: 7
```

Then:

```python
staff = AIStaff.from_config_file("config.yaml")
```

## Examples

See `examples/` directory:

- `code_gen.py` — Code generation with review loop
- `research.py` — Deep research with multi-round follow-ups
- `creative.py` — Creative writing with critique

## Requirements

- Python 3.10+
- httpx (required)
- pyyaml (optional, for YAML config)

## License

MIT
