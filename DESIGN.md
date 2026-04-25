# LoomLLM Design Philosophy

> Why it works the way it works.

---

## 1. Not Everything Needs a Roundtable

The #1 mistake in multi-agent frameworks: treating every question like it needs a committee meeting.

```
User: "1+1=?"
Framework: 🤖 Spinning up 5 agents... 💸💸💸
```

LoomLLM's `TaskClassifier` prevents this. It scores the input against 6 task types and picks the cheapest pipeline:

| Task Type | LLM Calls | Review? | Typical Tokens |
|-----------|-----------|---------|----------------|
| direct (Q&A) | 1 | No | ~100 |
| code | 2-3 | Yes | ~6K |
| research | 4-8 | No | ~20K |
| decision | 2-3 | Yes | ~8K |
| creative | 2-3 | Yes | ~6K |
| arena | 6+ | No | ~6K |

**A simple question costs 12 tokens, not 6,000.** This is 500x cheaper than running it through a full review loop.

### How TaskClassifier works

Keyword scoring + length heuristics + anti-keyword filtering:

- `"1+1"`, `"hello"`, `"什么是XX"` → direct (1 call)
- `"写代码"`, `"实现XX"`, `"debug"` → code (2-3 calls)
- `"研究"`, `"分析"`, `"深入"` → research (4+ calls)
- `"应该选"`, `"A还是B"`, `"对比"` → decision (2-3 calls)

Anti-keywords prevent misclassification: `"分析"` alone is research, but `"分析并实现"` is code.

---

## 2. Cost-Aware by Default

### The Two-Model Strategy

Most LLM output doesn't need a frontier model. A draft can come from a cheap model:

```
Writer (free/cheap model)     → Draft      (~3K tokens, $0.00)
Reviewer (strong model)       → Score+Fix  (~3K tokens, $0.01)
```

vs.

```
Single strong model call      → Output     (~3K tokens, $0.03)
```

**Same quality, 3x cheaper.** And when the draft is already good (score ≥ 80), the loop terminates after 1 iteration — no wasted rewrites.

### Model Tier System

```python
# SmartInit auto-classifies discovered models:
"free"    # gemini-2.5-flash-lite, glm-4-flash, qwen2.5-7b
"cheap"   # deepseek-chat, moonshot-v1
"standard" # gpt-4o-mini, gemini-2.5-flash
"premium"  # gpt-4o, claude-3.5-sonnet
```

Writer defaults to the cheapest available. Reviewer defaults to the strongest. You can override both.

---

## 3. Quality Gate, Not Guesswork

Without a quality gate, you're just hoping the LLM's first answer is good enough. With LoomLLM:

```python
result = staff.chat("Write quicksort", mode="code", return_details=True)
print(result.quality_score)    # 92/100
print(result.rounds_used)      # 1 (passed first try)
```

If the score is below threshold (default: 80/100), the Writer rewrites with specific feedback from the Reviewer. This isn't "try again" — it's "fix these 3 issues and resubmit."

### Review format

```json
{
  "score": 72,
  "issues": ["Missing edge case for empty arrays", "No complexity analysis"],
  "suggestions": ["Add base case for len(arr) <= 1", "Include Big-O annotation"],
  "strengths": ["Clean recursion", "Good naming"]
}
```

Concrete, actionable, not vague. The Writer knows *exactly* what to fix.

### Debate protocol

The Writer can argue back. If the Reviewer says "add error handling" but the code already has it, the Writer explains why the criticism is invalid. The score adjusts. This prevents over-iteration on stylistic preferences.

---

## 4. Zero Config is a Feature

Most LLM frameworks require:
1. Install the framework
2. Install a provider SDK
3. Write configuration code
4. Handle errors yourself

LoomLLM requires:
1. Set one environment variable
2. `staff = AIStaff.from_env()`

`SmartInit` does the rest:
- Scans all provider environment variables
- Tests connectivity to each (concurrent, 20s timeout)
- Lists available models with tier classification
- Picks the best default model
- Detects local proxy (Clash/V2Ray) for overseas providers

### Proxy detection

Chinese providers (DeepSeek, Zhipu, SiliconFlow, etc.) never get proxy — direct connect is faster. Overseas providers (Google AI, OpenAI, Groq) auto-detect local proxy on common ports (7890, 1080, 8080).

---

## 5. OpenAI Format Only

LoomLLM uses **one API format**: OpenAI-compatible (`/v1/chat/completions`).

Why:
- Every major provider offers an OpenAI-compatible endpoint
- No vendor-specific SDKs to install
- No proprietary APIs to learn
- Switching providers = changing one environment variable
- Future providers that speak OpenAI = zero code changes

This is the "USB-C of LLM APIs" principle. One connector, every device.

---

## 6. When to Use What

### Use `mode="auto"` (default) when:
- You don't know what type of task it is
- You want the framework to optimize cost vs quality

### Use `mode="code"` when:
- You're generating code that needs to work
- You want a quality score before shipping

### Use `mode="arena"` when:
- You want to compare multiple models side-by-side
- You're evaluating which model handles your use case best

### Use `mode="research"` when:
- You need deep, multi-perspective analysis
- One answer isn't enough — you want follow-up questions

### Use `mode="direct"` when:
- You know it's a simple question
- You want maximum speed, minimum tokens

---

## 7. Cost Optimization Checklist

1. **Use free-tier models when possible** — `gemini-2.5-flash-lite`, `glm-4-flash`, `qwen2.5-7b`
2. **Let TaskClassifier do its job** — Don't force `mode="code"` for simple questions
3. **Set quality_threshold wisely** — 80 is good for most tasks. 90+ burns more tokens.
4. **Use `auto_save=True`** (default) — Results are saved automatically, no lost work
5. **Monitor token usage** — `result.total_tokens` tells you exactly what you spent
6. **Configure daily budget** — Prevent runaway API costs with `budget.daily_limit_usd`

---

## 8. Architecture Decisions

| Decision | Why |
|----------|-----|
| Keyword-based classification, not LLM-based | Saves an API call per classification. Keywords + heuristics are fast and free. |
| OpenAI format only | Universal compatibility, zero vendor lock-in |
| SQLite for memory | Zero-config, no server, file-based, good enough |
| YAML for expert config | Human-editable, no code changes needed |
| Event bus for observability | Decoupled, anyone can subscribe to any event |
| Auto-save by default | Lost output = wasted tokens = wasted money |
| Debate protocol | Prevents over-iteration on subjective preferences |
