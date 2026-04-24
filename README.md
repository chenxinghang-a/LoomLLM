# 🦞 AI-Staff V4 — 全自动AI协作工作站

> 给任务 → 自动选专家 → 自动干活 → 自动审查 → 质量不过？自动重写

**3行代码启动，零配置：**

```python
from ai_staff_v4 import AIStaff

staff = AIStaff.from_env()          # 自动扫描API key，零配置
answer = staff.chat("写个快排")     # 自动分类→选专家→闭环协作
```

## ⚡ 为什么不用LangChain/CrewAI？

| | LangChain | CrewAI | **AI-Staff** |
|---|---|---|---|
| 启动 | 写50行chain | 配4个yaml | **3行代码** |
| 质量保障 | 自己写 | 手动配 | **自动审查+重写闭环** |
| 多模型 | 单模型 | 单模型 | **快模型写+强模型查** |
| 成本感知 | ❌ | ❌ | **实时Token+费用显示** |
| 降级容错 | 自己写 | ❌ | **429自动切后端** |

**一句话定位**：LangChain=积木，CrewAI=流水线，AI-Staff=全自动工作站。

## 🎯 核心特性

### 1. 零配置启动
```python
# 设置任一Provider的API Key，自动扫描所有可用模型
# 支持环境变量: GEMINI_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY 等
set GEMINI_API_KEY=your-key        # 或 DEEPSEEK_API_KEY, ZHIPU_API_KEY...
staff = AIStaff.from_env()
# → 自动发现: gemini-2.5-flash-lite(free), deepseek-chat(cheap)...
```

### 2. V5闭环协作（核心杀手锏）
```
Writer写初稿 → Reviewer审查 → 分数<80? → Writer带反馈重写 → Reviewer再查 → ...
```
- **快模型写，强模型查**（节省50%成本）
- **结构化反馈**：问题、建议、优点，不是笼统的"请改进"
- **辩论协议**：Writer可以对Reviewer的批评辩解，分数可能上调
- **自动终止**：达到质量阈值或最大迭代次数才停

### 3. 智能路由
```python
staff.chat("1+1等于几")        # → 快速问答，1次调用
staff.chat("写个快排")          # → 代码模式，编码+审查
staff.chat("AI趋势分析")        # → 研究模式，多轮追问
staff.chat("React vs Vue")      # → 决策模式，多维度分析
```
**不需要手动选模式**，AI-Staff自动分类+路由。

### 4. 10个Provider，开箱即用
| Provider | 直连 | 免费 | 需代理 |
|----------|------|------|--------|
| Gemini | ❌ | ✅ flash-lite | ✅ |
| DeepSeek | ✅ | ❌ | ❌ |
| 智谱GLM | ✅ | ✅ glm-4-flash | ❌ |
| 硅基流动 | ✅ | ✅ Qwen2.5-7B | ❌ |
| Kimi | ✅ | ❌ | ❌ |
| 通义千问 | ✅ | ❌ | ❌ |
| OpenAI | ❌ | ❌ | ✅ |
| Groq | ❌ | ✅ | ✅ |
| Anthropic | ❌ | ❌ | ✅ |
| Ollama | 本地 | ✅ | ❌ |

### 5. Token成本实时显示
```
  💰 Budget     gemini-3.1-flash-lite | 1,234 tok | free  ←  累计: 5,678 tokens (3 calls) | 2,345 tok/s | free | 2.4s
```

### 6. 彩色日志，过程可视化
```
  🟢 Writer      开始 #1
  🔵 Reviewer    评分:85/100 | 1.2s | 2,345ch
  ✅ Done        PASSED (score=85 >= 80)
```

## 🚀 Quick Start

### 安装
```bash
pip install httpx pyyaml
```

### 最简示例
```python
from ai_staff_v4 import AIStaff

# 方式1: 环境变量（推荐）
# set GEMINI_API_KEY=your-key   或  set DEEPSEEK_API_KEY=your-key
staff = AIStaff.from_env()

# 方式2: 直接传key
staff = AIStaff.quick_start("your-api-key", provider="gemini")  # provider可选: gemini/deepseek/openai/...

# 方式3: YAML配置（多后端）
staff = AIStaff.from_config_file("config.yaml")

# 开聊
answer = staff.chat("你好")
```

### 指定模式
```python
staff.chat("写个快排", mode="code")        # 代码+审查
staff.chat("AI趋势", mode="research")      # 多轮研究
staff.chat("React vs Vue", mode="decision")# 多维决策
staff.chat("写个slogan", mode="creative")  # 创意+审查
```

### 查看完整结果
```python
result = staff.chat("写个快排", mode="code", return_details=True)
print(f"质量评分: {result.quality_score}/10")
print(f"迭代次数: {result.rounds_used}")
print(f"Token消耗: {result.total_tokens:,}")
print(f"参与专家: {result.experts_used}")
```

## 📁 项目结构

```
ai_staff_v4/
├── core/           # 核心基础设施
│   ├── verbose.py  # 🆕 彩色日志+Token成本显示
│   ├── budget.py   # Token预算管理
│   ├── events.py   # 事件总线
│   ├── memory.py   # SQLite持久化记忆
│   └── validation.py # 输出校验
├── experts/        # 专家角色
│   ├── experts.yaml # 🆕 用户可编辑的专家配置
│   ├── registry.py  # 专家注册表
│   └── classifier.py # 任务自动分类
├── agents/         # AI子Agent
│   ├── collab_loop.py # V5闭环协作引擎
│   ├── cot.py       # 思维链规划
│   ├── executor.py  # 执行Agent
│   └── reviewer.py  # 审查Agent
├── backends/       # LLM后端
│   ├── client.py    # 统一API客户端
│   ├── smart_init.py # 零配置自动扫描
│   ├── multi_client.py # 多后端统一管理
│   ├── router.py    # 智能路由
│   └── fallback.py  # 级联降级
├── main_mod/
│   ├── staff.py     # AIStaff核心编排器
│   └── startup.py   # 零配置启动逻辑
├── examples/       # 🆕 3个最小示例
│   ├── simple.py    # 3行代码启动
│   ├── research_flow.py # V5闭环研究
│   └── expert_task.py   # 代码+审查
├── config_template.yaml # 配置模板（10个Provider）
└── README.md
```

## 🔧 配置

### 单环境变量（最简）
```bash
# 任选一个Provider，设置对应环境变量即可
set GEMINI_API_KEY=your-key        # Gemini (免费flash-lite)
set DEEPSEEK_API_KEY=your-key      # DeepSeek (性价比高)
set ZHIPU_API_KEY=your-key         # 智谱GLM (glm-4-flash免费)
```

### config.yaml（多后端）
```yaml
settings:
  proxy: ""                          # 海外Provider需设代理
  default_expert: generalist

profiles:
  deepseek:                          # 国内直连，无需代理
    base_url: "https://api.deepseek.com/v1"
    api_key: "${DEEPSEEK_API_KEY}"
    model: "deepseek-chat"
    tier: cheap
    priority: 9

  gemini_flash:                      # 需代理
    base_url: "https://generativelanguage.googleapis.com/v1beta/openai"
    api_key: "${GEMINI_API_KEY}"
    model: "gemini-2.5-flash-lite"
    tier: free
    priority: 10
```

### 自定义专家
编辑 `experts/experts.yaml`，无需改代码：
```yaml
- id: my_expert
  name: 我的专家
  system_prompt: "你是一个..."
  temperature: 0.5
  require_review: true
```

## 🧪 测试

```bash
# 快速验证
python -c "from ai_staff_v4 import AIStaff; print('OK')"

# 跑示例
python examples/simple.py
python examples/research_flow.py
python examples/expert_task.py
```

## 📊 架构图

```
用户输入 → chat()
  │
  ├─ TaskClassifier 自动分类
  │   ├─ direct(简单) → 单次调用
  │   ├─ code(代码)   → Coder + Critic
  │   ├─ research(研究)→ 多轮追问
  │   └─ complex(复杂) → V5闭环
  │
  └─ V5 CollabLoop
      ├─ Writer(快模型) 写初稿
      ├─ Reviewer(强模型) 审查
      ├─ 分数 < 80? → Writer带反馈重写
      ├─ 辩论协议: Writer可辩解
      └─ 自动终止: 达标或超时
```

## License

MIT
