# LoomLLM

> **3行代码，10个Provider，自动质量闭环。**

```python
from ai_staff_v4 import AIStaff

staff = AIStaff.from_env()                      # 设一个API Key，自动发现一切
result = staff.chat("写个快排")                   # 自动分类 → 起草 → 审查 → 迭代
print(f"质量评分: {result.quality_score}/100")    # 内置质量门控
```

---

## 核心差异

**别的框架让你调API。LoomLLM让输出真正靠谱。**

| | LangChain | CrewAI | AutoGen | **LoomLLM** |
|---|-----------|--------|---------|-------------|
| 第一次出结果要几行 | 50+ | 30+ | 40+ | **3** |
| 质量门控 | 自己写 | 没有 | 没有 | **自动审查+重写** |
| 便宜模型起草+强模型审查 | 手动 | 没有 | 没有 | **内置** |
| 429自动降级 | 自己写 | 没有 | 没有 | **自动** |
| "我的回答好不好？" | 靠猜 | 靠猜 | 靠猜 | **百分制评分** |
| 简单问题浪费token？ | 是 | 是 | 是 | **不——自动走快速路径** |

### 核心思路

```
1. 分类任务（简单问答？代码？研究？）
2. 选最便宜的模型搞定它
3. 起草 → 审查 → 迭代直到质量达标
4. 自动保存结果
```

简单问题 `"1+1=?"` 只需 **1次调用，12个token**。
代码任务 `"写个快排"` 需要 **3次调用，~6K token**，但评分 **92/100**。

不浪费token。不手动调prompt。不"希望这次够好"。

---

## 特性

### 🎯 智能路由——简单问题不烧贵模型

```python
staff.chat("1+1等于几")          # → direct: 1次调用，12 token
staff.chat("写个快排")            # → code: Coder+Reviewer闭环，92/100
staff.chat("AI趋势分析")          # → research: 多轮追问
staff.chat("React vs Vue")       # → decision: 多维度对比
```

`TaskClassifier` 对输入做关键词评分，路由到最便宜的管线。简单问答直接跳过审查闭环。

### 🔄 V5闭环——质量引擎

```
Writer（便宜模型）→ Reviewer（强模型）→ 分数 < 80？→ 带反馈重写 → 循环
```

- **成本感知**：快模型起草，强模型只审查（省~50% token）
- **结构化反馈**：审查返回具体问题+建议，不是"再改改"
- **辩论协议**：Writer可以反驳不合理的批评，防止过度迭代
- **自动终止**：分数达标/最大轮次/无改进 → 停

### 🔌 10个Provider，零厂商锁定

| Provider | 直连 | 免费额度 | 需代理 |
|----------|------|----------|--------|
| DeepSeek | ✅ | ❌ | ❌ |
| 智谱GLM | ✅ | ✅ glm-4-flash | ❌ |
| SiliconFlow | ✅ | ✅ Qwen2.5-7B | ❌ |
| Moonshot | ✅ | ❌ | ❌ |
| 通义千问 | ✅ | ✅ | ❌ |
| Google AI | ❌ | ✅ flash-lite | ✅ |
| OpenAI | ❌ | ❌ | ✅ |
| Groq | ❌ | ✅ | ✅ |
| Anthropic | ❌ | ❌ | ✅ |
| Ollama | 本地 | ✅ | ❌ |

全部使用 **OpenAI兼容格式**（`/v1/chat/completions`）。无厂商锁定，无私有API。会说OpenAI格式的，就能用。

### 🛡️ 级联降级

```
Provider A (429) → Provider B (503) → Provider C (200 OK ✓)
```

### 💰 Token预算 & 成本追踪

```
💰 gemini-2.5-flash-lite | 1,234 tok | free | 合计: 5,678 tokens (3次调用)
```

### 💾 自动保存

每次`chat()`自动保存输出到带时间戳的目录：

```
ai_staff_code_Write_quicksort_20260425_153000/
├── solution.py          # 代码
├── report.md            # 质量报告
└── transcript.txt       # 完整执行日志
```

### 🧠 持久化记忆

SQLite存储的对话历史，跨session保持：

```python
staff = AIStaff.from_env(session_id="project-x")
staff.chat("我在做Flask应用")    # 下次对话还记得
```

---

## 快速开始

### 安装
```bash
pip install httpx pyyaml
```

### 设置一个API Key（任选）
```bash
# 任选一个——都能直接用
export DEEPSEEK_API_KEY=your-key       # 性价比最高，国内直连
export ZHIPU_API_KEY=your-key          # 有免费额度（glm-4-flash）
export OPENAI_API_KEY=your-key         # 标准选择
export GEMINI_API_KEY=your-key         # 有免费额度（flash-lite）
```

<details>
<summary>所有支持的Key</summary>

| Key | Provider | 获取地址 | 免费？ |
|-----|----------|----------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek | [platform.deepseek.com](https://platform.deepseek.com/api_keys) | ❌ |
| `ZHIPU_API_KEY` | 智谱GLM | [open.bigmodel.cn](https://open.bigmodel.cn/usercenter/apikeys) | ✅ |
| `SILICONFLOW_API_KEY` | SiliconFlow | [cloud.siliconflow.cn](https://cloud.siliconflow.cn/account/ak) | ✅ |
| `MOONSHOT_API_KEY` | Moonshot | [platform.moonshot.cn](https://platform.moonshot.cn/console/api-keys) | ❌ |
| `QWEN_API_KEY` | 通义千问 | [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com/apiKey) | ✅ |
| `GEMINI_API_KEY` | Google AI | [aistudio.google.com](https://aistudio.google.com/apikey) | ✅ |
| `OPENAI_API_KEY` | OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | ❌ |
| `GROQ_API_KEY` | Groq | [console.groq.com](https://console.groq.com/keys) | ✅ |
| `ANTHROPIC_API_KEY` | Anthropic | [console.anthropic.com](https://console.anthropic.com/) | ❌ |

</details>

### 运行
```python
from ai_staff_v4 import AIStaff

# 零配置：自动检测Key、测试连通性、选最佳模型
staff = AIStaff.from_env()

# 完事。开始聊。
answer = staff.chat("你好")
```

### 或用交互式向导
```bash
python -m ai_staff_v4 setup
```

---

## 使用场景

### 简单问答——自动快速路径
```python
staff.chat("asyncio是什么？")     # 1次调用，不审查，~100 token
```

### 代码+质量门控
```python
result = staff.chat("写个快排", mode="code", return_details=True)
print(result.quality_score)        # 92/100
print(result.rounds_used)          # 1（首次审查即通过）
```

### 多模型竞技场
```python
report = staff.chat("解释量子纠缠", mode="arena")
# 6个模型同台竞技，按质量排名
```

### 深度研究
```python
report = staff.chat("对比Python asyncio和threading", mode="research")
# 多轮追问，层层深入
```

### 自定义专家
```yaml
# experts/experts.yaml
- id: code_reviewer
  name: 代码审查员
  system_prompt: "你是资深工程师，审查代码的正确性和性能。"
  temperature: 0.3
```

---

## 设计哲学

详见 [DESIGN.md](DESIGN.md)。要点：

1. **不是什么都需要圆桌会议**——简单问答1次调用搞定，不该开5个Agent开会
2. **默认成本感知**——便宜模型起草，强模型审查
3. **质量 > 速度 > 成本**——但绝不在简单任务上浪费token
4. **零配置是特性**——设一个Key，得10个Provider
5. **只用OpenAI格式**——无厂商锁定，无私有API

---

## 测试

```bash
# 单元测试（不需要API Key）
python -m unittest ai_staff_v4.tests.test_core -v

# 快速检查
python -c "from ai_staff_v4 import AIStaff; print('OK')"
```

---

## 依赖

- Python 3.10+
- httpx >= 0.27
- pyyaml >= 6.0

---

## 对比

| 特性 | LoomLLM | LangChain | CrewAI | AutoGen |
|------|---------|-----------|--------|---------|
| 第一次聊天几行代码 | **3** | 50+ | 30+ | 40+ |
| 迭代质量闭环 | **内置** | 手动 | 手动 | 手动 |
| 多Provider | **10个内置** | 手动 | 手动 | 手动 |
| 零配置启动 | **是** | 否 | 否 | 否 |
| 429自动降级 | **是** | 否 | 否 | 否 |
| Token成本追踪 | **实时** | 否 | 否 | 否 |
| 任务自动分类 | **是** | 否 | 否 | 否 |
| 仅OpenAI兼容 | **是** | 多格式 | 多格式 | 多格式 |

---

## 许可

[MIT](LICENSE)

---

**[English](README.md)** | **[设计哲学](DESIGN.md)**
