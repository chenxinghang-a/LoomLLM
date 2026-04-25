# LoomLLM — 迭代式LLM协作框架

> 写 → 审 → 改 → 再审。全自动。

**一行调用，10个Provider，零配置。**

```python
from ai_staff_v4 import AIStaff

staff = AIStaff.from_env()                          # 自动检测API Key，零配置
answer = staff.chat("写个快排")                      # 自动分类 → 选专家 → 闭环迭代到质量达标
print(f"质量评分: {answer.quality_score}/100")        # 内置质量评分
```

## 为什么用 LoomLLM？

现有框架让你**自己拼管线**。LoomLLM **就是**管线。

| | LangChain | CrewAI | **LoomLLM** |
|---|---|---|---|
| 启动 | 写50行chain | 配4个YAML | **3行代码** |
| 质量保障 | 自己写 | 手动配 | **自动审查+重写闭环** |
| 多模型 | 单模型 | 单模型 | **快模型写+强模型查** |
| 成本感知 | ❌ | ❌ | **实时Token+费用显示** |
| 降级容错 | 自己写 | ❌ | **429自动切后端** |

**一句话定位**：LangChain=积木，CrewAI=流水线，LoomLLM=全自动工作站。

---

## 核心特性

### 1. 零配置启动

设**一个**环境变量，LoomLLM自动发现所有可用模型：

```python
# 随便设一个Provider的Key就行
# export GEMINI_API_KEY=your-key     # 或 DEEPSEEK_API_KEY, OPENAI_API_KEY 等
staff = AIStaff.from_env()
# → 自动发现: gemini-2.5-flash-lite(免费), deepseek-chat(便宜)...
```

`SmartInit`自动扫描环境变量、测试连通性、列出可用模型、配置最优后端——全自动。

### 2. 迭代闭环（V5 CollabLoop）

这是LoomLLM的核心创新。不再是单次调用听天由命：

```
Writer(快/便宜模型) → Reviewer(强模型) → 分数<80? → 带反馈重写 → 再审 → ...
```

**如何省~50%成本：**
- **Writer**用快速便宜的模型（如`gemini-2.5-flash-lite`、`deepseek-chat`）
- **Reviewer**用更强的模型（如`gemini-3-flash`、`deepseek-reasoner`）
- 只有审查步骤需要贵模型——起草+重写都用便宜的

**结构化反馈，不是"请改进"这种废话：**
```json
{
  "score": 72,
  "issues": ["缺少空数组边界情况", "没有复杂度分析"],
  "suggestions": ["添加 len(arr) <= 1 的基础情况", "标注时间复杂度"],
  "strengths": ["递归结构清晰", "变量命名规范"]
}
```

**辩论协议**：Writer可以对Reviewer的批评进行辩解。如果论点有效，分数上调。避免因风格偏好过度迭代。

**自动终止**：循环在以下条件之一满足时停止：
- 质量评分 ≥ 阈值（默认80/100）
- 达到最大迭代次数（默认3次）
- 两轮之间没有实质性改进

### 3. 智能路由

你不需要手动选模式——LoomLLM自动分类你的输入：

```python
staff.chat("1+1等于几")            # → 直接回答：单次调用，无开销
staff.chat("写个快排")              # → 代码模式：编码+审查闭环
staff.chat("AI趋势分析")            # → 研究模式：多轮追问深挖
staff.chat("React vs Vue")         # → 决策模式：多维度对比分析
staff.chat("写个slogan")            # → 创意模式：创意+审查
```

**TaskClassifier**用LLM自身对输入分类，然后路由到最优管线：
- **direct**：简单问答 → 1次API调用，无额外开销
- **code**：编程 → Coder写，Critic审，闭环迭代到质量达标
- **research**：开放式研究 → 多轮追问，逐步深入
- **decision**：对比分析 → 多视角，结构化优缺点
- **creative**：创意写作 → Writer+Reviewer闭环，创意导向提示词

### 4. 10个Provider，统一API

| Provider | 国内直连 | 免费额度 | 需代理 |
|----------|---------|---------|--------|
| DeepSeek | ✅ | ❌ | ❌ |
| 智谱GLM | ✅ | ✅ glm-4-flash | ❌ |
| 硅基流动 | ✅ | ✅ Qwen2.5-7B | ❌ |
| Moonshot(Kimi) | ✅ | ❌ | ❌ |
| 通义千问 | ✅ | ❌ | ❌ |
| Gemini | ❌ | ✅ flash-lite | ✅ |
| OpenAI | ❌ | ❌ | ✅ |
| Groq | ❌ | ✅ | ✅ |
| Anthropic | ❌ | ❌ | ✅ |
| Ollama | 本地 | ✅ | ❌ |

**所有Provider使用OpenAI兼容API格式**（`/v1/chat/completions`）。无厂商锁定，无专属API。只要Provider提供OpenAI兼容端点，就能接入LoomLLM。

**Provider特性：**
- **智能代理检测**：自动扫描本地代理端口（Clash/V2Ray）
- **needs_proxy标记**：国产Provider（DeepSeek、智谱等）永远不传代理——避免延迟
- **模型列表发现**：Gemini支持API实时查询模型列表；其他Provider使用精选模型列表
- **Tier分级**：free / cheap / standard / premium——用于成本感知路由

### 5. 级联降级

当Provider失败（429额度、503过载、网络超时），LoomLLM自动降级：

```
gemini-2.5-flash-lite (429) → gemini-3-flash-preview (503) → deepseek-chat (200 OK ✓)
```

**降级策略：**
1. 同Provider不同模型（Tier降级）
2. 不同Provider相似模型（跨Provider）
3. 全部耗尽则停止（报告哪些Provider失败了）

### 6. Token预算与成本追踪

每次调用实时显示成本：

```
💰 Budget  gemini-2.5-flash-lite | 1,234 tok | free  ←  累计: 5,678 tokens (3 calls) | 2,345 tok/s | free | 2.4s
```

**功能：**
- 逐次Token统计（prompt + completion）
- 基于模型定价的费用估算
- 每日预算限制+预警阈值
- 会话累计汇总

### 7. 彩色过程日志

实时看到每一步在干什么：

```
🟢 Writer      开始 #1
🔴 Reviewer    评分: 62/100 | 问题: 3 | 建议: 2
🟡 Writer      带反馈重写 #2...
🔵 Reviewer    评分: 84/100 | 1.1s | 2,890字
✅ 完成        PASSED (score=84 >= 80) | 2轮 | 5,135 tokens
```

### 8. 自定义专家

编辑`experts/experts.yaml`——无需改代码：

```yaml
- id: code_reviewer
  name: 代码审查专家
  system_prompt: "你是一名高级软件工程师，审查代码的正确性、性能和可读性。"
  temperature: 0.3
  require_review: false    # 跳过审查闭环
```

**内置专家**：generalist、coder、critic、researcher、creative_writer、analyst、reviewer

### 9. 持久化记忆

SQLite支持的对话历史，跨会话保持：

```python
staff = AIStaff.from_env(session_id="project-x")
staff.chat("我在做一个Flask应用")      # 下次会话还记得
staff.chat("给应用加认证")             # 知道上文的上下文
```

---

## 快速开始

### 安装
```bash
pip install httpx pyyaml
```

### 运行
```python
from ai_staff_v4 import AIStaff

# 方式1: 环境变量（推荐）
# 设置任一Provider的Key: GEMINI_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY 等
staff = AIStaff.from_env()

# 方式2: 直接传Key
staff = AIStaff.quick_start("your-api-key", provider="gemini")  # provider可选: deepseek/openai/...

# 方式3: YAML配置（多后端）
staff = AIStaff.from_config_file("config.yaml")

# 开聊
answer = staff.chat("你好")
```

### 指定模式
```python
staff.chat("写个快排", mode="code")           # 代码+审查
staff.chat("AI趋势分析", mode="research")      # 多轮研究
staff.chat("React vs Vue", mode="decision")    # 多维决策
staff.chat("写个slogan", mode="creative")      # 创意+审查
```

### 查看完整结果
```python
result = staff.chat("写个快排", mode="code", return_details=True)
print(f"质量评分: {result.quality_score}/100")
print(f"迭代次数: {result.rounds_used}")
print(f"Token消耗: {result.total_tokens:,}")
print(f"参与专家: {result.experts_used}")
```

---

## 架构

```
用户输入 → chat()
  │
  ├─ TaskClassifier 自动分类
  │   ├─ direct(简单)  → 单次调用
  │   ├─ code(代码)    → Coder + Critic闭环
  │   ├─ research(研究)→ 多轮追问
  │   └─ complex(复杂) → V5完整闭环
  │
  └─ V5 CollabLoop
      ├─ Writer(快模型) 起草
      ├─ Reviewer(强模型) 评分+反馈
      ├─ 分数 < 80? → Writer带反馈重写
      ├─ 辩论协议: Writer可以辩解
      └─ 自动终止: 质量达标或超时
```

---

## 配置

### 单环境变量（最简）
```bash
# 任选一个:
export GEMINI_API_KEY=your-key        # Gemini (免费flash-lite)
export DEEPSEEK_API_KEY=your-key      # DeepSeek (性价比最高)
export ZHIPU_API_KEY=your-key         # 智谱GLM (glm-4-flash免费)
```

### config.yaml（多后端）
复制`config_template.yaml`填入你的Key。10个Provider预配置好，取消注释即可。

---

## 与同类框架对比

| 特性 | LoomLLM | LangChain | CrewAI | AutoGen |
|------|---------|-----------|--------|---------|
| 首次调用代码行数 | **3行** | 50+ | 30+ | 40+ |
| 迭代质量闭环 | **内置** | 手动 | 手动 | 手动 |
| 多Provider | **10个内置** | 手动 | 手动 | 手动 |
| 零配置启动 | **支持** | ❌ | ❌ | ❌ |
| 429自动降级 | **支持** | ❌ | ❌ | ❌ |
| Token成本追踪 | **实时** | ❌ | ❌ | ❌ |
| 任务自动分类 | **支持** | ❌ | ❌ | ❌ |
| 纯OpenAI兼容 | **是** | 多格式 | 多格式 | 多格式 |

---

## 测试

```bash
# 快速验证
python -c "from ai_staff_v4 import AIStaff; print('OK')"

# 跑示例
python examples/simple.py
python examples/research_flow.py
python examples/expert_task.py
```

## 依赖

- Python 3.10+
- httpx >= 0.27
- pyyaml >= 6.0

---

## 许可证

[MIT](LICENSE)
