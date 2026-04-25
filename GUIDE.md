# LoomLLM 使用指南

> 教你选对模式、看懂数据、用好人

---

## 一、模式选择（最重要！）

**核心原则：不是所有任务都需要讨论闭环。** 简单问题直接答，复杂任务才走审查循环。

| 你想干嘛 | 推荐模式 | 说明 |
|---------|---------|------|
| 问个问题、翻译、定义 | `auto` 或 `direct` | 1次调用，秒回，省token |
| 写代码、debug | `code` | 编码→审查，2次调用 |
| 技术趋势、深度分析 | `research` | 多轮追问，4次调用 |
| 选型对比、买哪个 | `decision` | 多视角分析+综合结论 |
| 文案、命名、方案 | `creative` | 创作→审查，2次调用 |
| 复杂多领域任务 | `collab` | 多专家分工+审查 |
| 比较不同模型表现 | `arena` | 同一问题问所有后端 |

**auto模式的行为**：
- 简单问答 → 自动走`direct`快速路径，**不进闭环**，1次调用搞定
- 复杂任务 → 自动走V5闭环（Writer→Reviewer→Judge），迭代直到达标

```python
# 这些走快速路径（1次调用）：
staff.chat("什么是递归")
staff.chat("Python的GIL是什么")
staff.chat("1+1等于几")

# 这些走闭环（2-3次迭代）：
staff.chat("写一个线程安全的单例模式")
staff.chat("分析React vs Vue的技术选型")
staff.chat("用Python实现二分查找，处理边界情况")
```

---

## 二、返回值解读

### 基础用法：纯文本

```python
response = staff.chat("你好")  # 返回 str
print(response)
```

### 进阶用法：获取完整元数据

```python
result = staff.chat("写个快排", return_details=True)
# result 是 CollaborationResult 对象

print(result.deliverables)     # {"solution.py": "..."}
print(result.quality_score)    # 7.5  (0-10)
print(result.total_tokens)     # 3280
print(result.trace_id)         # "a3f1b2c4d5e6"
print(result.rounds_used)      # 2
print(result.total_time_sec)   # 8.3
print(result.strategy_mode)    # "v5_code"
print(result.experts_used)     # ["coder", "critic"]
```

### CollaborationResult 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `goal` | str | 原始用户输入 |
| `status` | str | "success" / "failed" / "max_iterations_reached" |
| `strategy_mode` | str | 使用的策略（如"v5_code", "direct"） |
| `trace_id` | str | 追踪ID，贯穿整个执行过程 |
| `deliverables` | dict | 交付物 {"文件名": "内容"} |
| `quality_score` | float | 质量评分 (0-10) |
| `total_tokens` | int | 消耗的总token数 |
| `rounds_used` | int | 迭代轮数 |
| `total_time_sec` | float | 总耗时(秒) |
| `experts_used` | list | 使用的专家ID列表 |
| `transcript` | str | 完整执行记录 |

---

## 三、REST API 接口

### POST /chat

```json
// 请求
{"prompt": "写个快排", "mode": "auto"}

// 响应（现在含完整元数据！）
{
  "ok": true,
  "response": "def quicksort(arr): ...",
  "mode": "auto",
  "trace_id": "a3f1b2c4d5e6",
  "strategy": "v5_code",
  "quality_score": 8.5,
  "total_tokens": 3280,
  "rounds": 2,
  "duration_ms": 8300
}
```

### POST /run

```json
// 请求
{"prompt": "分析AI趋势", "max_iterations": 3, "quality_threshold": 80}

// 响应
{
  "ok": true,
  "status": "success",
  "trace_id": "b4c5d6e7f8a9",
  "strategy": "v5_research",
  "quality_score": 9.0,
  "total_tokens": 15200,
  "deliverables": ["research_report.md"],
  "iterations": 2,
  "experts_used": ["researcher", "critic"],
  "duration_ms": 15200
}
```

### GET /health | /status | /experts

标准健康检查和状态查询。

---

## 四、V5闭环协作流程

不是盲目讨论，是**质量驱动的迭代**：

```
用户输入 → 分类 → Writer生成初稿
                    ↓
              Reviewer结构化评分
                    ↓
          分数≥80? ──YES→ 返回
              ↓NO
          Writer辩解(Rebuttal)
              ↓
          Reviewer重新评估(Rejudge)
              ↓
          仍不通过 → Writer带反馈修正
              ↓
          再次Review → ...循环
```

**关键设计**：
- 简单问题（direct分类）**不进闭环**，直接返回
- 辩论协议：Writer有权为自己的选择辩护，不是盲目接受批评
- 上调递减：第1轮Rejudge最多+15分，后续最多+5分，防止刷分

---

## 五、启动方式

```python
# 零配置（推荐，自动扫描环境变量和~/.ai-staff/keys.json）
staff = AIStaff.from_env()

# 快速启动
staff = AIStaff.quick_start("your-key", provider="gemini")

# YAML配置（多后端）
staff = AIStaff.from_config_file("config.yaml")
```

---

## 六、省钱技巧

1. **简单问题用auto或direct** — 不走闭环，省2-3倍token
2. **`quality_threshold`适当降低** — 70就够了，80比较严格
3. **`max_iterations`限制迭代** — 设2就行，避免无限循环
4. **多配几个模型** — fallback机制自动切换，不会卡在429

```python
# 省钱参数
staff.chat("简单问题")                              # auto直接走快速路径
staff.chat("复杂任务", quality_threshold=70, max_iterations=2)  # 降低阈值
```

---

*AI-Staff V4 | 更新 2026-04-24*
