# AI-Staff V4 使用手册 & 踩坑指南
> 2026-04-24 实测总结 | 主人: cxx | 环境: Windows + Python 3.12 + Clash代理

---

## 一、快速开始（别再踩的坑）

### 1. 你只需要做一件事

编辑 **一个文件**：
```
ai_staff_v4/config.yaml    ← 把这个从 template 复制过来，填好就能跑
```

**必须填的3个字段：**
```yaml
settings:
  proxy: "http://127.0.0.1:7890"   # ← 必填！国内不挂代理连不上Gemini

profiles:
  gemini_main:
    api_key: "AIzaSy..."            # ← 必填！你的Key
    model: "gemini-2.5-flash-lite"  # ← 用下面验证过的模型名，别瞎填
```

### 2. 别用这些模型名（2026-04-24实测已死）

| 写法 | 状态 | 报错 |
|------|------|------|
| `gemini-1.5-flash` | ❌ 下线 | 404 not found |
| `gemini-1.5-pro` | ❌ 下线 | 404 |
| `gemini-2.0-flash` | ❌ 免费额度归零 | 429 quota exceeded |
| `gemini-2.0-flash-lite` | ❌ 同上 | 429 |
| `gemini-2.0-flash-exp` | ❌ 不存在 | 404 |
| `gemini-3-pro-preview` | ❌ 额度归零 | 429 |
| `gemini-3.1-pro-preview` | ❌ 额度归零 | 429 |
| `gemini-pro-latest` | ❌ 额度归零 | 429 |

> **教训**：网上教程/旧代码里的模型名大部分过期了。Google每几个月就改一次命名。

### 3. 用这些（实测200 OK）

| 模型名 | 特点 | 推荐场景 |
|--------|------|----------|
| `gemini-2.5-flash-lite` | ✅ 稳定、便宜、中文好 | **日常主力** |
| `gemini-3-flash-preview` | ✅ 能用但输出短 | 备选/简单任务 |
| `gemini-2.5-flash` | ⚠️ 偶发503高负载 | 高峰期可能不可用 |
| `gemini-3.1-flash-lite-preview` | ⚠️ 偶发503 | 输出质量最好但不太稳 |

---

## 二、Windows环境专用坑

### 坑1：PowerShell吃掉Python输出

**现象**：脚本明明print了，但工具返回空字符串。

**原因**：PowerShell的stdout捕获机制和Python stderr混在一起，Unicode字符直接炸。

**解决**：
```python
# ❌ 别这么调试
print(f"结果: {data}")  # 中文可能乱码或吞掉

# ✅ 写文件再看
with open("result.txt", "w", encoding="utf-8") as f:
    f.write(f"结果: {data}")
```

### 坑2：GBK编码爆炸

**现象**：
```
UnicodeEncodeError: 'gbk' codec can't encode character '\u274c'
```

**原因**：Windows默认GBK，emoji/特殊符号必炸。

**解决**：
```python
# ❌ 
print("✅ 成功")   # 炸
print("❌ 失败")   # 炸

# ✅ 用ASCII符号
print("[OK] 成功")
print("[FAIL] 失败")
```

### 坑3：引号地狱

**现象**：PowerShell里写内联Python脚本，引号转义让你怀疑人生。

**解决**：
```powershell
# ❌ 别在命令行里写长Python
python -c "print(f\"{data['error']['message']}\")"  # 引号地狱

# ✅ 写成.py文件再执行
# 先 write_to_file 写 .py 文件
# 再 execute_command 调 python xxx.py
```

---

## 三、网络问题（国内用户必看）

### 坑4：必须代理

```
generativelanguage.googleapis.com  →  国内直连 = 超时
```

**三种配法（按优先级）**：

```yaml
# 方法1：config.yaml全局设置（推荐）
settings:
  proxy: "http://127.0.0.1:7890"

# 方法2：代码里传参
client = LLMClient(..., proxy="http://127.0.0.1:7890")

# 方法3：环境变量（未实现，待加）
```

**注意**：Clash要开着！关了代理=所有请求超时。

### 坑5：429不是Key无效

```
400 INVALID_API_KEY     → Key错了，换一个
429 quota exceeded      → Key对，但额度用完了，等明天或付费
503 UNAVAILABLE         → Key对，服务器爆了，等几秒重试
404 not found           → 模型名写错了，看上面的可用列表
```

**判断流程**：
```
先试 gemini-2.5-flash-lite
├─ 200 OK → 完美，用它
├─ 429 → 换 gemini-3-flash-preview 试
│   ├─ 200 → 用备选
│   └─ 429 → 所有免费额度耗尽，需付费
├─ 503 → 等10秒再试（高峰期）
└─ 400 → Key错误，检查有没有复制完整
```

---

## 四、LLMClient使用细节

### 坑6：OpenAI兼容 vs 原生API

本项目用的是 **OpenAI兼容格式**（`/v1beta/openai/chat/completions`），不是Gemini原生格式。

```python
# ✅ 正确用法（本项目）
client = LLMClient(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    api_key="AIzaSy...",
    model="gemini-2.5-flash-lite",
    proxy="http://127.0.0.1:7890"
)
content, usage = client.chat_completion(
    messages=[{"role": "user", "content": "你好"}]
)
# 返回: ("回复文本", {"prompt_tokens": x, "completion_tokens": y})

# ❌ 如果你用原生API格式会怎样？
# base_url="/v1beta/" → 需要自己拼generateContent，结构完全不同
```

### 坑7：返回值是tuple不是字符串

```python
content, usage = client.chat_completion(...)
# content = str (回复内容)
# usage = dict (token用量)
# 别忘了解包！
```

### 坑8：429自动重试但有上限

LLMClient内置3次重试，429等待时间递增（3s→6s→12s）。但**如果所有模型都429，重试也没用**。

建议：多配几个不同tier的profile，fallback机制会自动切换。

---

## 五、config.yaml 完整参考（复制即用）

```yaml
default_model: "gemini-2.5-flash-lite"

settings:
  proxy: "http://127.0.0.1:7890"       # ← 改成你的代理地址
  default_expert: "generalist"
  timeout: 120
  max_retries: 3
  language: "zh-CN"

profiles:
  # 主力（稳定便宜）
  main:
    provider: "gemini"
    base_url: "https://generativelanguage.googleapis.com/v1beta/openai"
    api_key: "你的Key贴这里"             # ← 改
    model: "gemini-2.5-flash-lite"
    tier: "standard"
    priority: 10

  # 备选
  backup:
    provider: "gemini"
    base_url: "https://generativelanguage.googleapis.com/v1beta/openai"
    api_key: "你的Key贴这里"             # ← 改
    model: "gemini-3-flash-preview"
    tier: "premium"
    priority: 8

budget:
  daily_limit_usd: 1.0
  enable_tracking: true

self_improve:
  enabled: true

memory:
  db_path: ".ai_staff_memory.db"
```

---

## 六、如何查看有哪些模型可用？

运行这个（不需要装任何东西）：

```python
import httpx
KEY = "你的key"
PROXY = "http://127.0.0.1:7890"
resp = httpx.get(
    f"https://generativelanguage.googleapis.com/v1beta/models?key={KEY}",
    proxy=PROXY, timeout=15
)
models = [m["name"].replace("models/","") for m in resp.json().get("models",[])
          if "generateContent" in m.get("supportedGenerationMethods",[])]
for m in models: print(m)
```

这比网上搜靠谱100倍——**以API返回为准**。

---

## 七、常见问题速查

| 问题 | 原因 | 解决 |
|------|------|------|
| ConnectTimeout | 没开代理 | 开Clash |
| 429 quota exceeded | 免费额度用完 | 等明天/付费/换号 |
| 503 UNAVAILABLE | 服务器过载 | 等10秒重试 |
| 400 INVALID_API_KEY | Key错了 | 检查复制完整性 |
| 404 not found | 模型名错/下线 | 运行上面查模型脚本 |
| UnicodeEncodeError | Windows GBK编码 | 别用emoji，写文件输出 |
| 返回空字符串 | PowerShell吞输出 | 结果写文件再读 |

---

*最后更新: 2026-04-24 | 基于13+38个模型的实际测试结果*
