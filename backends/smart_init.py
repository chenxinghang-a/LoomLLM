"""
Smart Init V2 — 全Provider自动扫描 + 多Key来源 + AI决策路由

核心改变（对比V1）：
  V1: 只扫Gemini，单key，硬编码选模型
  V2: 扫所有provider，多key来源(env/config/明文)，为AI决策路由提供数据

工作流：
  1. KeyDiscovery — 从env/config/明文收集所有API key
  2. ProviderScan — 对每个provider探测可用模型
  3. ModelRegistry — 汇总所有可用模型+能力标签
  4. 缓存 — 结果写JSON，1h有效

产出的 ModelRegistry 是 AIRouter 的数据源：
  AIRouter 用最强可用模型分析任务 → 从 ModelRegistry 选执行模型
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import httpx

from ..core.constants import PACKAGE_ROOT


# ═══════════════════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════════════════

@dataclass
class ModelInfo:
    """单个模型探测结果"""
    name: str               # e.g. "gemini-2.5-flash-lite"
    provider: str           # e.g. "gemini", "openai", "deepseek", "ollama"
    base_url: str           # API endpoint
    status: int             # HTTP status (200=OK, 429=quota, 0=timeout)
    latency_ms: float       # 响应延迟
    tier: str               # free/cheap/standard/premium/local
    input_cost: float       # per 1K tokens
    output_cost: float      # per 1K tokens
    capabilities: list[str] = field(default_factory=list)  # 能力标签
    context_window: int = 0  # 上下文窗口大小
    key_source: str = ""    # key来源: env/config/plaintext

    @property
    def is_usable(self) -> bool:
        return self.status == 200

    @property
    def is_local(self) -> bool:
        return self.provider == "ollama" or "localhost" in self.base_url

    @property
    def is_free(self) -> bool:
        return self.tier in ("free", "local") or (self.input_cost == 0 and self.output_cost == 0)

    @property
    def strength_score(self) -> float:
        """综合能力评分（AIRouter用）: 越高越强"""
        score = 0.0
        # tier 权重
        tier_weights = {"local": 2, "free": 3, "cheap": 5, "standard": 7, "premium": 9}
        score += tier_weights.get(self.tier, 4)
        # 能力标签加分
        cap_bonus = {"reasoning": 3, "code": 2, "creative": 1, "vision": 2, "long_context": 1}
        for cap in self.capabilities:
            score += cap_bonus.get(cap, 0)
        # 上下文窗口加分
        if self.context_window >= 100000:
            score += 2
        elif self.context_window >= 30000:
            score += 1
        # 延迟惩罚
        if self.latency_ms > 5000:
            score -= 1
        if self.latency_ms > 10000:
            score -= 1
        return score


@dataclass
class ProviderScanResult:
    """单个provider的扫描结果"""
    provider: str
    api_key: str
    base_url: str
    proxy: str
    models: list[ModelInfo]
    best_model: str
    best_tier: str
    error: str = ""


@dataclass
class ModelRegistry:
    """
    全provider模型注册表 — SmartInit的最终产出
    
    这是 AIRouter 的数据源：包含所有已发现的可用模型及其能力标签
    """
    providers: dict[str, ProviderScanResult]   # provider → scan result
    all_models: list[ModelInfo]                # 扁平化模型列表
    best_overall: str                          # 全局最优模型
    best_free: str                             # 最优免费模型
    proxy: str                                 # 全局代理
    scanned_at: float = 0
    scan_duration_s: float = 0

    @property
    def usable_models(self) -> list[ModelInfo]:
        return [m for m in self.all_models if m.is_usable]

    @property
    def free_models(self) -> list[ModelInfo]:
        return [m for m in self.usable_models if m.is_free]

    @property
    def premium_models(self) -> list[ModelInfo]:
        return [m for m in self.usable_models if m.tier in ("premium", "standard")]

    def get_model(self, name: str) -> Optional[ModelInfo]:
        for m in self.all_models:
            if m.name == name and m.is_usable:
                return m
        return None

    def get_strongest(self) -> Optional[ModelInfo]:
        """返回能力最强的可用模型（用于AI决策路由的决策模型）"""
        usable = self.usable_models
        if not usable:
            return None
        return max(usable, key=lambda m: m.strength_score)

    def to_profiles_dict(self) -> dict:
        """转成 AIStaff(profiles=...) 需要的格式"""
        from .profile import BackendProfile
        profiles = {}
        for m in self.usable_models:
            # key格式: gemini_25_flash_lite (不加provider前缀，和旧版兼容)
            key = m.name.replace('-', '_').replace('.', '')
            # 去重：同名加provider前缀
            if key in profiles:
                key = f"{m.provider}_{key}"
            profiles[key] = {
                "name": key,
                "base_url": m.base_url,
                "api_key": next(
                    (p.api_key for p in self.providers.values() if p.provider == m.provider),
                    ""
                ),
                "model": m.name,
                "proxy": self.proxy,
                "tier": m.tier,
                "priority": 10 if m.name == self.best_overall else 5,
                "input_cost_per_1k": m.input_cost,
                "output_cost_per_1k": m.output_cost,
            }
        return profiles


# ═══════════════════════════════════════════════════════════
#  Provider 定义
# ═══════════════════════════════════════════════════════════

PROVIDER_DEFS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "list_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "env_keys": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "AI_STAFF_API_KEY"],
        "needs_proxy": True,    # 中国需要代理
        "can_list_models": True, # 支持 listModels API
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "list_url": "",
        "env_keys": ["OPENAI_API_KEY", "AI_STAFF_OPENAI_KEY"],
        "needs_proxy": True,
        "can_list_models": False,
        "known_models": [
            ("gpt-4o-mini", "cheap", 0.00015, 0.0006, ["reasoning", "code", "creative"]),
            ("gpt-4o", "premium", 0.0025, 0.01, ["reasoning", "code", "creative", "vision"]),
            ("gpt-3.5-turbo", "cheap", 0.0005, 0.0015, ["code"]),
        ],
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "list_url": "",
        "env_keys": ["DEEPSEEK_API_KEY", "AI_STAFF_DEEPSEEK_KEY"],
        "needs_proxy": False,
        "can_list_models": False,
        "known_models": [
            ("deepseek-chat", "cheap", 0.00014, 0.00028, ["reasoning", "code", "creative"]),
            ("deepseek-reasoner", "standard", 0.00055, 0.00219, ["reasoning", "code"]),
        ],
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "list_url": "http://localhost:11434/api/tags",
        "env_keys": [],
        "needs_proxy": False,
        "can_list_models": True,
        "local": True,
        "known_models": [
            ("qwen2.5:7b", "local", 0, 0, ["reasoning", "code"]),
            ("llama3:8b", "local", 0, 0, ["reasoning", "creative"]),
            ("codellama:7b", "local", 0, 0, ["code"]),
        ],
    },
}

# Gemini 模型能力标签
GEMINI_CAPABILITIES = {
    "gemini-2.5-flash-lite": ["reasoning", "code", "creative"],
    "gemini-2.5-flash": ["reasoning", "code", "creative", "vision", "long_context"],
    "gemini-2.5-pro": ["reasoning", "code", "creative", "vision", "long_context"],
    "gemini-3-flash-preview": ["reasoning", "code", "creative", "vision"],
    "gemini-3-pro-preview": ["reasoning", "code", "creative", "vision", "long_context"],
    "gemini-3.1-flash-lite-preview": ["reasoning", "code", "creative"],
}

# Gemini 模型 tier 映射
GEMINI_TIER_MAP = {
    "gemini-2.5-flash-lite": ("free", 0, 0),
    "gemini-2.5-flash": ("standard", 0.000075, 0.0003),
    "gemini-2.5-pro": ("premium", 0.00125, 0.005),
    "gemini-3-flash-preview": ("free", 0, 0),
    "gemini-3-pro-preview": ("standard", 0.0001, 0.0004),
    "gemini-3.1-flash-lite-preview": ("free", 0, 0),
    "gemini-2.0-flash": ("free", 0, 0),
    "gemini-2.0-flash-lite": ("free", 0, 0),
}

# 代理端口
PROXY_PORTS = [7890, 7891, 1080, 1081, 8080, 10808, 10809]

# tier 优先级
TIER_PRIORITY = {"local": -1, "free": 0, "cheap": 1, "standard": 2, "premium": 3}

# 缓存
CACHE_DIR = PACKAGE_ROOT / ".cache"
CACHE_FILE = CACHE_DIR / "smart_init_v2_cache.json"
CACHE_TTL = 3600


# ═══════════════════════════════════════════════════════════
#  KeyDiscovery — 多来源key收集
# ═══════════════════════════════════════════════════════════

class KeyDiscovery:
    """从多种来源收集API key"""

    @staticmethod
    def discover_all(extra_keys: dict[str, str] = None) -> dict[str, str]:
        """
        返回 {provider: api_key} 字典
        
        来源优先级:
          1. extra_keys（调用方显式传入）
          2. 环境变量（GEMINI_API_KEY等）
          3. config.yaml 文件
          4. ~/.ai-staff/keys.json（加密key存储，预留）
        """
        found: dict[str, str] = {}

        # 1. 显式传入
        if extra_keys:
            for provider, key in extra_keys.items():
                if key:
                    found[provider] = key

        # 2. 环境变量
        for provider, pdef in PROVIDER_DEFS.items():
            if provider in found:
                continue
            for env_key in pdef.get("env_keys", []):
                val = os.environ.get(env_key, "").strip()
                if val:
                    found[provider] = val
                    break

        # 3. config.yaml
        config_path = os.environ.get("AI_STAFF_CONFIG", "")
        if config_path and os.path.isfile(config_path):
            try:
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                for pname, pdata in cfg.get("profiles", {}).items():
                    # 从 profile 名推断 provider
                    provider = pdata.get("provider", "")
                    key = pdata.get("api_key", "")
                    # 展开 ${ENV_VAR}
                    if key.startswith("${") and key.endswith("}"):
                        env_name = key[2:-1]
                        key = os.environ.get(env_name, key)
                    if provider and key and provider not in found:
                        found[provider] = key
            except Exception:
                pass

        # 4. ~/.ai-staff/keys.json（预留加密存储）
        keys_file = Path.home() / ".ai-staff" / "keys.json"
        if keys_file.is_file():
            try:
                with open(keys_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for provider, key in data.items():
                    if provider not in found and key:
                        found[provider] = key
            except Exception:
                pass

        return found


# ═══════════════════════════════════════════════════════════
#  SmartInit V2 — 全Provider扫描引擎
# ═══════════════════════════════════════════════════════════

class SmartInit:
    """
    V2: 全Provider自动扫描 + 多Key来源 + AI决策数据源
    
    产出 ModelRegistry，供 AIRouter 做智能路由决策
    """

    @staticmethod
    def auto_configure(extra_keys: dict[str, str] = None,
                       proxy_hint: str = "",
                       force_rescan: bool = False) -> ModelRegistry:
        """
        主入口：扫描所有provider，构建模型注册表
        
        Args:
            extra_keys: 显式传入的 {provider: api_key}
            proxy_hint: 手动指定代理
            force_rescan: 忽略缓存
        """
        start = time.time()

        # 1. 读缓存
        if not force_rescan:
            cached = SmartInit._load_cache()
            if cached:
                age = time.time() - cached.scanned_at
                print(f"  [SmartInit] Cache hit (age={age:.0f}s, "
                      f"{len(cached.usable_models)} usable models)")
                return cached

        # 2. 发现所有key
        keys = KeyDiscovery.discover_all(extra_keys)
        print(f"  [SmartInit] Discovered keys for: {list(keys.keys())}")

        # 3. 探测代理
        proxy = proxy_hint or SmartInit._detect_proxy()
        if proxy:
            print(f"  [SmartInit] Proxy: {proxy}")

        # 4. 扫描每个provider
        provider_results: dict[str, ProviderScanResult] = {}
        all_models: list[ModelInfo] = []

        for provider, api_key in keys.items():
            print(f"  [SmartInit] Scanning {provider}...")
            result = SmartInit._scan_provider(provider, api_key, proxy)
            provider_results[provider] = result
            all_models.extend(result.models)
            if result.models:
                usable = [m for m in result.models if m.is_usable]
                print(f"    {provider}: {len(usable)}/{len(result.models)} usable")

        # 5. 扫描本地Ollama（即使没有key也要试）
        if "ollama" not in keys:
            ollama_result = SmartInit._scan_ollama(proxy)
            if ollama_result and ollama_result.models:
                provider_results["ollama"] = ollama_result
                all_models.extend(ollama_result.models)

        # 6. 选全局最优
        usable = [m for m in all_models if m.is_usable]
        best_overall = SmartInit._pick_best_overall(usable)
        best_free = SmartInit._pick_best_free(usable)

        registry = ModelRegistry(
            providers=provider_results,
            all_models=all_models,
            best_overall=best_overall,
            best_free=best_free,
            proxy=proxy,
            scanned_at=time.time(),
            scan_duration_s=0,
        )
        registry.scan_duration_s = round(time.time() - start, 1)

        # 7. 缓存
        SmartInit._save_cache(registry)

        print(f"  [SmartInit] Done in {registry.scan_duration_s}s: "
              f"{len(registry.usable_models)} usable, "
              f"best={best_overall}, best_free={best_free}")
        return registry

    # ── 代理探测 ──

    @staticmethod
    def _detect_proxy() -> str:
        for env in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "AI_STAFF_PROXY"):
            val = os.environ.get(env, "").strip()
            if val:
                return val
        for port in PROXY_PORTS:
            try:
                with httpx.Client(timeout=2) as c:
                    c.get(f"http://127.0.0.1:{port}", follow_redirects=True)
                    return f"http://127.0.0.1:{port}"
            except Exception:
                continue
        return ""

    # ── Provider扫描 ──

    @staticmethod
    def _scan_provider(provider: str, api_key: str, proxy: str) -> ProviderScanResult:
        """扫描单个provider"""
        pdef = PROVIDER_DEFS.get(provider)
        if not pdef:
            return ProviderScanResult(
                provider=provider, api_key=api_key,
                base_url="", proxy=proxy, models=[],
                best_model="", best_tier="free",
                error=f"Unknown provider: {provider}"
            )

        base_url = pdef["base_url"]
        models: list[ModelInfo] = []

        if provider == "gemini":
            models = SmartInit._scan_gemini_models(api_key, proxy)
        elif provider == "ollama":
            models = SmartInit._scan_ollama_models(proxy)
        else:
            # OpenAI/DeepSeek 等：用已知模型列表快速探测
            models = SmartInit._scan_known_models(
                api_key, base_url, provider, proxy,
                pdef.get("known_models", [])
            )

        usable = [m for m in models if m.is_usable]
        best = SmartInit._pick_best_for_provider(usable) if usable else None

        return ProviderScanResult(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            proxy=proxy,
            models=models,
            best_model=best.name if best else "",
            best_tier=best.tier if best else "free",
        )

    @staticmethod
    def _scan_gemini_models(api_key: str, proxy: str) -> list[ModelInfo]:
        """Gemini专用扫描：listModels API + 逐个测试"""
        all_names = SmartInit._list_gemini_models(api_key, proxy)
        if not all_names:
            all_names = list(GEMINI_TIER_MAP.keys())

        # 过滤+优先级
        skip = {"deep-research", "lyria", "tts", "image", "computer-use", "nano-banana"}
        priority = ["gemini-2.5-flash", "gemini-3-flash", "gemini-3.1-flash"]
        
        prio_list = []
        other_list = []
        for n in all_names:
            if any(s in n.lower() for s in skip):
                continue
            if any(n.startswith(p) for p in priority) or n.startswith("gemini-"):
                prio_list.append(n)
            else:
                other_list.append(n)

        models: list[ModelInfo] = []
        best_found = False

        for name in prio_list:
            info = SmartInit._test_gemini_model(api_key, name, proxy)
            models.append(info)
            if info.is_usable and info.is_free and "flash" in name:
                best_found = True
                break

        if not best_found:
            for name in other_list:
                info = SmartInit._test_gemini_model(api_key, name, proxy)
                models.append(info)
                if info.is_usable:
                    break

        return models

    @staticmethod
    def _list_gemini_models(api_key: str, proxy: str) -> list[str]:
        kwargs: dict = {"timeout": 10, "follow_redirects": True}
        if proxy:
            kwargs["proxy"] = proxy
        try:
            with httpx.Client(**kwargs) as c:
                r = c.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}&pageSize=100",
                    headers={"Content-Type": "application/json"}
                )
                if r.status_code != 200:
                    return []
                data = r.json()
                names = []
                for m in data.get("models", []):
                    mid = m.get("name", "")
                    methods = m.get("supportedGenerationMethods", [])
                    if "generateContent" in methods:
                        names.append(mid.replace("models/", ""))
                return names
        except Exception:
            return []

    @staticmethod
    def _test_gemini_model(api_key: str, model: str, proxy: str) -> ModelInfo:
        kwargs: dict = {"timeout": 15, "follow_redirects": True}
        if proxy:
            kwargs["proxy"] = proxy

        tier, inp, out = GEMINI_TIER_MAP.get(model, ("standard", 0.0001, 0.0004))
        caps = GEMINI_CAPABILITIES.get(model, ["reasoning"])

        try:
            with httpx.Client(**kwargs) as c:
                t0 = time.time()
                r = c.post(
                    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5},
                )
                latency = (time.time() - t0) * 1000
                return ModelInfo(name=model, provider="gemini",
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                    status=r.status_code, latency_ms=round(latency),
                    tier=tier, input_cost=inp, output_cost=out,
                    capabilities=caps, key_source="auto")
        except Exception:
            return ModelInfo(name=model, provider="gemini",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                status=0, latency_ms=0, tier=tier, input_cost=inp, output_cost=out,
                capabilities=caps)

    @staticmethod
    def _scan_known_models(api_key: str, base_url: str, provider: str,
                           proxy: str, known: list[tuple]) -> list[ModelInfo]:
        """用已知模型列表快速探测（OpenAI/DeepSeek等）"""
        models = []
        for item in known:
            name, tier, inp, out, caps = item
            kwargs: dict = {"timeout": 15, "follow_redirects": True}
            if proxy:
                kwargs["proxy"] = proxy
            try:
                with httpx.Client(**kwargs) as c:
                    t0 = time.time()
                    r = c.post(
                        f"{base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": name, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5},
                    )
                    latency = (time.time() - t0) * 1000
                    models.append(ModelInfo(name=name, provider=provider,
                        base_url=base_url, status=r.status_code,
                        latency_ms=round(latency), tier=tier,
                        input_cost=inp, output_cost=out, capabilities=caps))
            except Exception:
                models.append(ModelInfo(name=name, provider=provider,
                    base_url=base_url, status=0, latency_ms=0,
                    tier=tier, input_cost=inp, output_cost=out, capabilities=caps))
        return models

    @staticmethod
    def _scan_ollama(proxy: str) -> Optional[ProviderScanResult]:
        """探测本地Ollama"""
        models = SmartInit._scan_ollama_models(proxy)
        if not models:
            return None
        usable = [m for m in models if m.is_usable]
        best = usable[0] if usable else None
        return ProviderScanResult(
            provider="ollama", api_key="ollama",
            base_url="http://localhost:11434/v1", proxy=proxy,
            models=models, best_model=best.name if best else "",
            best_tier=best.tier if best else "local")

    @staticmethod
    def _scan_ollama_models(proxy: str) -> list[ModelInfo]:
        """扫描Ollama本地模型"""
        try:
            with httpx.Client(timeout=3) as c:
                r = c.get("http://localhost:11434/api/tags")
                if r.status_code != 200:
                    return []
                data = r.json()
                models = []
                for m in data.get("models", []):
                    name = m.get("name", "")
                    size = m.get("size", 0)
                    # 推测能力
                    caps = ["reasoning"]
                    if "code" in name.lower():
                        caps.append("code")
                    if "llama" in name.lower():
                        caps.extend(["reasoning", "creative"])
                    models.append(ModelInfo(
                        name=name, provider="ollama",
                        base_url="http://localhost:11434/v1",
                        status=200, latency_ms=10, tier="local",
                        input_cost=0, output_cost=0, capabilities=caps,
                        context_window=8000))
                return models
        except Exception:
            return []

    # ── 选模型 ──

    @staticmethod
    def _pick_best_overall(usable: list[ModelInfo]) -> str:
        if not usable:
            return "gemini-2.5-flash-lite"
        # 优先选稳定可靠的（flash-lite > flash > pro），同层级选strength最高的
        # 排序：tier优先级(低=便宜好) → 名字偏好(flash-lite最稳) → strength
        def sort_key(m: ModelInfo):
            tier_prio = {"local": -1, "free": 0, "cheap": 1, "standard": 2, "premium": 3}.get(m.tier, 99)
            name_bonus = 0
            if "flash-lite" in m.name:
                name_bonus = -2  # 最稳定
            elif "flash" in m.name and "pro" not in m.name:
                name_bonus = -1  # 也不错
            return (tier_prio, name_bonus, -m.strength_score)
        usable.sort(key=sort_key)
        return usable[0].name

    @staticmethod
    def _pick_best_free(usable: list[ModelInfo]) -> str:
        free = [m for m in usable if m.is_free]
        if not free:
            return usable[0].name if usable else "gemini-2.5-flash-lite"
        best = max(free, key=lambda m: m.strength_score)
        return best.name

    @staticmethod
    def _pick_best_for_provider(usable: list[ModelInfo]) -> Optional[ModelInfo]:
        if not usable:
            return None
        return max(usable, key=lambda m: m.strength_score)

    # ── 缓存 ──

    @staticmethod
    def _load_cache() -> Optional[ModelRegistry]:
        if not CACHE_FILE.is_file():
            return None
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if time.time() - data.get("scanned_at", 0) > CACHE_TTL:
                return None

            # 反序列化
            providers = {}
            for pname, pdata in data.get("providers", {}).items():
                models = [ModelInfo(**m) for m in pdata.get("models", [])]
                providers[pname] = ProviderScanResult(
                    provider=pdata["provider"], api_key=pdata["api_key"],
                    base_url=pdata["base_url"], proxy=pdata["proxy"],
                    models=models, best_model=pdata["best_model"],
                    best_tier=pdata["best_tier"], error=pdata.get("error", ""))

            all_models = [ModelInfo(**m) for m in data.get("all_models", [])]

            return ModelRegistry(
                providers=providers, all_models=all_models,
                best_overall=data["best_overall"],
                best_free=data["best_free"],
                proxy=data["proxy"],
                scanned_at=data["scanned_at"],
                scan_duration_s=data.get("scan_duration_s", 0),
            )
        except Exception:
            return None

    @staticmethod
    def _save_cache(registry: ModelRegistry):
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "providers": {
                    pname: {
                        "provider": p.provider, "api_key": p.api_key,  # 保留完整key
                        "base_url": p.base_url, "proxy": p.proxy,
                        "models": [asdict(m) for m in p.models],
                        "best_model": p.best_model, "best_tier": p.best_tier,
                        "error": p.error,
                    } for pname, p in registry.providers.items()
                },
                "all_models": [asdict(m) for m in registry.all_models],
                "best_overall": registry.best_overall,
                "best_free": registry.best_free,
                "proxy": registry.proxy,
                "scanned_at": registry.scanned_at,
                "scan_duration_s": registry.scan_duration_s,
            }
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  [SmartInit] Cache write failed: {e}")


__all__ = ['SmartInit', 'ModelRegistry', 'ProviderScanResult', 'ModelInfo', 'KeyDiscovery']
