"""
AI-Staff V4 启动模块 — 从 staff.py 拆分出来的零配置启动逻辑

包含:
  - PROVIDER_TEMPLATES: Provider预设模板
  - from_env(): 环境变量自动启动
  - quick_start(): 快速启动
  - discover_and_start(): 自动发现后端
  - from_config_file(): YAML配置加载
"""

from __future__ import annotations
import os, re
from typing import Optional


# Provider预设模板
PROVIDER_TEMPLATES = {
    "gemini": {
        "name": "Gemini (Google)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.5-flash-lite",
        "env_keys": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "AI_STAFF_API_KEY", "AI_STAFF_GEMINI_KEY"],
        "tier": "free",
        "max_rpm": 15,
    },
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env_keys": ["OPENAI_API_KEY", "AI_STAFF_OPENAI_KEY"],
        "tier": "cheap",
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "env_keys": ["DEEPSEEK_API_KEY"],
        "tier": "cheap",
    },
    "ollama": {
        "name": "Ollama (Local)",
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:7b",
        "api_key": "ollama",
        "tier": "free",
        "local": True,
        "env_keys": [],
    },
}


def from_env(cls, model: str = "") -> 'AIStaff':
    """V4: 零配置启动 — 从环境变量自动检测API key"""
    # Check for YAML config file first
    config_path = os.environ.get("AI_STAFF_CONFIG", "")
    if config_path and os.path.isfile(config_path):
        print(f"  [V4/from_env] Loading config from: {config_path}")
        return cls.from_config_file(config_path)
    
    proxy = os.environ.get("AI_STAFF_PROXY", "")
    default_model = model or os.environ.get("AI_STAFF_DEFAULT_MODEL", "")
    expert_id = os.environ.get("AI_STAFF_EXPERT", "generalist")
    
    # Try single-backend mode
    base_url = os.environ.get("AI_STAFF_BASE_URL", "")
    api_key = os.environ.get("AI_STAFF_API_KEY", "")
    
    if base_url and api_key:
        from ..backends.smart_init import SmartInit
        registry = SmartInit.auto_configure(
            extra_keys={"gemini": api_key},
            proxy_hint=proxy,
        )
        effective_model = default_model or registry.best_overall
        effective_proxy = proxy or registry.proxy
        print(f"  [V4/from_env] Single-backend: {base_url[:40]}... model={effective_model}")
        instance = cls(base_url=base_url, api_key=api_key, model=effective_model,
                  proxy=effective_proxy, expert_id=expert_id)
        instance._model_registry = registry
        return instance
    
    # Try provider-specific env vars
    for provider_name, tmpl in PROVIDER_TEMPLATES.items():
        if provider_name == "ollama":
            try:
                import httpx as _ht
                r = _ht.get("http://localhost:11434/api/tags", timeout=2)
                if r.status_code == 200:
                    print(f"  [V4/from_env] Detected {tmpl['name']} locally")
                    return cls(
                        base_url=tmpl["base_url"], api_key=tmpl.get("api_key", ""),
                        model=default_model or tmpl["model"], proxy=proxy,
                        expert_id=expert_id
                    )
            except Exception:
                continue
        
        for env_key in tmpl.get("env_keys", []):
            key = os.environ.get(env_key, "")
            if key:
                from ..backends.smart_init import SmartInit
                registry = SmartInit.auto_configure(
                    extra_keys={provider_name: key},
                    proxy_hint=proxy,
                )
                effective_model = default_model or registry.best_overall
                effective_proxy = proxy or registry.proxy
                print(f"  [V4/from_env] Found provider '{provider_name}' via {env_key}, model={effective_model}")
                profiles = registry.to_profiles_dict()
                if profiles:
                    instance = cls(profiles=profiles, proxy=effective_proxy, expert_id=expert_id)
                    instance._attach_ai_router(registry)
                    return instance
                else:
                    return cls(
                        base_url=tmpl["base_url"], api_key=key,
                        model=effective_model, proxy=effective_proxy, expert_id=expert_id
                    )
    
    # Nothing found — fall back to discover
    print("  [V4/from_env] No env vars found, trying auto-discover...")
    try:
        return cls.discover_and_start()
    except Exception as e:
        raise RuntimeError(
            "V4 from_env() failed: no API keys found in environment.\n"
            "Set one of:\n"
            "  - AI_STAFF_API_KEY + AI_STAFF_BASE_URL\n"
            "  - GEMINI_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY\n"
            "  - AI_STAFF_CONFIG=/path/to/config.yaml\n"
            "Or use: AIStaff.quick_start('your-api-key')"
        ) from e


def quick_start(cls, api_key: str = "", provider: str = "gemini",
                proxy: str = "", model: str = "",
                auto_detect: bool = True,
                extra_keys: dict = None) -> 'AIStaff':
    """V4: 快速启动 — 一个key搞定"""
    # Ollama特殊处理
    if provider == "ollama" and not extra_keys:
        tmpl = PROVIDER_TEMPLATES["ollama"]
        final_model = model or tmpl.get("model", "qwen2.5:7b")
        print(f"  [V4/quick_start] Ollama -> {final_model} (local)")
        return cls(
            base_url=tmpl["base_url"],
            api_key=tmpl.get("api_key", "ollama"),
            model=final_model,
            proxy=proxy,
        )
    
    # 手动指定模式
    if model and proxy and not auto_detect and api_key:
        tmpl = PROVIDER_TEMPLATES.get(provider, {})
        base_url = tmpl.get("base_url", "https://api.openai.com/v1")
        print(f"  [V4/quick_start] Manual: {provider} -> {model}")
        return cls(base_url=base_url, api_key=api_key, model=model, proxy=proxy)
    
    # SmartInit V2: 全provider扫描
    from ..backends.smart_init import SmartInit
    
    keys_dict = dict(extra_keys) if extra_keys else {}
    if api_key and provider not in keys_dict:
        keys_dict[provider] = api_key
    
    registry = SmartInit.auto_configure(
        extra_keys=keys_dict if keys_dict else None,
        proxy_hint=proxy,
    )
    
    cls._last_registry = registry
    final_model = model or registry.best_overall
    profiles = registry.to_profiles_dict()
    
    if profiles:
        usable_count = len(registry.usable_models)
        providers = set(m.provider for m in registry.usable_models)
        print(f"  [V4/quick_start] {usable_count} model(s) from {len(providers)} provider(s), "
              f"best={registry.best_overall}, proxy={'auto' if registry.proxy else 'none'}")
        instance = cls(profiles=profiles, proxy=registry.proxy)
        best_key = final_model.replace("-", "_").replace(".", "")
        for pkey in instance.multi_llm._clients:
            if final_model.replace("-", "_").replace(".", "") in pkey:
                best_key = pkey
                break
        if best_key in instance.multi_llm._clients:
            instance.llm = instance.multi_llm._clients[best_key]
        instance._attach_ai_router(registry)
        return instance
    else:
        print(f"  [V4/quick_start] No usable models, trying {final_model} anyway")
        pdef = SmartInit.PROVIDER_DEFS.get(provider, {}) if hasattr(SmartInit, 'PROVIDER_DEFS') else {}
        base_url = pdef.get("base_url", "https://api.openai.com/v1")
        return cls(
            base_url=base_url, api_key=api_key or "",
            model=final_model, proxy=registry.proxy,
        )


def discover_and_start(cls, proxy: str = "") -> 'AIStaff':
    """终极懒人模式：自动探测任何可用LLM后端"""
    all_env_patterns = {}
    for pname, tmpl in PROVIDER_TEMPLATES.items():
        for ek in tmpl.get("env_keys", []):
            all_env_patterns[ek] = (pname, tmpl)
    
    for env_key, (pname, tmpl) in sorted(all_env_patterns.items()):
        val = os.environ.get(env_key, "")
        if val:
            print(f"  [V4/discover] ✓ Found {env_key} → {tmpl['name']}")
            return cls(
                base_url=tmpl["base_url"], api_key=val,
                model=tmpl.get("model", ""), proxy=proxy
            )
    
    # Try Ollama
    try:
        import httpx as _ht
        r = _ht.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            tmpl = PROVIDER_TEMPLATES["ollama"]
            print(f"  [V4/discover] ✓ Found {tmpl['name']} running locally")
            return cls(base_url=tmpl["base_url"], api_key="ollama",
                      model=tmpl["model"], proxy=proxy)
    except Exception:
        pass
    
    # Try common config paths
    config_paths = [
        os.path.expanduser("~/.ai-staff/config.yaml"),
        os.path.expanduser("~/.config/ai-staff/config.yaml"),
        ".ai-staff/config.yaml",
    ]
    for cp in config_paths:
        if os.path.isfile(cp):
            print(f"  [V4/discover] ✓ Found config at: {cp}")
            return cls.from_config_file(cp)
    
    # Nothing found
    scanned = list(all_env_patterns.keys()) + ["Ollama (localhost:11434)"] + config_paths
    raise RuntimeError(
        "V4 discover_and_start() found no available LLM backend.\n"
        f"Scanned: {', '.join(scanned)}\n\n"
        "Quick fix options:\n"
        "  1. export GEMINI_API_KEY='your-key'\n"
        "  2. staff = AIStaff.quick_start('your-key')\n"
        "  3. Start Ollama: ollama serve\n"
        "  4. Create config.yaml (see config.example.yaml)"
    )


def from_config_file(cls, path: str) -> 'AIStaff':
    """从YAML配置文件加载多后端配置"""
    import yaml as _yaml
    
    with open(path, 'r', encoding='utf-8') as f:
        cfg = _yaml.safe_load(f)
    
    # 展开 ${ENV_VAR}
    def expand_env(value):
        if isinstance(value, str):
            def _replace_env(match):
                var_name = match.group(1)
                return os.environ.get(var_name, match.group(0))
            return re.sub(r'\$\{(\w+)\}', _replace_env, value)
        elif isinstance(value, dict):
            return {k: expand_env(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [expand_env(v) for v in value]
        return value
    
    cfg = expand_env(cfg)
    
    from ..backends.profile import BackendProfile
    profiles = {}
    for pname, pdata in cfg.get('profiles', {}).items():
        profiles[pname] = BackendProfile(
            name=pname,
            base_url=pdata['base_url'],
            api_key=pdata.get('api_key', ''),
            model=pdata.get('model', 'gpt-4o-mini'),
            proxy=pdata.get('proxy', ''),
            tier=pdata.get('tier', 'standard'),
            max_rpm=pdata.get('max_rpm', 0),
            priority=pdata.get('priority', 0),
            enabled=pdata.get('enabled', True),
        )
    
    settings = cfg.get('settings', {})
    proxy = settings.get('proxy', '') or cfg.get('default_proxy', '')
    expert_id = settings.get('default_expert', 'generalist') or cfg.get('default_expert', 'generalist')
    
    instance = cls(profiles=profiles, proxy=proxy, expert_id=expert_id)
    instance._config_path = path
    
    print(f"  [V4/from_config] Loaded {len(profiles)} backend(s) from {path}")
    return instance


__all__ = ['PROVIDER_TEMPLATES', 'from_env', 'quick_start', 'discover_and_start', 'from_config_file']
