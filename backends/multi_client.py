from __future__ import annotations
import time
from typing import Optional

# Intra-package imports
from .profile import BackendProfile
from .client import LLMClient
from .router import ModelRouter
from .fallback import FallbackManager
from ..core.budget import TokenBudgetManager
from ..core.verbose import log


class MultiLLMClient:
    """
    Unified client managing MULTIPLE LLM backends simultaneously.
    
    This is the KILLER FEATURE of ai-staff V2.2:
    - One call interface, N API backends
    - Auto-routing by task complexity
    - Cascade fallback on failure
    - Cross-API Arena mode
    - Per-expert model assignment
    - Cost-aware dispatch
    
    Usage:
        profiles = {
            "openai": BackendProfile("openai", "https://...", "sk-...", "gpt-4o-mini"),
            "gemini": BackendProfile("gemini", "https://...", "...", "gemini-2.5-flash"),
        }
        client = MultiLLMClient(profiles)
        response = client.chat(messages)           # Auto-routed
        response = client.chat(messages, profile="gemini")  # Force specific
        results = client.chat_all(messages)         # Call ALL backends
    """

    def __init__(self, profiles: dict[str, BackendProfile],
                 default_proxy: str = "",
                 default_profile: str = ""):
        self.profiles = profiles
        self.default_proxy = default_proxy
        self.default_profile = default_profile or (list(profiles.keys())[0] if profiles else "")
        
        # Create individual LLMClient instances per backend
        self._clients: dict[str, LLMClient] = {}
        for name, prof in profiles.items():
            proxy = prof.proxy or default_proxy
            client = LLMClient(prof.base_url, prof.api_key, prof.model, proxy)
            self._clients[name] = client
        
        # Sub-systems
        self.router = ModelRouter(profiles)
        self.fallback = FallbackManager(profiles)
        
        # Shared budget manager
        self.budget: Optional[TokenBudgetManager] = None
        for c in self._clients.values():
            c.budget = self.budget
        
        log.system(f"MultiLLM Initialized {len(profiles)} backend(s): "
              f"{', '.join(p.display_name for p in profiles.values())}")

    @property
    def active_profiles(self) -> list[str]:
        return [n for n, p in self.profiles.items() if p.enabled]

    def _get_client(self, profile_name: str) -> LLMClient:
        """Get or raise for a named backend."""
        if profile_name not in self._clients:
            raise ValueError(f"Unknown backend profile: '{profile_name}'. "
                           f"Available: {list(self._clients.keys())}")
        return self._clients[profile_name]

    def chat(self, messages: list[dict], temperature: float = 0.7,
              model: str = "", profile: str = "", max_tokens: int = 8192,
              user_input: str = "", expert: object = None) -> tuple[str, dict]:
        """
        Single-call unified interface with auto-routing + fallback.
        
        Returns (content_string, usage_dict_with_backend_info)
        """
        # Step 1: Determine target backend
        target_prof: Optional[BackendProfile] = None
        
        if profile and profile in self.profiles:
            target_prof = self.profiles[profile]
        elif expert and hasattr(expert, 'api_profile') and expert.api_profile in self.profiles:
            target_prof = self.profiles[expert.api_profile]
        elif user_input:
            target_prof = self.router.route(user_input, expert=expert)
        elif self.default_profile and self.default_profile in self.profiles:
            target_prof = self.profiles[self.default_profile]
        
        if not target_prof:
            raise RuntimeError("No backend selected and no default configured")

        # Step 2: Try primary backend with fallback chain
        tried = [target_prof.name]
        current = target_prof
        
        while True:
            try:
                client = self._get_client(current.name)
                effective_model = model or current.model
                
                resp_start = time.time()
                content, usage = client.chat_completion(
                    messages, temperature=temperature,
                    model=effective_model, max_tokens=max_tokens
                )
                elapsed = time.time() - resp_start
                
                # Record success
                self.fallback.record_success(current.name)
                
                # Enrich usage with routing info
                usage["backend"] = current.name
                usage["model"] = effective_model
                usage["tier"] = current.tier
                usage["time_seconds"] = round(elapsed, 2)
                
                # Estimate cost
                estimated_cost = (
                    usage["prompt_tokens"] * current.input_cost_per_1k / 1000 +
                    usage["completion_tokens"] * current.output_cost_per_1k / 1000
                )
                usage["estimated_cost_usd"] = round(estimated_cost, 6)
                
                if len(tried) > 1:
                    usage["fallback_from"] = tried[0]
                    log.success(f"Fallback OK: {current.display_name} (after {len(tried)-1} fail)")
                
                return content, usage
            
            except Exception as e:
                error_type = type(e).__name__
                self.fallback.record_failure(current.name)
                # Compact log: 只在首次和成功时详细
                log.system(f"  x {current.display_name}: {str(e)[:40]}")
                
                # Get fallback candidates
                candidates = self.fallback.get_fallback_chain(exclude=current.name)
                # Remove already-tried
                candidates = [c for c in candidates if c.name not in tried]
                
                if not candidates:
                    raise RuntimeError(
                        f"All backends exhausted after trying: {tried}. Last error: {e}"
                    )
                
                current = candidates[0]
                tried.append(current.name)
                log.system(f"-> {current.display_name} (attempt {len(tried)})")

    def chat_all(self, messages: list[dict], temperature: float = 0.7,
                 max_tokens: int = 8192, parallel: bool = True) -> dict[str, tuple[str, dict]]:
        """
        Call ALL enabled backends concurrently (or sequentially).
        
        Returns dict of {profile_name: (response, usage)}
        
        Use this for:
        - Cross-API model comparison (Arena++)
        - Consensus voting (multiple models agree/disagree)
        - Redundancy-critical applications
        """
        results = {}
        
        if not parallel:
            # 串行模式（保留兼容）
            for name in self.active_profiles:
                prof = self.profiles[name]
                try:
                    client = self._get_client(name)
                    content, usage = client.chat_completion(
                        messages, temperature=temperature,
                        model=prof.model, max_tokens=max_tokens
                    )
                    usage["backend"] = name
                    usage["model"] = prof.model
                    usage["tier"] = prof.tier
                    results[name] = (content, usage)
                except Exception as e:
                    results[name] = (f"[ERROR] {e}", {"backend": name, "error": str(e)})
                time.sleep(0.3)  # Polite rate-limit spacing
        else:
            # 并行模式 — ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def _call_one(name: str) -> tuple[str, tuple[str, dict]]:
                prof = self.profiles[name]
                try:
                    client = self._get_client(name)
                    content, usage = client.chat_completion(
                        messages, temperature=temperature,
                        model=prof.model, max_tokens=max_tokens
                    )
                    usage["backend"] = name
                    usage["model"] = prof.model
                    usage["tier"] = prof.tier
                    return name, (content, usage)
                except Exception as e:
                    return name, (f"[ERROR] {e}", {"backend": name, "error": str(e)})
            
            active = list(self.active_profiles)
            with ThreadPoolExecutor(max_workers=min(len(active), 6)) as pool:
                futures = {pool.submit(_call_one, n): n for n in active}
                for future in as_completed(futures, timeout=120):
                    try:
                        name, result = future.result(timeout=60)
                        results[name] = result
                    except Exception as e:
                        name = futures[future]
                        results[name] = (f"[ERROR] {e}", {"backend": name, "error": str(e)})
        
        return results


__all__ = ['MultiLLMClient']

