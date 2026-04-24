from __future__ import annotations
import time
from typing import Any, Optional

import httpx

# Intra-package imports
from ..core.constants import DEFAULT_TIMEOUT, MAX_RETRIES, RETRY_DELAY


class LLMClient:
    """Universal LLM API client supporting OpenAI-compatible endpoints."""
    
    def __init__(self, base_url: str, api_key: str, model: str,
                 proxy: str = "", timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.proxy = proxy
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # Create reusable client with proxy
        client_kwargs: dict = {"timeout": timeout, "follow_redirects": True}
        if proxy:
            # httpx accepts proxy as URL string for sync Client
            client_kwargs["proxy"] = proxy
        try:
            self._client = httpx.Client(**client_kwargs)
        except TypeError:
            # Fallback for older httpx versions that use different param names
            client_kwargs.pop("proxy", None)
            client_kwargs["proxies"] = {"all://": proxy} if proxy else {}
            self._client = httpx.Client(**client_kwargs)
        
        # Optional budget reference (set by AIStaff after init)
        self.budget = None
    
    def chat_completion(self, messages: list[dict], temperature: float = 0.7,
                        model: str = "", max_tokens: int = 8192) -> tuple[str, dict]:
        """
        Call chat completion API with retry logic.
        Returns (content_string, usage_dict)
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.post(url, headers=self.headers, json=payload)
                
                if resp.status_code == 429:
                    # 429快速失败：只短暂等待1次，让上层降级处理
                    if attempt < MAX_RETRIES - 1:
                        print(f"    [429] Retry {attempt+1}/{MAX_RETRIES}")
                        time.sleep(1)
                        continue
                    raise RuntimeError(f"429 Rate Limited after {MAX_RETRIES} attempts")
                
                resp.raise_for_status()
                data = resp.json()
                
                content = data["choices"][0]["message"]["content"]
                
                # Token stats
                usage = data.get("usage", {})
                usage_info = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                }
                
                # Record to budget manager if connected
                if self.budget:
                    self.budget.record(
                        usage_info["prompt_tokens"],
                        usage_info["completion_tokens"],
                        model or self.model
                    )
                
                return content, usage_info
            
            except httpx.HTTPStatusError as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"    [HTTP {e.response.status_code}] Retry {attempt+1}/{MAX_RETRIES}...")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise RuntimeError(f"API request failed after {MAX_RETRIES} attempts: {e}")
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"    [{type(e).__name__}] {str(e)[:80]} Retry {attempt+1}/{MAX_RETRIES}...")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise RuntimeError(f"API call failed: {e}")
        
        return "", usage_info


# ═══════════════════════════════════════════════════════════
# MULTI-BACKEND ENGINE — One Entry, All Models (KILLER FEATURE)
# ═══════════════════════════════════════════════════════════
