"""
LoomLLM Global Constants & Shared Configuration.

Central location for all magic numbers, defaults, and version info.
Every module imports from here — no more undefined name errors.
"""

from pathlib import Path

__all__ = [
    'VERSION', 'DEFAULT_TIMEOUT', 'MAX_RETRIES', 'RETRY_DELAY',
    'MEMORY_DB', 'DEFAULT_MODEL', 'DEFAULT_BASE_URL',
    'PACKAGE_ROOT',
]

# ── Version ──
VERSION = "1.0.0"

# ── API Defaults ──
DEFAULT_TIMEOUT = 120          # seconds per API call
MAX_RETRIES = 2                # retry attempts (429快速失败让上层降级)
RETRY_DELAY = 1.0              # base delay (秒，429不需要等太久)

DEFAULT_MODEL = ""           # No default — must be provided by user or auto-discovered
DEFAULT_BASE_URL = ""        # No default — provider-specific, set via config or SmartInit

# ── Paths ──
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MEMORY_DB = PACKAGE_ROOT / ".ai_staff_memory.db"
