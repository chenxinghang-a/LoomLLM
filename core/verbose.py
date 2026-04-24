"""
AI-Staff V4.2 — 彩色日志 + Token成本实时显示

替代所有裸 print()，让AI协作过程可视化：
  🟢 Writer(执行)    — 绿色
  🔵 Reviewer(审查)  — 蓝色
  🟡 Planner(规划)   — 黄色
  🔴 Error(错误)     — 红色
  ⚪ System(系统)    — 灰色
  💰 Budget(成本)    — 品红

用法：
  from .verbose import log, set_verbose
  set_verbose(True)   # 开启彩色（默认开）
  log.writer("正在执行...")
  log.reviewer("评分 85/100")
  log.error("429 Rate Limited")
  log.budget(tokens=1234, cost=0.002)
"""
from __future__ import annotations
import os, sys, time
from dataclasses import dataclass, field
from typing import Optional


# ── ANSI颜色码 ──
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    
    GREEN   = "\033[32m"
    BLUE    = "\033[34m"
    YELLOW  = "\033[33m"
    RED     = "\033[31m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    GRAY    = "\033[90m"
    
    BG_GREEN  = "\033[42m\033[30m"
    BG_BLUE   = "\033[44m\033[37m"
    BG_YELLOW = "\033[43m\033[30m"
    BG_RED    = "\033[41m\033[37m"


# Windows 终端彩色支持
def _enable_ansi():
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass
        # 强制stdout/stderr走UTF-8，根治GBK乱码
        try:
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass


_ansi_enabled = False
_verbose = True  # 默认开启

# Windows UTF-8强制 — 模块加载时立即生效
if sys.platform == "win32":
    _enable_ansi()


def _safe_print(text: str):
    """Windows GBK安全的print"""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        encoded = text.encode(sys.stdout.encoding or 'utf-8', errors='replace')
        print(encoded.decode(sys.stdout.encoding or 'utf-8', errors='replace'), flush=True)


def set_verbose(enabled: bool = True):
    """全局开关"""
    global _verbose
    _verbose = enabled


def _supports_color() -> bool:
    """检测终端是否支持彩色"""
    if not _verbose:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    # Windows 10+ 或 Unix
    return hasattr(sys.stdout, 'isatty') and (sys.stdout.isatty() or os.environ.get("FORCE_COLOR"))


def _fmt(color: str, icon: str, label: str, msg: str) -> str:
    """格式化一行彩色日志"""
    # Windows GBK兼容：emoji转安全字符
    _safe_icons = {
        "⚪": "[SYS]", "🟢": "[WRT]", "🔵": "[REV]", "🟡": "[PLN]",
        "🔴": "[ERR]", "✅": "[OK]", "⚠️": "[WRN]", "💰": "[$$]",
        "🔀": "[RTE]", "🟠": "[FIX]", "🟣": "[RBT]",
    }
    safe_icon = _safe_icons.get(icon, icon)
    try:
        safe_icon.encode(sys.stdout.encoding or 'utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        safe_icon = _safe_icons.get(icon, f"[{label[:3].upper()}]")
    
    if _supports_color():
        return f"  {color}{safe_icon} {label:12s}{C.RESET} {msg}"
    else:
        return f"  {safe_icon} [{label}] {msg}"


# ── Token/成本追踪器 ──
@dataclass
class _CostTracker:
    """会话级Token+成本累计"""
    total_input: int = 0
    total_output: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0
    start_time: float = field(default_factory=time.time)
    
    def record(self, prompt_tokens: int, completion_tokens: int,
               model: str = "", input_cost_per_1k: float = 0, output_cost_per_1k: float = 0):
        self.total_input += prompt_tokens
        self.total_output += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.call_count += 1
        cost = (prompt_tokens * input_cost_per_1k + completion_tokens * output_cost_per_1k) / 1000
        self.total_cost_usd += cost
    
    def summary_line(self) -> str:
        elapsed = time.time() - self.start_time
        tps = self.total_tokens / elapsed if elapsed > 0 else 0
        cost_str = f"${self.total_cost_usd:.4f}" if self.total_cost_usd > 0 else "free"
        return (f"[COST] {self.total_tokens:,} tokens ({self.call_count} calls) | "
                f"{tps:.0f} tok/s | {cost_str} | {elapsed:.1f}s")


# 全局成本追踪器实例
cost_tracker = _CostTracker()


class _Logger:
    """彩色日志输出器"""
    
    def __init__(self):
        self._indent = 0
        self._phase_start: dict[str, float] = {}  # phase → start_time
    
    def _phase(self, color: str, icon: str, label: str, msg: str, 
               timing: bool = False, phase_id: str = ""):
        if not _verbose:
            return
        parts = []
        if timing and phase_id:
            if phase_id in self._phase_start:
                elapsed = time.time() - self._phase_start[phase_id]
                parts.append(f"{C.DIM if _supports_color() else ''}{elapsed:.1f}s{C.RESET if _supports_color() else ''}")
            else:
                self._phase_start[phase_id] = time.time()
        
        timing_str = f" {parts[0]}" if parts else ""
        indent = "  " * self._indent
        line = _fmt(color, icon, label, f"{indent}{msg}{timing_str}")
        _safe_print(line)
    
    # ── 便捷方法 ──
    
    def system(self, msg: str, **kw):
        """⚪ 系统消息"""
        self._phase(C.GRAY, "⚪", "System", msg, **kw)
    
    def writer(self, msg: str, **kw):
        """🟢 Writer/执行阶段"""
        self._phase(C.GREEN, "🟢", "Writer", msg, **kw)
    
    def reviewer(self, msg: str, **kw):
        """🔵 Reviewer/审查阶段"""
        self._phase(C.BLUE, "🔵", "Reviewer", msg, **kw)
    
    def planner(self, msg: str, **kw):
        """🟡 Planner/规划阶段"""
        self._phase(C.YELLOW, "🟡", "Planner", msg, **kw)
    
    def error(self, msg: str, **kw):
        """🔴 错误"""
        self._phase(C.RED, "🔴", "Error", msg, **kw)
    
    def success(self, msg: str, **kw):
        """🟢 成功"""
        self._phase(C.GREEN, "✅", "Done", msg, **kw)
    
    def warn(self, msg: str, **kw):
        """🟡 警告"""
        self._phase(C.YELLOW, "⚠️", "Warn", msg, **kw)
    
    def budget(self, tokens: int = 0, cost: float = 0.0, model: str = "", 
               phase: str = "", **kw):
        """Token cost real-time display"""
        if not _verbose:
            return
        parts = []
        if model:
            parts.append(model)
        if tokens:
            parts.append(f"{tokens:,} tok")
        if cost > 0:
            parts.append(f"${cost:.4f}")
        elif tokens:
            parts.append("free")
        if phase:
            parts.append(f"[{phase}]")
        msg = " | ".join(parts) if parts else "---"
        self._phase(C.MAGENTA, "💰", "Budget", msg, **kw)
        # Session累计行（灰色缩进）
        if cost_tracker.call_count > 0:
            summary = cost_tracker.summary_line()
            if _supports_color():
                _safe_print(f"  {C.GRAY}              {summary}{C.RESET}")
            else:
                _safe_print(f"              {summary}")
    
    def route(self, task_type: str, writer: str, reviewer: str, **kw):
        """🔀 路由决策"""
        if not _verbose:
            return
        msg = f"{task_type} → Writer: {writer} | Reviewer: {reviewer}"
        self._phase(C.CYAN, "🔀", "Route", msg, **kw)
    
    def phase_start(self, phase: str, iteration: int = 0, **kw):
        """标记阶段开始"""
        self._phase_start[phase] = time.time()
        icon = {"exec": "🟢", "review": "🔵", "revise": "🟠", "rebuttal": "🟣"}.get(phase, "⚪")
        label = {"exec": "Writer", "review": "Reviewer", "revise": "Revising", "rebuttal": "Rebuttal"}.get(phase, phase)
        color = {"exec": C.GREEN, "review": C.BLUE, "revise": C.YELLOW, "rebuttal": C.MAGENTA}.get(phase, C.GRAY)
        iter_str = f" #{iteration}" if iteration else ""
        self._phase(color, icon, label, f"开始{iter_str}", timing=True, phase_id=phase)
    
    def phase_end(self, phase: str, score: int = -1, chars: int = 0, **kw):
        """标记阶段结束"""
        parts = []
        if phase in self._phase_start:
            elapsed = time.time() - self._phase_start[phase]
            parts.append(f"{elapsed:.1f}s")
        if score >= 0:
            color_code = C.GREEN if score >= 80 else (C.YELLOW if score >= 50 else C.RED)
            if _supports_color():
                parts.append(f"评分:{color_code}{score}/100{C.RESET}")
            else:
                parts.append(f"评分:{score}/100")
        if chars:
            parts.append(f"{chars:,}ch")
        msg = " | ".join(parts) if parts else "完成"
        icon = {"exec": "🟢", "review": "🔵", "revise": "🟠", "rebuttal": "🟣"}.get(phase, "✅")
        label = {"exec": "Writer", "review": "Reviewer", "revise": "Revising", "rebuttal": "Rebuttal"}.get(phase, phase)
        color = {"exec": C.GREEN, "review": C.BLUE, "revise": C.YELLOW, "rebuttal": C.MAGENTA}.get(phase, C.GRAY)
        self._phase(color, icon, label, msg)
    
    def divider(self, title: str = "", char: str = "─", width: int = 60):
        """分隔线"""
        if not _verbose:
            return
        if _supports_color():
            _safe_print(f"  {C.DIM}{char * width}{C.RESET}")
            if title:
                pad = (width - len(title) - 2) // 2
                _safe_print(f"  {C.BOLD}{char * pad} {title} {char * pad}{C.RESET}")
        else:
            _safe_print(f"  {char * width}")
            if title:
                pad = (width - len(title) - 2) // 2
                _safe_print(f"  {char * pad} {title} {char * pad}")


# 全局日志实例
log = _Logger()

# 初始化时启用ANSI
set_verbose(True)


__all__ = ['log', 'set_verbose', 'cost_tracker']
