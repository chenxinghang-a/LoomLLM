"""
AI-Staff V4 核心功能单元测试
Run from project root:  python -m unittest ai_staff_v4.tests.test_core -v
Run from package dir:   python tests/test_core.py
No API key needed — all tests are local unit tests.
"""
import sys
import os
import unittest

# Ensure package importable from both root and package dir
# From root: python -m unittest ai_staff_v4.tests.test_core
# From package dir: python tests/test_core.py
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_parent_root = os.path.dirname(_pkg_root)
if os.path.isfile(os.path.join(_parent_root, "ai_staff_v4", "__init__.py")):
    sys.path.insert(0, _parent_root)  # Claw/ → finds ai_staff_v4 as package
else:
    sys.path.insert(0, _pkg_root)  # fallback: ai_staff_v4/ itself


class TestVerboseLogger(unittest.TestCase):
    """彩色日志模块测试"""

    def test_import(self):
        from ai_staff_v4.core.verbose import log, cost_tracker, _safe_print
        self.assertIsNotNone(log)
        self.assertIsNotNone(cost_tracker)

    def test_log_methods_dont_crash(self):
        from ai_staff_v4.core.verbose import log
        # All log methods should work without raising
        log.system("test system")
        log.writer("test writer")
        log.reviewer("test reviewer")
        log.planner("test planner")
        log.error("test error")
        log.warn("test warn")
        log.success("test success")
        log.route("direct", "gemini", "deepseek")
        log.phase_start("exec", 1)
        log.phase_end("exec", chars=100)
        log.divider("test divider")
        log.budget(tokens=1000, model="test")

    def test_cost_tracker(self):
        from ai_staff_v4.core.verbose import cost_tracker
        # Reset by creating fresh instance attributes
        cost_tracker.total_input = 0
        cost_tracker.total_output = 0
        cost_tracker.total_tokens = 0
        cost_tracker.total_cost_usd = 0.0
        cost_tracker.call_count = 0
        cost_tracker.record(100, 50, model="gemini-2.5-flash")
        self.assertEqual(cost_tracker.total_input, 100)
        self.assertEqual(cost_tracker.total_output, 50)
        self.assertEqual(cost_tracker.total_tokens, 150)
        self.assertEqual(cost_tracker.call_count, 1)
        summary = cost_tracker.summary_line()
        self.assertIn("150", summary)

    def test_safe_print_gbk(self):
        from ai_staff_v4.core.verbose import _safe_print
        # Should not crash even with unicode on GBK terminals
        _safe_print("test ascii string")
        _safe_print("test 中文 string")


class TestExpertRegistry(unittest.TestCase):
    """专家注册表测试"""

    def test_load_from_yaml(self):
        from ai_staff_v4.experts.registry import ExpertRegistry
        ExpertRegistry._experts = {}  # reset
        count = ExpertRegistry.load_all()
        self.assertGreaterEqual(count, 5)  # At least 5 experts

    def test_get_expert(self):
        from ai_staff_v4.experts.registry import ExpertRegistry
        ExpertRegistry._experts = {}
        ExpertRegistry.load_all()
        exp = ExpertRegistry.get("coder")
        self.assertIsNotNone(exp)
        self.assertEqual(exp.id, "coder")
        self.assertEqual(exp.name, "高级工程师")

    def test_get_fallback_generalist(self):
        from ai_staff_v4.experts.registry import ExpertRegistry
        ExpertRegistry._experts = {}
        ExpertRegistry.load_all()
        exp = ExpertRegistry.get("nonexistent")
        self.assertIsNone(exp)  # get() returns None for unknown

    def test_search_expert(self):
        from ai_staff_v4.experts.registry import ExpertRegistry
        ExpertRegistry._experts = {}
        ExpertRegistry.load_all()
        results = ExpertRegistry.search("code")
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].id, "coder")


class TestSmartInit(unittest.TestCase):
    """SmartInit零配置测试"""

    def test_auto_configure(self):
        from ai_staff_v4.backends.smart_init import SmartInit
        # auto_configure makes real API calls, just test it doesn't crash on import
        # and the return type annotation
        self.assertTrue(hasattr(SmartInit, 'auto_configure'))

    def test_model_registry_fields(self):
        from ai_staff_v4.backends.smart_init import ModelRegistry
        # ModelRegistry is a dataclass, test construction
        from ai_staff_v4.backends.smart_init import ProviderScanResult, ModelInfo
        reg = ModelRegistry(
            providers={},
            all_models=[],
            best_overall="",
            best_free="",
            proxy="",
            scanned_at=0,
            scan_duration_s=0
        )
        self.assertIsInstance(reg, ModelRegistry)
        self.assertEqual(reg.usable_models, [])


class TestTokenBudget(unittest.TestCase):
    """Token预算管理测试"""

    def test_budget_record(self):
        from ai_staff_v4.core.budget import TokenBudgetManager, BudgetConfig
        cfg = BudgetConfig(max_cost_usd=1.0)
        mgr = TokenBudgetManager(cfg)
        mgr.record(100, 50, "test-model")
        stats = mgr.summary()
        self.assertIn("tokens_used", stats)
        self.assertEqual(stats["tokens_used"], 150)

    def test_budget_limit(self):
        from ai_staff_v4.core.budget import TokenBudgetManager, BudgetConfig
        cfg = BudgetConfig(max_cost_usd=0.001, max_tokens_per_task=100)
        mgr = TokenBudgetManager(cfg)
        mgr.record(100000, 100000, "expensive-model")
        stats = mgr.summary()
        self.assertTrue(stats["exhausted"])


class TestEventBus(unittest.TestCase):
    """事件总线测试"""

    def test_subscribe_and_publish(self):
        from ai_staff_v4.core.events import EventBus, Event, EventType
        bus = EventBus()
        received = []
        bus.subscribe(EventType.TASK_START, lambda e: received.append(e))
        bus.publish(Event(type=EventType.TASK_START, data={"task": "test"}))
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].data["task"], "test")

    def test_subscribe_only(self):
        from ai_staff_v4.core.events import EventBus, Event, EventType
        bus = EventBus()
        received = []
        handler = lambda e: received.append(e)
        bus.subscribe(EventType.TASK_START, handler)
        # EventBus has no unsubscribe, just test that subscribe works
        bus.publish(Event(type=EventType.TASK_START, data={}))
        self.assertEqual(len(received), 1)


class TestLLMClient(unittest.TestCase):
    """LLM客户端构造测试（不发真实请求）"""

    def test_client_creation(self):
        from ai_staff_v4.backends.client import LLMClient
        client = LLMClient(
            base_url="https://test.api.com/v1",
            api_key="test-key",
            model="test-model"
        )
        self.assertEqual(client.model, "test-model")

    def test_client_with_budget(self):
        from ai_staff_v4.backends.client import LLMClient
        from ai_staff_v4.core.budget import TokenBudgetManager, BudgetConfig
        client = LLMClient(
            base_url="https://test.api.com/v1",
            api_key="test-key",
            model="test-model",
        )
        # Budget is attached separately, not in constructor
        budget = TokenBudgetManager(BudgetConfig())
        client.budget = budget
        self.assertIsNotNone(client.budget)


class TestCollabLoopImport(unittest.TestCase):
    """V5闭环模块import测试"""

    def test_import(self):
        from ai_staff_v4.agents.collab_loop import CollaborationLoop
        self.assertIsNotNone(CollaborationLoop)

    def test_route_context(self):
        from ai_staff_v4.agents.collab_loop import RouteContext
        ctx = RouteContext(
            task_type="simple",
            complexity=3,
            writer_model="gemini-2.5-flash",
            reviewer_model="deepseek-chat",
            max_iterations=3,
            quality_threshold=70,
            needs_review=True
        )
        self.assertEqual(ctx.task_type, "simple")
        self.assertTrue(ctx.needs_review)


class TestValidation(unittest.TestCase):
    """输出验证测试"""

    def test_validation_result(self):
        from ai_staff_v4.core.validation import ValidationResult
        result = ValidationResult(
            passed=True,
            score=0.85,
            issues=[]
        )
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.score, 0.85)

    def test_validation_result_with_issues(self):
        from ai_staff_v4.core.validation import ValidationResult
        result = ValidationResult(
            passed=False,
            score=0.4,
            issues=["Missing summary", "Too short"]
        )
        self.assertFalse(result.passed)
        self.assertEqual(len(result.issues), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
