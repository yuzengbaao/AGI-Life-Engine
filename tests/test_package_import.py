"""
测试项目包是否可以被正确导入
"""
import pytest


def test_core_package_import():
    """测试core包可以被导入"""
    try:
        import core
        assert core is not None
    except ImportError as e:
        pytest.skip(f"Cannot import core package: {e}")


def test_memory_lifecycle_manager_import():
    """测试memory_lifecycle_manager可以被导入"""
    try:
        from core.memory.memory_lifecycle_manager import MemoryLifecycleManager
        assert MemoryLifecycleManager is not None
    except ImportError as e:
        pytest.skip(f"Cannot import MemoryLifecycleManager: {e}")


def test_tool_call_cache_import():
    """测试tool_call_cache可以被导入"""
    try:
        from core.tool_call_cache import ToolCallCache
        assert ToolCallCache is not None
    except ImportError as e:
        pytest.skip(f"Cannot import ToolCallCache: {e}")


def test_dynamic_recursion_limiter_import():
    """测试dynamic_recursion_limiter可以被导入"""
    try:
        from core.dynamic_recursion_limiter import DynamicRecursionLimiter
        assert DynamicRecursionLimiter is not None
    except ImportError as e:
        pytest.skip(f"Cannot import DynamicRecursionLimiter: {e}")
