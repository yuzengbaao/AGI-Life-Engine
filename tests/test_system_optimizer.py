#!/usr/bin/env python
"""
SystemOptimizer 集成测试
========================

目的：验证SystemOptimizer与AGI_Life_Engine的集成
"""

import sys
import os
import io
from pathlib import Path

# 🔧 Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

logger = logging.getLogger(__name__)


def print_test_banner():
    """打印测试横幅"""
    print("\n" + "=" * 70)
    print("🧪 SystemOptimizer 集成测试")
    print("=" * 70 + "\n")


def test_optimizer_import():
    """测试1: 验证SystemOptimizer可以导入"""
    print("📦 测试1: 导入SystemOptimizer...")

    try:
        from core.system_optimizer import SystemOptimizer
        print("✅ SystemOptimizer 导入成功\n")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}\n")
        return False


def test_optimizer_initialization():
    """测试2: 验证SystemOptimizer可以初始化"""
    print("🔧 测试2: 初始化SystemOptimizer...")

    try:
        from core.system_optimizer import SystemOptimizer

        # 创建一个模拟AGI引擎
        class MockAGIEngine:
            def __init__(self):
                # 模拟双螺旋引擎
                class MockHelix:
                    def __init__(self):
                        self.emergence_threshold = 0.5
                        self.divergence_amplification = 0.0

                # 模拟推理调度器
                class MockScheduler:
                    def __init__(self):
                        self.max_depth = 1000

                # 模拟自主目标系统
                class MockGoals:
                    def __init__(self):
                        self.generation_rate = 1.0

                # 模拟跨域迁移
                class MockTransfer:
                    def __init__(self):
                        self.auto_transfer = False
                        self.similarity_threshold = 0.5
                        self.confidence_threshold = 0.5

                self.double_helix_engine = MockHelix()
                self.reasoning_scheduler = MockScheduler()
                self.autonomous_goal_system = MockGoals()
                self.cross_domain_transfer = MockTransfer()

        mock_agi = MockAGIEngine()
        optimizer = SystemOptimizer(mock_agi)

        print("✅ SystemOptimizer 初始化成功")
        print(f"   - 配置项: {len(optimizer.config)} 个")
        print(f"   - 优化历史: {len(optimizer.optimization_history)} 条\n")
        return True, optimizer

    except Exception as e:
        print(f"❌ 初始化失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False, None


def test_param_preservation(optimizer):
    """测试3: 验证参数保存和恢复"""
    print("💾 测试3: 参数保存与恢复...")

    try:
        # 保存原始参数
        optimizer.save_original_params()

        print("✅ 原始参数已保存")
        print(f"   - 保存的参数组: {len(optimizer.original_params)} 个")

        for key, value in optimizer.original_params.items():
            print(f"     • {key}: {value}\n")

        return True

    except Exception as e:
        print(f"❌ 参数保存失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_creativity_optimization(optimizer):
    """测试4: 测试创造性涌现优化"""
    print("🎨 测试4: 创造性涌现优化...")

    try:
        result = optimizer.optimize_helix_emergence()

        print("✅ 创造性涌现优化完成")
        print(f"   - 优化前: {result.before}")
        print(f"   - 优化后: {result.after}")
        print(f"   - 提升幅度: {result.improvement:.1f}%")
        print(f"   - 状态: {result.status}\n")

        return True

    except Exception as e:
        print(f"❌ 优化失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_reasoning_optimization(optimizer):
    """测试5: 测试深度推理优化"""
    print("🧠 测试5: 深度推理优化...")

    try:
        result = optimizer.activate_deep_reasoning()

        print("✅ 深度推理优化完成")
        print(f"   - 优化前: {result.before}")
        print(f"   - 优化后: {result.after}")
        print(f"   - 提升幅度: {result.improvement:.1f}%")
        print(f"   - 状态: {result.status}\n")

        return True

    except Exception as e:
        print(f"❌ 优化失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_rollback(optimizer):
    """测试6: 测试回滚功能"""
    print("↩️  测试6: 参数回滚...")

    try:
        # 保存原始参数
        optimizer.save_original_params()

        # 应用一些优化
        optimizer.optimize_helix_emergence()
        optimizer.activate_deep_reasoning()

        print("   - 已应用 2 项优化")

        # 回滚
        optimizer.restore_original_params()

        print("✅ 参数回滚成功")
        print(f"   - 优化历史: {len(optimizer.optimization_history)} 条\n")

        return True

    except Exception as e:
        print(f"❌ 回滚失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_all_optimizations(optimizer):
    """测试7: 测试批量应用所有优化"""
    print("🚀 测试7: 批量应用所有优化...")

    try:
        results = optimizer.apply_all_optimizations()

        print("✅ 批量优化完成")
        print(f"   - 成功应用: {len([r for r in results.values() if r.status == 'applied'])} 项")
        print(f"   - 跳过: {len([r for r in results.values() if r.status == 'skipped'])} 项\n")

        return True

    except Exception as e:
        print(f"❌ 批量优化失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print_test_banner()

    # 测试计数
    total_tests = 7
    passed_tests = 0

    # 测试1: 导入
    if test_optimizer_import():
        passed_tests += 1

    # 测试2: 初始化
    init_success, optimizer = test_optimizer_initialization()
    if init_success and optimizer:
        passed_tests += 1
    else:
        print("❌ 无法继续测试（初始化失败）\n")
        return

    # 测试3: 参数保存
    if test_param_preservation(optimizer):
        passed_tests += 1

    # 测试4: 创造性优化
    if test_creativity_optimization(optimizer):
        passed_tests += 1

    # 测试5: 深度推理优化
    if test_reasoning_optimization(optimizer):
        passed_tests += 1

    # 测试6: 回滚
    if test_rollback(optimizer):
        passed_tests += 1

    # 测试7: 批量优化
    if test_all_optimizations(optimizer):
        passed_tests += 1

    # 打印总结
    print("=" * 70)
    print("📊 测试总结")
    print("=" * 70)
    print(f"总测试数: {total_tests}")
    print(f"通过: {passed_tests}")
    print(f"失败: {total_tests - passed_tests}")
    print(f"通过率: {passed_tests / total_tests * 100:.1f}%")
    print("=" * 70 + "\n")

    if passed_tests == total_tests:
        print("✅ 所有测试通过！SystemOptimizer集成成功。\n")
    else:
        print("⚠️ 部分测试失败，请检查上述错误信息。\n")


if __name__ == "__main__":
    main()
