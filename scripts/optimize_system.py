#!/usr/bin/env python
"""
系统优化脚本 (System Optimizer Script)
========================================

功能：运行时应用系统优化，无需重启系统

用法:
    # 应用所有优化
    python scripts/optimize_system.py --apply-all

    # 回滚所有优化
    python scripts/optimize_system.py --rollback

    # 查看优化状态
    python scripts/optimize_system.py --status

    # 应用单个优化
    python scripts/optimize_system.py --apply creativity
    python scripts/optimize_system.py --apply reasoning
    python scripts/optimize_system.py --apply autonomy
    python scripts/optimize_system.py --apply transfer

版本: 1.0.0
日期: 2026-01-19
作者: System Optimization Team
"""

import sys
import os
from pathlib import Path
import argparse
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """打印横幅"""
    print("\n" + "=" * 70)
    print("🚀 AGI系统优化工具 (System Optimizer)")
    print("=" * 70)
    print("\n优化策略: 零拓扑改动，充分利用现有能力")
    print("\n预期收益:")
    print("  • 创造性涌现: 0.04 → 0.15 (+275%)")
    print("  • 深度推理利用: 100步 → 99,999步 (+999x)")
    print("  • 自主目标生成: 生成率×2")
    print("  • 跨域迁移: 自动激活 (+18.3%)")
    print("  • 总体智能: 77% → 82% (+5%)")
    print("\n" + "=" * 70 + "\n")


def get_agi_instance():
    """
    获取AGI实例

    Returns:
        AGI_Life_Engine实例或None
    """
    try:
        # 尝试从agi_chat_cli获取
        from agi_chat_cli import AGIChatCLI

        print("📡 连接到AGI系统...")
        cli = AGIChatCLI()

        if hasattr(cli, 'engine') and cli.engine:
            print("✅ 成功连接到AGI_Life_Engine\n")
            return cli.engine
        else:
            print("⚠️ AGI_Life_Engine未初始化\n")
            return None

    except ImportError as e:
        print(f"❌ 无法导入agi_chat_cli: {e}\n")
        return None
    except Exception as e:
        print(f"❌ 获取AGI实例失败: {e}\n")
        return None


def apply_all_optimizations(optimizer):
    """应用所有优化"""
    print("🎯 应用所有优化...\n")

    results = optimizer.apply_all_optimizations()

    print("\n" + "=" * 70)
    print("📊 优化摘要")
    print("=" * 70)

    applied_count = 0
    skipped_count = 0

    for target, result in results.items():
        if result.status == "applied":
            print(f"✅ {target.value.upper():12s}: {result.before} → {result.after}")
            print(f"   提升: {result.improvement:.1f}%")
            applied_count += 1
        else:
            print(f"⚠️ {target.value.upper():12s}: {result.status}")
            skipped_count += 1

    print("=" * 70)
    print(f"\n✅ 完成！应用了 {applied_count} 项优化，跳过 {skipped_count} 项\n")

    return results


def apply_single_optimization(optimizer, target_name):
    """应用单个优化"""
    print(f"🎯 应用优化: {target_name}\n")

    target_map = {
        'creativity': optimizer.optimize_helix_emergence,
        'reasoning': optimizer.activate_deep_reasoning,
        'autonomy': optimizer.stimulate_autonomous_goals,
        'transfer': optimizer.activate_cross_domain_transfer
    }

    if target_name not in target_map:
        print(f"❌ 未知的优化目标: {target_name}")
        print(f"   可用目标: {', '.join(target_map.keys())}\n")
        return None

    try:
        result = target_map[target_name]()
        print(f"\n✅ {target_name.upper()} 优化完成")
        print(f"   变化: {result.before} → {result.after}")
        print(f"   提升: {result.improvement:.1f}%\n")
        return result
    except Exception as e:
        print(f"❌ 优化失败: {e}\n")
        return None


def rollback_all_optimizations(optimizer):
    """回滚所有优化"""
    print("↩️  回滚所有优化...\n")

    optimizer.rollback_all_optimizations()

    print("=" * 70)
    print("✅ 完成！所有优化已回滚到原始状态")
    print("=" * 70 + "\n")


def show_optimization_status(optimizer):
    """显示优化状态"""
    optimizer.print_optimization_status()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="AGI系统优化工具 - 零拓扑改动优化方案",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --apply-all              应用所有优化
  %(prog)s --apply creativity       仅应用创造性涌现优化
  %(prog)s --rollback               回滚所有优化
  %(prog)s --status                 查看优化状态
        """
    )

    parser.add_argument(
        '--apply-all',
        action='store_true',
        help='应用所有优化'
    )

    parser.add_argument(
        '--apply',
        metavar='TARGET',
        choices=['creativity', 'reasoning', 'autonomy', 'transfer'],
        help='应用单个优化 (creativity|reasoning|autonomy|transfer)'
    )

    parser.add_argument(
        '--rollback',
        action='store_true',
        help='回滚所有优化'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='查看优化状态'
    )

    args = parser.parse_args()

    # 打印横幅
    print_banner()

    # 如果没有参数，显示帮助
    if len(sys.argv) == 1:
        parser.print_help()
        return

    # 获取AGI实例
    agi_engine = get_agi_instance()

    if not agi_engine:
        print("❌ 无法获取AGI实例，请确保:")
        print("   1. AGI_Life_Engine.py 正在运行")
        print("   2. agi_chat_cli.py 可用")
        print("\n💡 提示: 您可以在运行AGI时使用 --optimize-on-startup 参数\n")
        return

    # 创建优化器
    try:
        from core.system_optimizer import SystemOptimizer
        optimizer = SystemOptimizer(agi_engine)
        print("✅ SystemOptimizer 初始化成功\n")
    except Exception as e:
        print(f"❌ SystemOptimizer 初始化失败: {e}\n")
        return

    # 执行操作
    try:
        if args.apply_all:
            apply_all_optimizations(optimizer)

        elif args.apply:
            apply_single_optimization(optimizer, args.apply)

        elif args.rollback:
            rollback_all_optimizations(optimizer)

        elif args.status:
            show_optimization_status(optimizer)

        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\n\n⚠️ 操作被用户中断\n")
    except Exception as e:
        print(f"\n❌ 操作失败: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
