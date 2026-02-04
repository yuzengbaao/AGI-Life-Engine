#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B系统观测脚本
实时查看B系统运行状态和监控指标

作者：Claude Code (Sonnet 5.0)
创建日期：2026-01-13
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_b_system_running():
    """检查B系统是否运行"""
    print("[检查] B系统运行状态...\n")

    # 检查启动信息
    startup_file = project_root / "monitoring" / "b_system_startup.json"
    if startup_file.exists():
        with open(startup_file, 'r', encoding='utf-8') as f:
            startup_info = json.load(f)

        print("[状态] B系统启动信息:")
        print(f"  启动时间: {startup_info.get('startup_time')}")
        print(f"  运行模式: {startup_info.get('mode')}")
        print(f"  置信度阈值: {startup_info.get('confidence_threshold')}")
        print(f"  设备: {startup_info.get('device')}")
        print(f"  监控: {startup_info.get('monitoring')}")
        print(f"  运行器: {startup_info.get('runner', 'N/A')}")
        return True
    else:
        print("[信息] 未找到B系统启动信息")
        print("[提示] B系统可能未启动")
        return False


def show_monitoring_dashboard():
    """显示监控仪表板"""
    print("\n" + "="*70)
    print(" " * 20 + "B系统监控仪表板")
    print("="*70)

    # 尝试加载监控数据
    monitoring_dir = project_root / "monitoring"

    # 检查监控文件
    metrics_files = list(monitoring_dir.glob("metrics_*.json"))

    if metrics_files:
        # 读取最新的监控文件
        latest_file = sorted(metrics_files)[-1]
        print(f"\n[数据] 最新监控文件: {latest_file.name}")

        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)

            if isinstance(metrics, list) and len(metrics) > 0:
                # 计算统计
                total = len(metrics)
                response_times = [m.get('response_time_ms', 0) for m in metrics]
                confidences = [m.get('confidence', 0) for m in metrics]
                entropies = [m.get('entropy', 0) for m in metrics]
                needs_val = sum(1 for m in metrics if m.get('needs_validation', False))

                # 显示统计
                print(f"\n[统计] 数据点数量: {total}")
                print(f"\n[响应时间]")
                print(f"  平均: {sum(response_times)/total:.2f}ms")
                print(f"  最小: {min(response_times):.2f}ms")
                print(f"  最大: {max(response_times):.2f}ms")

                print(f"\n[置信度]")
                print(f"  平均: {sum(confidences)/total:.4f}")
                print(f"  最小: {min(confidences):.4f}")
                print(f"  最大: {max(confidences):.4f}")

                print(f"\n[熵值]")
                print(f"  平均: {sum(entropies)/total:.4f}")

                print(f"\n[外部依赖]")
                print(f"  外部依赖率: {needs_val/total:.1%}")
                print(f"  本地决策率: {1-needs_val/total:.1%}")

                # 来源分布
                sources = {}
                for m in metrics:
                    source = m.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1

                print(f"\n[来源分布]")
                for source, count in sources.items():
                    print(f"  {source}: {count} ({count/total:.1%})")

        except Exception as e:
            print(f"[错误] 读取监控文件失败: {e}")
    else:
        print("[信息] 未找到监控数据文件")
        print("[提示] 启动B系统后会自动生成监控数据")


def show_recent_logs():
    """显示最近的日志"""
    print("\n" + "="*70)
    print(" " * 25 + "最近日志")
    print("="*70)

    # 检查A系统日志
    a_log = project_root / "docs" / "A系统日志.txt"
    if a_log.exists():
        print(f"\n[A系统] 最后10行日志:")
        print("-" * 70)
        with open(a_log, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(f"  {line.rstrip()}")
    else:
        print("\n[A系统] 日志文件不存在")


def show_quick_start():
    """显示快速启动指南"""
    print("\n" + "="*70)
    print(" " * 20 + "快速启动指南")
    print("="*70)

    guide = """
[启动B系统]

  1. 演示模式（自动执行决策）
     python run_b_system.py --mode demo

  2. 交互模式（手动执行命令）
     python run_b_system.py --mode interactive

  3. 服务模式（后台持续运行）
     python run_b_system.py --mode service

[观测B系统]

  查看系统状态:
     python scripts/observe_system.py

  检查系统进程:
     python scripts/check_system.py

[停止B系统]

  在B系统运行窗口按 Ctrl+C

[文件位置]

  B系统主脚本: run_b_system.py
  启动信息: monitoring/b_system_startup.json
  监控数据: monitoring/metrics_*.json
  最终统计: monitoring/b_system_final_stats.json

[推荐流程]

  1. 启动B系统（交互模式）
     > python run_b_system.py --mode interactive

  2. 在B系统窗口执行命令
     > status      # 查看状态
     > decision    # 执行决策
     > batch       # 批量执行10次

  3. 在另一个窗口观测状态
     > python scripts/observe_system.py
"""
    print(guide)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='B系统观测工具')
    parser.add_argument('--quick-start', action='store_true',
                       help='显示快速启动指南')

    args = parser.parse_args()

    print("\n" + "="*70)
    print(" " * 25 + "B系统观测工具")
    print("="*70)
    print(f"  查看时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 检查B系统
    has_b_system = check_b_system_running()

    # 显示监控
    if has_b_system:
        show_monitoring_dashboard()

    # 显示日志
    show_recent_logs()

    # 显示快速启动指南
    if args.quick_start:
        show_quick_start()

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
