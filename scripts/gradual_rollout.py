#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
灰度发布脚本 (Gradual Rollout Script)
自动化B方案的灰度发布流程

功能：
1. 阶段化发布（10% -> 50% -> 100%）
2. 实时监控和自动回滚
3. 流量分配
4. 健康检查

作者：Claude Code (Sonnet 5.0)
创建日期：2026-01-13
版本：v1.0
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.production_config import ProductionConfig, get_production_config
from monitoring.fractal_monitor import FractalMonitor, get_monitor


class RolloutManager:
    """灰度发布管理器"""

    def __init__(self, config: ProductionConfig):
        self.config = config
        self.monitor = get_monitor(config)
        self.current_stage = 0
        self.rollout_log: List[Dict[str, Any]] = []

    def execute_rollout(self):
        """执行完整的灰度发布流程"""
        stages = self.config.rollout.rollout_stages
        duration = self.config.rollout.stage_duration_minutes

        print("="*60)
        print(f"[发布] 开始灰度发布流程")
        print(f"[发布] 灰度阶段: {stages}")
        print(f"[发布] 每阶段时长: {duration}分钟")
        print("="*60)

        for i, percentage in enumerate(stages):
            self.current_stage = i
            stage_name = f"阶段{i+1}: {percentage}%流量"

            print(f"\n{'='*60}")
            print(f"[发布] {stage_name}")
            print(f"{'='*60}")

            # 设置流量百分比
            self.config.fractal_config.rollout_percentage = percentage

            # 记录开始
            start_time = datetime.now()
            self.log_event(
                stage=f"stage_{i+1}",
                event="start",
                percentage=percentage,
                timestamp=start_time.isoformat()
            )

            # 执行阶段
            success = self._execute_stage(percentage, duration)

            # 记录结束
            end_time = datetime.now()
            duration_actual = (end_time - start_time).total_seconds() / 60

            self.log_event(
                stage=f"stage_{i+1}",
                event="complete" if success else "failed",
                percentage=percentage,
                duration_minutes=duration_actual,
                timestamp=end_time.isoformat()
            )

            if not success:
                print(f"\n[发布] 阶段{i+1}失败，执行回滚")
                self._rollback()
                return False

        print(f"\n{'='*60}")
        print(f"[成功] 灰度发布完成！")
        print(f"{'='*60}")
        return True

    def _execute_stage(self, percentage: int, duration_minutes: int) -> bool:
        """执行单个阶段"""
        print(f"[发布] 流量设置为: {percentage}%")
        print(f"[发布] 监控时长: {duration_minutes}分钟")

        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        check_interval = 30  # 每30秒检查一次

        while datetime.now() < end_time:
            # 获取统计
            stats = self.monitor.collector.get_statistics(window_minutes=5)

            # 检查是否需要回滚
            if self._should_rollback(stats):
                print(f"\n[发布] 触发回滚条件")
                self._print_stats(stats)
                return False

            # 定期打印状态
            elapsed = (end_time - datetime.now()).total_seconds() / 60
            if elapsed % 5 < 1:  # 每5分钟打印一次
                self._print_stats(stats)

            time.sleep(check_interval)

        # 最终检查
        stats = self.monitor.collector.get_statistics(window_minutes=5)
        self._print_stats(stats)

        # 判断阶段是否成功
        return not self._should_rollback(stats)

    def _should_rollback(self, stats: Dict[str, Any]) -> bool:
        """判断是否应该回滚"""
        if stats['total_requests'] < 10:
            return False  # 样本太少

        # 检查错误率
        if stats['error_rate'] > self.config.rollout.max_error_rate_for_rollback:
            print(f"[回滚] 错误率过高: {stats['error_rate']:.2%}")
            return True

        # 检查延迟
        if stats['response_time']['p95'] > self.config.rollout.max_latency_for_rollback_ms:
            print(f"[回滚] 延迟过高: {stats['response_time']['p95']:.2f}ms")
            return True

        return False

    def _rollback(self):
        """回滚到A组"""
        print(f"\n[回滚] 开始回滚流程")
        print(f"[回滚] 切换回A组模式...")

        # 切换到A组
        self.config.fractal_config.mode = "GROUP_A"
        self.config.fractal_config.enable_fractal = False

        print(f"[回滚] 已回滚到A组模式")

        self.log_event(
            stage="rollback",
            event="executed",
            timestamp=datetime.now().isoformat()
        )

    def _print_stats(self, stats: Dict[str, Any]):
        """打印统计信息"""
        print(f"[状态] 请求数: {stats['total_requests']}, "
              f"错误率: {stats['error_rate']:.2%}, "
              f"P95延迟: {stats['response_time']['p95']:.2f}ms, "
              f"外部依赖: {stats['external_dependency']:.2%}")

    def log_event(self, **kwargs):
        """记录事件"""
        event = {
            **kwargs,
            'logged_at': datetime.now().isoformat()
        }
        self.rollout_log.append(event)

        # 保存到文件
        log_file = Path("monitoring/rollout_log.json")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.rollout_log, f, indent=2, ensure_ascii=False)

    def save_report(self, filename: str = "monitoring/rollout_report.json"):
        """保存发布报告"""
        report = {
            'stages_completed': self.current_stage + 1,
            'total_stages': len(self.config.rollout.rollout_stages),
            'log': self.rollout_log,
            'final_stats': self.monitor.collector.get_statistics(window_minutes=60),
            'timestamp': datetime.now().isoformat()
        }

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"[发布] 报告已保存: {filename}")


def quick_rollout(percentage: int = 10, duration_minutes: int = 5):
    """快速灰度（用于测试）"""
    config = get_production_config()
    config.rollout.rollout_stages = [percentage]
    config.rollout.stage_duration_minutes = duration_minutes

    manager = RolloutManager(config)
    return manager.execute_rollout()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='灰度发布脚本')
    parser.add_argument('--percentage', type=int, default=10, help='灰度百分比')
    parser.add_argument('--duration', type=int, default=5, help='持续时长（分钟）')
    parser.add_argument('--full', action='store_true', help='执行完整灰度流程')

    args = parser.parse_args()

    if args.full:
        # 完整灰度发布
        config = get_production_config()
        manager = RolloutManager(config)
        success = manager.execute_rollout()
        manager.save_report()

        if success:
            print("\n[成功] 灰度发布成功完成！")
            sys.exit(0)
        else:
            print("\n[失败] 灰度发布失败，已回滚")
            sys.exit(1)
    else:
        # 快速测试
        print(f"[测试] 快速灰度: {args.percentage}%流量, {args.duration}分钟")
        success = quick_rollout(args.percentage, args.duration)

        if success:
            print(f"\n[成功] 快速灰度完成")
        else:
            print(f"\n[失败] 快速灰度失败")
