#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产部署实际验证脚本
模拟真实场景验证外部依赖降低效果

关键改进：
1. 使用更真实的输入模式（不是完全随机）
2. 模拟学习过程（置信度应该逐渐提升）
3. 验证B组的实际外部依赖降低

作者：Claude Code (Sonnet 5.0)
创建日期：2026-01-13
"""

import sys
import time
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.fractal_adapter import create_fractal_seed_adapter, IntelligenceMode
from monitoring.fractal_monitor import FractalMonitor, MetricPoint
from config.production_config import get_production_config


class RealisticScenarioGenerator:
    """真实场景生成器"""

    def __init__(self, state_dim: int = 64):
        self.state_dim = state_dim
        # 模拟一些"有意义"的状态模式
        self.base_patterns = {
            'exploration': np.random.randn(state_dim) * 0.5,
            'exploitation': np.random.randn(state_dim) * 0.3,
            'curiosity': np.random.randn(state_dim) * 0.7,
            'goal_directed': np.random.randn(state_dim) * 0.2,
        }

    def generate_state(self, context: str = 'general') -> np.ndarray:
        """生成更真实的状态"""
        # 根据上下文混合基础模式
        if context == 'learning':
            # 学习阶段：更多探索
            weights = [0.4, 0.3, 0.2, 0.1]
        elif context == 'stable':
            # 稳定阶段：更多利用
            weights = [0.2, 0.5, 0.1, 0.2]
        else:
            # 一般情况：均衡
            weights = [0.25, 0.25, 0.25, 0.25]

        # 混合模式
        state = np.zeros(self.state_dim)
        for pattern, weight in zip(self.base_patterns.values(), weights):
            state += pattern * weight

        # 添加一些变化
        state += np.random.randn(self.state_dim) * 0.1

        return state


class ProductionValidator:
    """生产环境验证器"""

    def __init__(self):
        self.config = get_production_config()
        self.scenario_gen = RealisticScenarioGenerator(state_dim=64)

        # 创建适配器
        self.adapter_a = create_fractal_seed_adapter(
            state_dim=64,
            action_dim=4,
            mode="GROUP_A",
            device='cpu'
        )

        self.adapter_b = create_fractal_seed_adapter(
            state_dim=64,
            action_dim=4,
            mode="GROUP_B",
            device='cpu'
        )

        # 监控
        self.monitor_a = FractalMonitor(self.config)
        self.monitor_b = FractalMonitor(self.config)

    def simulate_realistic_usage(self, num_requests: int = 1000) -> Dict[str, Any]:
        """模拟真实使用场景"""
        print(f"\n{'='*60}")
        print(f"[验证] 模拟真实使用场景")
        print(f"[验证] 请求数量: {num_requests}")
        print(f"{'='*60}")

        # 模拟学习曲线：前30%是探索阶段，后面逐渐稳定
        learning_phase = int(num_requests * 0.3)

        results_a = []
        results_b = []

        for i in range(num_requests):
            # 确定场景上下文
            if i < learning_phase:
                context = 'learning'
            else:
                context = 'stable'

            # 生成状态
            state = self.scenario_gen.generate_state(context)

            # A组决策
            start_a = time.time()
            result_a = self.adapter_a.decide(state)
            time_a = (time.time() - start_a) * 1000

            # B组决策
            start_b = time.time()
            result_b = self.adapter_b.decide(state)
            time_b = (time.time() - start_b) * 1000

            # 记录指标
            self.monitor_a.record_decision(
                response_time_ms=time_a,
                confidence=result_a.confidence,
                entropy=result_a.entropy,
                source=result_a.source,
                needs_validation=result_a.needs_validation
            )

            self.monitor_b.record_decision(
                response_time_ms=time_b,
                confidence=result_b.confidence,
                entropy=result_b.entropy,
                source=result_b.source,
                needs_validation=result_b.needs_validation
            )

            results_a.append(result_a)
            results_b.append(result_b)

            # 进度显示
            if (i + 1) % 100 == 0:
                print(f"[进度] 已完成 {i+1}/{num_requests} 请求")

        # 计算统计
        stats_a = self.monitor_a.collector.get_statistics(window_minutes=60)
        stats_b = self.monitor_b.collector.get_statistics(window_minutes=60)

        return {
            'num_requests': num_requests,
            'group_a': self._extract_results(stats_a, results_a),
            'group_b': self._extract_results(stats_b, results_b),
            'improvement': self._calculate_improvement(stats_a, stats_b)
        }

    def _extract_results(self, stats: Dict, results: List) -> Dict[str, Any]:
        """提取结果"""
        return {
            'avg_response_time_ms': stats['response_time']['avg'],
            'p95_response_time_ms': stats['response_time']['p95'],
            'avg_confidence': stats['confidence']['avg'],
            'avg_entropy': stats['entropy']['avg'],
            'external_dependency_rate': stats['external_dependency'],
            'total_requests': stats['total_requests'],
            'source_distribution': stats['sources']
        }

    def _calculate_improvement(self, stats_a: Dict, stats_b: Dict) -> Dict[str, Any]:
        """计算改进"""
        return {
            'external_dependency_reduction': (
                stats_a['external_dependency'] - stats_b['external_dependency']
            ),
            'confidence_improvement': (
                stats_b['confidence']['avg'] - stats_a['confidence']['avg']
            ),
            'response_time_overhead_ms': (
                stats_b['response_time']['avg'] - stats_a['response_time']['avg']
            ),
            'entropy_change': (
                stats_b['entropy']['avg'] - stats_a['entropy']['avg']
            )
        }

    def print_validation_results(self, results: Dict[str, Any]):
        """打印验证结果"""
        print(f"\n{'='*60}")
        print(f"[验证] 生产环境验证结果")
        print(f"{'='*60}")

        a = results['group_a']
        b = results['group_b']
        imp = results['improvement']

        print(f"\n[指标] 请求数: {results['num_requests']}")

        print(f"\n[A组 - 组件组装]")
        print(f"  平均响应时间: {a['avg_response_time_ms']:.2f}ms")
        print(f"  P95响应时间: {a['p95_response_time_ms']:.2f}ms")
        print(f"  平均置信度: {a['avg_confidence']:.4f}")
        print(f"  平均熵: {a['avg_entropy']:.4f}")
        print(f"  外部依赖率: {a['external_dependency_rate']:.2%}")
        print(f"  来源分布: {a['source_distribution']}")

        print(f"\n[B组 - 分形拓扑]")
        print(f"  平均响应时间: {b['avg_response_time_ms']:.2f}ms")
        print(f"  P95响应时间: {b['p95_response_time_ms']:.2f}ms")
        print(f"  平均置信度: {b['avg_confidence']:.4f}")
        print(f"  平均熵: {b['avg_entropy']:.4f}")
        print(f"  外部依赖率: {b['external_dependency_rate']:.2%}")
        print(f"  来源分布: {b['source_distribution']}")

        print(f"\n[改进] B组 vs A组")
        print(f"  外部依赖降低: {imp['external_dependency_reduction']:.2%}")
        print(f"  置信度提升: {imp['confidence_improvement']:+.4f}")
        print(f"  响应时间开销: {imp['response_time_overhead_ms']:+.2f}ms")
        print(f"  熵变化: {imp['entropy_change']:+.4f}")

        # 关键验证
        external_dep_b = b['external_dependency_rate']
        target = 0.20  # 20%目标

        if external_dep_b <= target:
            print(f"\n[成功] 外部依赖率 {external_dep_b:.2%} <= 目标 {target:.2%}")
            print(f"[成功] B方案验证通过！外部依赖有效降低！")
        elif external_dep_b <= 0.30:
            print(f"\n[警告] 外部依赖率 {external_dep_b:.2%} 略高于目标 {target:.2%}")
            print(f"[警告] 但相比A组仍有 {imp['external_dependency_reduction']:.2%} 的降低")
        else:
            print(f"\n[失败] 外部依赖率 {external_dep_b:.2%} 未达到预期")
            print(f"[失败] 可能需要在真实生产数据中进一步验证")

        print(f"\n{'='*60}")

    def run_validation(self, num_requests: int = 1000) -> bool:
        """运行完整验证"""
        # 模拟真实使用
        results = self.simulate_realistic_usage(num_requests)

        # 打印结果
        self.print_validation_results(results)

        # 保存结果
        self._save_results(results)

        # 返回是否成功
        return results['group_b']['external_dependency_rate'] <= 0.20

    def _save_results(self, results: Dict[str, Any]):
        """保存验证结果"""
        output_file = Path("monitoring/validation_results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n[保存] 验证结果已保存: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='生产部署验证')
    parser.add_argument('--requests', type=int, default=1000, help='模拟请求数量')
    parser.add_argument('--quick', action='store_true', help='快速验证（100请求）')

    args = parser.parse_args()

    num_requests = 100 if args.quick else args.requests

    print("="*60)
    print("[部署] 生产环境验证开始")
    print("="*60)

    # 创建验证器
    validator = ProductionValidator()

    # 运行验证
    success = validator.run_validation(num_requests)

    # 输出最终结论
    print("\n" + "="*60)
    if success:
        print("[成功] 生产部署验证通过！")
        print("[成功] 外部依赖已有效降低，系统可以投入生产使用！")
    else:
        print("[注意] 外部依赖降低未完全达到目标")
        print("[建议] 可以在生产环境中使用真实数据继续验证")
    print("="*60)

    sys.exit(0 if success else 1)
