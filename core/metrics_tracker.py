#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指标追踪系统 (Metrics Tracking System)
功能：
1. 记录决策指标（置信度、响应时间、路径选择等）
2. 生成学习曲线报告
3. 可视化系统演化
4. 保存历史数据供分析

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-13
版本：v1.0
"""

import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class DecisionRecord:
    """单次决策记录"""
    timestamp: float
    episode: int
    decision_number: int
    path: str  # 'fractal', 'seed', 'llm'
    action: int
    confidence: float
    response_time_ms: float
    reward: float
    threshold: float
    gridworld_enabled: bool = False
    gridworld_position: Optional[tuple] = None
    gridworld_distance: Optional[int] = None


class MetricsTracker:
    """指标追踪器"""

    def __init__(self, save_dir: str = "data/metrics"):
        """
        初始化指标追踪器

        Args:
            save_dir: 保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 记录
        self.records: List[DecisionRecord] = []
        self.episode_records: Dict[int, List[DecisionRecord]] = {}

        # 统计
        self.start_time = time.time()
        self.total_decisions = 0
        self.total_reward = 0.0

    def record_decision(
        self,
        path: str,
        action: int,
        confidence: float,
        response_time_ms: float,
        reward: float,
        threshold: float,
        episode: int = 0,
        gridworld_enabled: bool = False,
        gridworld_position: Optional[tuple] = None,
        gridworld_distance: Optional[int] = None
    ):
        """记录一次决策"""
        record = DecisionRecord(
            timestamp=time.time(),
            episode=episode,
            decision_number=self.total_decisions,
            path=path,
            action=action,
            confidence=confidence,
            response_time_ms=response_time_ms,
            reward=reward,
            threshold=threshold,
            gridworld_enabled=gridworld_enabled,
            gridworld_position=gridworld_position,
            gridworld_distance=gridworld_distance
        )

        self.records.append(record)

        # 按episode分组
        if episode not in self.episode_records:
            self.episode_records[episode] = []
        self.episode_records[episode].append(record)

        # 更新统计
        self.total_decisions += 1
        self.total_reward += reward

    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        if len(self.records) == 0:
            return {
                'total_decisions': 0,
                'runtime_seconds': time.time() - self.start_time
            }

        # 提取数据
        confidences = [r.confidence for r in self.records]
        response_times = [r.response_time_ms for r in self.records]
        rewards = [r.reward for r in self.records]
        paths = [r.path for r in self.records]

        # 计算统计
        summary = {
            'total_decisions': len(self.records),
            'runtime_seconds': time.time() - self.start_time,
            'confidence': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'latest': float(confidences[-1]) if confidences else 0.0
            },
            'response_time_ms': {
                'mean': float(np.mean(response_times)),
                'std': float(np.std(response_times)),
                'min': float(np.min(response_times)),
                'max': float(np.max(response_times))
            },
            'reward': {
                'total': float(np.sum(rewards)),
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'latest': float(rewards[-1]) if rewards else 0.0
            },
            'path_distribution': {
                'fractal': paths.count('fractal'),
                'seed': paths.count('seed'),
                'llm': paths.count('llm'),
                'fractal_ratio': paths.count('fractal') / len(paths) if paths else 0,
                'seed_ratio': paths.count('seed') / len(paths) if paths else 0,
                'llm_ratio': paths.count('llm') / len(paths) if paths else 0
            },
            'learning_progress': {
                'first_10_confidence': float(np.mean(confidences[:10])) if len(confidences) >= 10 else 0,
                'last_10_confidence': float(np.mean(confidences[-10:])) if len(confidences) >= 10 else 0,
                'improvement': float(np.mean(confidences[-10:]) - np.mean(confidences[:10])) if len(confidences) >= 20 else 0
            }
        }

        return summary

    def get_learning_curve(self, window_size: int = 10) -> Dict[str, List]:
        """获取学习曲线数据"""
        if len(self.records) < window_size:
            return {'error': 'Not enough data points'}

        # 滑动窗口
        windows = []
        for i in range(window_size, len(self.records) + 1):
            window = self.records[i - window_size:i]
            windows.append(window)

        # 计算每个窗口的统计
        curve = {
            'decision_numbers': [w[-1].decision_number for w in windows],
            'confidences': [float(np.mean([r.confidence for r in w])) for w in windows],
            'rewards': [float(np.mean([r.reward for r in w])) for w in windows],
            'response_times': [float(np.mean([r.response_time_ms for r in w])) for w in windows],
            'fractal_ratios': [sum(1 for r in w if r.path == 'fractal') / window_size for w in windows]
        }

        return curve

    def save_records(self, filepath: Optional[str] = None):
        """保存记录到文件"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.save_dir / f"metrics_{timestamp}.json"

        filepath = Path(filepath)

        # 转换为可序列化格式
        data = {
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'total_decisions': len(self.records),
            'total_reward': self.total_reward,
            'summary': self.get_summary(),
            'learning_curve': self.get_learning_curve(),
            'records': []
        }

        for record in self.records:
            record_data = {
                'timestamp': record.timestamp,
                'datetime': datetime.fromtimestamp(record.timestamp).isoformat(),
                'episode': record.episode,
                'decision_number': record.decision_number,
                'path': record.path,
                'action': record.action,
                'confidence': record.confidence,
                'response_time_ms': record.response_time_ms,
                'reward': record.reward,
                'threshold': record.threshold,
                'gridworld_enabled': record.gridworld_enabled,
                'gridworld_position': record.gridworld_position,
                'gridworld_distance': record.gridworld_distance
            }
            data['records'].append(record_data)

        # 保存
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return filepath

    def generate_report(self) -> str:
        """生成文本报告"""
        summary = self.get_summary()
        curve = self.get_learning_curve()

        if 'error' in curve:
            curve_info = "\n[学习曲线] 数据不足，无法生成学习曲线"
        else:
            # 显示最近10个窗口
            recent_confidences = curve['confidences'][-10:]
            curve_info = f"""
[学习曲线（最近{len(recent_confidences)}个窗口）]
  置信度趋势: {recent_confidences[0]:.4f} -> {recent_confidences[-1]:.4f}
  变化: {((recent_confidences[-1] - recent_confidences[0]) / recent_confidences[0] * 100):.1f}%
"""

        runtime = summary['runtime_seconds']
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        seconds = int(runtime % 60)

        report = f"""
{'='*70}
                        指标追踪报告
{'='*70}

[运行信息]
  总决策数: {summary['total_decisions']}
  运行时长: {hours}h {minutes}m {seconds}s
  平均每秒: {summary['total_decisions'] / runtime:.2f} 决策

[置信度分析]
  平均值: {summary['confidence']['mean']:.4f} ± {summary['confidence']['std']:.4f}
  范围: [{summary['confidence']['min']:.4f}, {summary['confidence']['max']:.4f}]
  最新: {summary['confidence']['latest']:.4f}

[响应时间分析]
  平均值: {summary['response_time_ms']['mean']:.2f} ± {summary['response_time_ms']['std']:.2f} ms
  范围: [{summary['response_time_ms']['min']:.2f}, {summary['response_time_ms']['max']:.2f}] ms

[奖励分析]
  累计: {summary['reward']['total']:.2f}
  平均: {summary['reward']['mean']:.4f} ± {summary['reward']['std']:.4f}
  最新: {summary['reward']['latest']:.4f}

[决策路径分布]
  系统B（分形）: {summary['path_distribution']['fractal_ratio']:.1%} ({summary['path_distribution']['fractal']}次)
  系统A（TheSeed）: {summary['path_distribution']['seed_ratio']:.1%} ({summary['path_distribution']['seed']}次)
  外部LLM: {summary['path_distribution']['llm_ratio']:.1%} ({summary['path_distribution']['llm']}次)

[学习进度]
  前10次平均置信度: {summary['learning_progress']['first_10_confidence']:.4f}
  后10次平均置信度: {summary['learning_progress']['last_10_confidence']:.4f}
  改进幅度: {summary['learning_progress']['improvement']:+.4f} ({(summary['learning_progress']['improvement'] / summary['learning_progress']['first_10_confidence'] * 100 if summary['learning_progress']['first_10_confidence'] > 0 else 0):.1f}%)
{curve_info}
{'='*70}
"""
        return report

    def load_records(self, filepath: str):
        """从文件加载记录"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 恢复记录
        self.records = []
        for record_data in data.get('records', []):
            record = DecisionRecord(
                timestamp=record_data['timestamp'],
                episode=record_data['episode'],
                decision_number=record_data['decision_number'],
                path=record_data['path'],
                action=record_data['action'],
                confidence=record_data['confidence'],
                response_time_ms=record_data['response_time_ms'],
                reward=record_data['reward'],
                threshold=record_data['threshold'],
                gridworld_enabled=record_data.get('gridworld_enabled', False),
                gridworld_position=record_data.get('gridworld_position'),
                gridworld_distance=record_data.get('gridworld_distance')
            )
            self.records.append(record)

        # 恢复episode分组
        for record in self.records:
            if record.episode not in self.episode_records:
                self.episode_records[record.episode] = []
            self.episode_records[record.episode].append(record)

        # 恢复统计
        self.total_decisions = len(self.records)
        self.total_reward = data.get('total_reward', 0.0)

        return len(self.records)


def test_metrics_tracker():
    """测试指标追踪系统"""
    print("="*70)
    print(" "*20 + "测试指标追踪系统")
    print("="*70)

    # 创建追踪器
    tracker = MetricsTracker()

    # 模拟100次决策
    print("\n[模拟] 记录100次决策...")
    for i in range(100):
        import random
        path = 'fractal' if i % 2 == 0 else 'seed'
        confidence = 0.5 + (i / 200)  # 模拟提升
        response_time = random.uniform(2, 10)
        reward = random.uniform(-0.5, 1.0)
        threshold = 0.55 - (i * 0.002)

        tracker.record_decision(
            path=path,
            action=random.randint(0, 3),
            confidence=confidence,
            response_time_ms=response_time,
            reward=reward,
            threshold=threshold,
            episode=i // 10
        )

    # 生成报告
    print(tracker.generate_report())

    # 保存记录
    filepath = tracker.save_records()
    print(f"\n[保存] 记录已保存至: {filepath}")

    # 测试加载
    print("\n[测试] 加载记录...")
    new_tracker = MetricsTracker()
    count = new_tracker.load_records(str(filepath))
    print(f"[成功] 已加载 {count} 条记录")

    # 生成新报告
    print(new_tracker.generate_report())

    print("\n" + "="*70)
    print(" "*25 + "测试完成")
    print("="*70)


if __name__ == "__main__":
    test_metrics_tracker()
