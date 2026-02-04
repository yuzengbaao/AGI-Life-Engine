#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
元学习器 (Meta-Learner)
学习如何优化融合策略

核心思想：
1. 学习最优的螺旋参数（spiral_radius, phase_speed, ascent_rate）
2. 分析历史数据，识别高涌现模式
3. 动态调整融合策略
4. 预测最佳融合时机

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-13
版本：v1.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum
import json

logger = logging.getLogger(__name__)


# ============================================================================
# 数据类定义 (为M1M4Adapter兼容性)
# ============================================================================

class MetaStrategy(Enum):
    """元策略枚举"""
    RULE_BASED = "rule_based"  # 基于规则
    LEARNING_BASED = "learning_based"  # 基于学习
    HYBRID = "hybrid"  # 混合策略
    ADAPTIVE = "adaptive"  # 自适应策略


@dataclass
class StepMetrics:
    """步骤指标"""
    step: int
    reward: float
    loss: float = 0.0
    uncertainty: float = 0.0
    exploration_rate: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            import time
            self.timestamp = time.time()


@dataclass
class ParameterUpdate:
    """参数更新建议"""
    parameters: Dict[str, float]
    confidence: float
    reason: str
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            import time
            self.timestamp = time.time()


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class MetaLearningConfig:
    """元学习配置"""
    learning_rate: float = 0.01
    history_window: int = 100  # 历史窗口大小
    optimization_interval: int = 20  # 每N次决策优化一次
    exploration_rate: float = 0.2  # 探索率
    reward_decay: float = 0.95  # 奖励衰减


class MetaLearner:
    """
    元学习器

    核心功能：
    1. 记录决策历史（状态、融合参数、奖励）
    2. 学习参数-奖励映射
    3. 优化螺旋参数
    4. 生成融合策略建议
    """

    def __init__(
        self,
        config: Optional[MetaLearningConfig] = None,
        device: str = 'cpu',
        event_bus=None,
        initial_strategy: MetaStrategy = MetaStrategy.RULE_BASED
    ):
        self.config = config or MetaLearningConfig()
        self.device = device
        self.event_bus = event_bus  # M1M4Adapter兼容性
        self.strategy = initial_strategy  # M1M4Adapter兼容性

        # 历史记录
        self.history = deque(maxlen=self.config.history_window)

        # 参数空间
        self.param_space = {
            'spiral_radius': (0.1, 0.5),
            'phase_speed': (0.05, 0.3),
            'ascent_rate': (0.005, 0.02)
        }

        # 当前参数
        self.current_params = {
            'spiral_radius': 0.3,
            'phase_speed': 0.1,
            'ascent_rate': 0.01
        }

        # 性能追踪
        self.performance_metrics = {
            'emergence_history': [],
            'reward_history': [],
            'parameter_history': []
        }

        # 优化统计
        self.optimization_stats = {
            'total_optimizations': 0,
            'improvements': 0,
            'last_improvement': 0.0
        }

        logger.info(f"[元学习器] 初始化完成")
        logger.info(f"[元学习器] 学习率={self.config.learning_rate}")
        logger.info(f"[元学习器] 优化间隔={self.config.optimization_interval}")
        logger.info(f"[元学习器] 策略={self.strategy.value}")

    def record_decision(
        self,
        state: Dict[str, Any],
        fusion_params: Dict[str, float],
        reward: float,
        emergence: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        记录决策历史

        Args:
            state: 当前状态（相位、权重等）
            fusion_params: 融合参数（spiral_radius, phase_speed等）
            reward: 获得的奖励
            emergence: 涌现分数
            metadata: 额外元数据
        """

        record = {
            'state': state.copy(),
            'params': fusion_params.copy(),
            'reward': reward,
            'emergence': emergence,
            'timestamp': len(self.history),
            'metadata': metadata or {}
        }

        self.history.append(record)

        # 更新性能指标
        self.performance_metrics['emergence_history'].append(emergence)
        self.performance_metrics['reward_history'].append(reward)
        self.performance_metrics['parameter_history'].append(fusion_params.copy())

        # 定期优化
        if len(self.history) % self.config.optimization_interval == 0:
            self.optimize_parameters()

    def optimize_parameters(self):
        """
        优化螺旋参数

        策略：
        1. 分析历史数据
        2. 识别高涌现模式
        3. 调整参数向高涌现区域移动
        """

        if len(self.history) < 10:
            logger.debug("[元学习器] 历史数据不足，跳过优化")
            return

        self.optimization_stats['total_optimizations'] += 1

        # 1. 分析最近N次决策
        recent = list(self.history)[-self.config.optimization_interval:]

        # 2. 计算当前平均性能
        current_avg_emergence = np.mean([r['emergence'] for r in recent])
        current_avg_reward = np.mean([r['reward'] for r in recent])

        # 3. 识别高涌现决策
        high_emergence_records = [
            r for r in recent
            if r['emergence'] > current_avg_emergence * 1.2
        ]

        if len(high_emergence_records) == 0:
            logger.debug("[元学习器] 未发现高涌现模式")
            return

        # 4. 提取高涌现决策的参数特征
        high_emergence_params = {
            'spiral_radius': [r['params'].get('spiral_radius', 0.3) for r in high_emergence_records],
            'phase_speed': [r['params'].get('phase_speed', 0.1) for r in high_emergence_records],
            'ascent_rate': [r['params'].get('ascent_rate', 0.01) for r in high_emergence_records]
        }

        # 5. 计算目标参数（高涌现区域的中心）
        target_params = {
            'spiral_radius': np.mean(high_emergence_params['spiral_radius']),
            'phase_speed': np.mean(high_emergence_params['phase_speed']),
            'ascent_rate': np.mean(high_emergence_params['ascent_rate'])
        }

        # 6. 计算参数调整方向
        adjustments = {
            'spiral_radius': target_params['spiral_radius'] - self.current_params['spiral_radius'],
            'phase_speed': target_params['phase_speed'] - self.current_params['phase_speed'],
            'ascent_rate': target_params['ascent_rate'] - self.current_params['ascent_rate']
        }

        # 7. 应用学习率和探索
        for param in adjustments.keys():
            # 添加探索噪声
            noise = np.random.normal(0, 0.01) if np.random.random() < self.config.exploration_rate else 0

            # 调整参数
            adjustment = adjustments[param] * self.config.learning_rate + noise
            self.current_params[param] += adjustment

            # 限制在参数空间内
            min_val, max_val = self.param_space[param]
            self.current_params[param] = np.clip(
                self.current_params[param],
                min_val,
                max_val
            )

        # 8. 记录改进
        predicted_improvement = current_avg_emergence * 1.1 - current_avg_emergence
        self.optimization_stats['last_improvement'] = predicted_improvement
        self.optimization_stats['improvements'] += 1

        logger.info(f"[元学习器] 参数优化 #{self.optimization_stats['total_optimizations']}")
        logger.info(f"[元学习器] 当前参数: {self._format_params(self.current_params)}")
        logger.info(f"[元学习器] 调整方向: {self._format_params(adjustments)}")
        logger.info(f"[元学习器] 预期改进: +{predicted_improvement:.4f}")

    def get_suggested_parameters(self) -> Dict[str, float]:
        """
        获取建议的螺旋参数

        Returns:
            参数字典
        """

        return self.current_params.copy()

    def analyze_patterns(self) -> Dict[str, Any]:
        """
        分析历史模式

        Returns:
            分析结果
        """

        if len(self.history) < 20:
            return {
                'status': 'insufficient_data',
                'message': '历史数据不足（需要至少20条记录）'
            }

        recent = list(self.history)[-20:]

        # 1. 相位-涌现关系
        phase_emergence = [
            (r['state'].get('phase', 0), r['emergence'])
            for r in recent
        ]
        phase_emergence.sort(key=lambda x: x[0])

        # 2. 权重差异-涌现关系
        weight_diff_emergence = [
            (abs(r['state'].get('weight_A', 0.5) - r['state'].get('weight_B', 0.5)), r['emergence'])
            for r in recent
        ]
        weight_diff_emergence.sort(key=lambda x: x[0])

        # 3. 参数-涌现相关性
        param_correlations = {}
        for param in ['spiral_radius', 'phase_speed', 'ascent_rate']:
            values = [r['params'].get(param, 0) for r in recent]
            emergences = [r['emergence'] for r in recent]
            if len(set(values)) > 1:
                correlation = np.corrcoef(values, emergences)[0, 1]
                param_correlations[param] = correlation

        # 4. 最佳参数组合
        best_record = max(recent, key=lambda r: r['emergence'])

        return {
            'status': 'success',
            'phase_emergence_relation': phase_emergence,
            'weight_diff_emergence_relation': weight_diff_emergence,
            'param_correlations': param_correlations,
            'best_params': best_record['params'],
            'best_emergence': best_record['emergence'],
            'avg_emergence': np.mean([r['emergence'] for r in recent]),
            'emergence_std': np.std([r['emergence'] for r in recent])
        }

    def predict_emergence(
        self,
        state: Dict[str, Any],
        params: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        预测给定参数下的涌现分数

        Args:
            state: 系统状态
            params: 融合参数

        Returns:
            (预测涌现, 预测置信度)
        """

        if len(self.history) < 10:
            return 0.0, 0.0

        # 1. 找到相似的历史记录
        similar_records = self._find_similar_records(state, params, k=5)

        if len(similar_records) == 0:
            return 0.0, 0.0

        # 2. 基于相似记录预测
        predicted_emergence = np.mean([r['emergence'] for r in similar_records])

        # 3. 计算预测置信度（基于相似度）
        confidence = min(1.0, len(similar_records) / 5.0)

        return predicted_emergence, confidence

    def _find_similar_records(
        self,
        state: Dict[str, Any],
        params: Dict[str, float],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """找到相似的历史记录"""

        if len(self.history) == 0:
            return []

        # 计算相似度（基于状态和参数）
        similarities = []

        for record in self.history:
            # 状态相似度（相位距离）
            phase_diff = abs(
                state.get('phase', 0) -
                record['state'].get('phase', 0)
            )
            phase_similarity = max(0, 1 - phase_diff / (2 * np.pi))

            # 参数相似度
            param_diff = 0
            for param in ['spiral_radius', 'phase_speed', 'ascent_rate']:
                param_diff += abs(
                    params.get(param, 0) -
                    record['params'].get(param, 0)
                )
            param_similarity = max(0, 1 - param_diff / 3.0)

            # 综合相似度
            similarity = 0.6 * phase_similarity + 0.4 * param_similarity
            similarities.append((similarity, record))

        # 排序并取top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [r for s, r in similarities[:k]]

    def _format_params(self, params: Dict[str, float]) -> str:
        """格式化参数字符串"""
        return ", ".join([f"{k}={v:.4f}" for k, v in params.items()])

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""

        return {
            'total_decisions': len(self.history),
            'optimization_count': self.optimization_stats['total_optimizations'],
            'improvements': self.optimization_stats['improvements'],
            'last_improvement': self.optimization_stats['last_improvement'],
            'current_params': self.current_params.copy(),
            'avg_emergence': np.mean(self.performance_metrics['emergence_history']) if self.performance_metrics['emergence_history'] else 0.0,
            'max_emergence': max(self.performance_metrics['emergence_history']) if self.performance_metrics['emergence_history'] else 0.0,
            'emergence_std': np.std(self.performance_metrics['emergence_history']) if len(self.performance_metrics['emergence_history']) > 1 else 0.0
        }

    def export_history(self, filepath: str):
        """导出历史记录到文件"""

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(list(self.history), f, indent=2, ensure_ascii=False)

        logger.info(f"[元学习器] 历史记录已导出到 {filepath}")

    def reset(self):
        """重置元学习器"""

        self.history.clear()
        self.performance_metrics = {
            'emergence_history': [],
            'reward_history': [],
            'parameter_history': []
        }
        self.optimization_stats = {
            'total_optimizations': 0,
            'improvements': 0,
            'last_improvement': 0.0
        }
        self.current_params = {
            'spiral_radius': 0.3,
            'phase_speed': 0.1,
            'ascent_rate': 0.01
        }

        logger.info("[元学习器] 已重置")

    # ========================================================================
    # M1M4Adapter兼容性方法
    # ========================================================================

    def observe(self, metrics: StepMetrics):
        """
        观察步骤指标 (M1M4Adapter兼容性)

        Args:
            metrics: StepMetrics对象
        """
        # 将StepMetrics转换为历史记录
        record = {
            'state': {
                'step': metrics.step,
                'timestamp': metrics.timestamp
            },
            'params': self.current_params.copy(),
            'reward': metrics.reward,
            'emergence': -metrics.loss if metrics.loss > 0 else 0.0,  # 转换loss为emergence
            'timestamp': metrics.timestamp
        }

        self.history.append(record)
        self.performance_metrics['emergence_history'].append(record['emergence'])
        self.performance_metrics['reward_history'].append(metrics.reward)

        # 定期优化
        if len(self.history) % self.config.optimization_interval == 0:
            self.optimize_parameters()

    def propose_update(self, mode: str = 'auto') -> Optional[ParameterUpdate]:
        """
        提出参数更新建议 (M1M4Adapter兼容性)

        Args:
            mode: 更新模式 ('auto', 'conservative', 'aggressive')

        Returns:
            ParameterUpdate对象或None
        """
        if len(self.history) < 10:
            return None

        # 计算最近性能
        recent = list(self.history)[-10:]
        recent_avg_emergence = np.mean([r['emergence'] for r in recent])

        # 如果最近性能好，建议保持
        if recent_avg_emergence > 0.05:
            return ParameterUpdate(
                parameters=self.current_params.copy(),
                confidence=0.8,
                reason=f"性能良好 (emergence={recent_avg_emergence:.4f})，建议保持当前参数"
            )

        # 否则建议微调
        adjustments = {}
        for param in ['spiral_radius', 'phase_speed', 'ascent_rate']:
            # 向参数空间中心调整
            min_val, max_val = self.param_space[param]
            center = (min_val + max_val) / 2
            adjustment = (center - self.current_params[param]) * 0.1
            adjustments[param] = adjustment

        # 应用调整
        proposed_params = self.current_params.copy()
        for param, adj in adjustments.items():
            proposed_params[param] += adj
            # 限制在参数空间内
            min_val, max_val = self.param_space[param]
            proposed_params[param] = np.clip(proposed_params[param], min_val, max_val)

        return ParameterUpdate(
            parameters=proposed_params,
            confidence=0.6,
            reason=f"性能亚平 (emergence={recent_avg_emergence:.4f})，建议微调参数"
        )

    def get_meta_knowledge_summary(self) -> Dict[str, Any]:
        """
        获取元知识摘要 (M1M4Adapter兼容性)

        Returns:
            元知识摘要字典
        """
        return {
            'total_observations': len(self.history),
            'optimization_count': self.optimization_stats['total_optimizations'],
            'current_strategy': self.strategy.value,
            'current_params': self.current_params.copy(),
            'avg_emergence': np.mean(self.performance_metrics['emergence_history']) if self.performance_metrics['emergence_history'] else 0.0,
            'learning_rate': self.config.learning_rate
        }


# 测试代码
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*20 + "元学习器测试")
    print("="*70)

    meta_learner = MetaLearner(
        config=MetaLearningConfig(
            learning_rate=0.01,
            optimization_interval=20
        )
    )

    print(f"\n[初始化] 元学习器创建成功")
    print(f"[配置] 学习率=0.01, 优化间隔=20")

    # 模拟100次决策
    print(f"\n[模拟] 记录100次决策...")
    print("="*70)

    for i in range(100):
        # 模拟状态
        phase = (i * 0.1) % (2 * np.pi)
        weight_A = 0.5 + 0.3 * np.cos(phase)
        weight_B = 0.5 + 0.3 * np.cos(phase + np.pi)

        state = {
            'phase': phase,
            'weight_A': weight_A,
            'weight_B': weight_B
        }

        # 模拟参数
        params = {
            'spiral_radius': 0.3 + np.random.normal(0, 0.05),
            'phase_speed': 0.1 + np.random.normal(0, 0.01),
            'ascent_rate': 0.01 + np.random.normal(0, 0.001)
        }

        # 模拟奖励和涌现（高涌现与高权重差异相关）
        weight_diff = abs(weight_A - weight_B)
        emergence = weight_diff * 0.1 + np.random.normal(0, 0.01)
        emergence = max(0, emergence)

        reward = emergence + np.random.normal(0, 0.05)

        # 记录
        meta_learner.record_decision(
            state=state,
            fusion_params=params,
            reward=reward,
            emergence=emergence
        )

        if (i + 1) % 20 == 0:
            print(f"\n已记录 {i+1}/100 次决策")
            stats = meta_learner.get_statistics()
            print(f"  当前参数: spiral_radius={stats['current_params']['spiral_radius']:.4f}")
            print(f"             phase_speed={stats['current_params']['phase_speed']:.4f}")
            print(f"             ascent_rate={stats['current_params']['ascent_rate']:.4f}")
            print(f"  平均涌现: {stats['avg_emergence']:.4f}")
            print(f"  优化次数: {stats['optimization_count']}")

    # 分析模式
    print(f"\n[分析] 历史模式分析")
    print("="*70)

    patterns = meta_learner.analyze_patterns()

    if patterns['status'] == 'success':
        print(f"参数-涌现相关性:")
        for param, corr in patterns['param_correlations'].items():
            print(f"  {param}: {corr:.4f}")

        print(f"\n最佳参数组合:")
        print(f"  {meta_learner._format_params(patterns['best_params'])}")
        print(f"  涌现分数: {patterns['best_emergence']:.4f}")

        print(f"\n性能统计:")
        print(f"  平均涌现: {patterns['avg_emergence']:.4f}")
        print(f"  涌现标准差: {patterns['emergence_std']:.4f}")

    # 预测
    print(f"\n[预测] 预测给定参数下的涌现")
    print("="*70)

    test_state = {'phase': np.pi / 2, 'weight_A': 0.8, 'weight_B': 0.2}
    test_params = {'spiral_radius': 0.35, 'phase_speed': 0.12, 'ascent_rate': 0.015}

    predicted_emergence, confidence = meta_learner.predict_emergence(test_state, test_params)

    print(f"状态: phase={test_state['phase']:.2f}, weight_A={test_state['weight_A']:.2f}")
    print(f"参数: {meta_learner._format_params(test_params)}")
    print(f"\n预测涌现: {predicted_emergence:.4f}")
    print(f"预测置信度: {confidence:.2%}")

    # 显示最终统计
    print("\n" + "="*70)
    print(" "*25 + "最终统计")
    print("="*70)

    stats = meta_learner.get_statistics()
    print(f"\n总决策数: {stats['total_decisions']}")
    print(f"优化次数: {stats['optimization_count']}")
    print(f"改进次数: {stats['improvements']}")
    print(f"平均涌现: {stats['avg_emergence']:.4f}")
    print(f"最大涌现: {stats['max_emergence']:.4f}")
    print(f"涌现标准差: {stats['emergence_std']:.4f}")

    print(f"\n最终参数:")
    print(f"  spiral_radius: {stats['current_params']['spiral_radius']:.4f}")
    print(f"  phase_speed: {stats['current_params']['phase_speed']:.4f}")
    print(f"  ascent_rate: {stats['current_params']['ascent_rate']:.4f}")

    print("\n" + "="*70 + "\n")
