#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
互补区域分析器 (Complementary Region Analyzer)
识别系统A和系统B各自擅长的状态空间区域

核心理念：
- 不是总是融合，而是识别"谁更擅长当前场景"
- 建立状态→优势系统的映射
- 动态学习互补性模式

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-13
版本：v1.0 - 系统纠偏版本
"""

import numpy as np
import torch
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class SystemPreference(Enum):
    """系统偏好"""
    PREFER_A = "prefer_A"      # 偏好系统A
    PREFER_B = "prefer_B"      # 偏好系统B
    NEUTRAL = "neutral"        # 无明显偏好
    FUSE = "fuse"             # 需要融合


@dataclass
class ComplementaryAnalysis:
    """互补性分析结果"""
    preference: SystemPreference
    confidence: float
    reason: str
    state_features: Dict[str, float]
    historical_performance: Dict[str, float]


class ComplementaryAnalyzer:
    """
    互补区域分析器
    
    功能：
    1. 追踪A和B在不同状态下的表现
    2. 识别互补模式（A擅长什么场景，B擅长什么场景）
    3. 提供决策建议：是融合还是直接选择某个系统
    
    实现策略：
    - 基于历史表现的统计分析
    - 状态特征聚类
    - 在线学习更新
    
    🔧 [2026-01-18] 关键修复：置信度归一化
    - 系统A和B使用不同的置信度计算方式
    - 必须归一化后才能公平比较
    """
    
    def __init__(
        self,
        state_dim: int = 64,
        window_size: int = 100,  # 历史窗口大小
        min_samples: int = 10    # 最少样本数
    ):
        self.state_dim = state_dim
        self.window_size = window_size
        self.min_samples = min_samples

        # 历史表现追踪
        self.performance_history = {
            'A': [],  # (state_hash, reward)
            'B': [],  # (state_hash, reward)
        }

        # 状态特征→优势系统映射
        self.state_preference_map: Dict[str, SystemPreference] = {}

        # 统计数据
        self.stats = {
            'A_better_count': 0,
            'B_better_count': 0,
            'neutral_count': 0,
            'total_decisions': 0
        }

        # 🆕 平衡追踪：防止单一系统过度偏好
        self.recent_selections = []  # 最近20次选择
        self.balance_threshold = 0.55  # 单系统最大占比（55%，严格平衡）
        self.neutral_target = 0.30  # 目标中性/融合比例（30%）
        
        # 🔧 [2026-01-18] 置信度校准：使用基于系统特性的初始估计
        # 系统A (TheSeed): 使用 1-uncertainty，通常产生高置信度 (~0.7-0.8)
        # 系统B (Fractal): 使用 goal_score，通常产生较低置信度 (~0.4-0.5)
        self._confidence_stats = {
            'A': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0, 'mean': 0.75, 'std': 0.10},
            'B': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0, 'mean': 0.45, 'std': 0.10}
        }
        # 🔧 在统计样本不足时强制融合的最小样本数
        self._min_samples_for_preference = 20  # 前20次决策强制中立/融合

        logger.info(f"[互补分析] 初始化完成")
        logger.info(f"[互补分析] 窗口大小={window_size}, 最少样本={min_samples}")
        logger.info(f"[互补分析] 平衡阈值={self.balance_threshold*100}%, 中性目标={self.neutral_target*100}%")
        logger.info(f"[互补分析] 🔧 置信度归一化已启用 - 解决系统A/B不可比问题")
    
    def analyze(
        self,
        state: np.ndarray,
        result_A: Optional[Dict[str, Any]],
        result_B: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> ComplementaryAnalysis:
        """
        分析当前状态下的互补性
        
        Args:
            state: 当前状态
            result_A: 系统A的结果
            result_B: 系统B的结果
            context: 额外上下文
            
        Returns:
            互补性分析结果
        """
        context = context or {}
        
        # 处理单系统情况
        if result_A is None:
            return self._single_system_analysis('B', result_B, state)
        if result_B is None:
            return self._single_system_analysis('A', result_A, state)
        
        # 提取状态特征
        state_features = self._extract_state_features(state)
        state_hash = self._hash_state(state_features)
        
        # 查询历史表现
        historical_perf = self._query_historical_performance(state_hash)
        
        # 比较当前置信度
        conf_A = result_A.get('confidence', 0.5)
        conf_B = result_B.get('confidence', 0.5)
        
        # 🔧 [2026-01-18] 关键修复：更新置信度统计并归一化
        self._update_confidence_stats('A', conf_A)
        self._update_confidence_stats('B', conf_B)
        norm_A = self._normalize_confidence('A', conf_A)
        norm_B = self._normalize_confidence('B', conf_B)
        
        # 综合判断（使用归一化后的置信度）
        preference, confidence, reason = self._determine_preference(
            norm_A, norm_B, historical_perf, state_features,
            raw_conf_A=conf_A, raw_conf_B=conf_B  # 传递原始值用于日志
        )

        # 更新统计
        self.stats['total_decisions'] += 1
        if preference == SystemPreference.PREFER_A:
            self.stats['A_better_count'] += 1
            # 🆕 追踪最近选择（用于平衡）
            self.recent_selections.append('A')
            if len(self.recent_selections) > 20:
                self.recent_selections.pop(0)
        elif preference == SystemPreference.PREFER_B:
            self.stats['B_better_count'] += 1
            # 🆕 追踪最近选择（用于平衡）
            self.recent_selections.append('B')
            if len(self.recent_selections) > 20:
                self.recent_selections.pop(0)
        elif preference == SystemPreference.NEUTRAL:
            self.stats['neutral_count'] += 1
            # 🆕 追踪最近选择（用于平衡）
            self.recent_selections.append('NEUTRAL')
            if len(self.recent_selections) > 20:
                self.recent_selections.pop(0)
        elif preference == SystemPreference.FUSE:
            self.stats['neutral_count'] += 1
            # 🆕 FUSE也计入NEUTRAL（因为评估系统把FUSE当作中性处理）
            self.recent_selections.append('NEUTRAL')
            if len(self.recent_selections) > 20:
                self.recent_selections.pop(0)
        
        return ComplementaryAnalysis(
            preference=preference,
            confidence=confidence,
            reason=reason,
            state_features=state_features,
            historical_performance=historical_perf
        )
    
    def update_performance(
        self,
        state: np.ndarray,
        system: str,
        reward: float
    ):
        """更新系统表现"""
        state_features = self._extract_state_features(state)
        state_hash = self._hash_state(state_features)
        
        if system in self.performance_history:
            self.performance_history[system].append((state_hash, reward))
            
            # 维护窗口大小
            if len(self.performance_history[system]) > self.window_size:
                self.performance_history[system].pop(0)
    
    def _update_confidence_stats(self, system: str, confidence: float):
        """
        🆕 [2026-01-18] 在线更新置信度统计
        用于计算每个系统的置信度均值和标准差
        """
        stats = self._confidence_stats[system]
        stats['sum'] += confidence
        stats['sum_sq'] += confidence ** 2
        stats['count'] += 1
        
        n = stats['count']
        if n >= 5:  # 至少5个样本才更新统计
            stats['mean'] = stats['sum'] / n
            variance = (stats['sum_sq'] / n) - (stats['mean'] ** 2)
            stats['std'] = max(0.05, variance ** 0.5)  # 最小标准差0.05
    
    def _normalize_confidence(self, system: str, confidence: float) -> float:
        """
        🆕 [2026-01-18] 置信度归一化（Z-score标准化后映射到[0,1]）
        
        问题：系统A使用 1-uncertainty 计算置信度（通常~0.7）
              系统B使用 goal_score 计算置信度（通常~0.47）
              两者尺度不同，无法直接比较
        
        解决：使用在线学习的均值和标准差进行Z-score归一化
              然后用sigmoid映射到[0,1]，使两者可比
        """
        stats = self._confidence_stats[system]
        
        # Z-score: (x - mean) / std
        z_score = (confidence - stats['mean']) / stats['std']
        
        # Sigmoid映射到[0,1]：1 / (1 + exp(-z))
        # 使用较平缓的sigmoid（除以2）避免过度极化
        normalized = 1.0 / (1.0 + np.exp(-z_score / 2))
        
        return float(normalized)

    def _extract_state_features(self, state: np.ndarray) -> Dict[str, float]:
        """提取状态特征
        
        🆕 [2026-01-17] P0修复：添加输入类型检查
        """
        # 🆕 输入标准化
        if isinstance(state, dict):
            # 从字典提取数值
            values = []
            for v in state.values():
                if isinstance(v, (int, float)):
                    values.append(float(v))
                elif isinstance(v, (list, tuple)):
                    values.extend([float(x) for x in v if isinstance(x, (int, float))])
            state = np.array(values if values else [0.0], dtype=np.float32)
        elif not isinstance(state, np.ndarray):
            state = np.array([state] if isinstance(state, (int, float)) else [0.0], dtype=np.float32)
        
        # 确保是一维数组
        state = np.atleast_1d(state.flatten())
        
        # 简单特征提取：均值、方差、最大最小值等
        features = {
            'mean': float(np.mean(state)),
            'std': float(np.std(state)),
            'max': float(np.max(state)),
            'min': float(np.min(state)),
            'norm': float(np.linalg.norm(state))
        }
        return features
    
    def _hash_state(self, features: Dict[str, float]) -> str:
        """状态特征哈希（用于索引）"""
        # 离散化特征值
        discretized = {
            k: round(v, 2) for k, v in features.items()
        }
        return str(discretized)
    
    def _query_historical_performance(self, state_hash: str) -> Dict[str, float]:
        """查询历史表现"""
        perf = {'A': 0.0, 'B': 0.0, 'count_A': 0, 'count_B': 0}
        
        # 统计相似状态下的表现
        for system in ['A', 'B']:
            similar_rewards = [
                reward for s_hash, reward in self.performance_history[system]
                if s_hash == state_hash  # 精确匹配（可改为相似度匹配）
            ]
            if similar_rewards:
                perf[system] = np.mean(similar_rewards)
                perf[f'count_{system}'] = len(similar_rewards)
        
        return perf
    
    def _check_balance_needed(self) -> Optional[str]:
        """检查是否需要强制平衡

        Returns: 'A' if A needs more selection, 'B' if B needs more, 'NEUTRAL' if need more neutral, None if balanced
        """
        if len(self.recent_selections) < 10:
            return None  # 样本太少，不需要平衡

        a_count = self.recent_selections.count('A')
        b_count = self.recent_selections.count('B')
        neutral_count = self.recent_selections.count('NEUTRAL')
        total = len(self.recent_selections)

        a_rate = a_count / total
        b_rate = b_count / total
        neutral_rate = neutral_count / total

        # 优先级1：检查是否需要更多NEUTRAL（目标30%）
        if neutral_rate < 0.25:  # 如果中性<25%，强制中性（更激进）
            return 'NEUTRAL'

        # 优先级2：检查A/B平衡（防止某系统>60%）
        if a_rate > self.balance_threshold:
            return 'B'  # A太多，强制选B
        elif b_rate > self.balance_threshold:
            return 'A'  # B太多，强制选A

        return None  # 平衡良好

    def _determine_preference(
        self,
        conf_A: float,
        conf_B: float,
        historical_perf: Dict[str, float],
        state_features: Dict[str, float],
        raw_conf_A: float = None,
        raw_conf_B: float = None
    ) -> Tuple[SystemPreference, float, str]:
        """综合判断偏好
        
        🔧 [2026-01-18] 关键修复：conf_A/conf_B 现在是归一化后的值
        raw_conf_A/raw_conf_B 是原始值，用于日志

        修复版本2：添加平衡机制，防止单一系统过度偏好
        修复版本3：样本不足时强制融合，避免早期偏差
        """
        
        # 🔧 [2026-01-18] 策略-1：样本不足时强制融合（最高优先级）
        total_samples = self._confidence_stats['A']['count'] + self._confidence_stats['B']['count']
        if total_samples < self._min_samples_for_preference:
            return (
                SystemPreference.FUSE,
                0.5,
                f"样本积累中 ({total_samples}/{self._min_samples_for_preference}), 强制融合以收集数据"
            )

        # 🆕 策略0：平衡检查（最高优先级）
        balance_needed = self._check_balance_needed()
        if balance_needed:
            if balance_needed == 'NEUTRAL':
                return (
                    SystemPreference.NEUTRAL,
                    0.6,
                    f"平衡性选择: NEUTRAL (提升中性比例)"
                )
            return (
                SystemPreference.PREFER_A if balance_needed == 'A' else SystemPreference.PREFER_B,
                0.6,
                f"平衡性选择: {balance_needed} (防止单一系统过度偏好)"
            )

        # 🆕 优化：降低历史表现样本要求（10→5）
        min_samples_needed = 5

        # 策略1：历史表现主导（如果有足够样本）
        if historical_perf['count_A'] >= min_samples_needed and historical_perf['count_B'] >= min_samples_needed:
            perf_diff = historical_perf['A'] - historical_perf['B']

            # 🆕 优化：降低差异阈值（0.1→0.05）
            if perf_diff > 0.05:  # A明显更好
                return (
                    SystemPreference.PREFER_A,
                    0.8,
                    f"历史表现: A={historical_perf['A']:.3f} > B={historical_perf['B']:.3f}"
                )
            elif perf_diff < -0.05:  # B明显更好
                return (
                    SystemPreference.PREFER_B,
                    0.8,
                    f"历史表现: B={historical_perf['B']:.3f} > A={historical_perf['A']:.3f}"
                )

        # 策略2：当前置信度主导（无足够历史数据时）
        # 🔧 [2026-01-18] conf_A/conf_B 现在是归一化后的值，可以公平比较
        conf_diff = conf_A - conf_B
        
        # 用于日志显示的原始值
        raw_A = raw_conf_A if raw_conf_A is not None else conf_A
        raw_B = raw_conf_B if raw_conf_B is not None else conf_B

        # 🔧 [2026-01-17] 提高阈值(0.08→0.15)，增加融合机会，减少单系统偏好
        if conf_diff > 0.15:  # A显著更自信（归一化后）
            return (
                SystemPreference.PREFER_A,
                0.7,
                f"归一化置信度: A={conf_A:.3f} >> B={conf_B:.3f} (原始: A={raw_A:.3f}, B={raw_B:.3f})"
            )
        elif conf_diff < -0.15:  # B显著更自信（归一化后）
            return (
                SystemPreference.PREFER_B,
                0.7,
                f"归一化置信度: B={conf_B:.3f} >> A={conf_A:.3f} (原始: A={raw_A:.3f}, B={raw_B:.3f})"
            )

        # 策略3：需要融合（两者接近）
        # 🔧 [2026-01-17] 扩大融合区间(0.03→0.10)，差异在±15%内都建议融合
        if abs(conf_diff) < 0.10:
            return (
                SystemPreference.FUSE,
                0.6,
                f"归一化后接近 (A={conf_A:.3f}, B={conf_B:.3f}), 建议融合"
            )

        # 🆕 策略4：强制探索（早期决策）
        total_samples = historical_perf['count_A'] + historical_perf['count_B']
        if total_samples < 10:  # 前10次决策强制探索
            # 使用随机选择促进探索
            import random
            if random.random() < 0.5:
                return (
                    SystemPreference.PREFER_A,
                    0.5,
                    f"探索性选择: A (早期决策#{total_samples})"
                )
            else:
                return (
                    SystemPreference.PREFER_B,
                    0.5,
                    f"探索性选择: B (早期决策#{total_samples})"
                )

        # 默认：中性
        return (
            SystemPreference.NEUTRAL,
            0.5,
            f"无明显偏好 (A={conf_A:.3f}, B={conf_B:.3f})"
        )
    
    def _single_system_analysis(
        self,
        system: str,
        result: Dict[str, Any],
        state: np.ndarray
    ) -> ComplementaryAnalysis:
        """单系统分析"""
        preference = SystemPreference.PREFER_A if system == 'A' else SystemPreference.PREFER_B
        confidence = result.get('confidence', 0.5)
        
        state_features = self._extract_state_features(state)
        
        return ComplementaryAnalysis(
            preference=preference,
            confidence=confidence,
            reason=f"仅系统{system}可用",
            state_features=state_features,
            historical_performance={}
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self.stats['total_decisions']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'A_better_ratio': self.stats['A_better_count'] / total,
            'B_better_ratio': self.stats['B_better_count'] / total,
            'neutral_ratio': self.stats['neutral_count'] / total
        }
    
    def should_fuse(self, analysis: ComplementaryAnalysis) -> bool:
        """判断是否应该融合"""
        return analysis.preference in [SystemPreference.NEUTRAL, SystemPreference.FUSE]
    
    def get_preferred_system(self, analysis: ComplementaryAnalysis) -> Optional[str]:
        """获取偏好系统"""
        if analysis.preference == SystemPreference.PREFER_A:
            return 'A'
        elif analysis.preference == SystemPreference.PREFER_B:
            return 'B'
        return None
