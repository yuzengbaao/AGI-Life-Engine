#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creative Fusion Enhancement - 创造性融合增强
==============================================

功能：
1. 涌现行为检测
2. 自适应融合参数
3. 协同进化效果评估

作者：AGI Self-Improvement Module
创建日期：2026-02-04
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmergenceEvent:
    """涌现事件"""
    event_id: str
    timestamp: float
    fused_output: Any
    individual_A: Any
    individual_B: Any
    emergence_score: float  # 涌现评分
    improvement_A: float  # 相对于A的提升
    improvement_B: float  # 相对于B的提升
    novelty_contribution: float  # 新颖性贡献
    description: str


@dataclass
class FusionParameters:
    """融合参数"""
    interaction_strength: float  # 交互强度 [0, 1]
    fusion_method: str  # 融合方法（creative/balanced/conservative）
    complementarity_threshold: float  # 互补性阈值
    emergence_threshold: float  # 涌现阈值


class EmergenceDetector:
    """
    涌现行为检测器

    检测融合输出是否显著优于任一子系统
    """

    def __init__(self, emergence_threshold: float = 0.2):
        """
        初始化涌现检测器

        Args:
            emergence_threshold: 涌现阈值（默认0.2，即提升20%）
        """
        self.emergence_threshold = emergence_threshold
        self.emergence_history: List[EmergenceEvent] = []

        logger.info(f"[涌现检测] 初始化: 阈值={emergence_threshold}")

    def detect_emergence(
        self,
        fused_output: Any,
        individual_A: Any,
        individual_B: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        检测涌现行为

        Args:
            fused_output: 融合输出
            individual_A: 系统A的输出
            individual_B: 系统B的输出
            context: 上下文信息

        Returns:
            (是否涌现, 涌现评分, 详细指标)
        """
        if context is None:
            context = {}

        # 评估输出质量
        score_fused = self._evaluate_output(fused_output, context)
        score_A = self._evaluate_output(individual_A, context)
        score_B = self._evaluate_output(individual_B, context)

        # 计算提升
        improvement_A = score_fused - score_A
        improvement_B = score_fused - score_B
        max_improvement = max(improvement_A, improvement_B)

        # 涌现判断
        is_emergent = max_improvement > self.emergence_threshold

        # 新颖性贡献（融合输出相对于两个系统的差异）
        novelty_contribution = self._calculate_novelty_contribution(
            fused_output, individual_A, individual_B, context
        )

        metrics = {
            'score_fused': score_fused,
            'score_A': score_A,
            'score_B': score_B,
            'improvement_A': improvement_A,
            'improvement_B': improvement_B,
            'max_improvement': max_improvement,
            'novelty_contribution': novelty_contribution
        }

        # 记录涌现事件
        if is_emergent:
            event = EmergenceEvent(
                event_id=f"emergence_{int(time.time() * 1000)}",
                timestamp=time.time(),
                fused_output=fused_output,
                individual_A=individual_A,
                individual_B=individual_B,
                emergence_score=max_improvement,
                improvement_A=improvement_A,
                improvement_B=improvement_B,
                novelty_contribution=novelty_contribution,
                description=f"融合输出优于系统A {improvement_A:.2%}, 优于系统B {improvement_B:.2%}"
            )
            self.emergence_history.append(event)

            logger.info(
                f"[涌现检测] 发现涌现！提升={max_improvement:.2%}, "
                f"新颖性贡献={novelty_contribution:.2%}"
            )

        return is_emergent, max_improvement, metrics

    def _evaluate_output(self, output: Any, context: Dict) -> float:
        """
        评估输出质量

        Args:
            output: 输出（可以是字典、字符串、数值等）
            context: 上下文

        Returns:
            质量评分 [0, 1]
        """
        # 根据输出类型选择评估方法
        if isinstance(output, dict):
            return self._evaluate_dict(output, context)
        elif isinstance(output, (int, float)):
            return self._evaluate_numeric(output, context)
        elif isinstance(output, str):
            return self._evaluate_text(output, context)
        elif isinstance(output, np.ndarray):
            return self._evaluate_array(output, context)
        else:
            # 默认评分
            return 0.5

    def _evaluate_dict(self, output: Dict, context: Dict) -> float:
        """评估字典输出"""
        # 检查关键字段
        if 'confidence' in output:
            confidence = float(output['confidence'])
            return np.clip(confidence, 0, 1)

        if 'value_score' in output:
            value = float(output['value_score'])
            return np.clip(value, 0, 1)

        if 'score' in output:
            score = float(output['score'])
            return np.clip(score, 0, 1)

        # 默认：检查字段完整性
        required_fields = context.get('required_fields', [])
        if required_fields:
            present = sum(1 for f in required_fields if f in output)
            return present / len(required_fields)

        return 0.5

    def _evaluate_numeric(self, output: Union[int, float], context: Dict) -> float:
        """评估数值输出"""
        # 归一化到 [0, 1]
        value_range = context.get('value_range', (0, 1))
        min_val, max_val = value_range

        normalized = (output - min_val) / (max_val - min_val + 1e-8)
        return np.clip(normalized, 0, 1)

    def _evaluate_text(self, output: str, context: Dict) -> float:
        """评估文本输出"""
        # 简化版本：基于长度和关键词
        length_score = min(len(output) / 100, 1.0)  # 100字符为满分

        # 关键词匹配
        keywords = context.get('keywords', [])
        keyword_score = 0.0
        if keywords:
            matched = sum(1 for kw in keywords if kw.lower() in output.lower())
            keyword_score = matched / len(keywords)

        return (length_score + keyword_score) / 2

    def _evaluate_array(self, output: np.ndarray, context: Dict) -> float:
        """评估数组输出"""
        # 简化版本：基于范数
        norm = np.linalg.norm(output)
        return np.clip(norm / 10.0, 0, 1)

    def _calculate_novelty_contribution(
        self,
        fused: Any,
        individual_A: Any,
        individual_B: Any,
        context: Dict
    ) -> float:
        """
        计算新颖性贡献

        Args:
            fused: 融合输出
            individual_A: 系统A输出
            individual_B: 系统B输出

        Returns:
            新颖性贡献 [0, 1]
        """
        # 简化版本：比较融合输出与两个系统的差异
        # 使用距离度量

        # 将输出转换为可比较的向量
        vec_fused = self._to_vector(fused)
        vec_A = self._to_vector(individual_A)
        vec_B = self._to_vector(individual_B)

        if vec_fused is None or vec_A is None or vec_B is None:
            return 0.5

        # 计算距离
        dist_A = np.linalg.norm(vec_fused - vec_A)
        dist_B = np.linalg.norm(vec_fused - vec_B)

        # 归一化
        max_dist = np.linalg.norm(vec_A - vec_B) + 1e-8
        novelty = (dist_A + dist_B) / (2 * max_dist)

        return np.clip(novelty, 0, 1)

    def _to_vector(self, data: Any) -> Optional[np.ndarray]:
        """将数据转换为向量"""
        if isinstance(data, np.ndarray):
            return data.flatten()
        elif isinstance(data, (int, float)):
            return np.array([data])
        elif isinstance(data, dict):
            # 提取数值字段
            values = [v for v in data.values() if isinstance(v, (int, float))]
            if values:
                return np.array(values)
        elif isinstance(data, str):
            # 简化版本：字符频率
            char_freq = np.zeros(128)
            for char in data:
                code = ord(char)
                if code < 128:
                    char_freq[code] += 1
            return char_freq

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.emergence_history:
            return {
                'total_detections': 0,
                'emergence_rate': 0.0,
                'avg_emergence_score': 0.0,
                'max_emergence_score': 0.0
            }

        emergence_scores = [e.emergence_score for e in self.emergence_history]

        return {
            'total_detections': len(self.emergence_history),
            'emergence_rate': len(self.emergence_history) / max(len(self.emergence_history) + 100, 1),
            'avg_emergence_score': np.mean(emergence_scores),
            'max_emergence_score': max(emergence_scores),
            'avg_novelty_contribution': np.mean([e.novelty_contribution for e in self.emergence_history])
        }


class AdaptiveFusionEngine:
    """
    自适应融合引擎

    根据系统互补性和历史表现动态调整融合参数
    """

    def __init__(
        self,
        emergence_detector: Optional[EmergenceDetector] = None,
        learning_rate: float = 0.1
    ):
        """
        初始化自适应融合引擎

        Args:
            emergence_detector: 涌现检测器
            learning_rate: 学习率（参数调整速度）
        """
        self.emergence_detector = emergence_detector or EmergenceDetector()
        self.learning_rate = learning_rate

        # 当前融合参数
        self.current_params = FusionParameters(
            interaction_strength=0.3,
            fusion_method='balanced',
            complementarity_threshold=0.5,
            emergence_threshold=0.2
        )

        # 历史记录
        self.fusion_history: List[Dict[str, Any]] = []

        logger.info(f"[自适应融合] 初始化: 学习率={learning_rate}")

    def analyze_complementarity(
        self,
        system_A: Any,
        system_B: Any,
        context: Optional[Dict] = None
    ) -> float:
        """
        分析系统互补性

        Args:
            system_A: 系统A
            system_B: 系统B
            context: 上下文

        Returns:
            互补性评分 [0, 1]
        """
        # 互补性 = 1 - 相似性
        vec_A = self.emergence_detector._to_vector(system_A)
        vec_B = self.emergence_detector._to_vector(system_B)

        if vec_A is None or vec_B is None:
            return 0.5

        # 归一化
        vec_A = vec_A / (np.linalg.norm(vec_A) + 1e-8)
        vec_B = vec_B / (np.linalg.norm(vec_B) + 1e-8)

        # 计算相似度
        similarity = np.dot(vec_A, vec_B)

        # 互补性
        complementarity = 1.0 - similarity

        return np.clip(complementarity, 0, 1)

    def adaptive_fusion(
        self,
        system_A: Any,
        system_B: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, FusionParameters, Dict[str, Any]]:
        """
        自适应融合

        Args:
            system_A: 系统A
            system_B: 系统B
            context: 上下文

        Returns:
            (融合输出, 使用的融合参数, 统计信息)
        """
        if context is None:
            context = {}

        # 1. 分析互补性
        complementarity = self.analyze_complementarity(system_A, system_B, context)

        # 2. 动态调整融合参数
        if complementarity > 0.8:
            # 高互补性 → 强交互 + 创造性融合
            self.current_params.interaction_strength = 0.5
            self.current_params.fusion_method = 'creative'
        elif complementarity > 0.5:
            # 中等互补性 → 中等交互 + 平衡融合
            self.current_params.interaction_strength = 0.3
            self.current_params.fusion_method = 'balanced'
        else:
            # 低互补性 → 弱交互 + 保守融合
            self.current_params.interaction_strength = 0.1
            self.current_params.fusion_method = 'conservative'

        # 3. 执行融合
        fused_output = self._fusion_formula(
            system_A,
            system_B,
            self.current_params.interaction_strength,
            self.current_params.fusion_method
        )

        # 4. 检测涌现
        is_emergent, emergence_score, metrics = self.emergence_detector.detect_emergence(
            fused_output, system_A, system_B, context
        )

        # 5. 记录历史
        record = {
            'timestamp': time.time(),
            'complementarity': complementarity,
            'params': self.current_params,
            'emergence_detected': is_emergent,
            'emergence_score': emergence_score,
            'metrics': metrics
        }
        self.fusion_history.append(record)

        # 6. 自适应调整（基于涌现表现）
        if is_emergent:
            # 如果出现涌现，保持当前参数或略微增强
            self.current_params.interaction_strength = min(
                1.0, self.current_params.interaction_strength + self.learning_rate * 0.1
            )
            logger.info(
                f"[自适应融合] 涌现成功！交互强度提升至 "
                f"{self.current_params.interaction_strength:.2f}"
            )
        else:
            # 如果没有涌现，尝试调整参数
            if emergence_score < 0:
                # 融合效果差，降低交互强度
                self.current_params.interaction_strength = max(
                    0.0, self.current_params.interaction_strength - self.learning_rate * 0.2
                )

        stats = {
            'complementarity': complementarity,
            'interaction_strength': self.current_params.interaction_strength,
            'fusion_method': self.current_params.fusion_method,
            'emergence_detected': is_emergent,
            'emergence_score': emergence_score
        }

        return fused_output, self.current_params, stats

    def _fusion_formula(
        self,
        system_A: Any,
        system_B: Any,
        interaction_strength: float,
        method: str
    ) -> Any:
        """
        融合公式

        Args:
            system_A: 系统A
            system_B: 系统B
            interaction_strength: 交互强度
            method: 融合方法

        Returns:
            融合输出
        """
        # 如果都是数值
        if isinstance(system_A, (int, float)) and isinstance(system_B, (int, float)):
            return self._fuse_numeric(system_A, system_B, interaction_strength, method)

        # 如果都是数组
        if isinstance(system_A, np.ndarray) and isinstance(system_B, np.ndarray):
            return self._fuse_array(system_A, system_B, interaction_strength, method)

        # 如果都是字典
        if isinstance(system_A, dict) and isinstance(system_B, dict):
            return self._fuse_dict(system_A, system_B, interaction_strength, method)

        # 默认：返回系统A（保守）
        return system_A

    def _fuse_numeric(
        self,
        A: float,
        B: float,
        strength: float,
        method: str
    ) -> float:
        """融合数值"""
        if method == 'creative':
            # 创造性：非线性组合
            return A * (1 - strength) + B * strength + 0.1 * A * B
        elif method == 'balanced':
            # 平衡：加权平均
            return A * (1 - strength) + B * strength
        else:  # conservative
            # 保守：偏向A
            return A * (1 - strength * 0.5) + B * strength * 0.5

    def _fuse_array(
        self,
        A: np.ndarray,
        B: np.ndarray,
        strength: float,
        method: str
    ) -> np.ndarray:
        """融合数组"""
        if method == 'creative':
            # 创造性：非线性组合
            return A * (1 - strength) + B * strength + 0.1 * A * B
        elif method == 'balanced':
            # 平衡：加权平均
            return A * (1 - strength) + B * strength
        else:  # conservative
            # 保守：偏向A
            return A * (1 - strength * 0.5) + B * strength * 0.5

    def _fuse_dict(
        self,
        A: Dict,
        B: Dict,
        strength: float,
        method: str
    ) -> Dict:
        """融合字典"""
        result = {}

        # 获取所有键
        all_keys = set(A.keys()) | set(B.keys())

        for key in all_keys:
            if key in A and key in B:
                # 两边都有，融合
                val_A = A[key]
                val_B = B[key]

                if isinstance(val_A, (int, float)) and isinstance(val_B, (int, float)):
                    result[key] = self._fuse_numeric(val_A, val_B, strength, method)
                else:
                    # 非数值，取A
                    result[key] = val_A
            elif key in A:
                result[key] = A[key]
            else:
                result[key] = B[key]

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.fusion_history:
            return {
                'total_fusions': 0,
                'avg_complementarity': 0.0,
                'emergence_rate': 0.0,
                'current_interaction_strength': self.current_params.interaction_strength
            }

        complementarities = [r['complementarity'] for r in self.fusion_history]
        emergences = [r['emergence_detected'] for r in self.fusion_history]

        return {
            'total_fusions': len(self.fusion_history),
            'avg_complementarity': np.mean(complementarities),
            'emergence_rate': np.mean(emergences),
            'current_interaction_strength': self.current_params.interaction_strength,
            'fusion_method_distribution': {
                'creative': sum(1 for r in self.fusion_history if r['params'].fusion_method == 'creative'),
                'balanced': sum(1 for r in self.fusion_history if r['params'].fusion_method == 'balanced'),
                'conservative': sum(1 for r in self.fusion_history if r['params'].fusion_method == 'conservative')
            }
        }


# 全局单例
_global_emergence_detector: Optional[EmergenceDetector] = None
_global_adaptive_fusion: Optional[AdaptiveFusionEngine] = None


def get_emergence_detector() -> EmergenceDetector:
    """获取全局涌现检测器"""
    global _global_emergence_detector
    if _global_emergence_detector is None:
        _global_emergence_detector = EmergenceDetector()
    return _global_emergence_detector


def get_adaptive_fusion_engine() -> AdaptiveFusionEngine:
    """获取全局自适应融合引擎"""
    global _global_adaptive_fusion
    if _global_adaptive_fusion is None:
        _global_adaptive_fusion = AdaptiveFusionEngine()
    return _global_adaptive_fusion
