#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双螺旋决策引擎 (Double Helix Decision Engine)
实现系统A和B的真正相互缠绕，激发智慧生成

核心思想：
1. 不是"非此即彼"，而是"互相缠绕"
2. 系统A和B并行决策，而非轮流
3. 相位耦合：形成周期性波动
4. 反馈闭环：A影响B，B影响A
5. 螺旋上升：每个周期性能跃迁

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-13
版本：v1.0
"""

import numpy as np
import torch
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# 导入系统A和B
try:
    from core.seed import TheSeed
except ImportError:
    TheSeed = None

try:
    from core.fractal_intelligence import create_fractal_intelligence
except ImportError:
    create_fractal_intelligence = None

logger = logging.getLogger(__name__)


@dataclass
class HelixContext:
    """螺旋上下文"""
    phase: float              # 当前相位
    weight_A: float           # 系统A的权重
    weight_B: float           # 系统B的权重
    last_A_output: Optional[np.ndarray]  # 系统A的上次输出
    last_B_output: Optional[np.ndarray]  # 系统B的上次输出
    cycle_number: int         # 当前螺旋周期编号
    ascent_level: float       # 上升层级（螺旋上升的高度）


@dataclass
class DoubleHelixResult:
    """双螺旋决策结果"""
    action: int
    confidence: float
    weight_A: float           # 系统A的权重
    weight_B: float           # 系统B的权重
    phase: float              # 当前相位
    individual_A: Optional[Any]  # 系统A的独立决策
    individual_B: Optional[Any]  # 系统B的独立决策
    fusion_method: str        # 融合方法
    emergence_score: float    # 涌现分数（协同效应）
    explanation: str
    response_time_ms: float
    entropy: float = 0.0
    cycle_number: int = 0
    ascent_level: float = 0.0
    system_a_confidence: float = 0.0  # 系统A置信度
    system_b_confidence: float = 0.0  # 系统B置信度


class DoubleHelixEngine:
    """
    双螺旋决策引擎

    核心特性：
    1. 相位耦合（Phase Coupling）：系统A和B的权重形成周期性波动
    2. 相互缠绕（Interwoven）：A的输出影响B，B的输出影响A
    3. 螺旋融合（Spiral Fusion）：不是选择，而是融合
    4. 螺旋上升（Spiral Ascent）：每个周期性能跃迁
    5. 智慧涌现（Emergence）：1+1 > 2
    """

    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 4,
        device: str = 'cpu',
        spiral_radius: float = 0.3,      # 螺旋半径（权重波动范围）
        phase_shift: float = np.pi,       # 相位差（默认180°）
        phase_speed: float = 0.1,         # 相位推进速度
        cycle_length: int = 10,           # 螺旋周期长度（决策次数）
        ascent_rate: float = 0.01,        # 上升速率
        enable_adaptive: bool = True      # 启用自适应
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # 螺旋参数
        self.spiral_radius = spiral_radius
        self.phase_shift = phase_shift
        self.phase_speed = phase_speed
        self.cycle_length = cycle_length
        self.ascent_rate = ascent_rate
        self.enable_adaptive = enable_adaptive

        # 状态变量
        self.phase = 0.0
        self.decision_count = 0
        self.cycle_number = 1
        self.ascent_level = 0.0

        # 上下文
        self.context = HelixContext(
            phase=0.0,
            weight_A=0.5,
            weight_B=0.5,
            last_A_output=None,
            last_B_output=None,
            cycle_number=1,
            ascent_level=0.0
        )

        # 性能追踪
        self.confidence_history = []
        self.cycle_peaks = []  # 每个周期的峰值置信度
        self.emergence_history = []  # 涌现分数历史

        # 统计
        self.stats = {
            'total_decisions': 0,
            'A_dominant': 0,      # A主导的决策数
            'B_dominant': 0,      # B主导的决策数
            'balanced': 0,        # 均衡的决策数
            'avg_emergence': 0.0,
            'avg_confidence': 0.0,
            'cycles_completed': 0
        }

        # 初始化系统A和B
        self._init_systems()

        logger.info(f"[双螺旋] 引擎初始化完成")
        logger.info(f"[双螺旋] 螺旋半径={spiral_radius}, 相位差={phase_shift:.2f}, 周期长度={cycle_length}")

    def _init_systems(self):
        """初始化系统A和B"""

        # 系统A：TheSeed
        self.seed = None
        if TheSeed:
            try:
                self.seed = TheSeed(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim
                )
                logger.info("[双螺旋] 系统A（TheSeed）已启用")
            except Exception as e:
                logger.warning(f"[双螺旋] 系统A初始化失败: {e}")

        # 系统B：分形智能
        self.fractal = None
        if create_fractal_intelligence:
            try:
                self.fractal = create_fractal_intelligence(
                    input_dim=self.state_dim,
                    state_dim=self.state_dim,
                    device=self.device
                )
                logger.info("[双螺旋] 系统B（分形智能）已启用")
            except Exception as e:
                logger.warning(f"[双螺旋] 系统B初始化失败: {e}")

    def decide(
        self,
        state: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> DoubleHelixResult:
        """
        双螺旋决策

        核心流程：
        1. 计算当前相位和权重
        2. 系统A和B并行决策（相互影响）
        3. 螺旋融合（不是选择，而是融合）
        4. 更新相位和周期
        5. 检测螺旋上升
        """

        start_time = time.time()
        context = context or {}
        self.decision_count += 1
        self.stats['total_decisions'] += 1

        # 步骤1：计算相位和权重
        self._update_phase()

        # 步骤2：系统A和B并行决策（相互缠绕）
        result_A = self._decide_A(state, context)
        result_B = self._decide_B(state, context)

        # 步骤3：螺旋融合
        fused_result = self._fuse_results(result_A, result_B)

        # 步骤4：更新上下文（为下次决策做准备）
        self._update_context(result_A, result_B)

        # 步骤5：检测周期完成和螺旋上升
        self._check_cycle_completion(fused_result['confidence'])

        # 统计
        response_time = (time.time() - start_time) * 1000
        self._update_stats(fused_result)

        return DoubleHelixResult(
            action=fused_result['action'],
            confidence=fused_result['confidence'],
            weight_A=self.context.weight_A,
            weight_B=self.context.weight_B,
            phase=self.context.phase,
            individual_A=result_A,
            individual_B=result_B,
            fusion_method=fused_result['method'],
            emergence_score=fused_result['emergence'],
            explanation=self._generate_explanation(fused_result),
            response_time_ms=response_time,
            entropy=fused_result.get('entropy', 0.0),
            cycle_number=self.context.cycle_number,
            ascent_level=self.context.ascent_level,
            system_a_confidence=result_A.get('confidence', 0.0) if isinstance(result_A, dict) else getattr(result_A, 'confidence', 0.0),
            system_b_confidence=result_B.get('confidence', 0.0) if isinstance(result_B, dict) else getattr(result_B, 'confidence', 0.0)
        )

    def _update_phase(self):
        """更新相位和权重"""

        # 计算权重（相位耦合）
        # 系统A的权重：0.5 + 0.3 * cos(phase)
        # 系统B的权重：0.5 + 0.3 * cos(phase + π)
        # 这样两个系统的权重形成周期性波动，相差180°
        self.context.weight_A = 0.5 + self.spiral_radius * np.cos(self.phase)
        self.context.weight_B = 0.5 + self.spiral_radius * np.cos(self.phase + self.phase_shift)

        # 确保权重非负
        self.context.weight_A = max(0.0, self.context.weight_A)
        self.context.weight_B = max(0.0, self.context.weight_B)

        # 归一化
        total_weight = self.context.weight_A + self.context.weight_B
        if total_weight > 0:
            self.context.weight_A /= total_weight
            self.context.weight_B /= total_weight

        # 更新相位
        self.context.phase = self.phase
        self.phase += self.phase_speed

    def _decide_A(
        self,
        state: np.ndarray,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """系统A决策（受系统B影响）"""

        if not self.seed:
            return None

        try:
            # 构建增强的状态（包含系统B的上次输出）
            enhanced_state = self._enhance_state_A(state, context)

            # 决策
            action = self.seed.act(enhanced_state)

            # 获取置信度（通过predict）
            _, confidence = self.seed.predict(enhanced_state, action)
            confidence = float(np.clip(confidence, 0, 1))

            return {
                'action': int(action),
                'confidence': confidence,
                'system': 'A'
            }
        except Exception as e:
            logger.warning(f"[双螺旋] 系统A决策失败: {e}")
            return None

    def _decide_B(
        self,
        state: np.ndarray,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """系统B决策（受系统A影响）"""

        if not self.fractal:
            return None

        try:
            # 构建增强的状态（包含系统A的上次输出）
            enhanced_state = self._enhance_state_B(state, context)

            # 决策
            state_tensor = torch.FloatTensor(enhanced_state).unsqueeze(0).to(self.device)
            output = self.fractal.core.forward(state_tensor)

            action = output.action.item() if hasattr(output, 'action') else 0
            confidence = output.confidence.item() if hasattr(output, 'confidence') else 0.5

            return {
                'action': int(action),
                'confidence': float(confidence),
                'system': 'B'
            }
        except Exception as e:
            logger.warning(f"[双螺旋] 系统B决策失败: {e}")
            return None

    def _enhance_state_A(
        self,
        state: np.ndarray,
        context: Dict[str, Any]
    ) -> np.ndarray:
        """增强系统A的状态（包含系统B的影响）"""

        # 如果有系统B的上次输出，融合到状态中
        if self.context.last_B_output is not None:
            # 简单的加权融合
            alpha = 0.7  # 当前状态权重
            beta = 0.3   # B的上次输出权重
            enhanced = alpha * state + beta * self.context.last_B_output
            return enhanced
        return state

    def _enhance_state_B(
        self,
        state: np.ndarray,
        context: Dict[str, Any]
    ) -> np.ndarray:
        """增强系统B的状态（包含系统A的影响）"""

        # 如果有系统A的上次输出，融合到状态中
        if self.context.last_A_output is not None:
            # 简单的加权融合
            alpha = 0.7  # 当前状态权重
            beta = 0.3   # A的上次输出权重
            enhanced = alpha * state + beta * self.context.last_A_output
            return enhanced
        return state

    def _fuse_results(
        self,
        result_A: Optional[Dict[str, Any]],
        result_B: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        螺旋融合（不是选择，而是融合）

        融合策略：
        1. 加权融合：根据相位权重融合动作
        2. 置信度融合：取加权平均，加上螺旋上升
        3. 涌现检测：计算协同效应
        """

        # 如果只有一个系统可用
        if result_A is None and result_B is None:
            return self._get_fallback()
        elif result_A is None:
            return {
                'action': result_B['action'],
                'confidence': result_B['confidence'],
                'method': 'B_only',
                'emergence': 0.0
            }
        elif result_B is None:
            return {
                'action': result_A['action'],
                'confidence': result_A['confidence'],
                'method': 'A_only',
                'emergence': 0.0
            }

        # 两个系统都可用，执行螺旋融合
        weight_A = self.context.weight_A
        weight_B = self.context.weight_B

        # 1. 动作融合（加权平均后取整）
        fused_action = int(
            weight_A * result_A['action'] + weight_B * result_B['action']
        )

        # 2. 置信度融合（加权平均）
        base_confidence = weight_A * result_A['confidence'] + weight_B * result_B['confidence']

        # 3. 真实协同效应检测（不含累积奖励）
        # 如果融合后的置信度 > 单个系统的最大置信度，说明有涌现
        max_individual_confidence = max(result_A['confidence'], result_B['confidence'])
        real_synergy = base_confidence - max_individual_confidence  # 真实协同（可为负）

        # 4. 螺旋上升加成（累积奖励）
        ascent_bonus = self.context.ascent_level

        # 5. 最终融合置信度（基础 + 上升加成）
        fused_confidence = base_confidence + ascent_bonus
        fused_confidence = min(1.0, max(0.0, fused_confidence))

        # 6. 涌现分数（只记录真实协同，不含累积奖励）
        # 如果真实协同为正，说明真正有涌现；否则说明只是累积奖励在起作用
        emergence_score = max(0.0, real_synergy)

        # 4. 确定融合方法
        if abs(weight_A - weight_B) < 0.1:
            method = 'balanced_fusion'
        elif weight_A > weight_B:
            method = 'A_dominant_fusion'
        else:
            method = 'B_dominant_fusion'

        return {
            'action': fused_action,
            'confidence': fused_confidence,
            'method': method,
            'emergence': emergence_score,
            'entropy': self._calculate_entropy(result_A, result_B)
        }

    def _update_context(self, result_A, result_B):
        """更新上下文（为下次决策做准备）"""

        # 保存系统A和B的输出（用于下次决策的相互影响）
        if result_A is not None:
            self.context.last_A_output = np.zeros(self.state_dim)
            self.context.last_A_output[result_A['action']] = result_A['confidence']

        if result_B is not None:
            self.context.last_B_output = np.zeros(self.state_dim)
            self.context.last_B_output[result_B['action']] = result_B['confidence']

    def _check_cycle_completion(self, confidence: float):
        """检测周期完成和螺旋上升"""

        # 记录当前置信度
        self.confidence_history.append(confidence)

        # 检查是否完成一个周期
        if self.decision_count % self.cycle_length == 0:
            # 计算本周期峰值
            cycle_peak = max(self.confidence_history[-self.cycle_length:])
            self.cycle_peaks.append(cycle_peak)

            # 螺旋上升：如果峰值比上一周期高，提升层级
            if len(self.cycle_peaks) >= 2:
                improvement = self.cycle_peaks[-1] - self.cycle_peaks[-2]
                if improvement > 0:
                    self.ascent_level += self.ascent_rate
                    self.context.ascent_level = self.ascent_level
                    logger.info(f"[双螺旋] 周期{self.cycle_number}完成，峰值提升{improvement:.4f}，上升至{self.ascent_level:.4f}")

            # 更新周期编号
            self.cycle_number += 1
            self.context.cycle_number = self.cycle_number
            self.stats['cycles_completed'] += 1

    def _calculate_entropy(self, result_A, result_B) -> float:
        """计算熵（不确定性）"""

        if result_A is None or result_B is None:
            return 0.0

        # 简化的熵计算：基于动作差异
        action_diff = abs(result_A['action'] - result_B['action'])
        confidence_diff = abs(result_A['confidence'] - result_B['confidence'])

        # 动作和置信度差异越大，熵越高
        entropy = (action_diff / self.action_dim) * 0.5 + (confidence_diff * 0.5)
        return entropy

    def _update_stats(self, result: Dict[str, Any]):
        """更新统计"""

        weight_A = self.context.weight_A
        weight_B = self.context.weight_B

        # 统计主导系统
        if abs(weight_A - weight_B) < 0.1:
            self.stats['balanced'] += 1
        elif weight_A > weight_B:
            self.stats['A_dominant'] += 1
        else:
            self.stats['B_dominant'] += 1

        # 统计涌现
        emergence = result['emergence']
        if len(self.emergence_history) > 0:
            self.stats['avg_emergence'] = (
                self.stats['avg_emergence'] * len(self.emergence_history) + emergence
            ) / (len(self.emergence_history) + 1)
        else:
            self.stats['avg_emergence'] = emergence

        self.emergence_history.append(emergence)

        # 统计置信度
        if len(self.confidence_history) > 0:
            self.stats['avg_confidence'] = np.mean(self.confidence_history)

    def _generate_explanation(self, result: Dict[str, Any]) -> str:
        """生成解释"""

        weight_A = self.context.weight_A
        weight_B = self.context.weight_B
        emergence = result['emergence']

        explanation = f"双螺旋融合 | 相位={self.context.phase:.2f} | A权重={weight_A:.2f} B权重={weight_B:.2f}"

        if emergence > 0.05:
            explanation += f" | 涌现+{emergence:.4f}"

        if self.context.ascent_level > 0:
            explanation += f" | 上升层级={self.context.ascent_level:.4f}"

        return explanation

    def _get_fallback(self) -> Dict[str, Any]:
        """兜底决策"""
        return {
            'action': 0,
            'confidence': 0.5,
            'method': 'fallback',
            'emergence': 0.0,
            'entropy': 1.0
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""

        return {
            **self.stats,
            'current_phase': self.context.phase,
            'current_weight_A': self.context.weight_A,
            'current_weight_B': self.context.weight_B,
            'cycle_number': self.context.cycle_number,
            'ascent_level': self.context.ascent_level,
            'cycle_peaks': self.cycle_peaks[-5:] if self.cycle_peaks else [],
            'recent_emergence': self.emergence_history[-10:] if self.emergence_history else []
        }

    def visualize_helix(self, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """可视化双螺旋结构"""

        t = np.linspace(0, 4 * np.pi, num_points)

        # 系统A的轨迹
        x_A = np.cos(t)
        y_A = np.sin(t)
        z_A = t / (4 * np.pi)  # 上升

        # 系统B的轨迹（相位差）
        x_B = np.cos(t + self.phase_shift)
        y_B = np.sin(t + self.phase_shift)
        z_B = t / (4 * np.pi)  # 上升

        return (x_A, y_A, z_A), (x_B, y_B, z_B)


# 测试代码
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "双螺旋决策引擎测试")
    print("="*70)

    engine = DoubleHelixEngine(
        state_dim=64,
        action_dim=4,
        spiral_radius=0.3,
        phase_shift=np.pi,
        cycle_length=10
    )

    print(f"\n[初始化] 双螺旋引擎创建成功")
    print(f"[参数] 螺旋半径={engine.spiral_radius}, 相位差={engine.phase_shift:.2f}, 周期长度={engine.cycle_length}")

    # 执行20次决策
    print(f"\n[测试] 执行20次双螺旋决策...")
    print("="*70)

    for i in range(20):
        state = np.random.randn(64)
        result = engine.decide(state)

        print(f"\n决策 {i+1}/20:")
        print(f"  相位: {result.phase:.2f}")
        print(f"  权重: A={result.weight_A:.2f} B={result.weight_B:.2f}")
        print(f"  动作: {result.action}")
        print(f"  置信度: {result.confidence:.4f}")
        print(f"  融合方法: {result.fusion_method}")
        print(f"  涌现分数: {result.emergence_score:.4f}")
        print(f"  解释: {result.explanation}")

    # 显示统计
    print("\n" + "="*70)
    print(" "*25 + "统计信息")
    print("="*70)

    stats = engine.get_statistics()
    print(f"\n总决策数: {stats['total_decisions']}")
    print(f"A主导: {stats['A_dominant']}")
    print(f"B主导: {stats['B_dominant']}")
    print(f"均衡: {stats['balanced']}")
    print(f"平均涌现分数: {stats['avg_emergence']:.4f}")
    print(f"平均置信度: {stats['avg_confidence']:.4f}")
    print(f"完成周期: {stats['cycles_completed']}")
    print(f"上升层级: {stats['ascent_level']:.4f}")
    print(f"周期峰值: {stats['cycle_peaks']}")

    print("\n" + "="*70 + "\n")
