#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Temperature Controller - 自适应温度控制器
====================================================

功能：
1. 基于上下文动态调整温度
2. 温度范围从[0,1]扩展到[0,2.0]
3. 多模态温度策略
4. 成功率反馈机制

作者：AGI Self-Improvement Module
创建日期：2026-02-04
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TemperatureRecord:
    """温度记录"""
    timestamp: str
    temperature: float
    context: Dict[str, Any]
    outcome_success: bool
    creativity_score: float
    feasibility_score: float


class AdaptiveTemperatureController:
    """
    自适应温度控制器

    基于任务上下文和历史成功率动态调整温度参数
    """

    def __init__(self, base_temperature: float = 0.7):
        """
        初始化自适应温度控制器

        Args:
            base_temperature: 基础温度（默认0.7，向后兼容）
        """
        self.base_temperature = base_temperature
        # 温度范围从[0,1]扩展到[0,2.0]
        self.temperature_range = (0.0, 2.0)
        self.success_history: List[TemperatureRecord] = []

        # 多模态温度策略
        self.mode_temperatures = {
            'exploration': 0.8,      # 探索模式：高温度
            'exploitation': 0.3,     # 利用模式：低温度
            'creative': 1.2,          # 创造模式：超高温度
            'critical': 0.1           # 关键任务：极低温度
        }

        logger.info(
            f"[温度控制器] 初始化: "
            f"基础温度={base_temperature}, "
            f"范围={self.temperature_range}"
        )

    def get_temperature(self, context: Optional[Dict[str, Any]] = None) -> float:
        """
        基于上下文动态调整温度

        Args:
            context: 上下文信息

        Returns:
            动态调整后的温度
        """
        if context is None:
            context = {}

        # 基础温度
        temperature = self.base_temperature
        adjustments = []

        # 因素1: 任务复杂度
        complexity = context.get('task_complexity', 0.5)
        if complexity > 0.8:
            temperature += 0.3
            adjustments.append(f"高复杂度(+0.3)")
        elif complexity < 0.3:
            temperature -= 0.2
            adjustments.append(f"低复杂度(-0.2)")
        else:
            adjustments.append("中复杂度(0.0)")

        # 因素2: 历史成功率
        if self.success_history:
            recent = self.success_history[-20:]
            if recent:
                recent_success_rate = np.mean([1.0 if r.outcome_success else 0.0 for r in recent])

                # 成功率低 → 提高温度探索
                # 成功率高 → 降低温度利用
                temperature += (0.5 - recent_success_rate) * 0.5
                adjustments.append(f"成功率{recent_success_rate:.2f}")
        else:
            adjustments.append("无历史(0.0)")

        # 因素3: 熵值反馈
        entropy = context.get('entropy', 0.5)
        if entropy < 0.3:
            temperature += 0.2  # 低熵→提高温度
            adjustments.append("低熵(+0.2)")
        elif entropy > 0.8:
            temperature -= 0.2  # 高熵→降低温度
            adjustments.append("高熵(-0.2)")
        else:
            adjustments.append("正常熵(0.0)")

        # 因素4: 风险容忍度
        risk_tolerance = context.get('risk_tolerance', 0.5)
        if risk_tolerance > 0.7:
            temperature += 0.3
            adjustments.append(f"高风险容忍(+0.3)")
        elif risk_tolerance < 0.3:
            temperature -= 0.3
            adjustments.append(f"低风险容忍(-0.3)")

        # 应用温度范围限制
        temperature = np.clip(temperature, *self.temperature_range)

        logger.debug(
            f"[温度控制器] 温度调整: "
            f"{self.base_temperature:.2f} -> {temperature:.2f}, "
            f"调整: {', '.join(adjustments)}"
        )

        return temperature

    def get_temperature_for_mode(self, mode: str) -> float:
        """
        获取指定模式的温度

        Args:
            mode: 模式名称（exploration/exploitation/creative/critical）

        Returns:
            该模式的温度
        """
        return self.mode_temperatures.get(mode, self.base_temperature)

    def classify_mode(self, context: Dict[str, Any]) -> str:
        """
        分类任务模式

        Args:
            context: 上下文信息

        Returns:
            任务模式
        """
        novelty = context.get('novelty_required', 0.5)
        risk = context.get('risk_tolerance', 0.5)

        if novelty > 0.8 and risk > 0.7:
            return 'creative'
        elif novelty > 0.6:
            return 'exploration'
        elif risk < 0.2:
            return 'critical'
        else:
            return 'exploitation'

    def record_outcome(
        self,
        temperature: float,
        context: Dict[str, Any],
        success: bool,
        creativity_score: float = 0.5,
        feasibility_score: float = 0.5
    ):
        """
        记录温度使用结果

        Args:
            temperature: 使用的温度
            context: 上下文
            success: 是否成功
            creativity_score: 创造性评分（0-1）
            feasibility_score: 可行性评分（0-1）
        """
        record = TemperatureRecord(
            timestamp=datetime.now().isoformat(),
            temperature=temperature,
            context=context,
            outcome_success=success,
            creativity_score=creativity_score,
            feasibility_score=feasibility_score
        )

        self.success_history.append(record)

        # 保留最近100条记录
        if len(self.success_history) > 100:
            self.success_history.pop(0)

        logger.debug(
            f"[温度控制器] 记录结果: "
            f"温度={temperature:.2f}, 成功={success}, "
            f"创造性={creativity_score:.2f}, 可行性={feasibility_score:.2f}"
        )

    def get_optimal_temperature(self) -> float:
        """
        基于历史记录获取最优温度

        Returns:
            最优温度值
        """
        if not self.success_history:
            return self.base_temperature

        # 分析不同温度范围的成功率
        temp_ranges = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
        best_temp = self.base_temperature
        best_success_rate = 0.0

        for temp_min, temp_max in temp_ranges:
            records_in_range = [
                r for r in self.success_history
                if temp_min <= r.temperature < temp_max
            ]

            if records_in_range:
                success_rate = np.mean([1.0 if r.outcome_success else 0.0 for r in records_in_range])

                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_temp = (temp_min + temp_max) / 2

        return best_temp

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.success_history:
            return {
                'base_temperature': self.base_temperature,
                'temperature_range': self.temperature_range,
                'total_records': 0,
                'avg_temperature': self.base_temperature,
                'success_rate': 1.0,
                'avg_creativity': 0.0,
                'avg_feasibility': 0.0
            }

        temps = [r.temperature for r in self.success_history]
        successes = [1.0 if r.outcome_success else 0.0 for r in self.success_history]
        creativities = [r.creativity_score for r in self.success_history]
        feasibilities = [r.feasibility_score for r in self.success_history]

        return {
            'base_temperature': self.base_temperature,
            'temperature_range': self.temperature_range,
            'total_records': len(self.success_history),
            'avg_temperature': np.mean(temps),
            'min_temperature': np.min(temps),
            'max_temperature': np.max(temps),
            'success_rate': np.mean(successes),
            'avg_creativity': np.mean(creativities),
            'avg_feasibility': np.mean(feasibilities),
            'optimal_temperature': self.get_optimal_temperature(),
            'mode_temperatures': self.mode_temperatures
        }


class MultiModalTemperature:
    """多模态温度策略"""

    def __init__(self):
        """初始化多模态温度策略"""
        self.modes = {
            'exploration': 0.8,     # 探索模式
            'exploitation': 0.3,    # 利用模式
            'creative': 1.2,         # 创造模式
            'critical': 0.1,         # 关键任务模式
            'balanced': 0.7          # 平衡模式
        }
        self.current_mode = 'balanced'

    def set_mode(self, mode: str):
        """
        设置当前模式

        Args:
            mode: 模式名称
        """
        if mode in self.modes:
            self.current_mode = mode
            logger.info(f"[多模态温度] 模式切换: {mode}")
        else:
            logger.warning(f"[多模态温度] 未知模式: {mode}")

    def get_temperature(self, mode: Optional[str] = None) -> float:
        """
        获取温度

        Args:
            mode: 模式名称（None使用当前模式）

        Returns:
            温度值
        """
        if mode is None:
            mode = self.current_mode

        return self.modes.get(mode, 0.7)

    def auto_select_mode(self, context: Dict[str, Any]) -> str:
        """
        自动选择最佳模式

        Args:
            context: 上下文信息

        Returns:
            最佳模式
        """
        novelty = context.get('novelty_required', 0.5)
        risk_tolerance = context.get('risk_tolerance', 0.5)
        urgency = context.get('urgency', 0.5)

        # 决策树
        if urgency > 0.8:
            # 高紧急性 → 关键模式
            selected = 'critical'
        elif novelty > 0.8 and risk_tolerance > 0.7:
            # 高新颖性 + 高风险容忍 → 创造模式
            selected = 'creative'
        elif novelty > 0.6:
            # 高新颖性 → 探索模式
            selected = 'exploration'
        elif risk_tolerance < 0.2:
            # 低风险容忍 → 关键模式
            selected = 'critical'
        else:
            # 默认 → 平衡模式
            selected = 'balanced'

        self.current_mode = selected
        return selected


# 全局单例
_global_controller: Optional[AdaptiveTemperatureController] = None
_global_multimodal: Optional[MultiModalTemperature] = None


def get_temperature_controller() -> AdaptiveTemperatureController:
    """获取全局温度控制器"""
    global _global_controller
    if _global_controller is None:
        _global_controller = AdaptiveTemperatureController()
    return _global_controller


def get_multimodal_temperature() -> MultiModalTemperature:
    """获取全局多模态温度策略"""
    global _global_multimodal
    if _global_multimodal is None:
        _global_multimodal = MultiModalTemperature()
    return _global_multimodal
