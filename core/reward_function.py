#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奖励函数 (Reward Function)
定义决策的奖励信号

功能：
1. 置信度奖励（高置信度→正奖励）
2. 速度奖励（快响应→正奖励）
3. 探索奖励（适度熵→正奖励）
4. 任务特定奖励（外部提供）

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-13
"""

import numpy as np
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.hybrid_decision_engine import DecisionResult


def compute_reward(
    state: np.ndarray,
    action: int,
    result: Any,  # DecisionResult
    next_state: np.ndarray,
    context: Dict[str, Any] = None
) -> float:
    """
    计算奖励信号

    Args:
        state: 当前状态
        action: 执行的动作
        result: 决策结果（DecisionResult对象）
        next_state: 下一个状态
        context: 额外上下文

    Returns:
        reward: 奖励值（正值表示好，负值表示坏）
    """
    context = context or {}
    reward = 0.0

    # 1. 置信度奖励（高置信度→正奖励）
    if hasattr(result, 'confidence'):
        confidence = result.confidence
        if confidence > 0.6:
            reward += 0.2
        elif confidence > 0.5:
            reward += 0.1
        elif confidence < 0.4:
            reward -= 0.1

    # 2. 速度奖励（快响应→正奖励）
    if hasattr(result, 'response_time_ms'):
        response_time = result.response_time_ms
        if response_time < 20:
            reward += 0.3  # 极速
        elif response_time < 50:
            reward += 0.2  # 很快
        elif response_time < 100:
            reward += 0.1  # 快速
        elif response_time > 500:
            reward -= 0.2  # 慢

    # 3. 外部依赖惩罚
    if hasattr(result, 'needs_validation') and result.needs_validation:
        reward -= 0.2

    # 4. 探索奖励（适度熵→正奖励）
    if hasattr(result, 'entropy'):
        entropy = result.entropy
        if 0.1 < entropy < 0.5:
            reward += 0.1  # 适度探索
        elif entropy < 0.01:
            reward -= 0.05  # 过于确定
        elif entropy > 0.8:
            reward -= 0.05  # 过度随机

    # 5. 任务特定奖励（外部提供）
    if 'task_reward' in context:
        reward += context['task_reward']

    # 6. 路径偏好（鼓励使用系统B）
    if hasattr(result, 'path'):
        if result.path.value == 'fractal':
            reward += 0.05  # 鼓励分形决策
        elif result.path.value == 'llm':
            reward -= 0.05  # 不鼓励LLM（成本高）

    # 7. 归一化到[-1, 1]
    reward = np.clip(reward, -1.0, 1.0)

    return reward
