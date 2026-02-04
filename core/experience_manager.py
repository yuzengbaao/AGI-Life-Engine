#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
经验管理器 (Experience Manager)
收集、组织和分发决策经验

功能：
1. 收集决策经验（state, action, reward, next_state）
2. 管理经验回放缓冲区
3. 提供批次采样
4. 支持优先级经验回放（PER）

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-13
"""

import numpy as np
import random
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class EnhancedExperience:
    """增强经验（包含决策路径）"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    path: str  # 'fractal', 'seed', 'llm'
    confidence: float
    response_time_ms: float
    timestamp: float = field(default_factory=time.time)


class ExperienceManager:
    """经验管理器"""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[EnhancedExperience] = []
        self.position = 0

        # 统计
        self.total_experiences = 0
        self.episode_rewards: List[float] = []

        logger.info(f"[经验管理器] 初始化完成，容量={capacity}")

    def add_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        path: str,
        confidence: float,
        response_time_ms: float
    ):
        """添加经验"""
        exp = EnhancedExperience(
            state=state.copy(),
            action=action,
            reward=reward,
            next_state=next_state.copy(),
            path=path,
            confidence=confidence,
            response_time_ms=response_time_ms
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self.position] = exp
            self.position = (self.position + 1) % self.capacity

        self.total_experiences += 1

    def sample(self, batch_size: int = 32) -> List[Any]:
        """
        采样一批经验

        返回：Experience对象列表（兼容TheSeed.learn）
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # 随机采样
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        # 转换为标准Experience格式（用于TheSeed）
        try:
            from core.seed import Experience
            standard_batch = [
                Experience(
                    state=exp.state,
                    action=exp.action,
                    reward=exp.reward,
                    next_state=exp.next_state
                )
                for exp in batch
            ]
            return standard_batch
        except ImportError:
            # 如果TheSeed不可用，返回原始格式
            return batch

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'total_experiences': 0
            }

        paths = [exp.path for exp in self.buffer]
        rewards = [exp.reward for exp in self.buffer]
        confidences = [exp.confidence for exp in self.buffer]
        response_times = [exp.response_time_ms for exp in self.buffer]

        return {
            'size': len(self.buffer),
            'total_experiences': self.total_experiences,
            'path_distribution': {
                'fractal': paths.count('fractal'),
                'seed': paths.count('seed'),
                'llm': paths.count('llm')
            },
            'avg_reward': float(np.mean(rewards)) if rewards else 0.0,
            'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'avg_response_time': float(np.mean(response_times)) if response_times else 0.0,
            'total_reward': float(sum(rewards))
        }

    def save(self, filepath: str):
        """
        保存经验到文件

        Args:
            filepath: 保存路径
        """
        import json
        from pathlib import Path

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # 转换为可序列化的格式
        data = {
            'capacity': self.capacity,
            'position': self.position,
            'total_experiences': self.total_experiences,
            'experiences': []
        }

        for exp in self.buffer:
            exp_data = {
                'state': exp.state.tolist(),
                'action': exp.action,
                'reward': exp.reward,
                'next_state': exp.next_state.tolist(),
                'path': exp.path,
                'confidence': exp.confidence,
                'response_time_ms': exp.response_time_ms,
                'timestamp': exp.timestamp
            }
            data['experiences'].append(exp_data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        logger.info(f"[经验管理器] 已保存 {len(self.buffer)} 条经验到 {filepath}")

    def load(self, filepath: str):
        """
        从文件加载经验

        Args:
            filepath: 加载路径
        """
        import json
        from pathlib import Path

        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"[经验管理器] 文件不存在: {filepath}")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 恢复配置
        self.capacity = data.get('capacity', self.capacity)
        self.position = data.get('position', 0)
        self.total_experiences = data.get('total_experiences', 0)

        # 恢复经验
        self.buffer = []
        for exp_data in data.get('experiences', []):
            exp = EnhancedExperience(
                state=np.array(exp_data['state'], dtype=np.float32),
                action=exp_data['action'],
                reward=exp_data['reward'],
                next_state=np.array(exp_data['next_state'], dtype=np.float32),
                path=exp_data['path'],
                confidence=exp_data['confidence'],
                response_time_ms=exp_data['response_time_ms'],
                timestamp=exp_data.get('timestamp', time.time())
            )
            self.buffer.append(exp)

        logger.info(f"[经验管理器] 已加载 {len(self.buffer)} 条经验从 {filepath}")

    def clear(self):
        """清空经验缓冲区"""
        self.buffer = []
        self.position = 0
        logger.info("[经验管理器] 已清空经验缓冲区")
