#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infinite Recursion Engine - 无限递归引擎
===========================================

功能：
1. 通过状态压缩实现深层递归
2. 怪圈检测与利用
3. 收敛性检测
4. 递归深度动态管理

作者：AGI Self-Improvement Module
创建日期：2026-02-04
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RecursionIteration:
    """递归迭代记录"""
    iteration: int
    depth: int
    input_state: str
    output_state: str
    compression_ratio: float  # 压缩比
    novelty_score: float  # 新颖度
    timestamp: float


@dataclass
class StrangeLoop:
    """怪圈（有价值的自指涉循环）"""
    loop_id: str
    length: int  # 循环长度
    states: List[str]  # 循环状态序列
    novelty_score: float  # 新颖度
    consistency_score: float  # 一致性
    value_score: float  # 价值评分
    discovered_at: int  # 发现时的迭代次数
    timestamps: float = field(default_factory=time.time)


class InfiniteRecursion:
    """
    无限递归引擎

    通过状态压缩实现深层递归，避免堆栈溢出
    """

    def __init__(
        self,
        compression_interval: int = 5,
        max_iterations: int = 100,
        convergence_threshold: float = 0.95
    ):
        """
        初始化无限递归引擎

        Args:
            compression_interval: 状态压缩间隔（每N层压缩一次）
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值（相似度超过此值认为收敛）
        """
        self.compression_interval = compression_interval
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        # 递归历史
        self.recursion_history: List[RecursionIteration] = []
        self.state_embeddings: Dict[str, np.ndarray] = {}

        # 怪圈检测
        self.detected_loops: List[StrangeLoop] = []
        self.loop_detection_window = 10  # 检测窗口大小

        logger.info(
            f"[无限递归] 初始化: "
            f"压缩间隔={compression_interval}, "
            f"最大迭代={max_iterations}"
        )

    def recursive_reflection(
        self,
        current_understanding: str,
        context: Optional[Dict[str, Any]] = None,
        reflect_func: Optional[callable] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        无限递归反思（带状态压缩）

        Args:
            current_understanding: 当前理解/状态
            context: 上下文信息
            reflect_func: 反思函数（如果为None，使用默认）

        Returns:
            (最终理解, 统计信息)
        """
        if context is None:
            context = {}

        stats = {
            'total_iterations': 0,
            'total_compressions': 0,
            'max_depth_reached': 0,
            'converged': False,
            'strange_loops_found': 0,
            'final_novelty': 0.0
        }

        current_state = current_understanding
        iteration = 0

        for iteration in range(1, self.max_iterations + 1):
            stats['total_iterations'] = iteration

            # 记录当前深度
            depth = len(self.recursion_history)
            stats['max_depth_reached'] = max(stats['max_depth_reached'], depth)

            # 检查是否需要压缩
            if depth > 0 and depth % self.compression_interval == 0:
                old_length = len(current_state)
                current_state = self.compress_state(current_state)
                compression_ratio = len(current_state) / old_length

                logger.info(
                    f"[无限递归] 状态压缩（深度={depth}, "
                    f"压缩比={compression_ratio:.2f})"
                )
                stats['total_compressions'] += 1

            # 执行反思
            if reflect_func is not None:
                new_state = reflect_func(current_state, context)
            else:
                new_state = self._default_reflection(current_state, context)

            # 计算新颖度
            novelty = self._calculate_novelty(current_state, new_state)

            # 记录迭代
            iteration_record = RecursionIteration(
                iteration=iteration,
                depth=depth,
                input_state=current_state,
                output_state=new_state,
                compression_ratio=getattr(iteration_record, 'compression_ratio', 1.0),
                novelty_score=novelty,
                timestamp=time.time()
            )
            self.recursion_history.append(iteration_record)

            # 检查收敛
            if self.has_converged(current_state, new_state):
                logger.info(f"[无限递归] 收敛（迭代次数={iteration}）")
                stats['converged'] = True
                current_state = new_state
                break

            # 检测怪圈
            strange_loop = self.detect_strange_loop(
                [r.output_state for r in self.recursion_history[-self.loop_detection_window:]]
            )
            if strange_loop:
                logger.info(
                    f"[无限递归] 发现怪圈（长度={len(strange_loop.states)}, "
                    f"价值={strange_loop.value_score:.2f})"
                )
                self.detected_loops.append(strange_loop)
                stats['strange_loops_found'] += 1

                # 如果怪圈价值高，利用它
                if strange_loop.value_score > 0.7:
                    logger.info(f"[无限递归] 利用高价值怪圈")
                    current_state = strange_loop.states[-1]  # 使用怪圈的最后状态
                    stats['final_novelty'] = strange_loop.novelty_score
                    break

            current_state = new_state

        stats['final_novelty'] = self._calculate_novelty(current_understanding, current_state)

        return current_state, stats

    def compress_state(self, state: str) -> str:
        """
        压缩状态（保留关键信息）

        Args:
            state: 原始状态

        Returns:
            压缩后的状态
        """
        # 提取关键概念（简化版本：提取关键词）
        concepts = self._extract_key_concepts(state)

        # 生成摘要
        summary = self._generate_summary(concepts)

        return summary

    def _extract_key_concepts(self, state: str) -> List[str]:
        """
        提取关键概念

        Args:
            state: 状态文本

        Returns:
            关键概念列表
        """
        # 简化版本：按句子分割，保留前5个长句
        sentences = [s.strip() for s in state.split('.') if s.strip()]

        # 按长度排序，保留最长的5个
        sorted_sentences = sorted(sentences, key=len, reverse=True)
        key_concepts = sorted_sentences[:5]

        return key_concepts

    def _generate_summary(self, concepts: List[str]) -> str:
        """
        生成摘要

        Args:
            concepts: 关键概念列表

        Returns:
            摘要文本
        """
        # 简化版本：拼接概念
        summary = ". ".join(concepts)

        # 确保不超过原长度的50%
        if len(summary) > 500:
            summary = summary[:500] + "..."

        return summary

    def has_converged(self, old_state: str, new_state: str) -> bool:
        """
        检查是否收敛

        Args:
            old_state: 旧状态
            new_state: 新状态

        Returns:
            是否收敛
        """
        # 使用余弦相似度
        similarity = self._cosine_similarity(
            self._embed(old_state),
            self._embed(new_state)
        )

        return similarity > self.convergence_threshold

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _embed(self, text: str) -> np.ndarray:
        """
        将文本嵌入向量（简化版本）

        Args:
            text: 文本

        Returns:
            向量表示
        """
        # 简化版本：使用字符频率作为特征
        # 实际应用中应该使用真正的嵌入模型（如sentence-transformers）

        # 如果已经缓存过，直接返回
        if text in self.state_embeddings:
            return self.state_embeddings[text]

        # 字符频率统计
        char_freq = np.zeros(128)  # ASCII字符范围
        for char in text:
            code = ord(char)
            if code < 128:
                char_freq[code] += 1

        # 归一化
        char_freq = char_freq / (np.linalg.norm(char_freq) + 1e-8)

        # 缓存
        self.state_embeddings[text] = char_freq

        return char_freq

    def detect_strange_loop(self, state_sequence: List[str]) -> Optional[StrangeLoop]:
        """
        检测怪圈（有价值的自指涉循环）

        Args:
            state_sequence: 状态序列

        Returns:
            检测到的怪圈（如果存在）
        """
        if len(state_sequence) < 3:
            return None

        # 检查所有可能的循环
        for loop_length in range(3, min(len(state_sequence) + 1, 8)):
            for start_idx in range(len(state_sequence) - loop_length):
                # 提取候选循环
                candidate_loop = state_sequence[start_idx:start_idx + loop_length]

                # 检查是否形成循环（首尾相似）
                if self._is_loop(candidate_loop):
                    # 评估循环价值
                    novelty = self._calculate_loop_novelty(candidate_loop)
                    consistency = self._calculate_loop_consistency(candidate_loop)
                    value = (novelty + consistency) / 2

                    # 只返回高价值循环
                    if value > 0.6:
                        return StrangeLoop(
                            loop_id=f"loop_{int(time.time())}_{start_idx}",
                            length=loop_length,
                            states=candidate_loop,
                            novelty_score=novelty,
                            consistency_score=consistency,
                            value_score=value,
                            discovered_at=len(self.recursion_history)
                        )

        return None

    def _is_loop(self, states: List[str]) -> bool:
        """
        检查状态序列是否形成循环

        Args:
            states: 状态列表

        Returns:
            是否形成循环
        """
        if len(states) < 3:
            return False

        # 检查首尾状态的相似度
        first_embed = self._embed(states[0])
        last_embed = self._embed(states[-1])

        similarity = self._cosine_similarity(first_embed, last_embed)

        # 相似度 > 0.85 认为形成循环
        return similarity > 0.85

    def _calculate_loop_novelty(self, loop_states: List[str]) -> float:
        """
        计算循环的新颖度

        Args:
            loop_states: 循环状态列表

        Returns:
            新颖度评分
        """
        # 新颖度 = 1 - 与历史循环的平均相似度
        if not self.detected_loops:
            return 1.0

        # 计算当前循环的嵌入
        loop_embed = np.mean([self._embed(s) for s in loop_states], axis=0)

        # 与历史循环比较
        min_similarity = 1.0
        for loop in self.detected_loops:
            history_embed = np.mean([self._embed(s) for s in loop.states], axis=0)
            similarity = self._cosine_similarity(loop_embed, history_embed)
            min_similarity = min(min_similarity, similarity)

        novelty = 1.0 - min_similarity
        return novelty

    def _calculate_loop_consistency(self, loop_states: List[str]) -> float:
        """
        计算循环的一致性

        Args:
            loop_states: 循环状态列表

        Returns:
            一致性评分
        """
        if len(loop_states) < 2:
            return 1.0

        # 计算相邻状态的相似度
        similarities = []
        for i in range(len(loop_states) - 1):
            sim = self._cosine_similarity(
                self._embed(loop_states[i]),
                self._embed(loop_states[i + 1])
            )
            similarities.append(sim)

        # 一致性 = 平均相似度
        consistency = np.mean(similarities)

        return consistency

    def _calculate_novelty(self, old_state: str, new_state: str) -> float:
        """
        计算新状态的新颖度

        Args:
            old_state: 旧状态
            new_state: 新状态

        Returns:
            新颖度评分
        """
        # 新颖度 = 1 - 相似度
        similarity = self._cosine_similarity(
            self._embed(old_state),
            self._embed(new_state)
        )

        novelty = 1.0 - similarity
        return novelty

    def _default_reflection(self, current_state: str, context: Dict) -> str:
        """
        默认反思函数（简化版本）

        Args:
            current_state: 当前状态
            context: 上下文

        Returns:
            反思后的状态
        """
        # 简化版本：添加迭代标记
        iteration_count = len(self.recursion_history) + 1

        reflected = f"[反思 #{iteration_count}] {current_state}"

        return reflected

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.recursion_history:
            return {
                'total_iterations': 0,
                'max_depth': 0,
                'total_compressions': 0,
                'strange_loops_found': 0,
                'avg_novelty': 0.0
            }

        novelties = [r.novelty_score for r in self.recursion_history]

        return {
            'total_iterations': len(self.recursion_history),
            'max_depth': max(r.depth for r in self.recursion_history),
            'total_compressions': sum(1 for r in self.recursion_history if r.compression_ratio < 1.0),
            'strange_loops_found': len(self.detected_loops),
            'avg_novelty': np.mean(novelties) if novelties else 0.0,
            'max_novelty': max(novelties) if novelties else 0.0,
            'compression_interval': self.compression_interval,
            'max_iterations': self.max_iterations
        }


class StrangeLoopDetector:
    """怪圈检测器（独立版本）"""

    def __init__(self, min_loop_length: int = 3, max_loop_length: int = 7):
        """
        初始化怪圈检测器

        Args:
            min_loop_length: 最小循环长度
            max_loop_length: 最大循环长度
        """
        self.min_loop_length = min_loop_length
        self.max_loop_length = max_loop_length
        self.infinite_recursion = InfiniteRecursion()  # 复用嵌入方法

    def detect_loops(
        self,
        reflection_sequence: List[str]
    ) -> List[StrangeLoop]:
        """
        检测序列中的所有怪圈

        Args:
            reflection_sequence: 反思序列

        Returns:
            检测到的怪圈列表
        """
        detected_loops = []

        if len(reflection_sequence) < self.min_loop_length:
            return detected_loops

        # 滑动窗口检测
        for window_size in range(self.min_loop_length, min(self.max_loop_length + 1, len(reflection_sequence) + 1)):
            for start_idx in range(len(reflection_sequence) - window_size + 1):
                window = reflection_sequence[start_idx:start_size + window_size]

                if self.infinite_recursion._is_loop(window):
                    # 评估价值
                    novelty = self.infinite_recursion._calculate_loop_novelty(window)
                    consistency = self.infinite_recursion._calculate_loop_consistency(window)
                    value = (novelty + consistency) / 2

                    if value > 0.6:
                        loop = StrangeLoop(
                            loop_id=f"loop_{len(detected_loops)}_{start_idx}",
                            length=window_size,
                            states=window,
                            novelty_score=novelty,
                            consistency_score=consistency,
                            value_score=value,
                            discovered_at=start_idx
                        )
                        detected_loops.append(loop)

        # 按价值排序
        detected_loops.sort(key=lambda l: l.value_score, reverse=True)

        return detected_loops

    def is_valuable_loop(self, loop: StrangeLoop) -> bool:
        """
        检查循环是否有价值

        Args:
            loop: 怪圈

        Returns:
            是否有价值
        """
        # 价值标准：
        # 1. 深度（循环长度 >= 3）
        # 2. 新颖性（包含新概念）
        # 3. 一致性（逻辑连贯）

        if loop.length < self.min_loop_length:
            return False

        if loop.novelty_score < 0.6:
            return False

        if loop.consistency_score < 0.7:
            return False

        return True


# 全局单例
_global_infinite_recursion: Optional[InfiniteRecursion] = None
_global_loop_detector: Optional[StrangeLoopDetector] = None


def get_infinite_recursion() -> InfiniteRecursion:
    """获取全局无限递归引擎"""
    global _global_infinite_recursion
    if _global_infinite_recursion is None:
        _global_infinite_recursion = InfiniteRecursion()
    return _global_infinite_recursion


def get_strange_loop_detector() -> StrangeLoopDetector:
    """获取全局怪圈检测器"""
    global _global_loop_detector
    if _global_loop_detector is None:
        _global_loop_detector = StrangeLoopDetector()
    return _global_loop_detector
