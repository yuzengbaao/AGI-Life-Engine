#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理调度器（Reasoning Scheduler）
==================================

功能：智能调度推理引擎，优先使用因果推理，降级到LLM
目标：实现推理深度从15步提升至1000步+

版本: 1.0.0
"""

import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class ReasoningMode(Enum):
    """推理模式"""
    CAUSAL = "causal"          # 因果推理
    HYBRID = "hybrid"          # 混合推理
    LLM_FALLBACK = "llm"       # LLM降级
    PATTERN_MATCH = "pattern"  # 模式匹配


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: str
    mode: ReasoningMode
    timestamp: float
    input_data: Dict[str, Any]
    reasoning_process: str
    output: Any
    confidence: float
    depth: int
    execution_time: float


@dataclass
class ReasoningSession:
    """推理会话"""
    session_id: str
    start_time: float
    steps: List[ReasoningStep] = field(default_factory=list)
    current_depth: int = 0
    max_depth: int = 99999  # 🔧 [2026-01-20] 解除推理深度限制
    mode_history: List[ReasoningMode] = field(default_factory=list)

    def add_step(self, step: ReasoningStep):
        """添加推理步骤"""
        self.steps.append(step)
        self.current_depth = max(self.current_depth, step.depth)
        self.mode_history.append(step.mode)

    def get_summary(self) -> Dict[str, Any]:
        """获取会话摘要"""
        if not self.steps:
            return {
                'session_id': self.session_id,
                'total_steps': 0,
                'max_depth': 0,
                'avg_confidence': 0.0,
                'mode_distribution': {}
            }

        mode_counts = {}
        for mode in self.mode_history:
            mode_counts[mode.value] = mode_counts.get(mode.value, 0) + 1

        return {
            'session_id': self.session_id,
            'total_steps': len(self.steps),
            'max_depth': self.current_depth,
            'avg_confidence': sum(s.confidence for s in self.steps) / len(self.steps),
            'mode_distribution': mode_counts,
            'total_time': self.steps[-1].timestamp - self.start_time if self.steps else 0,
            'avg_step_time': sum(s.execution_time for s in self.steps) / len(self.steps) if self.steps else 0
        }


class ReasoningScheduler:
    """
    推理调度器

    核心功能：
    1. 智能选择推理引擎（因果推理优先）
    2. 追踪推理深度
    3. 降级策略管理
    4. 推理历史记录
    """

    def __init__(self, causal_engine=None, llm_service=None,
                 confidence_threshold: float = 0.6,
                 max_depth: int = 99999):  # 🔧 [2026-01-20] 解除推理深度限制
        """
        初始化推理调度器

        Args:
            causal_engine: 因果推理引擎实例
            llm_service: LLM服务实例
            confidence_threshold: 因果推理置信度阈值
            max_depth: 最大推理深度（已解除限制）
        """
        self.causal_engine = causal_engine
        self.llm_service = llm_service
        self.confidence_threshold = confidence_threshold
        self.max_depth = max_depth

        # 推理会话管理
        self.current_session: Optional[ReasoningSession] = None
        self.session_history: List[ReasoningSession] = []

        # 性能统计
        self.stats = {
            'total_reasoning_calls': 0,
            'causal_reasoning_used': 0,
            'llm_fallback_used': 0,
            'hybrid_reasoning_used': 0,
            'avg_depth_per_session': 0,
            'max_depth_achieved': 0
        }

        # 推理链缓存（用于避免重复推理）
        self.reasoning_cache: Dict[str, ReasoningStep] = {}

    def start_session(self, context: Optional[Dict] = None) -> str:
        """
        开始新的推理会话

        Args:
            context: 初始上下文

        Returns:
            session_id: 会话ID
        """
        session_id = f"session_{int(time.time() * 1000)}"
        self.current_session = ReasoningSession(
            session_id=session_id,
            start_time=time.time(),
            max_depth=self.max_depth
        )
        return session_id

    def reason(self, query: str, context: Optional[Dict] = None,
               prefer_causal: bool = True) -> Tuple[Any, ReasoningStep]:
        """
        执行推理

        Args:
            query: 推理查询
            context: 上下文信息
            prefer_causal: 是否优先使用因果推理

        Returns:
            (result, reasoning_step): 推理结果和推理步骤记录
        """
        if not self.current_session:
            self.start_session()

        self.stats['total_reasoning_calls'] += 1

        # 生成查询缓存键
        cache_key = self._generate_cache_key(query, context)
        if cache_key in self.reasoning_cache:
            cached_step = self.reasoning_cache[cache_key]
            return cached_step.output, cached_step

        start_time = time.time()

        # 决策：使用哪种推理引擎
        mode, result, confidence = self._select_reasoning_engine(
            query, context, prefer_causal
        )

        execution_time = time.time() - start_time

        # 创建推理步骤
        step = ReasoningStep(
            step_id=f"step_{len(self.current_session.steps)}",
            mode=mode,
            timestamp=time.time(),
            input_data={'query': query, 'context': context},
            reasoning_process=self._get_reasoning_process_description(mode),
            output=result,
            confidence=confidence,
            depth=len(self.current_session.steps) + 1,
            execution_time=execution_time
        )

        # 记录推理步骤
        self.current_session.add_step(step)
        self.reasoning_cache[cache_key] = step

        # 更新统计
        if mode == ReasoningMode.CAUSAL:
            self.stats['causal_reasoning_used'] += 1
        elif mode == ReasoningMode.LLM_FALLBACK:
            self.stats['llm_fallback_used'] += 1
        elif mode == ReasoningMode.HYBRID:
            self.stats['hybrid_reasoning_used'] += 1

        return result, step

    def _select_reasoning_engine(self, query: str, context: Optional[Dict],
                                 prefer_causal: bool) -> Tuple[ReasoningMode, Any, float]:
        """
        选择推理引擎

        决策流程：
        1. 如果有因果推理引擎且被启用 -> 尝试因果推理
        2. 如果因果推理置信度足够 -> 使用因果推理结果
        3. 否则 -> 降级到LLM
        """
        # 尝试因果推理
        if prefer_causal and self.causal_engine:
            try:
                # 检查是否适合因果推理
                if self._is_suitable_for_causal_reasoning(query, context):
                    # 执行因果推理
                    result, confidence = self._perform_causal_reasoning(query, context)

                    if confidence >= self.confidence_threshold:
                        return ReasoningMode.CAUSAL, result, confidence
                    else:
                        # 置信度不足，尝试混合推理
                        if self.llm_service:
                            enhanced_result = self._hybrid_reasoning(result, query, context)
                            return ReasoningMode.HYBRID, enhanced_result, confidence + 0.2

            except Exception as e:
                print(f"  [Scheduler] Causal reasoning failed: {e}, falling back to LLM")

        # 降级到LLM
        if self.llm_service:
            try:
                llm_result = self._perform_llm_reasoning(query, context)
                return ReasoningMode.LLM_FALLBACK, llm_result, 0.5  # LLM基础置信度
            except Exception as e:
                print(f"  [Scheduler] LLM reasoning failed: {e}")

        # 最后的降级：模式匹配
        return self._perform_pattern_matching(query, context)

    def _is_suitable_for_causal_reasoning(self, query: str, context: Optional[Dict]) -> bool:
        """
        判断是否适合因果推理

        适合场景：
        - 查询包含因果关键词（为什么、导致、因为）
        - 上下文中有事件序列
        - 需要预测干预效果
        """
        causal_keywords = ['为什么', 'why', '导致', 'cause', '因为', 'because',
                          '如果', 'if', '预测', 'predict', '影响', 'effect']

        query_lower = query.lower()
        has_causal_keyword = any(kw in query_lower for kw in causal_keywords)

        # 检查上下文中是否有事件
        has_events = context and 'events' in context

        return has_causal_keyword or has_events

    def _perform_causal_reasoning(self, query: str, context: Optional[Dict]) -> Tuple[Any, float]:
        """执行因果推理"""
        from core.causal_reasoning import Event

        # 从上下文提取事件
        events = []
        if context and 'events' in context:
            events = context['events']
        else:
            # 如果没有明确的事件，创建模拟事件
            events = [
                Event(id=f"E{i}", type="query", timestamp=time.time() + i * 0.1,
                      properties={"content": query})
                for i in range(3)
            ]

        # 执行因果推理
        causal_graph = self.causal_engine.infer_causality(events)

        # 尝试解释
        explanation = self.causal_engine.explain_reasoning(query)

        # 计算置信度（基于因果关系数量）
        confidence = min(0.5 + len(causal_graph.edges) * 0.1, 0.95)

        result = {
            'explanation': explanation,
            'causal_relations': len(causal_graph.edges),
            'graph': causal_graph
        }

        return result, confidence

    def _perform_llm_reasoning(self, query: str, context: Optional[Dict]) -> Any:
        """执行LLM推理"""
        # 构建提示词
        prompt = self._build_llm_prompt(query, context)

        # 调用LLM服务
        if hasattr(self.llm_service, 'query'):
            response = self.llm_service.query(prompt)
        elif hasattr(self.llm_service, 'generate'):
            response = self.llm_service.generate(prompt)
        else:
            response = f"LLM response for: {query}"

        return response

    def _hybrid_reasoning(self, causal_result: Any, query: str,
                          context: Optional[Dict]) -> Any:
        """混合推理：结合因果推理和LLM"""
        # 使用因果推理的结果作为上下文，让LLM生成更好的解释
        enhanced_context = {
            'causal_reasoning': causal_result.get('explanation', ''),
            'original_context': context
        }

        return self._perform_llm_reasoning(query, enhanced_context)

    def _perform_pattern_matching(self, query: str, context: Optional[Dict]) -> Tuple[ReasoningMode, Any, float]:
        """模式匹配（最后的降级方案）"""
        # 简单的关键词匹配
        response = f"Pattern matching result for query: {query}"

        if context:
            if 'action' in context:
                response = f"Action: {context['action']}"
            elif 'concept' in context:
                response = f"Concept: {context['concept']}"

        return ReasoningMode.PATTERN_MATCH, response, 0.3

    def _build_llm_prompt(self, query: str, context: Optional[Dict]) -> str:
        """构建LLM提示词"""
        prompt = f"Query: {query}\n"

        if context:
            prompt += "\nContext:\n"
            for key, value in context.items():
                if key != 'events':  # 事件太长，简化显示
                    prompt += f"  {key}: {value}\n"

        prompt += "\nPlease provide reasoning and conclusion."
        return prompt

    def _get_reasoning_process_description(self, mode: ReasoningMode) -> str:
        """获取推理过程描述"""
        descriptions = {
            ReasoningMode.CAUSAL: "Causal inference using temporal precedence, covariation, and confounding exclusion",
            ReasoningMode.HYBRID: "Hybrid reasoning combining causal inference with LLM enhancement",
            ReasoningMode.LLM_FALLBACK: "LLM-based reasoning (fallback due to low causal confidence)",
            ReasoningMode.PATTERN_MATCH: "Pattern matching fallback (lowest confidence)"
        }
        return descriptions.get(mode, "Unknown reasoning mode")

    def _generate_cache_key(self, query: str, context: Optional[Dict]) -> str:
        """生成缓存键"""
        content = query + str(context) if context else query
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def end_session(self) -> ReasoningSession:
        """结束当前推理会话"""
        if self.current_session:
            session = self.current_session
            self.session_history.append(session)

            # 更新统计
            self.stats['max_depth_achieved'] = max(
                self.stats['max_depth_achieved'], session.current_depth
            )
            if self.session_history:
                self.stats['avg_depth_per_session'] = sum(
                    s.current_depth for s in self.session_history
                ) / len(self.session_history)

            self.current_session = None
            return session

        return None

    def get_current_session_summary(self) -> Dict[str, Any]:
        """获取当前会话摘要"""
        if self.current_session:
            return self.current_session.get_summary()
        return {}

    def get_statistics(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        return {
            **self.stats,
            'current_session_depth': self.current_session.current_depth if self.current_session else 0,
            'total_sessions': len(self.session_history),
            'cache_size': len(self.reasoning_cache),
            'causal_ratio': self.stats['causal_reasoning_used'] / max(self.stats['total_reasoning_calls'], 1)
        }

    def get_reasoning_chain(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取最近n步推理链"""
        if not self.current_session:
            return []

        recent_steps = self.current_session.steps[-n:]
        return [
            {
                'step': s.step_id,
                'mode': s.mode.value,
                'depth': s.depth,
                'confidence': s.confidence,
                'time': s.execution_time
            }
            for s in recent_steps
        ]


# ============ 使用示例 ============

if __name__ == "__main__":
    print("=" * 60)
    print("推理调度器测试")
    print("=" * 60)

    # 创建调度器
    from core.causal_reasoning import CausalReasoningEngine

    causal_engine = CausalReasoningEngine()
    scheduler = ReasoningScheduler(
        causal_engine=causal_engine,
        llm_service=None,  # 无LLM时自动降级到模式匹配
        confidence_threshold=0.6,
        max_depth=99999  # 🔧 [2026-01-20] 无限推理深度
    )

    # 开始会话
    session_id = scheduler.start_session()
    print(f"\n[Session] {session_id}")

    # 模拟多次推理
    queries = [
        "为什么系统会陷入循环？",
        "如何打破思想循环？",
        "预测添加工作记忆的效果",
        "分析当前系统状态",
        "探索改进方案"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n[Step {i}] Query: {query}")
        result, step = scheduler.reason(query, prefer_causal=True)

        print(f"  Mode: {step.mode.value}")
        print(f"  Confidence: {step.confidence:.2f}")
        print(f"  Depth: {step.depth}")
        print(f"  Time: {step.execution_time:.3f}s")

        if isinstance(result, dict) and 'explanation' in result:
            print(f"  Result: {result['explanation'][:100]}...")

    # 获取会话摘要
    summary = scheduler.end_session()
    print("\n" + "=" * 60)
    print("[会话摘要]")
    print("=" * 60)
    for key, value in summary.get_summary().items():
        print(f"  {key}: {value}")

    # 获取统计
    stats = scheduler.get_statistics()
    print("\n[调度器统计]")
    for key, value in stats.items():
        print(f"  {key}: {value}")
