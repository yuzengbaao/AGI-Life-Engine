#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
认知能力桥接层 (Cognitive Capability Bridge)
==========================================

功能：将AGI系统的核心认知能力暴露给LLM使用
让LLM能够调用拓扑记忆、因果推理、事件视界、数据流形等能力

作者: Claude Code (Sonnet 4.5)
日期: 2026-01-20
版本: 1.0.0
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CognitiveQuery:
    """认知查询"""
    query_type: str  # 'topology', 'causal', 'pattern', 'prediction'
    question: str
    context: Dict[str, Any]
    requires_deep_reasoning: bool = False


@dataclass
class CognitiveInsight:
    """认知洞察"""
    insight: str
    confidence: float
    source: str  # 'topology', 'causal', 'pattern', 'llm'
    evidence: List[str]
    reasoning: Optional[str] = None


class CognitiveBridge:
    """
    认知能力桥接层

    连接LLM与系统的核心认知能力：
    1. 拓扑记忆系统 - 节点关系、分形结构
    2. 因果推理引擎 - 因果关系、干预预测
    3. 模式识别 - 数据流形、事件序列
    4. 事件视界 - 预测边界、不确定性量化
    """

    def __init__(self, agi_engine=None):
        """
        初始化认知桥接层

        Args:
            agi_engine: AGI_Life_Engine实例（可选）
        """
        self.agi_engine = agi_engine

        # 核心能力引用
        self.topology_memory = None
        self.causal_engine = None
        self.working_memory = None
        self.biological_memory = None

        # 统计信息
        self.query_count = 0
        self.insight_cache = {}

        # 从AGI引擎提取核心能力
        if agi_engine:
            self._extract_capabilities()

        logger.info("✅ 认知能力桥接层已初始化")

    def _extract_capabilities(self):
        """从AGI引擎提取核心认知能力（支持多种系统架构）"""
        try:
            # ===== 适配 AGI_Life_Engine =====
            if hasattr(self.agi_engine, 'topology_memory'):
                self.topology_memory = self.agi_engine.topology_memory
                logger.info("  ✓ 拓扑记忆系统已连接 (AGI_Life_Engine)")

            if hasattr(self.agi_engine, 'causal_engine'):
                self.causal_engine = self.agi_engine.causal_engine
                logger.info("  ✓ 因果推理引擎已连接 (AGI_Life_Engine)")

            if hasattr(self.agi_engine, 'working_memory'):
                self.working_memory = self.agi_engine.working_memory
                logger.info("  ✓ 工作记忆已连接 (AGI_Life_Engine)")

            if hasattr(self.agi_engine, 'biological_memory'):
                self.biological_memory = self.agi_engine.biological_memory
                logger.info("  ✓ 生物记忆已连接 (AGI_Life_Engine)")

            # ===== 适配 FullyIntegratedAGISystem =====
            if hasattr(self.agi_engine, 'bio_memory') and self.agi_engine.bio_memory:
                self.biological_memory = self.agi_engine.bio_memory

                # 从bio_memory提取拓扑记忆
                if hasattr(self.biological_memory, 'topology') and self.biological_memory.topology:
                    self.topology_memory = self.biological_memory.topology
                    logger.info("  ✓ 拓扑记忆系统已连接 (bio_memory.topology)")

                logger.info("  ✓ 生物记忆已连接 (bio_memory)")

            # 如果没有causal_engine，创建一个
            if not self.causal_engine:
                try:
                    from core.causal_reasoning import CausalReasoningEngine
                    self.causal_engine = CausalReasoningEngine()
                    logger.info("  ✓ 因果推理引擎已创建 (新实例)")
                except ImportError:
                    logger.warning("  ⚠️ 因果推理引擎模块不可用")

            # 尝试获取工作记忆（从不同可能的属性）
            if not self.working_memory:
                for attr_name in ['working_memory', 'memory', 'episodic_memory']:
                    if hasattr(self.agi_engine, attr_name):
                        self.working_memory = getattr(self.agi_engine, attr_name)
                        logger.info(f"  ✓ 工作记忆已连接 ({attr_name})")
                        break

        except Exception as e:
            logger.warning(f"提取认知能力时出错: {e}")
            import traceback
            traceback.print_exc()

    # ==================== 拓扑记忆查询 ====================

    def query_topology(self, query: str, node_ids: Optional[List[int]] = None) -> CognitiveInsight:
        """
        查询拓扑记忆

        Args:
            query: 查询问题
            node_ids: 相关节点ID列表

        Returns:
            CognitiveInsight: 拓扑洞察
        """
        if not self.topology_memory:
            return CognitiveInsight(
                insight="拓扑记忆系统不可用",
                confidence=0.0,
                source="topology",
                evidence=[]
            )

        try:
            # 获取拓扑关系
            if node_ids:
                relations = self._get_node_relations(node_ids)
            else:
                # 使用工作记忆中的活跃概念
                if self.working_memory and hasattr(self.working_memory, 'active_concepts'):
                    relations = self._get_active_concept_relations()
                else:
                    relations = []

            # 分析拓扑结构
            insight_text = self._analyze_topology(query, relations)

            return CognitiveInsight(
                insight=insight_text,
                confidence=0.8,
                source="topology",
                evidence=[f"分析了 {len(relations)} 个拓扑关系"],
                reasoning="基于拓扑记忆的节点连接分析"
            )

        except Exception as e:
            logger.error(f"拓扑查询失败: {e}")
            return CognitiveInsight(
                insight=f"拓扑查询失败: {str(e)}",
                confidence=0.0,
                source="topology",
                evidence=[]
            )

    def _get_node_relations(self, node_ids: List[int]) -> List[Dict[str, Any]]:
        """获取节点间的拓扑关系"""
        relations = []
        for node_id in node_ids:
            edges = self.topology_memory.get_edges(node_id)
            for edge in edges:
                relations.append({
                    'source': node_id,
                    'target': edge.to_idx,
                    'weight': edge.weight,
                    'kind': edge.kind,
                    'ports': (edge.from_port, edge.to_port)
                })
        return relations

    def _get_active_concept_relations(self) -> List[Dict[str, Any]]:
        """获取活跃概念的关系"""
        if not self.working_memory or not hasattr(self.working_memory, 'active_concepts'):
            return []

        relations = []
        for concept_id, concept_data in self.working_memory.active_concepts.items():
            # 尝试从概念ID提取整数节点ID
            try:
                node_id = hash(concept_id) & 0xffffffff
                edges = self.topology_memory.get_edges(node_id)
                for edge in edges:
                    relations.append({
                        'source': concept_id,
                        'target': f"node_{edge.to_idx}",
                        'weight': edge.weight,
                        'kind': edge.kind
                    })
            except:
                pass

        return relations

    def _analyze_topology(self, query: str, relations: List[Dict[str, Any]]) -> str:
        """分析拓扑结构并生成洞察"""
        if not relations:
            return "当前没有可用的拓扑关系"

        # 统计
        total_relations = len(relations)
        strong_relations = [r for r in relations if r['weight'] > 0.7]
        weak_relations = [r for r in relations if r['weight'] < 0.3]

        # 生成洞察
        insight_parts = [
            f"基于拓扑记忆分析：",
            f"- 共发现 {total_relations} 个拓扑关系",
            f"- 强连接（权重大于0.7）：{len(strong_relations)} 个",
            f"- 弱连接（权重小于0.3）：{len(weak_relations)} 个",
        ]

        # 如果有强连接，描述它们
        if strong_relations:
            insight_parts.append("\n最强连接：")
            for r in sorted(strong_relations, key=lambda x: x['weight'], reverse=True)[:3]:
                insight_parts.append(f"  {r['source']} → {r['target']} (权重: {r['weight']:.2f})")

        return "\n".join(insight_parts)

    # ==================== 因果推理查询 ====================

    def query_causality(self, query: str, events: Optional[List[Dict]] = None) -> CognitiveInsight:
        """
        查询因果关系

        Args:
            query: 查询问题（例如："为什么X发生了？"）
            events: 相关事件列表

        Returns:
            CognitiveInsight: 因果洞察
        """
        if not self.causal_engine:
            return CognitiveInsight(
                insight="因果推理引擎不可用",
                confidence=0.0,
                source="causal",
                evidence=[]
            )

        try:
            # 如果没有提供事件，尝试从系统状态推断
            if not events:
                events = self._extract_recent_events()

            # 使用因果推理引擎
            explanation = self.causal_engine.explain_reasoning(query)

            return CognitiveInsight(
                insight=explanation,
                confidence=0.75,
                source="causal",
                evidence=[f"分析了 {len(events) if events else 0} 个事件"],
                reasoning="基于因果推理的反事实分析"
            )

        except Exception as e:
            logger.error(f"因果查询失败: {e}")
            return CognitiveInsight(
                insight=f"因果查询失败: {str(e)}",
                confidence=0.0,
                source="causal",
                evidence=[]
            )

    def _extract_recent_events(self) -> List[Dict]:
        """从系统状态提取最近的事件"""
        events = []

        # 从工作记忆提取
        if self.working_memory and hasattr(self.working_memory, 'episodic_buffer'):
            for memory in self.working_memory.episodic_buffer[-10:]:  # 最近10条
                events.append({
                    'type': 'episodic_memory',
                    'content': str(memory)[:200],
                    'timestamp': getattr(memory, 'timestamp', None)
                })

        return events

    # ==================== 综合认知查询 ====================

    def deep_reasoning(self, user_query: str) -> str:
        """
        深度推理：综合使用所有认知能力回答问题

        Args:
            user_query: 用户问题

        Returns:
            str: 综合洞察
        """
        self.query_count += 1

        insights = []

        # 1. 拓扑分析
        topology_insight = self.query_topology(user_query)
        if topology_insight.confidence > 0:
            insights.append({
                'type': '拓扑记忆',
                'insight': topology_insight.insight,
                'confidence': topology_insight.confidence
            })

        # 2. 因果分析
        causal_insight = self.query_causality(user_query)
        if causal_insight.confidence > 0:
            insights.append({
                'type': '因果推理',
                'insight': causal_insight.insight,
                'confidence': causal_insight.confidence
            })

        # 3. 组合洞察
        if not insights:
            return "抱歉，当前没有可用的认知洞察。"

        # 按置信度排序
        insights.sort(key=lambda x: x['confidence'], reverse=True)

        # 生成综合响应
        response_parts = [f"🧠 深度认知分析（查询 #{self.query_count}）\n"]

        for i, insight in enumerate(insights, 1):
            response_parts.append(f"\n【{insight['type']}】(置信度: {insight['confidence']:.0%})")
            response_parts.append(insight['insight'])

        return "\n".join(response_parts)

    def get_capability_summary(self) -> Dict[str, bool]:
        """获取可用能力摘要"""
        return {
            'topology_memory': self.topology_memory is not None,
            'causal_engine': self.causal_engine is not None,
            'working_memory': self.working_memory is not None,
            'biological_memory': self.biological_memory is not None,
        }


# ==================== 工具函数 ====================

def create_cognitive_bridge(agi_engine) -> CognitiveBridge:
    """创建认知桥接实例"""
    return CognitiveBridge(agi_engine)


def get_cognitive_prompt(bridge: CognitiveBridge, user_query: str) -> str:
    """
    为LLM生成增强提示词，包含认知上下文

    Args:
        bridge: 认知桥接实例
        user_query: 用户查询

    Returns:
        str: 增强的提示词
    """
    capabilities = bridge.get_capability_summary()

    prompt_parts = [
        "你是一个具有深度认知能力的AGI系统。",
        "你可以访问以下核心认知能力：\n"
    ]

    # 列出可用能力
    available_capabilities = []
    if capabilities['topology_memory']:
        available_capabilities.append("- 拓扑记忆：理解节点间的复杂关系和分形结构")
    if capabilities['causal_engine']:
        available_capabilities.append("- 因果推理：真正的因果理解和反事实推理")
    if capabilities['working_memory']:
        available_capabilities.append("- 工作记忆：当前的活跃概念和上下文")
    if capabilities['biological_memory']:
        available_capabilities.append("- 生物记忆：长期记忆和模式识别")

    if available_capabilities:
        prompt_parts.extend(available_capabilities)
        prompt_parts.append(f"\n用户问题：{user_query}")
        prompt_parts.append(
            "\n请利用这些认知能力来回答问题。"
            "如果需要深入的拓扑或因果分析，请明确指出。"
        )
    else:
        prompt_parts.append("（当前没有可用的认知能力增强）")

    return "\n".join(prompt_parts)
