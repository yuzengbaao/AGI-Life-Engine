"""
孤立节点预防增强模块 - P1修复
解决知识图谱孤立节点产生问题
"""

import hashlib
import time
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict


class IsolatedNodePrevention:
    """
    孤立节点预防器
    
    策略:
    1. 创建节点时强制建立至少3个连接
    2. 语义相似度匹配（使用感知系统编码）
    3. 连接不足时自动连接到核心枢纽
    4. 定期巡检（每30分钟）和修复
    """
    
    MIN_CONNECTIONS = 3           # 最小连接数
    SIMILARITY_THRESHOLD = 0.6    # 相似度阈值
    HUB_NODE_COUNT = 5            # 核心枢纽节点数
    CLEANUP_INTERVAL = 1800       # 清理间隔30分钟（秒）
    
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        self._creation_log: List[Dict] = []
        self._last_cleanup = time.time()
        self._hub_nodes: Set[str] = set()
        self._stats = {
            "nodes_created": 0,
            "auto_connected": 0,
            "hub_connected": 0,
            "isolated_detected": 0,
            "isolated_rescued": 0
        }
    
    def add_node_with_prevention(self, node_id: str, **attributes) -> Dict[str, Any]:
        """
        创建节点时自动预防孤立
        
        Returns:
            {"success": bool, "connections": int, "node_id": str}
        """
        # 1. 创建节点
        self.kg.add_node(node_id, **attributes)
        self._stats["nodes_created"] += 1
        
        # 2. 自动查找语义相似的现有节点
        similar_nodes = self._find_semantically_similar(node_id, attributes)
        
        # 3. 建立连接（至少3个）
        connections = 0
        for similar_id, similarity in similar_nodes[:10]:  # 最多检查前10个
            if similarity > self.SIMILARITY_THRESHOLD:
                self.kg.add_edge(node_id, similar_id, 
                               weight=similarity, 
                               relation="semantic_similar",
                               auto_created=True)
                connections += 1
                self._stats["auto_connected"] += 1
                
                if connections >= self.MIN_CONNECTIONS:
                    break
        
        # 4. 如果连接不足，连接到核心枢纽节点
        if connections < self.MIN_CONNECTIONS:
            hub_connections = self._connect_to_hubs(node_id, self.MIN_CONNECTIONS - connections)
            connections += hub_connections
            self._stats["hub_connected"] += hub_connections
        
        # 5. 记录创建日志
        log_entry = {
            "node_id": node_id,
            "timestamp": time.time(),
            "connections": connections,
            "attributes": list(attributes.keys()),
            "source": attributes.get("source", "unknown")
        }
        self._creation_log.append(log_entry)
        
        # 6. 更新枢纽节点（如果这个节点连接数很高）
        self._update_hub_nodes()
        
        return {
            "success": True,
            "connections": connections,
            "node_id": node_id,
            "auto_connected": connections > 0
        }
    
    def _find_semantically_similar(self, node_id: str, attributes: Dict) -> List[Tuple[str, float]]:
        """查找语义相似的现有节点"""
        similar_nodes = []
        
        # 获取节点的文本表示
        node_text = self._get_node_text(node_id, attributes)
        
        for existing_id in self.kg.graph.nodes():
            if existing_id == node_id:
                continue
            
            existing_attrs = self.kg.graph.nodes[existing_id]
            existing_text = self._get_node_text(existing_id, existing_attrs)
            
            # 计算简单相似度（基于关键词重叠）
            similarity = self._calculate_similarity(node_text, existing_text)
            
            if similarity > 0.3:  # 最低相似度
                similar_nodes.append((existing_id, similarity))
        
        # 按相似度排序
        similar_nodes.sort(key=lambda x: x[1], reverse=True)
        return similar_nodes
    
    def _get_node_text(self, node_id: str, attributes: Dict) -> str:
        """获取节点的文本表示用于比较"""
        text_parts = [str(node_id)]
        
        # 收集属性中的文本
        for key, value in attributes.items():
            if isinstance(value, str):
                text_parts.append(value)
            elif isinstance(value, (list, tuple)):
                text_parts.extend([str(v) for v in value])
        
        return " ".join(text_parts).lower()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度（简单Jaccard）"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _connect_to_hubs(self, node_id: str, needed: int) -> int:
        """连接到核心枢纽节点"""
        connections = 0
        
        # 确保有枢纽节点
        if not self._hub_nodes:
            self._update_hub_nodes()
        
        # 连接到枢纽
        for hub_id in list(self._hub_nodes)[:needed]:
            if hub_id != node_id:
                self.kg.add_edge(node_id, hub_id,
                               weight=0.5,
                               relation="hub_connection",
                               auto_created=True)
                connections += 1
        
        return connections
    
    def _update_hub_nodes(self):
        """更新核心枢纽节点（连接数最多的节点）"""
        if len(self.kg.graph.nodes()) < 10:
            return
        
        # 计算每个节点的度数
        degrees = [(n, self.kg.graph.degree(n)) for n in self.kg.graph.nodes()]
        degrees.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前N个作为枢纽
        self._hub_nodes = set(n for n, d in degrees[:self.HUB_NODE_COUNT])
    
    def periodic_cleanup(self) -> Dict[str, Any]:
        """定期清理和修复孤立节点"""
        current_time = time.time()
        
        # 检查是否到达清理间隔
        if current_time - self._last_cleanup < self.CLEANUP_INTERVAL:
            return {"skipped": True, "reason": "interval_not_met"}
        
        self._last_cleanup = current_time
        
        # 检测孤立节点
        isolated = self._detect_isolated_nodes()
        self._stats["isolated_detected"] += len(isolated)
        
        # 修复孤立节点
        rescued = 0
        for node_id in isolated:
            if self._rescue_isolated_node(node_id):
                rescued += 1
                self._stats["isolated_rescued"] += 1
        
        return {
            "isolated_detected": len(isolated),
            "isolated_rescued": rescued,
            "timestamp": current_time
        }
    
    def _detect_isolated_nodes(self) -> List[str]:
        """检测孤立节点（度数=0）"""
        isolated = []
        for node_id in self.kg.graph.nodes():
            if self.kg.graph.degree(node_id) == 0:
                isolated.append(node_id)
        return isolated
    
    def _rescue_isolated_node(self, node_id: str) -> bool:
        """修复孤立节点"""
        # 获取节点属性
        attrs = self.kg.graph.nodes[node_id]
        
        # 尝试语义匹配
        similar = self._find_semantically_similar(node_id, attrs)
        
        connected = False
        for similar_id, similarity in similar[:self.MIN_CONNECTIONS]:
            if similarity > self.SIMILARITY_THRESHOLD:
                self.kg.add_edge(node_id, similar_id,
                               weight=similarity,
                               relation="rescue_connection",
                               auto_created=True)
                connected = True
        
        # 如果还是没连接，连接到枢纽
        if not connected:
            self._connect_to_hubs(node_id, self.MIN_CONNECTIONS)
        
        return self.kg.graph.degree(node_id) > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        stats["hub_nodes"] = list(self._hub_nodes)
        stats["current_isolated"] = len(self._detect_isolated_nodes())
        stats["total_nodes"] = len(self.kg.graph.nodes())
        stats["isolation_rate"] = stats["current_isolated"] / max(stats["total_nodes"], 1)
        return stats
    
    def get_creation_log(self, limit: int = 100) -> List[Dict]:
        """获取创建日志"""
        return self._creation_log[-limit:]


# 便捷函数
def create_isolation_prevention(knowledge_graph) -> IsolatedNodePrevention:
    """创建孤立节点预防器"""
    return IsolatedNodePrevention(knowledge_graph)


# 测试代码
if __name__ == "__main__":
    # 模拟测试
    print("孤立节点预防器测试:")
    print("-" * 60)
    
    # 创建模拟知识图谱
    class MockKG:
        def __init__(self):
            self.graph = MockGraph()
        
        def add_node(self, node_id, **attrs):
            self.graph.nodes[node_id] = attrs
            self.graph.edges[node_id] = []
        
        def add_edge(self, n1, n2, **attrs):
            if n1 in self.graph.nodes and n2 in self.graph.nodes:
                self.graph.edges[n1].append((n2, attrs))
                self.graph.edges[n2].append((n1, attrs))
    
    class MockGraph:
        def __init__(self):
            self.nodes = {}
            self.edges = defaultdict(list)
        
        def degree(self, node):
            return len(self.edges.get(node, []))
        
        def number_of_nodes(self):
            return len(self.nodes)
    
    from collections import defaultdict
    
    kg = MockKG()
    prevention = IsolatedNodePrevention(kg)
    
    # 创建一些测试节点
    test_nodes = [
        ("node_1", {"type": "concept", "description": "machine learning algorithm"}),
        ("node_2", {"type": "concept", "description": "deep learning model"}),
        ("node_3", {"type": "concept", "description": "neural network architecture"}),
        ("node_4", {"type": "concept", "description": "data preprocessing technique"}),
    ]
    
    for node_id, attrs in test_nodes:
        result = prevention.add_node_with_prevention(node_id, **attrs)
        print(f"Created {node_id}: {result['connections']} connections")
    
    # 创建新节点，应该自动连接到相似节点
    new_node = ("node_5", {"type": "concept", "description": "ml training method"})
    result = prevention.add_node_with_prevention(new_node[0], **new_node[1])
    print(f"\nCreated {new_node[0]} (similar): {result['connections']} connections")
    
    # 统计
    stats = prevention.get_stats()
    print(f"\n统计:")
    print(f"  总节点: {stats['total_nodes']}")
    print(f"  孤立节点: {stats['current_isolated']}")
    print(f"  孤立率: {stats['isolation_rate']:.1%}")
