"""
跨会话记忆系统 - 图查询引擎
Cross-Session Memory System - Graph Query Engine

版本: 1.0.0
日期: 2025-11-14
阶段: Phase 3 - 知识图谱构建
"""

import logging
from typing import List, Dict, Set, Optional, Any, Tuple, Union
import networkx as nx
from core.knowledge_graph import ArchitectureKnowledgeGraph


logger = logging.getLogger(__name__)


class GraphQueryEngineError(Exception):
    """图查询引擎异常"""


class GraphQueryEngine:
    """图查询引擎类，提供对知识图谱的高级查询功能"""

    def __init__(self, knowledge_graph: ArchitectureKnowledgeGraph) -> None:
        """
        初始化图查询引擎

        Args:
            knowledge_graph: 知识图谱对象

        Raises:
            TypeError: 当knowledge_graph不是ArchitectureKnowledgeGraph实例时
        """
        if not isinstance(knowledge_graph, ArchitectureKnowledgeGraph):
            raise TypeError("knowledge_graph must be an instance of ArchitectureKnowledgeGraph")
        
        self.kg = knowledge_graph

    # ==================== 路径查询 ====================

    def find_shortest_path(
        self, source: str, target: str, max_length: Optional[int] = None
    ) -> Optional[List[str]]:
        """
        查找最短路径

        Args:
            source: 源节点ID
            target: 目标节点ID
            max_length: 最大路径长度限制（边数）

        Returns:
            路径节点列表，不存在返回None

        Raises:
            GraphQueryEngineError: 当源或目标节点不存在时
        """
        source = source.lower()
        target = target.lower()

        if not self.kg.has_node(source):
            raise GraphQueryEngineError(f"Source node {source} does not exist")
        if not self.kg.has_node(target):
            raise GraphQueryEngineError(f"Target node {target} does not exist")

        try:
            path = nx.shortest_path(self.kg.graph, source, target)
            
            # 检查路径长度限制
            if max_length is not None and len(path) - 1 > max_length:
                return None
                
            return path
        except nx.NetworkXNoPath:
            return None

    def find_all_paths(
        self,
        source: str,
        target: str,
        max_length: int = 5,
        cutoff: Optional[int] = None,
    ) -> List[List[str]]:
        """
        查找所有简单路径

        Args:
            source: 源节点ID
            target: 目标节点ID
            max_length: 最大路径长度（跳数）
            cutoff: 返回路径数量上限

        Returns:
            路径列表

        Raises:
            GraphQueryEngineError: 当源或目标节点不存在时
        """
        source = source.lower()
        target = target.lower()

        if not self.kg.has_node(source):
            raise GraphQueryEngineError(f"Source node {source} does not exist")
        if not self.kg.has_node(target):
            raise GraphQueryEngineError(f"Target node {target} does not exist")

        try:
            paths_iter = nx.all_simple_paths(self.kg.graph, source, target, cutoff=max_length)
            paths = list(paths_iter)

            if cutoff:
                paths = paths[:cutoff]

            return paths
        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            logger.error(f"Error finding all paths from {source} to {target}: {e}")
            return []

    def find_paths_by_relation(
        self, source: str, target: str, relation_types: List[str]
    ) -> List[List[Dict[str, Any]]]:
        """
        按关系类型查找路径

        Args:
            source: 源节点ID
            target: 目标节点ID
            relation_types: 允许的关系类型列表

        Returns:
            过滤后的路径列表，每个路径包含边信息
        """
        if not relation_types:
            return []
            
        source = source.lower()
        target = target.lower()

        all_paths = self.find_all_paths(source, target, max_length=5)
        filtered_paths = []

        relation_set = set(relation_types)  # 使用集合提高查找效率

        for path in all_paths:
            path_edges = []
            valid = True

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_data = self.kg.get_edge(u, v)

                if edge_data and edge_data.get("relation_type") in relation_set:
                    path_edges.append({
                        "source": u,
                        "target": v,
                        "relation": edge_data["relation_type"]
                    })
                else:
                    valid = False
                    break

            if valid:
                filtered_paths.append(path_edges)

        return filtered_paths

    # ==================== N跳邻居查询 ====================

    def get_n_hop_neighbors(
        self, node_id: str, n: int = 1, direction: str = "both"
    ) -> Dict[int, Set[str]]:
        """
        获取N跳邻居

        Args:
            node_id: 节点ID
            n: 跳数（必须 >= 1）
            direction: 遍历方向 ("in"/"out"/"both")

        Returns:
            邻居字典 {跳数: {节点集合}, ...}

        Raises:
            GraphQueryEngineError: 当节点不存在时
            ValueError: 当n < 1或direction无效时
        """
        if n < 1:
            raise ValueError("n must be >= 1")
            
        if direction not in ("in", "out", "both"):
            raise ValueError("direction must be 'in', 'out', or 'both'")
            
        node_id = node_id.lower()

        if not self.kg.has_node(node_id):
            raise GraphQueryEngineError(f"Node {node_id} does not exist")

        # 根据方向选择图
        if direction == "out":
            graph = self.kg.graph
        elif direction == "in":
            graph = self.kg.graph.reverse(copy=False)
        else:  # both
            graph = self.kg.graph.to_undirected()

        neighbors_by_hop: Dict[int, Set[str]] = {}
        visited: Set[str] = {node_id}

        for hop in range(1, n + 1):
            current_level: Set[str] = set()

            # 获取上一跳的节点
            prev_level = neighbors_by_hop[hop - 1] if hop > 1 else {node_id}

            # 扩展邻居
            for node in prev_level:
                current_level.update(
                    neighbor for neighbor in graph.neighbors(node)
                    if neighbor not in visited
                )

            if not current_level:  # 提前终止，没有更多邻居
                break
                
            visited.update(current_level)
            neighbors_by_hop[hop] = current_level

        return neighbors_by_hop

    def get_common_neighbors(
        self, node1: str, node2: str, direction: str = "both"
    ) -> Set[str]:
        """
        获取两个节点的共同邻居

        Args:
            node1: 第一个节点ID
            node2: 第二个节点ID
            direction: 方向 ("in"/"out"/"both")

        Returns:
            共同邻居集合

        Raises:
            GraphQueryEngineError: 当任一节点不存在时
        """
        node1 = node1.lower()
        node2 = node2.lower()

        if not self.kg.has_node(node1):
            raise GraphQueryEngineError(f"Node {node1} does not exist")
        if not self.kg.has_node(node2):
            raise GraphQueryEngineError(f"Node {node2} does not exist")

        neighbors1 = self.kg.get_neighbors(node1, direction)
        neighbors2 = self.kg.get_neighbors(node2, direction)

        if direction == "both":
            set1 = set(neighbors1["in"]) | set(neighbors1["out"])
            set2 = set(neighbors2["in"]) | set(neighbors2["out"])
        elif direction == "in":
            set1 = set(neighbors1["in"])
            set2 = set(neighbors2["in"])
        else:  # out
            set1 = set(neighbors1["out"])
            set2 = set(neighbors2["out"])

        return set1 & set2

    # ==================== 图算法 ====================

    def calculate_pagerank(
        self, max_iter: int = 100, tol: float = 1e-6
    ) -> Dict[str, float]:
        """
        计算PageRank值

        Args:
            max_iter: 最大迭代次数
            tol: 收敛容差

        Returns:
            PageRank分数字典 (节点 -> 分数)
        """
        try:
            return nx.pagerank(self.kg.graph, max_iter=max_iter, tol=tol)
        except nx.PowerIterationFailedConvergence:
            logger.warning("PageRank did not converge within %d iterations", max_iter)
            return {}

    def calculate_betweenness_centrality(
        self, normalized: bool = True
    ) -> Dict[str, float]:
        """
        计算介数中心性

        Args:
            normalized: 是否进行归一化处理

        Returns:
            介数中心性字典 (节点 -> 中心性值)
        """
        try:
            return nx.betweenness_centrality(self.kg.graph, normalized=normalized)
        except Exception as e:
            logger.error("Betweenness centrality calculation failed: %s", e)
            return {}

    def calculate_closeness_centrality(
        self, distance: Optional[str] = None
    ) -> Dict[str, float]:
        """
        计算紧密中心性

        Args:
            distance: 用作距离度量的边属性名

        Returns:
            紧密中心性字典 (节点 -> 中心性值)
        """
        try:
            return nx.closeness_centrality(self.kg.graph, distance=distance)
        except Exception as e:
            logger.error("Closeness centrality calculation failed: %s", e)
            return {}

    def calculate_degree_centrality(self) -> Dict[str, float]:
        """
        计算度中心性

        Returns:
            度中心性字典 (节点 -> 中心性值)
        """
        try:
            return nx.degree_centrality(self.kg.graph)
        except Exception as e:
            logger.error("Degree centrality calculation failed: %s", e)
            return {}

    def get_centrality_scores(self) -> Dict[str, Dict[str, float]]:
        """
        获取所有类型的中心性指标

        Returns:
            包含多种中心性指标的嵌套字典
        """
        return {
            "pagerank": self.calculate_pagerank(),
            "betweenness": self.calculate_betweenness_centrality(),
            "closeness": self.calculate_closeness_centrality(),
            "degree": self.calculate_degree_centrality(),
        }

    # ==================== 社区发现 ====================

    def detect_communities_louvain(self) -> List[Set[str]]:
        """
        使用标签传播算法模拟Louvain社区检测

        Returns:
            社区集合列表
        """
        try:
            undirected = self.kg.graph.to_undirected()
            communities = nx.community.label_propagation_communities(undirected)
            return [set(c) for c in communities]
        except Exception as e:
            logger.error("Louvain-style community detection failed: %s", e)
            return []

    def detect_communities_greedy(self) -> List[Set[str]]:
        """
        使用贪心模块度优化算法检测社区

        Returns:
            社区集合列表
        """
        try:
            undirected = self.kg.graph.to_undirected()
            communities = nx.community.greedy_modularity_communities(undirected)
            return [set(c) for c in communities]
        except Exception as e:
            logger.error("Greedy modularity community detection failed: %s", e)
            return []

    def calculate_modularity(self, communities: List[Set[str]]) -> float:
        """
        计算给定社区划分的模块度

        Args:
            communities: 社区集合列表

        Returns:
            模块度值 (0-1之间，越高表示社区结构越明显)
        """
        if not communities:
            return 0.0
            
        try:
            undirected = self.kg.graph.to_undirected()
            return nx.community.modularity(undirected, communities)
        except Exception as e:
            logger.error("Modularity calculation failed: %s", e)
            return 0.0

    # ==================== 子图匹配 ====================

    def find_nodes_by_attribute(self, attribute: str, value: Any) -> List[str]:
        """
        根据节点属性值查找节点

        Args:
            attribute: 属性名称
            value: 要匹配的属性值

        Returns:
            匹配的节点ID列表
        """
        if not isinstance(attribute, str):
            raise TypeError("attribute must be a string")
            
        matching_nodes = [
            node_id for node_id, node_data in self.kg.graph.nodes(data=True)
            if node_data.get(attribute) == value
        ]
        
        return matching_nodes

    def find_edges_by_attribute(
        self, attribute: str, value: Any
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        根据边属性值查找边

        Args:
            attribute: 属性名称
            value: 要匹配的属性值

        Returns:
            匹配的边列表 [(源, 目标, 数据), ...]
        """
        if not isinstance(attribute, str):
            raise TypeError("attribute must be a string")
            
        matching_edges = [
            (u, v, edge_data) for u, v, edge_data in self.kg.graph.edges(data=True)
            if edge_data.get(attribute) == value
        ]
        
        return matching_edges

    def pattern_match_triangle(self) -> List[Tuple[str, str, str]]:
        """
        查找图中的三角形模式（三元闭包）

        Returns:
            唯一三角形列表，每个三角形按节点名排序
        """
        try:
            undirected = self.kg.graph.to_undirected()
            triangles: List[Tuple[str, str, str]] = []
            seen = set()

            for node in undirected.nodes():
                neighbors = list(undirected.neighbors(node))
                for i, n1 in enumerate(neighbors):
                    for n2 in neighbors[i + 1:]:
                        if undirected.has_edge(n1, n2):
                            triangle = tuple(sorted([node, n1, n2]))
                            if triangle not in seen:
                                seen.add(triangle)
                                triangles.append(triangle)

            return triangles
        except Exception as e:
            logger.error("Triangle pattern matching failed: %s", e)
            return []

    def pattern_match_star(self, center: str) -> List[str]:
        """
        查找星形模式（以指定节点为中心）

        Args:
            center: 中心节点ID

        Returns:
            叶子节点（直接连接的邻居）列表

        Raises:
            GraphQueryEngineError: 当中心节点不存在时
        """
        center = center.lower()

        if not self.kg.has_node(center):
            raise GraphQueryEngineError(f"Node {center} does not exist")

        neighbors = self.kg.get_neighbors(center, direction="both")
        all_neighbors = set(neighbors["in"]) | set(neighbors["out"])

        return list(all_neighbors)

    # ==================== 高级查询 ====================

    def find_strongly_connected_components(self) -> List[Set[str]]:
        """
        查找有向图中的强连通分量

        Returns:
            强连通分量列表
        """
        try:
            sccs = nx.strongly_connected_components(self.kg.graph)
            return [set(c) for c in sccs]
        except Exception as e:
            logger.error("Strongly connected components calculation failed: %s", e)
            return []

    def is_dag(self) -> bool:
        """
        检查图是否为有向无环图（DAG）

        Returns:
            True如果图是DAG，否则False
        """
        return nx.is_directed_acyclic_graph(self.kg.graph)

    def topological_sort(self) -> List[str]:
        """
        对DAG进行拓扑排序

        Returns:
            拓扑排序的节点列表

        Raises:
            GraphQueryEngineError: 如果图不是DAG
        """
        if not self.is_dag():
            raise GraphQueryEngineError("Graph is not a DAG, cannot perform topological sort")

        try:
            return list(nx.topological_sort(self.kg.graph))
        except Exception as e:
            logger.error("Topological sort failed: %s", e)
            raise GraphQueryEngineError(f"Topological sort failed: {e}")

    def calculate_clustering_coefficient(self, node_id: Optional[str] = None) -> float:
        """
        计算聚类系数

        Args:
            node_id: 指定节点ID，None表示计算平均聚类系数

        Returns:
            聚类系数值

        Raises:
            GraphQueryEngineError: 当指定节点不存在时
        """
        # 转换为无向简单图（clustering不支持MultiGraph）
        undirected = self.kg.graph.to_undirected()
        simple_graph = nx.Graph(undirected)

        if node_id is not None:
            node_id = node_id.lower()
            if not self.kg.has_node(node_id):
                raise GraphQueryEngineError(f"Node {node_id} does not exist")
            return nx.clustering(simple_graph, node_id)
        else:
            return nx.average_clustering(simple_graph)

    def get_graph_density(self) -> float:
        """
        计算图密度

        Returns:
            图密度值（0-1之间）
        """
        try:
            return nx.density(self.kg.graph)
        except Exception as e:
            logger.error("Graph density calculation failed: %s", e)
            return 0.0

    def get_graph_diameter(self) -> int:
        """
        计算图直径（最长的最短路径长度）

        Returns:
            图直径

        Raises:
            GraphQueryEngineError: 如果图不连通
        """
        try:
            undirected = self.kg.graph.to_undirected()
            
            if not nx.is_connected(undirected):
                raise GraphQueryEngineError("Graph is not connected, diameter undefined")
                
            return nx.diameter(undirected)
        except Exception as e:
            logger.error("Graph diameter calculation failed: %s", e)
            raise GraphQueryEngineError(f"Graph diameter calculation failed: {e}")


# 示例用法
if __name__ == "__main__":
    # Note: This example assumes the existence of Entity, EntityType, Relationship, etc.
    # In actual usage, these would be imported from appropriate modules
    pass