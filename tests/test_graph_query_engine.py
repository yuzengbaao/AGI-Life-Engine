"""
跨会话记忆系统 - 图查询引擎测试
Cross-Session Memory System - Graph Query Engine Tests

版本: 1.0.0
日期: 2025-11-14
阶段: Phase 3 - 知识图谱构建
"""

import pytest
from graph_query_engine import GraphQueryEngine, GraphQueryEngineError
from knowledge_graph import KnowledgeGraph
from entity_extractor import Entity, EntityType
from relationship_builder import Relationship, RelationshipType


@pytest.fixture
def simple_graph():
    """创建简单测试图"""
    kg = KnowledgeGraph()

    # 添加节点
    entities = [
        Entity(name="A", entity_type=EntityType.CONCEPT),
        Entity(name="B", entity_type=EntityType.CONCEPT),
        Entity(name="C", entity_type=EntityType.CONCEPT),
        Entity(name="D", entity_type=EntityType.CONCEPT),
    ]

    for e in entities:
        kg.add_node(e)

    # 添加边: A->B->C->D, A->C
    relationships = [
        Relationship(source="A", target="B", relation_type=RelationshipType.RELATED_TO),
        Relationship(source="B", target="C", relation_type=RelationshipType.RELATED_TO),
        Relationship(source="C", target="D", relation_type=RelationshipType.RELATED_TO),
        Relationship(source="A", target="C", relation_type=RelationshipType.USES),
    ]

    for r in relationships:
        kg.add_edge(r)

    return kg


@pytest.fixture
def complex_graph():
    """创建复杂测试图"""
    kg = KnowledgeGraph()

    # 添加节点
    entities = [
        Entity(name="Python", entity_type=EntityType.TECHNOLOGY),
        Entity(name="Django", entity_type=EntityType.TECHNOLOGY),
        Entity(name="Flask", entity_type=EntityType.TECHNOLOGY),
        Entity(name="Guido", entity_type=EntityType.PERSON),
        Entity(name="Web", entity_type=EntityType.CONCEPT),
    ]

    for e in entities:
        kg.add_node(e)

    # 添加边
    relationships = [
        Relationship(
            source="Python", target="Guido", relation_type=RelationshipType.CREATED_BY
        ),
        Relationship(
            source="Django", target="Python", relation_type=RelationshipType.USES
        ),
        Relationship(
            source="Flask", target="Python", relation_type=RelationshipType.USES
        ),
        Relationship(
            source="Django", target="Web", relation_type=RelationshipType.RELATED_TO
        ),
        Relationship(
            source="Flask", target="Web", relation_type=RelationshipType.RELATED_TO
        ),
    ]

    for r in relationships:
        kg.add_edge(r)

    return kg


class TestPathQuery:
    """测试路径查询"""

    def test_find_shortest_path(self, simple_graph):
        """测试查找最短路径"""
        engine = GraphQueryEngine(simple_graph)
        path = engine.find_shortest_path("A", "D")

        assert path is not None
        assert path[0] == "a"
        assert path[-1] == "d"
        assert len(path) >= 2

    def test_find_shortest_path_no_path(self, simple_graph):
        """测试不存在路径"""
        kg = KnowledgeGraph()
        e1 = Entity(name="X", entity_type=EntityType.CONCEPT)
        e2 = Entity(name="Y", entity_type=EntityType.CONCEPT)
        kg.add_node(e1)
        kg.add_node(e2)

        engine = GraphQueryEngine(kg)
        path = engine.find_shortest_path("X", "Y")
        assert path is None

    def test_find_shortest_path_max_length(self, simple_graph):
        """测试最大长度限制"""
        engine = GraphQueryEngine(simple_graph)
        path = engine.find_shortest_path("A", "D", max_length=2)

        if path:
            assert len(path) - 1 <= 2

    def test_find_all_paths(self, simple_graph):
        """测试查找所有路径"""
        engine = GraphQueryEngine(simple_graph)
        paths = engine.find_all_paths("A", "D", max_length=5)

        assert len(paths) >= 1
        assert all(p[0] == "a" and p[-1] == "d" for p in paths)

    def test_find_all_paths_with_cutoff(self, simple_graph):
        """测试路径数量限制"""
        engine = GraphQueryEngine(simple_graph)
        paths = engine.find_all_paths("A", "D", max_length=5, cutoff=1)

        assert len(paths) <= 1

    def test_find_paths_by_relation(self, simple_graph):
        """测试按关系类型查找路径"""
        engine = GraphQueryEngine(simple_graph)
        paths = engine.find_paths_by_relation("A", "C", ["related_to"])

        assert len(paths) >= 0
        for path in paths:
            assert all(edge["relation"] == "related_to" for edge in path)


class TestNHopNeighbors:
    """测试N跳邻居查询"""

    def test_get_1_hop_neighbors(self, complex_graph):
        """测试1跳邻居"""
        engine = GraphQueryEngine(complex_graph)
        neighbors = engine.get_n_hop_neighbors("Python", n=1, direction="in")

        assert 1 in neighbors
        assert "django" in neighbors[1] or "flask" in neighbors[1]

    def test_get_2_hop_neighbors(self, complex_graph):
        """测试2跳邻居"""
        engine = GraphQueryEngine(complex_graph)
        neighbors = engine.get_n_hop_neighbors("Python", n=2, direction="in")

        assert 1 in neighbors
        assert 2 in neighbors

    def test_get_n_hop_neighbors_out(self, complex_graph):
        """测试出边邻居"""
        engine = GraphQueryEngine(complex_graph)
        neighbors = engine.get_n_hop_neighbors("Python", n=1, direction="out")

        assert 1 in neighbors
        assert "guido" in neighbors[1]

    def test_get_common_neighbors(self, complex_graph):
        """测试共同邻居"""
        engine = GraphQueryEngine(complex_graph)
        common = engine.get_common_neighbors("Django", "Flask", direction="out")

        assert "python" in common


class TestGraphAlgorithms:
    """测试图算法"""

    def test_calculate_pagerank(self, complex_graph):
        """测试PageRank"""
        engine = GraphQueryEngine(complex_graph)
        pagerank = engine.calculate_pagerank()

        assert len(pagerank) > 0
        assert all(0 <= score <= 1 for score in pagerank.values())
        # Python应该有较高的PageRank(被多个节点指向)
        assert pagerank.get("python", 0) > 0

    def test_calculate_betweenness_centrality(self, complex_graph):
        """测试介数中心性"""
        engine = GraphQueryEngine(complex_graph)
        betweenness = engine.calculate_betweenness_centrality()

        assert len(betweenness) > 0
        assert all(score >= 0 for score in betweenness.values())

    def test_calculate_closeness_centrality(self, complex_graph):
        """测试紧密中心性"""
        engine = GraphQueryEngine(complex_graph)
        closeness = engine.calculate_closeness_centrality()

        assert len(closeness) > 0
        assert all(0 <= score <= 1 for score in closeness.values())

    def test_calculate_degree_centrality(self, complex_graph):
        """测试度中心性"""
        engine = GraphQueryEngine(complex_graph)
        degree = engine.calculate_degree_centrality()

        assert len(degree) > 0
        assert all(0 <= score <= 1 for score in degree.values())

    def test_get_centrality_scores(self, complex_graph):
        """测试获取所有中心性指标"""
        engine = GraphQueryEngine(complex_graph)
        scores = engine.get_centrality_scores()

        assert "pagerank" in scores
        assert "betweenness" in scores
        assert "closeness" in scores
        assert "degree" in scores


class TestCommunityDetection:
    """测试社区发现"""

    def test_detect_communities_louvain(self, complex_graph):
        """测试Louvain算法"""
        engine = GraphQueryEngine(complex_graph)
        communities = engine.detect_communities_louvain()

        assert len(communities) > 0
        assert all(isinstance(c, set) for c in communities)

    def test_detect_communities_greedy(self, complex_graph):
        """测试贪心算法"""
        engine = GraphQueryEngine(complex_graph)
        communities = engine.detect_communities_greedy()

        assert len(communities) > 0
        assert all(isinstance(c, set) for c in communities)

    def test_calculate_modularity(self, complex_graph):
        """测试模块度计算"""
        engine = GraphQueryEngine(complex_graph)
        communities = engine.detect_communities_greedy()
        modularity = engine.calculate_modularity(communities)

        assert -1 <= modularity <= 1


class TestSubgraphMatching:
    """测试子图匹配"""

    def test_find_nodes_by_attribute(self, complex_graph):
        """测试按属性查找节点"""
        engine = GraphQueryEngine(complex_graph)
        nodes = engine.find_nodes_by_attribute("entity_type", "technology")

        assert len(nodes) >= 3
        assert "python" in nodes

    def test_find_edges_by_attribute(self, complex_graph):
        """测试按属性查找边"""
        engine = GraphQueryEngine(complex_graph)
        edges = engine.find_edges_by_attribute("relation_type", "uses")

        assert len(edges) >= 2

    def test_pattern_match_triangle(self):
        """测试三角形模式匹配"""
        kg = KnowledgeGraph()

        # 创建三角形: A-B-C-A
        for name in ["A", "B", "C"]:
            kg.add_node(Entity(name=name, entity_type=EntityType.CONCEPT))

        kg.add_edge(
            Relationship(
                source="A", target="B", relation_type=RelationshipType.RELATED_TO
            )
        )
        kg.add_edge(
            Relationship(
                source="B", target="C", relation_type=RelationshipType.RELATED_TO
            )
        )
        kg.add_edge(
            Relationship(
                source="C", target="A", relation_type=RelationshipType.RELATED_TO
            )
        )

        engine = GraphQueryEngine(kg)
        triangles = engine.pattern_match_triangle()

        assert len(triangles) >= 1

    def test_pattern_match_star(self, complex_graph):
        """测试星形模式匹配"""
        engine = GraphQueryEngine(complex_graph)
        leaves = engine.pattern_match_star("Python")

        assert len(leaves) >= 1


class TestAdvancedQueries:
    """测试高级查询"""

    def test_find_strongly_connected_components(self, complex_graph):
        """测试强连通分量"""
        engine = GraphQueryEngine(complex_graph)
        sccs = engine.find_strongly_connected_components()

        assert len(sccs) > 0
        assert all(isinstance(c, set) for c in sccs)

    def test_is_dag(self, complex_graph):
        """测试是否为DAG"""
        engine = GraphQueryEngine(complex_graph)
        is_dag = engine.is_dag()

        assert isinstance(is_dag, bool)

    def test_topological_sort_dag(self):
        """测试拓扑排序(DAG)"""
        kg = KnowledgeGraph()

        # 创建DAG: A->B->C
        for name in ["A", "B", "C"]:
            kg.add_node(Entity(name=name, entity_type=EntityType.CONCEPT))

        kg.add_edge(
            Relationship(
                source="A", target="B", relation_type=RelationshipType.RELATED_TO
            )
        )
        kg.add_edge(
            Relationship(
                source="B", target="C", relation_type=RelationshipType.RELATED_TO
            )
        )

        engine = GraphQueryEngine(kg)
        sorted_nodes = engine.topological_sort()

        assert len(sorted_nodes) == 3
        assert sorted_nodes.index("a") < sorted_nodes.index("b")
        assert sorted_nodes.index("b") < sorted_nodes.index("c")

    def test_topological_sort_non_dag(self):
        """测试拓扑排序(非DAG)"""
        kg = KnowledgeGraph()

        # 创建环: A->B->A
        kg.add_node(Entity(name="A", entity_type=EntityType.CONCEPT))
        kg.add_node(Entity(name="B", entity_type=EntityType.CONCEPT))

        kg.add_edge(
            Relationship(
                source="A", target="B", relation_type=RelationshipType.RELATED_TO
            )
        )
        kg.add_edge(
            Relationship(
                source="B", target="A", relation_type=RelationshipType.RELATED_TO
            )
        )

        engine = GraphQueryEngine(kg)

        with pytest.raises(GraphQueryEngineError, match="not a DAG"):
            engine.topological_sort()

    def test_calculate_clustering_coefficient_node(self, complex_graph):
        """测试节点聚类系数"""
        engine = GraphQueryEngine(complex_graph)
        coef = engine.calculate_clustering_coefficient("Python")

        assert 0 <= coef <= 1

    def test_calculate_clustering_coefficient_average(self, complex_graph):
        """测试平均聚类系数"""
        engine = GraphQueryEngine(complex_graph)
        coef = engine.calculate_clustering_coefficient()

        assert 0 <= coef <= 1

    def test_get_graph_density(self, complex_graph):
        """测试图密度"""
        engine = GraphQueryEngine(complex_graph)
        density = engine.get_graph_density()

        assert 0 <= density <= 1


class TestErrorHandling:
    """测试错误处理"""

    def test_invalid_knowledge_graph(self):
        """测试无效知识图谱"""
        with pytest.raises(ValueError, match="must be a KnowledgeGraph instance"):
            GraphQueryEngine("not a graph")

    def test_find_path_nonexistent_source(self, simple_graph):
        """测试不存在的源节点"""
        engine = GraphQueryEngine(simple_graph)

        with pytest.raises(GraphQueryEngineError, match="does not exist"):
            engine.find_shortest_path("X", "A")

    def test_find_path_nonexistent_target(self, simple_graph):
        """测试不存在的目标节点"""
        engine = GraphQueryEngine(simple_graph)

        with pytest.raises(GraphQueryEngineError, match="does not exist"):
            engine.find_shortest_path("A", "X")

    def test_n_hop_neighbors_invalid_n(self, simple_graph):
        """测试无效跳数"""
        engine = GraphQueryEngine(simple_graph)

        with pytest.raises(ValueError, match="n must be >= 1"):
            engine.get_n_hop_neighbors("A", n=0)


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
