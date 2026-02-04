import networkx as nx
import json
import os
import time
import random
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

class ArchitectureKnowledgeGraph:
    def __init__(self, data_dir: str = "data/knowledge"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.graph_file = os.path.join(self.data_dir, "arch_graph.json")
        self.graph = nx.DiGraph()
        self._load_graph()

    def _load_graph(self):
        if os.path.exists(self.graph_file):
            try:
                with open(self.graph_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # [FIX 2026-01-15] NetworkX 3.x 默认期望 'edges' 键，但旧数据使用 'links'
                    # 自动检测并使用正确的参数
                    edges_key = 'edges' if 'edges' in data else 'links'
                    self.graph = nx.node_link_graph(data, edges=edges_key)
                    print(f"   [KnowledgeGraph] Loaded {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            except Exception as e:
                print(f"Failed to load graph: {e}")
                self.graph = nx.DiGraph()

    def _merge_with_disk(self):
        """[FIX 2026-01-15] 保存前合并磁盘上可能被其他进程更新的数据"""
        if os.path.exists(self.graph_file):
            try:
                with open(self.graph_file, 'r', encoding='utf-8') as f:
                    disk_data = json.load(f)
                edges_key = 'edges' if 'edges' in disk_data else 'links'
                disk_graph = nx.node_link_graph(disk_data, edges=edges_key)
                # 合并：保留两边的所有节点和边
                self.graph = nx.compose(disk_graph, self.graph)
            except Exception as e:
                pass  # 如果无法读取磁盘文件，继续使用内存中的图

    def save_graph(self):
        """带文件锁的安全保存机制"""
        # [FIX 2026-01-15] 保存前先合并磁盘上的数据，防止覆盖其他进程的更新
        self._merge_with_disk()
        # [FIX 2026-01-15] 使用 edges='links' 保持与历史数据格式一致
        data = nx.node_link_data(self.graph, edges='links')
        temp_file = self.graph_file + ".tmp"
        lock_file = Path(self.graph_file).with_suffix('.lock')
        
        # ✅ [FIX 2026-01-09] 添加文件锁防止并发写入
        max_retries = 15
        for attempt in range(max_retries):
            try:
                # 尝试获取锁（创建.lock文件）
                if not lock_file.exists():
                    lock_file.touch()
                    lock_acquired = True
                else:
                    # 检查锁文件是否过期（>5秒视为死锁）
                    lock_age = time.time() - lock_file.stat().st_mtime
                    if lock_age > 5.0:
                        lock_file.unlink()
                        lock_file.touch()
                        lock_acquired = True
                    else:
                        lock_acquired = False
                
                if lock_acquired:
                    try:
                        # 写入临时文件
                        with open(temp_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        
                        # 原子替换
                        if os.path.exists(self.graph_file):
                            os.replace(temp_file, self.graph_file)
                        else:
                            os.rename(temp_file, self.graph_file)
                        break
                    finally:
                        # 释放锁
                        if lock_file.exists():
                            try:
                                lock_file.unlink()
                            except:
                                pass
                else:
                    # 等待并重试
                    if attempt == max_retries - 1:
                        print(f"   [Memory] ⚠️ 无法获取文件锁 {self.graph_file} (超时)")
                    time.sleep(0.05 * (attempt + 1) + random.random() * 0.05)
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"   [Memory] ⚠️ 保存图失败: {e}")
                    # 清理残留文件
                    for cleanup_file in [temp_file, lock_file]:
                        try:
                            if Path(cleanup_file).exists():
                                Path(cleanup_file).unlink()
                        except:
                            pass
                time.sleep(0.05 * (attempt + 1))

    def add_node(self, node_id: str, **attributes):
        """
        Add a generic node with attributes to the graph.
        """
        self.graph.add_node(node_id, **attributes)
        self.save_graph()

    def add_decision_node(self, context: str, decision: str, outcome: float, metadata: Dict[str, Any] = None):
        """
        Add a decision node to the graph.
        Node ID: timestamp_hash
        Edges: Connect to similar contexts or previous decisions.
        """
        node_id = f"DECISION_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(decision)}"
        
        self.graph.add_node(node_id, 
                            type="decision",
                            context=context, 
                            decision=decision, 
                            outcome=outcome,
                            timestamp=datetime.now().isoformat(),
                            metadata=metadata or {})
        
        # Simple linking logic: link to previous node if exists
        nodes = list(self.graph.nodes())
        if len(nodes) > 1:
            prev_node = nodes[-2]
            self.graph.add_edge(prev_node, node_id, relation="followed_by")
            
        self.save_graph()
        return node_id

    def query_best_practice(self, context_keyword: str) -> List[Dict[str, Any]]:
        """
        Find successful decisions related to a context keyword.
        """
        candidates = []
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'decision' and context_keyword in data.get('context', ''):
                candidates.append(data)
        
        # Return top 3 successful decisions
        return sorted(candidates, key=lambda x: x.get('outcome', 0), reverse=True)[:3]

    def get_stats(self):
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges()
        }

    # --- Compatibility Methods for Reasoner ---
    def has_node(self, node_id: str) -> bool:
        return self.graph.has_node(node_id)

    def get_neighbors(self, node_id: str, direction: str = "out") -> Dict[str, List[str]]:
        if direction == "out":
            return {"out": list(self.graph.successors(node_id))}
        elif direction == "in":
            return {"in": list(self.graph.predecessors(node_id))}
        return {"out": [], "in": []}

    def get_edge(self, source: str, target: str) -> Dict[str, Any]:
        return self.graph.get_edge_data(source, target) or {}
