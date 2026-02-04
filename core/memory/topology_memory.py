from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Edge:
    to_idx: int
    weight: float
    from_port: str = "semantic"
    to_port: str = "semantic"
    kind: str = "exc"
    usage: int = 0


class TopologicalMemoryCore:
    def __init__(
        self,
        max_degree: int = 24,
        min_edge_weight: float = 0.15,
        weight_decay: float = 0.999,
        inhibition_topn: int = 256,
        seed_topk: int = 32,
        diffusion_steps: int = 4,
        diffusion_gain: float = 0.85,
        self_retention: float = 0.65,
    ) -> None:
        self.max_degree = int(max_degree)
        self.min_edge_weight = float(min_edge_weight)
        self.weight_decay = float(weight_decay)
        self.inhibition_topn = int(inhibition_topn)
        self.seed_topk = int(seed_topk)
        self.diffusion_steps = int(diffusion_steps)
        self.diffusion_gain = float(diffusion_gain)
        self.self_retention = float(self_retention)

        self._adj: Dict[int, List[Edge]] = {}
        self._subgraphs: Dict[int, 'TopologicalMemoryCore'] = {}
        self._n: int = 0

    def size(self) -> int:
        return self._n

    @property
    def graph(self) -> Dict[int, List[Edge]]:
        return self._adj

    def get_edges(self, u: int) -> List[Edge]:
        """Get all outgoing edges from node u."""
        return self._adj.get(int(u), [])

    def get_edge_weight(self, u: int, v: int) -> float:
        """Get the weight of the edge from u to v, or 0.0 if not connected."""
        edges = self._adj.get(int(u), [])
        for e in edges:
            if e.to_idx == int(v):
                return e.weight
        return 0.0

    def ensure_size(self, n: int) -> None:
        n = int(n)
        if n <= self._n:
            return
        for i in range(self._n, n):
            self._adj.setdefault(i, [])
        self._n = n

    def connect(
        self,
        a: int,
        b: int,
        weight: float,
        from_port: str = "semantic",
        to_port: str = "semantic",
        kind: str = "exc",
        bidirectional: bool = True,
    ) -> None:
        if a == b:
            return
        self.ensure_size(max(a, b) + 1)
        w = float(weight)
        if not math.isfinite(w):
            return
        if w < self.min_edge_weight:
            return

        self._upsert_edge(a, b, w, from_port, to_port, kind)
        if bidirectional:
            self._upsert_edge(b, a, w, to_port, from_port, kind)

        self._cap_degree(a)
        if bidirectional:
            self._cap_degree(b)

    def disconnect(self, a: int, b: int, bidirectional: bool = True) -> None:
        if a in self._adj:
            self._adj[a] = [e for e in self._adj[a] if e.to_idx != b]
        if bidirectional and b in self._adj:
            self._adj[b] = [e for e in self._adj[b] if e.to_idx != a]

    def clone_subgraph(
        self,
        node_indices: List[int],
        noise_std: float = 0.01,
        copy_edges: bool = True,
    ) -> Dict[int, int]:
        uniq = [int(x) for x in dict.fromkeys(node_indices)]
        if not uniq:
            return {}

        base_n = self._n
        self.ensure_size(base_n + len(uniq))
        mapping: Dict[int, int] = {old: base_n + i for i, old in enumerate(uniq)}

        if copy_edges:
            s = set(uniq)
            for old_a in uniq:
                new_a = mapping[old_a]
                for e in self._adj.get(old_a, []):
                    if e.to_idx not in s:
                        continue
                    new_b = mapping[e.to_idx]
                    self.connect(
                        new_a,
                        new_b,
                        weight=float(e.weight),
                        from_port=e.from_port,
                        to_port=e.to_port,
                        kind=e.kind,
                        bidirectional=False,
                    )

        for new_idx in mapping.values():
            for e in self._adj.get(new_idx, []):
                e.weight = float(max(self.min_edge_weight, e.weight + random.gauss(0.0, noise_std)))

        for new_idx in mapping.values():
            self._cap_degree(new_idx)

        return mapping

    def embed_subgraph(self, node_idx: int, subgraph: "TopologicalMemoryCore") -> None:
        """
        Embed a subgraph into a specific node, creating a fractal structure.
        """
        self.ensure_size(node_idx + 1)
        self._subgraphs[int(node_idx)] = subgraph

    def get_subgraph(self, node_idx: int) -> Optional["TopologicalMemoryCore"]:
        """
        Retrieve the embedded subgraph for a node, if it exists.
        """
        return self._subgraphs.get(int(node_idx))

    def fractal_expand(self, node_idx: int, init_params: Optional[Dict[str, Any]] = None) -> "TopologicalMemoryCore":
        """
        Recursively expand a node into a new subgraph (mycelium-like growth).
        If a subgraph already exists, it returns it.
        """
        node_idx = int(node_idx)
        if node_idx in self._subgraphs:
            return self._subgraphs[node_idx]

        # Create new subgraph with similar properties to self, or custom params
        if init_params is None:
            # Inherit properties
            new_core = TopologicalMemoryCore(
                max_degree=self.max_degree,
                min_edge_weight=self.min_edge_weight,
                weight_decay=self.weight_decay,
                inhibition_topn=self.inhibition_topn,
                seed_topk=self.seed_topk,
                diffusion_steps=self.diffusion_steps,
                diffusion_gain=self.diffusion_gain,
                self_retention=self.self_retention,
            )
        else:
            new_core = TopologicalMemoryCore(**init_params)

        self.ensure_size(node_idx + 1)
        self._subgraphs[node_idx] = new_core
        return new_core

    def prune(self) -> None:
        for a, edges in list(self._adj.items()):
            kept = []
            for e in edges:
                e.weight *= self.weight_decay
                if e.weight >= self.min_edge_weight and math.isfinite(e.weight):
                    kept.append(e)
            kept.sort(key=lambda x: x.weight, reverse=True)
            self._adj[a] = kept[: self.max_degree]

    def homeostasis(self) -> None:
        for a, edges in self._adj.items():
            if not edges:
                continue
            total = sum(max(0.0, float(e.weight)) for e in edges)
            if total <= 0:
                continue
            for e in edges:
                e.weight = float(e.weight / total)

    def build_edges_for_new_nodes(
        self,
        latents: np.ndarray,
        start_idx: int,
        end_idx: int,
        k: int = 12,
        min_sim: float = 0.22,
        force_connection: bool = True,
    ) -> None:
        if latents is None:
            return
        n = int(latents.shape[0])
        self.ensure_size(n)

        start_idx = max(0, int(start_idx))
        end_idx = min(n, int(end_idx))
        if start_idx >= end_idx:
            return

        base_end = start_idx
        if base_end <= 0:
            # If this is the first batch, connect them to each other
            if end_idx > 1:
                base = latents[start_idx:end_idx].astype(np.float32, copy=False)
                base_norm = np.linalg.norm(base, axis=1)
                base_norm[base_norm == 0] = 1e-9
                
                # Simple O(N^2) for small first batch
                for i in range(end_idx - start_idx):
                    v = base[i]
                    vn = float(base_norm[i])
                    sims = (base @ v) / (base_norm * vn)
                    # Exclude self
                    sims[i] = -1.0

                    kk = min(int(k), sims.size - 1)
                    if kk <= 0: continue

                    idxs = np.argpartition(sims, -kk)[-kk:]
                    connected_count = 0
                    best_idx = -1
                    best_sim = -1.0

                    for j in idxs:
                        s = float(sims[j])
                        if best_idx == -1 or s > best_sim:
                            best_idx = int(j)
                            best_sim = s
                        if s >= min_sim:
                             self.connect(start_idx + i, start_idx + int(j), weight=s, bidirectional=True)
                             connected_count += 1

                    # 🆕 [2026-01-15] 第一批节点也确保至少有一个连接
                    if force_connection and connected_count == 0 and best_idx != -1:
                        if best_sim > 0:
                            weight = max(0.05, best_sim)
                        else:
                            weight = 0.01
                        self.connect(start_idx + i, start_idx + best_idx, weight=weight, bidirectional=True)
            return

        base = latents[:base_end].astype(np.float32, copy=False)
        base_norm = np.linalg.norm(base, axis=1)
        base_norm[base_norm == 0] = 1e-9

        for i in range(start_idx, end_idx):
            v = latents[i].astype(np.float32, copy=False)
            vn = float(np.linalg.norm(v))
            if vn == 0:
                continue
            sims = (base @ v) / (base_norm * vn)
            if sims.size == 0:
                continue
            kk = min(int(k), sims.size)
            idxs = np.argpartition(sims, -kk)[-kk:]
            idxs = idxs[np.argsort(sims[idxs])[::-1]]

            connected_count = 0
            best_idx = -1
            best_sim = -1.0

            for j in idxs:
                s = float(sims[j])
                if best_idx == -1 or s > best_sim:
                    best_idx = int(j)
                    best_sim = s
                
                if s < float(min_sim):
                    continue
                self.connect(i, int(j), weight=s, bidirectional=True)
                connected_count += 1
            
            # 🆕 [2026-01-15] 增强孤立节点预防：确保每个新节点至少有一个连接
            if force_connection and connected_count == 0 and best_idx != -1:
                # 创建到最佳匹配的连接，即使相似度很低
                # 使用更小的最小权重确保连接建立
                if best_sim > 0:
                    weight = max(0.05, best_sim)  # 降低最小权重阈值
                else:
                    # 如果所有相似度都是负数或零，使用绝对最小权重
                    weight = 0.01
                self.connect(i, best_idx, weight=weight, bidirectional=True)
                connected_count += 1

    def optimize_isolated_nodes(self, latents: np.ndarray, min_sim: float = 0.15) -> int:
        """
        Find nodes with no outgoing edges and try to connect them to the graph.
        Returns the number of rescued nodes.
        """
        if latents is None:
            return 0
            
        rescued = 0
        n = self.size()
        base_norm = None
        
        for i in range(n):
            if i in self._adj and len(self._adj[i]) > 0:
                continue
                
            # Found isolated node
            if base_norm is None:
                base = latents.astype(np.float32, copy=False)
                base_norm = np.linalg.norm(base, axis=1)
                base_norm[base_norm == 0] = 1e-9
                
            v = latents[i]
            vn = np.linalg.norm(v)
            if vn == 0: continue
            
            sims = (latents @ v) / (base_norm * vn)
            sims[i] = -1.0 # Exclude self
            
            # Find best match
            best_idx = np.argmax(sims)
            best_sim = float(sims[best_idx])
            
            if best_sim > 0.05: # Extremely low threshold for rescue
                self.connect(i, int(best_idx), weight=max(min_sim, best_sim), bidirectional=True)
                rescued += 1
                
        return rescued

    def auto_fractal_organize(self, latents: np.ndarray, cluster_threshold: int = 100) -> int:
        """
        OPT-1: 自动检测高密度区域并创建分形子图。
        模拟大脑的「概念抽象」过程：高频访问的概念会发展出更细致的子结构。
        
        Args:
            latents: 节点向量表示 (numpy array)
            cluster_threshold: 触发分形化的节点数阈值
        
        Returns:
            创建的子图数量
        """
        # 延迟导入避免硬依赖
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            return 0
        
        if self._n < cluster_threshold:
            return 0
        
        # 聚类节点
        n_clusters = max(3, self._n // cluster_threshold)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        try:
            labels = kmeans.fit_predict(latents[:self._n])
        except Exception:
            return 0
        
        created = 0
        for cluster_id in range(n_clusters):
            cluster_nodes = np.where(labels == cluster_id)[0]
            if len(cluster_nodes) < cluster_threshold // 2:
                continue
            
            # 找到中心节点作为入口点
            center_idx = int(cluster_nodes[0])
            
            if center_idx not in self._subgraphs:
                # 创建子图并迁移连接
                if hasattr(self, 'fractal_expand'):
                    sub = self.fractal_expand(center_idx)
                    
                    # 在子图中重建连接
                    for node in cluster_nodes:
                        node_idx = int(node)
                        if node_idx != center_idx:
                            sub.ensure_size(node_idx + 1)
                            # 使用正确的属性名 edge.to_idx
                            for edge in self._adj.get(node_idx, []):
                                if edge.to_idx in cluster_nodes:
                                    sub.connect(node_idx, edge.to_idx, edge.weight)
                    created += 1
        
        return created

    def create_divergent_links(self, n_links: int = 50, min_dist: int = 100) -> int:
        """
        Create random long-range connections to stimulate divergent thinking.
        Connects nodes that are far apart in index (likely different time/context).
        """
        if self._n < 2:
            return 0
            
        created = 0
        for _ in range(n_links):
            a = random.randint(0, self._n - 1)
            b = random.randint(0, self._n - 1)
            
            if abs(a - b) < min_dist:
                continue
                
            # Don't connect if already connected
            if any(e.to_idx == b for e in self._adj.get(a, [])):
                continue
                
            # Create a weak "wormhole" connection
            # Weight is random but generally low to represent a "hunch" or "remote association"
            w = random.uniform(0.15, 0.4) 
            self.connect(a, b, weight=w, bidirectional=True, kind="divergent")
            created += 1
            
        return created

    def recall(
        self,
        latents: np.ndarray,
        query_vec: np.ndarray,
        top_k: int = 3,
        seed_topk: Optional[int] = None,
        steps: Optional[int] = None,
        allowed_from_ports: Optional[List[str]] = None,
        port_gates: Optional[Dict[str, float]] = None,
        enable_plasticity: bool = False,
        plasticity_lr: float = 0.02,
        enable_structural_plasticity: bool = False,
        structural_topn: int = 16,
        structural_min_sim: float = 0.35,
    ) -> List[Dict[str, Any]]:
        if latents is None or len(latents) == 0:
            return []

        n = int(latents.shape[0])
        self.ensure_size(n)

        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if q.size == 0:
            return []

        seed_k = int(seed_topk if seed_topk is not None else self.seed_topk)
        steps = int(steps if steps is not None else self.diffusion_steps)

        L = latents.astype(np.float32, copy=False)
        Ln = np.linalg.norm(L, axis=1)
        qn = float(np.linalg.norm(q))
        if qn == 0:
            return []

        denom = Ln * qn
        denom[denom == 0] = 1e-9
        sims = (L @ q) / denom

        seed_k = max(1, min(seed_k, n))
        seed_idx = np.argpartition(sims, -seed_k)[-seed_k:]
        seed_idx = seed_idx[np.argsort(sims[seed_idx])[::-1]]

        act = np.zeros((n,), dtype=np.float32)
        parent = np.full((n,), -1, dtype=np.int32)
        contrib = np.zeros((n,), dtype=np.float32)

        for i in seed_idx:
            s = float(max(0.0, sims[int(i)]))
            act[int(i)] = s
            contrib[int(i)] = s

        if act.max() > 0:
            act /= act.max()

        inh = max(int(self.inhibition_topn), max(32, top_k * 8))
        inh = min(inh, n)

        allowed_ports = None
        if allowed_from_ports:
            allowed_ports = {str(p) for p in allowed_from_ports if isinstance(p, str)}

        gates = port_gates or {}

        for _ in range(max(1, steps)):
            new_act = act * float(self.self_retention)

            active_nodes = np.argsort(act)[-inh:]
            active_nodes = active_nodes[act[active_nodes] > 0]

            for a in active_nodes.tolist():
                a_act = float(act[a])
                if a_act <= 0:
                    continue
                edges = self._adj.get(int(a), [])
                if not edges:
                    continue
                for e in edges:
                    if allowed_ports is not None and e.from_port not in allowed_ports:
                        continue
                    b = int(e.to_idx)
                    if b < 0 or b >= n:
                        continue
                    w = float(e.weight)
                    if w <= 0:
                        continue
                    g = float(gates.get(e.from_port, 1.0))
                    if g <= 0:
                        continue
                    kind_gain = 1.0
                    sign = 1.0
                    if e.kind == "inh":
                        sign = -1.0
                    elif e.kind == "divergent":
                        kind_gain = 0.6
                    delta = sign * a_act * w * float(self.diffusion_gain) * g * kind_gain
                    if delta <= 0:
                        continue
                    cand = float(new_act[b] + delta)
                    if cand > float(new_act[b]):
                        new_act[b] = cand
                        if delta > float(contrib[b]):
                            contrib[b] = delta
                            parent[b] = int(a)
                        e.usage += 1
                        if enable_plasticity and plasticity_lr > 0:
                            dw = float(plasticity_lr) * a_act * float(min(1.0, delta))
                            if dw > 0:
                                e.weight = float(min(1.0, e.weight + dw))

            if new_act.max() > 0:
                new_act /= new_act.max()

            keep = np.argsort(new_act)[-inh:]
            mask = np.zeros((n,), dtype=bool)
            mask[keep] = True
            new_act[~mask] = 0.0
            act = new_act

        if enable_structural_plasticity and n > 1:
            t = max(2, min(int(structural_topn), n))
            top = np.argsort(act)[-t:]
            top = top[act[top] > 0]
            if top.size >= 2:
                L = latents.astype(np.float32, copy=False)
                norms = np.linalg.norm(L, axis=1)
                norms[norms == 0] = 1e-9
                top_list = [int(x) for x in top.tolist()]
                for a in top_list:
                    cand_bs: List[Tuple[float, int]] = []
                    va = L[a]
                    for b in top_list:
                        if a == b:
                            continue
                        if any(ed.to_idx == int(b) for ed in self._adj.get(int(a), [])):
                            continue
                        sim = float((va @ L[b]) / (float(norms[a]) * float(norms[b])))
                        if sim >= float(structural_min_sim):
                            cand_bs.append((sim, int(b)))
                    if not cand_bs:
                        continue
                    cand_bs.sort(reverse=True)
                    for sim, b in cand_bs[:4]:
                        if random.random() < 0.35:
                            self.connect(
                                int(a),
                                int(b),
                                weight=float(max(self.min_edge_weight, min(0.4, sim))),
                                from_port="semantic",
                                to_port="semantic",
                                kind="assoc",
                                bidirectional=True,
                            )

        top_k = max(1, min(int(top_k), n))
        idxs = np.argsort(act)[-top_k:][::-1]

        out: List[Dict[str, Any]] = []
        for idx in idxs.tolist():
            score = float(act[int(idx)])
            if score <= 0:
                continue
            trace = self._trace_path(int(idx), parent, limit=12)
            out.append(
                {
                    "node_index": int(idx),
                    "score": score,
                    "trace": trace,
                }
            )
        return out

    def to_dict(self) -> Dict[str, Any]:
        adj: Dict[str, Any] = {}
        for a, edges in self._adj.items():
            adj[str(int(a))] = [
                {
                    "to_idx": int(e.to_idx),
                    "weight": float(e.weight),
                    "from_port": e.from_port,
                    "to_port": e.to_port,
                    "kind": e.kind,
                    "usage": int(e.usage),
                }
                for e in edges
            ]
        return {
            "n": int(self._n),
            "params": {
                "max_degree": self.max_degree,
                "min_edge_weight": self.min_edge_weight,
                "weight_decay": self.weight_decay,
                "inhibition_topn": self.inhibition_topn,
                "seed_topk": self.seed_topk,
                "diffusion_steps": self.diffusion_steps,
                "diffusion_gain": self.diffusion_gain,
                "self_retention": self.self_retention,
            },
            "adj": adj,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TopologicalMemoryCore":
        params = (payload or {}).get("params", {}) or {}
        core = cls(**{k: params[k] for k in params if k in {
            "max_degree",
            "min_edge_weight",
            "weight_decay",
            "inhibition_topn",
            "seed_topk",
            "diffusion_steps",
            "diffusion_gain",
            "self_retention",
        }})
        n = int((payload or {}).get("n", 0) or 0)
        core.ensure_size(n)
        adj = (payload or {}).get("adj", {}) or {}
        for a_str, edges in adj.items():
            a = int(a_str)
            core._adj.setdefault(a, [])
            for ed in edges or []:
                core._adj[a].append(
                    Edge(
                        to_idx=int(ed.get("to_idx", 0)),
                        weight=float(ed.get("weight", 0.0)),
                        from_port=str(ed.get("from_port", "semantic")),
                        to_port=str(ed.get("to_port", "semantic")),
                        kind=str(ed.get("kind", "exc")),
                        usage=int(ed.get("usage", 0)),
                    )
                )
            core._cap_degree(a)

        # Load fractal subgraphs
        subgraphs_payload = (payload or {}).get("subgraphs", {}) or {}
        for k_str, v_payload in subgraphs_payload.items():
            if v_payload:
                core._subgraphs[int(k_str)] = cls.from_dict(v_payload)

        return core

    def save_json(self, path: str) -> None:
        tmp = f"{path}.tmp_{os.getpid()}_{int(time.time() * 1000)}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)
        os.replace(tmp, path)

    @classmethod
    def load_json(cls, path: str) -> "TopologicalMemoryCore":
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            dec = json.JSONDecoder()
            payload, _ = dec.raw_decode(raw)
        return cls.from_dict(payload)

    def _upsert_edge(self, a: int, b: int, w: float, from_port: str, to_port: str, kind: str) -> None:
        edges = self._adj.setdefault(int(a), [])
        for e in edges:
            if e.to_idx == int(b) and e.from_port == from_port and e.to_port == to_port and e.kind == kind:
                e.weight = float(max(e.weight, w))
                return
        edges.append(Edge(to_idx=int(b), weight=float(w), from_port=from_port, to_port=to_port, kind=kind))

    def _cap_degree(self, a: int) -> None:
        edges = self._adj.get(int(a), [])
        if not edges:
            return
        edges.sort(key=lambda x: x.weight, reverse=True)
        if len(edges) > self.max_degree:
            self._adj[int(a)] = edges[: self.max_degree]

    def _trace_path(self, idx: int, parent: np.ndarray, limit: int = 12) -> List[int]:
        path: List[int] = [int(idx)]
        cur = int(idx)
        for _ in range(max(1, int(limit))):
            p = int(parent[cur])
            if p < 0:
                break
            path.append(p)
            cur = p
        return path
