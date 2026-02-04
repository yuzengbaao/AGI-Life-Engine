import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np
import logging
import time
import hashlib
from typing import List, Dict, Optional, Any, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuralMemory")

class InsightAutoEncoder(nn.Module):
    """
    A biological-inspired AutoEncoder that compresses high-dimensional semantic vectors
    into a compact latent space (the "concept space").
    """
    def __init__(self, input_dim=384, hidden_dim=128, latent_dim=64):
        super().__init__()
        # Encoder: Cortex -> Hippocampus compression
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh() # Latent space normalized to [-1, 1] for stability
        )
        # Decoder: Reconstruction for recall
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

from core.memory.topology_memory import TopologicalMemoryCore
# P2-1: 导入生命周期管理器
from core.memory.memory_lifecycle_manager import MemoryLifecycleManager, EvictionPolicy

class BiologicalMemorySystem:
    """
    The 'Second Brain' of the AGI.
    It does not rely on LLMs for storage or retrieval.
    It uses a biological neural network topology (AutoEncoder + TopologyGraph) to internalize skills and insights.

    P2-1 增强: 集成生命周期管理器，自动清理和压缩记忆
    """
    def __init__(self, base_dir="./data/neural_memory", enable_lifecycle: bool = True):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Biological Memory System on {self.device}...")
        
        # 1. Sensory Cortex (Embedding Model)
        # We use a small, efficient model to simulate sensory processing
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=str(self.device))
        except Exception as e:
            logger.warning(f"Failed to load sentence-transformer: {e}. Falling back to random projection.")
            self.embedder = None
            
        # 2. Hippocampus (The AutoEncoder)
        self.input_dim = 384
        self.model = InsightAutoEncoder(input_dim=self.input_dim).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        # 3. Topological Graph (The Connectome)
        self.topology = TopologicalMemoryCore()
        
        self.model_path = os.path.join(self.base_dir, "brain_weights.pt")
        self.memory_index_path = os.path.join(self.base_dir, "episodic_index.npy")
        self.topology_path = os.path.join(self.base_dir, "topology_graph.json")
        self.metadata_path = os.path.join(self.base_dir, "memory_metadata.json")
        
        self.memory_latents = None # Numpy array of stored latent vectors
        self.memory_metadata = []  # List of dicts corresponding to latents
        self._id_to_index: Dict[str, int] = {}
        self._last_exec_index: Optional[int] = None
        self._exec_events_since_macro: int = 0
        self._last_macro_ts: float = 0.0

        # P2-1: 生命周期管理器
        self.enable_lifecycle = enable_lifecycle
        self.lifecycle_manager: Optional[MemoryLifecycleManager] = None
        self.lifecycle_state_path = os.path.join(self.base_dir, "lifecycle_state.json")

        if enable_lifecycle:
            self.lifecycle_manager = MemoryLifecycleManager(
                max_records=100000,  # 最大保留 10 万条记录
                max_age_days=30.0,  # 最多保留 30 天
                eviction_policy=EvictionPolicy.HYBRID,
                auto_cleanup_interval=100,
                compression_threshold=10000,
            )
            # 尝试加载生命周期状态
            if os.path.exists(self.lifecycle_state_path):
                self.lifecycle_manager.load_state(self.lifecycle_state_path)
                logger.info("✅ 生命周期管理器状态已加载")

        self.load_brain()

    def load_brain(self):
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                logger.info("Brain weights loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load brain weights: {e}")
        
        if os.path.exists(self.memory_index_path):
            try:
                self.memory_latents = np.load(self.memory_index_path)
            except Exception as e:
                logger.error(f"Failed to load memory index: {e}")
        
        if os.path.exists(self.topology_path):
            try:
                self.topology = TopologicalMemoryCore.load_json(self.topology_path)
                logger.info(f"Topology graph loaded with {self.topology.size()} nodes.")
            except Exception as e:
                logger.error(f"Failed to load topology graph: {e}")
                
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    raw = f.read()
                try:
                    self.memory_metadata = json.loads(raw)
                except json.JSONDecodeError:
                    dec = json.JSONDecoder()
                    self.memory_metadata, _ = dec.raw_decode(raw)
            except Exception as e:
                logger.error(f"Failed to load memory metadata: {e}")
        self._rebuild_index()

    def save_brain(self):
        tmp_model = f"{self.model_path}.tmp_{os.getpid()}_{int(time.time() * 1000)}"
        torch.save(self.model.state_dict(), tmp_model)
        os.replace(tmp_model, self.model_path)

        if self.memory_latents is not None:
            tmp_idx = f"{self.memory_index_path}.tmp_{os.getpid()}_{int(time.time() * 1000)}"
            np.save(tmp_idx, self.memory_latents)
            if not tmp_idx.endswith(".npy") and os.path.exists(tmp_idx + ".npy"):
                tmp_idx = tmp_idx + ".npy"
            os.replace(tmp_idx, self.memory_index_path)

        self.topology.save_json(self.topology_path)

        tmp_meta = f"{self.metadata_path}.tmp_{os.getpid()}_{int(time.time() * 1000)}"
        with open(tmp_meta, 'w', encoding='utf-8') as f:
            json.dump(self.memory_metadata, f, ensure_ascii=False, indent=2)
        os.replace(tmp_meta, self.metadata_path)

        # P2-1: 保存生命周期状态
        if self.lifecycle_manager:
            self.lifecycle_manager.save_state(self.lifecycle_state_path)

        logger.info("Brain state saved.")
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._id_to_index = {}
        for i, meta in enumerate(self.memory_metadata):
            mid = meta.get("id")
            if isinstance(mid, str) and mid:
                self._id_to_index[mid] = int(i)

    def get_stats(self) -> Dict[str, object]:
        """
        Get current memory statistics for system monitoring.

        P2-1 增强: 包含生命周期管理统计
        """
        memories = 0
        if self.memory_latents is not None:
            try:
                memories = int(self.memory_latents.shape[0])
            except Exception:
                memories = int(len(self.memory_latents))

        stats = {
            "nodes": int(self.topology.size()),
            "memories": int(memories),
            "metadata_entries": int(len(self.memory_metadata)),
            "device": str(self.device),
            "embedder_available": self.embedder is not None,
            "lifecycle_enabled": self.enable_lifecycle,
        }

        # P2-1: 添加生命周期统计
        if self.lifecycle_manager:
            stats["lifecycle"] = self.lifecycle_manager.get_stats()

        return stats

    def cleanup_memories(self, force: bool = False) -> Dict[str, Any]:
        """
        P2-1: 手动触发记忆清理

        Args:
            force: 是否强制清理（忽略自动清理间隔）

        Returns:
            清理结果统计
        """
        if not self.lifecycle_manager:
            return {"status": "disabled", "message": "生命周期管理器未启用"}

        if force:
            # 强制清理：重置操作计数器并触发清理
            self.lifecycle_manager.operation_count = self.lifecycle_manager.auto_cleanup_interval

        result = self.lifecycle_manager.auto_cleanup()

        # 应用清理结果到实际内存数组
        if result["evicted"] > 0 or result["archived"] > 0:
            self.memory_latents, self.memory_metadata = (
                self.lifecycle_manager.export_records_for_cleanup(
                    self.memory_latents, self.memory_metadata
                )
            )
            self._rebuild_index()
            # 保存清理后的状态
            self.save_brain()

        result["status"] = "ok"
        return result

    def export_visualization(
        self,
        output_path: str = "./workspace/neural_brain_topology.html",
        max_nodes: int = 500,
        highlight_nodes: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        导出神经拓扑的交互式 HTML 可视化。
        
        Args:
            output_path: 输出 HTML 文件路径
            max_nodes: 最大渲染节点数（防止浏览器性能问题）
            highlight_nodes: 需要高亮的节点索引列表
        
        Returns:
            包含渲染结果的字典，包括:
            - status: "ok" 或 "error"
            - path: 输出文件路径
            - nodes_rendered: 渲染的节点数
            - edges_rendered: 渲染的边数
        
        Example:
            >>> result = biological_memory.export_visualization()
            >>> print(f"可视化已生成: {result['path']}")
        """
        try:
            from core.memory.topology_visualizer import TopologyVisualizer
            
            visualizer = TopologyVisualizer(
                topology=self.topology,
                metadata=self.memory_metadata,
            )
            
            result = visualizer.render_html(
                output_path=output_path,
                max_nodes=max_nodes,
                highlight_nodes=highlight_nodes,
                title="🧠 数字神经大脑 - 拓扑可视化",
            )
            
            logger.info(f"Topology visualization exported to {output_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to export visualization: {e}")
            return {"status": "error", "error": str(e)}

    def store(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store an experience/memory item. This is a convenience wrapper for record_online.
        
        Provides compatibility with code that expects a simple store() interface.
        
        Args:
            experience: Dict containing experience data. Common fields:
                - type: str, the type of experience (e.g., 'execution', 'observation')
                - intent_id: str, associated intent identifier
                - action: str, the action performed
                - result: str, the result of the action
                - success: bool, whether the action succeeded
                - timestamp: float, Unix timestamp
        
        Returns:
            Dict with status of the operation.
        
        Example:
            >>> bio_memory.store({
            ...     'type': 'execution',
            ...     'action': 'run_command',
            ...     'result': 'success',
            ...     'success': True
            ... })
        """
        if not isinstance(experience, dict):
            return {"status": "error", "error": "experience must be a dict"}
        
        # 构建content字段：将experience字典转换为可读的字符串
        content_parts = []
        for key, value in experience.items():
            if key not in ('timestamp',) and value is not None:
                # 截断过长的值
                str_val = str(value)
                if len(str_val) > 500:
                    str_val = str_val[:500] + "..."
                content_parts.append(f"{key}: {str_val}")
        
        content = " | ".join(content_parts) if content_parts else "empty experience"
        
        # 构建record_online期望的item格式
        item = {
            "id": f"exp_{int(time.time() * 1000)}_{hash(content) % 10000}",
            "content": content,
            "source": experience.get("type", "execution"),
            "type": experience.get("type", "episode"),
            "timestamp": experience.get("timestamp", time.time()),
            "tool": experience.get("action"),
            "args": experience.get("args"),
        }
        
        # 使用record_online进行实际存储
        return self.record_online([item], connect_sequence=True, save=True)

    def internalize_items(self, items: List[Dict], epochs=30) -> Dict[str, float]:
        """
        Unified API for internalizing a list of memory items.
        Items should be dicts with 'content' (str) and optionally 'id', 'source', 'tags'.
        """
        processed_items = []
        for item in items:
            if 'content' not in item:
                continue
            
            new_item = item.copy()
            if 'id' not in new_item:
                # Generate a safe ID if missing
                new_item['id'] = f"mem_{int(time.time())}_{len(processed_items)}"
            
            processed_items.append(new_item)
            
        if not processed_items:
            return {"status": "no_content"}
            
        return self.internalize(processed_items, epochs=epochs)

    def record_online(
        self,
        items: List[Dict[str, Any]],
        *,
        connect_sequence: bool = True,
        seq_weight: float = 0.9,
        seq_port: str = "exec",
        k: int = 8,
        min_sim: float = 0.25,
        save: bool = True,
    ) -> Dict[str, Any]:
        if not items:
            return {"status": "no_items"}

        processed: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            new_item = dict(item)
            if not isinstance(new_item.get("id"), str) or not new_item["id"]:
                new_item["id"] = f"mem_{int(time.time() * 1000)}_{len(processed)}"
            processed.append(new_item)

        if not processed:
            return {"status": "no_content"}

        texts = [it["content"] for it in processed]

        if self.embedder:
            with torch.no_grad():
                embeddings = self.embedder.encode(texts, convert_to_tensor=True)
            embeddings = embeddings.clone().detach()
        else:
            embeddings = torch.randn(len(texts), self.input_dim).to(self.device)

        self.model.eval()
        with torch.no_grad():
            _, latents = self.model(embeddings)
            new_latents = latents.cpu().numpy()

        if self.memory_latents is None:
            start_idx = 0
            self.memory_latents = new_latents
        else:
            start_idx = int(len(self.memory_latents))
            self.memory_latents = np.vstack([self.memory_latents, new_latents])

        end_idx = int(len(self.memory_latents))
        self.topology.ensure_size(end_idx)

        for it in processed:
            content = it.get("content", "")
            preview = content[:800]
            metadata = {
                "id": it.get("id", "unknown"),
                "timestamp": float(it.get("timestamp", time.time())),
                "source": it.get("source", "online"),
                "type": it.get("type", "episode"),
                "tool": it.get("tool"),
                "skill": it.get("skill"),
                "args": it.get("args"),
                "macro_signature": it.get("macro_signature"),
                "prototype_ids": it.get("prototype_ids"),
                "macro_id": it.get("macro_id"),
                "origin_id": it.get("origin_id"),
                "content_preview": preview,
            }

            # P2-1: 注册到生命周期管理器
            if self.lifecycle_manager:
                importance = self.lifecycle_manager.calculate_importance(metadata)
                self.lifecycle_manager.register_record(
                    memory_id=metadata["id"],
                    importance_score=importance,
                    tags=[metadata.get("type", "episode"), metadata.get("source", "online")]
                )

            self.memory_metadata.append(metadata)

        self.topology.build_edges_for_new_nodes(
            self.memory_latents,
            start_idx=start_idx,
            end_idx=end_idx,
            k=int(k),
            min_sim=float(min_sim),
            force_connection=True,
        )

        if connect_sequence:
            idxs = list(range(start_idx, end_idx))
            prev = self._last_exec_index
            for cur in idxs:
                if prev is not None:
                    self.topology.connect(
                        int(prev),
                        int(cur),
                        weight=float(seq_weight),
                        from_port=str(seq_port),
                        to_port=str(seq_port),
                        kind="exec",
                        bidirectional=False,
                    )
                prev = int(cur)
            self._last_exec_index = int(idxs[-1]) if idxs else self._last_exec_index

        self.topology.homeostasis()
        self.topology.prune()

        has_exec = any(it.get("type") in {"tool_call", "skill_call"} for it in processed)
        if has_exec:
            self._exec_events_since_macro += int(len(processed))

        should_induce = (
            has_exec
            and self._exec_events_since_macro >= 24
            and (time.time() - float(self._last_macro_ts)) >= 20.0
        )

        if should_induce:
            try:
                self.induce_macros_from_exec(max_scan=600, max_macros=3, save=False)
                self._exec_events_since_macro = 0
                self._last_macro_ts = float(time.time())
            except Exception:
                pass

        if save:
            self.save_brain()
        else:
            self._rebuild_index()

        return {
            "status": "ok",
            "added": int(end_idx - start_idx),
            "topology_size": int(self.topology.size()),
            "last_exec_index": self._last_exec_index,
        }

    def suggest_macros_for_goal(self, goal: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not goal:
            return []

        if self.memory_latents is None or len(self.memory_latents) == 0:
            return []

        if not self.embedder:
            q = str(goal).lower()
            parts = []
            cur = []
            for ch in q:
                if ch.isalnum() or ch in {"_", "-"}:
                    cur.append(ch)
                else:
                    if cur:
                        parts.append("".join(cur))
                        cur = []
            if cur:
                parts.append("".join(cur))
            tokens = [p for p in parts if len(p) >= 2]
            if not tokens:
                return []

            scored: List[Dict[str, Any]] = []
            for meta in self.memory_metadata:
                if not isinstance(meta, dict) or meta.get("type") != "macro":
                    continue
                text = (meta.get("content_preview") or "").lower()
                if not text:
                    continue
                hit = 0
                for t in tokens:
                    if t in text:
                        hit += 1
                if hit <= 0:
                    continue
                score = float(hit) / float(max(1, len(tokens)))
                scored.append(
                    {
                        "macro_id": meta.get("id"),
                        "score": score,
                        "content_preview": (meta.get("content_preview") or "")[:240],
                    }
                )
            scored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
            return scored[: int(top_k)]

        self.model.eval()
        with torch.no_grad():
            query_emb = self.embedder.encode(goal, convert_to_tensor=True)
            _, query_latent = self.model(query_emb.unsqueeze(0))
            query_vec = query_latent.cpu().numpy().flatten()

        results = self.topology.recall(
            latents=self.memory_latents,
            query_vec=query_vec,
            top_k=min(64, int(len(self.memory_latents))),
            seed_topk=min(128, max(16, int(len(self.memory_latents) // 4))),
            steps=3,
            allowed_from_ports=None,
            port_gates={"semantic": 1.0, "macro": 1.0, "exec": 0.6},
            enable_plasticity=False,
            plasticity_lr=0.0,
            enable_structural_plasticity=False,
        )

        out: List[Dict[str, Any]] = []
        for res in results:
            idx = int(res.get("node_index", -1))
            if idx < 0 or idx >= len(self.memory_metadata):
                continue
            meta = self.memory_metadata[idx]
            if not isinstance(meta, dict):
                continue
            t = meta.get("type")
            if t != "macro":
                continue
            out.append(
                {
                    "macro_id": meta.get("id"),
                    "score": float(res.get("score", 0.0)),
                    "content_preview": (meta.get("content_preview") or "")[:240],
                }
            )
            if len(out) >= int(top_k):
                break

        return out

    def expand_macro_pattern(self, macro_id: str) -> List[str]:
        if not macro_id:
            return []
        idx = self._id_to_index.get(macro_id)
        if idx is None or idx < 0 or idx >= len(self.memory_metadata):
            return []
        meta = self.memory_metadata[idx]
        if isinstance(meta, dict):
            sig = meta.get("macro_signature")
            if isinstance(sig, list) and sig:
                return [str(x) for x in sig if str(x)]
        text = ""
        if isinstance(meta, dict):
            text = meta.get("content_preview") or ""
        if not text:
            return []
        try:
            if "macro:" in text:
                body = text.split("macro:", 1)[1].strip()
            else:
                body = text
            parts = [p.strip() for p in body.replace("→", "->").split("->")]
            return [p for p in parts if p]
        except Exception:
            return []

    def expand_macro_toolcalls(
        self,
        macro_id: str,
        *,
        bindings: Optional[Dict[str, Any]] = None,
        max_steps: int = 12,
    ) -> List[Dict[str, Any]]:
        if not macro_id:
            return []
        idx = self._id_to_index.get(macro_id)
        if idx is None or idx < 0 or idx >= len(self.memory_metadata):
            return []
        meta = self.memory_metadata[idx]
        if not isinstance(meta, dict):
            return []
        proto = meta.get("prototype_ids")
        bindings = bindings or {}

        steps: List[Dict[str, Any]] = []

        if isinstance(proto, list) and proto:
            for pid in proto[: int(max_steps)]:
                if not isinstance(pid, str):
                    continue
                pidx = self._id_to_index.get(pid)
                if pidx is None or pidx < 0 or pidx >= len(self.memory_metadata):
                    continue
                pmeta = self.memory_metadata[pidx]
                if not isinstance(pmeta, dict):
                    continue
                tool = pmeta.get("tool")
                if not isinstance(tool, str) or not tool:
                    continue
                if tool == "execute_cognitive_skill":
                    skill = pmeta.get("skill")
                    if not isinstance(skill, str) or not skill:
                        continue
                    skill_args = pmeta.get("args") if isinstance(pmeta.get("args"), dict) else {}
                    steps.append({"tool": "execute_cognitive_skill", "args": {"skill_name": skill, "args": skill_args}})
                else:
                    targs = pmeta.get("args") if isinstance(pmeta.get("args"), dict) else {}
                    steps.append({"tool": tool, "args": targs})
            return steps

        pattern = self.expand_macro_pattern(macro_id)
        for token in pattern[: int(max_steps)]:
            tool = token
            skill = None
            if ":" in token:
                tool, skill = token.split(":", 1)
                tool = tool.strip()
                skill = skill.strip()
            if tool == "execute_cognitive_skill" and skill:
                key = f"{tool}:{skill}"
                skill_args = bindings.get(key) if isinstance(bindings.get(key), dict) else bindings.get(tool, {})
                if not isinstance(skill_args, dict):
                    skill_args = {}
                steps.append({"tool": "execute_cognitive_skill", "args": {"skill_name": skill, "args": skill_args}})
            else:
                targs = bindings.get(tool, {})
                if not isinstance(targs, dict):
                    targs = {}
                steps.append({"tool": tool, "args": targs})
        return steps

    def induce_macros_from_exec(
        self,
        *,
        max_scan: int = 800,
        session_gap_s: float = 90.0,
        min_len: int = 3,
        max_len: int = 6,
        min_count: int = 2,
        max_macros: int = 5,
        clone_circuit: bool = True,
        save: bool = True,
    ) -> Dict[str, Any]:
        if self.memory_latents is None or len(self.memory_latents) == 0:
            return {"status": "no_latents"}

        n_total = int(len(self.memory_metadata))
        if n_total == 0:
            return {"status": "no_metadata"}

        events: List[Tuple[int, float, str]] = []
        start = max(0, n_total - int(max_scan))
        for idx in range(start, n_total):
            meta = self.memory_metadata[idx] if idx < n_total else None
            if not isinstance(meta, dict):
                continue
            t = meta.get("type")
            if t not in {"tool_call", "skill_call"}:
                continue
            tool = meta.get("tool") or "tool"
            skill = meta.get("skill")
            token = f"{tool}:{skill}" if skill else str(tool)
            ts = float(meta.get("timestamp", 0.0) or 0.0)
            events.append((int(idx), ts, token))

        if len(events) < int(min_len):
            return {"status": "too_few_events", "events": int(len(events))}

        events.sort(key=lambda x: x[1])

        sessions: List[List[Tuple[int, float, str]]] = []
        cur: List[Tuple[int, float, str]] = []
        last_ts = None
        for ev in events:
            if last_ts is None:
                cur = [ev]
                last_ts = ev[1]
                continue
            if (ev[1] - float(last_ts)) > float(session_gap_s):
                if cur:
                    sessions.append(cur)
                cur = [ev]
            else:
                cur.append(ev)
            last_ts = ev[1]
        if cur:
            sessions.append(cur)

        counts: Dict[Tuple[str, ...], int] = {}
        occ: Dict[Tuple[str, ...], List[List[int]]] = {}

        for sess in sessions:
            tokens = [x[2] for x in sess]
            idxs = [int(x[0]) for x in sess]
            m = len(tokens)
            for L in range(int(min_len), int(max_len) + 1):
                if m < L:
                    continue
                for i in range(0, m - L + 1):
                    sig = tuple(tokens[i : i + L])
                    counts[sig] = int(counts.get(sig, 0) + 1)
                    occ.setdefault(sig, [])
                    if len(occ[sig]) < 3:
                        occ[sig].append(idxs[i : i + L])

        candidates = [(sig, c) for sig, c in counts.items() if int(c) >= int(min_count)]
        if not candidates:
            return {"status": "no_repeats"}

        candidates.sort(key=lambda x: (int(x[1]), len(x[0])), reverse=True)
        chosen = candidates[: int(max_macros)]

        created: List[Dict[str, Any]] = []

        for sig, c in chosen:
            macro_id = self._macro_id(sig)
            macro_idx = self._id_to_index.get(macro_id)

            if macro_idx is None:
                content = "macro: " + " -> ".join(sig)
                prototype_ids: List[str] = []
                occurrences = occ.get(sig, [])
                if occurrences:
                    for node_idx in occurrences[0]:
                        if 0 <= int(node_idx) < len(self.memory_metadata):
                            mid = self.memory_metadata[int(node_idx)].get("id")
                            if isinstance(mid, str) and mid:
                                prototype_ids.append(mid)
                res = self.record_online(
                    [
                        {
                            "id": macro_id,
                            "content": content,
                            "source": "macro_induction",
                            "type": "macro",
                            "tool": "macro_induction",
                            "macro_signature": list(sig),
                            "prototype_ids": prototype_ids,
                        }
                    ],
                    connect_sequence=False,
                    seq_port="macro",
                    save=False,
                )
                macro_idx = self._id_to_index.get(macro_id)
                if macro_idx is None:
                    continue

            occurrences = occ.get(sig, [])
            if 0 <= int(macro_idx) < len(self.memory_metadata):
                mmeta = self.memory_metadata[int(macro_idx)]
                if isinstance(mmeta, dict):
                    if not isinstance(mmeta.get("macro_signature"), list):
                        mmeta["macro_signature"] = list(sig)
                    if not isinstance(mmeta.get("prototype_ids"), list) or not mmeta.get("prototype_ids"):
                        proto_ids: List[str] = []
                        if occurrences:
                            for node_idx in occurrences[0]:
                                if 0 <= int(node_idx) < len(self.memory_metadata):
                                    mid = self.memory_metadata[int(node_idx)].get("id")
                                    if isinstance(mid, str) and mid:
                                        proto_ids.append(mid)
                        mmeta["prototype_ids"] = proto_ids
            for seq in occurrences:
                for pos, node_idx in enumerate(seq):
                    w = float(0.95 - 0.15 * (pos / max(1, (len(seq) - 1))))
                    self.topology.connect(
                        int(macro_idx),
                        int(node_idx),
                        weight=w,
                        from_port="macro",
                        to_port="exec",
                        kind="macro",
                        bidirectional=False,
                    )
                    self.topology.connect(
                        int(node_idx),
                        int(macro_idx),
                        weight=float(w * 0.7),
                        from_port="exec",
                        to_port="macro",
                        kind="macro",
                        bidirectional=False,
                    )
                for a, b in zip(seq[:-1], seq[1:]):
                    self.topology.connect(
                        int(a),
                        int(b),
                        weight=1.0,
                        from_port="exec",
                        to_port="exec",
                        kind="exec",
                        bidirectional=False,
                    )

            cloned = None
            if clone_circuit and occurrences:
                try:
                    tag = macro_id
                    mapping = self.clone_subgraph_memories(occurrences[0], tag=tag, noise_std=0.02)
                    cloned = mapping
                    for old_idx, new_idx in mapping.items():
                        self.topology.connect(
                            int(macro_idx),
                            int(new_idx),
                            weight=0.98,
                            from_port="macro",
                            to_port="exec",
                            kind="macro_clone",
                            bidirectional=False,
                        )
                        self.topology.connect(
                            int(new_idx),
                            int(macro_idx),
                            weight=0.8,
                            from_port="exec",
                            to_port="macro",
                            kind="macro_clone",
                            bidirectional=False,
                        )
                except Exception:
                    cloned = None

            created.append(
                {
                    "macro_id": macro_id,
                    "count": int(c),
                    "length": int(len(sig)),
                    "occurrences": int(len(occurrences)),
                    "cloned": cloned is not None,
                }
            )

        self.topology.homeostasis()
        self.topology.prune()

        if save:
            self.save_brain()
        else:
            self._rebuild_index()

        return {"status": "ok", "created": created}

    def clone_subgraph_memories(
        self,
        node_indices: List[int],
        *,
        tag: str,
        noise_std: float = 0.01,
    ) -> Dict[int, int]:
        if self.memory_latents is None or len(self.memory_latents) == 0:
            return {}

        uniq = [int(x) for x in dict.fromkeys(node_indices)]
        uniq = [x for x in uniq if 0 <= x < int(len(self.memory_latents))]
        if not uniq:
            return {}

        if int(self.topology.size()) != int(len(self.memory_latents)):
            self.topology.ensure_size(int(len(self.memory_latents)))

        base_n = int(len(self.memory_latents))
        mapping = self.topology.clone_subgraph(uniq, noise_std=0.01, copy_edges=True)

        clones = self.memory_latents[np.asarray(uniq, dtype=np.int64)].astype(np.float32, copy=True)
        if float(noise_std) > 0:
            clones = clones + np.random.normal(0.0, float(noise_std), size=clones.shape).astype(np.float32)
        self.memory_latents = np.vstack([self.memory_latents, clones])

        now = float(time.time())
        for old_idx in uniq:
            meta = self.memory_metadata[old_idx] if old_idx < len(self.memory_metadata) else {}
            origin_id = meta.get("id", f"idx_{old_idx}")
            new_id = f"clone_{tag}_{origin_id}_{int(now * 1000)}"
            self.memory_metadata.append(
                {
                    "id": new_id,
                    "timestamp": now,
                    "source": "macro_clone",
                    "type": "clone",
                    "macro_id": tag,
                    "origin_id": origin_id,
                    "tool": meta.get("tool"),
                    "skill": meta.get("skill"),
                    "content_preview": meta.get("content_preview", ""),
                }
            )

        if int(len(self.memory_latents)) != int(self.topology.size()):
            self.topology.ensure_size(int(len(self.memory_latents)))

        self._rebuild_index()
        return mapping

    def _macro_id(self, signature: Tuple[str, ...]) -> str:
        raw = ("|".join(signature)).encode("utf-8", errors="ignore")
        h = hashlib.sha1(raw).hexdigest()
        return f"macro_{h[:16]}"

    def recall_by_text(self, query: str, top_k=5) -> List[Dict]:
        """
        Unified API for text-based recall.
        """
        return self.recall(query, top_k=top_k)

    def internalize(self, insights: List[Dict[str, str]], epochs=100) -> Dict[str, float]:
        """
        The 'Sleep' process. Consolidates raw insights into neural weights and topological connections.
        
        Args:
            insights: List of dicts with 'content' and 'id' keys.
        """
        if not insights:
            return {"loss": 0.0}
            
        texts = [item['content'] for item in insights]
        
        # 1. Sensory Encoding
        if self.embedder:
            with torch.no_grad():
                embeddings = self.embedder.encode(texts, convert_to_tensor=True)
            # Ensure embeddings are standard tensors ready for training
            embeddings = embeddings.clone().detach()
        else:
            # Fallback random projection (not ideal but works for structure)
            embeddings = torch.randn(len(texts), self.input_dim).to(self.device)
            
        # 2. Neural Plasticity (Training)
        self.model.train()
        initial_loss = 0.0
        final_loss = 0.0
        
        logger.info(f"Starting consolidation of {len(texts)} memories...")
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            recon, latents = self.model(embeddings)
            
            # Reconstruction Loss (preserve meaning)
            rec_loss = nn.MSELoss()(recon, embeddings)
            
            # Regularization (keep latent space compact)
            reg_loss = torch.mean(latents ** 2) * 0.01
            
            total_loss = rec_loss + reg_loss
            
            total_loss.backward()
            self.optimizer.step()
            
            if epoch == 0:
                initial_loss = total_loss.item()
            final_loss = total_loss.item()
            
        # 3. Memory Indexing (Store the compressed latent vectors)
        with torch.no_grad():
            self.model.eval()
            _, latents = self.model(embeddings)
            new_latents = latents.cpu().numpy()
            
            start_idx = 0
            # Update memory bank
            if self.memory_latents is None:
                self.memory_latents = new_latents
                start_idx = 0
            else:
                start_idx = len(self.memory_latents)
                self.memory_latents = np.vstack([self.memory_latents, new_latents])
                
            # Update metadata
            for item in insights:
                self.memory_metadata.append({
                    "id": item.get('id', 'unknown'),
                    "timestamp": time.time(),
                    "source": "insight_consolidation"
                })
        
        # 4. Topological Growth (Connect new memories)
        # Ensure topology size matches latent store
        total_nodes = len(self.memory_latents)
        self.topology.ensure_size(total_nodes)
        
        # Grow connections for new nodes
        # Connect to existing nodes based on latent similarity (Hebbian-like wiring)
        logger.info(f"Growing topology connections for {len(new_latents)} new nodes...")
        self.topology.build_edges_for_new_nodes(
            self.memory_latents, 
            start_idx=start_idx, 
            end_idx=total_nodes,
            k=8, # Connect to top-8 most similar existing memories
            min_sim=0.3, # Minimum similarity threshold
            force_connection=True # Ensure no node is left behind
        )
        
        # Rescue isolated nodes (scattered points)
        rescued = self.topology.optimize_isolated_nodes(self.memory_latents, min_sim=0.1)
        if rescued > 0:
            logger.info(f"Rescued {rescued} isolated nodes by force-connecting to nearest neighbors.")

        # Inject divergent thinking (Random long-range connections)
        divergent_links = self.topology.create_divergent_links(n_links=int(len(new_latents) * 0.5))
        if divergent_links > 0:
            logger.info(f"Created {divergent_links} divergent links for lateral thinking.")
        
        # Apply homeostasis to keep network healthy
        self.topology.homeostasis()
        self.topology.prune()
                
        self.save_brain()
        
        return {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "improvement": initial_loss - final_loss,
            "topology_size": self.topology.size()
        }

    def recall(self, query_text: str, top_k=3) -> List[Dict]:
        """
        Recall memories using topological activation diffusion.
        Mechanism: Stimulus -> Seed Activation -> Diffusion -> Route Selection
        """
        if self.memory_latents is None or len(self.memory_latents) == 0:
            return []
            
        # 1. Encode query
        if self.embedder:
            with torch.no_grad():
                query_emb = self.embedder.encode(query_text, convert_to_tensor=True)
        else:
            return []
            
        # 2. Project to latent space (Stimulus)
        self.model.eval()
        with torch.no_grad():
            _, query_latent = self.model(query_emb.unsqueeze(0))
            query_vec = query_latent.cpu().numpy().flatten()
            
        # 3. Topological Activation
        # Instead of simple cosine similarity sort, we use the topology core
        # to simulate activation spreading through the memory graph.
        
        results = self.topology.recall(
            latents=self.memory_latents,
            query_vec=query_vec,
            top_k=top_k,
            seed_topk=max(16, top_k * 4),
            steps=3,
            allowed_from_ports=None,
            port_gates={"semantic": 1.0, "exec": 0.8},
            enable_plasticity=True,
            plasticity_lr=0.015,
            enable_structural_plasticity=True,
            structural_topn=max(16, top_k * 6),
            structural_min_sim=0.3,
        )
        
        # Enrich results with metadata
        enriched_results = []
        for res in results:
            idx = res['node_index']
            if idx < len(self.memory_metadata):
                meta = self.memory_metadata[idx]
                enriched_results.append({
                    "id": meta['id'],
                    "score": res['score'],
                    "trace": res['trace'], # Return the activation path
                    "type": meta.get("type"),
                    "tool": meta.get("tool"),
                    "skill": meta.get("skill"),
                    "args": meta.get("args"),
                    "content_preview": meta.get("content_preview", "")[:240]
                })
            
        return enriched_results

    def dream(self, n_samples=1):
        """
        Generate new ideas by sampling the latent space and decoding.
        This simulates 'dreaming' or 'imagination'.
        """
        self.model.eval()
        with torch.no_grad():
            # Sample random points in latent space
            z = torch.randn(n_samples, 64).to(self.device)
            reconstructed_vecs = self.model.decoder(z)
            
            # Note: We can't easily turn vectors back to text without a generative decoder,
            # but we can return the raw vectors as 'neural activity patterns'.
            return reconstructed_vecs.cpu().numpy()

