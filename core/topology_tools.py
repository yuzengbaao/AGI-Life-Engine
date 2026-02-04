import os
import json
import re
from typing import Dict, Any, List, Tuple


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _update_edge(adj: Dict[str, List[str]], a: str, b: str) -> None:
    if a not in adj:
        adj[a] = []
    if b not in adj[a]:
        adj[a].append(b)


def _safe_node_id(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_]", "_", name).strip("_")
    if not s:
        return "N"
    if s[0].isdigit():
        return f"N_{s}"
    return s


def _edge_list_from_adj(adj: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    for a, outs in adj.items():
        for b in outs:
            edges.append((a, b))
    return edges


def load_topology_graph(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def graph_to_visual_payload(graph_obj: Dict[str, Any]) -> Dict[str, Any]:
    nodes = list(graph_obj.get("nodes", []))
    adj = dict(graph_obj.get("adj", {}))
    node_items = [{"id": n, "label": n} for n in nodes]
    edges = [{"source": a, "target": b} for a, b in _edge_list_from_adj(adj)]
    return {
        "nodes": node_items,
        "edges": edges,
        "params": dict(graph_obj.get("params", {})),
    }


def write_topology_visual_payload(graph_obj: Dict[str, Any], output_path: str = "data/neural_memory/topology_visual.json") -> Dict[str, Any]:
    base_dir = os.path.dirname(output_path)
    if base_dir:
        _ensure_dir(base_dir)
    payload = graph_to_visual_payload(graph_obj)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return {"success": True, "nodes": len(payload["nodes"]), "edges": len(payload["edges"]), "path": output_path}


def graph_to_mermaid(graph_obj: Dict[str, Any], direction: str = "TD") -> str:
    nodes = list(graph_obj.get("nodes", []))
    adj = dict(graph_obj.get("adj", {}))
    id_map = {n: _safe_node_id(n) for n in nodes}
    lines: List[str] = [f"graph {direction}"]
    for n in nodes:
        nid = id_map[n]
        lines.append(f'{nid}["{n}"]')
    for a, b in _edge_list_from_adj(adj):
        if a not in id_map:
            id_map[a] = _safe_node_id(a)
            lines.append(f'{id_map[a]}["{a}"]')
        if b not in id_map:
            id_map[b] = _safe_node_id(b)
            lines.append(f'{id_map[b]}["{b}"]')
        lines.append(f"{id_map[a]} --> {id_map[b]}")
    return "\n".join(lines) + "\n"


def write_mermaid_graph(graph_obj: Dict[str, Any], output_path: str = "data/neural_memory/topology_graph.mmd") -> Dict[str, Any]:
    base_dir = os.path.dirname(output_path)
    if base_dir:
        _ensure_dir(base_dir)
    mermaid = graph_to_mermaid(graph_obj)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(mermaid)
    return {"success": True, "path": output_path}


def build_topology_graph(
    log_path: str = "logs/flow_cycle.jsonl",
    output_path: str = "data/neural_memory/topology_graph.json",
    limit: int = 200,
) -> Dict[str, Any]:
    if not os.path.exists(log_path):
        return {"success": False, "error": "log file not found", "path": log_path}
    entries: List[Dict[str, Any]] = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                entries.append(obj)
        entries = entries[-limit:]
    except Exception as e:
        return {"success": False, "error": str(e), "path": log_path}
    nodes: List[str] = []
    adj: Dict[str, List[str]] = {}
    core_components = [
        "AGI_Life_Engine",
        "PlannerAgent",
        "ExecutorAgent",
        "CriticAgent",
        "EvolutionController",
        "PerceptionManager",
        "MemorySystem",
    ]
    for name in core_components:
        if name not in nodes:
            nodes.append(name)
    last_ts = 0
    for entry in entries:
        ts = entry.get("timestamp", 0)
        if isinstance(ts, (int, float)):
            if ts > last_ts:
                last_ts = int(ts)
        has_goal = bool(entry.get("goal"))
        has_step = entry.get("step", 0) > 0
        has_verification = bool(entry.get("verification"))
        has_evo = bool(entry.get("evolution"))
        context_str = str(entry.get("context", "")).lower()
        memory_mention = "memory" in context_str or bool(entry.get("memory"))
        if has_goal:
            _update_edge(adj, "AGI_Life_Engine", "PlannerAgent")
            _update_edge(adj, "PlannerAgent", "AGI_Life_Engine")
        if has_step:
            _update_edge(adj, "PlannerAgent", "ExecutorAgent")
            _update_edge(adj, "ExecutorAgent", "PlannerAgent")
        if has_verification:
            _update_edge(adj, "ExecutorAgent", "CriticAgent")
            _update_edge(adj, "CriticAgent", "AGI_Life_Engine")
        if has_evo:
            _update_edge(adj, "AGI_Life_Engine", "EvolutionController")
            _update_edge(adj, "EvolutionController", "AGI_Life_Engine")
        if memory_mention:
            _update_edge(adj, "ExecutorAgent", "MemorySystem")
            _update_edge(adj, "MemorySystem", "ExecutorAgent")
    n = len(nodes)
    base_dir = os.path.dirname(output_path)
    if base_dir:
        _ensure_dir(base_dir)
    edge_list = [{"from": a, "to": b} for a, b in _edge_list_from_adj(adj)]
    graph_obj = {
        "n": n,
        "nodes": nodes,
        "adj": adj,
        "edges": edge_list,
        "params": {"timestamp": last_ts},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_obj, f, ensure_ascii=False)
    return {"success": True, "nodes": n, "edges": sum(len(v) for v in adj.values()), "path": output_path}
