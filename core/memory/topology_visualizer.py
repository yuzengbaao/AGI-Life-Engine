"""
🧠 数字神经大脑拓扑可视化器
Digital Neural Brain Topology Visualizer

提供交互式 HTML 可视化，展示神经拓扑的连接结构。
支持：
- 节点颜色编码（按类型/活跃度）
- 边权重可视化
- 分形子图高亮
- 统计信息导出
"""

from __future__ import annotations

import json
import os
import math
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.memory.topology_memory import TopologicalMemoryCore

# 尝试导入 pyvis，如果不可用则使用纯 HTML 方案
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False


class TopologyVisualizer:
    """
    数字神经大脑拓扑可视化器。
    
    支持两种渲染模式：
    1. PyVis 模式（推荐）：交互式物理布局
    2. 纯 HTML 模式：使用 vis.js CDN，无需额外依赖
    """
    
    def __init__(
        self,
        topology: "TopologicalMemoryCore",
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        初始化可视化器。
        
        Args:
            topology: TopologicalMemoryCore 实例
            metadata: 可选的节点元数据列表
        """
        self.topology = topology
        self.metadata = metadata or []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取拓扑统计信息。"""
        total_edges = sum(len(edges) for edges in self.topology._adj.values())
        fractal_count = len(getattr(self.topology, '_subgraphs', {}))
        
        return {
            "total_nodes": self.topology.size(),
            "total_edges": total_edges,
            "fractal_subgraphs": fractal_count,
            "avg_degree": total_edges / max(1, self.topology.size()),
            "max_degree": self.topology.max_degree,
            "min_edge_weight": self.topology.min_edge_weight,
        }
    
    def render_html(
        self,
        output_path: str = "./workspace/neural_brain_topology.html",
        max_nodes: int = 500,
        highlight_nodes: Optional[List[int]] = None,
        title: str = "🧠 数字神经大脑拓扑可视化",
    ) -> Dict[str, Any]:
        """
        渲染拓扑图到交互式 HTML 文件。
        
        Args:
            output_path: 输出 HTML 文件路径
            max_nodes: 最大渲染节点数（防止浏览器性能问题）
            highlight_nodes: 需要高亮的节点索引列表
            title: 页面标题
        
        Returns:
            包含渲染结果信息的字典
        """
        highlight_set = set(highlight_nodes or [])
        subgraphs = getattr(self.topology, '_subgraphs', {})
        
        # 确定要渲染的节点（采样策略）
        total_nodes = self.topology.size()
        if total_nodes <= max_nodes:
            sample_nodes = list(range(total_nodes))
        else:
            # 优先保留：高亮节点 + 分形节点 + 高连接度节点
            priority_nodes = set(highlight_set) | set(subgraphs.keys())
            
            # 计算每个节点的连接度
            degrees = {}
            for i in range(total_nodes):
                degrees[i] = len(self.topology._adj.get(i, []))
            
            # 按连接度排序，取 top
            sorted_by_degree = sorted(degrees.items(), key=lambda x: -x[1])
            remaining_slots = max_nodes - len(priority_nodes)
            
            sample_nodes = list(priority_nodes)
            for node_id, _ in sorted_by_degree:
                if len(sample_nodes) >= max_nodes:
                    break
                if node_id not in priority_nodes:
                    sample_nodes.append(node_id)
        
        sample_set = set(sample_nodes)
        
        if PYVIS_AVAILABLE:
            return self._render_with_pyvis(
                output_path, sample_nodes, sample_set, 
                highlight_set, subgraphs, title
            )
        else:
            return self._render_pure_html(
                output_path, sample_nodes, sample_set,
                highlight_set, subgraphs, title
            )
    
    def _render_with_pyvis(
        self,
        output_path: str,
        sample_nodes: List[int],
        sample_set: set,
        highlight_set: set,
        subgraphs: Dict[int, Any],
        title: str,
    ) -> Dict[str, Any]:
        """使用 PyVis 渲染。"""
        net = Network(
            height="900px",
            width="100%",
            bgcolor="#0a0a1a",
            font_color="#ffffff",
            heading=title,
        )
        net.barnes_hut(
            gravity=-3000,
            central_gravity=0.3,
            spring_length=150,
            spring_strength=0.01,
            damping=0.09,
        )
        
        # 添加节点
        for node_id in sample_nodes:
            # 确定节点颜色
            if node_id in highlight_set:
                color = "#ff4444"  # 红色 - 高亮
                size = 25
            elif node_id in subgraphs:
                color = "#00ff88"  # 绿色 - 分形节点
                size = 20
            else:
                # 根据连接度调整颜色
                degree = len(self.topology._adj.get(node_id, []))
                if degree >= 15:
                    color = "#ffaa00"  # 橙色 - 高连接度
                    size = 15
                elif degree >= 5:
                    color = "#4488ff"  # 蓝色 - 中连接度
                    size = 12
                else:
                    color = "#666688"  # 灰蓝 - 低连接度
                    size = 8
            
            # 获取元数据标签
            label = str(node_id)
            tooltip = f"Node {node_id}"
            if node_id < len(self.metadata):
                meta = self.metadata[node_id]
                if isinstance(meta, dict):
                    mem_type = meta.get("type", "unknown")
                    preview = meta.get("content_preview", "")[:100]
                    tooltip = f"[{mem_type}] {preview}"
            
            net.add_node(
                node_id,
                label=label,
                color=color,
                size=size,
                title=tooltip,
            )
        
        # 添加边
        edge_count = 0
        for src_id in sample_nodes:
            edges = self.topology._adj.get(src_id, [])
            for edge in edges:
                if edge.to_idx in sample_set:
                    net.add_edge(
                        src_id,
                        edge.to_idx,
                        value=edge.weight,
                        title=f"weight: {edge.weight:.3f}, kind: {edge.kind}",
                        color="#334455" if edge.kind != "divergent" else "#884488",
                    )
                    edge_count += 1
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        net.save_graph(output_path)
        
        return {
            "status": "ok",
            "renderer": "pyvis",
            "path": output_path,
            "nodes_rendered": len(sample_nodes),
            "edges_rendered": edge_count,
            "total_nodes": self.topology.size(),
        }
    
    def _render_pure_html(
        self,
        output_path: str,
        sample_nodes: List[int],
        sample_set: set,
        highlight_set: set,
        subgraphs: Dict[int, Any],
        title: str,
    ) -> Dict[str, Any]:
        """使用纯 HTML + vis.js CDN 渲染（无需 pyvis）。"""
        
        # 构建节点数据
        nodes_data = []
        for node_id in sample_nodes:
            if node_id in highlight_set:
                color = "#ff4444"
                size = 25
            elif node_id in subgraphs:
                color = "#00ff88"
                size = 20
            else:
                degree = len(self.topology._adj.get(node_id, []))
                if degree >= 15:
                    color = "#ffaa00"
                    size = 15
                elif degree >= 5:
                    color = "#4488ff"
                    size = 12
                else:
                    color = "#666688"
                    size = 8
            
            tooltip = f"Node {node_id}"
            if node_id < len(self.metadata):
                meta = self.metadata[node_id]
                if isinstance(meta, dict):
                    mem_type = meta.get("type", "unknown")
                    preview = (meta.get("content_preview", "") or "")[:80]
                    tooltip = f"[{mem_type}] {preview}"
            
            nodes_data.append({
                "id": node_id,
                "label": str(node_id),
                "color": color,
                "size": size,
                "title": tooltip,
            })
        
        # 构建边数据
        edges_data = []
        for src_id in sample_nodes:
            edges = self.topology._adj.get(src_id, [])
            for edge in edges:
                if edge.to_idx in sample_set:
                    edges_data.append({
                        "from": src_id,
                        "to": edge.to_idx,
                        "value": edge.weight,
                        "title": f"weight: {edge.weight:.3f}",
                        "color": "#884488" if edge.kind == "divergent" else "#334455",
                    })
        
        stats = self.get_stats()
        
        html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 100%);
            color: #ffffff;
            min-height: 100vh;
        }}
        .header {{
            padding: 20px;
            background: rgba(0,0,0,0.3);
            border-bottom: 1px solid #333;
        }}
        .header h1 {{
            font-size: 24px;
            margin-bottom: 10px;
        }}
        .stats {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .stat-item {{
            background: rgba(255,255,255,0.1);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 14px;
        }}
        .stat-value {{
            color: #00ff88;
            font-weight: bold;
        }}
        #network {{
            width: 100%;
            height: calc(100vh - 120px);
            background: #0a0a1a;
        }}
        .legend {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 10px;
            font-size: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 5px 0;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div class="stats">
            <div class="stat-item">节点: <span class="stat-value">{stats['total_nodes']:,}</span></div>
            <div class="stat-item">连接: <span class="stat-value">{stats['total_edges']:,}</span></div>
            <div class="stat-item">分形子图: <span class="stat-value">{stats['fractal_subgraphs']}</span></div>
            <div class="stat-item">平均度: <span class="stat-value">{stats['avg_degree']:.2f}</span></div>
            <div class="stat-item">渲染节点: <span class="stat-value">{len(sample_nodes)}</span></div>
        </div>
    </div>
    
    <div id="network"></div>
    
    <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:#ff4444"></div> 高亮节点</div>
        <div class="legend-item"><div class="legend-dot" style="background:#00ff88"></div> 分形节点</div>
        <div class="legend-item"><div class="legend-dot" style="background:#ffaa00"></div> 高连接度</div>
        <div class="legend-item"><div class="legend-dot" style="background:#4488ff"></div> 中连接度</div>
        <div class="legend-item"><div class="legend-dot" style="background:#666688"></div> 低连接度</div>
    </div>
    
    <script>
        const nodes = new vis.DataSet({json.dumps(nodes_data)});
        const edges = new vis.DataSet({json.dumps(edges_data)});
        
        const container = document.getElementById('network');
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
            nodes: {{
                shape: 'dot',
                font: {{ color: '#ffffff', size: 10 }},
                borderWidth: 1,
                borderWidthSelected: 3,
            }},
            edges: {{
                width: 0.5,
                smooth: {{ type: 'continuous' }},
            }},
            physics: {{
                barnesHut: {{
                    gravitationalConstant: -3000,
                    centralGravity: 0.3,
                    springLength: 120,
                    springConstant: 0.01,
                    damping: 0.09,
                }},
                stabilization: {{
                    iterations: 150,
                    updateInterval: 25,
                }},
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                hideEdgesOnDrag: true,
                hideEdgesOnZoom: true,
            }},
        }};
        
        const network = new vis.Network(container, data, options);
        
        network.on("stabilizationIterationsDone", function() {{
            network.setOptions({{ physics: false }});
        }});
    </script>
</body>
</html>'''
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return {
            "status": "ok",
            "renderer": "pure_html",
            "path": output_path,
            "nodes_rendered": len(sample_nodes),
            "edges_rendered": len(edges_data),
            "total_nodes": self.topology.size(),
        }
    
    def export_stats_json(self, output_path: str = "./workspace/topology_stats.json") -> str:
        """导出拓扑统计为 JSON 文件。"""
        stats = self.get_stats()
        
        # 添加连接度分布
        degree_distribution = {}
        for i in range(self.topology.size()):
            degree = len(self.topology._adj.get(i, []))
            bucket = (degree // 5) * 5  # 按5分桶
            degree_distribution[f"{bucket}-{bucket+4}"] = degree_distribution.get(f"{bucket}-{bucket+4}", 0) + 1
        
        stats["degree_distribution"] = degree_distribution
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        return output_path


def visualize_topology(
    topology: "TopologicalMemoryCore",
    metadata: Optional[List[Dict[str, Any]]] = None,
    output_path: str = "./workspace/neural_brain_topology.html",
    max_nodes: int = 500,
) -> Dict[str, Any]:
    """
    便捷函数：一键生成拓扑可视化。
    
    Args:
        topology: TopologicalMemoryCore 实例
        metadata: 可选的节点元数据
        output_path: 输出 HTML 路径
        max_nodes: 最大节点数
    
    Returns:
        渲染结果信息
    """
    visualizer = TopologyVisualizer(topology, metadata)
    return visualizer.render_html(output_path, max_nodes)
