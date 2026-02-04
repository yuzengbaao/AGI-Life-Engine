import json
import os
import sys
from collections import Counter

# 验证知识图谱覆盖率脚本
# Usage: python scripts/verify_knowledge.py

def verify_coverage():
    # 定位知识图谱文件
    graph_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "knowledge", "arch_graph.json")
    
    if not os.path.exists(graph_file):
        print(f"❌ 错误: 找不到图谱文件: {graph_file}")
        return

    print(f"🔍 正在分析知识图谱: {graph_file}")
    
    try:
        with open(graph_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        nodes = data.get("nodes", [])
        links = data.get("links", [])
        
        print(f"   - 总节点数 (Nodes): {len(nodes)}")
        print(f"   - 总连线数 (Links): {len(links)}")
        
        # 1. 分析节点类型分布
        node_types = Counter([n.get("type", "unknown") for n in nodes])
        print("\n📊 节点类型分布 (Node Types):")
        for type_name, count in node_types.most_common():
            print(f"   - {type_name}: {count}")
        
        # 2. 分析文档覆盖的目录结构
        print("\n📂 文档目录覆盖 (Top Directories):")
        doc_nodes = [n for n in nodes if n.get("type") == "document"]
        dir_counts = Counter()
        
        for n in doc_nodes:
            path = n.get("path", "")
            if path:
                # 获取顶级目录名称
                parts = path.replace("\\", "/").split("/")
                if len(parts) > 1:
                    top_dir = parts[0]
                    dir_counts[top_dir] += 1
                else:
                    dir_counts["root"] += 1
        
        for directory, count in dir_counts.most_common(15):
            print(f"   - {directory}/: {count} 个文件")
        
        # 3. 针对性检查核心研究范围
        print("\n🧪 研究范围检查 (Scope Check):")
        # 假设核心关注 core, visualization, scripts
        has_core = dir_counts["core"] > 0
        has_viz = dir_counts["visualization"] > 0 or dir_counts["world_visualizations"] > 0
        has_scripts = dir_counts["scripts"] > 0
        
        print(f"   - 核心逻辑 (Core Logic): {'✅ 已覆盖' if has_core else '❌ 未发现'}")
        print(f"   - 可视化模块 (Visualization): {'✅ 已覆盖' if has_viz else '❌ 未发现'}")
        print(f"   - 自动化脚本 (Scripts): {'✅ 已覆盖' if has_scripts else '❌ 未发现'}")

    except Exception as e:
        print(f"❌ 分析时发生错误: {e}")

if __name__ == "__main__":
    verify_coverage()
