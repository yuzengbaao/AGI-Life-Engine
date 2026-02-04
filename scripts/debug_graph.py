import json
import os

def check():
    graph_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "knowledge", "arch_graph.json")
    with open(graph_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    nodes = data.get("nodes", [])
    print(f"Total nodes: {len(nodes)}")
    
    viz_nodes = [n for n in nodes if "world_visualizations" in n.get("path", "") or "world_visualizations" in n.get("id", "")]
    print(f"Nodes with 'world_visualizations' in path/id: {len(viz_nodes)}")
    
    if viz_nodes:
        print("Sample node:", viz_nodes[0])

    viz_dir_nodes = [n for n in nodes if "visualization" in n.get("path", "")]
    print(f"Nodes with 'visualization' in path: {len(viz_dir_nodes)}")

if __name__ == "__main__":
    check()
