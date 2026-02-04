
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

def generate_topology_visualization(output_file="topology_viz.html"):
    base_dir = "./data/neural_memory"
    topology_path = os.path.join(base_dir, "topology_graph.json")
    metadata_path = os.path.join(base_dir, "memory_metadata.json")

    print(f"Loading topology from {topology_path}...")
    if not os.path.exists(topology_path):
        print("Topology file not found!")
        return

    with open(topology_path, "r", encoding="utf-8") as f:
        topo_data = json.load(f)

    print(f"Loading metadata from {metadata_path}...")
    meta_data = []
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

    nodes = []
    edges = []

    adj = topo_data.get("adj", {})
    
    # Process nodes
    # If metadata exists, use it for labels. Otherwise use indices.
    # To avoid too many nodes for browser rendering, we might need to limit or cluster.
    # For 2722 nodes, vis.js might be slow but manageable.
    
    existing_indices = set()
    for idx_str in adj.keys():
        existing_indices.add(int(idx_str))
        for e in adj[idx_str]:
            existing_indices.add(int(e["to_idx"]))

    print(f"Found {len(existing_indices)} active nodes in topology.")

    for idx in existing_indices:
        label = str(idx)
        title = f"Node {idx}"
        group = "default"
        
        if 0 <= idx < len(meta_data):
            meta = meta_data[idx]
            # Use a snippet of content as label
            content = meta.get("content", "")
            if content:
                label = content[:20] + "..." if len(content) > 20 else content
                title = content
            
            # Color by source or timestamp if available
            group = meta.get("source", "unknown")

        nodes.append({
            "id": int(idx),
            "label": label,
            "title": title,
            "group": group,
            "value": len(adj.get(str(idx), [])) # Size by degree
        })

    # Process edges
    edge_count = 0
    for from_idx_str, edge_list in adj.items():
        from_idx = int(from_idx_str)
        for e in edge_list:
            to_idx = int(e["to_idx"])
            weight = float(e["weight"])
            
            # Only add edge if weight is significant to reduce clutter?
            # Or just add all. Let's add all but make them thin.
            edges.append({
                "from": from_idx,
                "to": to_idx,
                "value": weight,
                "title": f"Weight: {weight:.2f}",
                "arrows": "to"
            })
            edge_count += 1

    print(f"Generated {len(nodes)} nodes and {len(edges)} edges.")

    # HTML Template with vis.js
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Fluid Intelligence Topology Visualization</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        #mynetwork {{
            width: 100%;
            height: 800px;
            border: 1px solid lightgray;
            background-color: #f8f9fa;
        }}
        .info {{
            padding: 10px;
            font-family: Arial, sans-serif;
        }}
    </style>
</head>
<body>
    <div class="info">
        <h1>Fluid Intelligence Topology</h1>
        <p>Nodes: {len(nodes)} | Edges: {len(edges)}</p>
        <p>Scroll to zoom, drag to pan. Click nodes to see details.</p>
    </div>
    <div id="mynetwork"></div>
    <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(nodes)});
        var edges = new vis.DataSet({json.dumps(edges)});

        var container = document.getElementById('mynetwork');
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        var options = {{
            nodes: {{
                shape: 'dot',
                scaling: {{
                    min: 10,
                    max: 30
                }},
                font: {{
                    size: 12,
                    face: 'Tahoma'
                }}
            }},
            edges: {{
                width: 0.15,
                color: {{ inherit: 'from' }},
                smooth: {{
                    type: 'continuous'
                }}
            }},
            physics: {{
                stabilization: false,
                barnesHut: {{
                    gravitationalConstant: -8000,
                    springConstant: 0.04,
                    springLength: 95
                }}
            }},
            interaction: {{
                tooltipDelay: 200,
                hideEdgesOnDrag: true
            }}
        }};
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>
    """

    out_path = os.path.abspath(output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Visualization saved to: {out_path}")

if __name__ == "__main__":
    generate_topology_visualization()
