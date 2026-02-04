import os
import time
from datetime import datetime

def check_file(path):
    if os.path.exists(path):
        mtime = os.path.getmtime(path)
        dt = datetime.fromtimestamp(mtime)
        print(f"File: {path}")
        print(f"  Exists: YES")
        print(f"  Modified: {dt}")
        print(f"  Size: {os.path.getsize(path)} bytes")
    else:
        print(f"File: {path}")
        print(f"  Exists: NO")

files = [
    r"d:\TRAE_PROJECT\AGI\data\knowledge_graph.json",
    r"d:\TRAE_PROJECT\AGI\data\knowledge\arch_graph.json",
    r"d:\TRAE_PROJECT\AGI\memory\embeddings.npy",
]

# Check latest insight
insight_dir = r"d:\TRAE_PROJECT\AGI\data\insights"
if os.path.exists(insight_dir):
    files_in_dir = [os.path.join(insight_dir, f) for f in os.listdir(insight_dir) if f.endswith('.md')]
    if files_in_dir:
        latest_insight = max(files_in_dir, key=os.path.getmtime)
        files.append(latest_insight)
    else:
        print(f"No insights found in {insight_dir}")
else:
    print(f"Insight dir not found: {insight_dir}")

for f in files:
    check_file(f)
    print("-" * 20)