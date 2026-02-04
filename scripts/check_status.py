import os
import time
import json

file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "knowledge", "arch_graph.json")

if os.path.exists(file_path):
    size = os.path.getsize(file_path)
    mtime = os.path.getmtime(file_path)
    print(f"File: {file_path}")
    print(f"Size: {size / 1024 / 1024:.2f} MB")
    print(f"Last Modified: {time.ctime(mtime)}")
    
    # Read a bit to see if 'document' is in there
    found = False
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read in chunks to avoid memory issues if huge
        while True:
            chunk = f.read(1024 * 1024) # 1MB chunk
            if not chunk:
                break
            if '"type": "document"' in chunk or '"type":"document"' in chunk:
                found = True
                break
    
    print(f"Contains 'document' nodes: {found}")
else:
    print(f"File not found: {file_path}")