import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory_bridge import MemoryBridge

def test_bridge():
    print("Testing MemoryBridge...")
    bridge = MemoryBridge()
    
    if not bridge.memory:
        print("❌ Failed to initialize bridge memory.")
        return
        
    print("✅ Bridge initialized.")
    
    query = "AGI构建的第一性原理"
    print(f"Searching for: {query}")
    
    results = bridge.search(query)
    
    if results and "error" not in results[0]:
        print(f"✅ Found {len(results)} results.")
        for i, res in enumerate(results):
            print(f"Result {i+1}: {res.get('context')[:50]}...")
    else:
        print(f"❌ Search failed or no results: {results}")

if __name__ == "__main__":
    test_bridge()
