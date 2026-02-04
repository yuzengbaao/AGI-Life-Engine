import chromadb
import os
import sys

def inspect_chroma_db(path):
    print(f"\n--- Inspecting ChromaDB at: {path} ---")
    
    if not os.path.exists(path):
        print(f"❌ Path does not exist: {path}")
        return

    try:
        # Initialize client in persistent mode
        client = chromadb.PersistentClient(path=path)
        print("✅ Client initialized successfully.")
        
        # List collections
        collections = client.list_collections()
        print(f"📚 Found {len(collections)} collections:")
        
        for col in collections:
            print(f"  - Name: {col.name}")
            print(f"    Count: {col.count()}")
            
            # Peek at first 3 items
            if col.count() > 0:
                peek = col.peek(limit=3)
                print(f"    Sample Metadata: {peek['metadatas']}")
                print(f"    Sample Documents (truncated): {[d[:50] + '...' for d in peek['documents']]}")
            else:
                print("    (Empty collection)")
                
    except Exception as e:
        print(f"💥 Error inspecting DB: {e}")

if __name__ == "__main__":
    # Potential paths found in previous glob search
    paths_to_check = [
        r"d:\TRAE_PROJECT\AGI\memory_db",
        r"d:\TRAE_PROJECT\AGI\test_memory_db",
        r"d:\TRAE_PROJECT\AGI\data\neural_memory"
    ]
    
    print("🔍 Starting ChromaDB Forensic Analysis...")
    
    for path in paths_to_check:
        inspect_chroma_db(path)
        
    print("\n🏁 Analysis Complete.")
