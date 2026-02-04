import os
import shutil
import time
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.memory_enhanced import EnhancedExperienceMemory

def test_enhanced_memory():
    print("\n🧪 Starting Enhanced Memory Test...")
    
    # 1. Setup Test Environment
    test_db_path = "test_chroma_db"
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)
        print("🧹 Cleaned up old test DB.")
        
    try:
        # 2. Initialize Memory
        print("🚀 Initializing Memory System...")
        memory = EnhancedExperienceMemory(memory_dir=test_db_path)
        
        # 3. Test: Add Experiences
        print("💾 Storing memories...")
        id1 = memory.add_experience(
            context="High system load detected (CPU > 90%)",
            action="TRIGGER_GC_AND_THROTTLE",
            outcome=0.9,
            details={"strategy": "conservative"}
        )
        
        id2 = memory.add_experience(
            context="User requested system status",
            action="DISPLAY_DASHBOARD",
            outcome=1.0,
            details={"user_id": "admin"}
        )
        
        id3 = memory.add_experience(
            context="Memory leak in vision module",
            action="RESTART_PROCESS",
            outcome=0.8,
            details={"module": "vision"}
        )
        
        print(f"✅ Stored 3 memories. IDs: {id1}, {id2}, {id3}")
        
        # 4. Test: Semantic Retrieval
        print("\n🔍 Testing Semantic Search:")
        
        # Query 1: Similar to first memory
        query1 = "System is running very slow and hot"
        results1 = memory.retrieve_relevant(query1)
        print(f"  Query: '{query1}'")
        if results1 and "High system load" in results1[0]['content']:
            print(f"  ✅ Match found: {results1[0]['content']}")
        else:
            print(f"  ❌ Failed match. Top result: {results1[0]['content'] if results1 else 'None'}")

        # Query 2: Similar to second memory
        query2 = "Show me the status panel"
        results2 = memory.retrieve_relevant(query2)
        print(f"  Query: '{query2}'")
        if results2 and "DISPLAY_DASHBOARD" in results2[0]['metadata']['action']:
            print(f"  ✅ Match found: {results2[0]['metadata']['action']}")
        else:
            print(f"  ❌ Failed match.")
            
        # 5. Test: Legacy Consolidation (Mock)
        print("\n📥 Testing Legacy Import...")
        mock_json = "test_legacy.json"
        with open(mock_json, "w", encoding="utf-8") as f:
            f.write('[{"context": "Old memory", "action": "DO_NOTHING", "outcome": 0.1}]')
            
        memory.consolidate_legacy_memory(mock_json)
        
        # Verify count
        stats = memory.get_stats()
        print(f"📊 Final Stats: {stats}")
        
        if stats['total_memories'] == 4:
            print("✅ Import successful (Total = 4)")
        else:
            print(f"❌ Import count mismatch: {stats['total_memories']}")
            
        # Cleanup
        if os.path.exists(mock_json):
            os.remove(mock_json)
            
    except Exception as e:
        print(f"💥 Test Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Optional: cleanup DB
        # shutil.rmtree(test_db_path)
        print("\n🏁 Test Complete.")

if __name__ == "__main__":
    test_enhanced_memory()
