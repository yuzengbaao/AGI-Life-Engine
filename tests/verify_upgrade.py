import asyncio
import sys
import os
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_agi.consciousness_engine import ContinuousConsciousness
from active_agi.event_driven_system import EnvironmentEventSource
from vector_memory_system import VectorMemorySystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_network_sensor():
    print("\n🌐 Testing Network Sensor...")
    env_source = EnvironmentEventSource()
    # Force initial state to False to trigger event if we are online
    env_source.network_online = False
    events = await env_source._check_network()
    
    if events:
        print(f"✅ Network event detected: {events[0].data}")
    else:
        print("⚠️ No network event (System might be offline)")

async def test_git_sensor():
    print("\n🐙 Testing Git Sensor...")
    env_source = EnvironmentEventSource()
    
    # Mock subprocess if git is not available or to force change
    # But let's try real first.
    # Fake a previous hash to trigger "Change"
    env_source.last_git_hash = "0000000000000000000000000000000000000000"
    
    events = await env_source._check_git()
    
    if events:
        print(f"✅ Git event detected: {events[0].data}")
    else:
        print("⚠️ No Git event (Not a git repo or git command failed)")

async def test_dreaming_phase():
    print("\n💤 Testing Dreaming Phase...")
    # Setup memory
    # Use a temporary db path
    db_path = os.path.abspath("./tests/test_dream_db")
    memory = VectorMemorySystem(save_dir=db_path)
    
    # Add some memories
    print("  Adding test memories...")
    # VectorMemorySystem.add_memory is synchronous
    memory.add_memory("Today I learned about Python async programming.", {"type": "experience"})
    memory.add_memory("The event system is now connected to real sensors.", {"type": "observation"})
    memory.add_memory("We need to optimize the memory retrieval latency.", {"type": "task"})
    
    engine = ContinuousConsciousness(memory_system=memory)
    
    # Force dream
    print("  Triggering dream phase (calling LLM)...")
    await engine._dream_phase()
    
    # Verify insight creation
    # Give it a moment to index if needed (VectorMemorySystem is sync usually but let's see)
    recent = memory.get_recent_memories(limit=5)
    found = False
    for m in recent:
        if "梦境洞察" in m.get('content', ''):
            print(f"✅ Dream insight created: {m['content']}")
            found = True
            break
            
    if not found:
        print("❌ Dream insight failed (check LLM availability or memory)")

async def test_associate_with_llm():
    print("\n🧠 Testing Association with LLM...")
    
    # Setup memory with related content
    db_path = os.path.abspath("./tests/test_assoc_db")
    memory = VectorMemorySystem(save_dir=db_path)
    
    # Add related memories
    print("  Adding related memories...")
    # Memory A: Cause
    await memory.store_memory("It rained heavily all night.", context={"id": "mem_1"})
    # Memory B: Effect
    await memory.store_memory("The streets are flooded this morning.", context={"id": "mem_2"})
    
    engine = ContinuousConsciousness(memory_system=memory)
    
    # Trigger associate
    print("  Triggering association...")
    
    # Mock active memories passed to associate
    active_mems = [
        {'id': 'mem_1', 'content': "It rained heavily all night."},
        # We don't pass mem_2, we expect it to be found via search
    ]
    
    # We need to make sure search finds mem_2. 
    # Since we just added it, and store_memory is async but maybe VectorMemorySystem needs manual save or is simple list?
    # VectorMemorySystem.search uses numpy on self.embeddings.
    # Check if store_memory updates embeddings immediately. Yes it does.
    
    associations = await engine._associate(active_mems)
    
    found_causal = False
    for assoc in associations:
        print(f"  Found association: {assoc.get('relation')} - {assoc.get('analysis', '')} (Score: {assoc.get('score'):.2f})")
        if 'cause' in str(assoc.get('relation', '')).lower() or 'cause' in str(assoc.get('analysis', '')).lower():
            found_causal = True
            
    if found_causal:
        print("✅ Deep causal analysis successful")
    else:
        print("⚠️ No causal link found (might be due to LLM unavailability or similarity threshold)")

async def main():
    try:
        await test_network_sensor()
        await test_git_sensor()
        await test_dreaming_phase()
        await test_associate_with_llm()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
