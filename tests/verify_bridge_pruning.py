import sys
import os
import asyncio
import shutil
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from core.neuro_symbolic_bridge import NeuroSymbolicBridge
from core.memory_enhanced_v2 import EnhancedExperienceMemory

# Mock Bridge
class MockBridge(NeuroSymbolicBridge):
    def __init__(self):
        super().__init__()
        self.concept_states = {}

async def test_pruning():
    print("Testing Bridge Pruning in Memory System...")
    
    # 1. Setup Memory (Use temp dir)
    test_dir = "tests/temp_memory_pruning"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    try:
        memory = EnhancedExperienceMemory(memory_dir=test_dir)
        bridge = MockBridge()
        
        # 2. Add Memories
        print("Adding memories...")
        # Mem 1: Good memory
        id1 = memory.add_experience("Good concept", "keep", 1.0, details={"concept_id": "concept_good"})
        # Mem 2: Bad memory (Hallucination)
        id2 = memory.add_experience("Bad concept", "discard", 0.5, details={"concept_id": "concept_bad"})
        
        # 3. Setup Bridge State
        bridge.concept_states["concept_good"] = "MAINTAIN"
        bridge.concept_states["concept_bad"] = "REJECT_NOISE"
        
        # 4. Run Consolidation
        print("Running consolidation...")
        await memory.forget_and_consolidate(bridge=bridge)
        
        # 5. Verify
        print("Verifying results...")
        results = memory.collection.get(ids=[id1, id2])
        found_ids = results['ids']
        
        success = True
        if id1 in found_ids:
            print("✅ Good memory preserved.")
        else:
            print("❌ Good memory WRONGLY deleted.")
            success = False
            
        if id2 not in found_ids:
            print("✅ Bad memory successfully pruned.")
        else:
            print("❌ Bad memory NOT pruned.")
            success = False
            
        if success:
            print("🎉 Test PASSED: Bridge Pruning Works!")
        else:
            print("💥 Test FAILED.")

    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    asyncio.run(test_pruning())
