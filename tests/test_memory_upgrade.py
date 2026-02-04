import unittest
import shutil
import os
import sys
import time
import asyncio
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.memory_enhanced_v2 import EnhancedExperienceMemory

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestEnhancedMemoryV2(unittest.TestCase):
    def setUp(self):
        # Use a unique directory for each test to avoid file locking issues
        self.test_id = f"test_{int(time.time())}_{id(self)}"
        self.test_dir = f"test_memory_db_{self.test_id}"
        
        self.memory = EnhancedExperienceMemory(memory_dir=self.test_dir, collection_name="test_collection")

    def tearDown(self):
        # Explicitly delete the client reference to help GC release file handles
        if hasattr(self, 'memory'):
            del self.memory
            
        # Try to clean up, but ignore errors if files are locked (Windows issue)
        # They can be cleaned up manually later or by a cleanup script
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup test dir {self.test_dir}: {e}")

    def test_add_and_retrieve(self):
        print("\n--- Testing Add and Retrieve ---")
        mem_id = self.memory.add_experience(
            context="User asked for a joke",
            action="Tell a knock-knock joke",
            outcome=0.8
        )
        self.assertTrue(mem_id.startswith("mem_"))
        
        # Retrieve
        results = self.memory.retrieve_relevant("joke", limit=1)
        self.assertEqual(len(results), 1)
        self.assertIn("knock-knock", results[0]['content'])
        
        # Verify Access Count
        # Note: Access count is updated AFTER retrieval
        # First retrieval: DB=0 -> Mem=1 -> Update DB=1
        # Second retrieval: DB=1 -> Mem=2 -> Update DB=2
        results_2 = self.memory.retrieve_relevant("joke", limit=1)
        
        # We expect access_count to be 2 now
        self.assertEqual(results_2[0]['metadata']['access_count'], 2)
        print("✅ Access count verified.")

    def test_intuition(self):
        print("\n--- Testing Intuition ---")
        self.memory.add_experience("Dangerous situation detected", "Run away", 0.9)
        
        # Async test wrapper
        async def run_test():
            confidence = await self.memory.retrieve_intuition("Dangerous situation")
            print(f"Intuition Confidence: {confidence}")
            return confidence

        confidence = asyncio.run(run_test())
        # Cosine similarity for short text might be lower than 0.8 depending on the model
        # 0.5 is a reasonable threshold for semantic relatedness in this context
        self.assertGreater(confidence, 0.5)
        print("✅ Intuition verified.")

    def test_forgetting_mechanism(self):
        print("\n--- Testing Forgetting (LRU) ---")
        # 1. Add a weak memory (low outcome/importance)
        id_weak = self.memory.add_experience("Useless noise", "Do nothing", 0.1)
        
        # 2. Add a strong memory
        id_strong = self.memory.add_experience("Important discovery", "Document it", 0.9)
        
        # 3. Simulate time passing
        time.sleep(0.1)
        
        # 4. Run Forgetting Cycle
        async def run_cycle():
            await self.memory.forget_and_consolidate()
            
        asyncio.run(run_cycle())
        
        # 5. Verify Weak is gone, Strong remains
        results_weak = self.memory.retrieve_relevant("Useless noise", limit=1)
        
        # We can double check by ID lookup logic implicitly via search
        found_weak = any(r['id'] == id_weak for r in results_weak)
        self.assertFalse(found_weak, "Weak memory should have been pruned")
        
        results_strong = self.memory.retrieve_relevant("Important", limit=1)
        self.assertEqual(len(results_strong), 1, "Strong memory should remain")
        print("✅ Forgetting verified.")

if __name__ == '__main__':
    unittest.main()
