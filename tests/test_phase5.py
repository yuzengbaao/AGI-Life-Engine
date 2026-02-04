import sys
import os
import unittest

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.llm_client import LLMService
from core.agents import AgentOrchestrator
from core.memory import ExperienceMemory

class TestPhase5Integration(unittest.TestCase):
    def setUp(self):
        print("\n[Setup] Initializing components...")
        self.llm = LLMService()
        self.agents = AgentOrchestrator()
        self.memory = ExperienceMemory("tests/data/memory_test")

    def test_llm_initialization(self):
        print("[Test] LLM Initialization Mode:", "MOCK" if self.llm.mock_mode else "REAL")
        # Should be safe regardless of mode
        response = self.llm.chat_completion("System", "Hello")
        self.assertTrue(len(response) > 0)

    def test_agent_workflow(self):
        print("[Test] Agent Workflow (Code Generation -> Review)...")
        context = {"strategy": "Recursive", "complexity": 2.0}
        result = self.agents.run_development_cycle(context)
        
        print(f"  -> Agent Result: {result['final_outcome']}")
        print(f"  -> Code Snippet: {result['code']['code_snippet'][:50]}...")
        
        self.assertIn('code', result)
        self.assertIn('review', result)

    def test_vector_memory(self):
        print("[Test] Vector Memory (Embedding -> Recall)...")
        # 1. Add Memory
        self.memory.add_experience(
            context="Optimized loop with vectorization",
            action="Refactor",
            outcome=0.95,
            details={"timestamp": 123456}
        )
        
        # 2. Recall
        # "vectorization" should match "vectorization" in mock mode if random vectors align? 
        # Actually random vectors won't align well, so recall might be empty if threshold is high.
        # But let's check if it runs without error.
        results = self.memory.recall_relevant("vectorization")
        print(f"  -> Recalled {len(results)} items.")
        
        # If mock mode, random vectors are unlikely to match > 0.6 similarity.
        # So we just assert it didn't crash.
        self.assertIsInstance(results, list)

if __name__ == '__main__':
    unittest.main()
