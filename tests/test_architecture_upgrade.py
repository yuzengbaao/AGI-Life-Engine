import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.layered_identity import LayeredIdentity

class TestArchitectureUpgrade(unittest.TestCase):
    def test_layered_identity_structure(self):
        """Verify LayeredIdentity has all three layers"""
        identity = LayeredIdentity(".")
        
        self.assertIsNotNone(identity.core)
        self.assertIsNotNone(identity.slow)
        self.assertIsNotNone(identity.fast)
        
        # Check Core
        self.assertEqual(identity.core.system_name, "TRAE AGI")
        
        # Check Context Generation
        context = identity.get_system_prompt_context()
        print(f"\n[Identity Context Preview]\n{context}")
        self.assertIn("SYSTEM IDENTITY", context)
        self.assertIn("PERSONALITY", context)
        self.assertIn("CURRENT STATE", context)

    def test_identity_feedback_loop(self):
        """Verify feedback updates the fast layer"""
        identity = LayeredIdentity(".")
        initial_energy = identity.fast.energy_level
        
        # Simulate success
        identity.update_after_action("TEST_ACTION", True, 0.8)
        
        # Energy should increase (or stay same if maxed, but mood should be happy)
        print(f"Energy: {initial_energy} -> {identity.fast.energy_level}")
        print(f"Mood: {identity.fast.emotional_state}")
        
        self.assertNotEqual(identity.fast.emotional_state, "Neutral") # Should update based on success

if __name__ == "__main__":
    unittest.main()
