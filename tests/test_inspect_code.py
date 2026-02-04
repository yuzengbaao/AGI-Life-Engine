import sys
import os
import unittest

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.system_tools import SystemTools

class TestSystemTools(unittest.TestCase):
    def setUp(self):
        self.tools = SystemTools()

    def test_inspect_code_summary(self):
        # Inspect self (this test file) or system_tools.py
        target = "core/system_tools.py"
        print(f"\nTesting inspect_code on {target}...")
        result = self.tools.inspect_code(target, mode="summary")
        print(f"Result:\n{result}")
        self.assertIn("Class: SystemTools", result)
        self.assertIn("Method: inspect_code", result)

    def test_run_command_existence(self):
        # Verify run_command still works (checking for regression)
        print("\nTesting run_command existence...")
        # We don't need to run a real command, just check if method exists
        self.assertTrue(hasattr(self.tools, 'run_command'))
        # Try a safe command
        result = self.tools.run_command("echo hello")
        self.assertIn("hello", result)

if __name__ == '__main__':
    unittest.main()