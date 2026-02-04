import os
import sys
import unittest

# Add project root to sys.path
sys.path.append(os.getcwd())

from core.system_tools import SystemTools

class TestFixesP3(unittest.TestCase):
    def setUp(self):
        self.tools = SystemTools()

    def test_run_python_missing_file(self):
        print("\nTesting run_python_script with missing file...")
        result = self.tools.run_python_script("non_existent_script_12345.py")
        print(f"Result: {result}")
        self.assertIn("Error: Python script 'non_existent_script_12345.py' not found", result)
        self.assertIn("Please create it first", result)

    def test_adapter_import(self):
        print("\nTesting adapter import (Absolute Import)...")
        try:
            from core.perception.processors.adapter import RealtimeVideoAdapter
            print("Successfully imported RealtimeVideoAdapter using absolute path.")
        except ImportError as e:
            self.fail(f"Failed to import adapter: {e}")
        except Exception as e:
            self.fail(f"Unexpected error importing adapter: {e}")

if __name__ == '__main__':
    unittest.main()
