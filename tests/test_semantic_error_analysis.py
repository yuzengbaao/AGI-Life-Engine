import unittest
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.system_tools import SystemTools

class TestSemanticErrorAnalysis(unittest.TestCase):
    def setUp(self):
        self.tools = SystemTools()

    def test_analyze_traceback_simple(self):
        """Test parsing a standard ZeroDivisionError traceback."""
        dummy_traceback = """
Traceback (most recent call last):
  File "d:\\TRAE_PROJECT\\AGI\\core\\research\\lab.py", line 42, in run_experiment
    result = 1 / 0
ZeroDivisionError: division by zero
"""
        diagnosis_json = self.tools.analyze_traceback(dummy_traceback)
        diagnosis = json.loads(diagnosis_json)
        
        print(f"\n[Test] Input Traceback:\n{dummy_traceback}")
        print(f"[Test] Output Diagnosis:\n{json.dumps(diagnosis, indent=2)}")

        self.assertEqual(diagnosis['error_type'], 'ZeroDivisionError')
        self.assertIn('division by zero', diagnosis['message'])
        self.assertIn('lab.py', diagnosis['location'])
        self.assertIn('42', diagnosis['location'])

    def test_analyze_traceback_complex(self):
        """Test parsing a nested traceback."""
        dummy_traceback = """
Traceback (most recent call last):
  File "main.py", line 10, in <module>
    main()
  File "main.py", line 5, in main
    helper()
  File "utils.py", line 20, in helper
    raise ValueError("Invalid configuration")
ValueError: Invalid configuration
"""
        diagnosis_json = self.tools.analyze_traceback(dummy_traceback)
        diagnosis = json.loads(diagnosis_json)
        
        # Should catch the LAST file (utils.py)
        self.assertIn('utils.py', diagnosis['location'])
        self.assertIn('20', diagnosis['location'])
        self.assertEqual(diagnosis['error_type'], 'ValueError')

if __name__ == '__main__':
    unittest.main()
