import unittest
import os
import sys
import time

# Add parent directory to path to import context_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_engine import PersistentContextEngine, ContextLifecycleManager, check_context_consistency

class TestContextEngine(unittest.TestCase):
    def setUp(self):
        self.test_db = "test_context.db"
        self.engine = PersistentContextEngine(self.test_db)
        self.lifecycle = ContextLifecycleManager(self.engine)
        self.session_id = "test_session_001"

    def tearDown(self):
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except:
                pass

    def test_save_and_retrieve(self):
        self.lifecycle.start_new_session(self.session_id)
        
        self.engine.save_message(self.session_id, "user", "Hello")
        self.engine.save_message(self.session_id, "assistant", "Hi there")
        
        messages = self.engine.retrieve_session(self.session_id)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['message'], "Hello")
        self.assertEqual(messages[1]['message'], "Hi there")
        self.assertEqual(messages[0]['speaker'], "user")

    def test_consistency_check_pass(self):
        messages = [
            {'speaker': 'user', 'message': 'A'},
            {'speaker': 'assistant', 'message': 'B'},
            {'speaker': 'user', 'message': 'C'}
        ]
        consistent, msg = check_context_consistency(messages)
        self.assertTrue(consistent)

    def test_consistency_check_fail_duplicate(self):
        messages = [
            {'speaker': 'user', 'message': 'A'},
            {'speaker': 'user', 'message': 'A'}
        ]
        consistent, msg = check_context_consistency(messages)
        self.assertFalse(consistent)
        self.assertIn("重复消息", msg)

    def test_consistency_check_fail_speaker(self):
        messages = [
            {'speaker': 'user', 'message': 'A'},
            {'speaker': 'user', 'message': 'B'},
            {'speaker': 'user', 'message': 'C'}
        ]
        consistent, msg = check_context_consistency(messages)
        self.assertFalse(consistent)
        self.assertIn("连续发言", msg)

if __name__ == '__main__':
    unittest.main()
