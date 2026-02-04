import os
import sys
import time
import unittest
import shutil
import tempfile
import json
from unittest.mock import MagicMock, patch

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.observer import PassiveObserver, ObservationSession
from core.macro_system import ActionEvent

class TestPassiveObserver(unittest.TestCase):
    def setUp(self):
        # Create a temp directory for test
        self.test_dir = tempfile.mkdtemp()
        self.observer = PassiveObserver(storage_dir=self.test_dir)
        
        # Mock LLM Service
        self.observer.llm_service = MagicMock()
        self.observer.llm_service.chat_with_vision.return_value = "**Intent:** Mocked Intent\n**Process:** Mocked Process"

    def tearDown(self):
        # Clean up
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch("core.observer.pyautogui")
    @patch("core.observer.MacroRecorder")
    def test_observation_flow(self, mock_recorder_cls, mock_pyautogui):
        """Tests the full flow of start -> observe -> stop -> analyze."""
        
        # 1. Setup Mocks
        mock_recorder = mock_recorder_cls.return_value
        # When stop() is called, it returns the events (and also likely sets self.events)
        # In the real code, observer reads self.recorder.events
        mock_recorder.events = [
            ActionEvent(time.time(), "click", {"x": 100, "y": 100, "button": "left"}).to_dict(),
            ActionEvent(time.time() + 0.5, "key_down", {"key": "Key.enter"}).to_dict()
        ]
        
        # Mock screenshot to return a dummy image
        from PIL import Image
        dummy_img = Image.new('RGB', (100, 100), color = 'red')
        mock_pyautogui.screenshot.return_value = dummy_img
        
        # 2. Start Observation
        print("\nStarting Observation...")
        self.observer.start_observation()
        self.assertTrue(self.observer.is_observing)
        self.assertIsNotNone(self.observer.session)
        
        # Verify recorder started
        mock_recorder.start.assert_called_once()
        
        # 3. Wait a bit (simulate time passing for screenshots)
        time.sleep(1.5)
        
        # 4. Stop Observation
        print("Stopping Observation...")
        # Force cleanup=False because our test is short and mocked events might be few
        # But wait, we added mocked events to recorder.events manually? 
        # Actually in the mock test we set mock_recorder.events. 
        # The real stop_observation merges these.
        # But duration is simulated by sleep(1.5). 
        # Our limit is 2.0s or <2 events.
        # We have 2 events. Time might be short.
        # Let's mock time.time() or just pass cleanup_if_empty=False for safety in test.
        session_path = self.observer.stop_observation(cleanup_if_empty=False)
        
        # Verify recorder stopped
        mock_recorder.stop.assert_called_once()
        
        # 5. Verify Files
        self.assertTrue(os.path.exists(session_path))
        self.assertTrue(os.path.exists(os.path.join(session_path, "manifest.json")))
        
        # Verify Manifest content
        with open(os.path.join(session_path, "manifest.json"), 'r') as f:
            data = json.load(f)
            self.assertEqual(len(data['events']), 2)
            self.assertGreaterEqual(len(data['screenshots']), 1) # Should have at least 1 screenshot
            
        print(f"Session saved to: {session_path}")
        print(f"Events captured: {len(data['events'])}")
        print(f"Screenshots captured: {len(data['screenshots'])}")

        # 6. Test Analysis
        print("Testing Analysis...")
        result = self.observer.analyze_session(session_path)
        print(f"Analysis Result: {result}")
        
        self.assertIn("Mocked Intent", result)
        self.observer.llm_service.chat_with_vision.assert_called_once()

if __name__ == "__main__":
    unittest.main()
