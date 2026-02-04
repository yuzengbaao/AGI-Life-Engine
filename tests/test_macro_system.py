import unittest
import time
import os
import threading
from core.macro_system import MacroRecorder, MacroPlayer, ActionEvent
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key

class TestMacroSystem(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_macro.json"
        self.mouse = MouseController()
        self.keyboard = KeyboardController()

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_recording_and_playback(self):
        recorder = MacroRecorder()
        
        # Start recording in a separate thread or just start it (listeners are threads)
        recorder.start()
        time.sleep(1) # Wait for listeners to be ready

        # Simulate actions
        # Move mouse slightly to avoid clicking on something important
        current_pos = self.mouse.position
        # We don't record move, only click
        
        # Click
        self.mouse.click(Button.left)
        time.sleep(0.1)
        
        # Type
        self.keyboard.press('a')
        self.keyboard.release('a')
        time.sleep(0.1)

        # Stop recording
        events = recorder.stop()
        
        # Verify events were captured
        # Note: In some CI/headless environments this might fail if no display, 
        # but on a local Windows machine it should work.
        print(f"Captured {len(events)} events")
        # We expect at least 1 click and 1 key press
        # Note: Listener might capture press and release, but we only record press in _on_press
        
        self.assertTrue(len(events) >= 2, "Should capture at least click and key press")
        
        # Save
        recorder.save_macro(self.test_file)
        self.assertTrue(os.path.exists(self.test_file))

        # Playback
        player = MacroPlayer()
        loaded_events = player.load_macro(self.test_file)
        self.assertEqual(len(loaded_events), len(events))
        
        # We won't verify physical side effects of playback here as it might interfere with the user
        # But we can verify no crash
        player.play(loaded_events, speed_factor=10.0)

if __name__ == '__main__':
    unittest.main()
