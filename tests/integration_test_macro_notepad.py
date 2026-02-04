import time
import os
import json
import threading
import pyautogui
from core.macro_system import MacroRecorder, MacroPlayer

# Define paths
WORK_LOG = r"D:\TRAE_PROJECT\AGI\WorkLog"
TEST_FILE = os.path.join(WORK_LOG, "test_macro_result.txt")
MACRO_FILE = os.path.join(WORK_LOG, "notepad_macro.json")

def clean_env():
    """Ensure test environment is clean."""
    if os.path.exists(TEST_FILE):
        try:
            os.remove(TEST_FILE)
            print(f"Cleaned up {TEST_FILE}")
        except Exception as e:
            print(f"Warning: Could not clean {TEST_FILE}: {e}")

def simulate_user_actions():
    """Simulates a human user operating Notepad."""
    print(">>> SIMULATION: Starting User Actions...")
    
    # 1. Open Run dialog
    pyautogui.hotkey('win', 'r')
    time.sleep(2.0)
    
    # 2. Open Notepad
    pyautogui.write('notepad')
    pyautogui.press('enter')
    time.sleep(5.0) # Wait for Notepad to open
    
    # 3. Type content
    pyautogui.write('Hello. This is a recorded macro test.', interval=0.1)
    time.sleep(2.0)
    
    # 4. Save file
    pyautogui.hotkey('ctrl', 's')
    time.sleep(4.0) # Wait for Save dialog
    
    # 5. Type path
    # Use raw string for path, but pyautogui write might need escaping if not careful.
    # We'll just type it out.
    pyautogui.write(TEST_FILE)
    time.sleep(2.0)
    
    # 6. Confirm Save
    pyautogui.press('enter')
    time.sleep(2.0)
    
    # Handle "Confirm Save As" if file exists (though we cleaned it)
    # Just in case, pressing 'y' or 'enter' might help if a dialog popped up, 
    # but 'enter' usually replaces.
    # Let's assume clean env.
    
    # 7. Close Notepad
    pyautogui.hotkey('alt', 'f4')
    
    print(">>> SIMULATION: User Actions Complete.")

def run_integration_test():
    # Ensure directories exist
    os.makedirs(WORK_LOG, exist_ok=True)
    clean_env()
    
    recorder = MacroRecorder()
    
    print("\n[Phase 1] Recording...")
    recorder.start()
    
    # Run simulation while recording
    # We sleep a bit to let recorder initialize
    time.sleep(1)
    
    try:
        simulate_user_actions()
    except Exception as e:
        print(f"Simulation failed: {e}")
        recorder.stop()
        return
        
    # Stop recording
    # Wait a bit to capture the final close
    time.sleep(1)
    events = recorder.stop()
    
    print(f"Recorded {len(events)} events.")
    
    # Save macro
    recorder.save_macro(MACRO_FILE)
    
    # Verify JSON content
    with open(MACRO_FILE, 'r') as f:
        data = json.load(f)
        print(f"Saved Macro JSON (first 3 events): {json.dumps(data[:3], indent=2)}")
    
    print("\n[Phase 2] Playback Verification...")
    # Clean up the file created by the "User" so we can prove the "Macro" creates it
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)
        print("Test file removed for playback verification.")
    else:
        print("Warning: Simulation did not create the file? Playback might fail too.")
    
    player = MacroPlayer()
    loaded_events = player.load_macro(MACRO_FILE)
    
    # Playback
    # Use normal speed and allow longer delays to ensure UI catches up
    player.play(loaded_events, speed_factor=1.0, max_delay=10.0)
    
    # Verification
    time.sleep(2) # Give OS time to flush file
    if os.path.exists(TEST_FILE):
        print("\nSUCCESS: Playback created the file!")
        with open(TEST_FILE, 'r') as f:
            content = f.read()
            print(f"File Content: '{content}'")
            if "recorded macro test" in content:
                print("Content verification passed.")
            else:
                print("Content verification FAILED.")
    else:
        print("\nFAILURE: Playback did not create the file.")

if __name__ == "__main__":
    run_integration_test()
