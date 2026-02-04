import time
import os
import json
import pyautogui
from core.macro_system import MacroRecorder, MacroPlayer
from core.desktop_automation import DesktopController

# Define paths
WORK_LOG = r"D:\TRAE_PROJECT\AGI\WorkLog"
TEST_FILE = os.path.join(WORK_LOG, "macro_test_cmd.txt")
MACRO_FILE = os.path.join(WORK_LOG, "cmd_macro.json")

def clean_env():
    """Ensure test environment is clean."""
    if os.path.exists(TEST_FILE):
        try:
            os.remove(TEST_FILE)
            print(f"Cleaned up {TEST_FILE}")
        except Exception as e:
            print(f"Warning: Could not clean {TEST_FILE}: {e}")

def simulate_user_actions():
    """Simulates a human user operating CMD."""
    print(">>> SIMULATION: Starting User Actions (CMD)...")
    
    # Ensure English Input for the simulation itself!
    dc = DesktopController()
    dc.ensure_english_input()
    time.sleep(0.5)
    
    # 1. Open Run dialog
    pyautogui.hotkey('win', 'r')
    time.sleep(1.0)
    
    # 2. Open CMD
    pyautogui.write('cmd')
    pyautogui.press('enter')
    time.sleep(2.0) # Wait for CMD to open
    
    # Ensure English again in CMD window? 
    # Usually inherits, but good practice if we want to be safe.
    dc.ensure_english_input()
    time.sleep(0.5)
    
    # 3. Type command to create file
    # We use echo to create a file
    cmd = f'echo SUCCESS > {TEST_FILE}'
    pyautogui.write(cmd, interval=0.05)
    time.sleep(1.0)
    pyautogui.press('enter')
    time.sleep(1.0)
    
    # 4. Exit CMD
    pyautogui.write('exit')
    pyautogui.press('enter')
    
    print(">>> SIMULATION: User Actions Complete.")

def run_integration_test():
    # Ensure directories exist
    os.makedirs(WORK_LOG, exist_ok=True)
    clean_env()
    
    recorder = MacroRecorder()
    
    print("\n[Phase 1] Recording...")
    recorder.start()
    
    # Run simulation while recording
    time.sleep(1)
    
    try:
        simulate_user_actions()
    except Exception as e:
        print(f"Simulation failed: {e}")
        recorder.stop()
        return
        
    # Stop recording
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
    # Clean up the file created by the "User"
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)
        print("Test file removed for playback verification.")
    else:
        print("Warning: Simulation did not create the file? Playback might fail too.")
    
    player = MacroPlayer()
    loaded_events = player.load_macro(MACRO_FILE)
    
    # Playback
    # Use normal speed
    player.play(loaded_events, speed_factor=1.0, max_delay=5.0)
    
    # Verification
    time.sleep(2) # Give OS time to flush file
    if os.path.exists(TEST_FILE):
        print("\nSUCCESS: Playback created the file!")
        with open(TEST_FILE, 'r') as f:
            content = f.read()
            print(f"File Content: '{content.strip()}'")
            if "SUCCESS" in content:
                print("Content verification passed.")
            else:
                print("Content verification FAILED.")
    else:
        print("\nFAILURE: Playback did not create the file.")

if __name__ == "__main__":
    run_integration_test()
