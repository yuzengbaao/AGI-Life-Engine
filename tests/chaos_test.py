import subprocess
import time
import os
import signal
import sys
import psutil

def run_chaos_test():
    print("🔥 STARTING CHAOS ENGINEERING TEST 🔥")
    print("=" * 50)

    # 1. Start the Engine (Main Process)
    print("\n[Step 1] Launching AGI Engine...")
    process = subprocess.Popen([sys.executable, "main.py"], 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE,
                             text=True)
    
    print(f"Process started with PID: {process.pid}")
    time.sleep(5)  # Let it run for a bit
    
    # 2. Simulate Power Failure (Kill Process)
    print("\n[Step 2] ⚡ SIMULATING SUDDEN POWER FAILURE (KILL -9) ⚡")
    try:
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
        print(">> Process Killed Successfully.")
    except psutil.NoSuchProcess:
        print(">> Process already died?")
    
    # Verify it's dead
    if process.poll() is None:
        print(">> Confirming kill...")
        process.kill()
        
    time.sleep(2)
    
    # 3. Recovery Test
    print("\n[Step 3] 🚑 ATTEMPTING SYSTEM RECOVERY...")
    print("Restarting Engine to check if it resumes from checkpoint...")
    
    # Start again
    process_recovery = subprocess.Popen([sys.executable, "main.py"], 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE,
                                      text=True)
    
    # Read output for a few seconds to check for "Restored state"
    start_time = time.time()
    restored = False
    
    print(">> Monitoring logs for recovery signature...")
    while time.time() - start_time < 10:
        line = process_recovery.stdout.readline()
        if line:
            print(f"   [LOG] {line.strip()}")
            if "Restored state" in line:
                restored = True
                print("\n✅ SUCCESS: SYSTEM RESTORED FROM CHECKPOINT!")
                break
    
    if not restored:
        print("\n❌ FAILURE: System did not report state restoration.")
    
    # Cleanup
    print("\n[Step 4] Cleaning up...")
    process_recovery.kill()
    print("Test Complete.")

if __name__ == "__main__":
    run_chaos_test()
