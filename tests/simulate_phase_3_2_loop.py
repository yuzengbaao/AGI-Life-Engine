
import sys
import os
import asyncio
import logging
import shutil
import time

# Ensure project root is in path
sys.path.append(os.getcwd())

from core.evolution.impl import EvolutionController, SandboxCompiler
from core.research.lab import ShadowRunner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SimulatePhase3.2")

async def simulate_loop():
    logger.info("🚀 Starting Phase 3.2 Closed Loop Simulation...")
    
    # 1. Initialize Controller
    controller = EvolutionController()
    logger.info("✅ EvolutionController Initialized.")
    
    # Verify ShadowRunner injection
    if not isinstance(controller.runtime.shadow_runner, ShadowRunner):
        logger.error("❌ ShadowRunner NOT injected into SandboxCompiler!")
        return
    logger.info("✅ ShadowRunner correctly injected.")

    # 2. Setup Dummy Target Module (in core/dummy_shadow_test.py)
    target_rel_path = "core/dummy_shadow_test.py"
    target_abs_path = os.path.abspath(target_rel_path)
    
    # Create the original file
    original_code = """
def hello():
    return "Original"
"""
    os.makedirs(os.path.dirname(target_abs_path), exist_ok=True)
    with open(target_abs_path, "w", encoding="utf-8") as f:
        f.write(original_code)
    logger.info(f"✅ Created dummy target file: {target_rel_path}")

    try:
        # 3. Define Test Case for Shadow Verification
        # We want to change "Original" to "Shadow" and verify it.
        shadow_code = """
def hello():
    return "Shadow"
"""
        test_script = """
import sys
import os
# Try to import the module
import core.dummy_shadow_test as target

print(f"Loaded module: {target}")
print(f"Module file: {target.__file__}")

val = target.hello()
print(f"Value: {val}")

if val != "Shadow":
    print("FAILURE: Value is not Shadow")
    sys.exit(1)
else:
    print("SUCCESS: Value is Shadow")
"""
        
        test_cases = [
            {
                "module_path": target_rel_path,
                "test_code": test_script
            }
        ]
        
        # 4. Run Verify in Sandbox
        logger.info("🧪 Running verify_in_sandbox...")
        verified = await controller.runtime.verify_in_sandbox(shadow_code, test_cases)
        
        if verified:
            logger.info("✅ verify_in_sandbox PASSED.")
        else:
            logger.error("❌ verify_in_sandbox FAILED.")
            # If failed, it might be the PYTHONPATH issue.
            if controller.runtime.last_verification:
                logger.error(f"Diagnosis: {controller.runtime.last_verification}")

        # 5. Test Failure Diagnosis (Bad Code)
        logger.info("🧪 Running verify_in_sandbox with BAD CODE...")
        bad_code = """
def hello():
    return "Shadow"
    syntax error here
"""
        verified_bad = await controller.runtime.verify_in_sandbox(bad_code, test_cases)
        if not verified_bad:
            logger.info("✅ verify_in_sandbox correctly rejected bad code.")
            diag = controller.runtime.last_verification.get("diagnosis")
            logger.info(f"   Diagnosis: {diag}")
        else:
            logger.error("❌ verify_in_sandbox improperly PASSED bad code.")

        # 6. Test Hot Swap (Rollback)
        logger.info("🔄 Testing Hot Swap (Rollback Capability)...")
        # We will try to swap with valid code first
        success = await controller.runtime.hot_swap_module(target_rel_path, shadow_code)
        if success:
            with open(target_abs_path, "r", encoding="utf-8") as f:
                content = f.read()
            if "Shadow" in content:
                logger.info("✅ Hot Swap successful (Content updated).")
                # Check backup
                backup_files = [f for f in os.listdir(os.path.dirname(target_abs_path)) if f.startswith("dummy_shadow_test.py.bak_")]
                if backup_files:
                    logger.info(f"✅ Backup file created: {backup_files[0]}")
                else:
                    logger.error("❌ No backup file found!")
            else:
                logger.error("❌ Hot Swap reported success but content mismatch.")
        else:
            logger.error("❌ Hot Swap failed.")

    finally:
        # Cleanup
        if os.path.exists(target_abs_path):
            os.remove(target_abs_path)
        # Cleanup backups
        for f in os.listdir(os.path.dirname(target_abs_path)):
            if f.startswith("dummy_shadow_test.py.bak_"):
                os.remove(os.path.join(os.path.dirname(target_abs_path), f))
        # Remove compiled pyc
        pyc = target_abs_path + "c" # .pyc not exactly like this usually but __pycache__
        # Just simple cleanup
        logger.info("🧹 Cleanup done.")

if __name__ == "__main__":
    asyncio.run(simulate_loop())
