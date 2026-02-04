
import os
import sys
import shutil
import logging
from dataclasses import dataclass

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.research.lab import ShadowRunner
from core.system_tools import SystemTools

# Mock Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntegrationVerify")

@dataclass
class MockPatch:
    target_file: str
    new_content: str
    description: str

def verify_integration():
    print("\n=== Simulating Phase 3.2 ShadowRunner Integration ===")
    
    # 1. Initialize Components
    project_root = os.getcwd()
    shadow_runner = ShadowRunner(project_root=project_root, sandbox_base="data/sandbox_test/integration_shadow")
    system_tools = SystemTools()
    
    print("✓ Components Initialized")
    
    # ==========================================
    # Scenario A: Bad Patch (Syntax Error) -> Diagnosis
    # ==========================================
    print("\n--- Scenario A: Handling Bad Patch (Syntax Error) ---")
    bad_patch = MockPatch(
        target_file="core/broken_module.py",
        new_content="def broken_func():\n    print('Missing closing parenthesis'", # Syntax Error
        description="A broken patch"
    )
    
    # 2. Create Shadow Env
    modified_files = {bad_patch.target_file: bad_patch.new_content}
    shadow_path = shadow_runner.create_shadow_env(modified_files)
    print(f"🌑 Shadow Env Created: {shadow_path}")
    
    try:
        # 3. Derive Module Name
        module_name = "core.broken_module"
        
        # 4. Dry Run
        print(f"🧪 Dry Run: {module_name}")
        success, output = shadow_runner.dry_run(shadow_path, module_name)
        
        if not success:
            print("❌ Dry Run Failed (Expected)")
            # 5. Analyze Traceback
            diagnosis = system_tools.analyze_traceback(output)
            print(f"🩺 Diagnosis Result:\n{diagnosis}")
            
            # Assertions
            if "SyntaxError" in output or "SyntaxError" in diagnosis:
                print("✅ Correctly identified SyntaxError")
            else:
                print("⚠️ Failed to identify SyntaxError")
        else:
            print("⚠️ Unexpected Success for bad patch")
            
    finally:
        # 6. Cleanup
        shadow_runner.cleanup(shadow_path)
        print("🗑️ Shadow Env Cleaned")

    # ==========================================
    # Scenario B: Good Patch -> Success
    # ==========================================
    print("\n--- Scenario B: Handling Good Patch ---")
    good_patch = MockPatch(
        target_file="core/working_module.py",
        new_content="def working_func():\n    return 'It works!'",
        description="A working patch"
    )
    
    modified_files = {good_patch.target_file: good_patch.new_content}
    shadow_path = shadow_runner.create_shadow_env(modified_files)
    
    try:
        module_name = "core.working_module"
        success, output = shadow_runner.dry_run(shadow_path, module_name)
        
        if success:
            print("✅ Dry Run Passed")
            if "It works!" in output: # Our dry run script doesn't run the func, just imports.
                pass 
            print("Output:", output)
        else:
            print(f"❌ Unexpected Failure:\n{output}")
            
    finally:
        shadow_runner.cleanup(shadow_path)
        print("🗑️ Shadow Env Cleaned")

if __name__ == "__main__":
    verify_integration()
