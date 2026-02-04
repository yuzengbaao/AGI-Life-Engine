import os
import sys
# Add project root to sys.path
sys.path.append(os.getcwd())

from core.research.lab import ShadowRunner

def test_full_context():
    print("Testing ShadowRunner with full_context=True...")
    runner = ShadowRunner(project_root=os.getcwd())
    
    # We want to test a file that has relative imports.
    # core/perception/processors/adapter.py has 'from .video import ...'
    target_file = "core/perception/processors/adapter.py"
    
    if not os.path.exists(target_file):
        print(f"Target file {target_file} not found, skipping specific test.")
        return

    with open(target_file, 'r', encoding='utf-8') as f:
        code = f.read()
        
    # We modify it slightly (comment update) to trigger shadow env creation
    modified_code = code + "\n# Verified by ShadowRunner"
    
    # Create shadow env with full_context=True
    shadow_path = runner.create_shadow_env({target_file: modified_code}, full_context=True)
    print(f"Shadow env created at: {shadow_path}")
    
    # Check if 'core' exists in shadow path
    if os.path.exists(os.path.join(shadow_path, "core")):
        print("SUCCESS: 'core' directory copied to shadow env.")
    else:
        print("FAILURE: 'core' directory NOT found in shadow env.")
        
    # Dry Run
    module_to_test = "core.perception.processors.adapter"
    ok, output = runner.dry_run(shadow_path, module_to_test)
    
    if ok:
        print("SUCCESS: Dry Run passed.")
    else:
        print("FAILURE: Dry Run failed.")
        print(output)
        
    runner.cleanup(shadow_path)

if __name__ == "__main__":
    test_full_context()
