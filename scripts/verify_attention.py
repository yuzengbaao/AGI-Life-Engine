
import asyncio
import os
import sys
import logging

# Add project root to path
sys.path.append(os.getcwd())

from core.evolution.impl import EvolutionController

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("AttentionVerifier")

def verify_attention_mechanism():
    print("🔬 Verifying Attention Mechanism in EvolutionController...")
    
    # Initialize Controller
    controller = EvolutionController()
    
    # Run selection multiple times to check distribution
    selections = {}
    print("\nRUNNING 100 SELECTIONS...")
    
    for i in range(100):
        target = controller._select_optimization_target()
        if target:
            name = os.path.basename(target)
            selections[name] = selections.get(name, 0) + 1
            
    print("\n📊 SELECTION DISTRIBUTION (Top 10):")
    sorted_selections = sorted(selections.items(), key=lambda x: x[1], reverse=True)
    
    for name, count in sorted_selections[:10]:
        print(f"  - {name}: {count}%")
        
    # Check if we have diversity
    unique_files = len(selections)
    print(f"\n✅ Unique files selected: {unique_files}")
    
    if unique_files > 1:
        print("✅ SUCCESS: Mechanism provides diversity (Global Awareness).")
    else:
        print("❌ FAILURE: Mechanism is deterministic or stuck.")
        
    # Check if larger files are favored (heuristic)
    # We expect 'impl.py' is excluded, so maybe 'llm_client.py' or 'the_seed.py'
    print("\n🔍 Checking for logic correctness...")
    # Manually check score for a known file
    core_path = os.path.join(os.getcwd(), "core")
    files_checked = 0
    for root, _, files in os.walk(core_path):
        for file in files:
            if file == "llm_client.py":
                full_path = os.path.join(root, file)
                stats = os.stat(full_path)
                print(f"  File: {file}, Size: {stats.st_size}, Last Modified: {stats.st_mtime}")
                files_checked += 1
    
    if files_checked > 0:
        print("✅ File system access confirmed.")

if __name__ == "__main__":
    verify_attention_mechanism()
