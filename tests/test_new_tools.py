import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.system_tools import SystemTools

def test_tools():
    print(f"Loaded SystemTools from: {sys.modules['core.system_tools'].__file__}")
    print("--- Initializing SystemTools ---")
    tools = SystemTools()
    
    print("\n[Test 1] inspect_project_structure (core/agents)")
    tree = tools.inspect_project_structure("core/agents", depth=1)
    print(tree)
    
    print("\n[Test 2] query_codebase (def:SystemTools)")
    # This might take a moment to build the index
    symbol_def = tools.query_codebase("def:SystemTools")
    print(symbol_def)
    
    print("\n[Test 3] search_knowledge (query='error handling')")
    # This might return "No relevant memories" or error if dependencies missing, which is expected
    memory = tools.search_knowledge("error handling")
    print(memory)

if __name__ == "__main__":
    test_tools()
