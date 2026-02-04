import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import ExperienceMemory

def verify():
    print("Verifying Knowledge Recall...")
    memory = ExperienceMemory()
    
    # Query about a specific ingested document
    query = "AGI构建的第一性原理"
    print(f"Query: {query}")
    
    results = memory.recall_relevant(query)
    
    if results:
        print(f"Found {len(results)} relevant memories.")
        top_result = results[0]
        print(f"Top Result Context: {top_result['context'][:200]}...")
        print(f"Source: {top_result['details'].get('file_path')}")
        print("✅ Verification SUCCESS: Knowledge successfully retrieved.")
    else:
        print("❌ Verification FAILED: No relevant memory found.")

if __name__ == "__main__":
    verify()
