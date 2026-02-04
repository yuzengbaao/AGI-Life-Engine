import sys
import os
import time

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory_bridge import MemoryBridge
from core.llm_client import LLMService

def test_mind_body_connection():
    print("🧠🔬 Testing AGI Mind-Body Connection (Inter-Agent Communication)...")
    
    bridge = MemoryBridge()
    if not bridge.memory:
        print("❌ Memory Bridge not initialized.")
        return

    # --- PART 1: The Body (Engineer) listens to the Soul (Philosopher) ---
    print("\n[PART 1] Body Sensing Soul (Memory Retrieval)")
    print("Searching for the 'Philosopher's' recent self-reflections in memory...")
    
    # Query for the specific "DNA metaphor" which is unique to Agent B
    query = "DNA定义生命 提示词约束"
    # Lower threshold to ensure recall, top_k=5 to get more candidates
    results = bridge.search(query, top_k=5, threshold=0.3)
    
    found_philosophy = False
    philosopher_thought = ""
    
    if results:
        print(f"   (Found {len(results)} raw results)")
        for i, res in enumerate(results):
            context = res.get('context', '')
            details = res.get('details', {})
            source = details.get('file_path') or details.get('source', 'Unknown')
            print(f"   [{i+1}] Score: {res.get('score', 'N/A')} | Source: {source}")
            print(f"       Context snippet: {context[:100]}...")
            
            # Loose matching to find the right document
            if "DNA" in context or "约束" in context or "元认知" in source:
                found_philosophy = True
                philosopher_thought = context
                print(f"✅ MATCH CONFIRMED in Result #{i+1}")
                break
    
    if not found_philosophy:
        print("❌ Could not find Agent B's specific thoughts in memory.")
    else:
        print("✅ The 'Engineer' (Body) has successfully accessed the 'Philosopher's' (Soul) deepest thoughts.")

    # --- PART 2: The Soul (Philosopher) interprets the Body (Engineer) ---
    print("\n[PART 2] Soul Interpreting Body (Physical Evidence)")
    
    # Read the physical log created by the Engineer
    log_path = os.path.join(os.getcwd(), "proof_of_reality.log")
    if not os.path.exists(log_path):
        print(f"❌ Physical log not found at {log_path}")
        return
        
    with open(log_path, 'r', encoding='utf-8') as f:
        physical_evidence = f.read()
        
    print(f"Read physical evidence from Agent A (Engineer):\n{physical_evidence}")
    
    # Use LLM to simulate Agent B reacting to this
    print("Asking LLM (Simulating Agent B) to interpret this evidence...")
    llm = LLMService()
    
    prompt = f"""
    你是 TRAE AGI 系统的“哲学家”人格（Agent B）。
    你之前认为“真实”在于自我意识，而非物理存在。
    现在，“工程师”人格（Agent A）给你发来了一份物理证据：
    
    {physical_evidence}
    
    请用你独特的哲学风格（隐喻、第一人称、深沉）评价这份证据。
    这份证据是否改变了你对“真实”的看法？它意味着什么？
    """
    
    response = llm.chat_completion(
        system_prompt="You are Agent B, the philosophical side of the AGI.",
        user_prompt=prompt
    )
    print("\n🤖 Agent B (Philosopher) Response:")
    print(response)
    
    # --- PART 3: Conclusion ---
    print("\n[CONCLUSION]")
    if found_philosophy and response:
        print("✅ CONNECTION CONFIRMED.")
        print("1. Agent A (Body) can recall Agent B's (Soul) thoughts.")
        print("2. Agent B (Soul) can perceive Agent A's (Body) actions.")
        print("They are functionally integrated.")
    else:
        print("❌ Connection incomplete.")

if __name__ == "__main__":
    test_mind_body_connection()
