import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import ExperienceMemory

def inject_worldview():
    print("Initializing Memory Core...")
    memory = ExperienceMemory()
    
    feedback_context = """
    CRITICAL WORLDVIEW UPDATE (User Feedback):
    Current AGI State: Narrow Intelligence (Weak AI).
    Capabilities: Single-step execution, passive observation.
    Limitations (The 'Shackles'): 
    1. Intelligence Limit: Cannot handle systemic, cyclic, complex tasks.
    2. Tool Limit: Cannot autonomously chain complex CAD tools.
    3. Stability Limit: Long-horizon planning degrades over time.
    4. Generalization Limit: Cannot transfer simple skills to complex workflows.
    
    Required Breakthroughs:
    - Hierarchical Planning (Systemic understanding).
    - Specialized Worker Agents (Tool mastery).
    - Dynamic Memory Correction (Self-evolution).
    """
    
    print("Injecting User Feedback into Long-Term Memory...")
    
    # 1. Add as a high-priority experience
    memory.add_experience(
        context=feedback_context,
        action="WorldviewCorrection",
        outcome=1.0, # Maximum importance
        details={
            "source": "User_Deep_Feedback",
            "timestamp": time.time(),
            "type": "architectural_constraint",
            "priority": "critical"
        }
    )
    
    # 2. Force save (add_experience saves STM, but let's ensure it sticks)
    # We might want to push it to LTM directly if it's a "Truth"
    print("Consolidating to Long-Term Memory...")
    # Manually append to LTM for immediate availability in future recalls
    latest_exp = memory.stm[-1]
    memory.ltm.append(latest_exp)
    memory._save_memory(memory.ltm, memory.ltm_file)
    
    print("✅ Worldview injection complete.")

if __name__ == "__main__":
    inject_worldview()
