
import os
import glob
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.memory.neural_memory import BiologicalMemorySystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Consolidation")

def load_insights():
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "insights")
    insight_files = glob.glob(os.path.join(base_dir, "insight_*.md"))
    
    insights = []
    for fpath in insight_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple extraction: treat the whole file as one memory block
                insights.append({
                    "id": os.path.basename(fpath),
                    "content": content
                })
        except Exception as e:
            logger.error(f"Error reading {fpath}: {e}")
            
    return insights

def main():
    print("==================================================")
    print("   AGI NEURAL CONSOLIDATION PROCESS (SLEEP MODE)   ")
    print("==================================================")
    
    # 1. Load Insights
    logger.info("Scanning short-term memory (Insights)...")
    insights = load_insights()
    logger.info(f"Found {len(insights)} insights to consolidate.")
    
    if not insights:
        logger.warning("No insights found. Aborting.")
        return

    # 2. Initialize Brain
    logger.info("Waking up Biological Memory System...")
    brain = BiologicalMemorySystem()
    
    # 3. Internalize (Train)
    logger.info("Starting internalization process (Training AutoEncoder)...")
    print("\n[Neural Activity Monitor]")
    stats = brain.internalize(insights, epochs=100)
    
    print(f"\nTraining Complete.")
    print(f"Initial Loss: {stats['initial_loss']:.6f}")
    print(f"Final Loss:   {stats['final_loss']:.6f}")
    print(f" plasticity:  {stats['improvement']:.6f} (Loss Reduction)")
    
    if stats['final_loss'] < 0.1:
        print("\nStatus: MEMORY CONSOLIDATED SUCCESSFULLY")
    else:
        print("\nStatus: PARTIAL CONSOLIDATION")

    # 4. Verification (Recall)
    print("\n==================================================")
    print("   VERIFYING NEURAL PATHWAYS (RECALL TEST)   ")
    print("==================================================")
    
    test_query = "optimization and efficiency"
    print(f"Stimulus: '{test_query}'")
    
    results = brain.recall(test_query)
    
    if results:
        print(f"Neural Response (Top activated memories):")
        for res in results:
            print(f" - {res['id']} (Activation: {res['score']:.4f})")
    else:
        print("No significant neural activation detected.")

    print("\n==================================================")
    print("   SECOND BRAIN STATUS: ONLINE & ACTIVE   ")
    print("==================================================")

if __name__ == "__main__":
    main()
