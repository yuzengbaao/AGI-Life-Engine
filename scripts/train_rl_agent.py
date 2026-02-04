
import sys
import os
import random
import time
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolving_agent import EvolvingAgent

# Configure simplified logging for training
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("RLTrainer")
logger.setLevel(logging.INFO)

def train_agent(iterations=1000):
    logger.info(f"🚀 Starting RL Training for {iterations} iterations...")
    
    agent = EvolvingAgent()
    
    # Disable main agent logging to avoid spamming the console, keep only trainer logs
    agent_logger = logging.getLogger("EvolvingAgent")
    agent_logger.setLevel(logging.ERROR)
    
    # States counters
    states_visited = {}
    
    start_time = time.time()
    
    for i in range(iterations):
        if i % 100 == 0:
             print(f"Progress: {i}/{iterations}")

        # 1. Inject Random Environment State (Simulate diverse operating conditions)
        # We want to cover all combinations of Success Rate (Low/Med/High) and Latency (Fast/Med/Slow)
        
        # Randomize task count (10-100)
        task_count = random.randint(10, 100)
        agent.metrics.task_count = task_count
        
        # Randomize success rate to cover L/M/H
        # 33% chance for each range
        rand_sr = random.random()
        if rand_sr < 0.33:
            # Low (< 0.6)
            success_rate = random.uniform(0.1, 0.59)
        elif rand_sr < 0.66:
            # Medium (0.6 - 0.9)
            success_rate = random.uniform(0.6, 0.89)
        else:
            # High (> 0.9)
            success_rate = random.uniform(0.91, 1.0)
            
        agent.metrics.success_count = int(task_count * success_rate)
        
        # Randomize latency to cover F/M/S
        # 33% chance for each
        rand_lat = random.random()
        if rand_lat < 0.33:
            # Fast (< 1.0)
            avg_latency = random.uniform(0.1, 0.9)
        elif rand_lat < 0.66:
            # Medium (1.0 - 3.0)
            avg_latency = random.uniform(1.1, 2.9)
        else:
            # Slow (> 3.0)
            avg_latency = random.uniform(3.1, 5.0)
            
        agent.metrics.total_latency = avg_latency * task_count
        
        # Randomize error count (occasionally introduce errors)
        if random.random() < 0.2:
            agent.metrics.error_count = random.randint(1, 5)
        else:
            agent.metrics.error_count = 0
            
        # 2. Run Evolution Step
        # This will trigger: Get State -> Choose Action -> Execute -> Learn -> Save
        report = agent.evolve()
        
        # Track visited states
        if "rl_state" in report:
            state = report["rl_state"]
            states_visited[state] = states_visited.get(state, 0) + 1
            
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info("\n" + "="*40)
    logger.info(f"✅ Training Completed in {duration:.2f}s")
    logger.info(f"⚡ Speed: {iterations/duration:.1f} iter/s")
    logger.info("="*40)
    logger.info("📊 State Coverage:")
    for state, count in sorted(states_visited.items()):
        logger.info(f"   - {state}: {count} visits")
        
    logger.info(f"\n💾 Q-Table saved to: {os.path.abspath('rl_q_table.json')}")

if __name__ == "__main__":
    # Ensure scripts directory exists
    os.makedirs("scripts", exist_ok=True)
    
    train_agent(1000)
