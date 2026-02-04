
import sys
import os
import asyncio
import logging
import time

# Ensure project root is in path
sys.path.append(os.getcwd())

from core.evolution.impl import EvolutionController
from core.llm_client import LLMService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RealOptimizationTest")

async def run_real_optimization():
    logger.info("🚀 Starting Real Self-Optimization Test...")
    
    # 1. Create Dummy Module (The Target)
    target_path = "core/dummy_math_module.py"
    target_abs_path = os.path.abspath(target_path)
    
    # Original inefficient code
    original_code = """
import time

def fibonacci(n):
    # Very inefficient implementation
    if n <= 1:
        return n
    # Artificial delay to simulate inefficiency
    time.sleep(0.001) 
    return fibonacci(n-1) + fibonacci(n-2)

def get_status():
    return "Legacy Version"
"""
    os.makedirs(os.path.dirname(target_abs_path), exist_ok=True)
    with open(target_abs_path, "w", encoding="utf-8") as f:
        f.write(original_code)
    logger.info(f"✅ Created dummy target: {target_path}")

    try:
        # 2. Initialize Controller with REAL LLM
        # NOTE: This requires a working LLM configuration in .env
        logger.info("🔌 Initializing LLM Service...")
        llm = LLMService() 
        
        if llm.mock_mode:
            logger.warning("⚠️ LLM is in MOCK MODE. This test might fail or produce mock results.")
            
        controller = EvolutionController(llm_service=llm)
        logger.info("✅ EvolutionController Ready.")
        
        # 3. Trigger Self-Optimization
        # Goal: Optimize fibonacci and update status string
        goal = "Optimize the fibonacci function to be iterative (O(n)) to remove the time.sleep delay. Also update get_status to return 'Optimized Version'."
        
        logger.info("🎯 Triggering attempt_self_optimization...")
        result = await controller.attempt_self_optimization(target_path, goal)
        
        # 4. Verification
        logger.info(f"📊 Result: {result}")
        
        if result.get("ok"):
            # Check content
            with open(target_abs_path, "r", encoding="utf-8") as f:
                new_content = f.read()
            
            if "Optimized Version" in new_content:
                logger.info("✅ Content check PASSED: Status updated.")
            else:
                logger.error("❌ Content check FAILED: Status not updated.")
                
            if "time.sleep" not in new_content:
                logger.info("✅ Content check PASSED: Inefficiency removed.")
            else:
                logger.warning("⚠️ Content check WARNING: time.sleep might still be there.")
        else:
            logger.error("❌ Optimization failed.")

    finally:
        # Cleanup
        if os.path.exists(target_abs_path):
            os.remove(target_abs_path)
        # Cleanup backups
        for f in os.listdir(os.path.dirname(target_abs_path)):
            if f.startswith("dummy_math_module.py.bak_"):
                os.remove(os.path.join(os.path.dirname(target_abs_path), f))
        logger.info("🧹 Cleanup done.")

if __name__ == "__main__":
    asyncio.run(run_real_optimization())
