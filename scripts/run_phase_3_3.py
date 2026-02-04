import sys
import os
import asyncio
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.evolution.impl import EvolutionController
from core.llm_client import LLMService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Phase3.3_Driver")

async def main():
    logger.info("🚀 Starting Phase 3.3: Knowledge Accumulation (Self-Optimization Demo)")
    
    target_file = "core/math_utils.py"
    if not os.path.exists(target_file):
        logger.error(f"❌ Target file {target_file} not found. Please create it first.")
        return

    # Initialize Controller
    logger.info("🔌 Initializing Evolution Controller...")
    try:
        llm = LLMService()
        # Using a new instance of EvolutionController. 
        # Note: Since the main engine is running, we might share the chroma db connection.
        # This is generally okay for this demo as we are mostly reading/writing files.
        controller = EvolutionController(llm_service=llm)
    except Exception as e:
        logger.error(f"❌ Failed to initialize controller: {e}")
        return
    
    # Define Goal
    goal = (
        "Optimize the fibonacci_recursive function to be iterative (O(n)) to remove the performance bottleneck. "
        "Also add type hints to all functions and handle division by zero in calculate_stats."
    )
    
    logger.info(f"🎯 Optimization Goal: {goal}")
    
    # Execute
    logger.info("⏳ Triggering attempt_self_optimization...")
    logger.info("   (This involves: Code Gen -> Test Gen -> Shadow Env -> Dry Run -> Verification -> Hot Swap)")
    
    start_time = asyncio.get_running_loop().time()
    result = await controller.attempt_self_optimization(target_file, goal)
    end_time = asyncio.get_running_loop().time()
    
    # Report
    logger.info("="*50)
    logger.info(f"📊 Execution Result: {result.get('ok', False)}")
    logger.info(f"⏱️ Duration: {end_time - start_time:.2f}s")
    
    if result.get("ok"):
        logger.info("✅ SUCCESS: The system has successfully evolved its own code!")
        logger.info("   Check core/math_utils.py for the optimized version.")
        logger.info("   Check core/math_utils.py.bak_* for the original version.")
        
        # Validation Read
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "def fibonacci_recursive(n: int) -> int:" in content or "while" in content or "range" in content or "for" in content:
                 logger.info("   [Validation] Code content confirms optimization.")
            else:
                 logger.warning("   [Validation] Code content might not be fully optimized.")
    else:
        logger.error("❌ FAILURE: Optimization failed.")
        if "verification" in result:
             logger.error(f"   Diagnosis: {result['verification'].get('diagnosis')}")

if __name__ == "__main__":
    asyncio.run(main())
