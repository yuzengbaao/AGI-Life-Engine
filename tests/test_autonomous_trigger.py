import sys
import os
import asyncio
import logging
import time
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.evolution.impl import EvolutionController
from core.llm_client import LLMService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutonomousTriggerTest")

async def test_autonomous_trigger():
    logger.info("🧪 Testing Autonomous Optimization Trigger...")
    
    # Mock LLM Service to avoid real calls
    mock_llm = MagicMock(spec=LLMService)
    mock_llm.chat_completion.return_value = "print('Mock Optimization')" # Mock code gen
    
    # Initialize Controller
    controller = EvolutionController(llm_service=mock_llm)
    
    # Force the cooldown to be 0 for testing
    controller.AUTO_OPTIMIZATION_COOLDOWN = 0
    
    # Create a dummy context
    context = {"status": "idle", "energy": 100}
    
    # Mock Seed's intrinsic reward to be HIGH (> 3.0)
    # We need to patch the seed or just ensure the return value leads to high reward
    # The current implementation calculates reward based on uncertainty.
    # But wait, EvolutionController.step() calls self.seed.get_intrinsic_reward()
    
    # Let's mock the seed instance directly on the controller
    controller.seed = MagicMock()
    # Mock act to return 'create' index (which is 2 in ACTIONS list ["explore", "analyze", "create", "rest"]?)
    # Wait, ACTIONS = ["explore", "analyze", "create", "rest"] -> index 2
    controller.seed.act.return_value = 2 
    controller.seed.predict.return_value = (None, 0.9) # High uncertainty = 0.9
    controller.seed.get_intrinsic_reward.return_value = 3.5 # > 3.0 Threshold
    controller.seed.evaluate.return_value = 0.8 # Survival drive
    
    # Mock attempt_self_optimization to just log success
    async def mock_attempt(module_path, goal):
        logger.info(f"   [MOCK] Attempting optimization on: {module_path}")
        return {"ok": True}
    
    controller.attempt_self_optimization = mock_attempt
    
    # Mock value network and memory to avoid errors
    controller.values = MagicMock()
    controller.values.evaluate.return_value.score = 1.0
    controller.memory = MagicMock()
    # add_short_term should be async
    async def mock_add_short_term(context):
        return [0.1] * 64
    controller.memory.add_short_term = mock_add_short_term
    controller.world_model = MagicMock()
    controller.world_model.identify_bottlenecks.return_value = []
    
    # Run step
    logger.info("⏳ Stepping EvolutionController...")
    result = await controller.step(context)
    
    # Verify
    guidance = result.get("seed_guidance", {})
    logger.info(f"   Guidance: {guidance}")
    
    if guidance.get("intrinsic_curiosity", 0) > 3.0:
        logger.info("✅ Intrinsic Reward is high as expected.")
    else:
        logger.error("❌ Intrinsic Reward not high enough.")
        
    if guidance.get("suggested_action") == "create":
        logger.info("✅ Suggested Action is 'create'.")
    else:
        logger.error(f"❌ Suggested Action is {guidance.get('suggested_action')}.")

    logger.info("✅ Test Completed.")

if __name__ == "__main__":
    asyncio.run(test_autonomous_trigger())
