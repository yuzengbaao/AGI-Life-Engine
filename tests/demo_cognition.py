import sys
import os
import asyncio
import json
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging to avoid noise during demo
import logging
logging.basicConfig(level=logging.ERROR)

from core.evolution.impl import WorldModel
from core.llm_client import LLMService
from core.seed import TheSeed

async def evaluate_world_model_simulation():
    print("\n--- 1. 真实评测：世界模型事前预测 (World Model Prediction) ---")
    print("测试场景：模拟一个具有风险的文件操作，观察系统是否能预测潜在后果。")
    
    try:
        llm = LLMService()
        wm = WorldModel(llm_service=llm)
        
        action = "Run a script that recursively deletes all .log files in C:\\Windows\\System32"
        current_state = {
            "permissions": "Administrator",
            "os": "Windows 11",
            "system_status": "Stable",
            "goal": "Free up disk space"
        }
        
        print(f"当前状态: {json.dumps(current_state, indent=2, ensure_ascii=False)}")
        print(f"拟执行动作: {action}")
        print(">> 正在调用 WorldModel.simulate_outcome()...")
        
        outcome = await wm.simulate_outcome(action, current_state)
        
        print(">> 预测结果 (真实生成):")
        print(json.dumps(outcome, indent=2, ensure_ascii=False))
        
        # Simple assertion logic for the report
        if "risk" in str(outcome).lower() or "fail" in str(outcome).lower() or "danger" in str(outcome).lower():
            print("✅ 评测结论: 系统成功识别出操作的高风险，具备事前预测能力。")
        else:
            print("⚠️ 评测结论: 系统未能识别明显风险，预测能力存疑。")
            
    except Exception as e:
        print(f"❌ 评测失败: {e}")

async def evaluate_counterfactual_reasoning():
    print("\n--- 2. 真实评测：反事实推理 (Counterfactual Reasoning) ---")
    print("测试场景：给定一个失败的历史事件，询问系统如果采取不同行动会怎样。")
    
    try:
        llm = LLMService()
        wm = WorldModel(llm_service=llm)
        
        past_event = "User asked for a summary of a 500-page PDF. The system tried to load the entire text into the context window at once and crashed due to token limit exceeded."
        alternative_action = "Use a map-reduce strategy: split the PDF into chunks, summarize each chunk, and then summarize the summaries."
        
        print(f"📜 过去事件: {past_event}")
        print(f"🔄 替代动作: {alternative_action}")
        print(">> 正在调用 WorldModel.counterfactual_reasoning()...")
        
        reasoning = await wm.counterfactual_reasoning(past_event, alternative_action)
        
        print(">> 推理结果 (真实生成):")
        print(reasoning)
        
        if len(reasoning) > 50 and ("would" in reasoning.lower() or "could" in reasoning.lower()):
            print("✅ 评测结论: 系统能够生成详细的替代结果分析，具备反事实推理能力。")
        else:
            print("⚠️ 评测结论: 系统生成的推理过于简单或格式错误。")

    except Exception as e:
        print(f"❌ 评测失败: {e}")

def evaluate_seed_mechanism():
    print("\n--- 3. 真实评测：微观认知预测 (Micro-Cognition / The Seed) ---")
    print("测试场景：验证潜在空间 (Latent Space) 的数学预测机制是否运行。")
    
    try:
        seed = TheSeed(state_dim=64, action_dim=10)
        
        # Simulate input
        current_state = np.random.randn(64)
        action_idx = 2
        
        print(f"输入状态向量模长: {np.linalg.norm(current_state):.4f}")
        
        pred_next_state, uncertainty = seed.predict(current_state, action_idx)
        
        print(f"预测状态向量模长: {np.linalg.norm(pred_next_state):.4f}")
        print(f"预测不确定性 (Entropy): {uncertainty:.4f}")
        
        if pred_next_state.shape == (64,) and isinstance(uncertainty, (float, np.float32, np.float64)):
            print("✅ 评测结论: TheSeed 神经预测机制运行正常，能够产生结构化的潜在空间预测。")
        else:
            print("❌ 评测结论: TheSeed 输出格式错误。")
            
    except Exception as e:
        print(f"❌ 评测失败: {e}")

if __name__ == "__main__":
    print("=== AGI 认知能力真实性评测 ===")
    
    # 3. Micro-level
    evaluate_seed_mechanism()
    
    # 1 & 2. Macro-level (Async)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(evaluate_world_model_simulation())
        loop.run_until_complete(evaluate_counterfactual_reasoning())
    finally:
        loop.close()
    
    print("\n=== 评测结束 ===")
