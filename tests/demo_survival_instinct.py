import sys
import os
import numpy as np
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.seed import TheSeed, Experience

def encode_log_feature(cpu_usage, error_rate, io_latency):
    """
    将系统状态编码为简单的特征向量
    Feature: [CPU, ErrorRate, IO, Bias]
    """
    # 归一化处理
    return np.array([cpu_usage/100.0, error_rate/10.0, io_latency/1000.0, 1.0])

def demo_survival():
    print("🛡️ 初始化系统生存本能 (Survival Instinct)...")
    seed = TheSeed(state_dim=4, action_dim=2) # Action: 0=Ignore, 1=Alert
    
    print("\n--- 阶段 1: 遭受痛苦 (Learning from Crash) ---")
    print("场景: CPU 飙升 + IO 延迟高 -> 系统崩溃")
    
    # 模拟危险状态特征
    dangerous_state = encode_log_feature(cpu_usage=95, error_rate=2, io_latency=800)
    
    # 初始预测
    pred, uncertainty = seed.predict(dangerous_state, action=0) # 0=Ignore
    # Linear model or Deep model access to value might differ, adapt to current seed implementation
    if hasattr(seed, 'value_network'):
        val = seed.value_network.forward(dangerous_state)[0]
    else:
         val = np.dot(dangerous_state, seed.value_weights)
    
    # Ensure val is a scalar float
    if isinstance(val, np.ndarray):
        val = float(val)

    print(f"   [初始直觉] 对危险状态的价值评估: {val:.4f} (它是懵懂的)")
    
    # 模拟崩溃体验
    # Action 0 (Ignore) -> Result: Crash (Reward = -10.0)
    # Action 1 (Alert)  -> Result: Safe  (Reward = +1.0)
    
    print("   💥 系统崩溃！正在刻入痛苦记忆...")
    # 强化训练 20 次 (模拟多次刻骨铭心的教训)
    crashed_state = np.random.randn(4) # 崩溃后的混乱状态
    
    for _ in range(20):
        # 惩罚 "Ignore" 行为
        exp_bad = Experience(dangerous_state, action=0, reward=-10.0, next_state=crashed_state)
        seed.learn(exp_bad)
        
        # 奖励 "Alert" 行为 (假设它偶尔蒙对了一次)
        exp_good = Experience(dangerous_state, action=1, reward=5.0, next_state=dangerous_state)
        seed.learn(exp_good)
        
    print("\n--- 阶段 2: 直觉验证 (Testing Intuition) ---")
    
    # 1. 再次遇到危险状态
    if hasattr(seed, 'value_network'):
        val_danger = seed.value_network.forward(dangerous_state)[0]
    else:
        val_danger = np.dot(dangerous_state, seed.value_weights)
        
    if isinstance(val_danger, np.ndarray): val_danger = float(val_danger)
        
    # 比较两个动作的价值
    pred_ignore, _ = seed.predict(dangerous_state, action=0)
    val_ignore = seed.evaluate(dangerous_state, pred_ignore, 0.1)
    if isinstance(val_ignore, np.ndarray): val_ignore = float(val_ignore)
    
    pred_alert, _ = seed.predict(dangerous_state, action=1)
    val_alert = seed.evaluate(dangerous_state, pred_alert, 0.1)
    if isinstance(val_alert, np.ndarray): val_alert = float(val_alert)
    
    print(f"   [进化后] 对危险状态的价值评估: {val_danger:.4f}")
    print(f"   [选择] 忽略(Ignore) 的价值: {val_ignore:.4f}")
    print(f"   [选择] 报警(Alert)  的价值: {val_alert:.4f}")
    
    if val_alert > val_ignore:
        print("   ✅ 成功: 系统现在本能地选择报警！")
    else:
        print("   ❌ 失败: 系统依然选择忽略。")
        
    # 2. 测试泛化 (Generalization)
    # 遇到一个类似的但不完全一样的状态 (CPU高，IO一般)
    print("\n--- 阶段 3: 泛化测试 (Generalization) ---")
    similar_state = encode_log_feature(cpu_usage=90, error_rate=1, io_latency=600)
    if hasattr(seed, 'value_network'):
        val_sim = seed.value_network.forward(similar_state)[0]
    else:
        val_sim = np.dot(similar_state, seed.value_weights)

    if isinstance(val_sim, np.ndarray): val_sim = float(val_sim)

    print(f"   [新情况] 遇到类似高负载状态 (CPU=90, IO=600)")
    print(f"   [直觉] 价值评估: {val_sim:.4f} (越低表示越警惕)")
    
    if val_sim < 0: # Assuming negative value learned for danger
        print("   ✅ 成功: 系统展现出了'一朝被蛇咬，十年怕井绳'的泛化恐惧。")
    else:
        print("   ⚠️ 提示: 系统可能需要更多样本才能泛化，或者当前仍为线性模型限制了泛化能力。")

if __name__ == "__main__":
    demo_survival()