import sys
import os
import numpy as np
import unittest

# 将项目根目录添加到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.seed import TheSeed

class MockSeed(TheSeed):
    """
    TheSeed 的子类，用于 Mock 内部神经网络，
    以便为测试逻辑提供确定性的行为。
    """
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        # 覆盖 DNA 进行测试：提高好奇心权重以使不确定性显现
        self.curiosity_weight = 1.0
        
    def predict(self, state: np.ndarray, action: int):
        """
        用于测试的确定性状态转换。
        为了简化，状态仅为一个包裹在数组中的浮点标量。
        
        场景地图:
        - 动作 0 (陷阱): 0 -> 1 (奖励 5) -> 1 (奖励 5) -> ... (总分低)
        - 动作 1 (投资): 0 -> 2 (奖励 0) -> 3 (奖励 100) -> ... (总分高)
        """
        s_val = state[0]
        next_state = state.copy()
        uncertainty = 0.0
        
        if s_val == 0: # 起点
            if action == 0: next_state[0] = 1 # 路径 A
            elif action == 1: next_state[0] = 2 # 路径 B
            
        elif s_val == 1: # 路径 A (停滞)
            next_state[0] = 1
            
        elif s_val == 2: # 路径 B (桥梁)
            next_state[0] = 3
            
        elif s_val == 3: # 路径 B (大奖)
            next_state[0] = 3
            
        return next_state, uncertainty

    def evaluate(self, state: np.ndarray, predicted_next_state: np.ndarray, uncertainty: float) -> float:
        """
        确定性估值函数。
        """
        s_val = predicted_next_state[0]
        extrinsic = 0.0
        
        if s_val == 1: extrinsic = 5.0   # 眼前的小利
        elif s_val == 2: extrinsic = 0.0   # 投资期无奖励
        elif s_val == 3: extrinsic = 100.0 # 未来的大回报
            
        return extrinsic + (uncertainty * self.curiosity_weight)

class TestActiveInference(unittest.TestCase):
    def setUp(self):
        self.seed = MockSeed(state_dim=1, action_dim=2)
        self.start_state = np.array([0.0])

    def test_lookahead_capability(self):
        """
        验证智能体是否向后看 3 步。
        动作 0 路径: 5 + 4.5 + 4.05 = 13.55
        动作 1 路径: 0 + 90 + 81 = 171
        智能体必须选择动作 1。
        """
        print("\nTesting Lookahead Capability (测试前瞻能力)...")
        action = self.seed.act(self.start_state)
        print(f"Chosen Action (选择的动作): {action}")
        
        if action == 1:
            print("PASS: Agent chose delayed gratification (Active Inference working). - 通过：智能体选择了延迟满足")
        else:
            print("FAIL: Agent chose immediate reward (Lookahead failed). - 失败：智能体选择了眼前利益")
            
        self.assertEqual(action, 1, "Agent failed to look ahead for higher future reward.")

if __name__ == '__main__':
    unittest.main()
