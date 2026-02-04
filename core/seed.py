"""
THE SEED OF INTELLIGENCE (智能的种子)

This file represents the minimal algorithmic description of the "Primal Seed" of AGI.
It is not the full system, but the mathematical essence that allows the system to grow.

Core Principle: Active Inference (主动推理) & Free Energy Minimization (自由能最小化)
"""

import numpy as np
from typing import List, Tuple, Any
from dataclasses import dataclass
import random
import logging

from core.evolution.dynamics import EvolutionaryDynamics

# Setup logger
logger = logging.getLogger(__name__)

@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray

class ExperienceReplay:
    """
    Experience Replay Buffer for Offline Training (Dreaming).
    Solves:
    1. Sample Scarcity: Reuses rare events (e.g., crashes) multiple times.
    2. Temporal Correlation: Breaks correlation by random sampling.
    """
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.position = 0
        
    def push(self, experience: Experience):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> List[Experience]:
        """Randomly samples a batch of experiences."""
        if len(self.buffer) < batch_size:
            return self.buffer
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class NeuralNetwork:
    """
    A compact implementation of a Multi-Layer Perceptron (Deep Neural Network).
    Supports:
    - Arbitrary depth and width
    - ReLU Activation for hidden layers
    - Linear Activation for output layer (Regression)
    - Backpropagation with SGD
    """
    def __init__(self, layers: List[int], learning_rate=0.01):
        self.weights = []
        self.biases = []
        self.layers = layers
        self.lr = learning_rate
        self.activations = [] # Store for backprop
        
        for i in range(len(layers) - 1):
            # He Initialization for ReLU
            scale = np.sqrt(2.0 / layers[i])
            w = np.random.randn(layers[i], layers[i+1]) * scale
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.activations = [x.reshape(1, -1)]
        output = self.activations[0]
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(output, w) + b
            if i < len(self.weights) - 1:
                output = np.maximum(0, z) # ReLU
            else:
                # 🆕 [2026-01-09] TRAE建议: 输出层界限化到[-1,1],防止数值爆炸
                output = np.tanh(z)  # Bounded output instead of linear
            self.activations.append(output)
            
        return output.flatten()
        
    def backward(self, target: np.ndarray) -> float:
        """
        Backpropagates error and updates weights.
        Returns MSE Loss.
        🔧 [2026-01-09] 增强稳定性: Loss检测 + 自适应梯度裁剪 + 权重范数约束
        """
        target = target.reshape(1, -1)
        output = self.activations[-1]
        error = output - target # dLoss/dOutput for MSE
        loss = np.mean(error**2)
        
        # 🆕 [2026-01-09] Loss异常检测 - 如果loss过大,跳过更新
        if loss > 1e6:
            logger.warning(f"⚠️ Loss异常 ({loss:.2e}), 跳过本次权重更新")
            return loss
        
        # Gradient Clipping to prevent explosion (放宽到±10)
        delta = np.clip(error, -10.0, 10.0)
        
        for i in reversed(range(len(self.weights))):
            input_act = self.activations[i]
            
            # Gradient Descent Update
            dW = np.dot(input_act.T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            
            # 🔧 [2026-01-09] 自适应梯度裁剪 - 按范数而非逐元素
            dW_norm = np.linalg.norm(dW)
            if dW_norm > 10.0:
                dW = dW * (10.0 / (dW_norm + 1e-8))
            
            db_norm = np.linalg.norm(db)
            if db_norm > 10.0:
                db = db * (10.0 / (db_norm + 1e-8))
            
            # Update weights
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db
            
            # 🆕 [2026-01-09] 权重范数约束 - 防止权重爆炸
            w_norm = np.linalg.norm(self.weights[i])
            if w_norm > 50.0:
                self.weights[i] = self.weights[i] * (50.0 / (w_norm + 1e-8))
                logger.debug(f"⚠️ Layer {i} 权重范数过大({w_norm:.2f}), 已归一化到50")
            
            # Prepare delta for next layer (if not first layer)
            if i > 0:
                # Backprop error: delta_prev = (delta @ W.T) * ReLU_derivative
                delta = np.dot(delta, self.weights[i].T)
                delta = delta * (input_act > 0) # ReLU derivative
                # Clip intermediate delta
                delta = np.clip(delta, -10.0, 10.0)
                
        return loss
    
    def check_health(self) -> dict:
        """
        🆕 [2026-01-09] 检查网络权重健康状态
        返回: {max_weight, avg_weight, has_nan, has_inf, is_healthy}
        """
        max_weight = max(np.max(np.abs(w)) for w in self.weights)
        avg_weight = np.mean([np.mean(np.abs(w)) for w in self.weights])
        
        has_nan = any(np.any(np.isnan(w)) for w in self.weights)
        has_inf = any(np.any(np.isinf(w)) for w in self.weights)
        
        # 健康标准: 最大权重<100, 无NaN/Inf
        return {
            'max_weight': float(max_weight),
            'avg_weight': float(avg_weight),
            'has_nan': has_nan,
            'has_inf': has_inf,
            'is_healthy': max_weight < 100 and not has_nan and not has_inf
        }
    
    def reset_if_unhealthy(self) -> bool:
        """
        🆕 [2026-01-09] 如果权重不健康,重新初始化
        返回: True=已重置, False=健康无需重置
        """
        health = self.check_health()
        
        if not health['is_healthy']:
            logger.warning(
                f"⚠️ 检测到权重异常,执行重置: max={health['max_weight']:.2e}, "
                f"avg={health['avg_weight']:.2e}, nan={health['has_nan']}, inf={health['has_inf']}"
            )
            
            # 重新初始化所有权重 (He Initialization)
            for i in range(len(self.layers) - 1):
                scale = np.sqrt(2.0 / self.layers[i])
                self.weights[i] = np.random.randn(self.layers[i], self.layers[i+1]) * scale
                self.biases[i] = np.zeros((1, self.layers[i+1]))
            
            logger.info("✅ 权重已重置为健康状态")
            return True
        return False

class TheSeed:
    """
    The Primal Seed (原始种子) - Evolved to Phase 2 (Deep Learning)
    
    Attributes:
        1. Internal Model (World Model): Deep Neural Network predicting next state.
        2. Value Function (Evaluator): Deep Neural Network estimating state value.
        3. Policy (Actor): Selects actions to maximize Value.
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # DNA: The unchangeable meta-rules
        self.learning_rate = 0.01
        self.curiosity_weight = 0.5  
        
        # The Body: Deep Neural Networks
        # World Model: [State + Action] -> [Hidden] -> [Hidden] -> [Next State]
        hidden_dim = 64
        self.world_model = NeuralNetwork(
            layers=[state_dim + action_dim, hidden_dim, hidden_dim, state_dim],
            learning_rate=self.learning_rate
        )
        
        # Value Function: [State] -> [Hidden] -> [Hidden] -> [Value]
        self.value_network = NeuralNetwork(
            layers=[state_dim, hidden_dim, hidden_dim, 1],
            learning_rate=self.learning_rate
        )
        
        # Memory System: Experience Replay (The Hippocampus)
        self.memory = ExperienceReplay(capacity=5000)
        self.batch_size = 32 # Dream batch size
    
    def perceive(self, raw_input: Any) -> np.ndarray:
        """
        Function 1: Compression (压缩)
        Transforms high-dimensional chaos (reality) into low-dimensional order (concept).
        """
        # Simplistic compression for demonstration
        return np.array(raw_input) if isinstance(raw_input, list) else np.random.randn(self.state_dim)

    def predict(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        """
        Function 2: Prediction (预测)
        Uses Deep Neural Network to predict the future.
        Returns: (Predicted Next State, Uncertainty/Entropy)
        🆕 [2026-01-09] 增加健康检查和输出防护
        """
        # 🆕 [2026-01-09] 定期检查world_model健康状态 (每100次)
        if not hasattr(self, '_health_check_counter'):
            self._health_check_counter = 0
        
        self._health_check_counter += 1
        if self._health_check_counter % 100 == 0:
            if self.world_model.reset_if_unhealthy():
                logger.error("⚠️ World Model权重异常已重置, 预测将临时降级")
        
        # One-hot encode action
        action_vec = np.zeros(self.action_dim)
        if 0 <= action < self.action_dim:
            action_vec[action] = 1.0
            
        input_vec = np.concatenate([state, action_vec])
        
        # Deep Forward Pass
        prediction = self.world_model.forward(input_vec)
        
        # 🆕 [2026-01-09] 输出防护 - 检测异常并降级处理
        pred_norm = np.linalg.norm(prediction)
        has_nan = np.any(np.isnan(prediction))
        has_inf = np.any(np.isinf(prediction))
        
        if has_nan or has_inf or pred_norm > 10:  # 🔧 降低阈值(1000→10),tanh输出已界限化
            logger.error(
                f"⚠️ World Model输出异常: norm={pred_norm:.2e}, "
                f"nan={has_nan}, inf={has_inf} - 执行应急降级"
            )
            
            # 应急降级: 返回状态的微小扰动 (保持连续性)
            prediction = state + np.random.randn(self.state_dim) * 0.01
            prediction = np.clip(prediction, -10, 10)
            uncertainty = 0.99  # 🆕 [TRAE] 极高但有界的不确定性
        else:
            # 正常情况: 计算标准不确定性
            # Uncertainty heuristic for deterministic NN:
            # High variance in the predicted state vector often implies complex/chaotic states.
            # Ideally, we would use an Ensemble or Dropout for epistemic uncertainty.
            # For now, we stick to state entropy proxy.
            raw_uncertainty = np.std(prediction)
            
            # 🆕 [2026-01-09] TRAE建议: 不确定性标定化,映射到(0,1)
            # 公式: u/(1+u) 单调递增,界限在[0,1)
            uncertainty = raw_uncertainty / (1.0 + raw_uncertainty)
        
        return prediction, uncertainty

    def evaluate(self, state: np.ndarray, predicted_next_state: np.ndarray, uncertainty: float) -> float:
        """
        Function 3: Valuation (价值评估) - The Compass
        Value = Extrinsic Reward (Food) + Intrinsic Reward (Curiosity)
        """
        # Extrinsic Value (Survival/Task Success) via Value Network
        survival_value = self.value_network.forward(state)[0]
        
        # Intrinsic Value (Curiosity)
        curiosity_value = uncertainty * self.curiosity_weight
        
        return survival_value + curiosity_value

    def get_intrinsic_reward(self, uncertainty: float) -> float:
        """Helper to expose just the curiosity component"""
        return uncertainty * self.curiosity_weight

    def act(self, state: np.ndarray) -> int:
        """
        Function 4: Action (行动) - Active Inference Upgrade
        Select action that maximizes Expected Free Energy over a time horizon.
        """
        # Use the new simulate_trajectory for robust lookahead
        best_action = 0
        max_efe = -float('inf')
        
        for a in range(self.action_dim):
            # Simulate a trajectory starting with action 'a'
            trajectory = self.simulate_trajectory(state, first_action=a, horizon=3)
            
            # Calculate EFE for this trajectory
            cumulative_value = 0.0
            gamma = 0.9
            for t, (s, u, _) in enumerate(trajectory):
                # evaluate(state, next_state, uncertainty)
                prev_s = state if t == 0 else trajectory[t-1][0]
                val = self.evaluate(prev_s, s, u)
                cumulative_value += (gamma ** t) * val
            
            if cumulative_value > max_efe:
                max_efe = cumulative_value
                best_action = a
                
        return best_action

    def simulate_trajectory(
        self, 
        start_state: np.ndarray, 
        first_action: int, 
        horizon: int = 100,  # 🔧 [2026-01-17] 与元认知SHALLOW_HORIZON一致
        adaptive: bool = True,  # 🔧 新增：启用自适应深度
        max_horizon_extension: int = 2000  # 🔧 [2026-01-17] 与元认知MAX_HORIZON一致
    ) -> List[Tuple[np.ndarray, float, int]]:
        """
        [Reasoning Internalization - 🔧 Enhanced with Adaptive Depth]
        Simulates a future timeline in the Latent Space.
        Returns a list of (State, Uncertainty, ActionTaken).
        
        Enhanced Features:
        1. Default horizon extended from 5 to 20 for deeper reasoning
        2. Configurable range: 15-25 steps for consciousness emergence support
        3. 🆕 Adaptive depth extension: Auto-extends horizon when high uncertainty detected
        4. 🆕 Early stopping: Terminates when state converges (optimization)
        
        Args:
            start_state: Initial state vector
            first_action: First action to take
            horizon: Base reasoning depth (default 20)
            adaptive: Enable adaptive depth extension (default True)
            max_horizon_extension: Maximum allowed horizon extension (default 30)
        
        Returns:
            List of (state, uncertainty, action) tuples
        """
        trajectory = []
        current_state = start_state
        effective_horizon = horizon
        
        # Step 1: Force the first action
        next_state, uncertainty = self.predict(current_state, first_action)
        trajectory.append((next_state, uncertainty, first_action))
        current_state = next_state
        
        # Step 2: Auto-regressive simulation with adaptive control
        for step in range(horizon - 1):
            # Policy: Choose best action for the simulated state
            # 🔧 [2026-01-22] 修复：添加动作多样性机制，防止陷入同一动作循环
            best_a = 0
            best_val = -float('inf')
            best_next = current_state
            best_unc = 0.0

            # 🔧 [2026-01-22] Epsilon-greedy with action cooldown to prevent loops
            # 检查最近5步的动作，避免重复
            if len(trajectory) >= 5:
                recent_actions = [t[2] for t in trajectory[-5:]]
                # 如果某个动作出现超过3次，降低其优先级
                action_counts = {}
                for act in recent_actions:
                    action_counts[act] = action_counts.get(act, 0) + 1
            else:
                action_counts = {}

            # 评估所有动作
            action_values = []
            for a in range(self.action_dim):
                s_next, s_unc = self.predict(current_state, a)
                val = self.evaluate(current_state, s_next, s_unc)

                # 🔧 [2026-01-22] 惩罚过度使用的动作
                penalty = action_counts.get(a, 0) * 0.5  # 每次重复降低0.5分
                adjusted_val = val - penalty
                action_values.append((a, adjusted_val, s_next, s_unc))

                if adjusted_val > best_val:
                    best_val = adjusted_val
                    best_next = s_next
                    best_unc = s_unc
                    best_a = a

            # 🔧 [2026-01-22] 添加10%随机探索，防止局部最优
            if random.random() < 0.10:
                # 从前3个最佳动作中随机选择
                sorted_actions = sorted(action_values, key=lambda x: x[1], reverse=True)[:3]
                if sorted_actions:
                    best_a, _, best_next, best_unc = random.choice(sorted_actions)
            
            # 🆕 [2026-01-09] Trajectory状态异常防护 - 防止NaN/Inf传播
            state_norm = np.linalg.norm(best_next) if not (np.any(np.isnan(best_next)) or np.any(np.isinf(best_next))) else float('inf')
            
            if np.any(np.isnan(best_next)) or np.any(np.isinf(best_next)):
                logger.warning(f"⚠️ Trajectory[{step}]: NaN/Inf detected, 执行裁剪+清理")
                best_next = np.clip(best_next, -10, 10)
                best_next = np.nan_to_num(best_next, nan=0.0, posinf=10.0, neginf=-10.0)
                state_norm = np.linalg.norm(best_next)
                # clip后可能norm仍>10(因64维),需二次检查
                if state_norm > 10:
                    logger.warning(f"⚠️ Trajectory[{step}]: Clip后norm={state_norm:.2e},执行归一化")
                    best_next = best_next / (state_norm + 1e-8)
                    state_norm = np.linalg.norm(best_next)
            elif state_norm > 100:
                logger.warning(f"⚠️ Trajectory[{step}]: Large norm {state_norm:.2e}, 执行归一化")
                best_next = best_next / (state_norm + 1e-8)
                state_norm = np.linalg.norm(best_next)
            
            trajectory.append((best_next, best_unc, best_a))
            current_state = best_next
            
            # 🔧 自适应深度延长 (降低不确定性阈值到2.0)
            if adaptive and step > int(horizon * 0.8) and effective_horizon < max_horizon_extension:
                # 检查最近5步的平均不确定性
                recent_uncertainties = [t[1] for t in trajectory[-5:]] if len(trajectory) >= 5 else []
                if recent_uncertainties:
                    avg_uncertainty = np.mean(recent_uncertainties)
                    if avg_uncertainty > 2.0:  # 中等偏高不确定性即触发
                        extension = min(5, max_horizon_extension - effective_horizon)
                        effective_horizon += extension
                        logger.info(f"🔍 自适应延长推理深度: {horizon} → {effective_horizon} (步骤{step}, 不确定性={avg_uncertainty:.3f})")
            
            # 🔧 早停机制：状态收敛检测
            # 🆕 [2026-01-22] 修复：提高最小步数要求和收敛阈值，避免推理深度不足
            # 动态最小步数：基于 horizon 的一定比例（至少50步，最多1000步）
            min_steps_before_early_stop = max(50, min(int(horizon * 0.1), 1000))

            if adaptive and len(trajectory) >= min_steps_before_early_stop:
                # 计算最近5步的状态变化
                recent_changes = []
                for i in range(-5, 0):
                    if i + len(trajectory) >= 0:
                        state_change = np.linalg.norm(trajectory[i][0] - trajectory[i-1][0])
                        recent_changes.append(state_change)

                # 🆕 [2026-01-09] TRAE建议: 加不确定性门槛,避免"假收敛"
                # 🔧 [2026-01-22] 优化: 提高收敛阈值，降低假阳性
                if recent_changes:
                    avg_change = np.mean(recent_changes)
                    # 计算最近5步的平均不确定性
                    recent_uncertainties = [trajectory[i][1] for i in range(-5, 0) if i + len(trajectory) >= 0]
                    avg_uncertainty = np.mean(recent_uncertainties) if recent_uncertainties else 1.0

                    # 🔧 [2026-01-22] 更严格的收敛阈值：
                    # - avg_change < 0.3 (从0.5降低到0.3，更严格)
                    # - avg_uncertainty < 0.3 (从0.4降低到0.3，更严格)
                    if avg_change < 0.3 and avg_uncertainty < 0.3:  # 低变化且低不确定性
                        logger.info(
                            f"✅ 推理真正收敛于步骤 {step+1}/{horizon} "
                            f"(变化={avg_change:.5f}, 不确定性={avg_uncertainty:.3f})"
                        )
                        break
                    elif avg_change < 0.3 and avg_uncertainty >= 0.3:
                        # 🆕 [2026-01-17] 添加假收敛计数器，防止无限循环
                        if not hasattr(self, '_false_convergence_count'):
                            self._false_convergence_count = 0
                        self._false_convergence_count += 1
                        
                        # 最多警告10次，之后强制接受为"软收敛"
                        if self._false_convergence_count <= 10:
                            logger.warning(
                                f"⚠️ 假收敛检测 [{self._false_convergence_count}/10]: "
                                f"变化低({avg_change:.5f})但不确定性高({avg_uncertainty:.3f})"
                            )
                        elif self._false_convergence_count == 11:
                            logger.info(
                                f"✅ 假收敛达到上限，强制接受为软收敛 "
                                f"(变化={avg_change:.5f}, 不确定性={avg_uncertainty:.3f})"
                            )
                            self._false_convergence_count = 0  # 重置计数器
                            break  # 强制收敛
                        # 不触发break,继续推理
            # 检查是否需要继续（考虑延长后的horizon）
            if step + 2 >= effective_horizon:
                break
        
        return trajectory

    def internalize_knowledge(self, concept_embedding: np.ndarray):
        """
        [Knowledge Compression]
        Train the World Model to 'recognize' this concept as a valid state transition.
        We treat reading a concept as an observation: State(Empty) -> State(Concept).
        """
        # Create a dummy experience: 
        # State: Zero (Ignorance) -> Action: Analyze (1) -> Next State: Concept
        zero_state = np.zeros(self.state_dim)
        action_idx = 1 # Analyze
        
        exp = Experience(
            state=zero_state,
            action=action_idx,
            reward=1.0, # High reward for learning
            next_state=concept_embedding
        )
        self.learn(exp)

    def project_thought(self, state_vector: np.ndarray) -> str:
        """
        [Language Decoder Stub]
        Projects the latent thought vector back to a crude string representation.
        In the future, this will be a small 0.5B LLM.
        For now, it returns a hash-based conceptual ID.
        """
        # Simple LSH (Locality Sensitive Hashing) simulation
        # Sign of the first few dimensions
        code = "".join(["1" if x > 0 else "0" for x in state_vector[:8]])
        return f"Concept_{code}"

    def learn(self, experience: Experience):
        """
        Function 5: Plasticity (可塑性) - The Growth
        Update internal Deep Neural Models based on prediction error (Backpropagation).
        Uses Experience Replay to learn from past (Dreaming).
        """
        # 1. Consolidate memory
        self.memory.push(experience)
        
        # 2. Dream (Sample from memory)
        # If we have enough memories, train on a batch
        experiences = self.memory.sample(self.batch_size)
        
        # Train on the batch
        total_wm_loss = 0
        total_vf_loss = 0
        
        for exp in experiences:
            # --- Train World Model ---
            action_vec = np.zeros(self.action_dim)
            if 0 <= exp.action < self.action_dim:
                action_vec[exp.action] = 1.0
            input_vec = np.concatenate([exp.state, action_vec])
            
            self.world_model.forward(input_vec)
            total_wm_loss += self.world_model.backward(exp.next_state)
            
            # --- Train Value Function ---
            next_val = self.value_network.forward(exp.next_state)[0]
            td_target = np.array([exp.reward + next_val])
            
            self.value_network.forward(exp.state)
            total_vf_loss += self.value_network.backward(td_target)
            
        # Optional: Return avg loss for monitoring
        return total_wm_loss / len(experiences), total_vf_loss / len(experiences)

    def exist(self, environment):
        """
        The Loop of Life
        """
        state = self.perceive(environment.get_observation())
        
        # 1. Observe & Compress
        action = self.act(state)
        
        # 2. Act on World
        raw_next, reward = environment.step(action)
        next_state = self.perceive(raw_next)
        
        # 3. Learn from Reality (Consolidate)
        exp = Experience(state, action, reward, next_state)
        self.learn(exp)
        
        # 4. Evolve
        return next_state
