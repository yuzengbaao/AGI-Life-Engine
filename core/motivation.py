"""
AGI Motivation System - 动机核心模块
实现类似马斯洛需求层次和多巴胺奖励机制的动机系统

改进版：支持目标系统的反馈闭环
"""

class MotivationCore:
    """
    AGI Motivation System (Simulating Maslow's Hierarchy & Dopamine)
    
    改进版: 支持目标系统的反馈闭环
    """
    def __init__(self):
        # Basic Stats (0.0 - 100.0)
        self.energy = 100.0         # 能量：随操作消耗，随时间恢复
        self.curiosity = 50.0       # 好奇心：随无聊增加，驱动探索
        self.satisfaction = 50.0    # 满足感：完成任务增加，随时间衰减
        self.boredom = 0.0          # 无聊度：无任务时线性增加
        self.frustration = 0.0      # 挫败感：失败时增加，成功时减少 (新增)
        self.needs_exploration_trigger = False # 🔧 NEW: Signal for AGI_Life_Engine to trigger evolution
        
        # Drives (Internal Goals)
        self.current_drive = "IDLE"
        
        # 历史追踪 (用于连胜/连败效应)
        self.recent_outcomes = []   # [(success: bool, score: float), ...]
        self.streak = 0             # 正=连胜, 负=连败
        
    def tick(self, active_task: bool):
        """Update internal state based on time passing"""
        if active_task:
            self.boredom = max(0, self.boredom - 5)
            self.energy = max(0, self.energy - 0.5)
            # --- MODIFICATION: Self-Correction Applied (0.1 -> 0.05) ---
            self.curiosity = max(0, self.curiosity - 0.05) # Focus reduces wandering curiosity
            # -----------------------------------------------------------
            self.frustration = max(0, self.frustration - 0.2)  # 工作中挫败感缓慢下降
        else:
            self.boredom = min(100, self.boredom + 2)
            self.energy = min(100, self.energy + 1)
            self.curiosity = min(100, self.curiosity + 1) # Boredom breeds curiosity
            
        # 满足感随时间自然衰减
        self.satisfaction = max(0, self.satisfaction - 0.1)
            
    def receive_goal_feedback(self, success: bool, score: float, is_timeout: bool = False):
        """
        接收目标系统的反馈，调整内部状态
        
        这是实现闭环的关键方法！
        
        Args:
            success: 目标是否成功完成
            score: 完成质量 (0.0-1.0)
            is_timeout: 是否因超时失败
        """
        self.recent_outcomes.append((success, score))
        if len(self.recent_outcomes) > 10:
            self.recent_outcomes.pop(0)
        
        if success:
            # 成功反馈：多巴胺奖励
            reward = 10 + score * 20  # 基础10分 + 质量加成(最高20)
            
            # 连胜加成 (Streak Bonus)
            if self.streak > 0:
                self.streak += 1
                # --- ANTI-GAMING MECHANISM ---
                # 如果连胜过多，且处于 MAINTAIN 模式，收益递减甚至产生厌倦
                if self.streak > 5 and self.current_drive == "MAINTAIN":
                    reward *= 0.5  # 收益减半
                    # 连胜反而增加无聊感 (太简单了)
                    self.boredom = min(100, self.boredom + 5) 
                else:
                    reward *= (1 + 0.1 * min(self.streak, 5))  # 正常连胜加成
                    self.boredom = max(0, self.boredom - 10) # 正常减少无聊
            else:
                self.streak = 1
                self.boredom = max(0, self.boredom - 10)
            
            self.satisfaction = min(100, self.satisfaction + reward)
            self.frustration = max(0, self.frustration - 15)
            self.energy = max(0, self.energy - 5)  # 成功消耗能量
            
        else:
            # 失败反馈：挫败感
            penalty = 10 if not is_timeout else 5  # 超时惩罚较轻
            
            # 连败加剧挫败感
            if self.streak < 0:
                self.streak -= 1
                penalty *= (1 + 0.1 * min(abs(self.streak), 5))
            else:
                self.streak = -1
            
            self.frustration = min(100, self.frustration + penalty)
            self.satisfaction = max(0, self.satisfaction - penalty / 2)
            
            # 如果连续失败，触发自省驱动
            if self.streak <= -3:
                self.current_drive = "REFLECT"

    def apply_external_feedback(self, feedback: dict):
        """
        Process aggregated feedback from GoalManager (or other sources) 
        to adjust motivation state.
        
        Expected feedback keys:
        - recent_success_rate (0.0 - 1.0)
        - recent_average_score (0.0 - 1.0)
        - pending_count (int)
        - streak (int)
        """
        # Sync streak
        self.streak = int(feedback.get("streak", 0))
        
        # Adjust Satisfaction based on recent performance
        avg_score = feedback.get("recent_average_score", 0.5)
        success_rate = feedback.get("recent_success_rate", 0.5)
        
        if success_rate > 0.7:
            self.satisfaction = min(100.0, self.satisfaction + 5.0)
        elif success_rate < 0.3:
            self.satisfaction = max(0.0, self.satisfaction - 5.0)
            self.frustration = min(100.0, self.frustration + 5.0)
            
        # Adjust Drive based on workload (pending_count)
        pending_count = feedback.get("pending_count", 0)
        if pending_count > 0:
            # If there is work to do, reduce boredom
            self.boredom = max(0.0, self.boredom - (pending_count * 2.0))
        else:
            # If no work, boredom creeps up slightly
            self.boredom = min(100.0, self.boredom + 1.0)

    def get_dominant_drive(self) -> str:
        """Determine what the AGI 'wants' to do"""
        # 优先级1: 能量不足需要休息
        if self.energy < 20:
            self.current_drive = "REST"
            return "REST"
        
        # 优先级2: 挫败感过高需要求助/反思
        if self.frustration > 60:
            self.current_drive = "REFLECT"
            return "REFLECT"

        # 优先级3: 无聊导致探索
        # [MODIFIED 2026-01-29] Lowered threshold from 80 to 30 to fix "Vegetative State"
        if self.boredom > 30:
            self.current_drive = "EXPLORE"
            self.needs_exploration_trigger = True
            return "EXPLORE"

        self.needs_exploration_trigger = False
        # 默认: 维持现状或工作
        self.current_drive = "MAINTAIN"
        return "MAINTAIN"

    def update_drive(self, active_task: bool = False) -> str:
        """
        封装 tick() + get_dominant_drive() 的便捷方法。
        
        此方法作为桥接接口，供外部模块（如 AGI_Life_Engine 主循环）调用。
        采用包装器模式复用现有逻辑，避免代码重复，保持 DRY 原则。
        
        Args:
            active_task: 是否有活动任务在执行。
                        True = 能量缓慢恢复，无聊减少
                        False = 能量快速恢复，无聊增加
        
        Returns:
            str: 当前主导驱动力类型
                 - "REST": 能量不足，需要休息
                 - "REFLECT": 挫败感过高，需要反思
                 - "EXPLORE": 无聊，需要探索新事物
                 - "MAINTAIN": 维持当前状态/继续工作
        
        Example:
            >>> motivation = MotivationCore()
            >>> drive = motivation.update_drive(active_task=True)
            >>> print(f"当前驱动力: {drive}")
            当前驱动力: MAINTAIN
        """
        self.tick(active_task)
        return self.get_dominant_drive()
