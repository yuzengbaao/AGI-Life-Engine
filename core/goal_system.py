"""
AGI Goal System - 可验证目标的闭环系统
解决原系统中目标抽象、无法验证、死循环的问题

核心设计原则：
1. 每个目标必须有明确的完成标准 (success_criteria)
2. 目标有超时机制防止无限执行 (timeout)
3. 目标有重试限制防止死循环 (max_attempts)
4. 目标完成后产生可量化的反馈 (outcome_score)
"""

import os
import time
import json
import logging
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger("GoalSystem")


class GoalStatus(Enum):
    """目标状态枚举"""
    PENDING = "pending"          # 等待执行
    IN_PROGRESS = "in_progress"  # 执行中
    COMPLETED = "completed"      # 成功完成
    FAILED = "failed"            # 执行失败
    TIMEOUT = "timeout"          # 超时
    ABANDONED = "abandoned"      # 主动放弃


class GoalType(Enum):
    """目标类型 - 决定验证方式"""
    FILE_CREATE = "file_create"           # 创建文件
    FILE_MODIFY = "file_modify"           # 修改文件
    COMMAND_EXECUTE = "command_execute"   # 执行命令
    OBSERVATION = "observation"           # 观察类
    ANALYSIS = "analysis"                 # 分析类
    COMMUNICATION = "communication"       # 交流类
    GUI_ACTION = "gui_action"             # GUI操作类 (Vision-Driven)
    DREAM = "dream"                       # 记忆固化/做梦 (Memory Consolidation)
    CUSTOM = "custom"                     # 自定义验证


@dataclass
class VerifiableGoal:
    """
    可验证的目标结构
    
    与原系统的区别：
    - 原系统: {"goal": "观察屏幕", "priority": "medium"}  ← 太抽象
    - 新系统: 完整的可验证目标结构
    """
    # 基础信息
    id: str                              # 唯一标识
    description: str                     # 目标描述
    goal_type: GoalType                  # 目标类型
    priority: str = "medium"             # 优先级
    
    # 验证标准
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    # 示例: {"file_exists": "output.txt", "min_size": 100}
    # 示例: {"contains_text": "成功", "output_file": "result.md"}
    
    # 执行约束
    timeout_seconds: int = 60            # 超时时间
    max_attempts: int = 3                # 最大尝试次数
    
    # 状态追踪
    status: GoalStatus = GoalStatus.PENDING
    attempts: int = 0                    # 当前尝试次数
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # 结果
    outcome_score: float = 0.0           # 完成质量 0.0-1.0
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典"""
        data = asdict(self)
        data['goal_type'] = self.goal_type.value
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerifiableGoal':
        """从字典恢复"""
        data['goal_type'] = GoalType(data['goal_type'])
        data['status'] = GoalStatus(data['status'])
        return cls(**data)


class GoalVerifier:
    """
    目标验证器 - 根据 success_criteria 判断目标是否达成
    """
    
    def __init__(self, base_path: str = "."):
        self.base_path = base_path
    
    def verify(self, goal: VerifiableGoal) -> Tuple[bool, float, str]:
        """
        验证目标是否完成
        
        Returns:
            (is_success, score, message)
        """
        criteria = goal.success_criteria
        
        if not criteria:
            # 无明确标准，视为观察类目标，执行即成功
            return True, 0.5, "目标无明确验证标准，按执行完成计"
        
        try:
            # 根据目标类型选择验证策略
            if goal.goal_type == GoalType.FILE_CREATE:
                return self._verify_file_create(criteria)
            elif goal.goal_type == GoalType.FILE_MODIFY:
                return self._verify_file_modify(criteria)
            elif goal.goal_type == GoalType.COMMAND_EXECUTE:
                return self._verify_command(criteria)
            elif goal.goal_type == GoalType.OBSERVATION:
                return self._verify_observation(criteria, goal.result_data)
            elif goal.goal_type == GoalType.ANALYSIS:
                return self._verify_analysis(criteria, goal.result_data)
            else:
                return self._verify_custom(criteria, goal.result_data)
                
        except Exception as e:
            logger.error(f"验证失败: {e}")
            return False, 0.0, str(e)
    
    def _verify_file_create(self, criteria: Dict) -> Tuple[bool, float, str]:
        """验证文件创建"""
        file_path = criteria.get("file_exists") or criteria.get("output_file")
        if not file_path:
            return False, 0.0, "未指定目标文件路径"
        
        full_path = os.path.join(self.base_path, file_path) if not os.path.isabs(file_path) else file_path
        
        if not os.path.exists(full_path):
            return False, 0.0, f"文件不存在: {file_path}"
        
        # 检查文件大小
        size = os.path.getsize(full_path)
        min_size = criteria.get("min_size", 0)
        if size < min_size:
            return False, 0.3, f"文件太小: {size} < {min_size} 字节"
        
        # 检查内容关键词
        contains_text = criteria.get("contains_text")
        if contains_text:
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if contains_text not in content:
                    return False, 0.5, f"文件不包含关键文本: {contains_text}"
            except:
                pass  # 二进制文件跳过
        
        return True, 1.0, f"文件创建成功: {file_path} ({size} 字节)"
    
    def _verify_file_modify(self, criteria: Dict) -> Tuple[bool, float, str]:
        """验证文件修改"""
        file_path = criteria.get("file_path")
        if not file_path:
            return False, 0.0, "未指定文件路径"
        
        full_path = os.path.join(self.base_path, file_path) if not os.path.isabs(file_path) else file_path
        
        # 检查修改时间
        expected_after = criteria.get("modified_after", 0)
        if os.path.exists(full_path):
            mtime = os.path.getmtime(full_path)
            if mtime > expected_after:
                return True, 1.0, f"文件已更新: {file_path}"
            else:
                return False, 0.3, "文件未被修改"
        
        return False, 0.0, f"文件不存在: {file_path}"
    
    def _verify_command(self, criteria: Dict) -> Tuple[bool, float, str]:
        """验证命令执行"""
        expected_exit_code = criteria.get("exit_code", 0)
        actual_exit_code = criteria.get("actual_exit_code")
        
        if actual_exit_code is None:
            return False, 0.0, "命令未执行"
        
        if actual_exit_code == expected_exit_code:
            return True, 1.0, f"命令执行成功 (exit code: {actual_exit_code})"
        else:
            return False, 0.3, f"命令返回非预期值: {actual_exit_code} != {expected_exit_code}"
    
    def _verify_observation(self, criteria: Dict, result: Dict) -> Tuple[bool, float, str]:
        """验证观察类目标"""
        # 观察类目标：只要有输出就算成功
        if result.get("observation") or result.get("vlm_result"):
            insight = result.get("observation", result.get("vlm_result", ""))
            score = min(1.0, len(insight) / 100)  # 根据洞察长度给分
            return True, score, f"观察完成，获得 {len(insight)} 字符洞察"
        
        return False, 0.0, "未获得有效观察结果"
    
    def _verify_analysis(self, criteria: Dict, result: Dict) -> Tuple[bool, float, str]:
        """验证分析类目标"""
        analysis = result.get("analysis", "") if isinstance(result, dict) else ""
        min_length = criteria.get("min_length", 50)

        if (not analysis) and isinstance(result, dict):
            candidate_files: List[str] = []
            if criteria.get("output_file"):
                candidate_files.append(str(criteria.get("output_file")))
            if result.get("output_file"):
                candidate_files.append(str(result.get("output_file")))
            if result.get("report_file"):
                candidate_files.append(str(result.get("report_file")))

            for p in candidate_files:
                if not p:
                    continue
                full_path = os.path.join(self.base_path, p) if not os.path.isabs(p) else p
                if not os.path.exists(full_path):
                    continue
                try:
                    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                        file_text = f.read()
                    if file_text:
                        analysis = file_text
                        break
                except Exception:
                    continue

        if len(analysis) >= min_length:
            # 检查是否包含必要关键词
            required_keywords = criteria.get("required_keywords", [])
            found = sum(1 for kw in required_keywords if kw in analysis)
            score = 0.5 + 0.5 * (found / max(len(required_keywords), 1))
            return True, score, f"分析完成 ({len(analysis)} 字符)"
        
        return False, 0.3, f"分析结果不足: {len(analysis)} < {min_length}"
    
    def _verify_custom(self, criteria: Dict, result: Dict) -> Tuple[bool, float, str]:
        """自定义验证"""
        # 检查自定义条件
        custom_check = criteria.get("custom_check")
        if custom_check and callable(custom_check):
            return custom_check(result)
        
        # 默认：有结果就算成功
        if result:
            return True, 0.7, "自定义目标已执行"
        return False, 0.0, "无执行结果"


class GoalManager:
    """
    目标管理器 - 管理目标栈的闭环系统
    
    核心改进：
    1. 超时检测
    2. 重试限制
    3. 完成验证
    4. 反馈闭环
    """
    
    def __init__(self, base_path: str = "."):
        self.goal_stack: List[VerifiableGoal] = []
        self.completed_goals: List[VerifiableGoal] = []
        self.failed_goals: List[VerifiableGoal] = []
        self.verifier = GoalVerifier(base_path)
        self.base_path = base_path
        
        # 统计
        self.stats = {
            "total_created": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_timeout": 0,
            "average_score": 0.0
        }
        
        # 持久化路径
        self.state_file = os.path.join(base_path, "data", "goal_state.json")
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
    
    def create_goal(
        self,
        description: str,
        goal_type: GoalType = GoalType.CUSTOM,
        success_criteria: Dict = None,
        priority: str = "medium",
        timeout: int = 60,
        max_attempts: int = 3,
        metadata: Dict = None  # 🆕 [2026-01-08] 支持传递系统状态上下文
    ) -> VerifiableGoal:
        """
        创建一个新的可验证目标
        
        Args:
            metadata: 🆕 可选的元数据，用于传递系统状态上下文
                - entropy: 当前系统熵值
                - curiosity: 当前好奇心水平
                - state_change_rate: 状态变化率
                - uncertainty: 不确定性
                (这些数据会被传递给 MetaCognition 用于复杂度评估)
        """
        import uuid
        
        goal = VerifiableGoal(
            id=str(uuid.uuid4())[:8],
            description=description,
            goal_type=goal_type,
            success_criteria=success_criteria or {},
            priority=priority,
            timeout_seconds=timeout,
            max_attempts=max_attempts
        )
        
        # 🆕 存储 metadata 到 result_data（临时方案，Goal 结构体没有 metadata 字段）
        if metadata:
            goal.result_data['_creation_metadata'] = metadata
            logger.debug(f"📊 Goal metadata: {metadata}")
        
        self.goal_stack.append(goal)
        self.stats["total_created"] += 1
        self._persist_state()
        
        logger.info(f"🎯 新目标创建: [{goal.id}] {description}")
        return goal
    
    def add_goal(self, *args, **kwargs):
        """兼容性接口: 映射到 create_goal"""
        logger.warning("⚠️ Deprecated 'add_goal' called. Redirecting to 'create_goal'.")
        return self.create_goal(*args, **kwargs)
    
    def get_current_goal(self) -> Optional[VerifiableGoal]:
        """获取当前最高优先级的活跃目标"""
        active_goals = [g for g in self.goal_stack if g.status in [GoalStatus.PENDING, GoalStatus.IN_PROGRESS]]
        if not active_goals:
            return None
        
        # 按优先级排序
        priority_order = {"highest": 0, "high": 1, "medium": 2, "low": 3}
        active_goals.sort(key=lambda g: priority_order.get(g.priority, 2))
        return active_goals[0]
    
    def start_goal(self, goal: VerifiableGoal):
        """开始执行目标"""
        goal.status = GoalStatus.IN_PROGRESS
        goal.started_at = time.time()
        goal.attempts += 1
        logger.info(f"▶️ 开始执行: [{goal.id}] {goal.description} (尝试 {goal.attempts}/{goal.max_attempts})")
        self._persist_state()
    
    def complete_goal(self, goal: VerifiableGoal, result_data: Dict = None):
        """完成目标并验证"""
        goal.result_data = result_data or {}
        goal.completed_at = time.time()
        
        # 验证目标
        is_success, score, message = self.verifier.verify(goal)
        goal.outcome_score = score
        
        if is_success:
            goal.status = GoalStatus.COMPLETED
            self.stats["total_completed"] += 1
            logger.info(f"✅ 目标完成: [{goal.id}] {message} (得分: {score:.2f})")
        else:
            # 检查是否还有重试机会
            if goal.attempts < goal.max_attempts:
                goal.status = GoalStatus.PENDING  # 重置为待执行
                goal.error_message = message
                logger.warning(f"⚠️ 目标未达成，将重试: [{goal.id}] {message}")
            else:
                goal.status = GoalStatus.FAILED
                goal.error_message = message
                self.stats["total_failed"] += 1
                logger.error(f"❌ 目标失败: [{goal.id}] {message}")
        
        # 移动到已完成/失败列表
        if goal.status in [GoalStatus.COMPLETED, GoalStatus.FAILED]:
            self.goal_stack = [g for g in self.goal_stack if g.id != goal.id]
            if goal.status == GoalStatus.COMPLETED:
                self.completed_goals.append(goal)
            else:
                self.failed_goals.append(goal)
        
        self._update_average_score()
        self._persist_state()
        
        return is_success, score, message
    
    def check_timeouts(self) -> List[VerifiableGoal]:
        """检查并处理超时的目标"""
        now = time.time()
        timed_out = []
        
        for goal in self.goal_stack:
            if goal.status == GoalStatus.IN_PROGRESS and goal.started_at:
                elapsed = now - goal.started_at
                if elapsed > goal.timeout_seconds:
                    goal.status = GoalStatus.TIMEOUT
                    goal.completed_at = now
                    goal.error_message = f"执行超时 ({elapsed:.1f}s > {goal.timeout_seconds}s)"
                    timed_out.append(goal)
                    self.stats["total_timeout"] += 1
                    logger.warning(f"⏰ 目标超时: [{goal.id}] {goal.description}")
        
        # 移动超时目标
        for goal in timed_out:
            self.goal_stack = [g for g in self.goal_stack if g.id != goal.id]
            self.failed_goals.append(goal)
        
        if timed_out:
            self._persist_state()
        
        return timed_out
    
    def fail_goal(self, goal: VerifiableGoal, reason: str = ""):
        """显式标记目标失败 (e.g. 安全拦截)"""
        goal.status = GoalStatus.FAILED
        goal.error_message = reason
        goal.completed_at = time.time()
        
        # 从堆栈中移除
        self.goal_stack = [g for g in self.goal_stack if g.id != goal.id]
        
        # 加入失败列表
        self.failed_goals.append(goal)
        self.stats["total_failed"] += 1
        
        logger.error(f"❌ 目标失败 (Explicit): [{goal.id}] {reason}")
        self._persist_state()

    def abandon_goal(self, goal: VerifiableGoal, reason: str = ""):
        """主动放弃目标"""
        goal.status = GoalStatus.ABANDONED
        goal.error_message = reason or "主动放弃"
        goal.completed_at = time.time()
        
        self.goal_stack = [g for g in self.goal_stack if g.id != goal.id]
        self.failed_goals.append(goal)
        
        logger.info(f"🚫 目标放弃: [{goal.id}] {reason}")
        self._persist_state()
    
    def get_feedback_for_motivation(self) -> Dict[str, float]:
        """
        获取用于动机系统的反馈数据
        
        Returns:
            {
                "recent_success_rate": 0.0-1.0,
                "recent_average_score": 0.0-1.0,
                "pending_count": int,
                "streak": int (连续成功/失败)
            }
        """
        recent = self.completed_goals[-10:] + self.failed_goals[-5:]
        if not recent:
            return {
                "recent_success_rate": 0.5,
                "recent_average_score": 0.5,
                "pending_count": len(self.goal_stack),
                "streak": 0
            }
        
        completed = [g for g in recent if g.status == GoalStatus.COMPLETED]
        success_rate = len(completed) / len(recent)
        avg_score = sum(g.outcome_score for g in completed) / max(len(completed), 1)
        
        # 计算连胜/连败
        streak = 0
        if self.completed_goals:
            for g in reversed(self.completed_goals):
                if g.status == GoalStatus.COMPLETED:
                    streak += 1
                else:
                    break
        
        return {
            "recent_success_rate": success_rate,
            "recent_average_score": avg_score,
            "pending_count": len([g for g in self.goal_stack if g.status == GoalStatus.PENDING]),
            "streak": streak
        }
    
    def _update_average_score(self):
        """更新平均分数"""
        all_completed = [g for g in self.completed_goals if g.outcome_score > 0]
        if all_completed:
            self.stats["average_score"] = sum(g.outcome_score for g in all_completed) / len(all_completed)
    
    def _persist_state(self):
        """持久化状态"""
        try:
            state = {
                "timestamp": time.time(),
                "stats": self.stats,
                "active_goals": [g.to_dict() for g in self.goal_stack],
                "recent_completed": [g.to_dict() for g in self.completed_goals[-10:]],
                "recent_failed": [g.to_dict() for g in self.failed_goals[-5:]]
            }
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"状态持久化失败: {e}")
    
    def get_status_summary(self) -> str:
        """获取状态摘要"""
        active = len([g for g in self.goal_stack if g.status in [GoalStatus.PENDING, GoalStatus.IN_PROGRESS]])
        return (
            f"📊 目标系统状态:\n"
            f"   活跃: {active} | 完成: {self.stats['total_completed']} | "
            f"失败: {self.stats['total_failed']} | 超时: {self.stats['total_timeout']}\n"
            f"   平均得分: {self.stats['average_score']:.2f}"
        )


# ========== 预定义工作任务模板 ==========

class WorkTemplates:
    """
    预定义的可执行工作模板
    解决 LLM 生成抽象目标的问题
    """
    
    @staticmethod
    def create_file_report(filename: str, topic: str) -> VerifiableGoal:
        """创建文件报告任务"""
        return VerifiableGoal(
            id=f"report_{int(time.time())}",
            description=f"生成关于 '{topic}' 的报告并保存到 {filename}",
            goal_type=GoalType.FILE_CREATE,
            success_criteria={
                "file_exists": filename,
                "min_size": 200
            },
            priority="medium",
            timeout_seconds=120,
            max_attempts=2
        )
    
    @staticmethod
    def observe_and_log(duration_seconds: int = 30) -> VerifiableGoal:
        """观察并记录任务"""
        log_file = f"observation_{int(time.time())}.txt"
        return VerifiableGoal(
            id=f"observe_{int(time.time())}",
            description=f"观察屏幕 {duration_seconds} 秒并记录到 {log_file}",
            goal_type=GoalType.FILE_CREATE,
            success_criteria={
                "file_exists": log_file,
                "min_size": 50
            },
            priority="low",
            timeout_seconds=duration_seconds + 30,
            max_attempts=1
        )
    
    @staticmethod
    def run_diagnostic() -> VerifiableGoal:
        """运行系统诊断任务"""
        return VerifiableGoal(
            id=f"diag_{int(time.time())}",
            description="运行系统自诊断并输出状态报告",
            goal_type=GoalType.FILE_CREATE,
            success_criteria={
                "file_exists": "data/logs/self_diagnostic.log",
                "modified_after": time.time()
            },
            priority="medium",
            timeout_seconds=60,
            max_attempts=1
        )
    
    @staticmethod
    def meta_cognitive_investigation(entropy: float, curiosity: float) -> VerifiableGoal:
        """
        元认知调查任务 - 调查高熵状态的根本原因
        
        🔧 [2026-01-11] 修复空转循环：绑定到产生实质证据的调查动作
        
        Args:
            entropy: 当前系统熵值
            curiosity: 当前好奇心水平
        
        Returns:
            带有明确验证标准的可验证目标
        """
        import time as _time

        ts = int(_time.time())
        report_path = f"data/entropy_investigation_{ts}.json"

        return VerifiableGoal(
            id=f"meta_inv_{ts}",
            description=(
                f"[Meta] Investigate high entropy state (Entropy: {entropy:.2f}, Curiosity: {curiosity:.2f}) | "
                f"Report: {report_path}"
            ),
            goal_type=GoalType.ANALYSIS,
            success_criteria={
                # 🆕 必须产生分析报告文件
                "output_file": report_path,
                "min_length": 200,
                # 🆕 必须包含以下关键词之一表明进行了实质分析
                "required_keywords": ["entropy_source", "memory_drift", "uncertainty_analysis", "root_cause"]
            },
            priority="high",
            timeout_seconds=120,
            max_attempts=1  # 🆕 不重试，避免循环
        )
    
    @staticmethod
    def analyze_file(file_path: str) -> VerifiableGoal:
        """分析文件任务"""
        return VerifiableGoal(
            id=f"analyze_{int(time.time())}",
            description=f"分析文件 {file_path} 并生成摘要",
            goal_type=GoalType.ANALYSIS,
            success_criteria={
                "min_length": 100
            },
            priority="medium",
            timeout_seconds=90,
            max_attempts=2
        )
    
    @staticmethod
    def user_command(command: str, output_file: str = None) -> VerifiableGoal:
        """用户命令任务"""
        criteria = {}
        goal_type = GoalType.CUSTOM
        
        if output_file:
            criteria["file_exists"] = output_file
            goal_type = GoalType.FILE_CREATE
        
        return VerifiableGoal(
            id=f"cmd_{int(time.time())}",
            description=f"执行用户指令: {command}",
            goal_type=goal_type,
            success_criteria=criteria,
            priority="highest",
            timeout_seconds=180,
            max_attempts=3
        )


# ========== 测试代码 ==========

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=" * 50)
    print("AGI Goal System - 闭环验证测试")
    print("=" * 50)
    
    # 创建管理器
    manager = GoalManager(base_path=".")
    
    # 测试1: 创建文件报告任务
    goal1 = WorkTemplates.create_file_report("test_output.md", "AGI系统状态")
    manager.goal_stack.append(goal1)
    
    # 模拟执行
    manager.start_goal(goal1)
    
    # 模拟完成（创建实际文件）
    with open("test_output.md", "w", encoding="utf-8") as f:
        f.write("# AGI系统状态报告\n\n这是一个测试报告，包含超过200字符的内容。" * 5)
    
    success, score, msg = manager.complete_goal(goal1)
    print(f"\n结果: success={success}, score={score}, msg={msg}")
    
    # 清理测试文件
    os.remove("test_output.md")
    
    # 打印统计
    print(manager.get_status_summary())
    print("\n反馈数据:", manager.get_feedback_for_motivation())
