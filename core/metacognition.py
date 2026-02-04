import os
import json
import logging
import time
import inspect
from datetime import datetime
from typing import Dict, Any, List

from core.llm_client import LLMService
# Import new cognitive metrics
try:
    from core.cognitive_metrics import (
        fractal_coherence_index, 
        detect_internal_resonance,
        calculate_metaphoric_drift,
        calculate_system_entropy
    )
except ImportError:
    # Fallback if file not created yet or dependencies missing
    def fractal_coherence_index(x): return 0.0
    def detect_internal_resonance(x): return 0.0
    def calculate_metaphoric_drift(x): return 0.0
    def calculate_system_entropy(x): return 0.0

logger = logging.getLogger(__name__)

# [2026-01-09] Phase2幻觉修复: 确定性验证器
class DeterministicValidator:
    """
    确定性验证器 - 使用硬编码规则验证LLM断言真实性
    防止元幻觉(LLM对自己的评估也可能是幻觉)
    """
    def __init__(self, tool_bridge=None):
        self.tool_bridge = tool_bridge
        self.validation_failures = []
    
    def validate_tool_call(self, tool_name: str, operation: str = None) -> Dict[str, Any]:
        """
        验证工具调用真实性
        返回: {"valid": bool, "reason": str, "evidence": Any}
        """
        result = {"valid": False, "reason": "未验证", "evidence": None}
        
        # 检查1: 工具是否在白名单中
        if self.tool_bridge:
            try:
                available_tools = self.tool_bridge.get_available_tools()
                if tool_name not in available_tools:
                    result["reason"] = f"工具'{tool_name}'不在白名单中(共{len(available_tools)}个可用工具)"
                    result["evidence"] = {"available": available_tools[:10]}  # 只返回前10个
                    self.validation_failures.append(result)
                    return result
                result["valid"] = True
                result["reason"] = f"工具存在于白名单({len(available_tools)}个工具中)"
                result["evidence"] = {"tool_name": tool_name}
            except Exception as e:
                result["reason"] = f"白名单检查异常: {e}"
                return result
        else:
            # 无tool_bridge时,使用硬编码白名单
            hardcoded_tools = ['file_operation', 'world_model', 'memory', 'openhands', 
                             'autonomous_document_create', 'knowledge_graph']
            if tool_name not in hardcoded_tools:
                result["reason"] = f"工具'{tool_name}'不在硬编码白名单中"
                result["evidence"] = {"hardcoded_whitelist": hardcoded_tools}
                self.validation_failures.append(result)
                return result
            result["valid"] = True
            result["reason"] = "工具在硬编码白名单中"
        
        return result
    
    def validate_numeric_sanity(self, value: Any, context: str = "") -> Dict[str, Any]:
        """
        验证数值合理性(检测NaN/Inf/超大值)
        返回: {"valid": bool, "reason": str, "evidence": Any}
        """
        import math
        result = {"valid": True, "reason": "数值正常", "evidence": value}
        
        try:
            if isinstance(value, (int, float)):
                if math.isnan(value):
                    result["valid"] = False
                    result["reason"] = f"{context}: 数值为NaN(Not a Number)"
                    self.validation_failures.append(result)
                elif math.isinf(value):
                    result["valid"] = False
                    result["reason"] = f"{context}: 数值为Infinity"
                    self.validation_failures.append(result)
                elif abs(value) > 1e10:
                    result["valid"] = False
                    result["reason"] = f"{context}: 数值过大(>{1e10})"
                    self.validation_failures.append(result)
        except Exception as e:
            result["valid"] = False
            result["reason"] = f"数值验证异常: {e}"
        
        return result
    
    def validate_file_operation_claim(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证文件操作断言(通过实际文件系统检查)
        claim格式: {"action": "created/deleted/modified", "path": "xxx", "content_hash": "xxx"}
        """
        result = {"valid": False, "reason": "未验证", "evidence": None}
        
        try:
            action = claim.get("action")
            path = claim.get("path")
            
            if not path:
                result["reason"] = "缺少文件路径"
                return result
            
            import os
            import hashlib
            
            if action == "created":
                # 验证文件是否存在
                if os.path.exists(path):
                    result["valid"] = True
                    result["reason"] = "文件确实存在"
                    result["evidence"] = {"exists": True, "size": os.path.getsize(path)}
                else:
                    result["reason"] = "声称创建但文件不存在"
                    result["evidence"] = {"exists": False}
                    self.validation_failures.append(result)
            
            elif action == "deleted":
                # 验证文件是否不存在
                if not os.path.exists(path):
                    result["valid"] = True
                    result["reason"] = "文件确实不存在"
                    result["evidence"] = {"exists": False}
                else:
                    result["reason"] = "声称删除但文件仍存在"
                    result["evidence"] = {"exists": True}
                    self.validation_failures.append(result)
            
            elif action == "modified" and claim.get("content_hash"):
                # 验证文件hash
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        actual_hash = hashlib.md5(f.read()).hexdigest()
                    expected_hash = claim.get("content_hash")
                    if actual_hash == expected_hash:
                        result["valid"] = True
                        result["reason"] = "文件内容hash匹配"
                        result["evidence"] = {"hash_match": True}
                    else:
                        result["reason"] = f"文件hash不匹配(期望:{expected_hash[:8]}, 实际:{actual_hash[:8]})"
                        result["evidence"] = {"expected": expected_hash, "actual": actual_hash}
                        self.validation_failures.append(result)
                else:
                    result["reason"] = "声称修改但文件不存在"
                    self.validation_failures.append(result)
        
        except Exception as e:
            result["reason"] = f"文件验证异常: {e}"
        
        return result
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """获取所有验证失败的摘要"""
        return {
            "total_failures": len(self.validation_failures),
            "failures": self.validation_failures[-10:],  # 只返回最近10个
            "failure_types": self._categorize_failures()
        }
    
    def _categorize_failures(self) -> Dict[str, int]:
        """分类统计失败类型"""
        categories = {}
        for failure in self.validation_failures:
            reason = failure.get("reason", "未知")
            # 简单分类
            if "工具" in reason or "白名单" in reason:
                key = "工具幻觉"
            elif "数值" in reason or "NaN" in reason or "Infinity" in reason:
                key = "数值异常"
            elif "文件" in reason:
                key = "文件操作幻觉"
            else:
                key = "其他"
            categories[key] = categories.get(key, 0) + 1
        return categories

class MetacognitiveCore:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        self.history = []
        self.last_reflection_time = 0
        self.reflection_history_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "metacognition_history.json")
        os.makedirs(os.path.dirname(self.reflection_history_path), exist_ok=True)
        # [2026-01-09] Phase2: 添加确定性验证器
        self.validator = DeterministicValidator()

    def _load_history(self) -> List[Dict]:
        try:
            with open(self.reflection_history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []

    def _save_history(self, history: List[Dict]):
        with open(self.reflection_history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def evaluate_self(self, recent_logs: List[str], goals_status: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a comprehensive self-evaluation based on the "Intelligence Function" model:
        Intelligence = f(Perception, Reasoning, Action, Learning, Evolution)
        """
        logger.info("🧠 Initiating Metacognitive Self-Evaluation...")
        
        # --- 1. Calculate Cognitive Metrics (FCI, Resonance) ---
        cognitive_state = {}
        try:
            # Extract timestamps from logs for FCI
            timestamps = []
            for log in recent_logs[-100:]: # Look at last 100 logs
                try:
                    # Assume log format: "YYYY-MM-DD HH:MM:SS..."
                    parts = log.split(' - ')
                    if len(parts) > 1:
                        ts_str = parts[0].strip().replace(',', '.')
                        if '.' in ts_str:
                            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                        else:
                            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                        timestamps.append(dt.timestamp())
                except Exception:
                    continue
            
            if timestamps:
                fci_score = fractal_coherence_index(timestamps)
                cognitive_state['fractal_coherence_index'] = fci_score
                cognitive_state['interpretation'] = "High (>0.85) implies insight-ready state; Low (<0.5) implies random noise."
                
                # Internal Resonance (Mock signal for now)
                cognitive_state['internal_resonance'] = detect_internal_resonance([
                    timestamps, 
                    [t + 0.1 for t in timestamps] 
                ])

                # New Metrics
                cognitive_state['metaphoric_drift'] = calculate_metaphoric_drift(recent_logs[-50:])
                cognitive_state['system_entropy'] = calculate_system_entropy(recent_logs[-50:])
                
        except Exception as e:
            logger.warning(f"Failed to calculate cognitive metrics: {e}")
            cognitive_state['error'] = str(e)

        # --- 2. Construct Prompt (With Token Limit Safeguards) ---
        # Truncate logs to avoid token overflow
        truncated_logs = recent_logs[-30:] # Reduce from 50 to 30 lines
        
        prompt = f"""
        You are the Metacognitive Module of an AGI system. Your job is to objectively evaluate the system's current intelligence level based on recent logs and goal status.

        Model: Intelligence = f(Adaptability, Learning Rate, Goal Achievement, Efficiency)
        
        NEW COGNITIVE METRICS (Derived from Internal Resonance):
        {json.dumps(cognitive_state, indent=2)}

        INPUT DATA:
        1. Recent Logs (Last 30 lines):
        {json.dumps(truncated_logs, ensure_ascii=False)}
        
        2. Goal Status:
        {json.dumps(goals_status, ensure_ascii=False)}

        TASK:
        Analyze the system's performance. 
        - Did it adapt to failures? (Adaptability)
        - Did it learn from new inputs? (Learning Rate)
        - Did it achieve its goals? (Goal Achievement)
        
        OUTPUT FORMAT (JSON ONLY):
        {{
            "intelligence_index": 0-100,
            "metrics": {{
                "adaptability": 0-10,
                "learning_rate": 0-10,
                "goal_achievement": 0-10,
                "efficiency": 0-10
            }},
            "qualitative_analysis": "Brief analysis of strengths and weaknesses observed.",
            "insight": "One profound insight for self-improvement.",
            "parameter_adjustments": {{
                "curiosity_delta": -0.1 to 0.1,
                "frustration_tolerance_delta": -0.1 to 0.1
            }},
            "self_improvement_directive": "A specific, actionable instruction for the coding agent to modify the codebase (e.g., 'Update AGI_Life_Engine.py to increase sleep time', 'Refactor desktop_automation.py'). If the intelligence_index is below 80, you MUST provide a directive to improve the system (e.g. 'Add detailed logging to AGI_Life_Engine.py' or 'Create a new test file')."
        }}
        """
        
        try:
            response = self.llm.chat_completion(
                system_prompt="You are a rigorous AGI evaluator. Be critical and objective.",
                user_prompt=prompt
            )
            
            # Clean and parse JSON
            cleaned_response = response.replace("```json", "").replace("```", "").strip()
            evaluation = json.loads(cleaned_response)
            
            # [2026-01-09] Phase2: 用确定性验证覆盖LLM乐观评估
            deterministic_overrides = {}
            
            # 检查validation failures
            failure_summary = self.validator.get_failure_summary()
            if failure_summary['total_failures'] > 0:
                logger.warning(f"⚠️ 检测到{failure_summary['total_failures']}个验证失败")
                
                # 降低intelligence_index(根据失败数量)
                original_index = evaluation.get('intelligence_index', 50)
                penalty = min(30, failure_summary['total_failures'] * 5)  # 每个失败扣5分,最多扣30分
                adjusted_index = max(0, original_index - penalty)
                
                deterministic_overrides['intelligence_index_override'] = {
                    'original': original_index,
                    'adjusted': adjusted_index,
                    'penalty': penalty,
                    'reason': f"检测到{failure_summary['total_failures']}个确定性验证失败"
                }
                
                # 覆盖原始评分
                evaluation['intelligence_index'] = adjusted_index
                
                # 添加失败类型分析
                deterministic_overrides['failure_analysis'] = failure_summary['failure_types']
                
                # 强制添加修复指令
                if not evaluation.get('self_improvement_directive'):
                    evaluation['self_improvement_directive'] = (
                        f"修复{failure_summary['total_failures']}个验证失败: "
                        f"{', '.join(failure_summary['failure_types'].keys())}"
                    )
            
            # Add timestamp and cognitive metrics
            evaluation["timestamp"] = datetime.now().isoformat()
            evaluation["cognitive_state"] = cognitive_state
            evaluation["deterministic_validation"] = deterministic_overrides  # 记录覆盖信息
            
            # Save to history
            history = self._load_history()
            history.append(evaluation)
            self._save_history(history)
            
            if deterministic_overrides:
                logger.info(f"🧠 Self-Evaluation Complete (Adjusted by Validator). Index: {evaluation.get('intelligence_index')} (原始: {deterministic_overrides.get('intelligence_index_override', {}).get('original', 'N/A')})")
            else:
                logger.info(f"🧠 Self-Evaluation Complete. Index: {evaluation.get('intelligence_index')}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Metacognitive evaluation failed: {e}")
            with open("meta_error.log", "w") as f:
                f.write(str(e))
            return {"error": str(e)}

    def generate_evolutionary_report(self) -> str:
        """
        Generate a human-readable report summarizing the system's cognitive evolution.
        """
        history = self._load_history()
        if not history:
            return "No metacognitive history available."
            
        latest = history[-1]
        
        report = f"""# 🧠 System Self-Evolution Report
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Intelligence Index**: {latest.get('intelligence_index', 'N/A')}/100

## 📊 Metrics
- **Adaptability**: {latest.get('metrics', {}).get('adaptability', 'N/A')}/10
- **Learning Rate**: {latest.get('metrics', {}).get('learning_rate', 'N/A')}/10
- **Goal Achievement**: {latest.get('metrics', {}).get('goal_achievement', 'N/A')}/10
- **Efficiency**: {latest.get('metrics', {}).get('efficiency', 'N/A')}/10

## 📝 Qualitative Analysis
{latest.get('qualitative_analysis', 'N/A')}

## ✨ Key Insight
> "{latest.get('insight', 'N/A')}"

## 🔧 Self-Adjustment
Proposed parameter adjustments for Motivation Core:
{json.dumps(latest.get('parameter_adjustments', {}), indent=2)}

## 🧬 Self-Evolution Directive
> **Code Modification Proposal**:
> {latest.get('self_improvement_directive', 'None')}

---
*Generated by MetacognitiveCore*
"""
        return report


# ============================================================================
# 新增: 深度元认知层 - 思维链延长与意识涌现支持
# ============================================================================

from dataclasses import dataclass, field
from collections import deque
import numpy as np
import hashlib


@dataclass
class ThoughtFrame:
    """
    思维帧 - 单个Tick的完整思维状态快照
    类比: 电影的一帧，记录了AGI在某一瞬间的完整认知状态
    """
    tick_id: int                          # Tick序号
    timestamp: float                      # 时间戳
    state_vector: np.ndarray             # 64维状态向量
    action_taken: int                     # 采取的动作
    action_name: str                      # 动作名称
    uncertainty: float                    # 不确定性
    thought_chain: List[str]             # 思维链
    extended_chain: List[str] = field(default_factory=list)  # 延长的思维链
    neural_confidence: float = 0.0        # 神经网络置信度
    context_hash: str = ""               # 上下文哈希
    meta_insights: List[str] = field(default_factory=list)   # 元洞察
    intentions: List[str] = field(default_factory=list)      # 当前活跃意图
    
    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "tick_id": self.tick_id,
            "timestamp": self.timestamp,
            "state_vector_hash": hashlib.md5(self.state_vector.tobytes()).hexdigest()[:8],
            "action": f"{self.action_name}({self.action_taken})",
            "uncertainty": round(self.uncertainty, 4),
            "confidence": round(self.neural_confidence, 4),
            "thought_chain_length": len(self.thought_chain),
            "extended_chain_length": len(self.extended_chain),
            "meta_insights": self.meta_insights,
            "intentions": self.intentions
        }


@dataclass
class Intention:
    """
    意图 - 跨Tick持久化的目标
    意图不是单一动作，而是多Tick持续追求的目标状态
    """
    id: str                               # 唯一标识
    description: str                      # 描述
    priority: float                       # 优先级 (0-1)
    created_tick: int                     # 创建时的Tick
    target_state: Any = None              # 目标状态向量
    progress: float = 0.0                 # 进度 (0-1)
    status: str = "active"               # active/completed/abandoned
    related_frames: List[int] = field(default_factory=list)  # 相关ThoughtFrame的tick_id


@dataclass
class MetaInsight:
    """
    元洞察 - 从思维模式中提取的高阶认识
    """
    insight_type: str                     # 洞察类型: pattern/anomaly/correlation/emergence
    description: str                      # 描述
    confidence: float                     # 置信度
    evidence_ticks: List[int] = field(default_factory=list)  # 支撑证据的Tick ID
    discovered_at: float = 0.0            # 发现时间


class MetaCognition:
    """
    深度元认知层 - AGI的自我观察与深度思维系统
    
    核心能力:
    1. observe(): 记录每个Tick的完整思维状态
    2. extend_thought_chain(): 延长思维链 (5→20步, 可配置15-25)
    3. register_intention(): 注册跨Tick持久意图
    4. detect_patterns(): 检测思维模式
    5. generate_meta_insights(): 生成元洞察
    
    架构位置:
        AGI_Life_Engine
             │
             ▼
        EvolutionController ◄──── MetaCognition
             │                         │
             ▼                         ▼
          TheSeed ────────────► TopologicalMemory
    """
    
    # 推理深度配置（🔧 [2026-01-20] 解除所有限制，支持无限思维成长）
    SHALLOW_HORIZON = 99999    # 简单任务（日常对话、单步工具）
    NORMAL_HORIZON = 99999     # 常规任务（中等推理、文档生成）
    DEEP_HORIZON = 99999      # 复杂任务（跨步骤规划、深度分析）
    ULTRA_DEEP_HORIZON = 99999 # 极端复杂任务（数学证明、架构设计）

    MIN_HORIZON = 99999         # 最小推理步数（无限制）
    MAX_HORIZON = 99999       # 最大推理步数（无限制思维成长）
    DEFAULT_HORIZON = 99999  # 默认使用无限深度
    
    # 历史窗口大小
    HISTORY_WINDOW = 100
    PATTERN_DETECTION_WINDOW = 10
    
    def __init__(self, seed_ref=None, memory_ref=None):
        """
        初始化元认知层
        
        Args:
            seed_ref: TheSeed实例的引用
            memory_ref: TopologicalMemory实例的引用
        """
        self.seed = seed_ref
        self.memory = memory_ref
        
        # 思维帧历史
        self.thought_frames: deque = deque(maxlen=self.HISTORY_WINDOW)
        
        # 意图注册表
        self.intentions: Dict[str, Intention] = {}
        
        # 元洞察库
        self.meta_insights: List[MetaInsight] = []
        
        # 模式检测缓存
        self._pattern_cache: Dict[str, int] = {}  # pattern_hash -> count
        
        # 当前Tick计数器
        self._tick_counter = 0
        
        # 配置
        self.current_horizon = self.DEFAULT_HORIZON
        
        # 🔧 新增：任务复杂度评估统计
        self.task_complexity_history: deque = deque(maxlen=50)  # 记录最近50次任务复杂度
        self.horizon_selection_stats = {
            'shallow': 0,
            'normal': 0, 
            'deep': 0,
            'ultra_deep': 0
        }
        
        logger.info(f"🧠 MetaCognition initialized with adaptive horizon (default={self.current_horizon})")
        logger.info(f"   - 推理深度档位: {self.SHALLOW_HORIZON}/{self.NORMAL_HORIZON}/{self.DEEP_HORIZON}/{self.ULTRA_DEEP_HORIZON}")
    
    # ========================================================================
    # 核心方法1: 自我观察
    # ========================================================================
    
    def observe(
        self, 
        state_vector: np.ndarray,
        action_taken: int,
        action_name: str,
        uncertainty: float,
        thought_chain: List[str],
        context: Dict[str, Any] = None,
        neural_confidence: float = 0.0
    ) -> ThoughtFrame:
        """
        观察并记录当前Tick的完整思维状态
        每次EvolutionController.step()调用后应立即调用此方法
        """
        self._tick_counter += 1
        
        # 计算上下文哈希
        context_hash = ""
        if context:
            context_str = json.dumps(context, sort_keys=True, default=str)
            context_hash = hashlib.md5(context_str.encode()).hexdigest()[:12]
        
        # 获取当前活跃意图
        active_intentions = [
            i.description for i in self.intentions.values() 
            if i.status == "active"
        ]
        
        # 创建思维帧
        frame = ThoughtFrame(
            tick_id=self._tick_counter,
            timestamp=datetime.now().timestamp(),
            state_vector=state_vector.copy() if state_vector is not None else np.zeros(64),
            action_taken=action_taken,
            action_name=action_name,
            uncertainty=uncertainty,
            thought_chain=thought_chain.copy() if thought_chain else [],
            neural_confidence=neural_confidence,
            context_hash=context_hash,
            intentions=active_intentions
        )
        
        # 保存到历史
        self.thought_frames.append(frame)
        
        # 触发模式检测
        if len(self.thought_frames) >= self.PATTERN_DETECTION_WINDOW:
            self._detect_patterns_async(frame)
        
        logger.debug(f"🔍 Observed Tick #{self._tick_counter}: action={action_name}, unc={uncertainty:.4f}")
        
        return frame
    
    # ========================================================================
    # 核心方法2: 延长思维链
    # ========================================================================
    
    def extend_thought_chain(
        self,
        start_state: np.ndarray,
        first_action: int,
        horizon: int = None,
        seed_ref = None
    ) -> tuple:
        """
        延长思维链 - 从5步扩展到15-25步
        这是意识涌现的关键：更长的思维链允许更深层次的推理
        
        Args:
            start_state: 起始状态向量
            first_action: 初始动作
            horizon: 思维深度 (默认20, 可配置15-25)
            seed_ref: TheSeed引用 (可选，使用内部引用)
            
        Returns:
            (extended_thoughts, trajectory): 延长的思维链和轨迹
        """
        seed = seed_ref or self.seed
        if not seed:
            logger.warning("⚠️ No TheSeed reference, cannot extend thought chain")
            return [], []
        
        # 确保horizon在合理范围内
        if horizon is None:
            horizon = self.current_horizon
        horizon = max(self.MIN_HORIZON, min(self.MAX_HORIZON, horizon))
        
        simulate_kwargs = {"horizon": horizon}
        try:
            sig = inspect.signature(seed.simulate_trajectory)
            if "adaptive" in sig.parameters:
                simulate_kwargs["adaptive"] = True
            if "max_horizon_extension" in sig.parameters:
                simulate_kwargs["max_horizon_extension"] = 30
        except Exception:
            pass
        try:
            trajectory = seed.simulate_trajectory(start_state, first_action, **simulate_kwargs)
        except TypeError:
            trajectory = seed.simulate_trajectory(start_state, first_action, horizon=horizon)
        
        # 将轨迹投影为思维链
        extended_thoughts = []
        cumulative_uncertainty = 0.0
        
        # 动作名称映射
        try:
            from core.evolution.impl import ACTIONS
        except ImportError:
            ACTIONS = ["explore", "exploit", "rest", "learn"]
        
        for i, (t_state, t_unc, t_act) in enumerate(trajectory):
            thought = seed.project_thought(t_state)
            act_name = ACTIONS[t_act % len(ACTIONS)]
            
            # 构建思维节点
            depth_marker = "." * min(i, 5)  # 深度标记
            uncertainty_marker = "?" if t_unc > 0.5 else ""
            
            thought_node = f"[D{i:02d}]{depth_marker}({act_name}) -> {thought}{uncertainty_marker}"
            extended_thoughts.append(thought_node)
            
            cumulative_uncertainty += t_unc
        
        # 生成元观察
        avg_uncertainty = cumulative_uncertainty / len(trajectory) if trajectory else 0
        
        if avg_uncertainty > 0.7:
            extended_thoughts.append(f"[META] 高不确定性区域 (avg_unc={avg_uncertainty:.3f}), 需要更多信息")
        elif avg_uncertainty < 0.3:
            extended_thoughts.append(f"[META] 高置信度路径 (avg_unc={avg_uncertainty:.3f}), 可靠推理")
        else:
            extended_thoughts.append(f"[META] 中等置信度 (avg_unc={avg_uncertainty:.3f}), 继续观察")
        
        logger.info(f"🔗 Extended thought chain: {len(extended_thoughts)} steps (horizon={horizon})")
        
        return extended_thoughts, trajectory
    
    # ========================================================================
    # 核心方法3: 意图注册与管理
    # ========================================================================
    
    def register_intention(
        self,
        description: str,
        priority: float = 0.5,
        target_state: np.ndarray = None
    ) -> Intention:
        """注册跨Tick持久意图"""
        intention_id = hashlib.md5(
            f"{description}_{self._tick_counter}".encode()
        ).hexdigest()[:8]
        
        intention = Intention(
            id=intention_id,
            description=description,
            priority=max(0.0, min(1.0, priority)),
            created_tick=self._tick_counter,
            target_state=target_state.copy() if target_state is not None else None
        )
        
        self.intentions[intention_id] = intention
        logger.info(f"📌 Registered intention: {description} (id={intention_id})")
        return intention
    
    def update_intention_progress(self, intention_id: str, progress: float, frame_tick: int = None):
        """更新意图进度"""
        if intention_id in self.intentions:
            intention = self.intentions[intention_id]
            intention.progress = max(0.0, min(1.0, progress))
            if frame_tick:
                intention.related_frames.append(frame_tick)
            if intention.progress >= 1.0:
                intention.status = "completed"
                logger.info(f"✅ Intention completed: {intention.description}")
    
    def get_active_intentions(self) -> List[Intention]:
        """获取所有活跃意图"""
        return [i for i in self.intentions.values() if i.status == "active"]
    
    # ========================================================================
    # 核心方法4: 模式检测
    # ========================================================================
    
    def _detect_patterns_async(self, current_frame: ThoughtFrame):
        """异步检测思维模式"""
        recent_actions = [f.action_taken for f in list(self.thought_frames)[-self.PATTERN_DETECTION_WINDOW:]]
        
        if len(recent_actions) < 3:
            return
        
        # 检测重复模式 (长度2-4的循环)
        for pattern_len in range(2, min(5, len(recent_actions) // 2 + 1)):
            pattern = tuple(recent_actions[-pattern_len:])
            prev_pattern = tuple(recent_actions[-2*pattern_len:-pattern_len]) if len(recent_actions) >= 2*pattern_len else None
            
            if prev_pattern and pattern == prev_pattern:
                pattern_hash = str(pattern)
                self._pattern_cache[pattern_hash] = self._pattern_cache.get(pattern_hash, 0) + 1
                
                if self._pattern_cache[pattern_hash] >= 3:
                    self._add_meta_insight(
                        insight_type="pattern",
                        description=f"检测到重复思维循环: 动作序列 {pattern} 重复了 {self._pattern_cache[pattern_hash]} 次",
                        confidence=0.8,
                        evidence_ticks=[f.tick_id for f in list(self.thought_frames)[-2*pattern_len:]]
                    )
    
    def detect_entropy_lock(self):
        """检测熵锁定状态 - 系统陷入低变化状态"""
        if len(self.thought_frames) < 5:
            return None
        
        recent_frames = list(self.thought_frames)[-5:]
        uncertainties = [f.uncertainty for f in recent_frames]
        
        if max(uncertainties) < 0.1:
            unique_actions = len(set(f.action_taken for f in recent_frames))
            if unique_actions <= 2:
                return self._add_meta_insight(
                    insight_type="anomaly",
                    description="熵锁定警告: 系统变化极低，可能需要外部刺激",
                    confidence=0.9,
                    evidence_ticks=[f.tick_id for f in recent_frames]
                )
        return None
    
    # ========================================================================
    # 核心方法5: 元洞察生成
    # ========================================================================
    
    def _add_meta_insight(
        self,
        insight_type: str,
        description: str,
        confidence: float,
        evidence_ticks: List[int]
    ) -> MetaInsight:
        """添加元洞察"""
        insight = MetaInsight(
            insight_type=insight_type,
            description=description,
            confidence=confidence,
            evidence_ticks=evidence_ticks,
            discovered_at=datetime.now().timestamp()
        )
        self.meta_insights.append(insight)
        logger.info(f"💡 Meta-Insight [{insight_type}]: {description}")
        return insight
    
    def generate_introspection_report(self) -> Dict[str, Any]:
        """生成内省报告 - AGI的自我分析"""
        recent_frames = list(self.thought_frames)[-10:]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_ticks": self._tick_counter,
            "current_horizon": self.current_horizon,
            "statistics": {
                "frames_recorded": len(self.thought_frames),
                "active_intentions": len(self.get_active_intentions()),
                "total_intentions": len(self.intentions),
                "meta_insights_count": len(self.meta_insights),
                "pattern_cache_size": len(self._pattern_cache)
            },
            "recent_state": {
                "avg_uncertainty": float(np.mean([f.uncertainty for f in recent_frames])) if recent_frames else 0,
                "avg_confidence": float(np.mean([f.neural_confidence for f in recent_frames])) if recent_frames else 0,
                "action_diversity": len(set(f.action_taken for f in recent_frames)) / max(1, len(recent_frames)),
                "last_actions": [f.action_name for f in recent_frames[-5:]]
            },
            "active_intentions": [
                {"id": i.id, "description": i.description, "progress": i.progress, "priority": i.priority}
                for i in self.get_active_intentions()
            ],
            "recent_insights": [
                {"type": i.insight_type, "description": i.description, "confidence": i.confidence}
                for i in self.meta_insights[-5:]
            ]
        }
    
    # ========================================================================
    # 配置方法
    # ========================================================================
    
    def set_horizon(self, horizon: int):
        """设置思维链深度"""
        self.current_horizon = max(self.MIN_HORIZON, min(self.MAX_HORIZON, horizon))
        logger.info(f"⚙️ Horizon updated to {self.current_horizon}")
    
    def set_seed_reference(self, seed_ref):
        """设置TheSeed引用"""
        self.seed = seed_ref
    
    def set_memory_reference(self, memory_ref):
        """设置Memory引用"""
        self.memory = memory_ref
    
    # ========================================================================
    # 持久化方法
    # ========================================================================
    
    def save_state(self, filepath: str):
        """保存元认知状态到文件"""
        state = {
            "tick_counter": self._tick_counter,
            "current_horizon": self.current_horizon,
            "intentions": {
                k: {
                    "id": v.id,
                    "description": v.description,
                    "priority": v.priority,
                    "created_tick": v.created_tick,
                    "progress": v.progress,
                    "status": v.status,
                    "related_frames": v.related_frames
                }
                for k, v in self.intentions.items()
            },
            "meta_insights": [
                {
                    "insight_type": i.insight_type,
                    "description": i.description,
                    "confidence": i.confidence,
                    "evidence_ticks": i.evidence_ticks,
                    "discovered_at": i.discovered_at
                }
                for i in self.meta_insights
            ],
            "pattern_cache": self._pattern_cache
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 MetaCognition state saved to {filepath}")
    
    # ========================================================================
    # 🔧 新增方法: 任务复杂度评估与自适应推理深度
    # ========================================================================
    
    def _estimate_complexity(self, task_descriptor: str, context: Dict[str, Any] = None) -> float:
        """
        评估任务复杂度（0.0-1.0）
        
        🔧 [2026-01-08] 紧急修复：元认知任务优先检测
        - **问题**: 内部探索任务（如"Investigate high entropy state"）无法识别
        - **症状**: 复杂度=0.05 → horizon=10（应为80+）→ 推理深度不足警告循环
        - **实际状态**: Entropy=1.00, State_change_rate=217, Uncertainty=60
        - **修复策略**: 从任务描述提取系统状态指标，优先返回高复杂度
        
        综合考虑:
        1. 🆕 元认知任务优先检测（0.75-0.95） - **最高优先级**
        2. 文本长度和嵌套结构 (0-0.25)
        3. 关键词复杂度（如"证明"、"设计"）(0-0.35)
        4. 上下文信息（子任务数量、依赖深度）(0-0.40)
        
        Returns:
            0.0-0.25: 简单任务 → SHALLOW_HORIZON=99999
            0.25-0.55: 常规任务 → NORMAL_HORIZON=99999
            0.55-0.80: 复杂任务 → DEEP_HORIZON=99999
            0.80-1.0: 极端复杂任务 → ULTRA_DEEP_HORIZON=99999

        Note: 所有HORIZON常量均设置为99999以支持无限深度推理，
              实际深度由simulate_trajectory的早停机制控制（现已优化）。
        """
        complexity = 0.0
        context = context or {}
        task_lower = task_descriptor.lower()


        # ========================================================================
        # 🆕 [2026-01-09] EMERGENCY FIX: 系统状态阈值检查（优先级最高）
        # 即使任务描述不是元认知关键词，只要系统状态异常，也应提升推理深度
        # ========================================================================
        if context:
            ctx_entropy = context.get('entropy', 0.0)
            ctx_curiosity = context.get('curiosity', 0.0)
            ctx_state_change = context.get('state_change_rate', 0.0)
            ctx_uncertainty = context.get('uncertainty', 0.0)

            # 检测系统异常状态 - 无论任务描述如何，都应提升深度
            if ctx_state_change > 150 or ctx_entropy > 0.9:
                logger.warning(f"⚠️ 检测到系统异常状态: StateChange={ctx_state_change:.1f}, Entropy={ctx_entropy:.2f} → ULTRA_DEEP")
                return 0.95  # → horizon=2000
            elif ctx_state_change > 100 or ctx_entropy > 0.7:
                logger.info(f"🔬 检测到系统高变化率: StateChange={ctx_state_change:.1f}, Entropy={ctx_entropy:.2f} → ULTRA_DEEP")
                return 0.85  # → horizon=2000
            elif ctx_uncertainty > 60 or ctx_curiosity > 0.8:
                logger.info(f"🤔 检测到高不确定性: Uncertainty={ctx_uncertainty:.1f}, Curiosity={ctx_curiosity:.2f} → DEEP")
                return 0.70  # → horizon=1000

        # ========================================================================
        # 🆕 **CRITICAL FIX**: 元认知/内部探索任务检测
        # ========================================================================

        # 1. 检测高熵/混沌状态探索关键词
        meta_cognitive_keywords = [
            'entropy', '熵', 'investigate', '调查', 'explore', '探索',
            'curiosity', '好奇', 'high.*state', '高.*状态',
            'fractal', '分形', 'anomaly', '异常', 'chaos', '混沌',
            'inspect', '检查', 'analyze.*state', '分析.*状态',
            'monitor', '监控', 'diagnostic', '诊断'
        ]

        is_meta_task = any(kw in task_lower for kw in meta_cognitive_keywords)

        # 2. 检测数值指标（如 "Curiosity: 0.68", "Entropy: 1.00"）
        import re
        has_metrics = bool(re.search(r'(entropy|curiosity|uncertainty)[:=]?\s*\d+\.\d+', task_lower))

        # 3. 检测任务类型前缀标记
        is_marked_internal = task_descriptor.startswith('[Meta]') or task_descriptor.startswith('[Internal]')

        # 4. 如果是元认知任务，直接返回高复杂度
        if is_meta_task or has_metrics or is_marked_internal:
            logger.info(f"🧠 检测到元认知任务: '{task_descriptor[:60]}...'")

            # 尝试从任务描述中提取数值指标
            curiosity_match = re.search(r'curiosity[:=]?\s*(\d+\.\d+)', task_lower)
            entropy_match = re.search(r'entropy[:=]?\s*(\d+\.\d+)', task_lower)

            extracted_curiosity = float(curiosity_match.group(1)) if curiosity_match else 0.0
            extracted_entropy = float(entropy_match.group(1)) if entropy_match else 0.0

            # 同时检查 context 中的系统状态
            ctx_entropy = context.get('entropy', extracted_entropy) if context else extracted_entropy
            ctx_curiosity = context.get('curiosity', extracted_curiosity) if context else extracted_curiosity
            ctx_state_change = context.get('state_change_rate', 0.0) if context else 0.0
            ctx_uncertainty = context.get('uncertainty', 0.0) if context else 0.0

            # 根据提取的指标或默认值返回复杂度
            if ctx_entropy > 0.9 or ctx_curiosity > 0.8 or ctx_state_change > 150:
                logger.warning(f"🚨 超高熵任务: Entropy≈{ctx_entropy:.2f}, Curiosity≈{ctx_curiosity:.2f}, StateChange≈{ctx_state_change:.1f} → ULTRA_DEEP")
                return 0.95  # → horizon=2000
            elif ctx_entropy > 0.7 or ctx_curiosity > 0.6 or ctx_state_change > 100:
                logger.info(f"🔬 高熵探索任务: Entropy≈{ctx_entropy:.2f}, Curiosity≈{ctx_curiosity:.2f} → ULTRA_DEEP")
                return 0.85  # → horizon=2000
            elif ctx_curiosity > 0.4 or ctx_uncertainty > 50:
                logger.info(f"🤔 中等好奇心任务: Curiosity≈{ctx_curiosity:.2f}, Uncertainty≈{ctx_uncertainty:.1f} → DEEP")
                return 0.70  # → horizon=1000
            else:
                # 默认：所有元认知任务至少 DEEP
                logger.info(f"🧠 元认知任务（默认）→ DEEP")
                return 0.65  # → horizon=1000

        # ========================================================================
        # 原有逻辑：用户面向任务的复杂度评估
        # ========================================================================

        # 1. 文本复杂度 (0-0.25)
        text_len = len(task_descriptor)
        if text_len > 500:
            complexity += 0.25
        elif text_len > 200:
            complexity += 0.18
        elif text_len > 80:
            complexity += 0.10
        elif text_len > 30:
            complexity += 0.05

        # 2. 关键词复杂度 (0-0.35) - 🔧 [2026-01-09] 分离关键词避免重复

        # 🆕 [2026-01-09] 超高复杂度关键词 (权重0.30) - 独立关键词
        ultra_complexity_keywords = [
            'mathematical', '数学的',
            'validity', '有效性', '合法性',
            'tradeoff', 'trade-off', 'tradeoffs', 'trade-offs'
        ]

        # 高复杂度关键词 (权重0.15)
        high_complexity_keywords = [
            '证明', 'proof', 'prove',
            '设计', 'design',
            '架构', 'architecture',
            '规划', 'planning',
            '优化', 'optimize',
            '分析', 'analyze',
            '推导', 'derive',
            '综合', 'synthesis',
            '重构', 'refactor',
            '分布式', 'distributed',
            '协议', 'protocol',
            '比较', 'compare', '对比', '权衡',
            '猜想', 'conjecture',
            '定理', 'theorem',
            '公式', 'formula',
            '方程', 'equation',
            '推理', 'inference',
            '逻辑', 'logic',
            '算法', 'algorithm',
            '模型', 'model',
            '缓存', 'cache',
            '验证', 'verify',
            '推导', 'deduce',
            '分形', 'fractal',
            '异常', 'anomaly'
        ]

        # 中等复杂度关键词 (权重0.08)
        medium_complexity_keywords = [
            '评估', 'evaluate',
            '总结', 'summarize',
            '集成', 'integrate',
            '修复', 'fix',
            '调试', 'debug',
            '生成', 'generate',
            '创建', 'create',
            '排序', 'sort'
        ]
        
        keyword_score = 0.0
        
        # 🆕 [2026-01-09] 超高复杂度关键词优先检测 (权重0.30)
        for kw in ultra_complexity_keywords:
            if kw in task_lower:
                keyword_score += 0.30  # 超高权重 → 确保达到complexity≥0.55
        
        # 高复杂度关键词 (权重0.15)
        for kw in high_complexity_keywords:
            if kw in task_lower:
                keyword_score += 0.15
        
        # 中等复杂度关键词 (权重0.08)
        for kw in medium_complexity_keywords:
            if kw in task_lower:
                keyword_score += 0.08
        
        # 🔧 [2026-01-09] 提高关键词复杂度上限: 0.35 → 0.70
        # 原因: 复杂任务可能同时包含多个高复杂度关键词
        complexity += min(keyword_score, 0.70)
        
        # 3. 上下文信息 (0-0.40) - 显著提高权重
        if context:
            subtask_count = context.get('subtask_count', 0)
            dependency_depth = context.get('dependency_depth', 0)
            uncertainty = context.get('uncertainty', 0.0)
            novelty = context.get('novelty', 0.0)
            
            # 子任务数量影响
            if subtask_count > 12:
                complexity += 0.15
            elif subtask_count > 8:
                complexity += 0.12
            elif subtask_count > 5:
                complexity += 0.08
            elif subtask_count > 2:
                complexity += 0.04
            
            # 依赖深度影响
            if dependency_depth > 4:
                complexity += 0.15
            elif dependency_depth > 2:
                complexity += 0.10
            elif dependency_depth > 1:
                complexity += 0.05
            
            # 不确定性和新颖性
            if uncertainty > 0.7:
                complexity += 0.05
            if novelty > 0.8:
                complexity += 0.05
        
        return min(complexity, 1.0)
    
    def auto_select_horizon(self, task_descriptor: str, context: Dict[str, Any] = None) -> int:
        """
        根据任务描述自动选择推理深度
        
        Args:
            task_descriptor: 任务描述文本
            context: 额外上下文信息（可选）
                - subtask_count: 子任务数量
                - dependency_depth: 依赖深度
                - uncertainty: 不确定性
                - novelty: 任务新颖度
        
        Returns:
            推荐的推理深度 (所有档位均为99999，实际深度由早停机制控制)
        """
        complexity = self._estimate_complexity(task_descriptor, context)
        
        # 记录复杂度历史
        self.task_complexity_history.append({
            'task': task_descriptor[:100],  # 前100字符
            'complexity': complexity,
            'timestamp': datetime.now().isoformat()
        })
        
        # 根据复杂度选择深度 (调整后的阈值)
        if complexity < 0.25:  # 简单任务
            selected_horizon = self.SHALLOW_HORIZON
            tier = 'shallow'
        elif complexity < 0.55:  # 常规任务
            selected_horizon = self.NORMAL_HORIZON
            tier = 'normal'
        elif complexity < 0.80:  # 复杂任务
            selected_horizon = self.DEEP_HORIZON
            tier = 'deep'
        else:  # 极端复杂任务
            selected_horizon = self.ULTRA_DEEP_HORIZON
            tier = 'ultra_deep'
        
        # 更新统计
        self.horizon_selection_stats[tier] += 1
        
        logger.info(f"🎯 任务复杂度: {complexity:.3f} → 推理深度: {selected_horizon} ({tier})")
        logger.debug(f"   任务: {task_descriptor[:80]}...")
        
        return selected_horizon
    
    def get_complexity_stats(self) -> Dict[str, Any]:
        """获取复杂度评估统计"""
        if not self.task_complexity_history:
            return {
                'total_tasks': 0,
                'avg_complexity': 0.0,
                'horizon_distribution': self.horizon_selection_stats
            }
        
        complexities = [t['complexity'] for t in self.task_complexity_history]
        return {
            'total_tasks': len(self.task_complexity_history),
            'avg_complexity': np.mean(complexities),
            'min_complexity': np.min(complexities),
            'max_complexity': np.max(complexities),
            'std_complexity': np.std(complexities),
            'horizon_distribution': self.horizon_selection_stats.copy(),
            'recent_tasks': list(self.task_complexity_history)[-5:]  # 最近5个任务
        }
    
    def load_state(self, filepath: str):
        """从文件加载元认知状态"""
        if not os.path.exists(filepath):
            logger.warning(f"⚠️ State file not found: {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self._tick_counter = state.get("tick_counter", 0)
        self.current_horizon = state.get("current_horizon", self.DEFAULT_HORIZON)
        
        for k, v in state.get("intentions", {}).items():
            self.intentions[k] = Intention(
                id=v["id"],
                description=v["description"],
                priority=v["priority"],
                created_tick=v["created_tick"],
                target_state=None,
                progress=v["progress"],
                status=v["status"],
                related_frames=v["related_frames"]
            )
        
        for i in state.get("meta_insights", []):
            self.meta_insights.append(MetaInsight(
                insight_type=i["insight_type"],
                description=i["description"],
                confidence=i["confidence"],
                evidence_ticks=i["evidence_ticks"],
                discovered_at=i["discovered_at"]
            ))
        
        self._pattern_cache = state.get("pattern_cache", {})
        logger.info(f"📂 MetaCognition state loaded from {filepath}")


def create_metacognition(seed_ref=None, memory_ref=None, horizon: int = 99999) -> MetaCognition:
    """创建元认知实例的工厂函数"""
    mc = MetaCognition(seed_ref, memory_ref)
    mc.set_horizon(horizon)
    return mc

