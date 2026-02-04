import time
import json
import logging
from typing import List, Dict, Any, Optional, Deque
from collections import deque
from core.llm_client import LLMService

# 🆕 导入模式匹配器和决策缓存
try:
    from core.pattern_matcher import PatternMatcher, get_pattern_matcher
    from core.decision_cache import DecisionCache, get_decision_cache
except ImportError:
    PatternMatcher = None
    DecisionCache = None

# Configure logging
logger = logging.getLogger("IntentTracker")

class IntentTracker:
    """
    Tracks user actions and infers high-level intent using LLM analysis.
    Acts as the 'Subconscious Inference Engine' for the AGI.
    """
    def __init__(self, history_size: int = 20):
        if history_size <= 0:
            raise ValueError("history_size must be a positive integer")

        self.action_history: Deque[Dict[str, Any]] = deque(maxlen=history_size)
        self.llm: LLMService = LLMService()
        self.current_hypothesis: Optional[Dict[str, Any]] = None
        self.last_inference_time: float = 0.0
        self.inference_interval: float = 30.0  # Analyze every 30 seconds or when buffer fills
        self.min_actions_for_inference: int = 3

        # Context State
        self.active_application: str = "Unknown"
        self.visual_context: str = "None"

        # 🆕 [P0级优化] 集成模式匹配器和决策缓存
        self.enable_fast_intent = True  # 配置开关，可禁用快速路径
        self.pattern_matcher: Optional[PatternMatcher] = get_pattern_matcher() if PatternMatcher else None
        self.intent_cache: Optional[DecisionCache] = get_decision_cache(max_size=1000) if DecisionCache else None

        if self.pattern_matcher:
            logger.info("[IntentTracker] ✅ 模式匹配器已启用 (延迟<5ms)")
        if self.intent_cache:
            logger.info("[IntentTracker] ✅ 意图缓存已启用 (命中率目标>60%)")

        # 统计信息
        self.fast_path_hits = 0
        self.cache_hits = 0
        self.llm_calls = 0
        
    def add_observation(self, observation: Dict[str, Any]) -> None:
        """
        Ingest a new observation from any observer (CAD, Global, etc.).
        Validates input and enriches with metadata before storing.
        """
        if not isinstance(observation, dict):
            logger.warning("Observation must be a dictionary.")
            return

        timestamp: float = observation.get("timestamp", time.time())
        raw_text: Optional[str] = observation.get("text")
        summary: Optional[str] = observation.get("summary")
        action_text: str = raw_text or summary or "Unknown Action"
        
        # Enriched entry
        try:
            entry: Dict[str, Any] = {
                "timestamp": timestamp,
                "source": observation.get("type", "general"),
                "content": action_text,
                "details": observation.get("vlm_context", "")
            }
            self.action_history.append(entry)
        except Exception as e:
            logger.error(f"Failed to record observation: {e}")

    def update_context(self, app_name: Optional[str], visual_summary: Optional[str] = None) -> None:
        """Update global context info with optional fields."""
        if app_name is not None:
            if not isinstance(app_name, str):
                logger.warning("Application name must be a string.")
                return
            self.active_application = app_name
        if visual_summary is not None:
            if not isinstance(visual_summary, str):
                logger.warning("Visual summary must be a string.")
                return
            self.visual_context = visual_summary

    async def infer_intent(self) -> Optional[Dict[str, Any]]:
        """
        Analyze recent history to infer user intent.
        Returns the intent dictionary if a new insight is found.
        Uses rate limiting and structured error handling.

        🆕 [P0级优化] 快速路径：
        1. 模式匹配（< 5ms）- 50-100个常见意图
        2. 缓存检索（< 10ms）- 基于向量相似度
        3. LLM调用（< 2000ms）- 仅当快速路径失败
        """
        # 1. Check constraints
        current_time = time.time()
        if len(self.action_history) < self.min_actions_for_inference:
            return None

        time_since_last = current_time - self.last_inference_time
        # Only infer if enough time passed OR buffer is reasonably full
        if time_since_last < self.inference_interval and len(self.action_history) < self.action_history.maxlen * 0.7:
            return None

        self.last_inference_time = current_time

        # 🆕 [P0优化] 快速路径1：模式匹配（< 5ms）
        if self.enable_fast_intent and self.pattern_matcher:
            # 提取最近的文本内容
            recent_text = self._extract_recent_text()
            if recent_text:
                match_result = self.pattern_matcher.match(recent_text)
                if match_result and match_result.confidence >= 0.9:
                    self.fast_path_hits += 1
                    logger.info(
                        f"[IntentTracker] 🎯 模式匹配命中 "
                        f"(intent={match_result.intent}, "
                        f"confidence={match_result.confidence:.2f}, "
                        f"fast_path_hits={self.fast_path_hits})"
                    )
                    # 构造标准格式的返回结果
                    return {
                        "intent": match_result.intent,
                        "confidence": match_result.confidence,
                        "next_prediction": f"Based on pattern: {match_result.matched_pattern}",
                        "suggestion": f"Tool: {match_result.metadata.get('tool', 'unknown')}",
                        "source": "pattern_matcher",
                        "matched_pattern": match_result.matched_pattern
                    }

        # 🆕 [P0优化] 快速路径2：缓存检索（< 10ms）
        if self.enable_fast_intent and self.intent_cache:
            # 生成历史文本的embedding（简化版，使用文本hash）
            history_key = self._generate_history_key()
            if history_key:
                # 这里简化处理：使用历史文本的第一个观察作为缓存key
                # 实际应用中应该使用真实的embedding
                cached_result = self._check_cache(history_key)
                if cached_result:
                    self.cache_hits += 1
                    logger.info(
                        f"[IntentTracker] 💾 缓存命中 "
                        f"(intent={cached_result.get('intent')}, "
                        f"cache_hits={self.cache_hits})"
                    )
                    return cached_result

        # 🆕 [P0优化] 记录LLM调用
        self.llm_calls += 1
        
        # 2. Prepare Prompt
        try:
            history_str = "\n".join([
                f"- [{time.strftime('%H:%M:%S', time.localtime(e['timestamp']))}] ({e['source']}) {e['content']} {e['details']}"
                for e in self.action_history
            ])
        except Exception as e:
            logger.error(f"Failed to format action history: {e}")
            return None

        prompt = f"""
You are the 'Subconscious Intent Analyst' of an AGI system.
The user is currently working in: {self.active_application}.
Visual Context: {self.visual_context}

Recent User Actions:
{history_str}

Analyze this stream of behavior.
1. What is the user trying to achieve? (The high-level goal)
2. What pattern are they following?
3. Predict their next likely step.

Output a concise JSON object:
{{
  "intent": "High level goal description",
  "confidence": 0.0-1.0,
  "next_prediction": "Prediction of next action",
  "suggestion": "How can the system help? (Optional)"
}}
"""

        # 3. Call LLM
        try:
            response: str = self.llm.chat_completion(
                system_prompt="You are the 'Subconscious Intent Analyst' of an AGI system.",
                user_prompt=prompt,
                model=None  # Use default
            )
            
            if not response or not response.strip():
                logger.warning("Empty response from LLM during intent inference.")
                return None

            # 4. Parse Response
            clean_resp: str = response.strip()
            if clean_resp.startswith("{"):
                parsed: Dict[str, Any] = json.loads(clean_resp)
                self.current_hypothesis = parsed
                return parsed
            else:
                logger.warning(f"LLM response does not start with '{{'. Got: {clean_resp[:200]}")
                return None

        except json.JSONDecodeError as je:
            logger.error(f"Failed to parse LLM response as JSON: {je}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during intent inference: {e}")
            return None

    # 🆕 [P0级优化] 快速路径辅助方法

    def _extract_recent_text(self) -> Optional[str]:
        """提取最近的文本内容用于模式匹配"""
        if not self.action_history:
            return None

        # 获取最近的观察
        recent = list(self.action_history)[-1]
        return recent.get('content', '')

    def _generate_history_key(self) -> Optional[str]:
        """生成历史记录的唯一key（用于缓存）"""
        if not self.action_history:
            return None

        # 简化版：使用最近3个观察的内容hash
        recent_texts = [e.get('content', '') for e in list(self.action_history)[-3:]]
        combined = '|'.join(recent_texts)
        return combined[:200]  # 限制长度

    def _check_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """检查缓存（简化实现）"""
        # 这里简化处理：实际应该使用embedding向量相似度
        # 当前实现：仅在完全匹配时返回缓存
        if hasattr(self, '_cache_store') and key in self._cache_store:
            return self._cache_store[key]
        return None

    def _store_cache(self, key: str, result: Dict[str, Any]) -> None:
        """存储到缓存"""
        if not hasattr(self, '_cache_store'):
            self._cache_store = {}

        # 限制缓存大小
        if len(self._cache_store) >= 1000:
            # 删除最旧的条目
            oldest_key = next(iter(self._cache_store))
            del self._cache_store[oldest_key]

        self._cache_store[key] = result

    def get_fast_path_statistics(self) -> Dict[str, Any]:
        """获取快速路径统计信息"""
        total = self.fast_path_hits + self.cache_hits + self.llm_calls
        return {
            'fast_path_hits': self.fast_path_hits,
            'cache_hits': self.cache_hits,
            'llm_calls': self.llm_calls,
            'total_inferences': total,
            'fast_path_rate': self.fast_path_hits / total if total > 0 else 0.0,
            'llm_call_rate': self.llm_calls / total if total > 0 else 0.0
        }