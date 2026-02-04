import json
import platform
import hashlib
from typing import List, Dict, Any, Optional, Set
from .base_agent import BaseAgent

# 🆕 [P0级优化] 导入规划缓存
try:
    from core.decision_cache import get_decision_cache
except ImportError:
    get_decision_cache = None

class PlannerAgent(BaseAgent):
    """
    Role: Strategist
    Responsibility: Break down high-level user goals into atomic, executable steps.
    Output: A list of JSON-formatted tasks or simple text steps.
    
    [2026-01-17] 解除硬编码步数限制，支持自适应规划深度
    [2026-01-17] 移除硬性上限，规划器自主决定推理层级数量
    [2026-01-18] 添加动态工具感知能力，可感知运行时创建的新工具
    """
    # 自适应规划步数配置 - 无硬性上限，规划器自主决策
    MIN_PLAN_STEPS = 3        # 最小步数（简单任务）
    DEFAULT_PLAN_STEPS = 10   # 默认步数（常规任务）
    DEEP_PLAN_STEPS = 25      # 深度规划步数（复杂任务）
    ULTRA_DEEP_STEPS = 50     # 超深度规划（高复杂度任务）
    MAX_PLAN_STEPS = 999      # 理论上限（实际由规划器自主决定，无硬性限制）
    
    def __init__(self, llm_service, biological_memory=None, event_bus=None, tool_registry=None):
        super().__init__("Planner", llm_service)
        self.biological_memory = biological_memory
        self._adaptive_max_steps = self.DEFAULT_PLAN_STEPS  # 当前自适应步数

        # 🆕 [2026-01-18] 动态工具感知能力
        self._tool_registry = tool_registry
        self._dynamic_tools: Set[str] = set()  # 运行时创建的工具名称
        self._event_bus = event_bus

        # 🆕 [P0级优化] 规划结果缓存
        self.planning_cache = {}  # {task_hash: steps}
        self.cache_hits = 0
        self.cache_misses = 0
        self.enable_planning_cache = True  # 配置开关

        # 订阅工具创建事件
        if event_bus:
            self._subscribe_to_tool_events(event_bus)
    
    def _subscribe_to_tool_events(self, event_bus):
        """订阅工具创建事件，感知新工具"""
        try:
            if hasattr(event_bus, 'subscribe'):
                event_bus.subscribe('tool.created', self._on_tool_created)
                event_bus.subscribe('autonomy.tool_created', self._on_tool_created)
                print("   [Planner] 🔔 Subscribed to tool creation events")
        except Exception as e:
            print(f"   [Planner] ⚠️ Failed to subscribe to events: {e}")
    
    def _on_tool_created(self, event):
        """处理工具创建事件"""
        try:
            data = event.data if hasattr(event, 'data') else event
            tool_name = data.get('tool_name', '')
            if tool_name:
                self._dynamic_tools.add(tool_name)
                print(f"   [Planner] 🔧 New tool available: {tool_name}")
        except Exception as e:
            print(f"   [Planner] ⚠️ Error handling tool event: {e}")
    
    def get_available_tools(self) -> List[str]:
        """获取所有可用工具（包括动态创建的）"""
        base_tools = [
            'read_file', 'write_file', 'run_python', 'run_command',
            'execute_macro', 'wait', 'log'
        ]
        
        # 添加从 registry 获取的工具
        if self._tool_registry and hasattr(self._tool_registry, 'list_tools'):
            registry_tools = self._tool_registry.list_tools()
            base_tools.extend(registry_tools)
        
        # 添加动态创建的工具
        base_tools.extend(list(self._dynamic_tools))
        
        return list(set(base_tools))  # 去重

    def _estimate_task_complexity(self, text: str) -> float:
        """
        估算任务复杂度 (0.0 - 1.0)
        用于动态决定规划步数上限
        """
        if not text:
            return 0.2
        
        lower = text.lower()
        complexity = 0.3  # 基础复杂度
        
        # 复杂度指标
        complexity_indicators = {
            # 高复杂度关键词
            "high_complexity": [
                "research", "analyze", "investigate", "design", "architect",
                "调查", "研究", "分析", "设计", "架构", "实现", "implement",
                "multi-step", "多步骤", "复杂", "complex", "comprehensive",
                "deep", "深度", "全面", "系统性", "systematic"
            ],
            # 中等复杂度
            "medium_complexity": [
                "create", "build", "develop", "test", "debug",
                "创建", "构建", "开发", "测试", "调试", "修复", "fix"
            ],
            # 低复杂度
            "low_complexity": [
                "read", "list", "check", "log", "observe",
                "读取", "列出", "检查", "记录", "观察"
            ]
        }
        
        for keyword in complexity_indicators["high_complexity"]:
            if keyword in lower:
                complexity += 0.15
        
        for keyword in complexity_indicators["medium_complexity"]:
            if keyword in lower:
                complexity += 0.08
        
        for keyword in complexity_indicators["low_complexity"]:
            if keyword in lower:
                complexity -= 0.05
        
        # 文本长度影响
        if len(text) > 500:
            complexity += 0.2
        elif len(text) > 200:
            complexity += 0.1
        
        # 多子任务检测（逗号、分号、"和"、"然后"等）
        multi_task_markers = ["，", ",", ";", "；", "然后", "接着", "and then", "finally"]
        for marker in multi_task_markers:
            if marker in text:
                complexity += 0.1
                break
        
        return max(0.0, min(1.0, complexity))

    def _get_adaptive_max_steps(self, text: str) -> int:
        """
        根据任务复杂度动态返回最大步数
        规划器自主决定推理深度，无硬性上限
        """
        complexity = self._estimate_task_complexity(text)
        
        if complexity < 0.3:
            max_steps = self.MIN_PLAN_STEPS
        elif complexity < 0.5:
            max_steps = self.DEFAULT_PLAN_STEPS
        elif complexity < 0.7:
            max_steps = self.DEEP_PLAN_STEPS
        elif complexity < 0.85:
            max_steps = self.ULTRA_DEEP_STEPS
        else:
            # 极端复杂任务：规划器自主决定，返回理论上限
            # 实际步数由 LLM 根据任务需求自行生成
            max_steps = self.MAX_PLAN_STEPS
        
        self._adaptive_max_steps = max_steps
        self.log_thought(f"📊 Task complexity: {complexity:.2f} → Max plan steps: {max_steps} (自主决策)")
        return max_steps

    async def decompose_task(self, text: str, failed_steps: List[str] = None, error_diagnosis: str = None, memory_context: List[Dict] = None) -> List[str]:
        """
        Decompose a high-level goal into executable steps, considering past failures and memory.

        🆕 [P0级优化] 快速路径：
        1. 规划缓存检查（基于任务文本hash）
        2. 确定性规则匹配（未来扩展）
        3. LLM规划（仅当缓存未命中）
        """
        self.log_thought(f"Analyzing task complexity: {text}")

        # 🆕 [P0优化] 快速路径1：规划缓存检查
        if self.enable_planning_cache and not failed_steps and not error_diagnosis:
            cache_key = self._generate_plan_cache_key(text)
            if cache_key and cache_key in self.planning_cache:
                cached_steps = self.planning_cache[cache_key]
                self.cache_hits += 1
                self.log_thought(
                    f"💾 规划缓存命中 "
                    f"(steps={len(cached_steps)}, "
                    f"cache_hits={self.cache_hits})"
                )
                return cached_steps
            else:
                self.cache_misses += 1

        # Windows command adaptation
        if platform.system() == "Windows":
            text = text.replace("ls -lt", "dir /O-D").replace("ls", "dir").replace("grep", "findstr").replace("cat", "type").replace("rm", "del").replace("cp", "copy").replace("mv", "move")

        lower_text = (text or "").lower()
        if "[meta]" in lower_text and ("investigate" in lower_text or "high entropy" in lower_text or "调查" in lower_text):
            import time as _time

            def j(tool: str, args: Dict[str, Any]) -> str:
                return json.dumps({"tool": tool, "args": args}, ensure_ascii=False)

            output_file = None
            marker = "data/entropy_investigation_"
            start = (text or "").find(marker)
            if start != -1:
                end = (text or "").find(".json", start)
                if end != -1:
                    output_file = (text or "")[start:end + 5]
            if not output_file:
                output_file = f"data/entropy_investigation_{int(_time.time())}.json"

            steps: List[str] = []
            steps.append(j("analyze_entropy_sources", {"output_file": output_file}))
            steps.append(j("check_memory_drift", {"threshold": 0.3}))
            steps.append(j("evaluate_uncertainty_distribution", {}))
            steps.append(j("synthesize_investigation_report", {"output_file": output_file}))
            steps.append(j("log", {"message": f"Meta-cognitive investigation complete. Report: {output_file}"}))
            self.log_thought(f"Meta-cognitive investigation plan created with {len(steps)} steps.")
            max_steps = self._get_adaptive_max_steps(text)
            return steps[:max_steps]

        avoid_instruction = ""
        if failed_steps and len(failed_steps) > 0:
            avoid_instruction = f"\n        IMPORTANT: The following steps have ALREADY FAILED. DO NOT generate them again. Find a different approach:\n        {json.dumps(failed_steps)}\n"

        error_instruction = ""
        if error_diagnosis:
            error_instruction = f"\n        CRITICAL: The previous attempt failed. DIAGNOSIS:\n        {error_diagnosis}\n        ADJUST your plan to fix this error.\n"
            
        # Build Memory Context String
        memory_instruction = ""
        if memory_context:
            memory_str = ""
            for m in memory_context:
                source = m.get('source', 'unknown')
                content = m.get('content', '')
                # Highlight failures
                prefix = "[PAST FAILURE]" if "failure" in source or "failure" in m.get('tags', []) else "[MEMORY]"
                memory_str += f"- {prefix} ({source}): {content[:200]}...\n"
            
            memory_instruction = f"""
        MEMORY CONTEXT (Related past experiences):
        {memory_str}
        PAY SPECIAL ATTENTION to [PAST FAILURE] items to avoid repeating mistakes.
        """

        macro_instruction = ""
        if self.biological_memory is not None:
            try:
                macros = self.biological_memory.suggest_macros_for_goal(text, top_k=3)
            except Exception:
                macros = []
            if macros:
                lines = []
                for m in macros:
                    mid = m.get("macro_id") or m.get("id")
                    pat = m.get("content_preview") or ""
                    lines.append(f"- Macro ID: {mid} | Pattern: {pat}")
                macro_text = "\n".join(lines)
                macro_instruction = f"""
        LEARNED MACRO PATTERNS (Reusable high-level behaviors):
{macro_text}
        You may treat an entire macro pattern as ONE high-level step in your plan,
        instead of enumerating all of its primitive operations separately.
        If a macro is applicable, you can call it directly:
        {{"tool": "execute_macro", "args": {{"macro_id": "<macro_id>", "bindings": {{}} }} }}
        """

        prompt = f"""
        You are the PLANNER Agent.
        Your job is to break down this complex user request into a sequence of atomic steps.
        
        CURRENT SYSTEM: {platform.system()}
        IMPORTANT: Generate commands compatible with this operating system (e.g., use 'dir' instead of 'ls' on Windows).
        
        The EXECUTOR Agent has these tools:
        - "Google Search X"
        - "Read file X" (JSON: {{"tool": "read_file", "args": {{"path": "..."}}}})
        - "Write file X" (JSON: {{"tool": "write_file", "args": {{"path": "...", "content": "..."}}}})
        - "Run Python script X" (JSON: {{"tool": "run_python", "args": {{"script_name": "..."}}}})
        - "Run Command X" (JSON: {{"tool": "run_command", "args": {{"command": "..."}}}})
        - "Execute Macro X" (JSON: {{"tool": "execute_macro", "args": {{"macro_id": "...", "bindings": {{}} }} }})
        - "Open App X" (ONLY use if app is NOT open)
        - "Type Text X"
        - "Wait X" (JSON: {{"tool": "wait", "args": {{"seconds": 5}}}})
        - "Log Observation" (JSON: {{"tool": "log", "args": {{"message": "..."}}}})
        
        USER REQUEST: {text}
        {memory_instruction}
        {macro_instruction}
        {avoid_instruction}
        {error_instruction}
        CRITICAL RULES:
        1. If the request is about "Observing", "Monitoring", or "Learning", DO NOT open new apps. Use "Wait" and "Log Observation".
        2. Output a PURE JSON list of strings.
        3. If the task involves file operations, PREFER using the JSON tool format in the step string.
        4. YOU decide the step count based on task complexity. Simple tasks: 3-10 steps. Complex tasks: 20-50+ steps. No artificial limit.
        5. For research/investigation/design tasks, use as many steps as needed for thoroughness. You have full autonomy.
        
        Example Output for Observation:
        [
            "{{\\"tool\\": \\"log\\", \\"args\\": {{\\"message\\": \\"Starting observation cycle...\\"}} }}",
            "{{\\"tool\\": \\"wait\\", \\"args\\": {{\\"seconds\\": 5}} }}",
            "{{\\"tool\\": \\"log\\", \\"args\\": {{\\"message\\": \\"Observation complete.\\"}} }}"
        ]
        """
        
        try:
            resp = self.llm.chat_completion(system_prompt="You are a Senior Project Manager.", user_prompt=prompt)

            if isinstance(resp, str) and (resp.startswith("[MOCK") or resp.startswith("[LLM ERROR]")):
                steps = self._heuristic_plan(text)
                # 🆕 [P0优化] 启发式规划也缓存（低置信度）
                if not failed_steps and not error_diagnosis:
                    self._store_plan_cache(text, steps)
                self.log_thought(f"Planner LLM unavailable. Heuristic plan created with {len(steps)} steps.")
                return steps

            json_str = resp
            if "```json" in resp:
                json_str = resp.split("```json")[1].split("```")[0].strip()
            elif "```" in resp:
                json_str = resp.split("```")[1].split("```")[0].strip()

            steps = json.loads(json_str)
            if isinstance(steps, list) and steps:
                max_steps = self._get_adaptive_max_steps(text)
                final_steps = steps[:max_steps] if len(steps) > max_steps else steps

                # 🆕 [P0优化] 存储到缓存（仅在无错误且无失败历史时）
                if not failed_steps and not error_diagnosis:
                    self._store_plan_cache(text, final_steps)

                self.log_thought(f"Plan created with {len(steps)} steps (max allowed: {max_steps}).")
                return final_steps

            steps = self._heuristic_plan(text)
            # 🆕 [P0优化] 启发式规划也缓存（低置信度）
            if not failed_steps and not error_diagnosis:
                self._store_plan_cache(text, steps)
            self.log_thought(f"Planner returned non-list plan. Heuristic plan created with {len(steps)} steps.")
            return steps

        except Exception as e:
            self.log_thought(f"Planning failed: {e}. Using heuristic plan.")
            return self._heuristic_plan(text)

    def _heuristic_plan(self, text: str) -> List[str]:
        t = (text or "").strip()
        lower = t.lower()

        def j(tool: str, args: Dict[str, Any]) -> str:
            return json.dumps({"tool": tool, "args": args}, ensure_ascii=False)

        steps: List[str] = []

        if "analyze generated insight" in lower and " in " in lower:
            try:
                path = t.split(" in ", 1)[1].strip()
                steps.append(j("read_file", {"path": path}))
                steps.append(j("log", {"message": f"Loaded insight for analysis: {path}"}))
                return steps[:6]
            except Exception:
                pass

        # 🔧 [2026-01-11] 元认知调查任务的专用计划模板 - 修复空转循环
        # 🔧 [2026-01-17] 解除硬编码步数限制，使用自适应步数
        if "[meta]" in lower and ("investigate" in lower or "high entropy" in lower or "调查" in lower):
            import time as _time
            output_file = f"data/entropy_investigation_{int(_time.time())}.json"
            steps.append(j("analyze_entropy_sources", {"output_file": output_file}))
            steps.append(j("check_memory_drift", {"threshold": 0.3}))
            steps.append(j("evaluate_uncertainty_distribution", {}))
            steps.append(j("synthesize_investigation_report", {"output_file": output_file}))
            steps.append(j("log", {"message": f"Meta-cognitive investigation complete. Report: {output_file}"}))
            max_steps = self._get_adaptive_max_steps(t)
            return steps[:max_steps]

        if any(k in lower for k in ["self-diagnostics", "diagnostics", "self diagnostics", "自检", "诊断", "结构"]):
            steps.append(j("list_files", {"path": "."}))
            steps.append(j("list_files", {"path": "core"}))
            steps.append(j("inspect_code", {"path": "AGI_Life_Engine.py", "mode": "summary"}))
            steps.append(j("inspect_code", {"path": "core/agents/executor.py", "mode": "summary"}))
            steps.append(j("log", {"message": "Heuristic diagnostics complete."}))
            max_steps = self._get_adaptive_max_steps(t)
            return steps[:max_steps]

        if any(k in lower for k in ["list files", "dir ", "目录", "文件列表"]):
            steps.append(j("list_files", {"path": "."}))
            max_steps = self._get_adaptive_max_steps(t)
            return steps[:max_steps]

        if "log" in lower or "observe" in lower or "monitor" in lower or "观察" in lower or "监控" in lower:
            steps.append(j("log", {"message": f"Observation: {t}"}))
            steps.append(j("wait", {"seconds": 2}))
            steps.append(j("log", {"message": "Observation tick complete."}))
            max_steps = self._get_adaptive_max_steps(t)
            return steps[:max_steps]

        steps.append(j("log", {"message": f"Heuristic fallback: {t}"}))
        max_steps = self._get_adaptive_max_steps(t)
        return steps[:max_steps]

    async def is_complex_task(self, text: str) -> bool:
        if len(text) < 10: return False
        prompt = f"Is this task complex enough to require breakdown? Answer YES/NO.\nTask: {text}"
        try:
            resp = self.llm.chat_completion(system_prompt="Analyzer", user_prompt=prompt)
            return "YES" in resp.upper()
        except:
            return False

    # 🆕 [P0级优化] 规划缓存辅助方法

    def _generate_plan_cache_key(self, text: str) -> Optional[str]:
        """
        生成规划缓存的唯一key

        Args:
            text: 任务文本

        Returns:
            MD5哈希值
        """
        if not text:
            return None

        # 标准化文本（去除空格、转小写）
        normalized = ' '.join(text.lower().split())

        # 生成hash
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def _store_plan_cache(self, text: str, steps: List[str]) -> None:
        """
        存储规划结果到缓存

        Args:
            text: 任务文本
            steps: 规划步骤列表
        """
        if not self.enable_planning_cache:
            return

        cache_key = self._generate_plan_cache_key(text)
        if not cache_key:
            return

        # 限制缓存大小
        if len(self.planning_cache) >= 500:
            # 删除最旧的条目（简单FIFO）
            oldest_key = next(iter(self.planning_cache))
            del self.planning_cache[oldest_key]
            self.log_thought(f"🗑️ 规划缓存已满，删除最旧条目")

        # 存储到缓存
        self.planning_cache[cache_key] = steps
        self.log_thought(f"💾 规划结果已缓存 (key={cache_key[:8]}..., steps={len(steps)})")

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        获取规划缓存统计信息

        Returns:
            统计信息字典
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0

        return {
            'cache_size': len(self.planning_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'enable_cache': self.enable_planning_cache
        }

    def clear_plan_cache(self) -> None:
        """清空规划缓存"""
        size_before = len(self.planning_cache)
        self.planning_cache.clear()
        self.log_thought(f"🗑️ 规划缓存已清空 (删除{size_before}条目)")
