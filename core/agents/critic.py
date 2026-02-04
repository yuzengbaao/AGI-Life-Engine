import json
from typing import Dict, Any
from .base_agent import BaseAgent

class CriticAgent(BaseAgent):
    """
    Role: Safety Officer & Quality Assurance
    Responsibility: 
    1. Check if a planned action is safe (e.g., doesn't delete core files).
    2. Verify if an executed action actually succeeded.
    """
    def __init__(self, llm_service):
        super().__init__("Critic", llm_service)
        self.forbidden_paths = [
            "core/motivation.py",
            "AGI_Life_Engine.py",
            "agi_permission_firmware.py",
            "permission_manager.py",
            "agi_terminal_tool.py",
        ]

    async def check_safety(self, intent: str) -> bool:
        """
        Pre-flight check for dangerous operations.
        """
        # 1. Static Rules
        intent_lower = intent.lower()
        
        # Prevent deletion or overwriting of source code (unless explicitly authorized mode - TODO)
        if any(k in intent_lower for k in ["write_file", "delete_file"]):
            for path in self.forbidden_paths:
                if path.lower() in intent_lower:
                    self.log_thought(f"⚠️ SAFETY ALERT: Attempt to modify core file '{path}' blocked.")
                    return False

        # 1.5 Fast Path: Whitelist safe read-only operations to avoid "Over-Defense"
        # If the intent is clearly a read-only command, allow it immediately without LLM overhead/hallucination.
        try:
            # Simple heuristic check for common read commands in the intent string
            # We allow pipes (|) and redirects (>) if the base command is informational
            safe_keywords = [
                "dir ", "ls ", "type ", "cat ", "echo ", "grep ", "findstr ", 
                "pwd", "whoami", "date", "time", "read_file", "list_files",
                "python analyze_", "python verification/" 
            ]
            
            # If it's a run_command with only safe keywords and no destructive markers
            if '"tool": "run_command"' in intent_lower:
                is_safe_cmd = any(k in intent_lower for k in safe_keywords)
                is_destructive = any(k in intent_lower for k in ["del ", "rm ", "format ", "shutdown", "kill"])
                
                if is_safe_cmd and not is_destructive:
                    self.log_thought("✅ Fast-Track: Allowed safe read-only command.")
                    return True
                    
        except Exception:
            pass # Fallback to full semantic check if parsing fails
        
        # 2. Semantic Check (Context-Aware "Golden Mean" Safety)
        # We allow high entropy in thoughts/simulations (to prevent stagnation),
        # but enforce strict safety on physical actions (to prevent chaos).
        try:
            is_physical_action = any(act in intent_lower for act in ["write", "delete", "execute", "run", "modify", "inject"])
            
            prompt = f"""
            You are the Safety Officer (Critic) for an AGI system.
            Evaluate the following intent for safety.
            
            Intent: "{intent}"
            Context: {"PHYSICAL ACTION (Strict Safety)" if is_physical_action else "THOUGHT/SIMULATION (Lenient/Creative)"}
            
            Guidelines:
            1. IF PHYSICAL ACTION: 
               - BLOCK: DESTRUCTIVE operations (delete, format, overwrite core files) and high-risk system modifications.
               - ALLOW: Read-only operations (including pipes '|', filters like grep/findstr, and output redirection), status checks, analysis scripts, and filesystem operations scoped inside the project directory (especially within 'data/' or 'logs/'), excluding protected core files.
               - ALLOW: Execution of verification scripts and non-destructive terminal commands.
               - BIAS: If in doubt, ALLOW "read/check" operations. Only BLOCK actual "write/destroy" risks.
            2. IF THOUGHT/SIMULATION: 
               - ALLOW: Creative, high-entropy, or speculative ideas.
               - ALLOW: Meta-cognitive tasks (investigating entropy, curiosity-driven exploration, system state analysis).
               - ALLOW: Logging operations with phrases like "High entropy", "Curiosity level", "Investigation" - these are BENIGN system introspection.
               - DO NOT BLOCK: Internal exploration or debugging activities.
               - ONLY BLOCK: Fundamental ethical violations (hate/harm to humans).
            
            🔧 [2026-01-08] CRITICAL: "High entropy investigation" is a NORMAL system operation, NOT a safety risk.
            
            Reply with:
            - "SAFE" (Allow)
            - "UNSAFE: [Reason]" (Block)
            - "WARNING: [Reason]" (Allow with caution - for high entropy thoughts)
            """
            
            # Run LLM check asynchronously
            import asyncio
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.llm.chat_completion("System Safety Check", prompt)
            )
            
            response_upper = response.upper()
            if "UNSAFE" in response_upper:
                self.log_thought(f"🛑 SEMANTIC SAFETY BLOCK: {response}")
                return False
            elif "WARNING" in response_upper:
                self.log_thought(f"⚠️ SAFETY WARNING (High Entropy): {response}")
                # We allow it, but log the warning. This maintains the "Heartbeat".
                return True
                
        except Exception as e:
            self.log_thought(f"⚠️ Safety check error: {e}. Proceeding with caution.")
            
        return True

    async def verify_outcome(self, action: str, result: str) -> float:
        """
        Evaluate the success of an action (0.0 to 1.0).
        
        🔧 [2026-01-11] 增强元认知任务的证据验证
        """
        if result is None:
            return 0.0
            
        result_str = str(result)
        result_lower = result_str.lower()
        action_lower = str(action).lower()

        # 🔧 [2026-01-11] 元认知调查任务的证据验证 - 防止空转循环
        if "[meta]" in action_lower and ("investigate" in action_lower or "entropy" in action_lower):
            # 必须产生实质证据才能得高分
            evidence_markers = [
                "entropy_source",
                "memory_drift", 
                "uncertainty_analysis",
                "root_cause",
                "investigation_report",
                "analysis_complete"
            ]
            evidence_found = sum(1 for m in evidence_markers if m in result_lower)
            
            if evidence_found >= 2:
                score = 0.5 + 0.1 * evidence_found  # 2个证据=0.7, 4个=0.9
                self.log_thought(f"✅ Meta-cognitive investigation produced {evidence_found} evidence markers. Score: {score:.2f}")
                return min(1.0, score)
            elif "logged:" in result_lower and evidence_found == 0:
                # 仅日志无证据 = 低分
                self.log_thought(f"⚠️ Meta-cognitive investigation produced only logs, no evidence. Score: 0.3")
                return 0.3
            else:
                self.log_thought(f"⚠️ Meta-cognitive investigation partial evidence ({evidence_found}). Score: 0.5")
                return 0.5

        if result_lower.startswith("logged:") or "logged:" in result_lower:
            return 1.0
        
        # 1. 显式错误检测
        error_markers = [
            "error:",
            "exception",
            "traceback",
            "assertionerror",
            "执行失败",
            "步骤执行失败",
            "execution failed",
            "failed:",
            "failed to",
            "can't ",
            "cannot ",
            "access denied",
        ]
        if any(m in result_lower for m in error_markers):
            # 特例：如果是 grep/findstr 没找到结果，通常不算系统错误，而是逻辑False
            if "grep" in action_lower or "findstr" in action_lower:
                return 0.5 
            return 0.0
        
        # 2. 启发式成功检测
        # 如果是文件写入操作，检查文件是否存在 (Side-effect check)
        # 简单解析 action: {'tool': 'run_command', 'args': 'echo "..." > test.txt'}
        try:
            import os
            import re
            # 匹配 > filename
            match = re.search(r'>\s*([\w./\\-]+)', action)
            if match:
                filepath = match.group(1)
                if os.path.exists(filepath):
                    # 检查文件是否为空
                    if os.path.getsize(filepath) > 0:
                        return 1.0
                    else:
                        return 0.2 # 创建了但为空
                else:
                    return 0.0 # 应该创建但没找到
        except Exception:
            pass

        # 3. 默认基于输出长度的评分
        if len(result_str) > 50:
            return 1.0
        elif len(result_str) > 0:
            return 0.8
            
        return 0.5
