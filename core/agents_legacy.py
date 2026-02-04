import time
import json
from typing import Dict, Any, Optional
from core.llm_client import LLMService

class AgentBase:
    def __init__(self, name: str, role: str, llm: LLMService):
        if not isinstance(llm, LLMService):
            raise TypeError("llm must be an instance of LLMService")
        self.name = name
        self.role = role
        self.llm = llm

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement 'process' method")

class CodeGenerationAgent(AgentBase):
    def __init__(self, llm: LLMService):
        super().__init__("Coder_01", "Code Generator", llm)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate code using LLM based on strategy.
        """
        try:
            strategy: str = input_data.get('strategy', 'Standard')
            complexity: float = input_data.get('complexity', 1.0)

            system_prompt = "You are an expert Python AGI developer. Write concise, high-performance Python code."
            user_prompt = f"Generate a Python function implementation demonstrating the '{strategy}' strategy. Keep it under 10 lines."

            response: Optional[str] = self.llm.chat_completion(system_prompt, user_prompt)
            if not response:
                raise RuntimeError("LLM returned empty response during code generation")

            return {
                "code_snippet": response.strip(),
                "quality_score": 0.9,
                "agent_comment": f"Generated using {strategy}"
            }
        except Exception as e:
            return {
                "code_snippet": "",
                "quality_score": 0.0,
                "agent_comment": f"Error in code generation: {str(e)}"
            }

class ArchitectureReviewAgent(AgentBase):
    def __init__(self, llm: LLMService):
        super().__init__("Reviewer_01", "Architecture Reviewer", llm)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review code using LLM for safety and efficiency.
        """
        try:
            code: str = input_data.get('code_snippet', '').strip()
            if not code:
                return {
                    "approved": False,
                    "risk_level": "HIGH",
                    "review_comment": "REJECTED: No code provided for review"
                }

            system_prompt = "You are a senior code reviewer. Analyze the code for safety and efficiency."
            user_prompt = f"Review this code:\n{code}"

            response: Optional[str] = self.llm.chat_completion(system_prompt, user_prompt)
            if not response:
                raise RuntimeError("LLM returned empty response during architecture review")

            # Dummy logic for illustration
            approved = "APPROVE" in response.upper()
            risk_level = "LOW" if approved else "HIGH"

            return {
                "approved": approved,
                "risk_level": risk_level,
                "review_comment": response.strip()
            }
        except Exception as e:
            return {
                "approved": False,
                "risk_level": "CRITICAL",
                "review_comment": f"Error in architecture review: {str(e)}"
            }

class AgentOrchestrator:
    def __init__(self):
        self.llm = LLMService()
        self.coder = CodeGenerationAgent(self.llm)
        self.reviewer = ArchitectureReviewAgent(self.llm)

    def run_development_cycle(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Generate Code
        code_result = self.coder.process(task_context)
        
        # 2. Review Code
        review_result = self.reviewer.process(code_result)
        
        return {
            "code": code_result,
            "review": review_result,
            "final_outcome": 1.0 if review_result.get('approved', False) else 0.0
        }

class AgentOrchestrator:
    def __init__(self):
        self.llm = LLMService()
        self.coder = CodeGenerationAgent(self.llm)
        self.reviewer = ArchitectureReviewAgent(self.llm)

    def run_development_cycle(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Generate Code
        code_result = self.coder.process(task_context)
        
        # 2. Review Code
        review_result = self.reviewer.process(code_result)
        
        return {
            "code": code_result,
            "review": review_result,
            "final_outcome": 1.0 if review_result.get('approved', False) else 0.0
        }