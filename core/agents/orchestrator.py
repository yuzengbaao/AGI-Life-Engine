from typing import Dict, Any
from core.llm_client import LLMService
from core.agents_legacy import CodeGenerationAgent, ArchitectureReviewAgent

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