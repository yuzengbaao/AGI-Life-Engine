from typing import List, Dict, Any, Optional

class BaseAgent:
    def __init__(self, name: str, llm_service):
        self.name = name
        self.llm = llm_service
        
    def log_thought(self, content: str):
        print(f"   [{self.name}] 💭 {content}")
