import logging
import numpy as np
import traceback
from typing import List, Dict, Any, Optional, Callable
from core.llm_client import LLMService

logger = logging.getLogger(__name__)

class ARCSolver:
    """
    Program Synthesis Engine for ARC (Abstraction and Reasoning Corpus).
    
    Bridging the gap between:
    - Fluid Intelligence (Topological Pattern Recognition)
    - Crystal Intelligence (Symbolic Program Execution)
    
    Methodology:
    1. Analyze input/output pairs to infer transformation rules.
    2. Use LLM to synthesize a Python function `transform(grid) -> grid`.
    3. Verify the function against training examples.
    4. If verification passes, apply to test input.
    """
    
    def __init__(self):
        self.llm = LLMService()
        self.max_attempts = 5
        
    def solve(self, train_examples: List[Dict[str, Any]], test_input: List[List[int]]) -> Optional[List[List[int]]]:
        """
        Main entry point to solve an ARC task.
        
        Args:
            train_examples: List of {"input": grid, "output": grid}
            test_input: Input grid for the test case
            
        Returns:
            Predicted output grid, or None if failed.
        """
        logger.info(f"ARC Solver: Attempting to solve task with {len(train_examples)} examples")
        
        # 1. Synthesize Candidates
        for attempt in range(self.max_attempts):
            logger.info(f"ARC Solver: Synthesis Attempt {attempt+1}/{self.max_attempts}")
            
            try:
                # Generate code
                code = self._generate_hypothesis(train_examples)
                if not code:
                    continue
                    
                # Verify code
                is_valid = self._verify_hypothesis(code, train_examples)
                
                if is_valid:
                    logger.info("ARC Solver: ✅ Hypothesis verified! Executing on test input.")
                    result = self._execute_safe(code, test_input)
                    return result
                else:
                    logger.warning("ARC Solver: ❌ Hypothesis failed verification.")
                    
            except Exception as e:
                logger.error(f"ARC Solver: Error in attempt {attempt}: {e}")
                
        logger.error("ARC Solver: Failed to find solution after max attempts.")
        return None

    def _generate_hypothesis(self, train_examples: List[Dict[str, Any]]) -> str:
        """
        Prompt the LLM to generate a Python function.
        """
        prompt = self._construct_prompt(train_examples)
        
        # Call LLM
        response = self.llm.chat_completion(
            system_prompt="You are an expert Python programmer and AI researcher specializing in the Abstraction and Reasoning Corpus (ARC). Your goal is to write a Python function that transforms an input grid into an output grid based on examples.",
            user_prompt=prompt,
            # model="gpt-4o" # Removed hardcoded model to use system default
        )
        
        # Extract code block
        code = self._extract_code(response)
        return code

    def _construct_prompt(self, examples: List[Dict[str, Any]]) -> str:
        prompt = "Write a Python function `transform(grid)` that maps the input grid to the output grid for the following examples.\n\n"
        prompt += "The grid is a list of lists of integers (0-9).\n"
        prompt += "You can use `import numpy as np`.\n"
        prompt += "The function MUST be named `transform` and take a single argument `grid`.\n"
        prompt += "Return the transformed grid as a list of lists.\n\n"
        
        for i, ex in enumerate(examples):
            prompt += f"Example {i+1}:\n"
            prompt += f"Input:\n{ex['input']}\n"
            prompt += f"Output:\n{ex['output']}\n\n"
            
        prompt += "Please provide ONLY the Python code block."
        return prompt

    def _extract_code(self, response: str) -> str:
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            return response[start:end].strip()
        return response.strip()

    def _verify_hypothesis(self, code: str, examples: List[Dict[str, Any]]) -> bool:
        """
        Check if the code correctly transforms all training inputs to outputs.
        """
        for ex in examples:
            inp = ex['input']
            expected_out = ex['output']
            
            try:
                actual_out = self._execute_safe(code, inp)
                if actual_out != expected_out:
                    return False
            except Exception:
                return False
                
        return True

    def _execute_safe(self, code: str, input_grid: List[List[int]]) -> List[List[int]]:
        """
        Execute the generated code in a restricted namespace.
        """
        # Create a local namespace
        local_scope = {}
        
        # Execute definition
        try:
            exec(code, {"np": np}, local_scope)
        except Exception as e:
            raise RuntimeError(f"Code definition failed: {e}")
            
        if 'transform' not in local_scope:
            raise RuntimeError("Function 'transform' not found in generated code.")
            
        transform_func = local_scope['transform']
        
        # Execute function
        try:
            result = transform_func(input_grid)
            # Ensure it's list of lists (not numpy array)
            if isinstance(result, np.ndarray):
                result = result.tolist()
            return result
        except Exception as e:
            raise RuntimeError(f"Execution failed: {e}")
