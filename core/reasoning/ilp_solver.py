import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import copy

logger = logging.getLogger(__name__)

@dataclass
class ILPExample:
    input_state: Any
    output_state: Any
    is_positive: bool = True

class ILPSolver:
    """
    ILP (Inductive Logic Programming) Solver for ARC.
    Synthesizes logical rules (Horn clauses) that explain the transformation 
    from input to output grids.
    """
    def __init__(self):
        self.hypothesis_space = []
        self.learned_rules = []

    def derive_rule(self, examples: List[ILPExample]) -> Optional[str]:
        """
        Derive a logical rule that satisfies all positive examples and no negative examples.
        """
        logger.info(f"ILP: Deriving rules for {len(examples)} examples")
        
        # 1. Feature Extraction (Predicates)
        # Extract features like "has_color(obj, red)", "is_square(obj)", "top_left(obj)"
        # predicates_map = self._extract_predicates(examples)
        
        # 2. Hypothesis Generation (Search)
        candidate_rule = self._search_hypothesis(examples)
        
        if candidate_rule:
            self.learned_rules.append(candidate_rule)
            return candidate_rule
        
        return None

    def _search_hypothesis(self, examples: List[ILPExample]) -> Optional[str]:
        """
        Search for a hypothesis (rule) that covers positive examples.
        Basic implementation checks for Identity and Constant Color Change.
        """
        valid_examples = [e for e in examples if e.is_positive]
        if not valid_examples:
            return None

        # Check for Identity
        if all(self._grids_equal(e.input_state, e.output_state) for e in valid_examples):
            return "rule(X, Y) :- equal(X, Y)"
            
        # Check for Single Color Transform (e.g. all Blue becomes Red)
        # Find if there is a consistent mapping
        color_map = self._find_consistent_color_map(valid_examples)
        if color_map:
            # Format: rule(X, Y) :- replace_color(X, Y, {src: dst, ...})
            map_str = str(color_map).replace(" ", "")
            return f"rule(X, Y) :- replace_color(X, Y, {map_str})"

        return None

    def _find_consistent_color_map(self, examples: List[ILPExample]) -> Optional[Dict[int, int]]:
        """Check if there is a global color mapping that works for all examples."""
        global_map = {}
        
        for ex in examples:
            inp = ex.input_state
            out = ex.output_state
            
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return None
                
            local_map = {}
            for r in range(len(inp)):
                for c in range(len(inp[0])):
                    src = inp[r][c]
                    dst = out[r][c]
                    
                    if src in local_map and local_map[src] != dst:
                        return None # Conflict within grid
                    local_map[src] = dst
            
            # Merge with global map
            for k, v in local_map.items():
                if k in global_map and global_map[k] != v:
                    return None # Conflict across examples
                global_map[k] = v
                
        # If identity map (no change), it's covered by Identity check, but valid here too.
        # Filter out identity mappings (k==v) to make it cleaner, unless empty
        real_changes = {k: v for k, v in global_map.items() if k != v}
        if not real_changes:
            return None # Just identity
            
        return real_changes

    def _grids_equal(self, grid_a, grid_b) -> bool:
        if not grid_a or not grid_b: return False
        if len(grid_a) != len(grid_b): return False
        if len(grid_a[0]) != len(grid_b[0]): return False
        return grid_a == grid_b

    def verify(self, rule: str, test_input: Any) -> Any:
        """
        Apply the learned rule to a test input.
        """
        if rule == "rule(X, Y) :- equal(X, Y)":
            return test_input
            
        if rule.startswith("rule(X, Y) :- replace_color"):
            # Extract map string
            try:
                start = rule.find("{")
                end = rule.find("}") + 1
                map_str = rule[start:end]
                color_map = eval(map_str) # Safe as we generated it
                
                # Apply map
                output = []
                for row in test_input:
                    new_row = []
                    for cell in row:
                        new_row.append(color_map.get(cell, cell))
                    output.append(new_row)
                return output
            except Exception as e:
                logger.error(f"Failed to execute rule {rule}: {e}")
                return test_input

        return test_input
