import logging
import numpy as np
from typing import List, Callable, Any, Optional

logger = logging.getLogger(__name__)

class LambdaEngine:
    """
    Lambda Calculus Reasoner for Program Synthesis.
    Uses a Domain Specific Language (DSL) based on grid primitives.
    """
    
    def __init__(self):
        self.primitives = self._register_primitives()
        
    def _register_primitives(self) -> dict:
        """Register DSL primitives"""
        return {
            'identity': lambda g: g,
            'rot90': lambda g: np.rot90(np.array(g), k=-1).tolist(),
            'rot180': lambda g: np.rot90(np.array(g), k=-2).tolist(),
            'rot270': lambda g: np.rot90(np.array(g), k=-3).tolist(),
            'flip_h': lambda g: np.fliplr(np.array(g)).tolist(),
            'flip_v': lambda g: np.flipud(np.array(g)).tolist(),
            # Higher order (mock)
            'map_color': lambda f, g: [[f(x) for x in r] for r in g],
        }

    def synthesize_program(self, input_grids: List[List[List[int]]], output_grids: List[List[List[int]]]) -> str:
        """
        Synthesize a lambda expression that transforms inputs to outputs.
        Uses Beam Search over the space of composed primitives.
        """
        logger.info("LambdaEngine: Starting program synthesis...")
        
        # 1. 1-step Primitive Search
        for name, func in self.primitives.items():
            if name == 'map_color': continue
            
            valid = True
            for inp, out in zip(input_grids, output_grids):
                try:
                    res = func(inp)
                    if not self._grids_equal(res, out):
                        valid = False
                        break
                except:
                    valid = False
                    break
            
            if valid:
                logger.info(f"LambdaEngine: Found simple primitive: {name}")
                return f"lambda x: {name}(x)"
                
        # 2. 2-step Composition Search (e.g. flip then rotate)
        prims = [k for k in self.primitives.keys() if k != 'map_color']
        for p1 in prims:
            for p2 in prims:
                # p2(p1(x))
                valid = True
                for inp, out in zip(input_grids, output_grids):
                    try:
                        res = self.primitives[p2](self.primitives[p1](inp))
                        if not self._grids_equal(res, out):
                            valid = False
                            break
                    except:
                        valid = False
                        break
                
                if valid:
                    logger.info(f"LambdaEngine: Found composite: {p2}({p1}(x))")
                    return f"lambda x: {p2}({p1}(x))"
        
        return "lambda x: identity(x)" # Fallback

    def execute(self, program_str: str, grid: List[List[int]]) -> List[List[int]]:
        """
        Execute a synthesized lambda expression.
        """
        # Context with primitives
        context = {**self.primitives}
        # Add numpy if needed by primitives (though they are closures, so they capture it)
        # But if the string is "lambda x: rot90(x)", rot90 must be in globals/locals.
        
        try:
            # We construct a closure
            # In production, use AST parsing for safety. Here we use eval with restricted globals.
            # program_str is like "lambda x: rot90(x)"
            # Passing context as globals ensures the lambda can resolve the function names.
            program = eval(program_str, {**context, "__builtins__": {}})
            result = program(grid)
            # Ensure list of lists
            if isinstance(result, np.ndarray):
                return result.tolist()
            return result
        except Exception as e:
            logger.error(f"Execution failed for {program_str}: {e}")
            return grid

    def _grids_equal(self, g1, g2):
        return np.array_equal(np.array(g1), np.array(g2))
