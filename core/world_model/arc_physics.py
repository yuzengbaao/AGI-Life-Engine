import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Optional

@dataclass
class ARCObject:
    id: int
    pixels: Set[Tuple[int, int]]
    color: int
    top_left: Tuple[int, int]
    width: int
    height: int

class ARCWorldModel:
    """
    World Model for ARC tasks.
    Simulates physics (gravity, collision) and manages object-centric representations.
    """
    def __init__(self, grid: List[List[int]]):
        self.grid = np.array(grid)
        self.height, self.width = self.grid.shape
        self.objects = self._segment_objects()
        self.background_color = 0 # Assume 0 is background usually

    def _segment_objects(self) -> List[ARCObject]:
        """
        Segment grid into objects using connected components (connectivity=4).
        """
        visited = set()
        objects = []
        obj_id = 1
        
        # Detect background color (most frequent?) 
        # For ARC, 0 (black) is standard background, but sometimes it's different.
        # Here we assume 0 for simplicity or pass it in.
        bg_color = 0
        
        for r in range(self.height):
            for c in range(self.width):
                color = self.grid[r, c]
                if color != bg_color and (r, c) not in visited:
                    # Flood fill
                    pixels = set()
                    stack = [(r, c)]
                    min_r, min_c = r, c
                    max_r, max_c = r, c
                    
                    while stack:
                        curr_r, curr_c = stack.pop()
                        if (curr_r, curr_c) in visited: continue
                        visited.add((curr_r, curr_c))
                        pixels.add((curr_r, curr_c))
                        
                        min_r = min(min_r, curr_r)
                        min_c = min(min_c, curr_c)
                        max_r = max(max_r, curr_r)
                        max_c = max(max_c, curr_c)
                        
                        # Neighbors
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < self.height and 0 <= nc < self.width:
                                if self.grid[nr, nc] == color and (nr, nc) not in visited:
                                    stack.append((nr, nc))
                    
                    objects.append(ARCObject(
                        id=obj_id,
                        pixels=pixels,
                        color=color,
                        top_left=(min_r, min_c),
                        width=max_c - min_c + 1,
                        height=max_r - min_r + 1
                    ))
                    obj_id += 1
        return objects

    def apply_gravity(self, direction='down') -> 'ARCWorldModel':
        """
        Simulate gravity: move all objects in direction until they hit boundary or other objects.
        Simplified version: Just stack them at bottom.
        """
        # Create empty grid
        new_grid = np.zeros_like(self.grid)
        
        # Sort objects: if moving down, process bottom-most first?
        # Actually, simpler logic for ARC "gravity" usually means columns fall independently
        # OR objects fall as blocks.
        # Let's implement "Column Gravity" which is common in ARC (tetris style)
        
        for c in range(self.width):
            col = self.grid[:, c]
            pixels = [x for x in col if x != 0]
            # Fill from bottom
            new_col = np.zeros(self.height, dtype=int)
            if pixels:
                new_col[-len(pixels):] = pixels
            new_grid[:, c] = new_col
            
        self.grid = new_grid
        self.objects = self._segment_objects() # Re-segment
        return self

    def get_state_matrix(self) -> List[List[int]]:
        return self.grid.tolist()
