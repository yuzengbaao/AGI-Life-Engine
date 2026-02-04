import os
import time
import logging
import hashlib
from typing import List, Dict, Any, Set
from core.evolution.dynamics import EvolutionaryDynamics
from core.evolution.impl import EvolutionController

# Configure logging
logger = logging.getLogger("ForagingSystem")

class ForagingAgent:
    """
    The sensory-motor agent responsible for scanning the environment (Foraging)
    and consuming resources (Digestion).
    """
    
    def __init__(self, root_path: str, evolution_controller: EvolutionController):
        if not os.path.exists(root_path):
            raise ValueError(f"Root path does not exist: {root_path}")
        if not isinstance(evolution_controller, EvolutionController):
            raise TypeError("evolution_controller must be an instance of EvolutionController")
            
        self.root_path = os.path.abspath(root_path)
        self.evolution_controller = evolution_controller
        self.known_files: Set[str] = set()  # Track already processed files
        self.last_forage_time: float = 0.0
        
    def perceive(self, capability_score: float) -> List[Dict[str, Any]]:
        """
        Scans the environment based on the current Visual Horizon.
        Uses efficient directory traversal with early termination.
        
        Args:
            capability_score: Float representing agent's perception capability
            
        Returns:
            List of candidate files sorted by nutrient density
        """
        try:
            horizon = EvolutionaryDynamics.calculate_visual_horizon(capability_score)
            logger.info(f"👁️ Visual Horizon: Depth={horizon['max_depth']}, Types={horizon['allowed_extensions']}")
            
            candidates: List[Dict[str, Any]] = []
            start_depth = self.root_path.count(os.path.sep)
            allowed_extensions = set(ext.lower() for ext in horizon['allowed_extensions'])
            max_file_size = horizon['max_file_size']
            
            for root, dirs, files in os.walk(self.root_path):
                current_depth = root.count(os.path.sep) - start_depth
                if current_depth > horizon['max_depth']:
                    del dirs[:]  # Prune directories to avoid deeper traversal
                    continue
                
                # Pre-filter files by extension
                filtered_files = [
                    f for f in files 
                    if os.path.splitext(f)[1].lower() in allowed_extensions
                ]
                
                for file in filtered_files:
                    full_path = os.path.join(root, file)
                    
                    # Memory filter
                    if full_path in self.known_files:
                        continue
                    
                    # Size check
                    try:
                        size = os.path.getsize(full_path)
                        if size == 0 or size > max_file_size:
                            continue
                            
                        # Calculate nutrient density
                        density = EvolutionaryDynamics.calculate_nutrient_density(full_path, "")
                        
                        candidates.append({
                            "path": full_path,
                            "density": density,
                            "size": size
                        })
                        
                    except (OSError, IOError) as e:
                        logger.debug(f"Skipping inaccessible file {full_path}: {e}")
                        continue
                        
            # Sort once with optimized key
            candidates.sort(key=lambda x: x["density"], reverse=True)
            return candidates
            
        except Exception as e:
            logger.error(f"Error during perception phase: {e}")
            return []

    async def forage_and_digest(self, capability_score: float) -> Dict[str, Any]:
        """
        Executes one full cycle of Foraging -> Selection -> Digestion.
        
        Args:
            capability_score: Perception capability determining scan depth and scope
            
        Returns:
            Dictionary with status and result details
        """
        self.last_forage_time = time.time()
        
        try:
            # Scan environment
            candidates = await self.perceive(capability_score)
            if not candidates:
                return {
                    "status": "empty", 
                    "message": "No suitable nutrients found within horizon."
                }
                
            # Select highest value target
            target = candidates[0]
            logger.info(f"🎯 Foraging Target Selected: {target['path']} (Density: {target['density']:.2f})")
            
            # Process the file
            success = await self.digest(target['path'])
            
            return {
                "status": "success" if success else "digestion_failed",
                "target": target['path'],
                "digestion_result": success
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in forage_and_digest cycle: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def digest(self, file_path: str) -> bool:
        """
        Ingests the file into the System Memory (ChromaDB + Graph).
        
        Args:
            file_path: Path to the file to be ingested
            
        Returns:
            Boolean indicating success of digestion
        """
        if not file_path or not isinstance(file_path, str):
            logger.error("Invalid file path provided for digestion")
            return False
            
        try:
            # Validate file existence and accessibility
            if not os.path.isfile(file_path):
                logger.warning(f"File not found or is not a regular file: {file_path}")
                return False
                
            # Read content safely
            content = ""
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except (IOError, OSError) as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                return False
                
            if not content.strip():
                logger.debug(f"Empty or whitespace-only content in {file_path}")
                return False
                
            # Mark as processed before expensive operations
            self.known_files.add(file_path)
            
            # Store in memory system
            try:
                metadata = {
                    "source": "foraging", 
                    "path": file_path, 
                    "timestamp": time.time(),
                    "size": len(content)
                }
                
                # Vector store ingestion
                self.evolution_controller.memory.store_memory(
                    content=content[:2000],  # Chunk limit
                    metadata=metadata
                )
                
                # Knowledge graph update
                doc_id = hashlib.md5(file_path.encode()).hexdigest()
                filename = os.path.basename(file_path)
                self.evolution_controller.knowledge_graph.add_concept(
                    name=f"DOC_{filename}",
                    category="ExternalResource",
                    properties={
                        "path": file_path, 
                        "density": "high",
                        "doc_id": doc_id
                    }
                )
                
                logger.info(f"✅ Digested: {filename}")
                return True
                
            except Exception as e:
                logger.error(f"Memory system ingestion failed for {file_path}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Unexpected error during digestion of {file_path}: {e}")
            return False