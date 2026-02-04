import os
import sys
import logging
from typing import List, Dict, Any, Optional

# Ensure the project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Try to import Phase 5 memory system
try:
    from core.memory import ExperienceMemory
    PHASE5_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Phase 5 'core.memory' module not available: {e}")
    PHASE5_AVAILABLE = False

logger = logging.getLogger(__name__)

class MemoryBridge:
    """
    Bridge to access Phase 5 ExperienceMemory from Phase 3/4 systems.
    Allows the older 'agi_chat_enhanced.py' to read knowledge ingested by Phase 5 tools.
    """
    def __init__(self):
        self.memory: Optional[ExperienceMemory] = None
        if PHASE5_AVAILABLE:
            try:
                # Initialize ExperienceMemory (Phase 5)
                # This will load short_term.json and long_term.json
                self.memory = ExperienceMemory()
                logger.info("✅ MemoryBridge connected to Phase 5 ExperienceMemory")
            except Exception as e:
                logger.error(f"❌ Failed to initialize ExperienceMemory: {e}", exc_info=True)
                self.memory = None
        else:
            logger.warning("⚠️ Phase 5 'core.memory' module not found.")

    def search(self, query: str, top_k: int = 5, threshold: float = 0.4) -> List[Dict[str, Any]]:
        """
        Search Phase 5 knowledge base using vector similarity.
        
        Args:
            query: The search query string
            top_k: Number of results to return (default: 5)
            threshold: Similarity threshold for filtering results (default: 0.4)
            
        Returns:
            List of memory items (dictionaries), or error object if failed
        """
        if not self.memory:
            error_result = {"error": "Phase 5 Memory system not initialized"}
            logger.warning("Attempted search but memory system is not available.")
            return [error_result]
        
        if not isinstance(query, str) or not query.strip():
            error_result = {"error": "Query must be a non-empty string"}
            logger.warning("Invalid query provided: %s", query)
            return [error_result]
        
        if top_k <= 0:
            logger.warning("top_k value %d is invalid; using default 1", top_k)
            top_k = 1

        try:
            # recall_relevant returns results sorted by relevance
            raw_results = self.memory.recall_relevant(query, threshold=threshold)
            
            # Validate and limit number of results
            if not isinstance(raw_results, list):
                logger.error("Expected list from recall_relevant, got %s", type(raw_results))
                return [{"error": "Invalid response format from memory system"}]

            # Early return for empty results
            if not raw_results:
                logger.debug("No results found for query: '%s'", query)
                return []

            # Return up to top_k results
            filtered_results = raw_results[:top_k]
            logger.debug("Returned %d results for query: '%s'", len(filtered_results), query)
            return filtered_results

        except AttributeError as e:
            logger.error("Memory object missing required method 'recall_relevant': %s", e, exc_info=True)
            return [{"error": "Memory system interface mismatch"}]
        except Exception as e:
            logger.error("Unexpected error during knowledge recall: %s", e, exc_info=True)
            return [{"error": f"Search failed due to internal error: {str(e)}"}]