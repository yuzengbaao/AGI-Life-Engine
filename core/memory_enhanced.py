import os
import json
import time
import logging
from typing import List, Dict, Any, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import chromadb
from chromadb.utils import embedding_functions
from core.llm_client import LLMService

# Configure logging
logger = logging.getLogger("EnhancedMemory")

class EnhancedExperienceMemory:
    """
    Enhanced Semantic Memory System backed by ChromaDB.
    
    Features:
    1. Persistent Vector Storage: Uses ChromaDB for efficient similarity search.
    2. Hybrid Retrieval: Combines semantic search (vector) with metadata filtering.
    3. Legacy Compatibility: Can load/import from old JSON memory files.
    4. Auto-Embedding: Uses lightweight default embedding or LLM service.
    """
    
    def __init__(self, memory_dir: str = "memory_db", collection_name: str = "semantic_memory"):
        """
        Initialize the Enhanced Memory System.
        
        Args:
            memory_dir: Path to the ChromaDB persistence directory.
            collection_name: Name of the collection to use/create.
        """
        self.memory_dir = os.path.abspath(memory_dir)
        self.collection_name = collection_name
        self.llm = LLMService()
        
        logger.info(f"🧠 Initializing Enhanced Memory at: {self.memory_dir}")
        
        # Ensure directory exists
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Initialize ChromaDB Client
        try:
            self.client = chromadb.PersistentClient(path=self.memory_dir)
            
            # Use a default embedding function (all-MiniLM-L6-v2 is standard and lightweight)
            # If we wanted to use OpenAI embeddings via LLMService, we'd need a custom function wrapper.
            # For now, let's stick to the default for reliability and speed.
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"description": "AGI Core Semantic Memory"}
            )
            
            count = self.collection.count()
            logger.info(f"✅ Memory loaded. Active memories: {count}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB: {e}")
            raise

    def add_experience(self, context: str, action: str, outcome: float, details: Dict[str, Any] = None) -> str:
        """
        Record a new experience.
        
        Args:
            context: The situational context (What happened?)
            action: The action taken
            outcome: Success score (0.0 - 1.0)
            details: Additional metadata
            
        Returns:
            memory_id: The unique ID of the stored memory
        """
        if details is None:
            details = {}
            
        # Enrich metadata
        metadata = details.copy()
        metadata.update({
            "action": action,
            "outcome": float(outcome),
            "timestamp": time.time(),
            "type": details.get("type", "experience")
        })
        
        # Create a unique ID
        memory_id = f"mem_{int(time.time())}_{hash(context) % 10000}"
        
        # The document text is what we search against. 
        # Combining context and action gives better semantic matches.
        document_text = f"Context: {context}\nAction: {action}\nOutcome: {outcome}"
        
        try:
            self.collection.add(
                documents=[document_text],
                metadatas=[metadata],
                ids=[memory_id]
            )
            logger.info(f"💾 Experience stored: {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"❌ Failed to store experience: {e}")
            return ""

    def retrieve_relevant(self, query: str, limit: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to the current query/context.
        
        Args:
            query: The current situation or question.
            limit: Max number of results.
            min_score: Minimum outcome score to filter for (only retrieve successful memories).
            
        Returns:
            List of memory dicts.
        """
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                # Optional: Filter by success score if needed in future
                # where={"outcome": {"$gte": min_score}} 
            )
            
            memories = []
            
            # Chroma returns lists of lists (one list per query)
            if results['ids'] and len(results['ids']) > 0:
                ids = results['ids'][0]
                metadatas = results['metadatas'][0]
                documents = results['documents'][0]
                distances = results['distances'][0]
                
                for i in range(len(ids)):
                    # Optional: Client-side filtering if metadata filtering is complex
                    if metadatas[i].get('outcome', 0.0) < min_score:
                        continue
                        
                    memories.append({
                        "id": ids[i],
                        "content": documents[i],
                        "metadata": metadatas[i],
                        "distance": distances[i] # Lower is better for cosine distance in Chroma
                    })
            
            logger.info(f"🔍 Retrieved {len(memories)} relevant memories for query: '{query[:30]}...'")
            return memories
            
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            return []

    def consolidate_legacy_memory(self, json_path: str):
        """
        Import data from old JSON memory files into ChromaDB.
        """
        if not os.path.exists(json_path):
            logger.warning(f"⚠️ Legacy memory file not found: {json_path}")
            return
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                logger.warning("Invalid JSON format (expected list)")
                return
                
            count = 0
            for item in data:
                # Map JSON fields to add_experience args
                context = item.get("context", "")
                action = item.get("action", "")
                outcome = item.get("outcome", 0.5)
                
                # If embedding exists in JSON, we currently ignore it and let Chroma re-embed 
                # to ensure vector space consistency.
                
                if context:
                    self.add_experience(context, action, outcome, details=item)
                    count += 1
                    
            logger.info(f"📥 Consolidated {count} memories from {json_path}")
            
        except Exception as e:
            logger.error(f"❌ Consolidation failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Return database statistics."""
        return {
            "total_memories": self.collection.count(),
            "collection_name": self.collection_name,
            "persistence_dir": self.memory_dir
        }
