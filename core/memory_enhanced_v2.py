import os
import json
import time
import logging
import math
import asyncio
from typing import List, Dict, Any, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from core.llm_client import LLMService
from core.evolution.dynamics import EvolutionaryDynamics

# Configure logging
logger = logging.getLogger("EnhancedMemoryV2")

class EnhancedExperienceMemory:
    """
    Enhanced Semantic Memory System backed by ChromaDB (Version 2).
    
    Upgrades:
    1. LRU Forgetting Mechanism: Prunes weak memories based on 'Vitality' (Age + Access + Importance).
    2. Intuition Retrieval: Fast similarity check for 'gut feeling'.
    3. Access Tracking: Updates access counts on retrieval to reinforce useful memories.
    4. Abstraction: Consolidates dense memory clusters into principles (via LLM).
    """
    
    def __init__(self, memory_dir: str = "memory_db", collection_name: str = "semantic_memory"):
        """
        Initialize the Enhanced Memory System V2.
        """
        self.memory_dir = os.path.abspath(memory_dir)
        self.collection_name = collection_name
        self.llm = LLMService()
        
        logger.info(f"🧠 Initializing Enhanced Memory V2 at: {self.memory_dir}")
        
        # Ensure directory exists
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Initialize ChromaDB Client
        try:
            self.client = chromadb.PersistentClient(path=self.memory_dir)
            
            # Use a default embedding function (all-MiniLM-L6-v2 is standard and lightweight)
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine", "description": "AGI Core Semantic Memory V2"}
            )
            
            # Secondary collection for abstract principles
            self.semantic_collection = self.client.get_or_create_collection(
                name=f"{self.collection_name}_principles",
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            
            count = self.collection.count()
            logger.info(f"✅ Memory loaded. Active memories: {count}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB: {e}")
            raise

    def add_experience(self, context: str, action: str, outcome: float, details: Dict[str, Any] = None) -> str:
        """
        Record a new experience.
        """
        if details is None:
            details = {}
            
        # Enrich metadata
        metadata = details.copy()
        metadata.update({
            "action": action,
            "outcome": float(outcome),
            "timestamp": time.time(),
            "access_count": 0, # Initialize access count
            "score": float(outcome), # Alias for vitality calc
            "type": details.get("type", "experience")
        })
        
        # Create a unique ID
        memory_id = f"mem_{int(time.time())}_{hash(context) % 10000}"
        
        # The document text is what we search against. 
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
        Retrieve memories relevant to the current query/context and update their vitality.
        """
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            memories = []
            ids_to_update = []
            metadatas_to_update = []
            
            if results['ids'] and len(results['ids']) > 0:
                ids = results['ids'][0]
                metadatas = results['metadatas'][0]
                documents = results['documents'][0]
                distances = results['distances'][0]
                
                for i in range(len(ids)):
                    meta = metadatas[i]
                    
                    # Filter by score
                    if meta.get('outcome', 0.0) < min_score:
                        continue
                        
                    # Update Access Stats (LRU Logic)
                    current_access = meta.get('access_count', 0)
                    meta['access_count'] = current_access + 1
                    meta['last_accessed'] = time.time()
                    
                    ids_to_update.append(ids[i])
                    metadatas_to_update.append(meta)
                    
                    memories.append({
                        "id": ids[i],
                        "content": documents[i],
                        "metadata": meta,
                        "distance": distances[i]
                    })
            
            # Batch Update Metadata (Reinforce Memory)
            if ids_to_update:
                try:
                    self.collection.update(
                        ids=ids_to_update,
                        metadatas=metadatas_to_update
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to update access counts: {e}")
            
            logger.info(f"🔍 Retrieved {len(memories)} relevant memories for query: '{query[:30]}...'")
            return memories
            
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            return []

    async def retrieve_intuition(self, stimulus: str) -> float:
        """
        Retrieve 'intuition' by checking if we have seen similar situations before.
        Returns a confidence score (0.0 - 1.0) based on distance to nearest neighbor.
        """
        try:
            results = self.collection.query(
                query_texts=[stimulus],
                n_results=1
            )
            
            if not results['distances'] or not results['distances'][0]:
                return 0.0
                
            # Chroma returns distance. Convert to similarity/confidence.
            # Cosine distance: 0 is identical, 1 is orthogonal.
            # We want 1.0 for identical.
            distance = results['distances'][0][0]
            confidence = max(0.0, 1.0 - distance)
            return confidence
            
        except Exception as e:
            logger.error(f"❌ Intuition retrieval failed: {e}")
            return 0.0

    async def forget_and_consolidate(self, bridge=None):
        """
        Executes the 'Sleep Cycle':
        1. Forgetting: Prune weak memories based on vitality (LRU + Age + Importance).
        2. Pruning: Use Neuro-Symbolic Bridge to remove 'Hallucinations' (High Drift, Low Surprise).
        3. Abstraction: Cluster dense memory regions into principles.
        """
        logger.info("[MemoryV2] Starting Sleep Cycle (Forgetting & Abstraction)...")
        
        try:
            # Get a batch of memories to check
            result = self.collection.get(limit=100, include=["metadatas", "documents"])
            ids = result['ids']
            metadatas = result['metadatas']
            documents = result['documents']
            
            if not ids:
                logger.info("   [Memory] No memories to consolidate.")
                return

            ids_to_delete = []
            clusters = {} # For abstraction
            
            for i, mem_id in enumerate(ids):
                meta = metadatas[i]
                
                # 1. Bridge Pruning
                if bridge:
                    concept_id = meta.get("concept_id") or mem_id
                    bridge_state = bridge.get_concept_state(concept_id)
                    if bridge_state == "REJECT_NOISE":
                         logger.info(f"   [Memory] ✂️ Pruning Hallucination (Bridge): {mem_id}")
                         ids_to_delete.append(mem_id)
                         continue 

                # 2. Vitality Logic
                access_count = meta.get("access_count", 0)
                score = meta.get("score", 0.0)
                age = time.time() - meta.get("timestamp", 0)
                
                # Simple logic: Old (>24h) + Unused + Low Score
                if age > 86400 and access_count < 2 and score < 0.3:
                     logger.info(f"   [Memory] 🍂 Forgetting weak memory: {mem_id}")
                     ids_to_delete.append(mem_id)
                     continue

                # 3. Collect for Abstraction (if high score)
                if score > 0.8:
                    clusters[mem_id] = documents[i]

            # Execute Deletion
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"   [Memory] 🗑️ Deleted {len(ids_to_delete)} memories.")
            
            # Execute Abstraction (Solidification)
            if len(clusters) > 3 and self.llm:
                logger.info(f"💎 Found {len(clusters)} high-value memories. Attempting Abstraction...")
                context_text = "\n".join(list(clusters.values())[:5])
                sys_prompt = "你是AGI的抽象思维模块。你的任务是将具体的记忆压缩为抽象的法则。"
                user_prompt = f"请阅读以下具体记忆，并提炼出一条普适的法则：\n{context_text}"
                
                # Async call
                loop = asyncio.get_running_loop()
                principle = await loop.run_in_executor(
                    None, 
                    lambda: self.llm.chat_completion(sys_prompt, user_prompt)
                )
                
                # Store Principle
                self.semantic_collection.add(
                    documents=[principle],
                    metadatas=[{"type": "crystallized_principle", "timestamp": time.time()}],
                    ids=[f"abs_{time.time()}"]
                )
                logger.info(f"✨ Crystallized Principle: {principle[:50]}...")

        except Exception as e:
            logger.error(f"❌ Sleep Cycle Error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Return database statistics."""
        return {
            "total_memories": self.collection.count(),
            "principles": self.semantic_collection.count(),
            "collection_name": self.collection_name,
            "persistence_dir": self.memory_dir
        }
