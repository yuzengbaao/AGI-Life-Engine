import os
import sys
import time
import logging
import argparse
import re
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory_enhanced_v2 import EnhancedExperienceMemory
from core.knowledge_graph import ArchitectureKnowledgeGraph
from core.llm_client import LLMService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ingest_knowledge.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("KnowledgeIngester")

class KnowledgeIngester:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.memory = EnhancedExperienceMemory()
        self.kg = ArchitectureKnowledgeGraph()
        self.llm = LLMService()
        
        # Files to process
        self.valid_extensions = {'.md', '.txt', '.py', '.json'}
        self.ignore_dirs = {
            '.git', '.venv', 'agi_env', '__pycache__', 'node_modules', 
            '.conda', 'backups', 'temp', 'logs', 'data', 'visualization', 'backbag'
        }
        
    def run(self):
        logger.info(f"🚀 Starting Knowledge Ingestion from: {self.root_dir}")
        logger.info(f"   - Memory: EnhancedExperienceMemory (ChromaDB)")
        logger.info(f"   - Graph: ArchitectureKnowledgeGraph (NetworkX)")
        
        count = 0
        new_nodes = 0
        
        # Walk through directories
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # Modify dirnames in-place to skip ignored directories
            dirnames[:] = [d for d in dirnames if d not in self.ignore_dirs and not d.startswith('.')]
            
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.valid_extensions:
                    file_path = os.path.join(dirpath, filename)
                    relative_path = os.path.relpath(file_path, self.root_dir)
                    
                    # Deduplication Check
                    node_id = f"FILE:{relative_path.replace(os.sep, '/')}"
                    
                    if self.kg.has_node(node_id):
                        logger.debug(f"⏭️  Skipping existing: {relative_path}")
                        continue
                        
                    # Process New File
                    if self._process_file(file_path, relative_path, node_id):
                        count += 1
                        new_nodes += 1
                        # Rate limit to be nice to LLM/Disk
                        if count % 10 == 0:
                            self.kg.save_graph()
                            logger.info(f"   [Checkpoint] Saved graph. Processed {count} new files.")

        self.kg.save_graph()
        logger.info(f"✅ Ingestion Complete.")
        logger.info(f"   - Processed Files: {count}")
        logger.info(f"   - New Nodes Added: {new_nodes}")
        stats = self.kg.get_stats()
        logger.info(f"   - Current Graph Stats: {stats}")

    def _process_file(self, file_path: str, relative_path: str, node_id: str) -> bool:
        try:
            # 1. Read Content
            size = os.path.getsize(file_path)
            if size == 0 or size > 100000: # Skip empty or very large files (>100KB)
                return False

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if not content.strip():
                return False

            logger.info(f"📥 Ingesting: {relative_path}")

            # 2. Add to Vector Memory (Semantic Search)
            # We add it as an "experience" of reading the code/doc
            self.memory.add_experience(
                context=f"Read file: {relative_path}\nContent Snippet: {content[:500]}...",
                action="IngestFile",
                outcome=1.0,
                details={
                    "file_path": relative_path,
                    "file_type": os.path.splitext(file_path)[1],
                    "content_length": len(content),
                    "full_content": content # Store full content in metadata for retrieval if needed (careful with size)
                }
            )

            # 3. Add to Knowledge Graph (Structural Relations)
            # Create the Document Node
            self.kg.graph.add_node(
                node_id,
                type="document",
                name=os.path.basename(file_path),
                path=relative_path,
                size=size,
                timestamp=datetime.now().isoformat()
            )

            # 4. Extract & Link Concepts (Simple Heuristics + Regex)
            self._extract_and_link(content, node_id, relative_path)

            return True

        except Exception as e:
            logger.error(f"❌ Error processing {relative_path}: {e}")
            return False

    def _extract_and_link(self, content: str, source_node_id: str, relative_path: str):
        # A. Python Imports Linking
        if relative_path.endswith('.py'):
            imports = re.findall(r'^(?:from|import)\s+([\w\.]+)', content, re.MULTILINE)
            for imp in imports:
                top_level = imp.split('.')[0]
                # Link to a Concept Node for the library
                lib_node_id = f"LIB:{top_level}"
                if not self.kg.has_node(lib_node_id):
                    self.kg.graph.add_node(lib_node_id, type="library", name=top_level)
                
                self.kg.graph.add_edge(source_node_id, lib_node_id, relation="imports")

        # B. Markdown Links (Internal Knowledge)
        if relative_path.endswith('.md'):
            # [[Link]] style or [Link](path)
            # Simple heuristic for key terms in headers
            headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
            for header in headers:
                concept = header.strip()
                if len(concept) < 50:
                    concept_id = f"CONCEPT:{concept.lower().replace(' ', '_')}"
                    if not self.kg.has_node(concept_id):
                        self.kg.graph.add_node(concept_id, type="concept", name=concept)
                    self.kg.graph.add_edge(source_node_id, concept_id, relation="defines_concept")

if __name__ == "__main__":
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ingester = KnowledgeIngester(root_path)
    ingester.run()
