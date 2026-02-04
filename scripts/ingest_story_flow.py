import os
import sys
import re
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm_client import LLMService
from knowledge_graph import KnowledgeGraph
from entity_extractor import Entity, EntityType
from relationship_builder import Relationship, RelationshipType

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ingest_flow.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StoryIngester")

class StoryIngester:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.kg = KnowledgeGraph()
        self.llm = LLMService()
        self.output_file = os.path.join(self.root_dir, "data", "knowledge_graph.json")
        
        # Load existing graph if available
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.kg.from_dict(data)
                logger.info(f"Loaded existing graph from {self.output_file}")
            except Exception as e:
                logger.warning(f"Failed to load existing graph: {e}")

    def scan_and_ingest(self, max_files=50):
        """
        Scans the directory and builds the knowledge graph.
        limit: max files to process to avoid long run times in this demo.
        """
        logger.info(f"Starting scan of {self.root_dir}...")
        count = 0
        
        # Priority: core/, scripts/, docs/
        target_dirs = ['core', 'scripts', 'docs']
        
        files_to_process = []
        for d in target_dirs:
            path = os.path.join(self.root_dir, d)
            if os.path.exists(path):
                for root, _, files in os.walk(path):
                    if "__pycache__" in root: continue
                    for file in files:
                        if file.endswith(('.py', '.md', '.txt')):
                            files_to_process.append(os.path.join(root, file))
        
        # Also add root files
        for file in os.listdir(self.root_dir):
            if os.path.isfile(os.path.join(self.root_dir, file)) and file.endswith(('.py', '.md', '.txt')):
                 files_to_process.append(os.path.join(self.root_dir, file))

        logger.info(f"Found {len(files_to_process)} candidate files.")

        for file_path in files_to_process[:max_files]:
            self._process_file(file_path)
            count += 1
            if count % 10 == 0:
                self._save_graph()
        
        self._save_graph()
        logger.info(f"Ingestion complete. Processed {count} files.")
        
    def _save_graph(self):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(self.kg.to_json())
        logger.info(f"Graph saved to {self.output_file}")

    def _process_file(self, file_path):
        logger.info(f"Processing {file_path}")
        relative_path = os.path.relpath(file_path, self.root_dir)
        file_name = os.path.basename(file_path)
        
        # 1. Create File Node
        # We use PRODUCT type as a proxy for 'File' or 'Module'
        file_entity = Entity(
            name=relative_path, # Unique ID
            entity_type=EntityType.PRODUCT,
            confidence=1.0,
            metadata={"path": file_path, "filename": file_name, "type": "file"}
        )
        self.kg.add_node(file_entity)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return

        # 2. Extract Python Imports (Dependencies)
        if file_path.endswith(".py"):
            self._extract_python_imports(content, relative_path)
            
        # 3. Extract Concepts (LLM) - only for manageable size
        # We skip very large files or auto-generated ones
        if len(content) < 5000 and "test" not in file_name.lower():
             self._extract_concepts(content, relative_path)

    def _extract_python_imports(self, content, source_node_name):
        # Regex for 'import X' or 'from X import Y'
        # Simple regex to catch top level modules
        imports = set()
        
        # import x, y
        matches = re.findall(r'^import\s+([\w,\s]+)', content, re.MULTILINE)
        for match in matches:
            for item in match.split(','):
                imports.add(item.strip().split('.')[0]) # Take top level package
                
        # from x import y
        matches = re.findall(r'^from\s+([\w\.]+)\s+import', content, re.MULTILINE)
        for match in matches:
            imports.add(match.split('.')[0])
            
        for imp in imports:
            if not imp: continue
            
            # Create dependency node
            # We treat external libs as TECHNOLOGY
            dep_entity = Entity(
                name=imp,
                entity_type=EntityType.TECHNOLOGY,
                confidence=0.9,
                metadata={"type": "library"}
            )
            self.kg.add_node(dep_entity)
            
            # Add edge: File -> DependsOn -> Lib
            rel = Relationship(
                source=source_node_name,
                target=imp,
                relation_type=RelationshipType.DEPENDS_ON,
                confidence=1.0
            )
            try:
                self.kg.add_edge(rel)
            except Exception as e:
                pass # Ignore duplicates or self-loops silently for now

    def _extract_concepts(self, content, source_node_name):
        # Use LLM to extract concepts
        # We'll use a simple prompt
        prompt = (
            "Identify 3 key technical concepts or architectural components in this code. "
            "Return ONLY a comma-separated list of terms. No explanation."
        )
        
        try:
            # Truncate content
            snippet = content[:2000]
            response = self.llm.chat_completion(prompt, snippet)
            
            if "[MOCK" in response or "[LLM ERROR]" in response:
                # Fallback to simple keyword extraction if LLM fails/mocked
                return 

            concepts = [c.strip() for c in response.split(',') if c.strip()]
            
            for concept in concepts:
                # Clean concept
                concept = re.sub(r'[^\w\s-]', '', concept)
                if len(concept) < 3 or len(concept) > 50: continue
                
                # Create Concept Node
                concept_entity = Entity(
                    name=concept,
                    entity_type=EntityType.CONCEPT,
                    confidence=0.7
                )
                self.kg.add_node(concept_entity)
                
                # Add edge: File -> RelatedTo -> Concept
                rel = Relationship(
                    source=source_node_name,
                    target=concept,
                    relation_type=RelationshipType.RELATED_TO,
                    confidence=0.7
                )
                try:
                    self.kg.add_edge(rel)
                except Exception as e:
                    pass
                    
        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")

if __name__ == "__main__":
    # Assuming script is in scripts/, root is one level up
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ingester = StoryIngester(root)
    ingester.scan_and_ingest(max_files=30) # Scan 30 files for demo
import os
import sys
import re
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm_client import LLMService
from knowledge_graph import KnowledgeGraph
from entity_extractor import Entity, EntityType
from relationship_builder import Relationship, RelationshipType

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ingest_flow.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StoryIngester")

class StoryIngester:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.kg = KnowledgeGraph()
        self.llm = LLMService()
        self.output_file = os.path.join(self.root_dir, "data", "knowledge_graph.json")
        
        # Load existing graph if available
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.kg.from_dict(data)
                logger.info(f"Loaded existing graph from {self.output_file}")
            except Exception as e:
                logger.warning(f"Failed to load existing graph: {e}")

    def scan_and_ingest(self, max_files=50):
        """
        Scans the directory and builds the knowledge graph.
        limit: max files to process to avoid long run times in this demo.
        """
        logger.info(f"Starting scan of {self.root_dir}...")
        count = 0
        
        # Priority: core/, scripts/, docs/
        target_dirs = ['core', 'scripts', 'docs']
        
        files_to_process = []
        for d in target_dirs:
            path = os.path.join(self.root_dir, d)
            if os.path.exists(path):
                for root, _, files in os.walk(path):
                    if "__pycache__" in root: continue
                    for file in files:
                        if file.endswith(('.py', '.md', '.txt')):
                            files_to_process.append(os.path.join(root, file))
        
        # Also add root files
        for file in os.listdir(self.root_dir):
            if os.path.isfile(os.path.join(self.root_dir, file)) and file.endswith(('.py', '.md', '.txt')):
                 files_to_process.append(os.path.join(self.root_dir, file))

        logger.info(f"Found {len(files_to_process)} candidate files.")

        for file_path in files_to_process[:max_files]:
            self._process_file(file_path)
            count += 1
            if count % 10 == 0:
                self._save_graph()
        
        self._save_graph()
        logger.info(f"Ingestion complete. Processed {count} files.")
        
    def _save_graph(self):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(self.kg.to_json())
        logger.info(f"Graph saved to {self.output_file}")

    def _process_file(self, file_path):
        logger.info(f"Processing {file_path}")
        relative_path = os.path.relpath(file_path, self.root_dir)
        file_name = os.path.basename(file_path)
        
        # 1. Create File Node
        # We use PRODUCT type as a proxy for 'File' or 'Module'
        file_entity = Entity(
            name=relative_path, # Unique ID
            entity_type=EntityType.PRODUCT,
            confidence=1.0,
            metadata={"path": file_path, "filename": file_name, "type": "file"}
        )
        self.kg.add_node(file_entity)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return

        # 2. Extract Python Imports (Dependencies)
        if file_path.endswith(".py"):
            self._extract_python_imports(content, relative_path)
            
        # 3. Extract Concepts (LLM) - only for manageable size
        # We skip very large files or auto-generated ones
        if len(content) < 5000 and "test" not in file_name.lower():
             self._extract_concepts(content, relative_path)

    def _extract_python_imports(self, content, source_node_name):
        # Regex for 'import X' or 'from X import Y'
        # Simple regex to catch top level modules
        imports = set()
        
        # import x, y
        matches = re.findall(r'^import\s+([\w,\s]+)', content, re.MULTILINE)
        for match in matches:
            for item in match.split(','):
                imports.add(item.strip().split('.')[0]) # Take top level package
                
        # from x import y
        matches = re.findall(r'^from\s+([\w\.]+)\s+import', content, re.MULTILINE)
        for match in matches:
            imports.add(match.split('.')[0])
            
        for imp in imports:
            if not imp: continue
            
            # Create dependency node
            # We treat external libs as TECHNOLOGY
            dep_entity = Entity(
                name=imp,
                entity_type=EntityType.TECHNOLOGY,
                confidence=0.9,
                metadata={"type": "library"}
            )
            self.kg.add_node(dep_entity)
            
            # Add edge: File -> DependsOn -> Lib
            rel = Relationship(
                source=source_node_name,
                target=imp,
                relation_type=RelationshipType.DEPENDS_ON,
                confidence=1.0
            )
            try:
                self.kg.add_edge(rel)
            except Exception as e:
                pass # Ignore duplicates or self-loops silently for now

    def _extract_concepts(self, content, source_node_name):
        # Use LLM to extract concepts
        # We'll use a simple prompt
        prompt = (
            "Identify 3 key technical concepts or architectural components in this code. "
            "Return ONLY a comma-separated list of terms. No explanation."
        )
        
        try:
            # Truncate content
            snippet = content[:2000]
            response = self.llm.chat_completion(prompt, snippet)
            
            if "[MOCK" in response or "[LLM ERROR]" in response:
                # Fallback to simple keyword extraction if LLM fails/mocked
                return 

            concepts = [c.strip() for c in response.split(',') if c.strip()]
            
            for concept in concepts:
                # Clean concept
                concept = re.sub(r'[^\w\s-]', '', concept)
                if len(concept) < 3 or len(concept) > 50: continue
                
                # Create Concept Node
                concept_entity = Entity(
                    name=concept,
                    entity_type=EntityType.CONCEPT,
                    confidence=0.7
                )
                self.kg.add_node(concept_entity)
                
                # Add edge: File -> RelatedTo -> Concept
                rel = Relationship(
                    source=source_node_name,
                    target=concept,
                    relation_type=RelationshipType.RELATED_TO,
                    confidence=0.7
                )
                try:
                    self.kg.add_edge(rel)
                except Exception as e:
                    pass
                    
        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")

if __name__ == "__main__":
    # Assuming script is in scripts/, root is one level up
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ingester = StoryIngester(root)
    ingester.scan_and_ingest(max_files=30) # Scan 30 files for demo
import os
import sys
import time
import logging

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import ExperienceMemory
from core.llm_client import LLMService
from core.knowledge_graph import ArchitectureKnowledgeGraph
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StoryIngester:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.memory = ExperienceMemory()
        self.llm = LLMService()
        self.graph = ArchitectureKnowledgeGraph()
        self.valid_extensions = {'.md', '.txt', '.py', '.json'}
        self.ignore_dirs = {'.git', '.venv', '__pycache__', 'node_modules', '.conda', 'backups', 'temp'}

    def ingest(self):
        logger.info(f"Starting TOPOLOGICAL Story Flow Ingestion from: {self.root_dir}")
        count = 0
        
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # Filter ignore dirs
            dirnames[:] = [d for d in dirnames if d not in self.ignore_dirs and not d.startswith('.')]
            
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.valid_extensions:
                    file_path = os.path.join(dirpath, filename)
                    if self._process_file(file_path):
                        count += 1
                        # Rate limit
                        time.sleep(1) 
        
        logger.info(f"Ingestion Complete. Processed {count} documents.")

    def _process_file(self, file_path):
        try:
            # Check file size
            size = os.path.getsize(file_path)
            if size == 0 or size > 50000:
                return False

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if not content.strip():
                return False

            logger.info(f"Processing: {os.path.basename(file_path)}")

            # 1. Extract Narrative & Topology
            analysis = self._analyze_content(os.path.basename(file_path), content)
            
            if not analysis:
                return False

            narrative = analysis.get("narrative", "")
            concepts = analysis.get("concepts", [])
            references = analysis.get("references", [])

            # 2. Store in Vector Memory (The Content)
            context = f"Document Story: {os.path.basename(file_path)}\nNarrative: {narrative}"
            self.memory.add_experience(
                context=context,
                action="IngestStoryFlow",
                outcome=1.0,
                details={
                    "file_path": file_path,
                    "file_type": os.path.splitext(file_path)[1],
                    "timestamp": time.time()
                }
            )

            # 3. Store in Knowledge Graph (The Web)
            node_id = self.graph.add_decision_node(
                context=f"Ingested {os.path.basename(file_path)}",
                decision="Knowledge Integration",
                outcome=1.0,
                metadata={
                    "file_path": file_path,
                    "narrative": narrative,
                    "type": "document_node"
                }
            )
            
            # 4. Create Edges (Weaving the Web)
            # Link to concepts
            for concept in concepts:
                # Naive concept linking: Create concept node if not exists, then link
                concept_id = f"CONCEPT_{hash(concept)}"
                if not self.graph.has_node(concept_id):
                    self.graph.graph.add_node(concept_id, type="concept", name=concept)
                self.graph.graph.add_edge(node_id, concept_id, relation="relates_to")
                
            # Link to referenced files (Code Imports / Markdown Links)
            # Simple heuristic for imports
            if file_path.endswith(".py"):
                imports = re.findall(r'^(?:from|import)\s+(\w+)', content, re.MULTILINE)
                for imp in imports:
                    # Try to find if this import corresponds to a known file node?
                    # For now, just create a "dependency" concept
                    dep_id = f"DEPENDENCY_{imp}"
                    if not self.graph.has_node(dep_id):
                        self.graph.graph.add_node(dep_id, type="dependency", name=imp)
                    self.graph.graph.add_edge(node_id, dep_id, relation="imports")

            self.graph.save_graph()
            return True

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False

    def _analyze_content(self, filename, content):
        """
        Uses LLM to extract Narrative AND Topological Connections.
        Returns dict: {narrative: str, concepts: [str], references: [str]}
        """
        system_prompt = (
            "You are the 'Knowledge Weaver' of an AGI system. "
            "Read the document and extract structured intelligence.\n"
            "Return a JSON object with:\n"
            "- 'narrative': A concise summary of the logic/story (under 100 words).\n"
            "- 'concepts': A list of 3-5 key abstract concepts (e.g., 'Evolution', 'Memory', 'Recursion').\n"
            "- 'references': A list of other files or modules mentioned or implied."
        )
        
        user_prompt = f"Document: {filename}\nContent:\n{content[:2000]}"
        
        try:
            # Force JSON response structure if possible, or parse text
            response = self.llm.chat_completion(system_prompt, user_prompt)
            
            # Simple parsing attempt (robustness needed for real prod)
            import json
            # Try to find JSON block
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                # Fallback
                return {"narrative": response, "concepts": [], "references": []}
        except Exception:
            return None

if __name__ == "__main__":
    # Target the project root
    target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ingester = StoryIngester(target_dir)
    ingester.ingest()