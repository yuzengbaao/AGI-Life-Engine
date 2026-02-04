import os
import sys
import logging

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.memory_enhanced import EnhancedExperienceMemory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MemoryMigration")

def migrate_legacy_data():
    """
    Migrate data from legacy JSON files (short_term.json, long_term.json) 
    to the new ChromaDB-backed Enhanced Memory.
    """
    logger.info("🚀 Starting Memory Migration Process...")
    
    # 1. Initialize Enhanced Memory (pointing to the REAL production DB)
    # Using 'memory_db' which we verified contains the 751 semantic memories
    memory_db_path = "memory_db" 
    memory = EnhancedExperienceMemory(memory_dir=memory_db_path, collection_name="semantic_memory")
    
    # 2. Define Legacy Paths
    legacy_paths = [
        "data/memory/short_term.json",
        "data/memory/long_term.json"
    ]
    
    # 3. Perform Migration
    total_migrated = 0
    for path in legacy_paths:
        full_path = os.path.abspath(path)
        if os.path.exists(full_path):
            logger.info(f"📂 Found legacy file: {full_path}")
            try:
                # Use the built-in consolidation method
                # Note: This method logs its own success/failure
                memory.consolidate_legacy_memory(full_path)
                logger.info(f"✅ Processed {path}")
            except Exception as e:
                logger.error(f"❌ Failed to process {path}: {e}")
        else:
            logger.warning(f"⚠️ Legacy file not found: {full_path}")
            
    # 4. Final Stats
    stats = memory.get_stats()
    logger.info(f"📊 Migration Complete. Final Database Stats: {stats}")

if __name__ == "__main__":
    migrate_legacy_data()
