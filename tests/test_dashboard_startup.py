
import sys
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("Testing agi_visual_dashboard import...")
    from agi_visual_dashboard import VisualDashboard
    logger.info("Import successful.")

    class MockAGI:
        def __init__(self):
            self.initialization_time = time.time()
            self.status = type('obj', (object,), {'active_modules': ['MockModule']})
            self.task_queue = []
    
    mock_agi = MockAGI()
    
    logger.info("Initializing VisualDashboard...")
    dashboard = VisualDashboard(mock_agi, port=5005) # Use different port for test
    
    logger.info("Starting dashboard...")
    dashboard.start()
    
    time.sleep(2)
    
    if dashboard.server_thread.is_alive():
        logger.info("Dashboard thread is running.")
    else:
        logger.error("Dashboard thread died.")
        sys.exit(1)
        
    logger.info("Stopping dashboard (mock)...")
    dashboard.stop()
    # Note: Flask server doesn't stop easily, so we just exit the process
    logger.info("Test passed.")
    
except Exception as e:
    logger.error(f"Test failed: {e}")
    sys.exit(1)
