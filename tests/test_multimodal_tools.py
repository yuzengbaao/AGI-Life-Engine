
import asyncio
import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agi_chat_enhanced import AGIChatInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_multimodal_tools():
    logger.info("Initializing AGIChatInterface...")
    chat_interface = AGIChatInterface(active_mode=False)
    
    logger.info("Testing capture_webcam tool...")
    # Directly call the handler
    result_cam = await chat_interface._handle_capture_webcam(filename="test_webcam.jpg")
    logger.info(f"Webcam Result: {result_cam}")
    
    if result_cam.get('status') == 'success':
        logger.info(f"✅ Webcam test passed. File saved at: {result_cam.get('file_path')}")
    else:
        logger.warning(f"⚠️ Webcam test failed (expected if no camera): {result_cam.get('error')}")

    logger.info("Testing record_audio tool...")
    # Directly call the handler
    result_audio = await chat_interface._handle_record_audio(duration=2, filename="test_audio.wav")
    logger.info(f"Audio Result: {result_audio}")
    
    if result_audio.get('status') == 'success':
        logger.info(f"✅ Audio test passed. File saved at: {result_audio.get('file_path')}")
    else:
        logger.warning(f"⚠️ Audio test failed (expected if no mic): {result_audio.get('error')}")

if __name__ == "__main__":
    asyncio.run(test_multimodal_tools())
