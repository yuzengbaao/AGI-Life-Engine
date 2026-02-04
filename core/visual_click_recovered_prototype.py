from typing import Optional, Tuple, Union
import time
import logging

# Configure logger for consistent logging
logger = logging.getLogger(__name__)

# Assume these are imported from other modules
try:
    from vision_observer import locate_on_screen  # Provides image locating functionality
    from desktop_automation import click_at_position  # Performs the actual click
except ImportError as e:
    logger.critical(f"Failed to import required modules: {e}")
    raise


def visual_click_with_recovery(
    target_image_path: str,
    max_retries: int = 3,
    confidence: float = 0.8,
    region: Optional[Tuple[int, int, int, int]] = None,
    grayscale: bool = False
) -> bool:
    """
    Perform a vision-based click with built-in error detection and automatic retry mechanism.
    
    Args:
        target_image_path (str): Path to the target image template.
        max_retries (int): Maximum number of retry attempts (default: 3).
        confidence (float): Confidence threshold for image matching [0, 1] (default: 0.8).
        region (Optional[Tuple[int, int, int, int]]): Search region as (x, y, width, height). 
            If None, searches full screen.
        grayscale (bool): Whether to convert screenshots to grayscale for faster matching.

    Returns:
        bool: True if click succeeds within retries, False otherwise.

    Raises:
        ValueError: If inputs are out of valid range or malformed.
        RuntimeError: If critical dependencies are unavailable.

    Notes:
        This function integrates with `vision_observer.locate_on_screen` to find the target image.
        On failure, it waits and optionally applies recovery steps before retrying.
        Recovery strategies (e.g., window refocus, scroll, popup handling) can be extended in future.
    """
    # Input validation
    if not isinstance(target_image_path, str) or not target_image_path.strip():
        raise ValueError("target_image_path must be a non-empty string.")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative.")
    if not (0.0 <= confidence <= 1.0):
        raise ValueError("confidence must be between 0.0 and 1.0.")
    if region is not None:
        if not isinstance(region, tuple) or len(region) != 4:
            raise ValueError("region must be a tuple of (x, y, width, height).")
        if any(not isinstance(coord, int) or coord < 0 for coord in region):
            raise ValueError("region coordinates must be non-negative integers.")

    for attempt in range(1, max_retries + 1):
        try:
            position = locate_on_screen(
                image_path=target_image_path,
                confidence=confidence,
                region=region,
                grayscale=grayscale
            )

            if position:
                success = click_at_position(position)
                if success:
                    logger.info(f"Visual click successful at {position}, attempt {attempt}")
                    return True
                else:
                    logger.warning(f"Click failed at position {position}, even though image was found.")
            else:
                logger.info(f"Target image '{target_image_path}' not found during attempt {attempt}")

        except Exception as e:
            logger.error(f"Exception during visual click attempt {attempt}: {type(e).__name__}: {e}")

        # Decide whether to retry
        if attempt < max_retries:
            time.sleep(1)
            # Optional recovery logic placeholder (can be expanded)
            # Example: recover_from_failure(attempt)
        else:
            logger.error(f"Max retries ({max_retries}) exceeded for target: {target_image_path}")

    return False