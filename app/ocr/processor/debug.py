import os
import cv2
import logging

logger = logging.getLogger(__name__)
DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

def save_debug_image(image, name, debug=True):
    if not name.lower().endswith(".png"):
        name += ".png"
    if debug:
        path = os.path.join(DEBUG_DIR, name)
        cv2.imwrite(path, image)
        logger.debug(f"[DEBUG] Saved {path}")
