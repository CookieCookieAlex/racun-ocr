import numpy as np
import cv2
from fastapi import UploadFile, HTTPException

async def load_image(file: UploadFile):
    """
    Reads and decodes an uploaded image file into OpenCV format.
    """
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Image could not be read")
    return img
