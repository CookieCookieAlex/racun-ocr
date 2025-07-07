from fastapi import APIRouter, File, UploadFile
import numpy as np
import cv2
import pytesseract

from app.ocr import processor, parser

router = APIRouter()

@router.post("/ocr/")
async def ocr_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Failed to load image"}

    processor.save_debug_image(img, "original")

    edged = processor.sharpen_edge(img)
    processor.save_debug_image(edged, "edged")

    bin_img = processor.binarize(edged)
    processor.save_debug_image(bin_img, "binarized")

    cnts = cv2.findContours(bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if isinstance(cnts, tuple) else cnts
    if not cnts:
        return {"error": "No contours found"}

    largest = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) < 4:
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="float32")
    elif len(approx) == 4:
        box = approx.reshape(4, 2).astype("float32")
    else:
        hull = cv2.convexHull(largest)
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="float32")

    img_box = img.copy()
    cv2.drawContours(img_box, [box.astype(int)], -1, (0, 255, 0), 2)
    processor.save_debug_image(img_box, "box")

    cropped = processor.crop_by_corners(img, box)
    processor.save_debug_image(cropped, "cropped")

    text_img = processor.enhance_text(cropped)
    processor.save_debug_image(text_img, "text_enhanced")

    ocr_text = pytesseract.image_to_string(text_img, lang='eng+deu+slk+ces')

    lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]
    result = parser.parse_receipt(lines)

    return result
