import pytesseract
import logging
import cv2

from app.ocr import processor


def run_ocr_with_fallback(processed_img, original_img):
    """
    Runs OCR on the processed image.
    If text is too weak, tries fallback on raw cropped receipt.
    """
    ocr_text = pytesseract.image_to_string(processed_img, lang='hrv')
    lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]

    # Fallback if OCR text is too short
    if len(lines) < 5:
        logging.info("Fallback: OCR result weak, trying raw cropped image")
        contour = processor.find_receipt_contour(original_img)
        if contour is not None:
            cropped = processor.crop_by_corners(original_img, contour)
            ocr_raw = pytesseract.image_to_string(cropped, lang='hrv')
            lines_raw = [line.strip() for line in ocr_raw.split("\n") if line.strip()]
            if len(lines_raw) > len(lines):
                lines = lines_raw

    if not lines:
        raise ValueError("No text detected in image")

    return lines
