import logging
import tempfile
import cv2
from app.ocr import processor
from app.ocr.main_ocr import OCRPipeline

logger = logging.getLogger(__name__)
pipeline = OCRPipeline(debug=True)


def run_ocr_with_fallback(processed_img, original_img):
    """
    Run OCR using the full OCRPipeline. Falls back to cropped raw image
    if the main OCR result is weak (less than 5 lines).
    """
    try:
        # Save original image temporarily
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            cv2.imwrite(tmp.name, original_img)
            result = pipeline.process_single_receipt(tmp.name)

        if not result["success"]:
            raise ValueError("Main OCR pipeline failed")

        # Get lines from raw text
        lines = result["raw_text"].splitlines()
        lines = [line.strip() for line in lines if line.strip()]

        if len(lines) >= 5:
            return lines

        # Fallback: Try cropped image
        logger.info("Fallback: OCR result weak, trying cropped raw image")
        contour = processor.find_receipt_contour(original_img)
        if contour is not None:
            cropped = processor.crop_by_corners(original_img, contour)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_crop:
                cv2.imwrite(tmp_crop.name, cropped)
                fallback_result = pipeline.process_single_receipt(tmp_crop.name)

            if fallback_result["success"]:
                fallback_lines = fallback_result["raw_text"].splitlines()
                fallback_lines = [line.strip() for line in fallback_lines if line.strip()]
                if len(fallback_lines) > len(lines):
                    return fallback_lines

        return lines

    except Exception as e:
        logger.error(f"OCR pipeline error: {e}")
        raise ValueError("OCR failed")
