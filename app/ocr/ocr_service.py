import logging
import tempfile
import os
import cv2
from app.ocr.engine.main_ocr import OCRPipeline

logger = logging.getLogger(__name__)

# Create OCR pipeline instance
ocr_pipeline = OCRPipeline(debug=True, use_fast=False)

def run_ocr_with_fallback(image) -> dict:
    """
    Main OCR entry point for uploaded receipt images.
    Accepts OpenCV image, returns parsed receipt data.
    
    This maintains the exact same interface as your original function
    but uses the improved processing pipeline internally.
    """
    try:
        # Method 1: Try processing directly from image array
        logger.info("[OCR SERVICE] Attempting direct image processing")
        result = ocr_pipeline.process_image_array(image)
        
        if result.get("success", False):
            logger.info("[OCR SERVICE] Direct processing successful")
            return result
        
        # Method 2: Fallback to file-based processing
        logger.warning("[OCR SERVICE] Direct processing failed, trying file-based approach")
        
        # Save image temporarily for file-based processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
            temp_path = temp.name
            cv2.imwrite(temp_path, image)

        # Run file-based OCR pipeline
        logger.info(f"[OCR SERVICE] Processing temporary file: {temp_path}")
        result = ocr_pipeline.process_single_receipt(temp_path)

        # Clean up the temp file
        try:
            os.remove(temp_path)
            logger.debug(f"[OCR SERVICE] Cleaned up temporary file: {temp_path}")
        except Exception as e:
            logger.warning(f"[OCR SERVICE] Could not delete temp file {temp_path}: {e}")

        if result.get("success", False):
            logger.info("[OCR SERVICE] File-based processing successful")
            return result
        else:
            logger.error(f"[OCR SERVICE] File-based processing also failed: {result.get('error', 'Unknown error')}")
            return result

    except Exception as e:
        logger.exception("[OCR SERVICE] Critical failure in OCR pipeline")
        return {
            "success": False,
            "error": f"OCR service critical failure: {str(e)}",
            "raw_text": "",
            "parsed_data": {},
            "image_path": None,
            "confidence": 0.0,
            "ocr_attempts": 0,
            "processing_attempts": 0
        }

def validate_ocr_result(result: dict) -> dict:
    """
    Validate and enhance OCR result with additional quality checks.
    """
    if not result.get("success", False):
        return result
    
    parsed_data = result.get("parsed_data", {})
    items = parsed_data.get("items", [])
    total = parsed_data.get("total")
    
    # Quality validation
    quality_issues = []
    
    # Check if we have meaningful items
    if not items:
        quality_issues.append("no_items_found")
    elif len(items) == 1 and not items[0].get("name", "").strip():
        quality_issues.append("empty_item_names")
    
    # Check if total makes sense
    if total is None:
        quality_issues.append("no_total_found")
    elif total <= 0:
        quality_issues.append("invalid_total")
    elif items:
        calculated_total = sum(item.get("total", 0) for item in items)
        if calculated_total > 0 and abs(total - calculated_total) > 1.0:
            quality_issues.append("total_mismatch")
    
    # Check store name
    store = parsed_data.get("store", "")
    if not store or store == "Unknown Store":
        quality_issues.append("no_store_name")
    
    # Add quality assessment
    result["quality_issues"] = quality_issues
    result["quality_score"] = max(0, 1.0 - (len(quality_issues) * 0.2))
    
    # Adjust confidence based on quality issues
    original_confidence = result.get("confidence", 0.0)
    quality_penalty = len(quality_issues) * 0.1
    result["confidence"] = max(0, original_confidence - quality_penalty)
    
    if quality_issues:
        logger.warning(f"[OCR SERVICE] Quality issues detected: {', '.join(quality_issues)}")
    else:
        logger.info("[OCR SERVICE] High quality result - no major issues detected")
    
    return result

def run_ocr_with_validation(image) -> dict:
    """
    OCR with additional validation - alternative entry point.
    """
    result = run_ocr_with_fallback(image)
    return validate_ocr_result(result)

# Backward compatibility - keeping your original function name
def run_ocr_with_fallback_legacy(image) -> dict:
    """
    Legacy function name support - calls the improved version.
    """
    return run_ocr_with_fallback(image)