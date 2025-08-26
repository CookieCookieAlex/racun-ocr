import os
import cv2
import logging
from typing import Any, Dict
from app.ocr.processor.core import ReceiptProcessor  # Import your improved processor
from app.ocr.parser import ReceiptParser
from app.ocr.engine.ocr_engine import run_ocr_on_image  # Import your improved OCR engine
from app.ocr.engine.scoring import score_ocr_text

logger = logging.getLogger(__name__)
DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

class OCRPipeline:
    def __init__(self, debug: bool = True, use_fast: bool = False):
        self.debug = debug
        self.use_fast = use_fast
        self.processor = ReceiptProcessor(debug=debug)
        self.parser = ReceiptParser()

    def process_single_receipt(self, image_path: str) -> Dict[str, Any]:
        """
        Main processing function - maintains same interface as your original.
        Enhanced with better error handling and processing pipeline.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")

            logger.info(f"[MAIN] Processing {image_path}")
            
            # Step 1: Process image with improved processor
            processed_images, status, processing_info = self.processor.process_receipt(img)
            
            if not processed_images:
                logger.error(f"[MAIN] No processed images returned from processor")
                return {
                    "success": False,
                    "error": "Image processing failed - no processed images",
                    "image_path": image_path,
                    "process_status": status
                }

            # Step 2: Run OCR on all processed images
            best_text = ""
            best_score = -1
            best_info = ("", -1, "")
            all_ocr_attempts = []

            logger.info(f"[OCR] Running OCR on {len(processed_images)} processed images")

            for i, p_img in enumerate(processed_images):
                logger.debug(f"[OCR] Processing image variant {i+1}/{len(processed_images)}")
                
                # Run enhanced OCR
                ocr_results = run_ocr_on_image(
                    p_img, 
                    debug=self.debug, 
                    image_index=i,
                    lang='hrv'  # Start with English only
                )
                all_ocr_attempts.extend(ocr_results)

                # Evaluate each OCR result
                for text, config_idx, config_str in ocr_results:
                    if not text or len(text.strip()) < 10:
                        continue
                    
                    score = score_ocr_text(text)
                    
                    if score > best_score:
                        best_score = score
                        best_text = text
                        best_info = (f"image_{i}", config_idx, config_str)
                        
                        if self.debug:
                            logger.info(f"[OCR] New best result (score {score}): "
                                      f"config='{config_str}', preview='{text[:80]}...'")

            # Step 3: Validate OCR results
            if not best_text or len(best_text.strip()) < 15:
                logger.warning(f"[OCR] Insufficient text extracted from {len(processed_images)} images")
                logger.warning(f"[OCR] Tried {len(all_ocr_attempts)} OCR configurations")
                
                # Try to provide more helpful error info
                if all_ocr_attempts:
                    longest_text = max(all_ocr_attempts, key=lambda x: len(x[0]))[0]
                    logger.info(f"[OCR] Longest text found: '{longest_text[:100]}...' ({len(longest_text)} chars)")
                
                return {
                    "success": False,
                    "error": f"OCR extraction failed - insufficient text (best: {len(best_text)} chars)",
                    "image_path": image_path,
                    "process_status": status,
                    "processing_info": processing_info,
                    "ocr_attempts": len(all_ocr_attempts),
                    "best_text_preview": best_text[:100] if best_text else ""
                }
            
            logger.info(f"[OCR] Best OCR result: {best_info[2]} (score: {best_score}, length: {len(best_text)})")
            
            # Step 4: Save debug information
            if self.debug:
                self._save_debug_info(processed_images, best_info, best_text, best_score)
            
            # Step 5: Parse the extracted text
            lines = [line.strip() for line in best_text.split("\n") if line.strip()]
            logger.debug(f"[PARSE] Parsing {len(lines)} text lines")
            
            parsed = self.parser.parse_receipt(lines, log_debug=self.debug)
            
            # Step 6: Calculate overall confidence
            ocr_confidence = min(best_score / 50.0, 1.0)  # Normalize to 0-1
            parsing_confidence = parsed.get("confidence", 0.0)
            overall_confidence = (ocr_confidence * 0.6) + (parsing_confidence * 0.4)
            
            # Step 7: Validate parsed results
            items_count = len(parsed.get('items', []))
            has_total = parsed.get('total') is not None
            has_store = parsed.get('store') and parsed.get('store') != "Unknown Store"
            
            logger.info(f"[PARSE] Results: {items_count} items, "
                       f"total: {parsed.get('total')}, "
                       f"store: '{parsed.get('store')}'")
            
            # Step 8: Build final result
            result = {
                "success": True,
                "image_path": image_path,
                "process_status": status,
                "processing_info": processing_info,
                "ocr_versions_tried": len(all_ocr_attempts),
                "raw_text": best_text,
                "parsed_data": parsed,
                "preprocessing_attempts": len(processed_images),
                "confidence": overall_confidence,
                "ocr_confidence": ocr_confidence,
                "parsing_confidence": parsing_confidence,
                "best_ocr_config": best_info[2],
                "ocr_score": best_score,
                
                # Quality indicators
                "quality_indicators": {
                    "has_items": items_count > 0,
                    "has_total": has_total,
                    "has_store": has_store,
                    "items_count": items_count,
                    "text_length": len(best_text),
                    "lines_count": len(lines)
                }
            }
            
            # Log success summary
            if overall_confidence > 0.7:
                logger.info(f"[MAIN] HIGH QUALITY result: confidence {overall_confidence:.2f}")
            elif overall_confidence > 0.4:
                logger.info(f"[MAIN] MEDIUM QUALITY result: confidence {overall_confidence:.2f}")
            else:
                logger.warning(f"[MAIN] LOW QUALITY result: confidence {overall_confidence:.2f}")
            
            return result

        except Exception as e:
            logger.error(f"[MAIN] Critical error processing {image_path}: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Critical processing error: {str(e)}",
                "image_path": image_path,
                "confidence": 0.0,
                "ocr_attempts": 0,
                "processing_attempts": 0
            }

    def _save_debug_info(self, processed_images, best_info, best_text, best_score):
        """Save debug information for analysis."""
        try:
            # Save the best OCR source image
            if best_info[0].startswith("image_"):
                index = int(best_info[0].split("_")[-1])
                if 0 <= index < len(processed_images):
                    best_img = processed_images[index]
                    cv2.imwrite(os.path.join(DEBUG_DIR, "final_best_ocr_source.png"), best_img)
            
            # Save OCR text result
            with open(os.path.join(DEBUG_DIR, "best_ocr_result.txt"), 'w', encoding='utf-8') as f:
                f.write(f"OCR Score: {best_score}\n")
                f.write(f"Config: {best_info[2]}\n")
                f.write(f"Source: {best_info[0]}\n")
                f.write(f"Text Length: {len(best_text)} characters\n")
                f.write(f"Lines: {len(best_text.splitlines())}\n")
                f.write("\n" + "="*50 + "\n")
                f.write("EXTRACTED TEXT:\n")
                f.write("="*50 + "\n")
                f.write(best_text)
                
        except Exception as e:
            logger.warning(f"[DEBUG] Could not save debug info: {e}")

    def process_image_array(self, image: any) -> Dict[str, Any]:
        """
        Process image from array/OpenCV format (for API usage).
        """
        try:
            logger.info("[MAIN] Processing image from array")
            
            # Step 1: Process image 
            processed_images, status, processing_info = self.processor.process_receipt(image)
            
            if not processed_images:
                return {
                    "success": False,
                    "error": "Image processing failed",
                    "process_status": status
                }

            # Step 2: Run OCR (same as file processing)
            best_text = ""
            best_score = -1
            all_ocr_attempts = []

            for i, p_img in enumerate(processed_images):
                ocr_results = run_ocr_on_image(p_img, debug=self.debug, image_index=i)
                all_ocr_attempts.extend(ocr_results)

                for text, config_idx, config_str in ocr_results:
                    if not text or len(text.strip()) < 10:
                        continue
                    
                    score = score_ocr_text(text)
                    if score > best_score:
                        best_score = score
                        best_text = text
            
            if not best_text or len(best_text.strip()) < 15:
                return {
                    "success": False,
                    "error": "OCR extraction failed - insufficient text",
                    "process_status": status,
                    "ocr_attempts": len(all_ocr_attempts)
                }
            
            # Step 3: Parse text
            lines = [line.strip() for line in best_text.split("\n") if line.strip()]
            parsed = self.parser.parse_receipt(lines, log_debug=self.debug)
            
            # Step 4: Calculate confidence
            ocr_confidence = min(best_score / 50.0, 1.0)
            parsing_confidence = parsed.get("confidence", 0.0)
            overall_confidence = (ocr_confidence * 0.6) + (parsing_confidence * 0.4)
            
            return {
                "success": True,
                "process_status": status,
                "raw_text": best_text,
                "parsed_data": parsed,
                "confidence": overall_confidence,
                "ocr_confidence": ocr_confidence,
                "parsing_confidence": parsing_confidence,
                "ocr_attempts": len(all_ocr_attempts)
            }
            
        except Exception as e:
            logger.error(f"[MAIN] Error processing image array: {e}")
            return {
                "success": False,
                "error": str(e),
                "confidence": 0.0
            }