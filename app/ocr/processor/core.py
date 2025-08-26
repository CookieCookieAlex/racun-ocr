import cv2
import numpy as np
import logging
import os
from app.ocr.processor.contour import ReceiptContourDetector
from app.ocr.processor.cropper import ReceiptCropper

logger = logging.getLogger(__name__)
DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

class ReceiptProcessor:
    def __init__(self, debug=True):
        self.debug = debug
        # Use the improved contour detector and dedicated cropper
        self.contour_detector = ReceiptContourDetector(debug=debug)
        self.cropper = ReceiptCropper(debug=debug)

    def save_debug_image(self, image, name):
        if self.debug:
            path = os.path.join(DEBUG_DIR, name if name.endswith(".png") else name + ".png")
            cv2.imwrite(path, image)
            logger.debug(f"[DEBUG] Saved {path}")

    def detect_receipt_orientation(self, image):
        """Detect and correct receipt orientation"""
        h, w = image.shape[:2]
        
        # If landscape, rotate to portrait
        if w > h:
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            self.save_debug_image(rotated, "step1_orientation_corrected.png")
            return rotated
        
        return image

    def enhance_for_text_recognition(self, image):
        """
        GENTLE text enhancement with DPI optimization
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        self.save_debug_image(gray, "text_01_grayscale.png")
        
        # DPI optimization - upscale if too small
        h, w = gray.shape
        if w < 800:  # If less than 800px wide, upscale for better OCR
            scale_factor = 800 / w
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            
            upscaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            self.save_debug_image(upscaled, "text_02_upscaled.png")
            gray = upscaled
            logger.info(f"[PROCESSOR] Upscaled from {w}x{h} to {new_w}x{new_h}")
        
        # Very gentle contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        self.save_debug_image(enhanced, "text_03_enhanced.png")
        
        return enhanced
    
    # Legacy compatibility methods
    def four_point_crop(self, image, contour):
        """Legacy method - now redirects to dedicated cropper"""
        logger.info("[PROCESSOR] Legacy four_point_crop called - using dedicated cropper")
        return self.cropper.crop_receipt(image, contour)

    def order_points(self, pts):
        """Legacy method - now redirects to dedicated cropper"""
        return self.cropper.order_points(pts)

    def process_receipt(self, image):
        """
        Main processing pipeline using improved contour detection
        """
        logger.info("[PROCESSOR] Starting receipt processing with improved contour detection")
        
        try:
            # Step 1: Fix orientation
            oriented = self.detect_receipt_orientation(image)
            
            # Step 2: Find receipt using improved contour detection
            logger.info("[PROCESSOR] Finding receipt contour...")
            receipt_contour = self.contour_detector.find_receipt_contour(oriented)
            
            processed_images = []
            
            if receipt_contour is not None:
                logger.info("[PROCESSOR] ✅ Receipt found! Applying perspective correction...")
                
                # Step 3: Crop the receipt using dedicated cropper
                cropped = self.cropper.crop_receipt(oriented, receipt_contour)
                
                if cropped is not None:
                    # Save as step7 for compatibility
                    self.save_debug_image(cropped, "step7_cropped_and_corrected.png")
                    
                    # Step 4: Enhance for OCR
                    enhanced = self.enhance_for_text_recognition(cropped)
                    processed_images.append(enhanced)
                    
                    # Also add simple grayscale version
                    if len(cropped.shape) == 3:
                        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = cropped.copy()
                    processed_images.append(gray)
                    
                    status = "success_with_crop"
                    logger.info(f"[PROCESSOR] 🎯 SUCCESS! Dedicated cropping applied, {len(processed_images)} versions ready")
                else:
                    logger.warning("[PROCESSOR] Cropping failed, using enhanced full image")
                    enhanced_full = self.enhance_for_text_recognition(oriented)
                    processed_images.append(enhanced_full)
                    status = "crop_failed_fallback"
                
            else:
                logger.warning("[PROCESSOR] ❌ No receipt found, using enhanced full image")
                
                # Fallback: enhance the whole image
                enhanced_full = self.enhance_for_text_recognition(oriented)
                processed_images.append(enhanced_full)
                
                status = "fallback_no_crop"
                logger.info("[PROCESSOR] ⚠️ Fallback mode - processing full image")
            
            logger.info(f"[PROCESSOR] Complete: {status}, {len(processed_images)} images ready for OCR")
            return processed_images, status, {}
            
        except Exception as e:
            logger.error(f"[PROCESSOR] Error: {e}")
            
            # Final fallback
            try:
                gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                return [gray], "error_fallback", {"error": str(e)}
            except:
                return [image], "complete_failure", {"error": str(e)}