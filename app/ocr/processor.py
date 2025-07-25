import cv2
import numpy as np
import os
from typing import Optional, Tuple, List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

class ReceiptProcessor:
    def __init__(self, debug=True):
        self.debug = debug
        self.preprocessing_attempts = []
    
    def save_debug_image(self, image, name: str, step: str = ""):
        """Save debug image with step information"""
        if not self.debug:
            return
        
        step_prefix = f"{step}_" if step else ""
        path = os.path.join(DEBUG_DIR, f"{step_prefix}{name}.png")
        cv2.imwrite(path, image)
        logger.info(f"[DEBUG] Saved {path}")
    
    def assess_image_quality(self, img) -> Dict[str, float]:
        """Assess various image quality metrics"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Contrast assessment
        contrast = gray.std()
        
        # Brightness assessment
        brightness = gray.mean()
        
        # Blur assessment (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Noise assessment
        noise_score = cv2.fastNlMeansDenoising(gray).std() / gray.std()
        
        return {
            'contrast': contrast,
            'brightness': brightness,
            'blur_score': blur_score,
            'noise_score': noise_score
        }
    
    def adaptive_deskew(self, img) -> Tuple[np.ndarray, float]:
        """Improved deskewing with multiple methods and validation"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        angles = []
        
        # Method 1: Hough line detection
        try:
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            if lines is not None:
                for rho, theta in lines[:10]:  # Check top 10 lines
                    angle = np.degrees(theta) - 90
                    if abs(angle) < 45:  # Only consider reasonable angles
                        angles.append(angle)
        except:
            pass
        
        # Method 2: Minimum area rectangle (your original method)
        try:
            coords = np.column_stack(np.where(gray > 0))
            if len(coords) > 0:
                rect_angle = cv2.minAreaRect(coords)[-1]
                if rect_angle < -45:
                    rect_angle = -(90 + rect_angle)
                else:
                    rect_angle = -rect_angle
                
                if abs(rect_angle) < 45:
                    angles.append(rect_angle)
        except:
            pass
        
        # Method 3: Text line detection
        try:
            # Create horizontal and vertical kernels
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours[:5]:  # Check top 5 horizontal lines
                if cv2.contourArea(contour) > 100:
                    rect = cv2.minAreaRect(contour)
                    angle = rect[2]
                    if angle < -45:
                        angle = -(90 + angle)
                    if abs(angle) < 30:
                        angles.append(angle)
        except:
            pass
        
        # Determine final angle
        if not angles:
            logger.info("[DESKEW] No reliable angle found, skipping rotation")
            return img, 0.0
        
        # Use median angle for robustness
        final_angle = np.median(angles)
        
        # Only rotate if angle is reasonable
        if abs(final_angle) < 1:
            logger.info(f"[DESKEW] Angle too small ({final_angle:.2f}°), skipping")
            return img, final_angle
        
        if abs(final_angle) > 20:
            logger.info(f"[DESKEW] Angle too large ({final_angle:.2f}°), using fallback")
            final_angle = np.clip(final_angle, -15, 15)
        
        # Apply rotation
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, final_angle, 1.0)
        
        # Calculate new image size to avoid cropping
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int((h * sin_a) + (w * cos_a))
        new_h = int((h * cos_a) + (w * sin_a))
        
        # Adjust transformation matrix
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(img, M, (new_w, new_h), 
                                flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(255, 255, 255))
        
        self.save_debug_image(rotated, "deskewed", "step1")
        logger.info(f"[DESKEW] Applied rotation: {final_angle:.2f}°")
        return rotated, final_angle
    
    def adaptive_enhancement(self, img) -> List[np.ndarray]:
        """Generate multiple enhanced versions using different techniques"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        enhanced_versions = []
        quality = self.assess_image_quality(gray)
        
        # Version 1: Adaptive CLAHE based on contrast
        clip_limit = 3.0 if quality['contrast'] < 30 else 2.0
        tile_size = (16, 16) if quality['contrast'] < 20 else (8, 8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        clahe_enhanced = clahe.apply(gray)
        enhanced_versions.append(("clahe", clahe_enhanced))
        
        # Version 2: Gaussian blur + adaptive threshold
        blur_kernel = 5 if quality['blur_score'] < 100 else 3
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        
        # Adaptive threshold with multiple block sizes
        for block_size in [11, 15, 21]:
            try:
                thresh = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, block_size, 2
                )
                enhanced_versions.append((f"adaptive_{block_size}", thresh))
            except:
                continue
        
        # Version 3: Otsu's thresholding if image has good contrast
        if quality['contrast'] > 25:
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            enhanced_versions.append(("otsu", otsu))
        
        # Version 4: Morphological operations for noisy images
        if quality['noise_score'] < 0.8:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            _, morph_thresh = cv2.threshold(morph, 127, 255, cv2.THRESH_BINARY)
            enhanced_versions.append(("morphological", morph_thresh))
        
        # Version 5: Simple binary threshold as fallback
        _, simple = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        enhanced_versions.append(("simple", simple))
        
        # Save debug images
        for i, (name, version) in enumerate(enhanced_versions):
            self.save_debug_image(version, f"enhanced_{name}", "step3")
        
        return [version for _, version in enhanced_versions]
    
    def smart_contour_detection(self, img) -> Optional[np.ndarray]:
        """Improved contour detection with multiple strategies, starting with binarized image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        contour_candidates = []


        # Try multiple binarization strategies and pick the best by largest contour area
        binarization_methods = []
        try:
            # Otsu
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binarization_methods.append(("otsu", otsu))
            # Otsu inverted
            _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            binarization_methods.append(("otsu_inv", otsu_inv))
            # Adaptive mean
            adaptive_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
            binarization_methods.append(("adaptive_mean", adaptive_mean))
            # Adaptive mean inverted
            adaptive_mean_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
            binarization_methods.append(("adaptive_mean_inv", adaptive_mean_inv))
            # Adaptive gaussian
            adaptive_gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
            binarization_methods.append(("adaptive_gauss", adaptive_gauss))
            # Adaptive gaussian inverted
            adaptive_gauss_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
            binarization_methods.append(("adaptive_gauss_inv", adaptive_gauss_inv))
        except Exception as e:
            logger.warning(f"Binarization methods failed: {e}")

        # Try all binarizations, keep the best contour(s)
        best_area = 0
        best_method = None
        for method_name, bin_img in binarization_methods:
            try:
                contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                self._evaluate_contours(contours, img, contour_candidates, f"binarized_{method_name}")
                if self.debug:
                    self.save_debug_image(bin_img, f"binarized_for_contour_{method_name}", "step2")
                # Find largest contour area for this method
                if contours:
                    max_area = max(cv2.contourArea(c) for c in contours)
                    if max_area > best_area:
                        best_area = max_area
                        best_method = method_name
            except Exception as e:
                logger.warning(f"Binarized detection failed for {method_name}: {e}")

        # Strategy 2: Edge-based detection (your original method)
        try:
            edges = cv2.Canny(gray, 30, 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self._evaluate_contours(contours, img, contour_candidates, "edge_based")
        except Exception as e:
            logger.warning(f"Edge-based detection failed: {e}")

        # Strategy 3: Morphological operations
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self._evaluate_contours(contours, img, contour_candidates, "morphological")
        except Exception as e:
            logger.warning(f"Morphological detection failed: {e}")

        # Select best contour
        if contour_candidates:
            # Sort by score (higher is better)
            contour_candidates.sort(key=lambda x: x['score'], reverse=True)
            best_contour = contour_candidates[0]
            logger.info(f"[CONTOUR] Selected {best_contour['method']} contour with score {best_contour['score']:.2f}")
            self.save_debug_image(self._draw_contour(img, best_contour['points']), "selected_contour", "step2")
            return best_contour['points']

        logger.warning("[CONTOUR] No suitable contour found")
        return None
    
    def _evaluate_contours(self, contours, img, candidates, method):
        """Evaluate contours and add good candidates to list, rejecting those too close to image border."""
        image_h, image_w = img.shape[:2]
        image_area = image_h * image_w
        margin = 15  # pixels from edge to consider as 'too close'

        for contour in contours:
            try:
                # Approximate contour
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                if len(approx) != 4:
                    continue

                area = cv2.contourArea(approx)
                if area < image_area * 0.1:
                    continue

                # Calculate bounding box and aspect ratio
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = h / float(w) if w > 0 else 0

                # Reject contours too close to the image border
                if (
                    x < margin or y < margin or
                    x + w > image_w - margin or
                    y + h > image_h - margin
                ):
                    logger.debug(f"[CONTOUR] Rejected for being too close to border: x={x}, y={y}, w={w}, h={h}")
                    continue

                # Penalize contours that are too large (almost whole image)
                area_ratio = area / image_area
                if area_ratio > 0.92:
                    logger.debug(f"[CONTOUR] Rejected for being too large: area_ratio={area_ratio:.2f}")
                    continue

                # Score the contour
                score = self._score_contour(approx, area, aspect_ratio, image_area)

                # Stricter minimum threshold
                if score > 0.4:
                    candidates.append({
                        'points': approx.reshape(4, 2).astype(np.float32),
                        'score': score,
                        'method': method,
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    })
            except Exception as e:
                logger.debug(f"[CONTOUR] Exception in evaluation: {e}")
                continue
    
    def _score_contour(self, contour, area, aspect_ratio, image_area):
        """Score contour based on receipt-like properties"""
        score = 0.0
        
        # Area score (prefer medium to large areas)
        area_ratio = area / image_area
        if 0.2 <= area_ratio <= 0.9:
            score += 0.4
        elif 0.1 <= area_ratio <= 0.95:
            score += 0.2
        
        # Aspect ratio score (receipts are typically tall)
        if 1.2 <= aspect_ratio <= 4.0:
            score += 0.3
        elif 1.0 <= aspect_ratio <= 5.0:
            score += 0.2
        
        # Convexity score
        hull = cv2.convexHull(contour)
        convexity = cv2.contourArea(contour) / cv2.contourArea(hull)
        if convexity > 0.8:
            score += 0.2
        
        # Rectangularity score
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0
        if rectangularity > 0.7:
            score += 0.1
        
        return score
    
    def _draw_contour(self, img, points):
        """Draw contour points for debugging"""
        debug_img = img.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(debug_img, (int(x), int(y)), 10, (0, 255, 0), -1)
            cv2.putText(debug_img, str(i), (int(x) + 15, int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return debug_img
    
    def robust_crop(self, img, points) -> Optional[np.ndarray]:
        """Robust perspective correction with error handling"""
        try:
            if points is None or len(points) != 4:
                return None
            
            # Order points consistently
            rect = self._order_points(points)
            
            # Calculate dimensions
            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxHeight = max(int(heightA), int(heightB))
            
            # Ensure minimum size
            if maxWidth < 100 or maxHeight < 100:
                logger.warning("[CROP] Resulting image would be too small")
                return None
            
            # Apply padding
            padding = min(20, maxWidth // 10, maxHeight // 10)
            
            dst = np.array([
                [padding, padding],
                [maxWidth - padding, padding],
                [maxWidth - padding, maxHeight - padding],
                [padding, maxHeight - padding]
            ], dtype="float32")
            
            # Get perspective transform
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
            
            self.save_debug_image(warped, "cropped", "step2")
            return warped
            
        except Exception as e:
            logger.error(f"[CROP] Failed: {e}")
            return None
    
    def _order_points(self, pts):
        """Order points consistently: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")
        
        # Sum coordinates
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        # Difference of coordinates
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        return rect
    
    def process_receipt(self, img) -> Tuple[List[np.ndarray], str]:
        """
        Main processing function that returns multiple processed versions
        Returns: (list_of_processed_images, status_message)
        """
        self.preprocessing_attempts = []
        processed_images = []
        
        try:
            # Step 1: Assess input image quality
            quality = self.assess_image_quality(img)
            logger.info(f"[QUALITY] Contrast: {quality['contrast']:.1f}, "
                       f"Brightness: {quality['brightness']:.1f}, "
                       f"Blur: {quality['blur_score']:.1f}")
            
            self.save_debug_image(img, "original", "step0")
            
            # Step 2: Deskew if needed
            deskewed, angle = self.adaptive_deskew(img)
            
            # Step 3: Try to find receipt contour
            contour = self.smart_contour_detection(deskewed)
            
            # Step 4: Process with and without cropping
            if contour is not None:
                # Process cropped version
                cropped = self.robust_crop(deskewed, contour)
                if cropped is not None:
                    enhanced_cropped = self.adaptive_enhancement(cropped)
                    processed_images.extend(enhanced_cropped)
                    self.preprocessing_attempts.append("cropped_enhanced")
            
            # Always process full image as fallback
            enhanced_full = self.adaptive_enhancement(deskewed)
            processed_images.extend(enhanced_full)
            self.preprocessing_attempts.append("full_image_enhanced")
            
            # If no processing worked, return simple grayscale
            if not processed_images:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                processed_images.append(gray)
                self.preprocessing_attempts.append("simple_grayscale")
            
            status = f"Success: {len(processed_images)} versions generated"
            logger.info(f"[PROCESS] {status}")
            
            return processed_images, status
            
        except Exception as e:
            logger.error(f"[PROCESS] Fatal error: {e}")
            # Emergency fallback
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                return [gray], f"Emergency fallback: {str(e)}"
            except:
                return [], f"Complete failure: {str(e)}"

# Convenience function for backward compatibility
def preprocess_for_ocr(img):
    """
    Backward compatible function that returns the best processed image
    """
    processor = ReceiptProcessor(debug=True)
    processed_images, status = processor.process_receipt(img)
    
    if processed_images:
        return processed_images[0], "ok"
    else:
        return None, status