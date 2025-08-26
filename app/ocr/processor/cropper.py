import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class ReceiptCropper:
    def __init__(self, debug=True):
        self.debug = debug
        self.debug_dir = "debug_images"
        os.makedirs(self.debug_dir, exist_ok=True)

    def save_debug_image(self, image, name):
        """Save debug image if debug mode is enabled"""
        if self.debug:
            path = os.path.join(self.debug_dir, name if name.endswith(".png") else name + ".png")
            cv2.imwrite(path, image)
            logger.info(f"[CROPPER] Saved {path}")

    def order_points(self, pts):
        """Order points for perspective transformation: TL, TR, BR, BL"""
        
        # Handle different input formats
        if pts.shape == (4, 1, 2):
            pts = pts.reshape(4, 2)
        
        pts = pts.astype(np.float32)
        rect = np.zeros((4, 2), dtype="float32")
        
        # Sum and difference method for robust corner ordering
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left: smallest sum (x + y)
        rect[2] = pts[np.argmax(s)]  # bottom-right: largest sum (x + y)
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right: smallest difference (x - y)
        rect[3] = pts[np.argmax(diff)]  # bottom-left: largest difference (x - y)
        
        return rect

    def validate_contour(self, contour):
        """Validate that contour is suitable for cropping"""
        if contour is None:
            return False, "No contour provided"
        
        if len(contour) < 3:
            return False, "Contour has too few points"
        
        area = cv2.contourArea(contour)
        if area < 1000:
            return False, f"Contour area too small: {area}"
        
        return True, "Valid contour"

    def prepare_contour_for_cropping(self, contour):
        """Convert any contour to a 4-point rectangle suitable for perspective transform"""
        
        # Validate input
        is_valid, message = self.validate_contour(contour)
        if not is_valid:
            logger.warning(f"[CROPPER] {message}")
            return None
        
        # If already 4 points, try to use directly
        if len(contour) == 4:
            logger.info("[CROPPER] Using 4-point contour directly")
            return contour
        
        # Method 1: Polygon approximation to get 4 points
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            logger.info("[CROPPER] ✅ Polygon approximation gave 4 points")
            return approx
        
        # Method 2: Minimum area rectangle (fallback)
        logger.info("[CROPPER] Using minimum area rectangle fallback")
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.int32)
        
        # Convert to standard contour format
        return box.reshape(4, 1, 2)

    def calculate_output_dimensions(self, corners):
        """Calculate optimal output dimensions from corner points"""
        
        ordered_corners = self.order_points(corners)
        (tl, tr, br, bl) = ordered_corners
        
        # Calculate width by taking maximum of top and bottom edge lengths
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        width = max(int(width_top), int(width_bottom))
        
        # Calculate height by taking maximum of left and right edge lengths
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        height = max(int(height_left), int(height_right))
        
        # Ensure minimum dimensions for good OCR
        width = max(width, 400)
        height = max(height, 600)
        
        # Ensure reasonable maximum dimensions (prevent memory issues)
        width = min(width, 3000)
        height = min(height, 4000)
        
        logger.info(f"[CROPPER] Calculated output dimensions: {width} x {height}")
        
        return width, height

    def apply_perspective_transform(self, image, contour):
        """Apply perspective transformation to extract receipt"""
        
        logger.info("[CROPPER] Starting perspective transformation")
        
        # Prepare contour for cropping
        rect_contour = self.prepare_contour_for_cropping(contour)
        if rect_contour is None:
            logger.error("[CROPPER] Could not prepare contour for cropping")
            return None
        
        # Extract and order the corner points
        pts = rect_contour.reshape(4, 2).astype(np.float32)
        ordered_corners = self.order_points(pts)
        (tl, tr, br, bl) = ordered_corners
        
        logger.info(f"[CROPPER] Corner points:")
        logger.info(f"  Top-left: {tl}")
        logger.info(f"  Top-right: {tr}")
        logger.info(f"  Bottom-right: {br}")
        logger.info(f"  Bottom-left: {bl}")
        
        # Calculate output dimensions
        width, height = self.calculate_output_dimensions(rect_contour)
        
        # Define destination points (perfect rectangle)
        dst_points = np.array([
            [0, 0],                    # top-left
            [width - 1, 0],           # top-right
            [width - 1, height - 1],  # bottom-right
            [0, height - 1]           # bottom-left
        ], dtype="float32")
        
        logger.info(f"[CROPPER] Destination rectangle: {width} x {height}")
        
        try:
            # Calculate perspective transformation matrix
            transform_matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)
            
            # Apply the transformation
            warped = cv2.warpPerspective(image, transform_matrix, (width, height))
            
            # Save debug images
            self.save_debug_image(warped, "cropper_result.png")
            
            logger.info("[CROPPER] ✅ Perspective transformation successful")
            return warped
            
        except Exception as e:
            logger.error(f"[CROPPER] Perspective transformation failed: {e}")
            return None

    def fallback_crop(self, image, contour):
        """Fallback cropping using bounding rectangle"""
        
        logger.warning("[CROPPER] Using fallback bounding rectangle crop")
        
        try:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add small margin if possible
            margin = 5
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image.shape[1] - x, w + 2 * margin)
            h = min(image.shape[0] - y, h + 2 * margin)
            
            # Crop the image
            cropped = image[y:y+h, x:x+w]
            
            self.save_debug_image(cropped, "cropper_fallback.png")
            
            logger.info(f"[CROPPER] Fallback crop: {w} x {h}")
            return cropped
            
        except Exception as e:
            logger.error(f"[CROPPER] Fallback crop failed: {e}")
            return image  # Return original image as last resort

    def crop_receipt(self, image, contour):
        """
        Main cropping function - extract receipt with perspective correction
        
        Args:
            image: Input image (numpy array)
            contour: Receipt contour (4-point or approximable to 4-point)
            
        Returns:
            Cropped and perspective-corrected receipt image
        """
        
        logger.info("[CROPPER] Starting receipt cropping")
        
        if image is None:
            logger.error("[CROPPER] No image provided")
            return None
        
        if contour is None:
            logger.error("[CROPPER] No contour provided")
            return image  # Return original image
        
        # Save input for debugging
        if self.debug:
            input_img = image.copy()
            if len(input_img.shape) == 2:
                input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
            
            # Draw contour on input image
            cv2.drawContours(input_img, [contour], -1, (0, 255, 0), 3)
            
            # Draw corner points if it's a 4-point contour
            if len(contour) == 4:
                for i, point in enumerate(contour):
                    x, y = point[0] if len(point.shape) > 1 else point
                    cv2.circle(input_img, (int(x), int(y)), 8, (255, 0, 0), -1)
                    cv2.putText(input_img, str(i), (int(x)+12, int(y)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            self.save_debug_image(input_img, "cropper_input_with_contour.png")
        
        # Try perspective transformation first
        result = self.apply_perspective_transform(image, contour)
        
        if result is not None:
            logger.info("[CROPPER] ✅ Perspective transformation successful")
            return result
        
        # If perspective transform fails, use fallback
        logger.warning("[CROPPER] Perspective transform failed, using fallback")
        result = self.fallback_crop(image, contour)
        
        return result

    def enhance_cropped_receipt(self, cropped_image):
        """Optional post-processing enhancement for cropped receipt"""
        
        if cropped_image is None:
            return None
        
        # Convert to grayscale if needed
        if len(cropped_image.shape) == 3:
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped_image.copy()
        
        # Light enhancement for better OCR
        # Very gentle CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        self.save_debug_image(enhanced, "cropper_enhanced.png")
        
        return enhanced

    def get_crop_info(self, original_image, cropped_image, contour):
        """Get information about the cropping operation"""
        
        if original_image is None or cropped_image is None:
            return {}
        
        orig_h, orig_w = original_image.shape[:2]
        crop_h, crop_w = cropped_image.shape[:2]
        
        # Calculate area ratios
        orig_area = orig_h * orig_w
        crop_area = crop_h * crop_w
        contour_area = cv2.contourArea(contour) if contour is not None else 0
        
        info = {
            "original_size": (orig_w, orig_h),
            "cropped_size": (crop_w, crop_h),
            "size_ratio": crop_area / orig_area if orig_area > 0 else 0,
            "contour_area": contour_area,
            "contour_area_ratio": contour_area / orig_area if orig_area > 0 else 0,
            "scale_factor": crop_w / orig_w if orig_w > 0 else 1.0
        }
        
        logger.info(f"[CROPPER] Crop info: {info}")
        return info