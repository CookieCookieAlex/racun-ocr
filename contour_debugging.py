#!/usr/bin/env python3
"""
Standalone Contour Test
======================
Test contour detection without complex imports
"""

import cv2
import numpy as np
import os

# Copy your ReceiptContourDetector class directly here
class ReceiptContourDetector:
    def __init__(self, debug=True):
        self.debug = debug
        self.debug_dir = "debug_images"
        os.makedirs(self.debug_dir, exist_ok=True)

    def save_debug_image(self, image, name):
        """Save debug image if debug mode is enabled"""
        if self.debug:
            path = os.path.join(self.debug_dir, name if name.endswith(".png") else name + ".png")
            cv2.imwrite(path, image)
            print(f"💾 Saved {path}")

    def sharpen_edge(self, img):
        """Edge detection from your old code"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        dil = cv2.dilate(blur, kern, iterations=2)
        closed = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, kern, iterations=2)
        edged = cv2.Canny(closed, 30, 100)
        
        self.save_debug_image(gray, "01_gray.png")
        self.save_debug_image(blur, "02_blur.png")
        self.save_debug_image(dil, "03_dilated.png")
        self.save_debug_image(closed, "04_closed.png")
        self.save_debug_image(edged, "05_edged.png")
        
        return edged

    def create_binary_methods(self, image):
        """Enhanced binary methods - GENTLE approach for finger-held receipts"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # CRITICAL: Much gentler filtering to preserve edges near fingers
        filtered = cv2.GaussianBlur(gray, (3, 3), 0)  # Much lighter blur
        
        methods = []
        
        # Method 1: GENTLE Otsu (works better with fingers)
        _, gentle_otsu = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(("gentle_otsu", gentle_otsu))
        
        # Method 2: HIGH threshold binary (fingers are usually darker than receipt)
        # Use a high threshold to separate white receipt from everything else
        high_thresh = int(np.percentile(filtered, 75))  # Use 75th percentile as threshold
        _, high_binary = cv2.threshold(filtered, high_thresh, 255, cv2.THRESH_BINARY)
        methods.append(("high_threshold", high_binary))
        
        # Method 3: VERY gentle adaptive (large block size to ignore finger details)
        adaptive_gentle = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 31, 15)  # Large block, gentle
        methods.append(("adaptive_gentle", adaptive_gentle))
        
        # Method 4: Multiple Otsu thresholds to separate finger/receipt/background
        # Try different threshold levels
        mean_val = int(np.mean(filtered))
        
        _, mean_plus = cv2.threshold(filtered, mean_val + 20, 255, cv2.THRESH_BINARY)
        methods.append(("mean_plus", mean_plus))
        
        _, mean_minus = cv2.threshold(filtered, mean_val - 20, 255, cv2.THRESH_BINARY)
        methods.append(("mean_minus", mean_minus))
        
        # Method 5: Edge-based but MUCH gentler
        edges_gentle = cv2.Canny(filtered, 30, 90)  # Lower thresholds
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Smaller kernel
        edges_dilated = cv2.dilate(edges_gentle, kernel_small, iterations=1)
        methods.append(("gentle_edges", edges_dilated))
        
        # CRITICAL: Only light morphological operations
        cleaned_methods = []
        for name, binary in methods:
            # MINIMAL morphological operations - don't destroy finger boundaries
            kernel_tiny = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_tiny, iterations=1)
            
            cleaned_methods.append((name, cleaned))
            self.save_debug_image(cleaned, f"contour_gentle_{name}.png")
        
        return cleaned_methods

    def find_best_contour(self, binary_methods, original_image):
        """Enhanced contour finding for full receipt"""
        h, w = original_image.shape[:2]
        image_area = h * w
        
        best_contour = None
        best_score = 0
        best_method = None
        
        for method_name, binary_img in binary_methods:
            # Find contours
            contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            print(f"🔍 Method {method_name}: Found {len(contours)} contours")
            
            for i, contour in enumerate(contours[:5]):  # Check top 5 contours
                area = cv2.contourArea(contour)
                score = self.score_full_receipt_contour(contour, image_area, (h, w))
                
                print(f"   Contour {i}: area={area}, score={score:.2f}")
                
                if score > best_score:
                    best_score = score
                    # Convert to rectangle
                    rect_contour = self.contour_to_rectangle(contour)
                    best_contour = rect_contour
                    best_method = method_name
                    
                    print(f"   🏆 New best: {method_name}, contour #{i}, score={score:.2f}")
        
        if best_contour is not None:
            print(f"✅ Selected contour from {best_method} with score {best_score:.2f}")
            
            # Draw the selected contour for debugging
            debug_img = original_image.copy()
            cv2.drawContours(debug_img, [best_contour], -1, (0, 255, 0), 3)
            
            # Draw corner points
            for i, point in enumerate(best_contour):
                x, y = point[0] if len(point.shape) > 1 else point
                cv2.circle(debug_img, (int(x), int(y)), 8, (255, 0, 0), -1)
                cv2.putText(debug_img, str(i), (int(x)+12, int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            self.save_debug_image(debug_img, f"contour_best_{best_method}.png")
        
        return best_contour

    def score_full_receipt_contour(self, contour, image_area, image_shape):
        """UPDATED: Score contour with finger-awareness"""
        
        area = cv2.contourArea(contour)
        if area < 10000:  # Must be substantial for full receipt
            return 0.0
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        h_img, w_img = image_shape
        
        score = 0.0
        
        # 1. Area score - ADJUSTED for finger occlusion
        area_ratio = area / image_area
        if 0.25 <= area_ratio <= 0.70:  # LOWER range - fingers reduce visible area
            score += 5.0
        elif 0.15 <= area_ratio <= 0.80:  # More forgiving range
            score += 3.0
        elif 0.1 <= area_ratio <= 0.85:
            score += 2.0
        
        # 2. Aspect ratio - receipts are tall (even with fingers)
        aspect_ratio = h / w if w > 0 else 0
        if 1.5 <= aspect_ratio <= 3.5:  # Slightly more forgiving
            score += 3.0
        elif 1.2 <= aspect_ratio <= 4.0:
            score += 2.0
        elif 1.0 <= aspect_ratio <= 5.0:  # Very forgiving
            score += 1.0
        
        # 3. Width coverage - ADJUSTED for fingers potentially blocking edges
        width_ratio = w / w_img
        if width_ratio >= 0.6:  # LOWER threshold - fingers reduce width
            score += 3.0
        elif width_ratio >= 0.5:  # More forgiving
            score += 2.0
        elif width_ratio >= 0.4:  # Very forgiving
            score += 1.0
        
        # 4. Position - BONUS for being somewhat centered (fingers usually on sides/bottom)
        center_x_ratio = (x + w/2) / w_img
        center_y_ratio = (y + h/2) / h_img
        
        # More forgiving centering
        if 0.25 <= center_x_ratio <= 0.75:
            score += 1.0
        
        # 5. Rectangularity - MORE FORGIVING (fingers disrupt perfect rectangles)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)  # Less strict approximation
        if len(approx) == 4:
            score += 2.0  # Reduced from 3.0
        elif len(approx) <= 6:
            score += 1.5  # More generous
        elif len(approx) <= 10:  # Even more forgiving
            score += 1.0
        
        # 6. Edge proximity - BONUS for touching image edges (common with finger-held receipts)
        edge_margin = 10
        touches_edge = (x < edge_margin or y < edge_margin or 
                       (x + w) > (w_img - edge_margin) or (y + h) > (h_img - edge_margin))
        
        if touches_edge:
            score += 1.0  # This is actually GOOD for finger-held receipts
        
        return score

    def contour_to_rectangle(self, contour):
        """Convert any contour to best 4-point rectangle"""
        
        # Method 1: Polygon approximation
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            return approx
        
        # Method 2: Minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.int32)
        
        # Convert to standard contour format
        return box.reshape(4, 1, 2)

    def find_receipt_contour(self, img):
        """
        IMPROVED: Find full receipt contour including handling for occlusion
        """
        print("[CONTOUR] Starting improved receipt contour detection")
        
        # Step 1: Enhanced edge detection
        edged = self.sharpen_edge(img)
        
        # Step 2: Try multiple binarization methods for full receipt
        binary_methods = self.create_binary_methods(img)
        
        # Step 3: Find best contour across all methods
        best_contour = self.find_best_contour(binary_methods, img)
        
        if best_contour is not None:
            print("[CONTOUR] ✅ Found receipt contour!")
            return best_contour
        
        print("[CONTOUR] ❌ No suitable receipt contour found")
        return None

def test_contour_detection(image_path):
    """Test the contour detection"""
    print(f"🔍 TESTING CONTOUR DETECTION")
    print("=" * 40)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load {image_path}")
        return
    
    print(f"📁 Image: {image_path}")
    print(f"📏 Size: {image.shape}")
    
    # Create detector
    detector = ReceiptContourDetector(debug=True)
    
    print(f"\n🎯 RUNNING CONTOUR DETECTION")
    print("-" * 35)
    
    # Run detection
    contour = detector.find_receipt_contour(image)
    
    if contour is not None:
        print("✅ Detection successful!")
        
        # Analyze result
        area = cv2.contourArea(contour)
        image_area = image.shape[0] * image.shape[1]
        area_ratio = area / image_area
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w if w > 0 else 0
        
        print(f"📊 RESULTS:")
        print(f"   Area: {area} ({area_ratio:.1%} of image)")
        print(f"   Bounding box: {w}x{h} at ({x}, {y})")
        print(f"   Aspect ratio: {aspect_ratio:.2f}")
        
        # Analyze if this looks right
        print(f"\n🤔 ANALYSIS:")
        if area_ratio < 0.15:
            print("   ⚠️  Area seems small - might be cropping a piece")
        elif area_ratio > 0.8:
            print("   ⚠️  Area seems large - might be whole image")
        else:
            print("   ✅ Area looks reasonable")
            
        if aspect_ratio < 1.2:
            print("   ⚠️  Too wide for a typical receipt")
        elif aspect_ratio > 4.0:
            print("   ⚠️  Very tall aspect ratio")
        else:
            print("   ✅ Aspect ratio looks good")
    else:
        print("❌ No contour found")
    
    # Show debug images created
    debug_dir = "debug_images"
    if os.path.exists(debug_dir):
        debug_files = [f for f in os.listdir(debug_dir) if f.endswith('.png')]
        print(f"\n📁 Created {len(debug_files)} debug images:")
        for file in sorted(debug_files):
            print(f"   📄 {file}")
    
    return contour

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python standalone_contour_test.py <image_path>")
        sys.exit(1)
    
    test_contour_detection(sys.argv[1])
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Check debug_images/ folder for all processing steps")  
    print("2. Look at contour_best_*.png to see what was selected")
    print("3. If wrong area, we can adjust the scoring parameters")