#!/usr/bin/env python3
"""
Quick Contour Debug
==================
Simple test to see what your contour detection is finding
"""

import cv2
import numpy as np
import os

def quick_contour_analysis(image_path):
    """Quick analysis of what contour detection finds"""
    
    print("🔍 QUICK CONTOUR ANALYSIS")
    print("=" * 30)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load {image_path}")
        return
    
    print(f"📁 Image: {image_path}")
    print(f"📏 Size: {image.shape}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Try different thresholding methods to see what each finds
    methods = []
    
    # Method 1: Simple Otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    methods.append(("otsu", otsu))
    
    # Method 2: High threshold (should separate white receipt from everything else)
    high_thresh = int(np.percentile(gray, 75))  # 75th percentile
    _, high_binary = cv2.threshold(gray, high_thresh, 255, cv2.THRESH_BINARY)
    methods.append(("high_threshold", high_binary))
    
    # Method 3: Try to isolate white areas specifically
    white_thresh = 180  # Pretty high threshold for white
    _, white_binary = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY)
    methods.append(("white_only", white_binary))
    
    # Method 4: Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 21, 10)
    methods.append(("adaptive", adaptive))
    
    # Save debug images and analyze contours
    os.makedirs("debug_images", exist_ok=True)
    
    best_method = None
    best_score = 0
    
    for method_name, binary_img in methods:
        # Save the binary image
        cv2.imwrite(f"debug_images/quick_{method_name}.png", binary_img)
        
        # Find contours
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        print(f"\n🔍 Method: {method_name}")
        print(f"   Found {len(contours)} contours")
        
        # Analyze top contours
        image_area = image.shape[0] * image.shape[1]
        
        for i, contour in enumerate(contours[:3]):  # Top 3 contours
            area = cv2.contourArea(contour)
            area_ratio = area / image_area
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0
            
            # Simple scoring - favor receipt-like properties
            score = 0
            if 0.2 <= area_ratio <= 0.7:  # Good size
                score += 3
            if 1.3 <= aspect_ratio <= 3.5:  # Receipt-like aspect ratio
                score += 3
            if w < image.shape[1] * 0.9:  # Not full width (avoids selecting whole image)
                score += 2
                
            # Check if it's mostly in the center-right (where receipt should be)
            center_x_ratio = (x + w/2) / image.shape[1]
            if 0.3 <= center_x_ratio <= 0.8:  # Receipt should be center-right
                score += 2
                
            print(f"   Contour {i}: area={area} ({area_ratio:.1%}), "
                  f"box={w}x{h} at ({x},{y}), aspect={aspect_ratio:.2f}, score={score}")
            
            if score > best_score:
                best_score = score
                best_method = (method_name, i, contour)
        
        # Draw contours on original image for visualization
        debug_img = image.copy()
        cv2.drawContours(debug_img, contours[:3], -1, (0, 255, 0), 2)
        
        # Number the top contours
        for i, contour in enumerate(contours[:3]):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(debug_img, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imwrite(f"debug_images/quick_{method_name}_contours.png", debug_img)
        print(f"   💾 Saved: debug_images/quick_{method_name}_contours.png")
    
    # Report best finding
    if best_method:
        method_name, contour_idx, contour = best_method
        print(f"\n🏆 BEST CONTOUR FOUND:")
        print(f"   Method: {method_name}")
        print(f"   Contour index: {contour_idx}")
        print(f"   Score: {best_score}")
        
        # Draw the best contour
        best_img = image.copy()
        cv2.drawContours(best_img, [contour], -1, (0, 255, 0), 3)
        
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(best_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imwrite("debug_images/quick_best_contour.png", best_img)
        print(f"   💾 Saved: debug_images/quick_best_contour.png")
        
        # Analysis
        area = cv2.contourArea(contour)
        area_ratio = area / image_area
        aspect_ratio = h / w if w > 0 else 0
        center_x_ratio = (x + w/2) / image.shape[1]
        
        print(f"\n📊 BEST CONTOUR ANALYSIS:")
        print(f"   Area: {area} ({area_ratio:.1%} of image)")
        print(f"   Bounding box: {w}x{h} at ({x}, {y})")
        print(f"   Aspect ratio: {aspect_ratio:.2f}")
        print(f"   Center X: {center_x_ratio:.2f} (0=left, 1=right)")
        
        # Check if this looks like the receipt
        if center_x_ratio > 0.6 and 1.5 <= aspect_ratio <= 3.5 and 0.2 <= area_ratio <= 0.6:
            print("   ✅ This looks like it could be the receipt!")
        else:
            print("   ⚠️  This doesn't look like the receipt")
            if center_x_ratio <= 0.6:
                print("       - Too far left (receipt should be center-right)")
            if aspect_ratio < 1.5:
                print("       - Too wide (receipts are tall)")
            if area_ratio < 0.2:
                print("       - Too small")
            if area_ratio > 0.6:
                print("       - Too large")
    
    else:
        print(f"\n❌ No good contours found")
    
    print(f"\n📁 Check debug_images/quick_*.png to see what each method found")
    return best_method

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python quick_contour_debug.py <image_path>")
        sys.exit(1)
    
    quick_contour_analysis(sys.argv[1])