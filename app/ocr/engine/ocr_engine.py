import pytesseract
import re
import logging
import cv2
import numpy as np
import os
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Croatian-optimized configurations for receipts
DEFAULT_CONFIGS = [
    # Primary Croatian receipt configurations
    '--psm 6 --oem 3',  # Single uniform block of text
    '--psm 4 --oem 3',  # Single column of text with variable sizes
    '--psm 3 --oem 3',  # Fully automatic page segmentation
    
    # With Croatian character whitelist - FIXED: Proper escaping
    '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzčćžšđĆČŽŠĐ.,: €/-',
    '--psm 4 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzčćžšđĆČŽŠĐ.,: €/-',
    
    # Single line mode (good for totals)
    '--psm 7 --oem 3',  # Single text line
    '--psm 8 --oem 3',  # Single word
    
    # Sparse text configurations for difficult receipts
    '--psm 11 --oem 3',  # Sparse text detection
    '--psm 12 --oem 3',  # Sparse text with orientation detection
]

def assess_image_quality(image: np.ndarray) -> dict:
    """Assess image quality to choose optimal Croatian OCR preprocessing."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Calculate quality metrics
    mean_brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Laplacian variance for sharpness assessment
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Quality flags
    is_dark = mean_brightness < 100
    is_low_contrast = contrast < 40
    is_blurry = laplacian_var < 100
    
    return {
        'mean_brightness': mean_brightness,
        'contrast': contrast,
        'sharpness': laplacian_var,
        'is_dark': is_dark,
        'is_low_contrast': is_low_contrast,
        'is_blurry': is_blurry
    }

def preprocess_for_ocr(image: np.ndarray, enhancement_type: str = "standard") -> np.ndarray:
    """CONSERVATIVE preprocessing - don't destroy good cropped receipts!"""
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if enhancement_type == "minimal":
        # MINIMAL processing - just return the grayscale
        return gray
    
    elif enhancement_type == "very_gentle_clahe":
        # Very gentle CLAHE only
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced
    
    elif enhancement_type == "super_gentle_binary":
        # Super gentle binarization - ONLY for very poor images
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # EXTREMELY gentle adaptive threshold
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 25, 12  # Very large block, very gentle
        )
        return binary
    
    elif enhancement_type == "gentle_otsu":
        # Gentle Otsu for contrast improvement
        clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    elif enhancement_type == "high_contrast":
        # For very dark or low contrast images - GENTLER approach
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Gentler gamma correction
        enhanced = np.power(enhanced / 255.0, 0.8) * 255
        enhanced = enhanced.astype(np.uint8)
        
        # MUCH gentler adaptive threshold
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, 6
        )
        return binary
        
    elif enhancement_type == "sharp":
        # For blurry images - unsharp masking
        gaussian = cv2.GaussianBlur(gray, (9, 9), 10)
        unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        
        # Gentler adaptive threshold
        binary = cv2.adaptiveThreshold(
            unsharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 13, 4
        )
        return binary
        
    elif enhancement_type == "noise_reduction":
        # For noisy images
        # Bilateral filter preserves edges while reducing noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Gentle CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        # Morphological cleaning
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
        
        return cleaned
        
    else:  # standard processing
        # Standard CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

def clean_text(text: str) -> str:
    """Enhanced Croatian text cleaning - PRESERVE line breaks for parsing!"""
    if not text:
        return ""
    
    # FIRST: Try to restore line breaks that OCR might have missed
    # Look for patterns that should be on separate lines
    text = re.sub(r'(UKUPNO.*?)([A-ZČĆŽŠĐ][a-zčćžšđ])', r'\1\n\2', text)  # After UKUPNO
    text = re.sub(r'(\d{2}\.\d{2}\.\d{4}.*?\d{2}:\d{2})', r'\1\n', text)  # After datetime
    text = re.sub(r'(Račun brj.*?\d+)', r'\1\n', text)  # After receipt number
    text = re.sub(r'(\d+\.\d{2})(\s+[A-ZČĆŽŠĐ])', r'\1\n\2', text)  # After prices
    
    # Split into lines for processing
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if not line.strip():
            continue
            
        line = line.strip()
        
        # Basic spacing fixes - but preserve structure
        line = re.sub(r'(?<=\d)(?=[A-Za-zčćžšđČĆŽŠĐ])', ' ', line)  # Space between numbers and letters
        line = re.sub(r'(?<=[A-Za-zčćžšđČĆŽŠĐ])(?=\d)', ' ', line)  # Space between letters and numbers
        
        # Common OCR character fixes
        line = re.sub(r'[|!¡]', 'l', line)  # Vertical bars to 'l'
        line = re.sub(r'[0oO](?=[a-zA-ZčćžšđČĆŽŠĐ])', '', line)  # Remove leading zeros before letters
        
        # Croatian character fixes (common OCR errors)
        line = re.sub(r'c\s*\'\s*', 'ć', line)
        line = re.sub(r'z\s*\'\s*', 'ž', line)
        line = re.sub(r's\s*\'\s*', 'š', line)
        line = re.sub(r'd\s*\'\s*', 'đ', line)
        line = re.sub(r'C\s*\'\s*', 'Ć', line)
        line = re.sub(r'Z\s*\'\s*', 'Ž', line)
        line = re.sub(r'S\s*\'\s*', 'Š', line)
        line = re.sub(r'D\s*\'\s*', 'Đ', line)
        
        # Price format fixes for EUR
        line = re.sub(r'(\d)\s*[,\.]\s*(\d{2})(?=\s|$)', r'\1.\2', line)  # Fix price decimals
        line = re.sub(r'(\d+)\s*[eE][uU][rR]', r'\1 EUR', line)  # Fix Euro currency
        
        # Clean up excessive whitespace within the line
        line = re.sub(r'\s+', ' ', line)
        line = line.strip()
        
        # Skip lines that are only symbols or very short - but be less aggressive
        if len(line) < 2:
            continue
        
        # Skip lines with only symbols/punctuation - FIXED regex
        if re.match(r'^[^\w\s]+$', line):
            continue
        
        # Skip obvious garbage lines - FIXED regex
        if not re.search(r'[a-zA-ZčćžšđČĆŽŠĐ\d]', line):
            continue
            
        cleaned_lines.append(line)
    
    # Join lines back together - PRESERVE the line breaks!
    result = '\n'.join(cleaned_lines)
    
    return result

def run_ocr_on_image(image: np.ndarray, configs: List[str] = None, lang: str = 'bos+hrv+eng', 
                    debug: bool = False, image_index: int = 0) -> List[Tuple[str, int, str]]:
    """
    Enhanced OCR with adaptive preprocessing - Croatian only.
    Maintains same interface as your original function.
    """
    results = []
    configs = configs or DEFAULT_CONFIGS
    
    # Assess image quality to determine best preprocessing strategy
    quality_metrics = assess_image_quality(image)
    
    if debug:
        logger.info(f"[OCR] Croatian Image #{image_index} - Brightness: {quality_metrics['mean_brightness']:.1f}, "
                   f"Contrast: {quality_metrics['contrast']:.1f}, Sharpness: {quality_metrics['sharpness']:.1f}")
    
    # Choose preprocessing strategies based on image quality - IMPROVED
    preprocessing_strategies = ["minimal"]  # Start with minimal processing
    
    ## For well-cropped receipts, add gentle options first
    #preprocessing_strategies.extend(["gentle_binary", "otsu_only"])
    
    #if quality_metrics['is_dark'] or quality_metrics['is_low_contrast']:
    #    preprocessing_strategies.append("high_contrast")
    
    #if quality_metrics['is_blurry']:
    #    preprocessing_strategies.append("sharp")
    
    #if quality_metrics['sharpness'] < 50:  # Very poor quality
    #    preprocessing_strategies.append("noise_reduction")
    
    # Try different preprocessing strategies
    for strategy in preprocessing_strategies:
        processed_image = preprocess_for_ocr(image, strategy)
        
        # SAVE DEBUG IMAGE for each strategy
        if debug:
            debug_dir = "debug_images"
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f"ocr_preprocess_{image_index}_{strategy}.png")
            cv2.imwrite(debug_path, processed_image)
            logger.info(f"[OCR] Saved debug image: {debug_path}")
        
        # Try each OCR configuration on this processed image
        for i, config in enumerate(configs):
            try:
                # Run Tesseract OCR with Croatian language
                text = pytesseract.image_to_string(processed_image, config=config, lang=lang)
                
                # Clean the extracted Croatian text
                cleaned_text = clean_text(text)
                
                # Only accept meaningful results
                if cleaned_text and len(cleaned_text.strip()) > 5:  # Lowered from 10 to 5
                    config_name = f"{strategy}_{config.split()[0]}"
                    results.append((cleaned_text, i, config_name))
                    
                    if debug:
                        logger.info(f"[OCR] Croatian Image #{image_index}, strategy '{strategy}', "
                                  f"config #{i}: extracted {len(cleaned_text)} chars")
                        logger.info(f"[OCR] Preview: {cleaned_text[:100]}...")
                        
            except Exception as e:
                if debug:
                    logger.warning(f"[OCR] Croatian config #{i} failed on image #{image_index} "
                                 f"with strategy '{strategy}': {e}")
                continue
    
    # If no good results with preprocessing, try basic OCR on original
    if not results:
        if debug:
            logger.info(f"[OCR] No Croatian results from preprocessing, trying original image")
        
        for i, config in enumerate(configs[:3]):  # Try only basic configs on original
            try:
                text = pytesseract.image_to_string(image, config=config, lang=lang)
                cleaned_text = clean_text(text)
                
                if cleaned_text and len(cleaned_text.strip()) > 3:  # Even more lenient for fallback
                    results.append((cleaned_text, i, f"original_{config.split()[0]}"))
                    if debug:
                        logger.info(f"[OCR] Original Croatian result: {cleaned_text[:50]}...")
                    
            except Exception as e:
                if debug:
                    logger.warning(f"[OCR] Croatian original image config #{i} failed: {e}")
                continue
    
    if debug:
        logger.info(f"[OCR] Croatian Image #{image_index} - Total successful extractions: {len(results)}")
        if results:
            best_result = max(results, key=lambda x: len(x[0]))
            logger.info(f"[OCR] Croatian best result preview: {best_result[0][:100]}...")
    
    return results