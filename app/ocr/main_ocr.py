"""
Complete Robust OCR Receipt Processing System
Combines improved image preprocessing with enhanced text parsing
"""

import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Any, Tuple, Optional
import logging
import os
import json
from pathlib import Path


# Import actual processor and parser from app.ocr
from app.ocr.processor import ReceiptProcessor
from app.ocr.parser import ReceiptParser

# Ensure debug directory exists
DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

class OCRPipeline:
    def __init__(self, debug: bool = True, tesseract_config: str = None):
        """
        Initialize the robust OCR system
        
        Args:
            debug: Enable debug image saving
            tesseract_config: Custom Tesseract configuration
        """
        self.processor = ReceiptProcessor(debug=debug)
        self.parser = ReceiptParser()
        self.debug = debug
        
        # Default Tesseract configurations for receipts
        self.tesseract_configs = [
            '--psm 1',
            '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzčćžšđČĆŽŠĐ.,:-/ ',
            '--psm 4 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzčćžšđČĆŽŠĐ.,:-/ ',
            '--psm 7',
            '--psm 8',
            '--psm 13',
        ]
        
        if tesseract_config:
            self.tesseract_configs.insert(0, tesseract_config)
    
    def extract_text_multiple_configs(self, image: np.ndarray) -> List[str]:
        """
        Extract text using multiple Tesseract configurations
        Returns list of text results for comparison
        """
        text_results = []
        
        for i, config in enumerate(self.tesseract_configs):
            try:
                text = pytesseract.image_to_string(image, config=config, lang='eng+hrv')
                if text.strip():
                    text_results.append(text.strip())
                    if self.debug:
                        logger.info(f"[OCR] Config {i+1} extracted {len(text.strip())} characters")
            except Exception as e:
                logger.warning(f"[OCR] Config {i+1} failed: {e}")
                continue
        
        return text_results
    
    def select_best_ocr_result(self, text_results: List[str]) -> str:
        """
        Select the best OCR result based on various criteria
        """
        if not text_results:
            return ""
        
        if len(text_results) == 1:
            return text_results[0]
        
        scored_results = []
        
        for text in text_results:
            score = 0
            lines = text.split('\n')
            
            # Score based on number of meaningful lines
            meaningful_lines = [line for line in lines if len(line.strip()) > 2]
            score += len(meaningful_lines) * 2
            
            # Score based on presence of prices
            price_count = len([line for line in lines if self._has_price_pattern(line)])
            score += price_count * 3
            
            # Score based on presence of date
            if self._has_date_pattern(text):
                score += 5
            
            # Score based on presence of store indicators
            if any(keyword in text.lower() for keyword in ['caffe', 'bar', 'restoran', 'pekara', 'market']):
                score += 3
            
            # Penalize very short results
            if len(text) < 50:
                score -= 5
            
            scored_results.append((text, score))
        
        # Return the highest scoring result
        scored_results.sort(key=lambda x: x[1], reverse=True)
        best_result = scored_results[0][0]
        
        if self.debug:
            logger.info(f"[OCR] Selected result with score {scored_results[0][1]}")
        
        return best_result
    
    def _has_price_pattern(self, text: str) -> bool:
        """Check if text contains price patterns"""
        import re
        return bool(re.search(r'\d+[,.]\d{2}', text))
    
    def _has_date_pattern(self, text: str) -> bool:
        """Check if text contains date patterns"""
        import re
        return bool(re.search(r'\d{1,2}[.,/-]\d{1,2}[.,/-]\d{2,4}', text))
    
    def process_single_receipt(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single receipt image and return structured data
        
        Args:
            image_path: Path to the receipt image
            
        Returns:
            Dictionary containing parsed receipt data and metadata
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            logger.info(f"[MAIN] Processing {image_path}")
            
            # Step 1: Image preprocessing
            processed_images, process_status = self.processor.process_receipt(img)
            
            if not processed_images:
                return {
                    "success": False,
                    "error": f"Image preprocessing failed: {process_status}",
                    "image_path": image_path
                }
            
            logger.info(f"[MAIN] Generated {len(processed_images)} processed versions")
            
            # Step 2: OCR on multiple processed versions
            all_text_results = []
            
            for i, processed_img in enumerate(processed_images):
                text_results = self.extract_text_multiple_configs(processed_img)
                all_text_results.extend(text_results)
                
                if self.debug:
                    # Save processed image for debugging
                    debug_path = os.path.join(DEBUG_DIR, f"processed_version_{i}.png")
                    cv2.imwrite(debug_path, processed_img)
            
            # Step 3: Select best OCR result
            best_text = self.select_best_ocr_result(all_text_results)
            
            if not best_text:
                return {
                    "success": False,
                    "error": "OCR failed to extract any text",
                    "image_path": image_path,
                    "process_status": process_status
                }
            
            # Step 4: Parse the text
            lines = [line.strip() for line in best_text.split('\n') if line.strip()]
            parsed_data = self.parser.parse_receipt(lines, log_debug=self.debug)
            
            # Step 5: Compile results
            result = {
                "success": True,
                "image_path": image_path,
                "process_status": process_status,
                "ocr_versions_tried": len(all_text_results),
                "raw_text": best_text,
                "parsed_data": parsed_data,
                "preprocessing_attempts": self.processor.preprocessing_attempts,
                "confidence": parsed_data.get('confidence', 0.0)
            }
            
            logger.info(f"[MAIN] Success! Confidence: {result['confidence']:.2f}, "
                       f"Items: {len(parsed_data['items'])}")
            
            return result
            
        except Exception as e:
            logger.error(f"[MAIN] Error processing {image_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path
            }
    
    def process_batch(self, image_directory: str, output_file: str = None) -> List[Dict[str, Any]]:
        """
        Process a batch of receipt images
        
        Args:
            image_directory: Directory containing receipt images
            output_file: Optional JSON file to save results
            
        Returns:
            List of processing results
        """
        image_dir = Path(image_directory)
        if not image_dir.exists():
            raise ValueError(f"Directory does not exist: {image_directory}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in image_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            logger.warning(f"No image files found in {image_directory}")
            return []
        
        logger.info(f"[BATCH] Processing {len(image_files)} images")
        
        results = []
        for image_file in image_files:
            result = self.process_single_receipt(str(image_file))
            results.append(result)
            
            # Log progress
            success_count = sum(1 for r in results if r['success'])
            logger.info(f"[BATCH] Progress: {len(results)}/{len(image_files)}, "
                       f"Success rate: {success_count/len(results)*100:.1f}%")
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"[BATCH] Results saved to {output_file}")
        
        # Summary statistics
        successful = [r for r in results if r['success']]
        logger.info(f"[BATCH] Summary: {len(successful)}/{len(results)} successful")
        
        if successful:
            avg_confidence = sum(r['confidence'] for r in successful) / len(successful)
            logger.info(f"[BATCH] Average confidence: {avg_confidence:.2f}")
        
        return results
    
    def analyze_failures(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze failed processing attempts to identify common issues
        """
        failures = [r for r in results if not r['success']]
        
        if not failures:
            return {"message": "No failures to analyze"}
        
        error_types = {}
        for failure in failures:
            error = failure.get('error', 'Unknown error')
            error_type = error.split(':')[0] if ':' in error else error
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_failures": len(failures),
            "failure_rate": len(failures) / len(results) * 100,
            "error_types": error_types,
            "recommendations": self._get_failure_recommendations(error_types)
        }
    
    def _get_failure_recommendations(self, error_types: Dict[str, int]) -> List[str]:
        """Generate recommendations based on failure patterns"""
        recommendations = []
        
        if "Image preprocessing failed" in error_types:
            recommendations.append("Consider adjusting image preprocessing parameters")
            recommendations.append("Check image quality - very blurry or dark images may fail")
        
        if "OCR failed to extract any text" in error_types:
            recommendations.append("Try different Tesseract configurations")
            recommendations.append("Consider image enhancement before OCR")
        
        if "Could not load image" in error_types:
            recommendations.append("Check image file paths and formats")
            recommendations.append("Ensure images are not corrupted")
        
        return recommendations

# Example usage and testing
def main():
    """Example usage of the robust OCR system"""
    
    # Initialize the system
    ocr_system = OCRPipeline(debug=True)
    
    # Process a single receipt
    # result = ocr_system.process_single_receipt("path/to/receipt.jpg")
    # print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Process a batch of receipts
    # results = ocr_system.process_batch("path/to/receipt/directory", "results.json")
    
    # Analyze failures
    # failure_analysis = ocr_system.analyze_failures(results)
    # print("Failure Analysis:", failure_analysis)
    
    print("Robust OCR system initialized successfully!")
    print("Use ocr_system.process_single_receipt(image_path) to process images")

if __name__ == "__main__":
    main()