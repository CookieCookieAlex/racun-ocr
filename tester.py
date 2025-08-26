#!/usr/bin/env python3
"""
Receipt OCR Tester
==================
Simple tool to test your receipt OCR pipeline on local images.
Usage: python tester.py path/to/receipt.jpg
"""

import sys
import os
import cv2
import json
import argparse
from datetime import datetime

# Add the parent directory to Python path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from app.ocr.engine.main_ocr import OCRPipeline
    from app.ocr.processor.core import ReceiptProcessor
    from app.ocr.engine.ocr_engine import run_ocr_on_image
    from app.ocr.engine.scoring import score_ocr_text
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from your project root directory!")
    print("Available modules:")
    try:
        import app
        print("✅ app module found")
    except:
        print("❌ app module not found")
    sys.exit(1)

class ReceiptTester:
    def __init__(self, debug=True):
        self.debug = debug
        self.results = []
    
    def test_image(self, image_path):
        """Test a single receipt image"""
        print(f"\n🧪 TESTING: {os.path.basename(image_path)}")
        print("=" * 60)
        
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            return None
        
        # Load image
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Could not load image: {image_path}")
                return None
            
            print(f"✅ Image loaded: {img.shape}")
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
        
        result = {
            'file': os.path.basename(image_path),
            'timestamp': datetime.now().isoformat(),
            'image_size': img.shape,
            'tests': {}
        }
        
        # Test 1: Full OCR Pipeline
        print("\n📋 TEST 1: Full OCR Pipeline")
        print("-" * 30)
        try:
            ocr_pipeline = OCRPipeline(debug=self.debug)
            pipeline_result = ocr_pipeline.process_single_receipt(image_path)
            
            if pipeline_result.get('success'):
                print(f"✅ Pipeline Success")
                print(f"   Status: {pipeline_result.get('process_status')}")
                print(f"   Confidence: {pipeline_result.get('confidence', 0):.2f}")
                parsed_data = pipeline_result.get('parsed_data', {})
                items = parsed_data.get('items', [])
                print(f"   Items Found: {len(items)}")
                print(f"   Store: {parsed_data.get('store', 'N/A')}")
                print(f"   Total: {parsed_data.get('total', 'N/A')}")
                print(f"   Date: {parsed_data.get('date', 'N/A')}")
                
                # Show first few items
                if items:
                    print(f"   Sample items:")
                    for i, item in enumerate(items[:3]):
                        print(f"     {i+1}. {item.get('name', 'N/A')} - {item.get('quantity', 1)}x {item.get('price_per_item', 0)} = {item.get('total', 0)}")
                
                result['tests']['full_pipeline'] = {
                    'success': True,
                    'confidence': pipeline_result.get('confidence'),
                    'items_count': len(items),
                    'total': parsed_data.get('total'),
                    'store': parsed_data.get('store'),
                    'date': parsed_data.get('date'),
                    'time': parsed_data.get('time'),
                    'raw_text_length': len(pipeline_result.get('raw_text', '')),
                    'items': items,  # Full items array
                    'full_parsed_data': parsed_data  # Complete parsed data for verification
                }
            else:
                print(f"❌ Pipeline Failed: {pipeline_result.get('error', 'Unknown error')}")
                result['tests']['full_pipeline'] = {'success': False, 'error': pipeline_result.get('error')}
                
        except Exception as e:
            print(f"❌ Pipeline Exception: {e}")
            result['tests']['full_pipeline'] = {'success': False, 'error': str(e)}
        
        # Test 2: Just Image Processing
        print("\n🖼️  TEST 2: Image Processing Only")
        print("-" * 35)
        try:
            processor = ReceiptProcessor(debug=self.debug)
            processed_images, status, info = processor.process_receipt(img)
            
            print(f"✅ Processing Status: {status}")
            print(f"   Images Generated: {len(processed_images)}")
            print(f"   Info: {info}")
            
            result['tests']['image_processing'] = {
                'success': True,
                'status': status,
                'images_generated': len(processed_images),
                'info': info
            }
            
        except Exception as e:
            print(f"❌ Processing Exception: {e}")
            result['tests']['image_processing'] = {'success': False, 'error': str(e)}
        
        # Test 3: Direct OCR Test
        print("\n👁️  TEST 3: Direct OCR Test")
        print("-" * 30)
        try:
            # Test OCR on original image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            # Try different language combinations
            language_tests = [
                ('hrv+eng', 'Croatian + English'),
                ('eng', 'English only'),
                ('bos+hrv+eng', 'Bosnian + Croatian + English'),
            ]
            
            best_score = 0
            best_result = None
            
            for lang_code, lang_name in language_tests:
                try:
                    ocr_results = run_ocr_on_image(gray, lang=lang_code, debug=False)
                    
                    if ocr_results:
                        best_ocr = max(ocr_results, key=lambda x: len(x[0]))
                        text, config_idx, config_name = best_ocr
                        score = score_ocr_text(text)
                        
                        print(f"   {lang_name}: Score={score}, Length={len(text)}")
                        
                        if score > best_score:
                            best_score = score
                            best_result = (text, lang_name, score)
                    
                except Exception as e:
                    print(f"   {lang_name}: ❌ Failed - {e}")
            
            if best_result:
                text, lang_name, score = best_result
                print(f"✅ Best OCR: {lang_name} (Score: {score})")
                print(f"   Text Preview: {text[:100]}...")
                
                result['tests']['direct_ocr'] = {
                    'success': True,
                    'best_language': lang_name,
                    'score': score,
                    'text_length': len(text),
                    'text_preview': text[:200]
                }
            else:
                print("❌ No successful OCR results")
                result['tests']['direct_ocr'] = {'success': False}
                
        except Exception as e:
            print(f"❌ OCR Exception: {e}")
            result['tests']['direct_ocr'] = {'success': False, 'error': str(e)}
        
        # Test 4: Debug Images Check
        print("\n🖼️  TEST 4: Debug Images")
        print("-" * 25)
        debug_dir = "debug_images"
        if os.path.exists(debug_dir):
            debug_files = [f for f in os.listdir(debug_dir) if f.endswith('.png')]
            print(f"✅ Found {len(debug_files)} debug images in {debug_dir}/")
            
            key_images = [
                'step7_cropped_and_corrected.png',
                'step8c_text_denoised.png',
                'final_best_ocr_source.png'
            ]
            
            found_key_images = [img for img in key_images if img in debug_files]
            print(f"   Key images found: {len(found_key_images)}/{len(key_images)}")
            
            if debug_files:
                print(f"   Recent debug images: {', '.join(debug_files[:5])}")
            
            result['tests']['debug_images'] = {
                'total_images': len(debug_files),
                'key_images_found': found_key_images
            }
        else:
            print(f"❌ Debug directory not found: {debug_dir}")
            result['tests']['debug_images'] = {'success': False, 'error': 'No debug directory'}
        
        self.results.append(result)
        return result
    
    def save_results(self, output_file='test_results.json'):
        """Save test results to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Could not save results: {e}")
    
    def print_summary(self):
        """Print a summary of all tests"""
        if not self.results:
            return
        
        print(f"\n📊 TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.results)
        successful_pipelines = sum(1 for r in self.results if r['tests'].get('full_pipeline', {}).get('success'))
        successful_processing = sum(1 for r in self.results if r['tests'].get('image_processing', {}).get('success'))
        successful_ocr = sum(1 for r in self.results if r['tests'].get('direct_ocr', {}).get('success'))
        
        print(f"Total Images Tested: {total_tests}")
        print(f"Full Pipeline Success: {successful_pipelines}/{total_tests}")
        print(f"Image Processing Success: {successful_processing}/{total_tests}")
        print(f"Direct OCR Success: {successful_ocr}/{total_tests}")
        
        if successful_pipelines > 0:
            confidences = [r['tests'].get('full_pipeline', {}).get('confidence', 0) 
                          for r in self.results if r['tests'].get('full_pipeline', {}).get('success')]
            avg_confidence = sum(confidences) / len(confidences)
            print(f"Average Confidence: {avg_confidence:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Test Receipt OCR Pipeline')
    parser.add_argument('images', nargs='+', help='Path to receipt image(s) or directory containing images')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug output')
    parser.add_argument('--output', '-o', default='test_results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    print("🧪 RECEIPT OCR TESTER")
    print("===================")
    
    tester = ReceiptTester(debug=not args.no_debug)
    
    # Expand directories and collect all image files
    image_files = []
    for path in args.images:
        if os.path.isdir(path):
            # If it's a directory, find all image files
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                import glob
                image_files.extend(glob.glob(os.path.join(path, ext)))
        else:
            image_files.append(path)
    
    if not image_files:
        print("❌ No image files found!")
        return
    
    print(f"📁 Found {len(image_files)} image files to test")
    
    for image_path in image_files:
        tester.test_image(image_path)
    
    tester.print_summary()
    tester.save_results(args.output)
    
    print(f"\n✨ Testing complete!")
    print(f"Check debug_images/ folder for detailed debug images")

if __name__ == "__main__":
    main()