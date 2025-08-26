#!/usr/bin/env python3
"""
Receipt OCR Test Results Analyzer
=================================
Analyzes the test results and provides insights
"""

import json
import os
from datetime import datetime

def analyze_results(results_file='test_results.json'):
    """Analyze the test results and provide detailed insights"""
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"❌ Error reading results: {e}")
        return
    
    if not results:
        print("❌ No results found in file")
        return
    
    print("📊 RECEIPT OCR TEST RESULTS ANALYSIS")
    print("=" * 50)
    
    total_images = len(results)
    successful_full_pipeline = 0
    successful_processing = 0
    successful_ocr = 0
    
    print(f"📁 Total Images Tested: {total_images}")
    print(f"⏰ Test Period: {results[0].get('timestamp', 'Unknown')} to {results[-1].get('timestamp', 'Unknown')}")
    print()
    
    # Detailed analysis for each image
    for i, result in enumerate(results, 1):
        print(f"🖼️  IMAGE {i}: {result.get('file', 'Unknown')}")
        print("-" * 40)
        
        # Image info
        if 'image_size' in result:
            size = result['image_size']
            print(f"   📏 Dimensions: {size[1]}x{size[0]} pixels")
        
        tests = result.get('tests', {})
        
        # Full Pipeline Results
        pipeline = tests.get('full_pipeline', {})
        if pipeline.get('success'):
            successful_full_pipeline += 1
            print(f"   ✅ PIPELINE: SUCCESS")
            print(f"      📊 Confidence: {pipeline.get('confidence', 0):.2f}")
            print(f"      🏪 Store: {pipeline.get('store', 'N/A')}")
            print(f"      💰 Total: {pipeline.get('total', 'N/A')}")
            print(f"      📅 Date: {pipeline.get('date', 'N/A')}")
            print(f"      📦 Items: {pipeline.get('items_count', 0)}")
            print(f"      📝 Text Length: {pipeline.get('raw_text_length', 0)} chars")
        else:
            print(f"   ❌ PIPELINE: FAILED")
            print(f"      🚨 Error: {pipeline.get('error', 'Unknown error')}")
        
        # Image Processing Results
        processing = tests.get('image_processing', {})
        if processing.get('success'):
            successful_processing += 1
            print(f"   ✅ PROCESSING: {processing.get('status', 'Unknown status')}")
            print(f"      🖼️  Images Generated: {processing.get('images_generated', 0)}")
        else:
            print(f"   ❌ PROCESSING: FAILED")
            print(f"      🚨 Error: {processing.get('error', 'Unknown error')}")
        
        # OCR Results
        ocr = tests.get('direct_ocr', {})
        if ocr.get('success'):
            successful_ocr += 1
            print(f"   ✅ OCR: SUCCESS")
            print(f"      🌍 Best Language: {ocr.get('best_language', 'Unknown')}")
            print(f"      📊 Score: {ocr.get('score', 0)}")
            print(f"      📝 Text Length: {ocr.get('text_length', 0)} chars")
            text_preview = ocr.get('text_preview', '')
            if text_preview:
                print(f"      👁️  Preview: {text_preview[:100]}...")
        else:
            print(f"   ❌ OCR: FAILED")
        
        # Debug Images
        debug = tests.get('debug_images', {})
        if 'total_images' in debug:
            print(f"   🖼️  Debug Images: {debug.get('total_images', 0)} generated")
            key_images = debug.get('key_images_found', [])
            if key_images:
                print(f"      🔑 Key Images: {', '.join(key_images)}")
        
        print()
    
    # Summary Statistics
    print("📈 SUMMARY STATISTICS")
    print("=" * 30)
    print(f"Full Pipeline Success Rate: {successful_full_pipeline}/{total_images} ({successful_full_pipeline/total_images*100:.1f}%)")
    print(f"Image Processing Success Rate: {successful_processing}/{total_images} ({successful_processing/total_images*100:.1f}%)")
    print(f"OCR Success Rate: {successful_ocr}/{total_images} ({successful_ocr/total_images*100:.1f}%)")
    
    # Calculate average confidence for successful pipelines
    successful_results = [r for r in results if r.get('tests', {}).get('full_pipeline', {}).get('success')]
    if successful_results:
        confidences = [r['tests']['full_pipeline']['confidence'] for r in successful_results]
        avg_confidence = sum(confidences) / len(confidences)
        print(f"Average Confidence: {avg_confidence:.2f}")
        print(f"Confidence Range: {min(confidences):.2f} - {max(confidences):.2f}")
    
    # Common Issues Analysis
    print(f"\n🔍 COMMON ISSUES ANALYSIS")
    print("=" * 30)
    
    failed_results = [r for r in results if not r.get('tests', {}).get('full_pipeline', {}).get('success')]
    if failed_results:
        error_types = {}
        for result in failed_results:
            error = result.get('tests', {}).get('full_pipeline', {}).get('error', 'Unknown')
            error_types[error] = error_types.get(error, 0) + 1
        
        print("Most Common Errors:")
        for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {error}: {count} times")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 20)
    
    if successful_full_pipeline == 0:
        print("🚨 CRITICAL: No successful pipeline runs!")
        print("   1. Check if Tesseract is installed and configured")
        print("   2. Verify image quality and format")
        print("   3. Check debug images for processing issues")
    elif successful_full_pipeline < total_images * 0.5:
        print("⚠️  LOW SUCCESS RATE:")
        print("   1. Review failed image characteristics")
        print("   2. Consider image preprocessing improvements")
        print("   3. Check OCR language settings")
    else:
        print("✅ GOOD SUCCESS RATE:")
        print("   1. Review confidence scores for optimization")
        print("   2. Fine-tune parsing rules if needed")
    
    if successful_processing > successful_full_pipeline:
        print("📝 OCR/PARSING ISSUES:")
        print("   1. Images are being processed but OCR/parsing fails")
        print("   2. Check Tesseract installation and language packs")
        print("   3. Review parsing logic for your receipt formats")
    
    # Save analysis to JSON file
    analysis_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_images": total_images,
        "success_rates": {
            "full_pipeline": f"{successful_full_pipeline}/{total_images} ({successful_full_pipeline/total_images*100:.1f}%)",
            "image_processing": f"{successful_processing}/{total_images} ({successful_processing/total_images*100:.1f}%)",
            "ocr": f"{successful_ocr}/{total_images} ({successful_ocr/total_images*100:.1f}%)"
        },
        "confidence_stats": {},
        "failed_results": [],
        "successful_results": []
    }
    
    # Add confidence statistics if available
    if successful_results:
        confidences = [r['tests']['full_pipeline']['confidence'] for r in successful_results]
        analysis_summary["confidence_stats"] = {
            "average": round(sum(confidences) / len(confidences), 2),
            "min": round(min(confidences), 2),
            "max": round(max(confidences), 2),
            "count": len(confidences)
        }
    
    # Categorize results
    for result in results:
        pipeline_test = result.get('tests', {}).get('full_pipeline', {})
        result_summary = {
            "file": result.get('file'),
            "success": pipeline_test.get('success', False),
            "confidence": pipeline_test.get('confidence'),
            "store": pipeline_test.get('store'),
            "total": pipeline_test.get('total'),
            "items_count": pipeline_test.get('items_count'),
            "date": pipeline_test.get('date'),
            "time": pipeline_test.get('time'),
            "error": pipeline_test.get('error')
        }
        
        if pipeline_test.get('success'):
            analysis_summary["successful_results"].append(result_summary)
        else:
            analysis_summary["failed_results"].append(result_summary)
    
    # Add common errors analysis
    if failed_results:
        error_types = {}
        for result in failed_results:
            error = result.get('tests', {}).get('full_pipeline', {}).get('error', 'Unknown')
            error_types[error] = error_types.get(error, 0) + 1
        analysis_summary["common_errors"] = error_types
    
    # Save to JSON file
    analysis_file = 'analysis_summary.json'
    try:
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Analysis summary saved to: {analysis_file}")
    except Exception as e:
        print(f"❌ Could not save analysis summary: {e}")
    
    return results

def show_debug_images():
    """Show available debug images"""
    debug_dir = "debug_images"
    if not os.path.exists(debug_dir):
        print(f"❌ Debug directory not found: {debug_dir}")
        return
    
    debug_files = [f for f in os.listdir(debug_dir) if f.endswith('.png')]
    if not debug_files:
        print(f"❌ No debug images found in {debug_dir}")
        return
    
    print(f"\n🖼️  DEBUG IMAGES AVAILABLE ({len(debug_files)} files)")
    print("=" * 40)
    
    # Group by processing step
    steps = {}
    for file in debug_files:
        if 'step' in file:
            step = file.split('_')[0]
            if step not in steps:
                steps[step] = []
            steps[step].append(file)
        else:
            if 'misc' not in steps:
                steps['misc'] = []
            steps['misc'].append(file)
    
    for step, files in sorted(steps.items()):
        print(f"\n{step.upper()}:")
        for file in sorted(files):
            print(f"   📁 {file}")
    
    print(f"\n💡 To view images, open files in: {os.path.abspath(debug_dir)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Receipt OCR Test Results')
    parser.add_argument('--results', '-r', default='test_results.json', help='Results JSON file')
    parser.add_argument('--debug-images', '-d', action='store_true', help='Show debug images info')
    
    args = parser.parse_args()
    
    # Analyze results
    results = analyze_results(args.results)
    
    # Show debug images if requested
    if args.debug_images:
        show_debug_images()
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Review debug images: python results_analyzer.py --debug-images")
    print("2. Check specific failed images for patterns")
    print("3. Adjust OCR settings or image preprocessing based on results")
    print("4. Test with different image types if needed")