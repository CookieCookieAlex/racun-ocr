#!/usr/bin/env python3
"""
Receipt OCR Detailed Debugger
============================
Shows exactly what's being extracted from your receipts
"""

import sys
import os
import json

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

try:
    from app.ocr.engine.main_ocr import OCRPipeline
    from app.ocr.parser import ReceiptParser
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def debug_single_receipt(image_path):
    """Debug a single receipt in detail"""
    print(f"\n🔍 DETAILED RECEIPT ANALYSIS")
    print(f"📁 File: {image_path}")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
    
    # Run OCR Pipeline
    ocr_pipeline = OCRPipeline(debug=True)
    result = ocr_pipeline.process_single_receipt(image_path)
    
    if not result.get('success'):
        print(f"❌ OCR Pipeline Failed: {result.get('error')}")
        return
    
    raw_text = result.get('raw_text', '')
    parsed_data = result.get('parsed_data', {})
    
    print(f"\n📝 RAW OCR TEXT ({len(raw_text)} characters)")
    print("-" * 30)
    if raw_text:
        lines = raw_text.split('\n')
        for i, line in enumerate(lines, 1):
            if line.strip():
                print(f"{i:2d}: {line}")
    else:
        print("❌ No text extracted")
    
    print(f"\n🎯 TARGET EXTRACTION RESULTS")
    print("-" * 35)
    
    # Store Name
    store = parsed_data.get('store', 'Not found')
    print(f"🏪 Store Name: {store}")
    if store == "Unknown Store":
        print("   💡 Tip: Store name not detected in first 10 lines")
    
    # Date
    date = parsed_data.get('date', 'Not found')
    print(f"📅 Date: {date}")
    if not date:
        print("   💡 Tip: No date patterns found (datum, dd.mm.yyyy)")
    
    # Time  
    time = parsed_data.get('time', 'Not found')
    print(f"⏰ Time: {time}")
    if not time:
        print("   💡 Tip: No time patterns found (HH:MM)")
    
    # Total
    total = parsed_data.get('total', 'Not found')
    print(f"💰 Total: {total}")
    if not total:
        print("   💡 Tip: No total keywords found (ukupno, total, suma)")
    
    # Items
    items = parsed_data.get('items', [])
    print(f"📦 Items Found: {len(items)}")
    
    if items:
        print(f"\n📋 ITEM DETAILS")
        print("-" * 20)
        for i, item in enumerate(items, 1):
            print(f"   {i}. {item.get('name', 'No name')}")
            print(f"      Quantity: {item.get('quantity', 'N/A')}")
            print(f"      Price per item: {item.get('price_per_item', 'N/A')}")
            print(f"      Total: {item.get('total', 'N/A')}")
            print()
    else:
        print("   ❌ No items detected")
        print("   💡 Tip: Items should have format like 'Name 2x 1.50 3.00'")
    
    # Overall confidence
    confidence = result.get('confidence', 0)
    print(f"📊 Overall Confidence: {confidence:.2f}")
    
    if confidence < 0.3:
        print("   🚨 Very low confidence - major issues")
    elif confidence < 0.6:
        print("   ⚠️  Low confidence - needs improvement")
    elif confidence < 0.8:
        print("   ✅ Good confidence - minor tweaks needed")
    else:
        print("   🏆 Excellent confidence!")
    
    return {
        'raw_text': raw_text,
        'parsed_data': parsed_data,
        'confidence': confidence,
        'success_breakdown': {
            'store_found': store != "Unknown Store" and store != "Not found",
            'date_found': date != "Not found" and date is not None,
            'time_found': time != "Not found" and time is not None,
            'total_found': total != "Not found" and total is not None,
            'items_found': len(items) > 0
        }
    }

def suggest_improvements(debug_results):
    """Suggest specific improvements based on debug results"""
    print(f"\n💡 SPECIFIC IMPROVEMENT SUGGESTIONS")
    print("=" * 40)
    
    success = debug_results['success_breakdown']
    raw_text = debug_results['raw_text']
    
    if not success['store_found']:
        print("🏪 Store Name Issues:")
        lines = [line.strip() for line in raw_text.split('\n')[:5] if line.strip()]
        if lines:
            print("   First few lines of receipt:")
            for line in lines:
                print(f"     '{line}'")
            print("   💡 Add these patterns to store_patterns in parser.py")
        else:
            print("   ❌ No readable text in header - OCR issue")
    
    if not success['date_found']:
        print("📅 Date Issues:")
        import re
        date_candidates = []
        for line in raw_text.split('\n'):
            if re.search(r'\d{1,2}[.,/-]\d{1,2}[.,/-]\d{2,4}', line):
                date_candidates.append(line.strip())
        if date_candidates:
            print("   Found potential dates:")
            for candidate in date_candidates[:3]:
                print(f"     '{candidate}'")
            print("   💡 These should be detected - check date patterns")
        else:
            print("   ❌ No date-like patterns found in OCR text")
    
    if not success['total_found']:
        print("💰 Total Amount Issues:")
        import re
        price_candidates = []
        for line in raw_text.split('\n'):
            if any(word in line.lower() for word in ['ukupno', 'total', 'suma']):
                price_candidates.append(line.strip())
        if price_candidates:
            print("   Found lines with total keywords:")
            for candidate in price_candidates:
                print(f"     '{candidate}'")
            print("   💡 Price extraction may need improvement")
        else:
            print("   ❌ No total keywords found - may need more patterns")
    
    if not success['items_found']:
        print("📦 Items Issues:")
        import re
        item_candidates = []
        for line in raw_text.split('\n'):
            if re.search(r'\d+[,.]\d{2}', line) and len(line.strip()) > 5:
                item_candidates.append(line.strip())
        if item_candidates:
            print("   Found lines with prices (potential items):")
            for candidate in item_candidates[:5]:
                print(f"     '{candidate}'")
            print("   💡 Item parsing patterns may need adjustment")
        else:
            print("   ❌ No price-containing lines found")
    
    # Overall recommendation
    success_count = sum(success.values())
    print(f"\n🎯 PRIORITY ACTIONS:")
    if success_count == 0:
        print("1. 🚨 CRITICAL: Fix OCR - no data extracted")
        print("2. Check debug images for processing issues")
        print("3. Verify Tesseract installation")
    elif success_count < 3:
        print("1. ⚠️  Focus on parsing improvements")
        print("2. Add missing patterns for your receipt format")
        print("3. Test with higher quality images")
    else:
        print("1. ✅ System mostly working!")
        print("2. Fine-tune remaining extraction issues")
        print("3. Test with more receipt varieties")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug Receipt OCR Extraction')
    parser.add_argument('image', help='Path to receipt image')
    
    args = parser.parse_args()
    
    # Debug the receipt
    debug_results = debug_single_receipt(args.image)
    
    if debug_results:
        # Provide improvement suggestions
        suggest_improvements(debug_results)
        
        # Show debug files
        print(f"\n📁 Check these files for more details:")
        print("   • debug_images/ - Processing steps")
        print("   • debug_images/best_ocr_text.txt - Raw extracted text")
    
    print(f"\n🚀 Next: Run this on multiple receipts to find patterns!")

if __name__ == "__main__":
    main()