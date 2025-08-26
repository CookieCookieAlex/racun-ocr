import re
import cv2
import numpy as np

def score_ocr_text(text: str) -> int:
    """
    Croatian-specific OCR text scoring function.
    Higher scores indicate better OCR quality for Croatian receipts.
    """
    if not text or len(text.strip()) < 5:
        return 0
    
    lines = [line for line in text.split('\n') if len(line.strip()) > 2]
    
    # Base score from number of meaningful lines
    score = len(lines) * 2
    
    # Price detection (Croatian decimal format)
    price_hits = sum(bool(re.search(r'\d+[,.]\d{2}', line)) for line in lines)
    score += price_hits * 3
    
    # Croatian store/business words
    croatian_business_words = [
        'caffe', 'bar', 'restoran', 'pekara', 'market', 'trgovina', 
        'konzum', 'tommy', 'spar', 'kaufland', 'lidl', 'plodine',
        'ina', 'omv', 'lukoil', 'mol', 'ljekarna', 'apoteka',
        'dućan', 'prodavaonica', 'kavana'
    ]
    
    has_croatian_business = any(word in text.lower() for word in croatian_business_words)
    if has_croatian_business:
        score += 5
    
    # Croatian receipt keywords
    croatian_receipt_words = [
        'ukupno', 'suma', 'račun', 'bon', 'datum', 'vrijeme',
        'pdv', 'porez', 'blagajna', 'kasa', 'operater'
    ]
    
    croatian_keyword_hits = sum(1 for word in croatian_receipt_words if word in text.lower())
    score += croatian_keyword_hits * 2
    
    # Date pattern (Croatian format DD.MM.YYYY or DD.MM.YY)
    if re.search(r'\d{1,2}\.\d{1,2}\.\d{2,4}', text):
        score += 5
    
    # Time pattern (HH:MM or HH.MM)
    if re.search(r'\d{1,2}[:.]\d{2}', text):
        score += 3
    
    # EUR currency detection
    if re.search(r'(\d+[,.]\d{2})\s*(?:eur|€)', text.lower()):
        score += 4
    
    # Croatian characters bonus (ć, č, ž, š, đ)
    croatian_chars = len(re.findall(r'[ćčžšđĆČŽŠĐ]', text))
    score += min(croatian_chars, 5)  # Max 5 points for Croatian characters
    
    # Penalty for very short text
    if len(text) < 50:
        score -= 5
    
    # Penalty for excessive garbage characters
    garbage_chars = len(re.findall(r'[^\w\sčćžšđĆČŽŠĐ.,:\/€()[\]{}-]', text))
    if garbage_chars > len(text) * 0.2:  # More than 20% garbage
        score -= 10
    
    # Bonus for reasonable text length
    if 100 <= len(text) <= 1000:
        score += 3
    
    # Bonus for multiple price entries (typical for receipts)
    if price_hits >= 3:
        score += 5
    
    return max(score, 0)  # Ensure non-negative score

def score_receipt_contour(contour, image_area, image_shape, gray_image):
    """
    Score contour for receipt detection - prioritizes white receipt areas over dark backgrounds.
    This fixes the wood grain detection issue by heavily favoring white areas.
    """
    area = cv2.contourArea(contour)
    if area < 10000:  # Must be substantial for receipt
        return 0.0
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    h_img, w_img = image_shape
    
    score = 0.0
    
    # 1. WHITENESS SCORE - Key fix for wood grain issue
    mask = np.zeros((h_img, w_img), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_intensity = cv2.mean(gray_image, mask=mask)[0]
    
    if mean_intensity > 200:  # Very white (receipt paper)
        score += 50.0
    elif mean_intensity > 180:
        score += 30.0
    elif mean_intensity > 160:
        score += 10.0
    else:
        score -= 20.0  # Penalty for dark areas (wood grain)
    
    # 2. Position score - receipts are usually center-right in images
    center_x_ratio = (x + w/2) / w_img
    if 0.5 <= center_x_ratio <= 0.8:  # Center-right where receipts typically are
        score += 20.0
    elif 0.4 <= center_x_ratio <= 0.9:
        score += 10.0
    elif center_x_ratio < 0.4:  # Too far left (likely background)
        score -= 15.0
    
    # 3. Aspect ratio - receipts are tall
    aspect_ratio = h / w if w > 0 else 0
    if 1.5 <= aspect_ratio <= 3.5:  # Good receipt aspect ratio
        score += 25.0
    elif 1.2 <= aspect_ratio <= 4.0:
        score += 15.0
    elif aspect_ratio < 1.0:  # Wide shapes (wood grain patterns)
        score -= 25.0  # Big penalty for wide shapes
    else:
        score += 5.0
    
    # 4. Area score
    area_ratio = area / image_area
    if 0.15 <= area_ratio <= 0.5:  # Good receipt size
        score += 15.0
    elif 0.1 <= area_ratio <= 0.7:
        score += 8.0
    elif area_ratio > 0.8:  # Too large (probably whole image)
        score -= 30.0
    elif area_ratio < 0.05:  # Too small
        score -= 10.0
    
    # 5. Width coverage - receipts shouldn't span full width
    width_ratio = w / w_img
    if 0.3 <= width_ratio <= 0.6:  # Good receipt width
        score += 15.0
    elif width_ratio > 0.8:  # Too wide (probably background)
        score -= 20.0
    
    # 6. Rectangularity bonus
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    if len(approx) == 4:
        score += 10.0
    elif len(approx) <= 8:
        score += 5.0
    
    # 7. Margin bonus - receipts usually don't touch image edges
    edge_margin = 20
    touches_edge = (x < edge_margin or y < edge_margin or 
                   (x + w) > (w_img - edge_margin) or (y + h) > (h_img - edge_margin))
    
    if not touches_edge:
        score += 10.0  # Bonus for having margins around receipt
    
    return score

def score_croatian_receipt_quality(parsed_data: dict) -> float:
    """
    Additional scoring function for parsed Croatian receipt data.
    Returns confidence score between 0.0 and 1.0.
    """
    if not parsed_data:
        return 0.0
    
    confidence = 0.0
    
    # Store name quality
    store = parsed_data.get('store', '')
    if store and store != 'Nepoznata trgovina':
        confidence += 0.2
        
        # Bonus for Croatian store patterns
        croatian_stores = ['konzum', 'tommy', 'spar', 'kaufland', 'lidl', 'plodine', 'ina', 'omv']
        if any(pattern in store.lower() for pattern in croatian_stores):
            confidence += 0.1
    
    # Date quality
    date = parsed_data.get('date')
    if date:
        confidence += 0.15
        if re.match(r'\d{4}-\d{2}-\d{2}', date):
            confidence += 0.05
    
    # Items quality
    items = parsed_data.get('items', [])
    if items:
        confidence += 0.3
        
        # Quality of item names
        good_items = sum(1 for item in items if len(item.get('name', '')) > 3)
        item_quality = good_items / len(items) if items else 0
        confidence += item_quality * 0.1
        
        # Multiple items bonus
        if len(items) > 1:
            confidence += 0.05
    
    # Total amount quality
    total = parsed_data.get('total')
    if total and total > 0:
        confidence += 0.2
        
        # Check if total matches items
        if items:
            calculated_total = sum(item.get('total', 0) for item in items)
            if calculated_total > 0 and abs(total - calculated_total) < 0.01:
                confidence += 0.1
    
    return min(confidence, 1.0)