import re
from datetime import datetime

ocr_fixes = {
    'ukupko': 'ukupno',
    'ukupko (eur);': 'ukupno',
    'ukupko (eur)': 'ukupno',
    'ukupko:': 'ukupno',
    'suma': 'ukupno',
    'totai': 'total',
    'totál': 'total',
    'iznos': 'ukupno',
    'bar,': 'lubar',
    'lu bar': 'lubar',
    '"lubar"': 'lubar',
    'račun broj': 'datum',
    'račun': 'datum',
    'u račun': 'datum',
    'gg': '',
    'g9': '',
}

def fix_text(text):
    for wrong, right in ocr_fixes.items():
        text = text.replace(wrong, right)
    return text

def detect_store_name(lines):
    """Detects the store name from top lines."""
    for i, line in enumerate(lines[:5]):
        fixed = fix_text(line.lower())
        if any(x in fixed for x in ['caffe bar', 'restoran', 'pekara']):
            return "LuBar" if 'lubar' in fixed else line.strip()
        elif len(line.strip()) > 3:
            return line.strip()

    # Fallback: first clean non-numeric line
    for line in lines[:5]:
        text = line.strip('"\'')
        if any(char.isdigit() for char in text) or ',' in text or ':' in text:
            continue
        if re.match(r'^[A-ZČĆŽŠĐa-zčćžšđ ]+$', text):
            return text
    return "Unknown"

def extract_date(lines):
    """Extracts date and optional time from lines."""
    for line in lines:
        fixed_line = fix_text(line.lower())
        match = re.search(r'datum\s*[:\-]?\s*(\d{1,2}[.,/-]\d{1,2}[.,/-]\d{2,4})[^0-9]*(\d{1,2}:\d{2})?', fixed_line)
        if match:
            try:
                date = match.group(1).replace(',', '.').replace('/', '.').replace('-', '.')
                parsed_date = datetime.strptime(date, "%d.%m.%Y").strftime("%d-%m-%Y")
            except:
                parsed_date = date
            return parsed_date, match.group(2)
    return None, None

def parse_items(lines):
    """Attempts to extract individual items from OCR lines."""
    items = []
    for line in lines:
        line_fixed = fix_text(line.lower())

        # Skip known labels
        if any(k in line_fixed for k in ['ukupno', 'total', 'datum', 'vrijeme', 'račun', 'bon', 'pdv', 'tax']):
            continue

        # Match strict item line
        item_match = re.match(r'^(.+?)\s+(\d+[xX])?\s*(\d+[,.]\d{2})\s+(\d+[,.]\d{2})$', line)
        if item_match:
            try:
                name = item_match.group(1).strip()
                quantity = item_match.group(2)[:-1] if item_match.group(2) else '1'
                price = item_match.group(4).replace(',', '.')
                items.append({
                    "name": name,
                    "quantity": int(quantity),
                    "price_per_item": float(item_match.group(3).replace(',', '.')),
                    "total": float(price)
                })
                continue
            except:
                pass

        # Fallback: find two prices
        prices = re.findall(r'(\d+[,.]\d{2})', line)
        if len(prices) >= 2:
            name = line.split(prices[0])[0].strip()
            try:
                items.append({
                    "name": name,
                    "quantity": 1,
                    "price_per_item": float(prices[-2].replace(',', '.')),
                    "total": float(prices[-1].replace(',', '.'))
                })
            except:
                continue
    return items

def parse_receipt(lines, log_debug=False):
    """
    Main OCR parsing function. Returns structured dict with receipt data.
    """
    def dbg(msg):
        if log_debug:
            print("[DEBUG]", msg)

    dbg("Starting parse_receipt")
    lines = [fix_text(line) for line in lines]

    store = detect_store_name(lines)
    date, time = extract_date(lines)
    items = parse_items(lines)
    total = None  # This could be improved later with total line detection

    return {
        "store": store,
        "location_label": None,
        "date": date,
        "time": time,
        "total": total,
        "items": items
    }
