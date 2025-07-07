import re
from datetime import datetime

def parse_receipt(lines):
    store_name = lines[0] if lines else "Unknown"
    date, time = None, None
    total = None
    items = []

    for line in lines:
        lower_line = line.lower()

        # --- Handle common OCR mistakes ---
        lower_line = lower_line.replace('ukupko', 'ukupno')
        lower_line = lower_line.replace('datum', 'datum')

        # --- Date and time ---
        if not date or not time:
            dt_match = re.search(r'datum\s*[:\-]?\s*(\d{1,2}[.,/-]\d{1,2}[.,/-]\d{2,4})[^0-9]*(\d{1,2}:\d{2})?', lower_line)
            if dt_match:
                date_raw = dt_match.group(1).replace(',', '.').replace('/', '.').replace('-', '.')
                time_raw = dt_match.group(2)
                try:
                    date = datetime.strptime(date_raw, "%d.%m.%Y").strftime("%d-%m-%Y")
                except:
                    date = date_raw
                time = time_raw

        # --- Total price ---
        if 'ukupno' in lower_line or 'total' in lower_line:
            price_match = re.search(r'(\d+[,.]\d{2})', line)
            if price_match:
                total = price_match.group(1).replace(',', '.')

        # --- Items ---
        item_match = re.match(r'^(.+?)\s+(\d+[xX])?\s*(\d+[,.]\d{2})\s+(\d+[,.]\d{2})$', line)
        if item_match:
            name = item_match.group(1).strip()
            quantity = item_match.group(2)[:-1] if item_match.group(2) else '1'
            price = item_match.group(4).replace(',', '.')
            items.append({
                "name": name,
                "quantity": int(quantity),
                "price": float(price)
            })
        else:
            # fallback pattern
            split = re.findall(r'(\d+[,.]\d{2})', line)
            if len(split) >= 1:
                name_part = re.split(r'\d+[,.]\d{2}', line)[0].strip()
                price_val = split[-1].replace(',', '.')
                items.append({
                    "name": name_part,
                    "quantity": 1,
                    "price": float(price_val)
                })

    return {
        "store": store_name,
        "date": date,
        "time": time,
        "total": float(total) if total else None,
        "items": items
    }
