import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class ReceiptParser:
    def __init__(self):
        self.ocr_fixes = {
            'ukupko': 'ukupno',
            'suma': 'ukupno',
            'prirdna': 'prirodna',
            'haraneada': 'narančada',
            'datun': 'datum',
        }

        self.store_indicators = ['caffe bar', 'cafe bar', 'bar', 'pekara', 'restoran', 'market']

        self.date_patterns = [
            r'datum\s*[:\-]?\s*(\d{1,2}[.,/-]\d{1,2}[.,/-]\d{2,4})',
            r'(\d{1,2}[.,/-]\d{1,2}[.,/-]\d{4})\s+(\d{1,2}:\d{2})',
            r'(\d{1,2}[.,/-]\d{1,2}[.,/-]\d{4})',
        ]

        self.time_patterns = [
            r'(\d{1,2}:\d{2}(?::\d{2})?)',
        ]

    def fuzzy_fix_text(self, text: str) -> str:
        fixed_text = text.lower()
        for wrong, right in self.ocr_fixes.items():
            fixed_text = fixed_text.replace(wrong, right)
        return fixed_text

    def detect_store_name_contextual(self, lines: List[str]) -> str:
        for i, line in enumerate(lines[:10]):
            line_lower = line.lower().strip().strip('"\'')
            
            for indicator in self.store_indicators:
                if indicator in line_lower:
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip().strip('"\'')
                        if (len(next_line) > 1 and 
                            not re.search(r'\d{5,}', next_line) and
                            not re.search(r'\d{2}[.,/-]\d{2}[.,/-]\d{2,4}', next_line) and
                            re.search(r'[a-zA-ZčćžšđČĆŽŠĐ]', next_line)):
                            return f"{line.strip()} {next_line}"
                    
                    remaining = line[line.lower().find(indicator) + len(indicator):].strip()
                    if remaining:
                        return line.strip()
                    
                    return line.strip()

        for line in lines[:5]:
            clean_line = line.strip().strip('"\'')
            if (len(clean_line) > 2 and 
                re.search(r'[a-zA-ZčćžšđČĆŽŠĐ]', clean_line) and
                not re.search(r'\d{5,}', clean_line) and
                not re.search(r'\d{2}[.,/-]\d{2}[.,/-]\d{2,4}', clean_line)):
                return clean_line

        return "Unknown Store"

    def extract_datetime(self, lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
        found_date = None
        found_time = None

        for line in lines:
            fixed_line = line.lower().strip()

            for pattern in self.date_patterns:
                match = re.search(pattern, fixed_line)
                if match:
                    date_str = match.group(1)
                    date_str = re.sub(r'[,/-]', '.', date_str)
                    time_str = match.group(2) if len(match.groups()) > 1 else None

                    try:
                        if time_str:
                            dt = datetime.strptime(f"{date_str} {time_str}", "%d.%m.%Y %H:%M")
                            return dt.strftime("%Y-%m-%d"), time_str
                        else:
                            dt = datetime.strptime(date_str, "%d.%m.%Y")
                            found_date = dt.strftime("%Y-%m-%d")
                    except:
                        continue

            if not found_time:
                for pattern in self.time_patterns:
                    match = re.search(pattern, fixed_line)
                    if match:
                        found_time = match.group(1)

        return found_date, found_time

    def extract_total_contextual(self, lines: List[str]) -> Optional[float]:
        for i, line in enumerate(lines):
            fixed_line = self.fuzzy_fix_text(line)

            if any(keyword in fixed_line for keyword in ['ukupno', 'total']):
                for j in range(i, min(len(lines), i + 3)):
                    check_line = lines[j].strip()
                    
                    if re.match(r'^\d+[;,.]\d{2}$', check_line):
                        try:
                            price = float(re.sub(r'[;,]', '.', check_line))
                            if 0.1 <= price <= 1000:
                                return price
                        except:
                            continue
                    
                    match = re.search(r'(\d+[;,.]\d{2})', check_line)
                    if match:
                        try:
                            price = float(re.sub(r'[;,]', '.', match.group(1)))
                            if 0.1 <= price <= 1000:
                                return price
                        except:
                            continue
        return None

    def parse_items_contextual(self, lines: List[str]) -> List[Dict[str, Any]]:
        items = []
        
        items_start = -1
        items_end = len(lines)
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            if any(keyword in line_lower for keyword in ['naziv', 'cijena']):
                items_start = i + 1
                break
        
        if items_start == -1:
            for i, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in ['dib:', 'oib:']):
                    items_start = i + 1
                    break
            if items_start == -1:
                items_start = 5
        
        for i, line in enumerate(lines[items_start:], items_start):
            line_lower = line.lower().strip()
            if any(keyword in line_lower for keyword in ['ukupno', 'total', 'način plaćanja', 'račun']):
                items_end = i
                break
        
        i = items_start
        while i < items_end:
            line = lines[i].strip()
            
            # Skip headers and short lines
            if (not line or 
                len(line) < 2 or
                re.match(r'^[\d\s.,;=*-]+$', line) or
                line.lower().strip() in ['cijena', 'iznos', 'izns', 'naziv', 'ko)', 'kol', 'cijena ko)'] or
                any(skip in line.lower() for skip in ['pdv', 'porez', 'način', 'račun', 'konobar'])):
                i += 1
                continue
            
            # Check if this looks like an item name
            if (re.search(r'[a-zA-ZčćžšđČĆŽŠĐ]', line) and 
                not re.match(r'^\d+[;,.]\d{2}$', line)):
                
                item_name = line
                
                # Look for prices in next lines
                price_lines = []
                for j in range(i + 1, min(items_end, i + 5)):
                    if j < len(lines):
                        next_line = lines[j].strip()
                        if re.match(r'^\d+[;,.]\d{2}$', next_line):
                            try:
                                price = float(re.sub(r'[;,]', '.', next_line))
                                if 0.1 <= price <= 1000:
                                    price_lines.append(price)
                            except:
                                continue
                
                # Parse price information
                if len(price_lines) >= 3:
                    price_per_item = price_lines[0]
                    quantity = int(round(price_lines[1])) if price_lines[1] <= 10 else 1
                    total = price_lines[2]
                elif len(price_lines) >= 2:
                    price_per_item = price_lines[0]
                    total = price_lines[1]
                    quantity = int(round(total / price_per_item)) if price_per_item > 0 else 1
                elif len(price_lines) >= 1:
                    price_per_item = price_lines[0]
                    total = price_per_item
                    quantity = 1
                else:
                    i += 1
                    continue
                
                # Create item if valid
                if price_per_item and total and price_per_item > 0:
                    clean_name = re.sub(r'^[\d\s.,;-]+', '', item_name)
                    clean_name = re.sub(r'[\d\s.,;-]+$', '', clean_name)
                    clean_name = clean_name.strip()
                    
                    if len(clean_name) >= 2:
                        items.append({
                            "name": clean_name,
                            "quantity": quantity,
                            "price_per_item": price_per_item,
                            "total": total
                        })
                
                i += len(price_lines) + 1
            else:
                i += 1
        
        return items

    def parse_receipt(self, lines: List[str], log_debug: bool = False) -> Dict[str, Any]:
        if log_debug:
            logger.setLevel(logging.DEBUG)

        clean_lines = []
        for line in lines:
            if isinstance(line, str):
                clean_line = line.strip()
                if clean_line:
                    clean_lines.append(clean_line)

        if not clean_lines:
            return self._empty_receipt()

        try:
            store = self.detect_store_name_contextual(clean_lines)
            date, time = self.extract_datetime(clean_lines)
            items = self.parse_items_contextual(clean_lines)
            total = self.extract_total_contextual(clean_lines)

            if total is None and items:
                total = sum(item['total'] for item in items)

            result = {
                "store": store,
                "location_label": None,
                "date": date,
                "time": time,
                "total": total,
                "items": items,
                "confidence": self._calculate_confidence(store, date, items, total),
                "raw_lines_count": len(clean_lines),
                "parsed_items_count": len(items)
            }

            return result

        except Exception as e:
            logger.error(f"[PARSE] Error: {e}")
            return self._empty_receipt(error=str(e))

    def _calculate_confidence(self, store: str, date: Optional[str],
                              items: List[Dict], total: Optional[float]) -> float:
        confidence = 0.0

        if store and store != "Unknown Store":
            confidence += 0.3

        if date:
            confidence += 0.2

        if items:
            confidence += 0.3

        if total is not None:
            confidence += 0.2

        return min(confidence, 1.0)

    def _empty_receipt(self, error: str = None) -> Dict[str, Any]:
        return {
            "store": "Unknown Store",
            "location_label": None,
            "date": None,
            "time": None,
            "total": None,
            "items": [],
            "confidence": 0.0,
            "error": error,
            "raw_lines_count": 0,
            "parsed_items_count": 0
        }

def parse_receipt(lines: List[str], log_debug: bool = False) -> Dict[str, Any]:
    parser = ReceiptParser()
    return parser.parse_receipt(lines, log_debug)