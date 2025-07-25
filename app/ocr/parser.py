import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import difflib
import logging

logger = logging.getLogger(__name__)

class ReceiptParser:
    def __init__(self):
        # Expanded OCR corrections with fuzzy matching
        self.ocr_fixes = {
            # Croatian specific
            'ukupko': 'ukupno',
            'ukupko (eur);': 'ukupno',
            'ukupko (eur)': 'ukupno',
            'ukupko:': 'ukupno',
            'suma': 'ukupno',
            'iznos': 'ukupno',
            'svota': 'ukupno',

            # English/International
            'totai': 'total',
            'totál': 'total',
            'totsl': 'total',
            'totak': 'total',
            'sub total': 'subtotal',
            'sub-total': 'subtotal',

            # Date related
            'račun broj': 'datum',
            'račun': 'datum',
            'u račun': 'datum',
            'datun': 'datum',
            'datur': 'datum',

            # Common OCR errors
            'gg': '',
            'g9': '',
            'bar,': 'lubar',
            'lu bar': 'lubar',
            '"lubar"': 'lubar',
            '|': 'l',
            '0': 'o',  # Context dependent
            '5': 's',  # Context dependent
        }

        # Known store patterns for better recognition
        self.store_patterns = {
            'lubar': ['lubar', 'lu bar', 'caffe bar', 'bar lu'],
            'pekara': ['pekara', 'bakery'],
            'restoran': ['restoran', 'restaurant'],
            'market': ['market', 'trgovina', 'shop'],
            'benzinska': ['benzinska', 'petrol', 'gas station', 'ina', 'omv'],
        }

        # Date patterns for different locales
        self.date_patterns = [
            r'datum\s*[:\-]?\s*(\d{1,2}[.,/-]\d{1,2}[.,/-]\d{2,4})',
            r'date\s*[:\-]?\s*(\d{1,2}[.,/-]\d{1,2}[.,/-]\d{2,4})',
            r'(\d{1,2}[.,/-]\d{1,2}[.,/-]\d{4})',
            r'(\d{4}[.,/-]\d{1,2}[.,/-]\d{1,2})',
        ]

        # Time patterns
        self.time_patterns = [
            r'(\d{1,2}:\d{2}(?::\d{2})?)',
            r'vrijeme\s*[:\-]?\s*(\d{1,2}:\d{2})',
            r'time\s*[:\-]?\s*(\d{1,2}:\d{2})',
        ]

        # Price patterns
        self.price_patterns = [
            r'(\d+[,.]\d{2})',  # Basic price
            r'(\d+\.\d{2})',    # Decimal point
            r'(\d+,\d{2})',     # Comma decimal
            r'(\d+(?:[.,]\d{2})?)\s*(?:kn|eur|€|\$)',  # With currency
        ]

        # Skip patterns for lines that shouldn't be parsed as items
        self.skip_patterns = [
            r'ukupno|total|suma|subtotal',
            r'datum|date|vrijeme|time',
            r'račun|bill|receipt|bon',
            r'pdv|tax|vat|porez',
            r'blagajna|cashier|cash',
            r'hvala|thank|grazie',
            r'dobrodošli|welcome',
            r'^\s*[-=*]+\s*$',  # Separator lines
            r'^\s*\d+\s*$',     # Just numbers
        ]

    def fuzzy_fix_text(self, text: str, threshold: float = 0.8) -> str:
        """Apply OCR fixes with fuzzy matching for better correction"""
        fixed_text = text.lower()

        # Apply direct fixes first
        for wrong, right in self.ocr_fixes.items():
            fixed_text = fixed_text.replace(wrong, right)

        # Fuzzy matching for common words
        words = fixed_text.split()
        fixed_words = []

        for word in words:
            best_match = word
            best_ratio = 0

            # Check against known corrections
            for wrong, right in self.ocr_fixes.items():
                if len(wrong) > 2:  # Only fuzzy match longer words
                    ratio = difflib.SequenceMatcher(None, word, wrong).ratio()
                    if ratio > threshold and ratio > best_ratio:
                        best_match = right
                        best_ratio = ratio

            fixed_words.append(best_match)

        return ' '.join(fixed_words)

    def detect_store_name(self, lines: List[str]) -> str:
        """Enhanced store name detection with fuzzy matching"""
        candidates = []

        # Check first 10 lines for store name
        for i, line in enumerate(lines[:10]):
            original_line = line.strip().strip('"\'')
            fixed_line = self.fuzzy_fix_text(line)

            # Skip if line looks like date, price, or receipt number
            if any(pattern in fixed_line for pattern in ['datum', 'ukupno', 'total', 'račun']):
                continue

            if re.search(r'\d{2}[.,/-]\d{2}[.,/-]\d{2,4}', line):  # Date pattern
                continue

            if re.search(r'\d+[.,]\d{2}', line):  # Price pattern
                continue

            # Check against known store patterns
            for store_type, patterns in self.store_patterns.items():
                for pattern in patterns:
                    if pattern in fixed_line:
                        score = len(pattern) / len(fixed_line) if fixed_line else 0
                        candidates.append((original_line, score, i))
                        break

            # Generic store name criteria
            if len(original_line) > 3 and original_line.replace(' ', '').isalpha():
                # Prefer lines near the top
                score = 1.0 / (i + 1)
                candidates.append((original_line, score, i))

        if candidates:
            # Sort by score and position
            candidates.sort(key=lambda x: (x[1], -x[2]), reverse=True)
            return candidates[0][0]

        # Fallback: first clean non-numeric line
        for line in lines[:5]:
            clean_line = line.strip().strip('"\'')
            if len(clean_line) > 2 and not any(char.isdigit() for char in clean_line):
                return clean_line

        return "Unknown Store"

    def extract_datetime(self, lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """Enhanced date and time extraction with multiple patterns"""
        found_date = None
        found_time = None

        for line in lines:
            fixed_line = self.fuzzy_fix_text(line)

            # Try date patterns
            if not found_date:
                for pattern in self.date_patterns:
                    match = re.search(pattern, fixed_line, re.IGNORECASE)
                    if match:
                        try:
                            date_str = match.group(1)
                            # Normalize separators
                            date_str = re.sub(r'[.,/-]', '.', date_str)

                            # Try different date formats
                            for fmt in ['%d.%m.%Y', '%d.%m.%y', '%Y.%m.%d']:
                                try:
                                    parsed_date = datetime.strptime(date_str, fmt)
                                    found_date = parsed_date.strftime("%d-%m-%Y")
                                    break
                                except ValueError:
                                    continue

                            if not found_date:
                                found_date = date_str  # Keep original if parsing fails
                            break
                        except Exception as e:
                            logger.warning(f"Date parsing error: {e}")

            # Try time patterns
            if not found_time:
                for pattern in self.time_patterns:
                    match = re.search(pattern, fixed_line, re.IGNORECASE)
                    if match:
                        found_time = match.group(1)
                        break

            if found_date and found_time:
                break

        return found_date, found_time

    def extract_total(self, lines: List[str]) -> Optional[float]:
        """Extract total amount from receipt"""
        total_candidates = []

        for i, line in enumerate(lines):
            fixed_line = self.fuzzy_fix_text(line)

            # Look for total keywords
            if any(keyword in fixed_line for keyword in ['ukupno', 'total', 'suma']):
                # Extract prices from this line
                prices = re.findall(r'(\d+[,.]\d{2})', line)
                for price_str in prices:
                    try:
                        price = float(price_str.replace(',', '.'))
                        # Prefer prices that appear later in the receipt
                        score = i / len(lines) + (price / 1000)  # Boost score for larger amounts
                        total_candidates.append((price, score))
                    except ValueError:
                        continue

        if total_candidates:
            # Return the total with highest score
            total_candidates.sort(key=lambda x: x[1], reverse=True)
            return total_candidates[0][0]

        return None

    def parse_item_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse individual item line with multiple strategies"""
        original_line = line.strip()
        fixed_line = self.fuzzy_fix_text(line)

        # Skip lines that match skip patterns
        for pattern in self.skip_patterns:
            if re.search(pattern, fixed_line, re.IGNORECASE):
                return None

        # Skip lines that are too short or contain only numbers/symbols
        if len(original_line.strip()) < 3:
            return None

        if re.match(r'^[\d\s.,\-=*]+$', original_line):
            return None

        # Strategy 1: Perfect item format (name quantity price total)
        # Define regex patterns for item lines.  The final pattern only applies
        # when there is a single decimal number in the line.  To prevent
        # misinterpreting lines with multiple numbers (e.g. "Name 3,80. 1,00"),
        # we use a negative lookahead to ensure there is not more than one
        # price-like number in the string.
        patterns = [
            r'^(.+?)\s+(\d+)[xX×]\s*(\d+[,.]\d{2})\s+(\d+[,.]\d{2})$',   # Name 2x 1.50 3.00
            r'^(.+?)\s+(\d+[,.]\d{2})\s+(\d+[,.]\d{2})$',                 # Name 1.50 3.00
            r'^(.+?)\s+(\d+)\s*[xX×]\s*(\d+[,.]\d{2})$',                  # Name 2x 1.50
            r'^(?!.*\d+[,.]\d{2}.*\d+[,.]\d{2})(.+?)\s+(\d+[,.]\d{2})$',  # Name 1.50 when only one price present
        ]

        for pattern in patterns:
            match = re.match(pattern, original_line)
            if match:
                try:
                    groups = match.groups()
                    name = groups[0].strip()

                    if len(groups) == 4:  # name qty price total
                        quantity = int(groups[1])
                        price_per_item = float(groups[2].replace(',', '.'))
                        total = float(groups[3].replace(',', '.'))
                    elif len(groups) == 3:
                        if 'x' in groups[1].lower() or '×' in groups[1]:  # name qty price
                            # e.g. "Name 2x 1,50" or "Name 2× 1,50"
                            try:
                                quantity = int(re.findall(r'\d+', groups[1])[0])
                            except Exception:
                                quantity = 1
                            price_per_item = float(groups[2].replace(',', '.'))
                            total = quantity * price_per_item
                        else:  # name price total (two numbers separated by space)
                            # Convert both numeric strings to floats
                            v1 = float(groups[1].replace(',', '.'))
                            v2 = float(groups[2].replace(',', '.'))
                            # If first value is larger than second, it may be price and second may be quantity
                            if v1 > v2:
                                # If the second value is close to 1, treat it as quantity
                                if v2 <= 1.1:
                                    quantity = max(1, int(round(v2)))
                                    price_per_item = v1
                                    total = price_per_item * quantity
                                else:
                                    # Otherwise assume second is price and first is total
                                    price_per_item = v2
                                    total = v1
                                    quantity = max(1, round(total / price_per_item)) if price_per_item > 0 else 1
                            else:
                                # Normal case: v1 is price, v2 is total
                                price_per_item = v1
                                total = v2
                                quantity = max(1, round(total / price_per_item)) if price_per_item > 0 else 1
                    else:  # name price
                        quantity = 1
                        price_per_item = float(groups[1].replace(',', '.'))
                        total = price_per_item

                    return {
                        "name": name,
                        "quantity": quantity,
                        "price_per_item": price_per_item,
                        "total": total
                    }
                except (ValueError, IndexError) as e:
                    logger.debug(f"Pattern match failed for '{original_line}': {e}")
                    continue

        # Strategy 2: Flexible price extraction
        prices = re.findall(r'(\d+[,.]\d{2})', original_line)
        if len(prices) >= 1:
            try:
                # Extract name (everything before first price occurrence)
                name_part = original_line.split(prices[0])[0].strip()
                if len(name_part) < 2:
                    return None

                # Clean up name by stripping leading/trailing numbers and symbols
                name = re.sub(r'^\d+\.?\s*', '', name_part)
                name = name.strip()
                if len(name) < 2:
                    return None

                # Convert price strings to floats
                price_values = [float(p.replace(',', '.')) for p in prices]

                quantity = 1
                price_per_item = price_values[0]
                total = price_values[-1]

                if len(price_values) >= 2:
                    # If one of the values equals 1 (quantity), treat it accordingly
                    # e.g. "3,80 1,00" -> quantity=1, price=3.80, total=3.80
                    # Otherwise, if the ratio of larger to smaller is near an integer,
                    # infer quantity from ratio.
                    sorted_vals = sorted(price_values)
                    smallest, largest = sorted_vals[0], sorted_vals[-1]
                    # Check for quantity equal to 1.00
                    if any(abs(v - 1.0) < 0.05 for v in price_values):
                        quantity = 1
                        # pick the value that is not ~1.00 as price per item and total
                        other_vals = [v for v in price_values if abs(v - 1.0) >= 0.05]
                        if other_vals:
                            price_per_item = other_vals[0]
                            total = price_per_item * quantity
                    else:
                        ratio = largest / smallest if smallest > 0 else 1
                        # If ratio is near an integer >1, interpret as quantity
                        if ratio > 1.2 and abs(ratio - round(ratio)) < 0.3:
                            quantity = int(round(ratio))
                            price_per_item = smallest
                            total = largest
                        else:
                            # Fallback: assume first value is price and last is total
                            price_per_item = price_values[0]
                            total = price_values[-1]
                            quantity = max(1, round(total / price_per_item)) if price_per_item > 0 else 1
                else:
                    # Only one numeric value found
                    price_per_item = price_values[0]
                    total = price_per_item
                    quantity = 1

                return {
                    "name": name,
                    "quantity": quantity,
                    "price_per_item": price_per_item,
                    "total": total,
                }
            except (ValueError, ZeroDivisionError) as e:
                logger.debug(f"Price extraction failed for '{original_line}': {e}")

        return None

    def parse_items(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Parse all items from receipt lines"""
        items = []

        # Skip header lines (first few lines usually contain store info)
        start_idx = 0
        for i, line in enumerate(lines[:10]):
            fixed_line = self.fuzzy_fix_text(line)
            if any(keyword in fixed_line for keyword in ['datum', 'date', 'time', 'račun']):
                start_idx = i + 1
                break

        # Parse items from middle section
        for i, line in enumerate(lines[start_idx:], start_idx):
            # Stop parsing when we hit total/summary section
            fixed_line = self.fuzzy_fix_text(line)
            if any(keyword in fixed_line for keyword in ['ukupno', 'total', 'suma', 'subtotal']):
                break

            item = self.parse_item_line(line)
            if item:
                items.append(item)

        # Post-process items
        return self._post_process_items(items)

    def _post_process_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean up and validate parsed items"""
        cleaned_items = []

        for item in items:
            # Validate name
            if len(item['name']) < 2:
                continue

            # Remove common OCR artifacts from names
            name = item['name']
            name = re.sub(r'^[\d\s.,\-]+', '', name)  # Remove leading numbers/symbols
            name = re.sub(r'[\d\s.,\-]+$', '', name)  # Remove trailing numbers/symbols
            name = name.strip()

            if len(name) < 2:
                continue

            # Validate prices
            if item['price_per_item'] <= 0 or item['total'] <= 0:
                continue

            if item['quantity'] <= 0:
                item['quantity'] = 1

            # Reasonable price check (adjust based on your currency)
            if item['total'] > 10000:  # Suspiciously high
                continue

            item['name'] = name
            cleaned_items.append(item)

        return cleaned_items

    def parse_receipt(self, lines: List[str], log_debug: bool = False) -> Dict[str, Any]:
        """
        Main parsing function with enhanced error handling and multiple strategies
        """
        if log_debug:
            logger.setLevel(logging.DEBUG)

        logger.debug(f"[PARSE] Processing {len(lines)} lines")

        # Clean input lines
        clean_lines = []
        for line in lines:
            if isinstance(line, str):
                clean_line = line.strip()
                if clean_line:  # Skip empty lines
                    clean_lines.append(clean_line)

        if not clean_lines:
            logger.warning("[PARSE] No valid lines to process")
            return self._empty_receipt()

        try:
            # Extract components
            store = self.detect_store_name(clean_lines)
            date, time = self.extract_datetime(clean_lines)
            items = self.parse_items(clean_lines)
            total = self.extract_total(clean_lines)

            # Calculate total from items if not found
            if total is None and items:
                total = sum(item['total'] for item in items)

            result = {
                "store": store,
                "location_label": None,  # Could be enhanced later
                "date": date,
                "time": time,
                "total": total,
                "items": items,
                "confidence": self._calculate_confidence(store, date, items, total),
                "raw_lines_count": len(clean_lines),
                "parsed_items_count": len(items)
            }

            logger.info(f"[PARSE] Success: {len(items)} items, total: {total}")
            return result

        except Exception as e:
            logger.error(f"[PARSE] Error: {e}")
            return self._empty_receipt(error=str(e))

    def _calculate_confidence(self, store: str, date: Optional[str],
                              items: List[Dict], total: Optional[float]) -> float:
        """Calculate parsing confidence score"""
        confidence = 0.0

        # Store name confidence
        if store and store != "Unknown Store":
            confidence += 0.2
            if any(pattern in store.lower() for patterns in self.store_patterns.values() for pattern in patterns):
                confidence += 0.1

        # Date confidence
        if date:
            confidence += 0.2
            if re.match(r'\d{2}-\d{2}-\d{4}', date):
                confidence += 0.1

        # Items confidence
        if items:
            confidence += 0.3
            if len(items) > 1:
                confidence += 0.1

        # Total confidence
        if total is not None:
            confidence += 0.2

        return min(confidence, 1.0)

    def _empty_receipt(self, error: str = None) -> Dict[str, Any]:
        """Return empty receipt structure"""
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

# Convenience function for backward compatibility
def parse_receipt(lines: List[str], log_debug: bool = False) -> Dict[str, Any]:
    """Backward compatible parsing function"""
    parser = ReceiptParser()
    return parser.parse_receipt(lines, log_debug)
