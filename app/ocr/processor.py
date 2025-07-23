import cv2
import numpy as np
import os

# Directory to save debug images for troubleshooting
DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

# -------------------- DEBUG --------------------

def save_debug_image(image, name):
    """
    Save an image with the given name in the debug directory.
    Helps visually inspect preprocessing steps.
    """
    path = os.path.join(DEBUG_DIR, f"{name}.png")
    cv2.imwrite(path, image)
    print(f"[DEBUG] Saved {path}")

# -------------------- PREPROCESSING --------------------

def preprocess_for_ocr(img):
    """
    Full OCR preprocessing pipeline:
    - Find receipt contour
    - Crop it
    - Enhance text for better OCR
    Returns: (preprocessed image, status label)
    """
    contour = find_receipt_contour(img)

    if contour is None:
        print("[WARN] Could not find receipt contour.")
        return None, "no_contour"

    cropped = crop_by_corners(img, contour)

    if cropped is None:
        print("[WARN] Cropping failed.")
        return None, "crop_failed"

    enhanced = enhance_text(cropped)

    if enhanced is None:
        print("[WARN] Text enhancement failed.")
        return None, "enhance_failed"

    return enhanced, "ok"

# -------------------- EDGE DETECTION --------------------

def sharpen_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    dil = cv2.dilate(blur, kern, iterations=2)
    closed = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, kern, iterations=2)
    edged = cv2.Canny(closed, 30, 100)
    return edged

def binarize(edged):
    mean = np.mean(edged)
    _, binary = cv2.threshold(edged, mean, 255, cv2.THRESH_BINARY)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(binary, kern, iterations=2)
    return dilated

# -------------------- CONTOUR + CROP --------------------

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def draw_points(img, pts, color=(0, 0, 255)):
    for i, (x, y) in enumerate(pts):
        cv2.circle(img, (int(x), int(y)), 15, color, -1)
        cv2.putText(img, f"P{i}", (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return img

def crop_by_corners(img, pts, padding=10):
    rect = order_points(pts)
    img_dbg = img.copy()
    draw_points(img_dbg, rect)
    save_debug_image(img_dbg, "debug_corners")

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    pad_w = min(padding, maxWidth // 4)
    pad_h = min(padding, maxHeight // 4)

    dst = np.array([
        [pad_w, pad_h],
        [maxWidth - 1 - pad_w, pad_h],
        [maxWidth - 1 - pad_w, maxHeight - 1 - pad_h],
        [pad_w, maxHeight - 1 - pad_h]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

# -------------------- TEXT ENHANCEMENT --------------------

def enhance_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 10
    )
    return binary

# -------------------- RECEIPT CONTOUR --------------------

def find_receipt_contour(img):
    edged = sharpen_edge(img)
    binary = binarize(edged)

    save_debug_image(edged, "edged")
    save_debug_image(binary, "binarized")

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    receipt_contour = None
    image_area = img.shape[0] * img.shape[1]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            if (0.2 < aspect_ratio < 1.5 and image_area * 0.2 < area < image_area * 0.95):
                if area > max_area:
                    max_area = area
                    receipt_contour = approx

    if receipt_contour is not None:
        return receipt_contour.reshape(4, 2)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        return np.array(box, dtype="float32")

    return None