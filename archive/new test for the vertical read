from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import pytesseract
import os

app = FastAPI()

DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

def save_debug_image(image, name):
    path = os.path.join(DEBUG_DIR, f"{name}.png")
    cv2.imwrite(path, image)
    print(f"[DEBUG] Saved {path}")

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

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def draw_points(img, pts, color=(0, 0, 255)):
    for i, (x, y) in enumerate(pts):
        cv2.circle(img, (int(x), int(y)), 15, color, -1)
        cv2.putText(img, f"P{i}", (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
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

    # Ensure padding doesn't invert coords
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

def enhance_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 10
    )
    return binary

@app.post("/ocr/")
async def ocr_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Failed to load image"}

    save_debug_image(img, "original")

    edged = sharpen_edge(img)
    save_debug_image(edged, "edged")

    bin_img = binarize(edged)
    save_debug_image(bin_img, "binarized")

    cnts = cv2.findContours(bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if isinstance(cnts, tuple) else cnts
    if not cnts:
        return {"error": "No contours found"}

    # Find largest contour by area
    largest = max(cnts, key=cv2.contourArea)

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    # Accept 3 or 4 points, else fallback
    if len(approx) < 4:
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="float32")
    elif len(approx) == 4:
        box = approx.reshape(4, 2).astype("float32")
    else:
        hull = cv2.convexHull(largest)
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="float32")

    # Draw contour box
    img_box = img.copy()
    cv2.drawContours(img_box, [box.astype(int)], -1, (0, 255, 0), 5)
    save_debug_image(img_box, "contour_box")

    cropped = crop_by_corners(img, box, padding=10)
    save_debug_image(cropped, "cropped")

    enhanced = enhance_text(cropped)
    save_debug_image(enhanced, "enhanced")

    txt = pytesseract.image_to_string(enhanced, lang='hrv', config='--psm 3 --oem 3')
    lines = [line.strip() for line in txt.strip().split('\n') if line.strip()]

    print(f"[DEBUG] OCR extracted lines:\n{lines}")

    return {
        "lines": lines,
        "debug_folder": os.path.abspath(DEBUG_DIR)
    }
