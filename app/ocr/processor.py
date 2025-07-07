import cv2
import numpy as np
import os

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
