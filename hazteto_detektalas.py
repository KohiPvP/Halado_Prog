import cv2
import numpy as np
import pytesseract
import re

IMAGE_PATH = "map.png"   # <-- ide írd a saját fájlneved


# 1. KÉP BEOLVASÁSA
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Nem sikerült beolvasni a képet: {IMAGE_PATH}")

h, w = img.shape[:2]
print(f"Kép mérete: {w}x{h}")


# 2. SKÁLA FELIRAT + CSÍK DETEKTÁLÁSA
roi = img[int(h * 0.78):h, int(w * 0.55):w]
roi_h, roi_w = roi.shape[:2]
print(f"Skála ROI mérete: {roi_w}x{roi_h}")

gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
gray_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)

gray_big = cv2.resize(gray_roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

_, thresh = cv2.threshold(gray_big, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

config = "--psm 7 -c tessedit_char_whitelist=0123456789m,k "
text = pytesseract.image_to_string(thresh, config=config)
print("OCR nyers szöveg:", repr(text))

match = re.search(r"(\d+(?:[\.,]\d+)?)\s*(m|km)", text.lower())

scale_m = None
if match:
    value = float(match.group(1).replace(",", "."))
    unit = match.group(2)
    if unit == "km":
        scale_m = value * 1000.0
    else:
        scale_m = value

if scale_m is None or scale_m < 10:
    print("⚠️ OCR gyanús értéket adott, kézzel felülírjuk 50 m-re.")
    scale_m = 50.0

print("Skála valós hossza (m):", scale_m)

edges = cv2.Canny(gray_roi, 50, 150)

lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=30,
    minLineLength=40,
    maxLineGap=10
)

if lines is None:
    raise RuntimeError("Nem találtam skálacsíkot vonaldetektálással.")

best_len = 0
best_line = None
for l in lines:
    x1, y1, x2, y2 = l[0]
    if abs(y2 - y1) < 4:
        length = abs(x2 - x1)
        if length > best_len:
            best_len = length
            best_line = (x1, y1, x2, y2)

if best_line is not None:
    x1, y1, x2, y2 = best_line
    cv2.line(roi, (x1, y1), (x2, y2), (0, 0, 255), 1)

scale_px = best_len
if scale_px == 0:
    raise RuntimeError("A detektált skálacsík hossza 0 pixel.")

print("Skálacsík hossza pixelekben:", scale_px)

meters_per_pixel = scale_m / scale_px
print(f"1 pixel ≈ {meters_per_pixel:.4f} m")

cv2.imwrite("debug_scale_roi.png", roi)
print("Skála ROI mentve: debug_scale_roi.png")


# 3. HÁZAK DETEKTÁLÁSA (háttérszínhez képest)
img_bgr = img.copy()
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

flat = img_rgb.reshape(-1, 3)

q = (flat // 4) * 4
uniq, counts = np.unique(q, axis=0, return_counts=True)
bg_color = uniq[np.argmax(counts)].astype(np.int16)

print("Becsült háttérszín (RGB):", bg_color)

diff = img_rgb.astype(np.int16) - bg_color
dist = np.linalg.norm(diff, axis=2)

gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

LIGHT_MIN = 230  
DIST_MIN = 6  

light_mask = (gray_full >= LIGHT_MIN).astype(np.uint8) * 255
dist_mask = (dist >= DIST_MIN).astype(np.uint8) * 255

mask = cv2.bitwise_and(light_mask, dist_mask)

kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

cv2.imwrite("debug_house_mask.png", mask)
print("Ház-maszk mentve: debug_house_mask.png")

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

out = img_rgb.copy()
min_area_px = 200   

print("\nDetektált házak:")

house_idx = 1
for cnt in contours:
    area_px = cv2.contourArea(cnt)
    if area_px < min_area_px:
        continue

    area_m2 = area_px * (meters_per_pixel ** 2)

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    print(f"Ház #{house_idx}: {area_px:.1f} px²  ≈  {area_m2:.1f} m²")
    house_idx += 1

    cv2.drawContours(out, [cnt], -1, (255, 0, 0), 1)
    cv2.putText(out, f"{int(area_m2)} m2", (cx - 20, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

# 4. EREDMÉNY MENTÉSE
out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
cv2.imwrite("hazak_terulettel.png", out_bgr)

print("\nKész: hazak_terulettel.png (házak kontúrral és m² felirattal)")
