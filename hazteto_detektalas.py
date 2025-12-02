import cv2
import numpy as np
import pytesseract
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional


IMAGE_PATH = "map.png" 



@dataclass
class ScaleInfo:
    """A térképskála adatai."""
    scale_m: float            # skála valós hossza (m)
    scale_px: int             # skálacsík hossza pixelben
    meters_per_pixel: float   # 1 pixel hány méter


@dataclass
class HouseMeasurement:
    """Egy detektált ház adatai."""
    index: int
    area_px: float
    area_m2: float
    centroid: Tuple[int, int]


# ---------- KÉPKEZELŐ OSZTÁLY ----------

class MapImage:
    """Kép beolvasása, alap infók, ROI kivágás stb."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.img_bgr: Optional[np.ndarray] = None
        self.height: int = 0
        self.width: int = 0

    def load(self) -> None:
        self.img_bgr = cv2.imread(self.path)
        if self.img_bgr is None:
            raise FileNotFoundError(f"Nem sikerült beolvasni a képet: {self.path}")
        self.height, self.width = self.img_bgr.shape[:2]
        print(f"Kép mérete: {self.width}x{self.height}")

    def get_rgb(self) -> np.ndarray:
        if self.img_bgr is None:
            raise RuntimeError("A kép még nincs beolvasva.")
        return cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)

    def get_gray(self) -> np.ndarray:
        if self.img_bgr is None:
            raise RuntimeError("A kép még nincs beolvasva.")
        return cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)

    def get_roi(
        self,
        y_start_ratio: float,
        y_end_ratio: float,
        x_start_ratio: float,
        x_end_ratio: float,
    ) -> np.ndarray:
        """Egyszerű arányos ROI kivágás (0–1 tartomány)."""
        if self.img_bgr is None:
            raise RuntimeError("A kép még nincs beolvasva.")
        h, w = self.height, self.width
        y1 = int(h * y_start_ratio)
        y2 = int(h * y_end_ratio)
        x1 = int(w * x_start_ratio)
        x2 = int(w * x_end_ratio)
        roi = self.img_bgr[y1:y2, x1:x2]
        print(f"Skála ROI mérete: {roi.shape[1]}x{roi.shape[0]}")
        return roi


# ---------- SKÁLA DETEKTÁLÓ OSZTÁLY ----------

class ScaleDetector:
    """
    Skálafelirat OCR-rel, skálacsík HoughLinesP-vel.
    ROI-t arányokkal adjuk meg (alapértelmezés: eredeti kód).
    """

    def __init__(
        self,
        y_start_ratio: float = 0.85,
        y_end_ratio: float = 1.0,
        x_start_ratio: float = 0.70,
        x_end_ratio: float = 0.95,
    ) -> None:
        self.y_start_ratio = y_start_ratio
        self.y_end_ratio = y_end_ratio
        self.x_start_ratio = x_start_ratio
        self.x_end_ratio = x_end_ratio

    def detect(self, map_img: MapImage) -> ScaleInfo:
        """Fő belépési pont: skála detektálása és m/pixel számítás."""
        roi = map_img.get_roi(
            self.y_start_ratio,
            self.y_end_ratio,
            self.x_start_ratio,
            self.x_end_ratio,
        )

        scale_m = self._detect_scale_text(roi)
        scale_px = self._detect_scale_bar_length(roi)

        if scale_px == 0:
            raise RuntimeError("A detektált skálacsík hossza 0 pixel.")

        meters_per_pixel = scale_m / scale_px
        print("Skála valós hossza (m):", scale_m)
        print("Skálacsík hossza pixelekben:", scale_px)
        print(f"1 pixel ≈ {meters_per_pixel:.4f} m")

        # debug ROI mentése
        #cv2.imwrite("debug_scale_roi.png", roi)
        #print("Skála ROI mentve: debug_scale_roi.png")

        return ScaleInfo(
            scale_m=scale_m,
            scale_px=scale_px,
            meters_per_pixel=meters_per_pixel,
        )

    def _detect_scale_text(self, roi_bgr: np.ndarray) -> float:
        """OCR-rel kiolvassa a skála értékét (m vagy km)."""
        # 1) Eredeti ROI mentése (BGR)
        #cv2.imwrite("debug_scale_roi_original.png", roi_bgr)

        gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)

        # 2) Szürke, elmosott ROI mentése
        #cv2.imwrite("debug_scale_gray_blur.png", gray_roi)

        gray_big = cv2.resize(
            gray_roi,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC
        )

        # 3) Felnagyított szürke ROI mentése
        #cv2.imwrite("debug_scale_gray_big.png", gray_big)

        _, thresh = cv2.threshold(
            gray_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 4) Thresholdolt kép mentése – EZT látja a Tesseract
        cv2.imwrite("debug_scale_thresh_for_ocr.png", thresh)

        config = "--psm 7 -c tessedit_char_whitelist=0123456789m,k "
        text = pytesseract.image_to_string(thresh, config=config)
        print("OCR nyers szöveg:", repr(text))

        match = re.search(r"(\d+(?:[\.,]\d+)?)\s*(m|km)", text.lower())

        scale_m: Optional[float] = None
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

        return scale_m


    def _detect_scale_bar_length(self, roi_bgr: np.ndarray) -> int:
        """HoughLinesP segítségével megkeresi a vízszintes skálacsíkot."""
        gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=40,
            maxLineGap=10,
        )

        if lines is None:
            raise RuntimeError("Nem találtam skálacsíkot vonaldetektálással.")

        best_len = 0
        best_line = None

        for l in lines:
            x1, y1, x2, y2 = l[0]
            # közel vízszintes vonal
            if abs(y2 - y1) < 4:
                length = abs(x2 - x1)
                if length > best_len:
                    best_len = length
                    best_line = (x1, y1, x2, y2)

        if best_line is not None:
            x1, y1, x2, y2 = best_line
            cv2.line(roi_bgr, (x1, y1), (x2, y2), (0, 0, 255), 1)

        return best_len


# ---------- HÁZ DETEKTÁLÓ OSZTÁLY ----------

class HouseDetector:
    """
    Háttérszín-alapú házdetektálás, terület m²-ben kiírása.
    """

    def __init__(
        self,
        light_min: int = 230,
        dist_min: int = 6,
        min_area_px: int = 200,
    ) -> None:
        self.LIGHT_MIN = light_min
        self.DIST_MIN = dist_min
        self.min_area_px = min_area_px

    def detect_houses(self, map_img: MapImage, meters_per_pixel: float) -> Tuple[np.ndarray, List[HouseMeasurement]]:
        """
        Házak detektálása és annotálása.
        Visszaadja az annotált RGB képet és a mérések listáját.
        """
        img_bgr = map_img.img_bgr
        if img_bgr is None:
            raise RuntimeError("A kép még nincs beolvasva.")

        img_rgb = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)
        gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # DEBUG: bemenetek mentése
        #cv2.imwrite("HouseDetector_input_bgr.png", img_bgr)
        #cv2.imwrite("HouseDetector_input_rgb.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        #cv2.imwrite("HouseDetector_gray_full.png", gray_full)

        bg_color = self._estimate_background_color(img_rgb)
        print("Becsült háttérszín (RGB):", bg_color)

        mask = self._create_house_mask(img_rgb, gray_full, bg_color)

        # debug maszk
        cv2.imwrite("HouseDetector_mask_morph.png", mask)
        print("Ház-maszk mentve: HouseDetector_mask_morph.png")

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        measurements: List[HouseMeasurement] = []
        out = img_rgb.copy()

        print("\nDetektált házak:")

        house_idx = 1
        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < self.min_area_px:
                continue

            area_m2 = area_px * (meters_per_pixel ** 2)

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            print(f"Ház #{house_idx}: {area_px:.1f} px²  ≈  {area_m2:.1f} m²")

            measurements.append(
                HouseMeasurement(
                    index=house_idx,
                    area_px=area_px,
                    area_m2=area_m2,
                    centroid=(cx, cy),
                )
            )

            cv2.drawContours(out, [cnt], -1, (255, 0, 0), 1)
            cv2.putText(
                out,
                f"{int(area_m2)} m2",
                (cx - 20, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
            )

            house_idx += 1

        # DEBUG: kontúros / annotált kép mentése
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        #cv2.imwrite("HouseDetector_contours_annotated.png", out_bgr)

        return out, measurements


    def _estimate_background_color(self, img_rgb: np.ndarray) -> np.ndarray:
        """Kép leggyakoribb színe (kvantálva) mint háttérszín."""
        flat = img_rgb.reshape(-1, 3)
        q = (flat // 4) * 4
        uniq, counts = np.unique(q, axis=0, return_counts=True)
        bg_color = uniq[np.argmax(counts)].astype(np.int16)
        return bg_color

    def _create_house_mask(
        self,
        img_rgb: np.ndarray,
        gray_full: np.ndarray,
        bg_color: np.ndarray,
    ) -> np.ndarray:
        """Maszk generálása fényesség + háttérszín távolság alapján."""
        diff = img_rgb.astype(np.int16) - bg_color
        dist = np.linalg.norm(diff, axis=2)

        light_mask = (gray_full >= self.LIGHT_MIN).astype(np.uint8) * 255
        dist_mask = (dist >= self.DIST_MIN).astype(np.uint8) * 255

        mask = cv2.bitwise_and(light_mask, dist_mask)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return mask


# ---------- FŐ FELDOLGOZÓ OSZTÁLY ----------

class MapAnalyzer:
    """Magas szintű vezérlő, ami összefűzi a lépéseket."""

    def __init__(self, image_path: str) -> None:
        self.map_img = MapImage(image_path)
        self.scale_detector = ScaleDetector()
        self.house_detector = HouseDetector()

    def run(self) -> None:
        # 1. Kép beolvasása
        self.map_img.load()

        # 2. Skála detektálása
        scale_info = self.scale_detector.detect(self.map_img)

        # 3. Házak detektálása
        out_rgb, _ = self.house_detector.detect_houses(
            self.map_img, scale_info.meters_per_pixel
        )

        # 4. Eredmény mentése
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite("hazak_terulettel.png", out_bgr)
        print("\nKész: hazak_terulettel.png (házak kontúrral és m² felirattal)")


# ---------- FUTTATÁS ----------

if __name__ == "__main__":
    analyzer = MapAnalyzer(IMAGE_PATH)
    analyzer.run()
