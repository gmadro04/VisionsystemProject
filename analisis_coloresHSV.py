# ============================================================
#  analisis_coloresHSV.py
#  Segmentación de blobs de color (HSV) sobre imágenes undist, sin distorsión
#  - Lee todas las imágenes de UNDIST_DIR
#  - Segmenta varios colores definidos en una paleta (HSV)
#  - Limpia, agrupa y saca blobs (bboxes, centroides, áreas)
#  - Guarda overlays y (opcional) máscaras por color
#  - Exporta un CSV con los resultados
#  Requisitos: opencv-python, pyyaml (opcional), pandas (opcional)
#  Ejecuta:   python analisis_colores_hsv.py
# ============================================================

from pathlib import Path
import os, glob
import numpy as np
import cv2

try:
    import pandas as pd
except Exception:
    pd = None  # si no hay pandas, usaremos csv nativo

import csv  # fallback para CSV si no hay pandas

# ----------------------------- CONFIG -----------------------------
CONFIG = {
    # Carpeta con imágenes "undist" (por ejemplo, *_und.png de tu pipeline)
    "UNDIST_DIR": r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\undist",

    # Carpeta de salida para overlays y máscaras
    "OUT_DIR": r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\seg_color",

    # ¿Guardar máscaras binarias por color?
    "SAVE_MASKS": False,

    # Filtro de región válida (ROI): usar pixeles no-negros como máscara general
    # (útil si tu undist ya trae negro fuera de la zona útil)
    "USE_VALID_ROI": True,

    # Limpieza morfológica y filtrado
    "MIN_AREA_PX": 500,          # área mínima de blob
    "MEDIAN_K": 3,               # mediana (3 = kernel 3x3), 0 desactiva
    "OPEN_DISK": 3,              # imopen con disco, 0 desactiva
    "MAX_BLOBS_PER_COLOR": 2,    # máximo de blobs a reportar por color (los más grandes)

    # Paleta de colores (HSV normalizado 0..1 para H,S,V)
    # Puedes ajustar los umbrales por color y añadir más entradas.
    "PALETTE": [
        # Colores “básicos”
        {"name":"rojo",     "hue_windows":[(0.95,1.00),(0.00,0.05)], "sat_min":0.30, "val_min":0.15},
        {"name":"naranja",  "hue_windows":[(0.06,0.10)],             "sat_min":0.30, "val_min":0.15},
        {"name":"amarillo", "hue_windows":[(0.11,0.18)],             "sat_min":0.25, "val_min":0.20},
        {"name":"verde",    "hue_windows":[(0.25,0.45)],             "sat_min":0.25, "val_min":0.15},
        {"name":"cian",     "hue_windows":[(0.48,0.56)],             "sat_min":0.25, "val_min":0.15},
        {"name":"azul",     "hue_windows":[(0.56,0.75)],             "sat_min":0.25, "val_min":0.15},
        {"name":"purpura",  "hue_windows":[(0.76,0.83)],             "sat_min":0.25, "val_min":0.15},
        {"name":"magenta",  "hue_windows":[(0.82,0.94)],             "sat_min":0.30, "val_min":0.15},

        # “Combinaciones” / variantes claras (RGB mixtos en HSV)
        {"name":"rosado",   "hue_windows":[(0.90,0.05)],             "sat_min":0.18, "val_min":0.55},
        {"name":"celeste",  "hue_windows":[(0.52,0.60)],             "sat_min":0.18, "val_min":0.55},
    ],

    # Colores de dibujo (B,G,R en 0..255) para cada etiqueta
    "DRAW_COLORS": {
        "rojo":    (60, 60, 230),
        "naranja": (60, 160, 255),
        "amarillo":(60, 220, 240),
        "verde":   (80, 200, 60),
        "cian":    (220, 220, 60),
        "azul":    (230, 120, 70),
        "purpura": (230, 90, 170),
        "magenta": (200, 70, 230),
        "rosado":  (200, 120, 255),
        "celeste": (255, 200, 120),
    },

    # Export CSV
    "WRITE_CSV": True,
    "CSV_PATH": r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\detecciones_colores.csv"
}
# -------------------------------------------------------------------

# ---------- Utilidades HSV ----------
def _to_opencv_hsv_ranges(h_norm_lo, h_norm_hi, s_min_norm, v_min_norm):
    """
    Convierte umbrales normalizados [0..1] a escala OpenCV:
    - Hue: 0..179   (OpenCV usa H/2)
    - Sat: 0..255
    - Val: 0..255
    Devuelve (lower, upper) para cv2.inRange.
    Maneja “wrap” si lo > hi devolviendo None (se manejará fuera).
    """
    loH = int(round(h_norm_lo * 179))
    hiH = int(round(h_norm_hi * 179))
    sMin = int(round(s_min_norm * 255))
    vMin = int(round(v_min_norm * 255))
    # Para rango alto, permitimos todo S/V hasta 255
    lower = np.array([min(loH,179), sMin, vMin], dtype=np.uint8)
    upper = np.array([min(hiH,179), 255, 255],   dtype=np.uint8)
    return lower, upper

def _mask_hue_windows(hsv, hue_windows, s_min, v_min):
    """
    Crea una máscara combinando una o varias ventanas de tono.
    Soporta “wrap” 1->0 (ej: (0.95,1.00) U (0.00,0.05) para rojo).
    """
    H, W = hsv.shape[:2]
    mask_total = np.zeros((H,W), dtype=np.uint8)
    for (lo, hi) in hue_windows:
        lower, upper = _to_opencv_hsv_ranges(lo, hi, s_min, v_min)
        if lo <= hi:
            m = cv2.inRange(hsv, lower, upper)       # [lo..hi]
            mask_total = cv2.bitwise_or(mask_total, m)
        else:
            # wrap: [lo..1] U [0..hi] -> dos rangos
            lower1, upper1 = _to_opencv_hsv_ranges(lo, 1.00, s_min, v_min)
            lower2, upper2 = _to_opencv_hsv_ranges(0.00, hi,  s_min, v_min)
            m1 = cv2.inRange(hsv, lower1, upper1)
            m2 = cv2.inRange(hsv, lower2, upper2)
            mask_total = cv2.bitwise_or(mask_total, cv2.bitwise_or(m1, m2))
    return mask_total

# ---------- Lógica principal ----------
def main(cfg):
    und_dir = Path(cfg["UNDIST_DIR"])
    out_dir = Path(cfg["OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Subcarpetas de salida
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)
    masks_dir = out_dir / "masks"
    if cfg["SAVE_MASKS"]:
        masks_dir.mkdir(exist_ok=True)

    # Recolectar imágenes
    patterns = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")
    paths = []
    for p in patterns:
        paths += glob.glob(str(und_dir / p))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No hay imágenes en {und_dir}")

    rows = []  # filas para CSV (archivo, color, cx, cy, x, y, w, h, area)

    for i, p in enumerate(paths, 1):
        img_path = Path(p)
        I = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if I is None:
            print(f"[!] No pude leer {img_path.name}, sigo...")
            continue

        H, W = I.shape[:2]
        # HSV de OpenCV: cv2.cvtColor convierte BGR -> HSV (H:0..179, S:0..255, V:0..255)
        hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)

        # ROI válida: si tu undist trae negro fuera del espejo, esto restringe la búsqueda
        if cfg["USE_VALID_ROI"]:
            gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            valid_roi = (gray > 0).astype(np.uint8) * 255
            # mantener componente mayor para evitar “islas”
            num, lab, stats, _ = cv2.connectedComponentsWithStats(valid_roi, connectivity=8)
            if num >= 2:
                # índice del componente con área máxima (ignorando etiqueta 0 = fondo)
                idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                valid_roi = np.where(lab == idx, 255, 0).astype(np.uint8)
        else:
            valid_roi = np.ones((H,W), dtype=np.uint8) * 255

        overlay = I.copy()

        for color_def in cfg["PALETTE"]:
            name = color_def["name"]
            hue_windows = color_def["hue_windows"]
            s_min = color_def["sat_min"]
            v_min = color_def["val_min"]

            # 1) Máscara por tono + S/V mínimos
            m = _mask_hue_windows(hsv, hue_windows, s_min, v_min)

            # 2) Restringir a ROI válida
            m = cv2.bitwise_and(m, valid_roi)

            # 3) Limpieza morfológica
            if cfg["MEDIAN_K"] and cfg["MEDIAN_K"] >= 3:
                m = cv2.medianBlur(m, cfg["MEDIAN_K"])
            if cfg["OPEN_DISK"] and cfg["OPEN_DISK"] > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["OPEN_DISK"], cfg["OPEN_DISK"]))
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

            # 4) Filtrar por área mínima y ordenar blobs
            num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
            # stats: [label, x, y, w, h, area]
            if num <= 1:
                # no hay componentes (solo fondo)
                if cfg["SAVE_MASKS"]:
                    _save_mask(masks_dir, img_path.stem, name, m)
                continue

            # Crear lista de blobs (ignorando 0=fondo)
            blobs = []
            for lbl in range(1, num):
                x, y, w, h, area = stats[lbl, cv2.CC_STAT_LEFT], stats[lbl, cv2.CC_STAT_TOP], \
                                    stats[lbl, cv2.CC_STAT_WIDTH], stats[lbl, cv2.CC_STAT_HEIGHT], \
                                    stats[lbl, cv2.CC_STAT_AREA]
                if area < cfg["MIN_AREA_PX"]:
                    continue
                cx, cy = centroids[lbl]
                blobs.append({"x":x, "y":y, "w":w, "h":h, "area":area, "cx":cx, "cy":cy, "label":lbl})

            if not blobs:
                if cfg["SAVE_MASKS"]:
                    _save_mask(masks_dir, img_path.stem, name, m)
                continue

            # Ordenar por área descendente y quedarnos con top-N
            blobs.sort(key=lambda b: b["area"], reverse=True)
            blobs = blobs[:CONFIG["MAX_BLOBS_PER_COLOR"]]

            # 5) Dibujar y guardar filas
            draw_col = CONFIG["DRAW_COLORS"].get(name, (0,255,255))  # por defecto, amarillo
            for b in blobs:
                x, y, w, h = b["x"], b["y"], b["w"], b["h"]
                cx, cy, area = b["cx"], b["cy"], b["area"]
                cv2.rectangle(overlay, (x,y), (x+w, y+h), draw_col, 2)
                label = f"{name} | A={area}"
                cv2.putText(overlay, label, (x, max(0, y-7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_col, 2, cv2.LINE_AA)

                rows.append({
                    "file": img_path.name,
                    "color": name,
                    "cx": float(cx), "cy": float(cy),
                    "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                    "area": int(area),
                    # campos extra útiles para después (distancia / pano):
                    "img_w": W, "img_h": H
                })

            # Guardar máscara por color (opcional)
            if cfg["SAVE_MASKS"]:
                _save_mask(masks_dir, img_path.stem, name, m)

        # 6) Guardar overlay por imagen
        out_path = overlays_dir / f"{img_path.stem}_colors.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"✓ {img_path.name} -> {out_path.name}")

    # 7) Escribir CSV
    if CONFIG["WRITE_CSV"]:
        csv_path = Path(CONFIG["CSV_PATH"])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(rows, csv_path)
        print(f">> CSV guardado en: {csv_path}")

# ----------------------------- Helpers -----------------------------
def _save_mask(masks_dir: Path, stem: str, color_name: str, mask_u8: np.ndarray):
    """Guarda la máscara binaria de un color (8-bit 0/255)."""
    color_dir = masks_dir / color_name
    color_dir.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(color_dir / f"{stem}_{color_name}.png"), mask_u8)

def _write_csv(rows, csv_path: Path):
    """Escribe detecciones a CSV. Usa pandas si está disponible; si no, csv nativo."""
    if not rows:
        # Crear CSV vacío con encabezados básicos
        headers = ["file","color","cx","cy","x","y","w","h","area","img_w","img_h"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(headers)
        return

    if pd is not None:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, encoding="utf-8")
    else:
        headers = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in rows:
                w.writerow(r)

# ------------------------------- Main ------------------------------
if __name__ == "__main__":
    main(CONFIG)
