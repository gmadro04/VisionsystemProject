# ============================================================
#  segmentacion_HSV.py  
#  Segmentación de blobs de color (HSV) sobre imágenes undist
#  - Lee todas las imágenes de ENTRADA_DIR
#  - Segmenta colores definidos en una paleta (HSV normalizado 0..1)
#  - Limpia, agrupa y extrae blobs (bboxes, centroides, áreas)
#  - Guarda overlays y (opcional) máscaras por color
#  - Exporta un CSV con los resultados
#
#  Optimizaciones clave para Raspberry Pi:
#    * Opción de procesar a menor escala (FACTOR_ESCALA)
#    * ROI a partir del canal V (evita conversión extra a gris)
#    * Morfología ligera y mediana desactivada por defecto
#    * Control de hilos OpenCV y compresión de PNG
# ============================================================

from pathlib import Path
import os, glob
import numpy as np
import cv2

try:
    import pandas as pd
except Exception:
    pd = None  # fallback a csv nativo

import csv  # fallback para CSV

# ----------------------------- CONFIG -----------------------------
CONFIG = {
    # Carpeta con imágenes "undist"
    "ENTRADA_DIR": r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\undist",

    # Carpeta de salida para overlays y máscaras
    "SALIDA_DIR": r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\seg_color",

    # ¿Guardar máscaras binarias por color?
    "GUARDAR_MASCARAS": True,

    # Usar ROI válida desde canal V (pixeles V>0) y quedarse con el componente mayor
    "USAR_ROI_VALIDA": True,
    "ROI_COMPONENTE_MAYOR": True,

    # Escalado para acelerar en Raspberry Pi (1.0 = sin cambio; p.ej. 0.75 ó 0.5 acelera bastante)
    "FACTOR_ESCALA": 1.0,

    # Limpieza morfológica y filtrado
    "AREA_MIN_PX": 500,           # área mínima de blob
    "MEDIANA_K": 0,               # 0 = desactiva (3 = kernel 3x3)
    "APERTURA_DISCO": 3,          # imopen con disco (0 desactiva)
    "MAX_BLOBS_POR_COLOR": 2,     # reportar los N más grandes por color

    # Paleta de colores (HSV normalizado 0..1) -> ¡OJO! sin amarillo ni naranja
    "PALETA": [
        {"nombre":"rojo",    "ventanas_h":[(0.95,1.00),(0.00,0.05)], "s_min":0.30, "v_min":0.15},
        {"nombre":"verde",   "ventanas_h":[(0.25,0.45)],             "s_min":0.25, "v_min":0.15},
        {"nombre":"cian",    "ventanas_h":[(0.48,0.56)],             "s_min":0.25, "v_min":0.15},
        {"nombre":"azul",    "ventanas_h":[(0.56,0.75)],             "s_min":0.25, "v_min":0.15},
        {"nombre":"purpura", "ventanas_h":[(0.76,0.83)],             "s_min":0.25, "v_min":0.15},
        {"nombre":"magenta", "ventanas_h":[(0.82,0.94)],             "s_min":0.30, "v_min":0.15},
        # variantes claras (combinaciones en HSV)
        {"nombre":"rosado",  "ventanas_h":[(0.90,0.05)],             "s_min":0.18, "v_min":0.55},
        {"nombre":"celeste", "ventanas_h":[(0.52,0.60)],             "s_min":0.18, "v_min":0.55},
    ],

    # Colores de dibujo (B,G,R en 0..255)
    "COLORES_DIBUJO": {
        "rojo":    (60, 60, 230),
        "verde":   (80, 200, 60),
        "cian":    (220, 220, 60),
        "azul":    (230, 120, 70),
        "purpura": (230, 90, 170),
        "magenta": (200, 70, 230),
        "rosado":  (200, 120, 255),
        "celeste": (255, 200, 120),
    },

    # Export CSV
    "ESCRIBIR_CSV": True,
    "CSV_PATH": r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\detecciones_colores.csv",

    # OpenCV / guardado
    "OPENCV_HILOS": 2,                   # 0 = auto; 1..N para fijar hilos
    "PNG_COMPRESION": 3                  # 0 (rápido/bajo) .. 9 (lento/alto)
}
# -------------------------------------------------------------------

# ============================================================
# Utilidades HSV
# ============================================================

def a_rango_hsv_opencv(h_norm_lo, h_norm_hi, s_min_norm, v_min_norm):
    """
    Convierte umbrales normalizados [0..1] a escala OpenCV:
      Hue: 0..179   (OpenCV usa H/2)
      Sat: 0..255
      Val: 0..255
    Devuelve (lower, upper) para cv2.inRange.
    """
    loH = int(round(h_norm_lo * 179))
    hiH = int(round(h_norm_hi * 179))
    sMin = int(round(s_min_norm * 255))
    vMin = int(round(v_min_norm * 255))
    lower = np.array([min(loH,179), sMin, vMin], dtype=np.uint8)
    upper = np.array([min(hiH,179), 255, 255],   dtype=np.uint8)
    return lower, upper

def mascara_por_tono(hsv, ventanas_h, s_min, v_min):
    """
    Crea una máscara combinando una o varias ventanas de tono (soporta wrap 1->0).
    Ej: rojo: [(0.95,1.00),(0.00,0.05)]
    """
    H, W = hsv.shape[:2]
    mask_total = np.zeros((H,W), dtype=np.uint8)
    for (lo, hi) in ventanas_h:
        if lo <= hi:
            lower, upper = a_rango_hsv_opencv(lo, hi, s_min, v_min)
            m = cv2.inRange(hsv, lower, upper)
            mask_total |= m
        else:
            # Rango envuelto: [lo..1] U [0..hi]
            lower1, upper1 = a_rango_hsv_opencv(lo, 1.00, s_min, v_min)
            lower2, upper2 = a_rango_hsv_opencv(0.00, hi,  s_min, v_min)
            m1 = cv2.inRange(hsv, lower1, upper1)
            m2 = cv2.inRange(hsv, lower2, upper2)
            mask_total |= (m1 | m2)
    return mask_total

# ============================================================
# Helpers de E/S
# ============================================================

def guardar_mascara(dir_masks: Path, stem: str, nombre_color: str, mask_u8: np.ndarray, comp_png: int = 3):
    """Guarda la máscara binaria de un color (8-bit 0/255)."""
    color_dir = dir_masks / nombre_color
    color_dir.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(color_dir / f"{stem}_{nombre_color}.png"), mask_u8,
                [cv2.IMWRITE_PNG_COMPRESSION, comp_png])

def escribir_csv(filas, csv_path: Path):
    """Escribe detecciones a CSV (pandas si está, si no csv nativo)."""
    headers = ["file","color","cx","cy","x","y","w","h","area","img_w","img_h"]
    if not filas:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(headers)
        return

    if pd is not None:
        df = pd.DataFrame(filas)
        df.to_csv(csv_path, index=False, encoding="utf-8")
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in filas:
                w.writerow(r)

# ============================================================
# Lógica principal
# ============================================================

def ejecutar(cfg):
    # --- afinamiento OpenCV para RPi ---
    try:
        if cfg.get("OPENCV_HILOS", 0) > 0:
            cv2.setNumThreads(int(cfg["OPENCV_HILOS"]))
    except Exception:
        pass

    ent_dir = Path(cfg["ENTRADA_DIR"])
    sal_dir = Path(cfg["SALIDA_DIR"])
    sal_dir.mkdir(parents=True, exist_ok=True)

    # Subcarpetas de salida
    overlays_dir = sal_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)
    masks_dir = sal_dir / "masks"
    if cfg["GUARDAR_MASCARAS"]:
        masks_dir.mkdir(exist_ok=True)

    # Recolectar imágenes
    patrones = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")
    rutas = []
    for p in patrones:
        rutas += glob.glob(str(ent_dir / p))
    rutas = sorted(rutas)
    if not rutas:
        raise FileNotFoundError(f"No hay imágenes en {ent_dir}")

    filas_csv = []  # filas para CSV (archivo, color, cx, cy, x, y, w, h, area)
    escala = float(cfg.get("FACTOR_ESCALA", 1.0))
    comp_png = int(cfg.get("PNG_COMPRESION", 3))

    for i, p in enumerate(rutas, 1):
        ruta_img = Path(p)
        I = cv2.imread(str(ruta_img), cv2.IMREAD_COLOR)
        if I is None:
            print(f"[!] No pude leer {ruta_img.name}, sigo...")
            continue

        # --- escalado opcional para acelerar ---
        if escala != 1.0:
            I = cv2.resize(I, None, fx=escala, fy=escala, interpolation=cv2.INTER_LINEAR)

        H, W = I.shape[:2]

        # HSV de OpenCV (H:0..179, S:0..255, V:0..255)
        hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
        V = hsv[:,:,2]  # canal V para ROI válida

        # ROI válida: usar V>0; opcionalmente quedarse con el componente mayor
        if cfg["USAR_ROI_VALIDA"]:
            roi = (V > 0).astype(np.uint8) * 255
            if cfg.get("ROI_COMPONENTE_MAYOR", True):
                num, lab, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
                if num >= 2:
                    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    roi = np.where(lab == idx, 255, 0).astype(np.uint8)
        else:
            roi = np.ones((H,W), dtype=np.uint8) * 255

        overlay = I.copy()

        for color_def in cfg["PALETA"]:
            nombre = color_def["nombre"]
            ventanas_h = color_def["ventanas_h"]
            s_min = color_def["s_min"]
            v_min = color_def["v_min"]

            # 1) Máscara por tono + S/V mínimos
            m = mascara_por_tono(hsv, ventanas_h, s_min, v_min)

            # 2) Restringir a ROI válida
            m = cv2.bitwise_and(m, roi)

            # 3) Limpieza morfológica ligera (pensado para RPi)
            if cfg["MEDIANA_K"] and cfg["MEDIANA_K"] >= 3:
                m = cv2.medianBlur(m, cfg["MEDIANA_K"])
            if cfg["APERTURA_DISCO"] and cfg["APERTURA_DISCO"] > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (cfg["APERTURA_DISCO"], cfg["APERTURA_DISCO"]))
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

            # 4) Componentes conectadas y filtro por área
            num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
            if num <= 1:
                if cfg["GUARDAR_MASCARAS"]:
                    guardar_mascara(masks_dir, ruta_img.stem, nombre, m, comp_png)
                continue

            blobs = []
            for lbl in range(1, num):
                x = stats[lbl, cv2.CC_STAT_LEFT]
                y = stats[lbl, cv2.CC_STAT_TOP]
                w = stats[lbl, cv2.CC_STAT_WIDTH]
                h = stats[lbl, cv2.CC_STAT_HEIGHT]
                area = stats[lbl, cv2.CC_STAT_AREA]
                if area < cfg["AREA_MIN_PX"]:
                    continue
                cx, cy = centroids[lbl]
                blobs.append({"x":x, "y":y, "w":w, "h":h, "area":area, "cx":cx, "cy":cy, "label":lbl})

            if not blobs:
                if cfg["GUARDAR_MASCARAS"]:
                    guardar_mascara(masks_dir, ruta_img.stem, nombre, m, comp_png)
                continue

            # 5) Ordenar por área y recortar top-N
            blobs.sort(key=lambda b: b["area"], reverse=True)
            blobs = blobs[:cfg["MAX_BLOBS_POR_COLOR"]]

            # 6) Dibujar y preparar filas CSV
            col_dibujo = cfg["COLORES_DIBUJO"].get(nombre, (0,255,255))
            for b in blobs:
                x, y, w, h = b["x"], b["y"], b["w"], b["h"]
                cx, cy, area = b["cx"], b["cy"], b["area"]
                cv2.rectangle(overlay, (x,y), (x+w, y+h), col_dibujo, 2)
                etiqueta = f"{nombre} | A={area}"
                cv2.putText(overlay, etiqueta, (x, max(0, y-7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_dibujo, 2, cv2.LINE_AA)

                filas_csv.append({
                    "file": ruta_img.name,
                    "color": nombre,
                    "cx": float(cx), "cy": float(cy),
                    "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                    "area": int(area),
                    "img_w": W, "img_h": H
                })

            # 7) Guardar máscara por color (opcional)
            if cfg["GUARDAR_MASCARAS"]:
                guardar_mascara(masks_dir, ruta_img.stem, nombre, m, comp_png)

        # 8) Guardar overlay por imagen (compresión moderada para rapidez)
        out_path = overlays_dir / f"{ruta_img.stem}_colors.png"
        cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_PNG_COMPRESSION, comp_png])
        print(f"✓ {ruta_img.name} -> {out_path.name}")

    # 9) Escribir CSV
    if CONFIG["ESCRIBIR_CSV"]:
        csv_path = Path(CONFIG["CSV_PATH"])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        escribir_csv(filas_csv, csv_path)
        print(f">> CSV guardado en: {csv_path}")

# ------------------------------- Main ------------------------------
if __name__ == "__main__":
    ejecutar(CONFIG)
