# ============================================================
# pano_color_distance.py
# Lote sobre PANORÁMICAS:
#   - Segmenta colores (HSV), encuentra blobs
#   - Horizonte automático (energía vertical por fila)
#   - Azimut por columna, distancia por elevación lineal
#   - Guarda overlays y CSV
# Ejecuta:  python pano_color_distance.py
# Reqs: opencv-python, numpy, (pandas opcional)
# ============================================================

from pathlib import Path
import glob, math
import numpy as np
import cv2

try:
    import pandas as pd
except Exception:
    pd = None

# --------------------------- CONFIG ---------------------------
CONFIG = {
    # Carpeta con panorámicas (ej: *_pano.png)
    "PANO_DIR": r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\panos",

    # Carpeta de salida
    "OUT_DIR": r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\pano_color",

    # Altura de la cámara (m). None => distancia RELATIVA
    "CAM_HEIGHT_M": None,        # None para relativa

    # Ángulo de elevación para la última fila (abajo del pano)
    "ALPHA_BOTTOM_DEG": 80.0,

    # Limitar a franja inferior (suelo) [0..1]; 0 = todo
    "BOTTOM_FRAC": 0.45,

    # Morfología/ruido
    "MIN_AREA_PX": 500,
    "MEDIAN_K": 3,
    "OPEN_DISK": 3,
    "MAX_BLOBS_PER_COLOR": 2,

    # Paleta HSV (0..1)
    "PALETTE": [
        {"name":"rojo",     "hue_windows":[(0.95,1.00),(0.00,0.05)], "sat_min":0.30, "val_min":0.15},
        {"name":"naranja",  "hue_windows":[(0.06,0.10)],             "sat_min":0.30, "val_min":0.15},
        {"name":"amarillo", "hue_windows":[(0.11,0.18)],             "sat_min":0.25, "val_min":0.20},
        {"name":"verde",    "hue_windows":[(0.25,0.45)],             "sat_min":0.25, "val_min":0.15},
        {"name":"cian",     "hue_windows":[(0.48,0.56)],             "sat_min":0.25, "val_min":0.15},
        {"name":"azul",     "hue_windows":[(0.56,0.75)],             "sat_min":0.25, "val_min":0.15},
        {"name":"purpura",  "hue_windows":[(0.76,0.83)],             "sat_min":0.25, "val_min":0.15},
        {"name":"magenta",  "hue_windows":[(0.82,0.94)],             "sat_min":0.30, "val_min":0.15},
        {"name":"rosado",   "hue_windows":[(0.90,0.05)],             "sat_min":0.18, "val_min":0.55},
        {"name":"celeste",  "hue_windows":[(0.52,0.60)],             "sat_min":0.18, "val_min":0.55},
    ],

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

    # CSV
    "WRITE_CSV": True,
    "CSV_PATH": r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\detecciones_pano.csv"
}
# ---------------------------------------------------------------

# ===== Helpers HSV =====
def to_hsv_inrange(hsv, lo, hi, s_min, v_min):
    def norm_to_cv(hn, sn, vn):
        return int(round(hn*179)), int(round(sn*255)), int(round(vn*255))
    if lo <= hi:
        lh, ls, lv = norm_to_cv(lo, s_min, v_min)
        hh, _, _   = norm_to_cv(hi, 1.0, 1.0)
        return cv2.inRange(hsv, (lh, ls, lv), (hh, 255, 255))
    else:
        lh1, ls, lv = norm_to_cv(lo, s_min, v_min)
        m1 = cv2.inRange(hsv, (lh1, ls, lv), (179, 255, 255))
        hh2, _, _   = norm_to_cv(hi, 1.0, 1.0)
        m2 = cv2.inRange(hsv, (0,   ls, lv), (hh2, 255, 255))
        return cv2.bitwise_or(m1, m2)

def mask_from_palette(hsv, hue_windows, s_min, v_min):
    m = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in hue_windows:
        m = cv2.bitwise_or(m, to_hsv_inrange(hsv, lo, hi, s_min, v_min))
    return m

# ===== Horizonte AUTO en PANO =====
def estimate_horizon_row(gray):
    """
    Horizonte = fila de máxima energía de gradiente vertical.
    Evita los bordes (15%..85%).
    """
    R, _ = gray.shape
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gRow = np.mean(np.abs(gy), axis=1)
    # suave
    k = max(5, int(R*0.01))
    gRow = cv2.blur(gRow.reshape(-1,1), (1,k), borderType=cv2.BORDER_REPLICATE).ravel()
    r1, r2 = int(0.15*R), int(0.85*R)
    idx = np.argmax(gRow[r1:r2])
    return r1 + int(idx), gRow

# ==================== MAIN ====================
def main(cfg):
    pano_dir = Path(cfg["PANO_DIR"])
    out_dir = Path(cfg["OUT_DIR"]); out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = out_dir / "overlays"; overlays_dir.mkdir(exist_ok=True)

    paths = []
    for p in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"):
        paths += glob.glob(str(pano_dir / p))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No hay panorámicas en {pano_dir}")

    rows = []
    for i, p in enumerate(paths, 1):
        img_path = Path(p)
        I = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if I is None:
            print(f"[!] No pude leer {img_path.name}, sigo…")
            continue

        R, C = I.shape[0], I.shape[1]
        gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        y_h, gRow = estimate_horizon_row(gray)

        hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
        overlay = I.copy()

        # Si solo quieres suelo, restringe máscara a filas y>=y0:
        if cfg["BOTTOM_FRAC"] > 0:
            y0 = max(0, int(R*(1-cfg["BOTTOM_FRAC"])))
        else:
            y0 = 0

        for col in cfg["PALETTE"]:
            m = mask_from_palette(hsv, col["hue_windows"], col["sat_min"], col["val_min"])
            # Franja inferior
            if y0 > 0:
                m[:y0,:] = 0
            if cfg["MEDIAN_K"] and cfg["MEDIAN_K"] >= 3:
                m = cv2.medianBlur(m, cfg["MEDIAN_K"])
            if cfg["OPEN_DISK"] and cfg["OPEN_DISK"] > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["OPEN_DISK"], cfg["OPEN_DISK"]))
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

            num, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
            blobs = []
            for lbl in range(1, num):
                x, y, w, h, area = stats[lbl, cv2.CC_STAT_LEFT], stats[lbl, cv2.CC_STAT_TOP], \
                                    stats[lbl, cv2.CC_STAT_WIDTH], stats[lbl, cv2.CC_STAT_HEIGHT], \
                                    stats[lbl, cv2.CC_STAT_AREA]
                if area < cfg["MIN_AREA_PX"]:
                    continue
                cx_b, cy_b = cents[lbl]
                blobs.append({"x":x,"y":y,"w":w,"h":h,"area":area,"cx":cx_b,"cy":cy_b})

            if not blobs:
                continue

            blobs.sort(key=lambda b: b["area"], reverse=True)
            blobs = blobs[:cfg["MAX_BLOBS_PER_COLOR"]]

            draw_col = cfg["DRAW_COLORS"].get(col["name"], (0,255,255))
            for b in blobs:
                x, y, w, h = b["x"], b["y"], b["w"], b["h"]
                cx_b, cy_b = b["cx"], b["cy"]
                y_base = y + h
                x_c = cx_b

                # --- Acimut por columna (0..360)
                azim = ((x_c) * 360.0 / C) % 360.0

                # --- Elevación lineal por fila (horizonte -> 0°, fondo -> -ALPHA_BOTTOM_DEG)
                alpha_deg = -cfg["ALPHA_BOTTOM_DEG"] * ((y_base - y_h) / max(1.0, (R - y_h)))
                alpha_rad = math.radians(alpha_deg)

                if cfg["CAM_HEIGHT_M"] is None:
                    dist_val = 1.0 / max(1e-6, math.tan(abs(alpha_rad)))  # relativa
                    dist_txt = f"{dist_val:.2f} rel"
                else:
                    dist_m = cfg["CAM_HEIGHT_M"] / max(1e-6, math.tan(abs(alpha_rad)))
                    dist_val = dist_m
                    dist_txt = f"{dist_m:.2f} m"

                # Dibujar
                cv2.rectangle(overlay, (x,y), (x+w, y+h), draw_col, 2)
                label = f"{col['name']} | θ={azim:.1f}° | d={dist_txt}"
                cv2.putText(overlay, label, (x, max(0, y-7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_col, 2, cv2.LINE_AA)

                rows.append({
                    "file": img_path.name,
                    "color": col["name"],
                    "azim_deg": round(azim,2),
                    "dist": round(dist_val,3),
                    "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                    "cx": float(cx_b), "cy": float(cy_b),
                    "img_w": C, "img_h": R,
                    "alpha_deg": round(alpha_deg,2),
                    "horizon_row": int(y_h)
                })

        out_path = Path(cfg["OUT_DIR"]) / "overlays" / f"{img_path.stem}_pano_colors.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"✓ {img_path.name} -> {out_path.name}")

    # CSV
    if CONFIG["WRITE_CSV"]:
        csv_path = Path(CONFIG["CSV_PATH"])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if pd is not None:
            pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")
        else:
            import csv
            if rows:
                headers = list(rows[0].keys())
            else:
                headers = ["file","color","azim_deg","dist","x","y","w","h","cx","cy","img_w","img_h","alpha_deg","horizon_row"]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=headers); w.writeheader()
                for r in rows: w.writerow(r)
        print(f">> CSV: {csv_path}")

if __name__ == "__main__":
    main(CONFIG)
