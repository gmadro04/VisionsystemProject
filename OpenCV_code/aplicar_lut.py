# ============================================================
# aplicar_lut_a_csv.py
#  - Aplica LUT (y_pix -> m) a las detecciones de panos
#  - Agrega columnas: dist_m, theta_deg
#  - (Opcional) genera overlays con distancias dibujadas
# ============================================================

from pathlib import Path
import os, csv, math
import numpy as np
import cv2

try:
    import pandas as pd
except Exception:
    pd = None

from OpenCV_code.lut_distancia import DistanciaPanoLUT

CONFIG = {
    # Entradas
    "CSV_DETECCIONES": r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\detecciones_colores_panos.csv",
    "CSV_LUT":         r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\pano_calib.csv",

    # Azimut (de tu proceso de pano). Debe coincidir con OFFSET_AZIMUT_GRADOS
    "OFFSET_AZIMUT_GRADOS": 0.0,

    # Salida CSV
    "CSV_SALIDA":      r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\detecciones_colores_panos_con_dist.csv",

    # (Opcional) Overlays con distancias
    "DIBUJAR_OVERLAYS": True,
    "DIR_PANOS":        r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\panos",
    "DIR_OVERLAYS":     r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\panos_overlays_dist",

    # Visual
    "DECIMALES_M": 2
}

# ------------------------ IO helpers ------------------------
def leer_csv_dicts(ruta):
    if pd is not None:
        df = pd.read_csv(ruta)
        return df.to_dict(orient="records"), list(df.columns)
    # csv nativo
    with open(ruta, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = [row for row in r]
        return rows, r.fieldnames or []

def escribir_csv_dicts(ruta, rows, headers=None):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    if pd is not None:
        import pandas as pd
        df = pd.DataFrame(rows, columns=headers if headers else None)
        df.to_csv(ruta, index=False, encoding="utf-8")
        return
    if not headers:
        headers = list(rows[0].keys()) if rows else []
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# -------------------- Overlays por archivo -------------------
def dibujar_overlays(dir_panos: Path, dir_out: Path, filas_por_archivo, offset_deg: float, decimales=2):
    dir_out.mkdir(parents=True, exist_ok=True)

    # Cargar-dibujar-guardar por imagen
    for file_name, filas in filas_por_archivo.items():
        pano_path = dir_panos / file_name
        I = cv2.imread(str(pano_path), cv2.IMREAD_COLOR)
        if I is None:
            print(f"[!] No pude abrir pano: {file_name}")
            continue

        H, W = I.shape[:2]
        for r in filas:
            x = int(float(r["x"])); y = int(float(r["y"]))
            w = int(float(r["w"])); h = int(float(r["h"]))
            dist_m = float(r["dist_m"])
            theta  = float(r["theta_deg"])
            color  = (0, 220, 255)  # BGR ámbar

            cv2.rectangle(I, (x,y), (x+w, y+h), color, 2)
            etiqueta = f"{dist_m:.{decimales}f} m | {theta:.1f}°"
            # Caja de texto con fondo para legibilidad
            (tw, th), base = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            tx, ty = x, max(0, y-8)
            cv2.rectangle(I, (tx, ty-th-6), (tx+tw+6, ty+base), (0,0,0), -1)
            cv2.putText(I, etiqueta, (tx+3, ty-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

        out_path = dir_out / file_name.replace(".png", "_dist.png")
        cv2.imwrite(str(out_path), I)
        print(f"Overlay -> {out_path.name}")

# ----------------------------- Main -----------------------------
def main(cfg):
    # 1) Cargar detecciones y LUT
    rows, headers_in = leer_csv_dicts(cfg["CSV_DETECCIONES"])
    if not rows:
        print("[!] CSV de detecciones vacío.")
        return

    lut = DistanciaPanoLUT.desde_csv(cfg["CSV_LUT"])

    # 2) Enriquecer filas
    out_rows = []
    for r in rows:
        try:
            yb = float(r.get("y_bottom", r.get("y") + r.get("h", 0)))  # fallback por si no estaba
            img_w = int(float(r["img_w"]))
            x_c = float(r["x"]) + float(r["w"]) * 0.5

            dist = lut.evaluar(yb)
            theta = DistanciaPanoLUT.theta_de_x(x_c, img_w, cfg["OFFSET_AZIMUT_GRADOS"])

            r2 = dict(r)  # copia
            r2["dist_m"]   = float(dist)
            r2["theta_deg"] = float(theta)
            out_rows.append(r2)
        except Exception as e:
            print(f"[!] Fila con error ({r.get('file','?')}): {e}")

    # 3) Guardar CSV
    headers_out = list(headers_in)
    for extra in ["dist_m", "theta_deg"]:
        if extra not in headers_out:
            headers_out.append(extra)

    escribir_csv_dicts(cfg["CSV_SALIDA"], out_rows, headers=headers_out)
    print(f">> CSV con distancias: {cfg['CSV_SALIDA']}")

    # 4) (Opcional) Overlays
    if cfg.get("DIBUJAR_OVERLAYS", False):
        dir_panos = Path(cfg["DIR_PANOS"])
        filas_por_archivo = {}
        for r in out_rows:
            fn = r["file"]
            filas_por_archivo.setdefault(fn, []).append(r)
        dibujar_overlays(dir_panos, Path(cfg["DIR_OVERLAYS"]), filas_por_archivo,
                        cfg["OFFSET_AZIMUT_GRADOS"], decimales=cfg.get("DECIMALES_M", 2))

if __name__ == "__main__":
    main(CONFIG)
