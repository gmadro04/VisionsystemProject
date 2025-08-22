# ============================================================
#  Script para: quitar distorsión -> panorámica 
#  Ajusta CONFIG y ejecuta:  py img_pro.py
#
#   - Ancho de panorámica fijo PANO_ANCHO = 1280 por defecto se puede cambiar para baja resolución
#   - Alto opcional de panorámica (PANO_ALTO: 0 = automático)
#   - Pre-cálculo de mapas de undistort y unwrapping (más rápido)
#   - Interpolación lineal en remap (rápida y estable)
# ============================================================

from pathlib import Path
import os, math, glob
import numpy as np
import cv2
import yaml
import pyfiglet as pf

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None

# ----------------------------- CONFIG -----------------------------
CONFIG = {
    # Carpeta de imágenes de entrada (png/jpg/jpeg/bmp/tif)
    "DIR_ORIGEN": r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\img_1280_curadas",
    # img_1280x1080 ---- una carpeta de procesamiento con base de 55 mm
    # img_1280_curadas --- otra carpeta con imágenes con base de 60 mm
    # Calibración .yml exportada desde MATLAB (recomendado)
    "RUTA_CALIB": r"calib_output\cameraParams_1280x1080.yml",

    # Máscara .yml con masks.mascaraRes_WxH o .mat con mask_ref / pos+rad
    "RUTA_MASCARA": "espejo_mask_1.mat", 
    # Si se trabaja con base 60 mm usar espejo_mask.mat, si no base = 55 mm usar espejo_mask_1.mat

    # Carpetas de salida
    "DIR_UNDIST": r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\undist",
    "DIR_PANO":   r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\panos",

    # Parámetros de panorama / orientación
    # PANO_ANCHO: fija el ancho de la pano (en columnas theta). Úsalo para 1280.
    # PANO_ALTO: 0 = automático (número de radios = rMax-rMin+1). Si >0, fija el alto.
    "PANO_ANCHO": 1280,         # resolución de la panoramica configurable
    "PANO_ALTO":  0,            # 0 = auto por rMax-rMin; pon >0 para fijar alto
    "RMIN_FRAC": 0.06,          # recorte interior (porcentaje del radio)
    "OFFSET_AZIMUT_GRADOS": 0,  # desplazar columnas (giro)
    "VOLTEAR_180": True,        # voltear panorama 180° (rot90 sin reinterpolar)
    "GAUSS_SIGMA": 0.4,         # suavizado final (0 desactiva)

    # Si alguna imagen no coincide con el tamaño de la calibración:
    "AJUSTAR_A_CALIB": False,   # True = encaja isotrópico con padding negro

    # Rendimiento
    "OPENCV_HILOS": 2,          # 0 = auto; 1..N para fijar hilos
    "INTERP_REMAPPING": "linear" # "linear" | "cubic"
}
# -------------------------------------------------------------------

# -------------------------- Calibración ----------------------------
def cargar_calibracion(ruta_calib):
    """
    Lee parámetros de calibración (.yml) y regresa K, D y (H,W).
    """
    ext = Path(ruta_calib).suffix.lower()
    if ext not in [".yml", ".yaml"]:
        raise ValueError("En Python usa el .yml/.yaml exportado desde MATLAB.")
    with open(ruta_calib, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cam = data.get("camera", data)
    H, W = int(cam["imageSize"][0]), int(cam["imageSize"][1])
    fx, fy = float(cam["focalLength"][0]), float(cam["focalLength"][1])
    cx, cy = float(cam["principalPoint"][0]), float(cam["principalPoint"][1])
    k = cam.get("radialDistortion", [0, 0, 0])
    if len(k) == 2: k = [k[0], k[1], 0.0]
    p = cam.get("tangentialDistortion", [0.0, 0.0])

    K = np.array([[fx, 0,  cx],
                    [0,  fy, cy],
                    [0,   0,  1]], dtype=np.float64)
    # OpenCV: [k1, k2, p1, p2, k3]
    D = np.array([k[0], k[1], p[0], p[1], k[2]], dtype=np.float64).reshape(1,5)
    return K, D, (H, W)

# ----------------------------- Máscara -----------------------------
def cargar_mascara(ruta_mask, image_size):
    """
    Carga máscara de espejo (yml/mat) y la ajusta al tamaño de imagen.
    Devuelve uint8 {0,255}.
    """
    H, W = image_size
    ext = Path(ruta_mask).suffix.lower()

    if ext in [".yml", ".yaml"]:
        with open(ruta_mask, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        masks = data.get("masks", data)
        key = f"mascaraRes_{W}x{H}"
        if key not in masks:
            raise KeyError(f"No existe {key} en {ruta_mask}")
        e = masks[key]
        cx, cy = float(e["center"][0]), float(e["center"][1])
        r = float(e["radius"])
        X, Y = np.meshgrid(np.arange(1, W+1), np.arange(1, H+1))
        m = (((X-cx)**2 + (Y-cy)**2) <= r**2).astype(np.uint8) * 255
        return m

    if ext == ".mat":
        if loadmat is None:
            raise RuntimeError("Instala scipy para leer .mat:  pip install scipy")
        S = loadmat(ruta_mask)
        if "mask_ref" in S:
            mr = S["mask_ref"].astype(bool)
            Hm, Wm = mr.shape[:2]
            if (Hm, Wm) == (H, W):
                return (mr.astype(np.uint8) * 255)
            if "pos" in S and "rad" in S:
                pos = np.array(S["pos"]).ravel().astype(float)
                cx, cy = pos[0], pos[1]
                r = float(np.array(S["rad"]).ravel()[0])
                sx, sy = W/Wm, H/Hm
                cx, cy, r = cx*sx, cy*sy, r*min(sx, sy)
                X, Y = np.meshgrid(np.arange(1, W+1), np.arange(1, H+1))
                return ((((X-cx)**2 + (Y-cy)**2) <= r**2).astype(np.uint8) * 255)
            return (cv2.resize(mr.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) * 255)
        if "pos" in S and "rad" in S:
            pos = np.array(S["pos"]).ravel().astype(float)
            cx, cy = pos[0], pos[1]
            r = float(np.array(S["rad"]).ravel()[0])
            if "imageSize" in S:
                Hm, Wm = int(S["imageSize"][0,0]), int(S["imageSize"][0,1])
                sx, sy = W/Wm, H/Hm
                cx, cy, r = cx*sx, cy*sy, r*min(sx, sy)
            X, Y = np.meshgrid(np.arange(1, W+1), np.arange(1, H+1))
            return ((((X-cx)**2 + (Y-cy)**2) <= r**2).astype(np.uint8) * 255)
        raise KeyError("El .mat no contiene 'mask_ref' ni 'pos'/'rad'.")

    raise ValueError(f"Formato de máscara no soportado: {ext}")

# -------------------- Undistort + Unwrap helpers -------------------
def circulo_inscrito_maximo(mask_u8):
    """
    Encuentra el círculo inscrito máximo dentro de la máscara (1 canal uint8).
    Devuelve: (cx, cy, radio)
    """
    m = (mask_u8 > 0).astype(np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(dist)
    cy, cx = float(maxLoc[1]), float(maxLoc[0])  # (x,y) -> (cx,cy)
    return cx, cy, float(maxVal)

def construir_mapas_despliegue(cx, cy, rMin, rMax, ancho_pano, alto_pano, H, W):
    """
    Precalcula mapas (mapX, mapY) para desplegar (unwrap) la panorámica.
    - ancho_pano: número de columnas (theta samples)
    - alto_pano:  número de filas (radial samples). Si 0, usa rMax-rMin+1.
    """
    if alto_pano is None or alto_pano <= 0:
        alto_pano = int(max(1, round(rMax - rMin + 1)))

    thetas = np.linspace(0, 2*np.pi, int(ancho_pano), endpoint=False).astype(np.float32)
    rs = np.linspace(rMin, rMax, int(alto_pano)).astype(np.float32)
    TH, R = np.meshgrid(thetas, rs)
    mapX = (cx + R*np.cos(TH)).astype(np.float32)
    mapY = (cy + R*np.sin(TH)).astype(np.float32)

    # Asegurar límites por si hay bordes
    mapX = np.clip(mapX, 0, W-1)
    mapY = np.clip(mapY, 0, H-1)
    return mapX, mapY

def desplegar_con_mapas(Iu, mapX, mapY, interp="linear", offset_deg=0, voltear_180=False, sigma=0.0):
    """
    Aplica remap con los mapas precalculados y ajustes de orientación.
    """
    inter = cv2.INTER_LINEAR if interp == "linear" else cv2.INTER_CUBIC
    pano = cv2.remap(Iu, mapX, mapY, interpolation=inter,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if offset_deg:
        shift = int(round((offset_deg/360.0) * pano.shape[1]))
        pano = np.roll(pano, shift, axis=1)
    if voltear_180:
        pano = np.rot90(pano, 2)
    if sigma and sigma > 0:
        k = max(3, (int(6*sigma)+1) | 1)
        pano = cv2.GaussianBlur(pano, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    return pano

def ajustar_a_lienzo(I, Ht, Wt):
    """
    Ajuste isotrópico + padding a lienzo [Ht, Wt]. Mantiene aspect y no recorta.
    """
    H, W = I.shape[:2]
    s = min(Wt/W, Ht/H)
    newW, newH = max(1, round(W*s)), max(1, round(H*s))
    R = cv2.resize(I, (newW, newH), interpolation=cv2.INTER_CUBIC)
    J = np.zeros((Ht, Wt, I.shape[2]), dtype=I.dtype)
    px, py = (Wt-newW)//2, (Ht-newH)//2
    J[py:py+newH, px:px+newW, :] = R
    return J

# --------------------------- Procesos en lote -----------------------
def procesar_lote(cfg):
    # Hilos de OpenCV
    try:
        if cfg.get("OPENCV_HILOS", 0) > 0:
            cv2.setNumThreads(int(cfg["OPENCV_HILOS"]))
    except Exception:
        pass

    dir_src   = Path(cfg["DIR_ORIGEN"])
    ruta_cal  = Path(cfg["RUTA_CALIB"])
    ruta_mask = Path(cfg["RUTA_MASCARA"])
    dir_und   = Path(cfg["DIR_UNDIST"])
    dir_pano  = Path(cfg["DIR_PANO"])

    dir_und.mkdir(parents=True, exist_ok=True)
    dir_pano.mkdir(parents=True, exist_ok=True)

    # 1) Calibración y máscara (una vez)
    K, D, (Hc, Wc) = cargar_calibracion(str(ruta_cal))
    Mdist = cargar_mascara(str(ruta_mask), (Hc, Wc))  # uint8 {0,255}

    # 1.1) Precalcular mapas de undistort (acelera el lote)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (Wc, Hc), cv2.CV_32FC1)

    # 2) Geometría (centro/radios) con la máscara UNDISTORSIONADA
    Mund = cv2.remap(Mdist, map1, map2, interpolation=cv2.INTER_NEAREST)
    _, Mund = cv2.threshold(Mund, 127, 255, cv2.THRESH_BINARY)
    Mund = cv2.morphologyEx(Mund, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    cx, cy, rMax0 = circulo_inscrito_maximo(Mund)
    rMin = max(2.0, round(cfg["RMIN_FRAC"] * rMax0))
    rMax = max(rMin + 5.0, math.floor(rMax0) - 2.0)

    # 2.1) Definir tamaño deseado de la pano
    pano_ancho = int(cfg.get("PANO_ANCHO", 1280))  # <- 1280 por defecto
    pano_alto  = int(cfg.get("PANO_ALTO", 0))      # 0 = auto por rMax-rMin

    # 2.2) Precalcular mapas de unwrapping (para todo el lote)
    mapX, mapY = construir_mapas_despliegue(cx, cy, rMin, rMax, pano_ancho, pano_alto, Hc, Wc)

    print(f">> Calib {Wc}x{Hc} | centro=({cx:.1f},{cy:.1f}) r=[{int(rMin)}..{int(rMax)}] | pano={mapX.shape[1]}x{mapX.shape[0]}")

    # 3) Archivos a procesar
    patrones = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")
    rutas = []
    for p in patrones:
        rutas += glob.glob(str(dir_src / p))
    rutas = sorted(rutas)
    if not rutas:
        raise FileNotFoundError(f"No hay imágenes en {dir_src}")

    # 4) Lote
    inter = cfg.get("INTERP_REMAPPING", "linear")
    for i, p in enumerate(rutas, 1):
        ruta_img = Path(p)
        I = cv2.imread(str(ruta_img), cv2.IMREAD_COLOR)
        if I is None:
            print(f"[!] No pude leer {ruta_img.name}, sigo...")
            continue

        # Normalizar tamaño si se pide (para coincidir con calibración)
        if cfg["AJUSTAR_A_CALIB"]:
            if I.shape[:2] != (Hc, Wc):
                I = ajustar_a_lienzo(I, Hc, Wc)
        else:
            if I.shape[:2] != (Hc, Wc):
                print(f"[!] {ruta_img.name}: tamaño {I.shape[1]}x{I.shape[0]} ≠ calib {Wc}x{Hc}. Saltando...")
                continue

        # Aplicar máscara en dominio distorsionado
        Im = I.copy()
        Im[Mdist == 0] = 0

        # Undistort (con mapas precalculados)
        Iu = cv2.remap(Im, map1, map2, interpolation=cv2.INTER_LINEAR)

        # Unwrap (con mapas precalculados)
        pano = desplegar_con_mapas(
            Iu, mapX, mapY,
            interp=inter,
            offset_deg=cfg["OFFSET_AZIMUT_GRADOS"],
            voltear_180=cfg["VOLTEAR_180"],
            sigma=cfg["GAUSS_SIGMA"]
        )

        # Guardar
        stem = ruta_img.stem
        und_out  = dir_und  / f"{stem}_und.png"
        pano_out = dir_pano / f"{stem}_pano.png"
        cv2.imwrite(str(und_out),  Iu)
        cv2.imwrite(str(pano_out), pano)

        print(f"  [{i:03d}/{len(rutas)}] ✓ {ruta_img.name} -> und:{und_out.name} | pano:{pano_out.name} ({pano.shape[1]}x{pano.shape[0]})")

    print(pf.figlet_format("Lote finalizado", font="bubble"))
    print("===== salidas de archivos ===== \n")
    print(f"Undist:    {dir_und}")
    print(f"Panoramas: {dir_pano}")

# ------------------------------ Main -------------------------------
if __name__ == "__main__":
    procesar_lote(CONFIG)