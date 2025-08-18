# ============================================================
#  Lote: segmentar -> quitar distorsión -> panorámica
#  Ajusta CONFIG y ejecuta:  python proceso_general.py
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
    "SRC_DIR": r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\img_1280x1080",

    # Calibración .yml exportada desde MATLAB (recomendado)
    "CALIB_PATH": r"calib_output\cameraParams_1280x1080.yml",

    # Máscara .yml con masks.mascaraRes_WxH o .mat con mask_ref / pos+rad
    "MASK_PATH": "espejo_mask.mat",

    # Carpetas de salida se crean si no existen
    "UNDIST_DIR": r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\undist",
    "PANO_DIR":   r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\panos",

    # Parámetros de unwrap / orientación
    "N_THETA": 720,            # ancho del panorama (o pon 0 para auto por perímetro)
    "RMIN_FRAC": 0.06,         # recorte interior (porcentaje del radio)
    "THETA_OFFSET_DEG": 0,     # desplazar columnas (giro azimutal)
    "FLIP_180": True,          # voltear panorama 180° (rot90 sin reinterpolar)
    "GAUSS_SIGMA": 0.4,        # suavizado final (0 para desactivar)

    # Si alguna imagen no coincide con el tamaño de la calibración:
    "AUTO_FIT_TO_CALIB": False # True = encaja isotrópico con padding negro
}
# -------------------------------------------------------------------

# -------------------------- Calibración ----------------------------
def load_calibration(calib_path):
    ext = Path(calib_path).suffix.lower()
    if ext not in [".yml", ".yaml"]:
        raise ValueError("En Python usa el .yml/.yaml exportado desde MATLAB.")
    with open(calib_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cam = data.get("camera", data)
    H, W = int(cam["imageSize"][0]), int(cam["imageSize"][1])
    fx, fy = float(cam["focalLength"][0]), float(cam["focalLength"][1])
    cx, cy = float(cam["principalPoint"][0]), float(cam["principalPoint"][1])
    k = cam.get("radialDistortion", [0, 0, 0])
    if len(k) == 2: k = [k[0], k[1], 0.0]
    p = cam.get("tangentialDistortion", [0.0, 0.0])

    K = np.array(  [[fx, 0,  cx],
                    [0,  fy, cy],
                    [0,   0,  1]], dtype=np.float64)
    # OpenCV: [k1, k2, p1, p2, k3]
    D = np.array([k[0], k[1], p[0], p[1], k[2]], dtype=np.float64).reshape(1,5)
    return K, D, (H, W)

# ----------------------------- Máscara -----------------------------
def load_mask(mask_path, image_size):
    H, W = image_size
    ext = Path(mask_path).suffix.lower()

    if ext in [".yml", ".yaml"]:
        with open(mask_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        masks = data.get("masks", data)
        key = f"mascaraRes_{W}x{H}"
        if key not in masks:
            raise KeyError(f"No existe {key} en {mask_path}")
        e = masks[key]
        cx, cy = float(e["center"][0]), float(e["center"][1])
        r = float(e["radius"])
        X, Y = np.meshgrid(np.arange(1, W+1), np.arange(1, H+1))
        return (((X-cx)**2 + (Y-cy)**2) <= r**2).astype(np.uint8)

    if ext == ".mat":
        if loadmat is None:
            raise RuntimeError("Instala scipy para leer .mat:  pip install scipy")
        S = loadmat(mask_path)
        if "mask_ref" in S:
            mr = S["mask_ref"].astype(bool)
            Hm, Wm = mr.shape[:2]
            if (Hm, Wm) == (H, W):
                return mr.astype(np.uint8)
            if "pos" in S and "rad" in S:
                pos = np.array(S["pos"]).ravel().astype(float)
                cx, cy = pos[0], pos[1]
                r = float(np.array(S["rad"]).ravel()[0])
                sx, sy = W/Wm, H/Hm
                cx, cy, r = cx*sx, cy*sy, r*min(sx, sy)
                X, Y = np.meshgrid(np.arange(1, W+1), np.arange(1, H+1))
                return (((X-cx)**2 + (Y-cy)**2) <= r**2).astype(np.uint8)
            return cv2.resize(mr.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        if "pos" in S and "rad" in S:
            pos = np.array(S["pos"]).ravel().astype(float)
            cx, cy = pos[0], pos[1]
            r = float(np.array(S["rad"]).ravel()[0])
            if "imageSize" in S:
                Hm, Wm = int(S["imageSize"][0,0]), int(S["imageSize"][0,1])
                sx, sy = W/Wm, H/Hm
                cx, cy, r = cx*sx, cy*sy, r*min(sx, sy)
            X, Y = np.meshgrid(np.arange(1, W+1), np.arange(1, H+1))
            return (((X-cx)**2 + (Y-cy)**2) <= r**2).astype(np.uint8)
        raise KeyError("El .mat no contiene 'mask_ref' ni 'pos'/'rad'.")

    raise ValueError(f"Formato de máscara no soportado: {ext}")

# -------------------- Undistort + Unwrap helpers -------------------
def maximum_inscribed_circle(mask_u8):
    m = (mask_u8 > 0).astype(np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(dist)
    cy, cx = float(maxLoc[1]), float(maxLoc[0])  # (x,y) -> (cx,cy)
    return cx, cy, float(maxVal)

def unwrap_panorama(Iu, cx, cy, rMin, rMax, nTheta=720, thetaOffsetDeg=0, flip180=True, sigma=0.0):
    thetas = np.linspace(0, 2*np.pi, nTheta, endpoint=False).astype(np.float32)
    rs = np.linspace(rMin, rMax, int(max(1, round(rMax - rMin + 1)))).astype(np.float32)
    TH, R = np.meshgrid(thetas, rs)
    mapX = (cx + R*np.cos(TH)).astype(np.float32)
    mapY = (cy + R*np.sin(TH)).astype(np.float32)
    pano = cv2.remap(Iu, mapX, mapY, interpolation=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if thetaOffsetDeg:
        shift = int(round((thetaOffsetDeg/360.0) * nTheta))
        pano = np.roll(pano, shift, axis=1)
    if flip180:
        pano = np.rot90(pano, 2)
    if sigma and sigma > 0:
        k = max(3, (int(6*sigma)+1) | 1)
        pano = cv2.GaussianBlur(pano, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    return pano

def fitpad_to_canvas(I, Ht, Wt):
    """Ajuste isotrópico + padding a lienzo [Ht, Wt]. Mantiene aspect y no recorta."""
    H, W = I.shape[:2]
    s = min(Wt/W, Ht/H)
    newW, newH = max(1, round(W*s)), max(1, round(H*s))
    R = cv2.resize(I, (newW, newH), interpolation=cv2.INTER_CUBIC)
    J = np.zeros((Ht, Wt, I.shape[2]), dtype=I.dtype)
    px, py = (Wt-newW)//2, (Ht-newH)//2
    J[py:py+newH, px:px+newW, :] = R
    return J

# --------------------------- Procesos en lote -----------------------
def process_batch(cfg):
    src_dir   = Path(cfg["SRC_DIR"])
    calib_path = Path(cfg["CALIB_PATH"])
    mask_path  = Path(cfg["MASK_PATH"])
    und_dir    = Path(cfg["UNDIST_DIR"])
    pano_dir   = Path(cfg["PANO_DIR"])

    und_dir.mkdir(parents=True, exist_ok=True)
    pano_dir.mkdir(parents=True, exist_ok=True)

    # 1) Calibración y máscara (una vez)
    K, D, (Hc, Wc) = load_calibration(str(calib_path))
    Mdist = load_mask(str(mask_path), (Hc, Wc))
    if Mdist.max() == 1:
        Mdist = (Mdist*255).astype(np.uint8)

    # 2) Geometría (centro / radios / grilla) con la máscara undistorsionada
    Mund = cv2.undistort(Mdist, K, D, None, K)
    _, Mund = cv2.threshold(Mund, 127, 255, cv2.THRESH_BINARY)
    Mund = cv2.morphologyEx(Mund, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    cx, cy, rMax0 = maximum_inscribed_circle(Mund)
    rMin = max(2.0, round(cfg["RMIN_FRAC"] * rMax0))
    rMax = max(rMin + 5.0, math.floor(rMax0) - 2.0)

    # nTheta: fijo o automático por perímetro
    if cfg["N_THETA"] and cfg["N_THETA"] > 0:
        nTheta = int(cfg["N_THETA"])
    else:
        rMid = 0.5*(rMin + rMax)
        nTheta = max(360, int(round(2*math.pi*rMid)))  # muestreo ~perímetro

    print(f">> Calib {Wc}x{Hc} | centro=({cx:.1f},{cy:.1f}) r=[{int(rMin)}..{int(rMax)}] | nTheta={nTheta}")

    # 3) Archivos a procesar
    patterns = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")
    paths = []
    for p in patterns:
        paths += glob.glob(str(src_dir / p))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No hay imágenes en {src_dir}")

    # 4) Lote
    for i, p in enumerate(paths, 1):
        img_path = Path(p)
        I = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if I is None:
            print(f"[!] No pude leer {img_path.name}, sigo...")
            continue

        # Normalizar tamaño si se pide
        if cfg["AUTO_FIT_TO_CALIB"]:
            if I.shape[:2] != (Hc, Wc):
                I = fitpad_to_canvas(I, Hc, Wc)
        else:
            if I.shape[:2] != (Hc, Wc):
                print(f"[!] {img_path.name}: tamaño {I.shape[1]}x{I.shape[0]} ≠ calib {Wc}x{Hc}. Saltando...")
                continue

        # Segmentación con máscara (dominio distorsionado)
        Im = I.copy()
        Im[Mdist == 0] = 0

        # Undistort
        Iu = cv2.undistort(Im, K, D, None, K)

        # Unwrap
        pano = unwrap_panorama(
            Iu, cx, cy, rMin, rMax,
            nTheta=nTheta,
            thetaOffsetDeg=cfg["THETA_OFFSET_DEG"],
            flip180=cfg["FLIP_180"],
            sigma=cfg["GAUSS_SIGMA"]
        )

        # Guardar
        stem = img_path.stem
        und_out  = und_dir  / f"{stem}_und.png"
        pano_out = pano_dir / f"{stem}_pano.png"
        cv2.imwrite(str(und_out),  Iu)
        cv2.imwrite(str(pano_out), pano)

        print(f"  [{i:03d}/{len(paths)}] ✓ {img_path.name} -> und:{und_out.name} | pano:{pano_out.name}")

    print(pf.figlet_format("Lote finalizado", font="bubble"))
    print("===== salidas de archivos ===== \n")
    print(f"Undist: {und_dir}")
    print(f"Panoramas: {pano_dir}")

# ------------------------------ Main -------------------------------
if __name__ == "__main__":
    process_batch(CONFIG)
