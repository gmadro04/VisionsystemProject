# ============================================================
#  Omni: segmentar -> undistort -> panorámica 
#  Edita CONFIG y ejecuta: py proceso_general.py
# ============================================================

from pathlib import Path
import os, math
import numpy as np
import cv2
import yaml

# (Opcionales)
try:
    from scipy.io import loadmat
except Exception:
    loadmat = None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ----------------------------- CONFIG -----------------------------
CONFIG = {
    # Imagen de entrada (1280x1080):
    "IMG_PATH": r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\img_qupas\img_20250724_145948.jpg",

    # Calibración (exportada desde MATLAB a .yml):
    "CALIB_PATH": r"C:\Users\GMADRO04\InternxtDrive - 9b57733c-7089-4577-8a63-3bde573d87c2\Maestría\PRIMER SEMESTRE\VISION POR COMPUTADOR\VisionsystemProject\calib_output\cameraParams_1280x1080.yml",

    # Máscara (.yml con "masks: mascaraRes_WxH" o .mat con mask_ref / pos+rad):
    "MASK_PATH": r"C:\Users\GMADRO04\InternxtDrive - 9b57733c-7089-4577-8a63-3bde573d87c2\Maestría\PRIMER SEMESTRE\VISION POR COMPUTADOR\VisionsystemProject\espejo_mask.mat",

    # Carpeta de salida (se crea si no existe):
    "OUT_DIR": r"C:\Users\GMADRO04\InternxtDrive - 9b57733c-7089-4577-8a63-3bde573d87c2\Maestría\PRIMER SEMESTRE\VISION POR COMPUTADOR\VisionsystemProject\salidas",

    # Parámetros de unwrap:
    "N_THETA": 720,
    "RMIN_FRAC": 0.06,
    "THETA_OFFSET_DEG": 0,   # gíralo si quieres otro “norte”
    "FLIP_180": True,        # True para ver la pano “derecha”
    "GAUSS_SIGMA": 0.0,      # 0.4 si quieres suavizar
    "MOSTRAR": True          # False en Raspberry headless
}
# -------------------------------------------------------------------

# -------------------------- Calibración ----------------------------

def load_calibration(calib_path):
    ext = Path(calib_path).suffix.lower()
    if ext not in [".yml", ".yaml"]:
        raise ValueError("Calibración en Python: usa el .yml exportado desde MATLAB.")
    with open(calib_path, "r", encoding="utf-8") as f:
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
    D = np.array([k[0], k[1], p[0], p[1], k[2]], dtype=np.float64).reshape(1, 5)  # k1 k2 p1 p2 k3
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
        cx, cy, r = float(e["center"][0]), float(e["center"][1]), float(e["radius"])
        X, Y = np.meshgrid(np.arange(1, W+1), np.arange(1, H+1))
        return (((X-cx)**2 + (Y-cy)**2) <= r**2).astype(np.uint8)

    if ext == ".mat":
        if loadmat is None:
            raise RuntimeError("Instala scipy para leer .mat (pip install scipy).")
        S = loadmat(mask_path)
        if "mask_ref" in S:
            mr = S["mask_ref"].astype(bool)
            Hm, Wm = mr.shape[:2]
            if (Hm, Wm) == (H, W):
                return mr.astype(np.uint8)
            # intentar reconstruir con pos/rad
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
    H, W = Iu.shape[:2]
    thetas = np.linspace(0, 2*np.pi, nTheta, endpoint=False).astype(np.float32)
    rs = np.linspace(rMin, rMax, int(max(1, round(rMax - rMin + 1)))).astype(np.float32)

    TH, R = np.meshgrid(thetas, rs)
    mapX = (cx + R*np.cos(TH)).astype(np.float32)
    mapY = (cy + R*np.sin(TH)).astype(np.float32)
    pano = cv2.remap(Iu, mapX, mapY, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if thetaOffsetDeg:
        shift = int(round((thetaOffsetDeg/360.0) * nTheta))
        pano = np.roll(pano, shift, axis=1)
    if flip180:
        pano = np.rot90(pano, 2)
    if sigma and sigma > 0:
        k = max(3, (int(6*sigma)+1) | 1)
        pano = cv2.GaussianBlur(pano, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    return pano

# --------------------------- Pipeline -----------------------------

def process_one(cfg):
    img_path   = Path(cfg["IMG_PATH"])
    calib_path = Path(cfg["CALIB_PATH"])
    mask_path  = Path(cfg["MASK_PATH"])
    outdir     = Path(cfg["OUT_DIR"])

    K, D, (Hc, Wc) = load_calibration(str(calib_path))

    I = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    assert I is not None, f"No pude leer: {img_path}"
    H, W = I.shape[:2]
    assert (H, W) == (Hc, Wc), f"Tamaño imagen {W}x{H} ≠ calibración {Wc}x{Hc}"

    M = load_mask(str(mask_path), (H, W))  # 0/1
    if M.max() == 1:
        M = (M*255).astype(np.uint8)

    Im = I.copy()
    Im[M == 0] = 0

    Iu   = cv2.undistort(Im, K, D, None, K)
    Mund = cv2.undistort(M,  K, D, None, K)
    _, Mund = cv2.threshold(Mund, 127, 255, cv2.THRESH_BINARY)
    Mund = cv2.morphologyEx(Mund, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    cx, cy, rMax = maximum_inscribed_circle(Mund)
    rMin = max(2.0, round(cfg["RMIN_FRAC"] * rMax))
    rMax = max(rMin + 5.0, math.floor(rMax) - 2.0)

    pano = unwrap_panorama(
        Iu, cx, cy, rMin, rMax,
        nTheta=cfg["N_THETA"],
        thetaOffsetDeg=cfg["THETA_OFFSET_DEG"],
        flip180=cfg["FLIP_180"],
        sigma=cfg["GAUSS_SIGMA"]
    )

    outdir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem
    cv2.imwrite(str(outdir / f"{stem}_masked.png"), Im)
    cv2.imwrite(str(outdir / f"{stem}_und.png"), Iu)
    cv2.imwrite(str(outdir / f"{stem}_pano.png"), pano)

    if cfg["MOSTRAR"] and plt is not None:
        plt.figure("Omni - Python (embebido)")
        plt.subplot(231); plt.imshow(cv2.cvtColor(I,  cv2.COLOR_BGR2RGB));  plt.title("Original");     plt.axis("off")
        plt.subplot(232); plt.imshow(M, cmap="gray");                        plt.title("Máscara");      plt.axis("off")
        plt.subplot(233); plt.imshow(cv2.cvtColor(Im, cv2.COLOR_BGR2RGB));   plt.title("Segmentada");   plt.axis("off")
        plt.subplot(234); plt.imshow(cv2.cvtColor(Iu, cv2.COLOR_BGR2RGB));   plt.title("Undistort");    plt.axis("off")
        ov = cv2.addWeighted(Iu, 1.0, cv2.cvtColor(Mund, cv2.COLOR_GRAY2BGR), 0.3, 0)
        plt.subplot(235); plt.imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB));   plt.title("ROI undistort");plt.axis("off")
        plt.subplot(236); plt.imshow(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)); plt.title("Panorámica");   plt.axis("off")
        plt.tight_layout(); plt.show()
    elif cfg["MOSTRAR"] and plt is None:
        print("[Aviso] matplotlib no está instalado; se guardaron las imágenes pero no se muestran.")

    return {"I": I, "Im": Im, "Iu": Iu, "M": M, "Mund": Mund, "pano": pano,
            "cx": cx, "cy": cy, "K": K, "D": D, "rMin": rMin, "rMax": rMax}

# ------------------------------ Run ------------------------------

if __name__ == "__main__":
    _ = process_one(CONFIG)
    print(">> Proceso terminado y archivos guardados en:", CONFIG["OUT_DIR"])
