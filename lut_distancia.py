# ============================================================
# lut_distancia.py
#  Mapeo LUT (y_pix -> distancia_m) para panorámicas
#  - Interpolación lineal monótona con extrapolación por pendiente
#  - Carga/guarda desde CSV
#  - Evalúa escalar o arreglo de y_pix
# ============================================================

from __future__ import annotations
import csv
from typing import Iterable, Tuple, Union
import numpy as np

Number = Union[int, float, np.number]

class DistanciaPanoLUT:
    """
    LUT: y_pix -> distancia_m para panos (mismo pipeline y resolución).
    Asume objetos en contacto con el suelo: usar y_bottom = y + h del bbox.
    """

    def __init__(self, y_pix: np.ndarray, dist_m: np.ndarray):
        assert len(y_pix) >= 2, "Se requieren al menos 2 puntos para una LUT"
        assert len(y_pix) == len(dist_m), "y_pix y dist_m deben tener misma longitud"

        y = np.asarray(y_pix, dtype=np.float32).ravel()
        d = np.asarray(dist_m, dtype=np.float32).ravel()

        # Ordenar por y ascendente
        idx = np.argsort(y)
        self.y = y[idx]
        self.d = d[idx]

        # Evitar y duplicados exactos (seguridad numérica)
        for i in range(1, len(self.y)):
            if self.y[i] <= self.y[i-1]:
                self.y[i] = self.y[i-1] + 1e-3

    # ----------------- Carga/guardado -----------------
    @classmethod
    def desde_csv(cls, ruta_csv: str, columnas: Tuple[str, str] = ("y_pix", "dist_m")) -> "DistanciaPanoLUT":
        ys, ds = [], []
        with open(ruta_csv, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            # Soporta encabezados flexibles
            y_key, d_key = columnas
            # Si no existen, intenta nombres alternativos comunes
            if r.fieldnames:
                lower = [c.lower() for c in r.fieldnames]
                if y_key not in r.fieldnames:
                    for cand in ("y", "y_bottom", "ypix"):
                        if cand in lower:
                            y_key = r.fieldnames[lower.index(cand)]
                            break
                if d_key not in r.fieldnames:
                    for cand in ("dist", "distance", "m"):
                        if cand in lower:
                            d_key = r.fieldnames[lower.index(cand)]
                            break

            for row in r:
                ys.append(float(row[y_key]))
                ds.append(float(row[d_key]))
        return cls(np.array(ys, dtype=np.float32), np.array(ds, dtype=np.float32))

    def a_csv(self, ruta_csv: str):
        with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["y_pix", "dist_m"])
            for y, d in zip(self.y, self.d):
                w.writerow([float(y), float(d)])

    # ----------------- Evaluación -----------------
    def _bordes_con_pendiente(self, yq: np.ndarray) -> np.ndarray:
        """
        Extrapolación lineal en extremos usando la pendiente local.
        """
        # Pendiente al inicio y al final
        m_left  = (self.d[1]   - self.d[0])   / (self.y[1]   - self.y[0])
        m_right = (self.d[-1]  - self.d[-2]) / (self.y[-1]  - self.y[-2])

        out = np.empty_like(yq, dtype=np.float32)
        left_mask  = yq <  self.y[0]
        right_mask = yq >  self.y[-1]
        mid_mask   = ~(left_mask | right_mask)

        out[left_mask]  = self.d[0]  + (yq[left_mask]  - self.y[0])  * m_left
        out[right_mask] = self.d[-1] + (yq[right_mask] - self.y[-1]) * m_right
        # Rango medio: interp. lineal
        out[mid_mask] = np.interp(yq[mid_mask], self.y, self.d).astype(np.float32)
        return out

    def evaluar(self, y_pix: Union[Number, Iterable[Number], np.ndarray]) -> Union[float, np.ndarray]:
        arr = np.asarray(y_pix, dtype=np.float32)
        if arr.ndim == 0:
            return float(self._bordes_con_pendiente(arr.reshape(1))[0])
        return self._bordes_con_pendiente(arr)

    # Azimut útil (opcional) si quieres ángulo con columnas -> grados
    @staticmethod
    def theta_de_x(x_centro: Number, img_w: int, offset_azimut_deg: float = 0.0) -> float:
        """
        Convierte columna (x_center) a ángulo en grados [0,360).
        Debe coincidir con el ancho de pano usado en el unwrap.
        """
        theta = (float(x_centro) / float(img_w)) * 360.0 - float(offset_azimut_deg)
        theta = theta % 360.0
        return theta
