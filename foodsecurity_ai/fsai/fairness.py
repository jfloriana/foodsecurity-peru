"""fairness.py - Brechas de desempeño por grupo con IC bootstrap y alerta de equidad.

Para cada grupo se estiman TPR (sensibilidad), FPR y tasa de error de la clase positiva
(uno-contra-resto) con pesos muestrales; la brecha TPR se mide frente a un grupo de
referencia y su IC percentil se obtiene remuestreando hogares dentro de cada grupo.
Una brecha |TPR_g - TPR_ref| > UMBRAL_BRECHA_TPR (5 pp) activa la alerta.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from fsai.config import ALPHA, COL_PESO, N_BOOT, SEED, UMBRAL_BRECHA_TPR

logger = logging.getLogger(__name__)


def _tasas(pos_real: np.ndarray, pos_pred: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """(TPR, FPR, error) ponderados; NaN si no hay positivos/negativos."""
    def razon(num: float, den: float) -> float:
        return float(num / den) if den > 0 else float("nan")
    tpr = razon(w[pos_real & pos_pred].sum(), w[pos_real].sum())
    fpr = razon(w[~pos_real & pos_pred].sum(), w[~pos_real].sum())
    err = razon(w[pos_real != pos_pred].sum(), w.sum())
    return tpr, fpr, err


def brechas_por_grupo(df: pd.DataFrame, y, pred, grupo_col: str, clase_positiva: int | Iterable[int],
                      grupo_referencia: str | None = None, n_boot: int = N_BOOT,
                      alpha: float = ALPHA, seed: int = SEED) -> pd.DataFrame:
    """Tabla por grupo: n, tpr, fpr, error_rate, brecha_tpr, ic_inf, ic_sup, alerta.

    ``clase_positiva``: entero (p. ej. 3 = severa) o colección (p. ej. [2, 3] = IA moderada/severa).
    ``grupo_referencia``: por defecto, el grupo de mayor tamaño. ``y`` y ``pred`` alineados con ``df``.
    """
    if grupo_col not in df.columns:
        raise KeyError(f"La columna de grupo '{grupo_col}' no existe.")
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    if not (len(df) == len(y) == len(pred)):
        raise ValueError("df, y y pred deben tener la misma longitud.")
    positivas = np.atleast_1d(clase_positiva)
    pos_real, pos_pred = np.isin(y, positivas), np.isin(pred, positivas)
    w = df[COL_PESO].to_numpy(dtype=float) if COL_PESO in df.columns else np.ones(len(df))
    g = df[grupo_col].astype(str).to_numpy()
    grupos = sorted(pd.unique(g))
    if len(grupos) < 2:
        raise ValueError(f"'{grupo_col}' tiene un solo grupo: no hay brecha que medir.")
    ref = grupo_referencia or max(grupos, key=lambda k: (g == k).sum())
    if ref not in grupos:
        raise ValueError(f"grupo_referencia '{ref}' no existe en {grupos}")
    idx = {k: np.flatnonzero(g == k) for k in grupos}
    rng = np.random.default_rng(seed)
    tpr_ref = _tasas(pos_real[idx[ref]], pos_pred[idx[ref]], w[idx[ref]])[0]
    filas = []
    for k in grupos:
        i = idx[k]
        tpr, fpr, err = _tasas(pos_real[i], pos_pred[i], w[i])
        brecha = tpr - tpr_ref if k != ref else 0.0
        lo = hi = float("nan")
        if k != ref and np.isfinite(brecha):
            dif = []
            for _ in range(n_boot):
                bk = rng.choice(i, size=len(i), replace=True)
                br = rng.choice(idx[ref], size=len(idx[ref]), replace=True)
                t_k = _tasas(pos_real[bk], pos_pred[bk], w[bk])[0]
                t_r = _tasas(pos_real[br], pos_pred[br], w[br])[0]
                if np.isfinite(t_k) and np.isfinite(t_r):
                    dif.append(t_k - t_r)
            if len(dif) >= 0.5 * n_boot:
                lo, hi = np.percentile(dif, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        filas.append({"grupo_col": grupo_col, "grupo": k, "n": int(len(i)), "es_referencia": k == ref,
                      "tpr": tpr, "fpr": fpr, "error_rate": err, "brecha_tpr": brecha,
                      "ic_inf": float(lo), "ic_sup": float(hi),
                      "alerta": bool(np.isfinite(brecha) and abs(brecha) > UMBRAL_BRECHA_TPR)})
    out = pd.DataFrame(filas)
    n_alertas = int(out["alerta"].sum())
    if n_alertas:
        logger.warning("Equidad (%s): %d grupo(s) con brecha TPR > %.0f pp", grupo_col, n_alertas,
                       100 * UMBRAL_BRECHA_TPR)
    return out
