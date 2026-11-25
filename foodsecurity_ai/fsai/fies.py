"""fies.py - Puntaje crudo, escala Rasch y clasificación en 4 niveles de la FIES.

IMPORTANTE: los parámetros de dificultad de los ítems y los umbrales de
clasificación son [POR VALIDAR]. Este módulo NO incluye parámetros calibrados
para el Perú; los recibe por inyección. Los umbrales crudos por defecto son una
parametrización de trabajo, no una calibración nacional.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

from fsai.config import COL_BINARIA, FEATURES_FIES

logger = logging.getLogger(__name__)

ITEMS: list[str] = FEATURES_FIES[:8]   # fies_q1..fies_q8

# [POR VALIDAR] cortes inferiores (>=) para leve / moderada / severa sobre el puntaje crudo.
UMBRALES_RAW: dict[str, float] = {"leve": 1, "moderada": 4, "severa": 7}


def calcular_raw_score(df: pd.DataFrame) -> pd.Series:
    """Suma de las 8 respuestas afirmativas (0-8). Si falta algún ítem, el puntaje es NaN."""
    faltan = [c for c in ITEMS if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan ítems FIES: {faltan}")
    items = df[ITEMS]
    valores = items.to_numpy(dtype=float)
    if not set(np.unique(valores[~np.isnan(valores)])) <= {0.0, 1.0}:
        raise ValueError("Los ítems FIES deben estar recodificados a 0/1 (ver recodificar_fies_crudo).")
    raw = items.sum(axis=1, min_count=len(ITEMS))
    raw.name = "fies_raw_score"
    return raw


def _theta_por_puntaje(dificultades: np.ndarray) -> dict[int, float]:
    """Resuelve sum_i expit(theta - b_i) = r para r=0..8 (máx. verosimilitud dado el puntaje).

    Para r=0 y r=8 la solución diverge; se usa la corrección habitual de puntajes
    extremos (0,5 y 7,5), declarada como supuesto metodológico.
    """
    k = len(dificultades)
    tabla: dict[int, float] = {}
    for r in range(k + 1):
        objetivo = min(max(r, 0.5), k - 0.5)
        tabla[r] = float(brentq(lambda t: expit(t - dificultades).sum() - objetivo, -15.0, 15.0))
    return tabla


def calcular_rasch_logit(df: pd.DataFrame, params_items: dict | None) -> pd.Series:
    """Medida de severidad (logit) por hogar a partir de dificultades de ítem inyectadas.

    ``params_items``: {'fies_q1': b1, ..., 'fies_q8': b8}. [POR VALIDAR]. Si es None
    devuelve NaN y registra una advertencia (no se inventan parámetros).
    """
    if params_items is None:
        logger.warning("params_items=None: fies_rasch_logit queda en NaN [POR VALIDAR].")
        return pd.Series(np.nan, index=df.index, name="fies_rasch_logit")
    faltan = [c for c in ITEMS if c not in params_items]
    if faltan:
        raise KeyError(f"params_items sin dificultad para: {faltan}")
    b = np.array([float(params_items[c]) for c in ITEMS])
    tabla = _theta_por_puntaje(b)
    raw = calcular_raw_score(df)
    logit = raw.map(lambda r: tabla[int(r)] if pd.notna(r) else np.nan).astype(float)
    logit.name = "fies_rasch_logit"
    return logit


def clasificar_fies(raw_o_logit: pd.Series, umbrales: dict) -> pd.Series:
    """Clasifica en 4 niveles: 0 seguridad (< leve), 1 leve, 2 moderada, 3 severa.

    ``umbrales``: cortes inferiores inclusivos con claves 'leve' < 'moderada' < 'severa'.
    Sirve tanto para el puntaje crudo como para el logit (con umbrales en su escala).
    """
    try:
        cortes = [float(umbrales["leve"]), float(umbrales["moderada"]), float(umbrales["severa"])]
    except KeyError as exc:
        raise KeyError(f"Falta la clave de umbral {exc}") from exc
    if not (cortes[0] < cortes[1] < cortes[2]):
        raise ValueError(f"Los umbrales deben ser estrictamente crecientes: {cortes}")
    x = pd.to_numeric(raw_o_logit, errors="coerce")
    nivel = pd.Series(np.digitize(x.fillna(-np.inf).to_numpy(), cortes, right=False), index=x.index)
    nivel = nivel.where(x.notna()).astype("Int64")
    nivel.name = "fies_nivel"
    return nivel


def agregar_columnas_fies(df: pd.DataFrame, umbrales: dict | None = None,
                          params_items: dict | None = None) -> pd.DataFrame:
    """Añade fies_raw_score, fies_rasch_logit, fies_nivel e ia_binaria (nivel>=2)."""
    out = df.copy()
    out["fies_raw_score"] = calcular_raw_score(out)
    out["fies_rasch_logit"] = calcular_rasch_logit(out, params_items)
    out["fies_nivel"] = clasificar_fies(out["fies_raw_score"], umbrales or UMBRALES_RAW)
    out[COL_BINARIA] = (out["fies_nivel"] >= 2).astype("Int64")
    out.loc[out["fies_nivel"].isna(), COL_BINARIA] = pd.NA
    return out
