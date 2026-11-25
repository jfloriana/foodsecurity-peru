"""inference.py - Fase 6: predicción individual y por lote con incertidumbre del ENSAMBLE.

Método declarado: el ensamble está formado por los 5 modelos de los pliegues externos de la CV anidada (cada uno
con sus propios hiperparámetros, entrenado con 4/5 de Desarrollo) más el modelo final (entrenado con todo
Desarrollo). La probabilidad puntual es el PROMEDIO de los miembros. El intervalo es un bootstrap percentil
(B remuestras con reemplazo de los miembros, IC del promedio, nivel 95 %) y se informa también el mínimo y el
máximo entre miembros. Es una medida de VARIABILIDAD DEL MODELO, no un IC frecuentista de la probabilidad
verdadera ni incorpora la incertidumbre de muestreo de la encuesta. El umbral operativo de la clase severa es el
fijado en Validación (15 %). Nada de esto es causal ni debe usarse para negar ayudas.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fsai import config as C
from fsai.metrics_weighted import predecir_con_umbral
from fsai.models import predecir_proba
from fsai.preprocessing import preparar_matriz

logger = logging.getLogger(__name__)

MAX_FILAS_LOTE = 5000


def cargar_ensamble(dir_run: Path, ident: str) -> dict[str, Any]:
    """Carga resultado, modelo final y modelos externos de una corrida de Fase 2."""
    res = json.loads((dir_run / f"resultado_{ident}.json").read_text(encoding="utf-8"))
    final = joblib.load(dir_run / res["final"]["modelo"])
    externos = [joblib.load(dir_run / "modelos" / f"outer_{ident}_f{f['pliegue']}.joblib") for f in res["folds"]]
    return {"resultado": res, "final": final, "miembros": externos + [final], "umbral": res["final"]["validacion"]["umbral_severa"]}


def especificacion_formulario(resultado: dict[str, Any], ref: pd.DataFrame) -> list[dict[str, Any]]:
    """Campos del formulario (variables del escenario) con rango/opciones observados en ``ref`` (nunca la Prueba)."""
    campos = []
    for c in resultado["features_num"]:
        s = pd.to_numeric(ref[c], errors="coerce").dropna()
        binaria = set(s.unique()) <= {0, 1}
        campos.append({"variable": c, "tipo": "binaria" if binaria else "numerica", "min": float(s.min()), "max": float(s.max()),
                       "defecto": float(s.median())})
    for c in resultado["features_cat"]:
        op = sorted(ref[c].dropna().astype(str).unique().tolist())
        campos.append({"variable": c, "tipo": "categorica", "opciones": op, "defecto": ref[c].astype(str).mode().iloc[0]})
    return campos


def predecir_ensamble(miembros: list, X: pd.DataFrame, n_boot: int = 1000, semilla: int = C.SEED) -> dict[str, np.ndarray]:
    """Probabilidades por miembro (m x n x 4), media, mín, máx e IC bootstrap percentil 2,5-97,5 de la media."""
    P = np.stack([predecir_proba(m, X) for m in miembros], axis=0)
    rng = np.random.default_rng(semilla)
    m = P.shape[0]
    idx = rng.integers(0, m, size=(n_boot, m))
    medias = P[idx].mean(axis=1)                                          # (B, n, 4)
    return {"por_miembro": P, "media": P.mean(axis=0), "min": P.min(axis=0), "max": P.max(axis=0),
            "ic_inf": np.percentile(medias, 2.5, axis=0), "ic_sup": np.percentile(medias, 97.5, axis=0)}


def severidad(media: np.ndarray, umbral: float) -> dict[str, np.ndarray]:
    """Nivel por argmax (0-3), nivel con el umbral operativo y P(IA severa)."""
    return {"nivel_argmax": media.argmax(axis=1), "nivel_umbral": predecir_con_umbral(media, umbral),
            "p_severa": media[:, C.CLASE_SEVERA]}


def percentil_poblacional(valor: float, ref: np.ndarray, pesos: np.ndarray | None = None) -> float:
    """Fracción (ponderada) de la población de referencia con valor menor o igual al dado."""
    ref = np.asarray(ref, float)
    w = np.ones(len(ref)) if pesos is None else np.asarray(pesos, float)
    return float(w[ref <= valor].sum() / w.sum())


def fila_desde_formulario(valores: dict[str, Any], resultado: dict[str, Any]) -> pd.DataFrame:
    """DataFrame de 1 fila con tipos homogéneos (mismo preparador que el entrenamiento)."""
    df = pd.DataFrame([valores])
    return preparar_matriz(df, resultado["features_num"], resultado["features_cat"])


def validar_lote(df: pd.DataFrame, resultado: dict[str, Any]) -> pd.DataFrame:
    """Valida columnas y tamaño de un CSV de lote y devuelve la matriz de predictores."""
    req = resultado["features_num"] + resultado["features_cat"]
    faltan = [c for c in req if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en el CSV: {faltan}")
    if len(df) == 0 or len(df) > MAX_FILAS_LOTE:
        raise ValueError(f"El lote debe tener entre 1 y {MAX_FILAS_LOTE} filas.")
    return preparar_matriz(df, resultado["features_num"], resultado["features_cat"])


def predecir_lote(ens: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Predicción por lote: nivel, P por clase, IC de P(severa) y alerta con el umbral operativo."""
    X = validar_lote(df, ens["resultado"])
    r = predecir_ensamble(ens["miembros"], X, n_boot=300)
    s = severidad(r["media"], ens["umbral"])
    out = pd.DataFrame({f"p_clase_{k}": r["media"][:, k] for k in C.CLASES})
    out["nivel_predicho"] = s["nivel_argmax"]
    out["p_severa_ic_inf"] = r["ic_inf"][:, C.CLASE_SEVERA]
    out["p_severa_ic_sup"] = r["ic_sup"][:, C.CLASE_SEVERA]
    out["alerta_severa_umbral"] = (s["p_severa"] >= ens["umbral"]).astype(int)
    return out
