"""metrics_weighted.py - Métricas de desempeño con ponderación muestral (factor_expansion).

Todas las métricas aceptan ``pesos``; con ``pesos=None`` se calculan sin ponderar.
Las probabilidades deben tener 4 columnas alineadas con CLASES = [0,1,2,3].
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             matthews_corrcoef, precision_recall_fscore_support, roc_auc_score)

from fsai.config import CLASE_SEVERA, CLASES, ESPECIFICIDAD_MIN

logger = logging.getLogger(__name__)

NOMBRES_METRICAS = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
                    "roc_auc_ovr", "pr_auc_macro", "mcc", "brier", "recall_severa", "ece"]


def _validar(y_true, y_proba, pesos):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_proba, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(CLASES):
        raise ValueError(f"y_proba debe tener forma (n, {len(CLASES)}); recibió {p.shape}")
    if len(y) != len(p):
        raise ValueError("y_true y y_proba difieren en longitud")
    if np.isnan(p).any():
        raise ValueError("y_proba contiene NaN")
    w = None if pesos is None else np.asarray(pesos, dtype=float)
    if w is not None and (len(w) != len(y) or (w < 0).any()):
        raise ValueError("pesos inválidos (longitud o valores negativos)")
    return y, p, w


def brier_multiclase(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    """Brier multiclase: media ponderada de sum_k (p_k - 1[y=k])^2."""
    onehot = np.eye(len(CLASES))[y]
    return float(np.average(((p - onehot) ** 2).sum(axis=1), weights=w))


def ece_top_label(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None, n_bins: int = 15) -> float:
    """Error de calibración esperado (ECE) de la etiqueta más probable, con bins de igual ancho."""
    w = np.ones(len(y)) if w is None else w
    conf = p.max(axis=1)
    acierto = (p.argmax(axis=1) == y).astype(float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = w.sum()
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            peso_b = w[m].sum() / total
            ece += peso_b * abs(np.average(acierto[m], weights=w[m]) - np.average(conf[m], weights=w[m]))
    return float(ece)


def recall_clase(y: np.ndarray, pred: np.ndarray, clase: int, w: np.ndarray | None = None) -> float:
    """Sensibilidad (TPR) ponderada de una clase en esquema uno-contra-resto."""
    pos = y == clase
    if not pos.any():
        return float("nan")
    return float(np.average((pred[pos] == clase).astype(float), weights=None if w is None else w[pos]))


def calcular_metricas(y_true, y_proba, pesos=None) -> dict[str, float]:
    """Calcula las 11 métricas del contrato (con pesos si se proveen)."""
    y, p, w = _validar(y_true, y_proba, pesos)
    pred = p.argmax(axis=1)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=CLASES, average="macro",
                                                       zero_division=0, sample_weight=w)
    aucs, aps = [], []
    for k in CLASES:   # uno-contra-resto; se omiten clases ausentes en y (AUC indefinido)
        yk = (y == k).astype(int)
        if 0 < yk.sum() < len(yk):
            aucs.append(roc_auc_score(yk, p[:, k], sample_weight=w))
            aps.append(average_precision_score(yk, p[:, k], sample_weight=w))
    if len(aucs) < len(CLASES):
        logger.warning("Se promediaron %d de %d clases en AUC/PR-AUC (clases ausentes).", len(aucs), len(CLASES))
    return {
        "accuracy": float(accuracy_score(y, pred, sample_weight=w)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred, sample_weight=w)),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "roc_auc_ovr": float(np.mean(aucs)) if aucs else float("nan"),
        "pr_auc_macro": float(np.mean(aps)) if aps else float("nan"),
        "mcc": float(matthews_corrcoef(y, pred, sample_weight=w)),
        "brier": brier_multiclase(y, p, w),
        "recall_severa": recall_clase(y, pred, CLASE_SEVERA, w),
        "ece": ece_top_label(y, p, w),
    }


def umbral_clase_severa(y_train, p_train, pesos=None, especificidad_min: float = ESPECIFICIDAD_MIN) -> float:
    """Umbral sobre P(clase severa) que maximiza la sensibilidad con especificidad >= mínimo.

    Debe estimarse SOLO con datos de entrenamiento (idealmente predicciones fuera de
    muestra de la CV interna) y aplicarse después, sin reajuste, al conjunto de prueba.
    """
    y = (np.asarray(y_train).astype(int) == CLASE_SEVERA)
    s = np.asarray(p_train, dtype=float)[:, CLASE_SEVERA]
    w = np.ones(len(y)) if pesos is None else np.asarray(pesos, dtype=float)
    if y.all() or not y.any():
        raise ValueError("Se requieren ejemplos de clase severa y de otras clases para fijar el umbral.")
    mejor_u, mejor_sens = 1.0, -1.0
    for u in np.unique(s):
        pred = s >= u
        esp = w[~y & ~pred].sum() / w[~y].sum()
        sens = w[y & pred].sum() / w[y].sum()
        if esp >= especificidad_min and sens > mejor_sens:
            mejor_u, mejor_sens = float(u), float(sens)
    if mejor_sens < 0:
        logger.warning("Ningún umbral cumple especificidad >= %.2f; se usa 0,5.", especificidad_min)
        return 0.5
    return mejor_u


def predecir_con_umbral(y_proba, umbral_severa: float) -> np.ndarray:
    """Etiquetas: severa si P(severa) >= umbral; si no, la mejor de las otras tres clases."""
    p = np.asarray(y_proba, dtype=float)
    pred = np.delete(p, CLASE_SEVERA, axis=1).argmax(axis=1)
    pred = np.array([c if c < CLASE_SEVERA else c + 1 for c in pred])
    pred[p[:, CLASE_SEVERA] >= umbral_severa] = CLASE_SEVERA
    return pred


def preparar_pesos_ajuste(pesos, recorte_percentil: float | None = None, normalizar: bool = False) -> np.ndarray | None:
    """Pesos muestrales PARA AJUSTAR modelos: recorte opcional en un percentil y normalización a media 1 (ambos OPCIONALES).

    Por defecto (``recorte_percentil=None``, ``normalizar=False``) devuelve los pesos sin cambios (comportamiento histórico).
    - ``recorte_percentil`` (p. ej. 99): ``w = min(w, percentil_p(w))`` reduce el efecto de pesos extremos (menor varianza,
      algo de sesgo). Los pesos de diseño no dependen del objetivo, por lo que calcular el percentil sobre el conjunto de
      Desarrollo no introduce fuga de etiquetas.
    - ``normalizar``: reescala a media 1 (no cambia el ajuste de modelos invariantes a la escala; sí el de los regularizados).
    Las MÉTRICAS siguen usando los pesos originales (``factor_expansion``); esto solo afecta a ``sample_weight`` del ajuste.
    """
    if pesos is None:
        return None
    w = np.asarray(pesos, dtype=float).copy()
    if (w < 0).any() or np.isnan(w).any():
        raise ValueError("pesos inválidos (negativos o NaN)")
    if recorte_percentil is not None:
        if not 50.0 <= recorte_percentil <= 100.0:
            raise ValueError("recorte_percentil debe estar entre 50 y 100")
        w = np.minimum(w, np.percentile(w, recorte_percentil))
    if normalizar:
        m = w.mean()
        if m <= 0:
            raise ValueError("los pesos suman 0; no se pueden normalizar")
        w = w / m
    return w
