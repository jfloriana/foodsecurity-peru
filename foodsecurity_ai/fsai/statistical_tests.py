"""statistical_tests.py - Comparación estadística de modelos.

DeLong (algoritmo rápido por rangos medios, equivalente a la estructura de componentes de
DeLong et al., 1988), Friedman + Nemenyi, Wilcoxon pareado, McNemar ponderado con
corrección de continuidad, bootstrap estratificado por clase y corrección de Holm.
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from fsai.config import ALPHA, N_BOOT, SEED

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# DeLong
# --------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    """Rangos medios (empates reciben el promedio de sus posiciones), base 1."""
    orden = np.argsort(x)
    z = x[orden]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(n, dtype=float)
    out[orden] = t + 1
    return out


def _fast_delong(preds_ordenadas: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """AUCs y matriz de covarianza de k clasificadores. Columnas: m positivos, luego negativos."""
    n = preds_ordenadas.shape[1] - m
    k = preds_ordenadas.shape[0]
    pos, neg = preds_ordenadas[:, :m], preds_ordenadas[:, m:]
    tx, ty, tz = np.empty((k, m)), np.empty((k, n)), np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(preds_ordenadas[r])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n            # componentes estructurales de los positivos
    v10 = 1.0 - (tz[:, m:] - ty) / m      # componentes estructurales de los negativos
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_test(y_true, p1, p2) -> dict[str, float]:
    """Contraste bilateral de AUC correlacionadas (mismos sujetos) de dos clasificadores binarios.

    ``y_true`` en {0,1}; ``p1`` y ``p2`` son puntajes/probabilidades de la clase positiva.
    NO incorpora pesos muestrales (la teoría de DeLong supone muestras simples):
    para comparaciones ponderadas use ``bootstrap_estratificado``.
    """
    y = np.asarray(y_true).astype(int)
    if not set(np.unique(y)) <= {0, 1} or y.min() == y.max():
        raise ValueError("DeLong requiere y binaria con ambas clases presentes.")
    a, b = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    if not (len(y) == len(a) == len(b)):
        raise ValueError("y_true, p1 y p2 deben tener igual longitud.")
    orden = np.argsort(-y, kind="stable")           # positivos primero
    m = int(y.sum())
    aucs, cov = _fast_delong(np.vstack([a, b])[:, orden], m)
    var_dif = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    dif = float(aucs[0] - aucs[1])
    if var_dif <= 0:
        z, p = (0.0, 1.0) if dif == 0 else (float("inf"), 0.0)
    else:
        z = dif / np.sqrt(var_dif)
        p = float(2 * stats.norm.sf(abs(z)))
    ee = float(np.sqrt(max(var_dif, 0.0)))
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diferencia": dif, "error_estandar": ee,
            "z": float(z), "p_valor": p, "ic_inf": dif - 1.959964 * ee, "ic_sup": dif + 1.959964 * ee}


def delong_ovr(y_true, P1: np.ndarray, P2: np.ndarray, clase: int) -> dict[str, float]:
    """DeLong uno-contra-resto para una clase de un problema multiclase (p. ej. la severa)."""
    y = (np.asarray(y_true).astype(int) == clase).astype(int)
    return delong_test(y, np.asarray(P1)[:, clase], np.asarray(P2)[:, clase])


# --------------------------------------------------------------------------
# Friedman + Nemenyi, Wilcoxon, McNemar
# --------------------------------------------------------------------------
def friedman_nemenyi(matriz, nombres: list[str] | None = None, alpha: float = ALPHA,
                     mayor_es_mejor: bool = True) -> dict:
    """Friedman sobre una matriz (n_bloques x k_modelos) y post hoc de Nemenyi.

    Los bloques típicos son los pliegues (o repeticiones) de validación y la medida, F1-macro.
    Con pocos bloques (p. ej. 5) la potencia es baja: interprételo con cautela.
    """
    M = np.asarray(matriz, dtype=float)
    if M.ndim != 2 or M.shape[1] < 3 or M.shape[0] < 2:
        raise ValueError("Friedman requiere >= 2 bloques y >= 3 modelos.")
    if np.isnan(M).any():
        raise ValueError("La matriz contiene NaN.")
    n, k = M.shape
    nombres = nombres or [f"M{i + 1}" for i in range(k)]
    chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    rangos = np.vstack([stats.rankdata(-fila if mayor_es_mejor else fila) for fila in M]).mean(axis=0)
    se = np.sqrt(k * (k + 1) / (12.0 * n))
    p_nem = np.ones((k, k))
    for i, j in combinations(range(k), 2):
        q = abs(rangos[i] - rangos[j]) / se
        p_nem[i, j] = p_nem[j, i] = float(stats.studentized_range.sf(q, k, np.inf))
    dc = float(stats.studentized_range.ppf(1 - alpha, k, np.inf) * se)   # diferencia crítica
    return {"chi2": float(chi2), "p_valor": float(p), "rangos_medios": dict(zip(nombres, rangos.round(4))),
            "nemenyi_p": pd.DataFrame(p_nem, index=nombres, columns=nombres), "diferencia_critica": dc,
            "n_bloques": n, "k_modelos": k}


def wilcoxon_pareado(a, b) -> dict[str, float]:
    """Wilcoxon de rangos con signo para dos medidas pareadas (bilateral).

    Con n=5 pliegues el p-valor exacto mínimo bilateral es 2/32 = 0,0625 > 0,05: no puede declararse
    significancia. Por eso V1/V1c se repiten N_REPETICIONES=4 veces y el pipeline entrega 5x4 = 20 pares
    (p-valor exacto mínimo 2/2^20, aprox. 1,9e-6).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("Las series pareadas deben tener la misma longitud.")
    d = a - b
    if np.allclose(d, 0):
        return {"estadistico": 0.0, "p_valor": 1.0, "n": int(len(a)), "mediana_diferencia": 0.0,
                "p_minimo_posible": float(2 / 2 ** len(a))}
    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"estadistico": float(res.statistic), "p_valor": float(res.pvalue), "n": int(len(a)),
            "mediana_diferencia": float(np.median(d)), "p_minimo_posible": float(2 / 2 ** len(a))}


def mcnemar_ponderado(y, pred1, pred2, pesos=None, n_efectivo_kish: bool = True) -> dict[str, float]:
    """McNemar con corrección de continuidad sobre aciertos/errores discordantes, ponderado.

    b: modelo 1 acierta y 2 falla; c: modelo 2 acierta y 1 falla. Estadístico
    (|b-c|-1)^2/(b+c) ~ chi2(1). Si ``n_efectivo_kish``, los pesos se reescalan para sumar el
    tamaño muestral efectivo de Kish, evitando inflar la significancia con el factor de expansión.
    """
    y, p1, p2 = (np.asarray(v).astype(int) for v in (y, pred1, pred2))
    w = np.ones(len(y)) if pesos is None else np.asarray(pesos, dtype=float)
    if not (len(y) == len(p1) == len(p2) == len(w)):
        raise ValueError("Longitudes inconsistentes en mcnemar_ponderado.")
    if pesos is not None and n_efectivo_kish:
        w = w * (w.sum() / (w ** 2).sum())          # suma de pesos = (sum w)^2 / sum w^2
    ok1, ok2 = p1 == y, p2 == y
    b = float(w[ok1 & ~ok2].sum())
    c = float(w[~ok1 & ok2].sum())
    if b + c == 0:
        return {"b": b, "c": c, "estadistico": 0.0, "p_valor": 1.0}
    est = (abs(b - c) - 1.0) ** 2 / (b + c) if abs(b - c) > 1.0 else 0.0
    return {"b": b, "c": c, "estadistico": float(est), "p_valor": float(stats.chi2.sf(est, 1))}


# --------------------------------------------------------------------------
# Bootstrap estratificado y Holm
# --------------------------------------------------------------------------
def bootstrap_estratificado(y, p, fn: Callable, n_boot: int = N_BOOT, pesos=None,
                            alpha: float = ALPHA, seed: int = SEED) -> dict:
    """Bootstrap que remuestrea DENTRO de cada clase de ``y`` y devuelve IC percentil.

    ``fn(y, p)`` (o ``fn(y, p, pesos)`` si se entregan pesos) debe devolver un escalar.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    w = None if pesos is None else np.asarray(pesos, dtype=float)
    rng = np.random.default_rng(seed)
    por_clase = [np.flatnonzero(y == c) for c in np.unique(y)]

    def evaluar(idx: np.ndarray) -> float:
        return float(fn(y[idx], p[idx]) if w is None else fn(y[idx], p[idx], w[idx]))

    estimado = evaluar(np.arange(len(y)))
    reps = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in por_clase])
        try:
            reps.append(evaluar(idx))
        except ValueError:      # réplica degenerada (p. ej. métrica indefinida)
            continue
    reps_a = np.array(reps)
    if len(reps_a) < 0.9 * n_boot:
        logger.warning("Bootstrap: solo %d/%d réplicas válidas.", len(reps_a), n_boot)
    lo, hi = np.nanpercentile(reps_a, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimado": estimado, "ic_inf": float(lo), "ic_sup": float(hi),
            "error_estandar": float(np.nanstd(reps_a, ddof=1)), "n_validas": int(len(reps_a))}


def holm(pvalores) -> np.ndarray:
    """Corrección de Holm (1979) por pasos descendentes; devuelve p-valores ajustados."""
    p = np.asarray(pvalores, dtype=float)
    m = len(p)
    orden = np.argsort(p)
    ajustados = np.empty(m)
    maximo = 0.0
    for rango, idx in enumerate(orden):
        maximo = max(maximo, (m - rango) * p[idx])
        ajustados[idx] = min(1.0, maximo)
    return ajustados
