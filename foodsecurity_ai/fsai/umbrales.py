"""umbrales.py - Pesos/umbrales de decisión por clase optimizados para F1-macro, ajustados SOLO en Validación.

Decisión: ``pred = argmax_k (m_k * p_k)`` con multiplicadores ``m`` (m_0 = 1 fijo: la decisión es invariante a la escala global).
Los multiplicadores se buscan por búsqueda coordenada (una clase a la vez sobre una malla geométrica, varias rondas) para
maximizar el F1-macro (ponderado por ``factor_expansion`` si se dan pesos) sobre las probabilidades CALIBRADAS de Validación.
Luego se aplican UNA vez a Prueba junto con el argmax simple, y se reportan ambos.

Anti-fuga (garantizada por la firma, verificada en tests):
* ``optimizar_multiplicadores`` solo recibe etiquetas/probabilidades/pesos de VALIDACIÓN; no existe parámetro para Prueba.
* ``aplicar_multiplicadores`` solo recibe probabilidades.
* ``evaluar_prueba_con_umbrales`` es el ÚNICO lugar donde se leen etiquetas de Prueba y solo para MEDIR (los multiplicadores
  vienen congelados de Validación); escribe un registro y se niega a repetirse salvo ``permitir_repetir=True``.

Limitación declarada: las probabilidades calibradas de Validación se ajustaron con esa misma Validación (isotónica), de modo que
la ganancia medida en Validación es optimista; la ganancia que importa es la de Prueba.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from fsai.config import CLASES

logger = logging.getLogger(__name__)

MALLA_DEFECTO: np.ndarray = np.geomspace(0.25, 4.0, 17)   # x0,25 ... x4 (17 puntos, incluye 1,0)


def aplicar_multiplicadores(proba, multiplicadores) -> np.ndarray:
    """Clase predicha = argmax de ``p * m`` (solo usa probabilidades; nunca etiquetas)."""
    p = np.asarray(proba, dtype=float)
    m = np.asarray(multiplicadores, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(m):
        raise ValueError(f"proba debe tener {len(m)} columnas; tiene {p.shape}")
    if (m <= 0).any():
        raise ValueError("los multiplicadores deben ser > 0")
    return np.asarray(CLASES)[np.argmax(p * m, axis=1)]


def _f1m(y, pred, w) -> float:
    return float(f1_score(y, pred, labels=CLASES, average="macro", zero_division=0, sample_weight=w))


def optimizar_multiplicadores(y_val, proba_val, pesos_val=None, *, malla=None, n_rondas: int = 3,
                              mejora_min: float = 1e-4) -> dict[str, Any]:
    """Búsqueda coordenada de multiplicadores por clase que maximizan el F1-macro en VALIDACIÓN.

    Un cambio solo se acepta si mejora el F1 al menos ``mejora_min`` (evita sobreajuste por empates mínimos). Determinista.
    Devuelve {'multiplicadores', 'f1_argmax_val', 'f1_optimizado_val', 'rondas', 'malla'}.
    """
    y = np.asarray(y_val).astype(int)
    p = np.asarray(proba_val, dtype=float)
    w = None if pesos_val is None else np.asarray(pesos_val, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(CLASES) or len(p) != len(y):
        raise ValueError(f"proba_val debe ser (n, {len(CLASES)}) con n = len(y_val)")
    if np.isnan(p).any():
        raise ValueError("proba_val contiene NaN")
    malla = MALLA_DEFECTO if malla is None else np.asarray(malla, dtype=float)
    m = np.ones(len(CLASES))
    f_base = f_actual = _f1m(y, aplicar_multiplicadores(p, m), w)
    rondas = 0
    for _ in range(n_rondas):
        rondas += 1
        cambio = False
        for k in CLASES[1:]:                       # la clase 0 queda fija en 1
            mejor_v, mejor_f = m[k], f_actual
            for v in malla:
                cand = m.copy()
                cand[k] = v
                f = _f1m(y, aplicar_multiplicadores(p, cand), w)
                if f > mejor_f + mejora_min:
                    mejor_v, mejor_f = float(v), f
            if mejor_v != m[k]:
                m[k], f_actual, cambio = mejor_v, mejor_f, True
        if not cambio:
            break
    return {"multiplicadores": [float(x) for x in m], "f1_argmax_val": f_base, "f1_optimizado_val": f_actual,
            "rondas": rondas, "malla": [float(x) for x in malla], "mejora_min": mejora_min,
            "nota": "ajustado solo con Validación; no se usó Prueba"}


def metricas_comparadas(y, proba, pesos, multiplicadores) -> dict[str, Any]:
    """F1-macro, recall y precisión macro, exactitud balanceada: argmax simple vs multiplicadores (mismos datos)."""
    from sklearn.metrics import balanced_accuracy_score
    y = np.asarray(y).astype(int)
    w = None if pesos is None else np.asarray(pesos, dtype=float)
    out = {}
    for nombre, m in (("argmax", np.ones(len(CLASES))), ("optimizado", np.asarray(multiplicadores, dtype=float))):
        pred = aplicar_multiplicadores(proba, m)
        out[nombre] = {"f1_macro": _f1m(y, pred, w),
                       "recall_macro": float(f1_score(y, pred, labels=CLASES, average=None, zero_division=0, sample_weight=w).mean()),
                       "balanced_accuracy": float(balanced_accuracy_score(y, pred, sample_weight=w)),
                       "f1_por_clase": [float(x) for x in f1_score(y, pred, labels=CLASES, average=None, zero_division=0, sample_weight=w)]}
    return out


def evaluar_prueba_con_umbrales(df, part: dict[str, np.ndarray], dir_run: Path, ident: str, *, permitir_repetir: bool = False) -> dict[str, Any]:
    """Aplica UNA vez a Prueba el modelo final + calibrador + multiplicadores guardados en ``resultado_<ident>.json``.

    Requiere que ``ejecutar_modelo(..., optimizar_umbrales=True)`` haya fijado los multiplicadores en Validación. Escribe
    ``prueba_<ident>.json`` y se niega a ejecutarse de nuevo (Prueba es de un solo uso) salvo ``permitir_repetir=True``.
    """
    import joblib
    from fsai import config as C
    from fsai.nested_cv import aplicar_calibrador
    from fsai.models import predecir_proba
    from fsai.preprocessing import enriquecer_para_escenario, preparar_matriz
    salida = Path(dir_run) / f"prueba_{ident}.json"
    if salida.exists() and not permitir_repetir:
        raise RuntimeError(f"{salida.name} ya existe: la Prueba se evalúa una sola vez (use permitir_repetir=True bajo su responsabilidad)")
    res = json.loads((Path(dir_run) / f"resultado_{ident}.json").read_text(encoding="utf-8"))
    um = res["final"].get("umbrales_clase")
    if not um:
        raise ValueError(f"{ident}: no hay multiplicadores; ejecute con optimizar_umbrales=True")
    df_e = enriquecer_para_escenario(df, res["escenario"], ingenieria=any(c in res["features_num"] for c in C.FEATURES_INGENIERIA_NUM))
    prueba = df_e.iloc[part["prueba"]].reset_index(drop=True)
    X = preparar_matriz(prueba, res["features_num"], res["features_cat"])
    pipe = joblib.load(Path(dir_run) / res["final"]["modelo"])
    p = aplicar_calibrador(res["final"]["calibrador"], predecir_proba(pipe, X))
    y, w = prueba[C.COL_OBJETIVO].to_numpy(), prueba[C.COL_PESO].to_numpy()
    reg = {"id": ident, "n_prueba": int(len(y)), "multiplicadores": um["multiplicadores"],
           "ponderada": metricas_comparadas(y, p, w, um["multiplicadores"]),
           "no_ponderada": metricas_comparadas(y, p, None, um["multiplicadores"]),
           "nota": "Prueba evaluada una vez; multiplicadores congelados desde Validación."}
    salida.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")
    return reg
