"""optuna_optimizer.py - Optimización bayesiana (TPE) con poda y persistencia en PostgreSQL/SQLite.

Criterio de optimización: F1-macro PONDERADO (sample_weight = factor_expansion) en una
validación cruzada interna GroupKFold(3) por hogar_uid, calculada SOLO sobre las filas
de entrenamiento que se le entregan. El preprocesado se reajusta en cada pliegue interno.
Los estudios y ensayos se guardan con el almacenamiento RDB nativo de Optuna
(DATABASE_URL o una URL SQLite); sus tablas son gestionadas por Optuna y no forman parte de las
12 del esquema fsai.

Fase 7 (Optuna UI): el espacio de búsqueda es una tabla (``ESPACIO_DEFECTO``) que la interfaz puede
editar DENTRO de los rangos válidos (``espacio_efectivo`` recorta cualquier rango al valor por defecto);
``optimizar_modelo`` admite además ``callbacks``, ``timeout``, semilla, sampler y pruner declarados.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from fsai.config import CLASES, DATABASE_URL, INNER_FOLDS, N_TRIALS, SEED
from fsai.models import ajustar_pipeline, construir_pipeline, predecir_proba

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

CONFIGS_CAPAS: dict[str, list[int]] = {"128-64": [128, 64], "256-128": [256, 128], "64-32": [64, 32]}

SAMPLERS = ("tpe",)          # TPE multivariante no se expone: el protocolo fija TPE
PRUNERS = ("mediana", "ninguno")


def _f(lo: float, hi: float, log: bool = False) -> dict[str, Any]:
    return {"tipo": "float", "lo": lo, "hi": hi, "log": log}


def _i(lo: int, hi: int, step: int = 1, log: bool = False) -> dict[str, Any]:
    return {"tipo": "int", "lo": lo, "hi": hi, "step": step, "log": log}


def _c(*opciones: Any) -> dict[str, Any]:
    return {"tipo": "cat", "opciones": list(opciones)}


# Espacio de búsqueda por defecto (sección 7 del protocolo). El ORDEN de los parámetros es el de muestreo.
ESPACIO_DEFECTO: dict[str, dict[str, dict[str, Any]]] = {
    "M1": {"C": _f(1e-3, 1e2, True), "penalty": _c("l2", "l1"), "class_weight": _c("balanced", "ninguno")},
    "M2": {"n_estimators": _i(100, 600, 50), "max_depth": _i(4, 24), "min_samples_leaf": _i(1, 30, 1, True)},
    "M3": {"depth": _i(4, 10), "learning_rate": _f(0.01, 0.3, True), "iterations": _i(200, 800, 100)},
    "M4": {"cb_depth": _i(4, 8), "cb_learning_rate": _f(0.02, 0.3, True), "cb_iterations": _i(100, 400, 100),
           "rf_n_estimators": _i(100, 400, 100), "rf_max_depth": _i(4, 16),
           "rf_min_samples_leaf": _i(1, 20, 1, True), "lr_C": _f(1e-3, 1e2, True), "C_meta": _f(1e-2, 1e2, True)},
    "M5": {"capas_cfg": _c(*CONFIGS_CAPAS), "dim_latente": _c(8, 16, 32), "dropout": _f(0.1, 0.4),
           "lr_ae": _f(1e-4, 1e-2, True), "modo_entrada": _c("x_z", "z"), "n_estimators": _i(100, 600, 50),
           "max_depth": _i(3, 10), "learning_rate": _f(0.01, 0.3, True), "subsample": _f(0.6, 1.0),
           "colsample_bytree": _f(0.5, 1.0)},
    # M6 (extra opcional): HistGradientBoosting; no altera M1-M5
    "M6": {"learning_rate": _f(0.02, 0.3, True), "max_iter": _i(100, 500, 50), "max_depth": _i(3, 12),
           "max_leaf_nodes": _i(15, 63), "min_samples_leaf": _i(10, 100, 1, True),
           "l2_regularization": _f(1e-3, 10.0, True), "class_weight": _c("balanced", "ninguno")},
}


def espacio_efectivo(codigo: str, espacio: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """Combina el espacio por defecto con ajustes del usuario RECORTADOS a los rangos válidos.

    ``espacio[param]`` puede traer ``lo``/``hi`` (numéricos) u ``opciones`` (subconjunto de las válidas).
    Un rango fuera del dominio se recorta; un rango vacío o invertido lanza ``ValueError``.
    """
    if codigo not in ESPACIO_DEFECTO:
        raise ValueError(f"Código de modelo desconocido: {codigo}")
    base = copy.deepcopy(ESPACIO_DEFECTO[codigo])
    for nombre, ajuste in (espacio or {}).items():
        if nombre not in base:
            raise ValueError(f"Hiperparámetro '{nombre}' no existe en {codigo}.")
        spec = base[nombre]
        if spec["tipo"] == "cat":
            ops = [o for o in ajuste.get("opciones", spec["opciones"]) if o in spec["opciones"]]
            if not ops:
                raise ValueError(f"'{nombre}': ninguna opción válida seleccionada.")
            spec["opciones"] = ops
        else:
            lo = max(spec["lo"], ajuste.get("lo", spec["lo"]))
            hi = min(spec["hi"], ajuste.get("hi", spec["hi"]))
            if lo > hi:
                raise ValueError(f"'{nombre}': rango vacío tras recortar a [{spec['lo']}, {spec['hi']}].")
            if spec["tipo"] == "int":
                lo, hi = int(round(lo)), int(round(hi))
            spec["lo"], spec["hi"] = lo, hi
    return base


def _sugerir(trial: optuna.Trial, nombre: str, spec: dict[str, Any]) -> Any:
    if spec["tipo"] == "cat":
        return trial.suggest_categorical(nombre, spec["opciones"])
    if spec["tipo"] == "int":
        if spec["lo"] == spec["hi"]:
            return trial.suggest_int(nombre, spec["lo"], spec["hi"])
        paso = 1 if spec.get("log") else spec.get("step", 1)
        return trial.suggest_int(nombre, spec["lo"], spec["hi"], step=paso, log=spec.get("log", False))
    if spec["lo"] == spec["hi"]:
        return trial.suggest_float(nombre, spec["lo"], spec["hi"])
    return trial.suggest_float(nombre, spec["lo"], spec["hi"], log=spec.get("log", False))


def sugerir_params(trial: optuna.Trial, codigo: str, espacio: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Espacio de búsqueda por modelo (valores JSON-serializables). ``espacio`` ajusta rangos (ver arriba)."""
    spec = espacio_efectivo(codigo, espacio)
    p = {nombre: _sugerir(trial, nombre, s) for nombre, s in spec.items()}
    if codigo == "M5":                        # la configuración de capas se guarda como lista
        p["capas"] = CONFIGS_CAPAS[p.pop("capas_cfg")]
    return p


def params_desde_estudio(estudio: optuna.Study, codigo: str) -> dict[str, Any]:
    """Mejores hiperparámetros listos para ``crear_modelo`` (M5: reconstruye ``capas``)."""
    p = dict(estudio.best_params)
    if codigo == "M5" and "capas_cfg" in p:
        p["capas"] = CONFIGS_CAPAS[p.pop("capas_cfg")]
    return p


def _inferir_features(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    return num, [c for c in X.columns if c not in num]


def crear_estudio(nombre: str | None, storage_url: str | None, semilla: int = SEED, pruner: str = "mediana",
                  sampler: str = "tpe", n_startup: int = 10) -> optuna.Study:
    """Crea (o reanuda si ya existe) un estudio ``maximize`` con TPE y el pruner declarado."""
    if sampler not in SAMPLERS or pruner not in PRUNERS:
        raise ValueError(f"sampler/pruner no admitido: {sampler}/{pruner}")
    storage = optuna.storages.RDBStorage(url=storage_url, engine_kwargs={"pool_pre_ping": True}) if storage_url else None
    pr = (optuna.pruners.MedianPruner(n_startup_trials=n_startup, n_warmup_steps=1) if pruner == "mediana"
          else optuna.pruners.NopPruner())
    return optuna.create_study(study_name=nombre, storage=storage, load_if_exists=bool(storage and nombre),
                               direction="maximize", sampler=optuna.samplers.TPESampler(seed=semilla), pruner=pr)


def optimizar_modelo(codigo: str, X: pd.DataFrame, y, grupos, pesos=None, n_trials: int = N_TRIALS,
                     *, features_num: list[str] | None = None, features_cat: list[str] | None = None,
                     usar_smote: bool = False, ponderar_ajuste: bool = True,
                     nombre_estudio: str | None = None, storage_url: str | None = None,
                     espacio: dict[str, dict[str, Any]] | None = None,
                     callbacks: list[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]] | None = None,
                     timeout: float | None = None, semilla: int = SEED, pruner: str = "mediana",
                     sampler: str = "tpe", pesos_ajuste=None) -> optuna.Study:
    """Ejecuta el estudio TPE de un modelo y devuelve el ``optuna.Study``.

    Parameters
    ----------
    X : DataFrame de PREDICTORES crudos (sin imputar/escalar) de las filas de entrenamiento.
    grupos : hogar_uid de esas filas (define los pliegues internos GroupKFold(INNER_FOLDS)).
    ponderar_ajuste : si False, los pesos solo ponderan el F1 de validación (no el ajuste).
    pesos_ajuste : (opcional) pesos ya recortados/normalizados para el AJUSTE (ver ``metrics_weighted.preparar_pesos_ajuste``);
        si es None se usan ``pesos``. El F1 de validación siempre usa ``pesos`` originales.
    nombre_estudio : si se da junto con almacenamiento, el estudio se reanuda (load_if_exists).
    espacio / callbacks / timeout / semilla / pruner / sampler : controles de la Fase 7 (todos opcionales).
    """
    y = np.asarray(y).astype(int)
    grupos = np.asarray(grupos)
    w = None if pesos is None else np.asarray(pesos, dtype=float)
    w_fit = w if pesos_ajuste is None else np.asarray(pesos_ajuste, dtype=float)
    if w_fit is not None and len(w_fit) != len(y):
        raise ValueError("pesos_ajuste debe tener la misma longitud que y.")
    if not (len(X) == len(y) == len(grupos)):
        raise ValueError("X, y y grupos deben tener la misma longitud.")
    if features_num is None or features_cat is None:
        features_num, features_cat = _inferir_features(X)
    # storage_url=None -> DATABASE_URL (comportamiento histórico); storage_url="" -> estudio en memoria
    url = (DATABASE_URL if storage_url is None else storage_url) or None
    if url is None:
        logger.info("Estudio Optuna en memoria (sin almacenamiento persistente).")
    estudio = crear_estudio(nombre_estudio, url, semilla, pruner, sampler)
    hechos = len([t for t in estudio.trials if t.state in (optuna.trial.TrialState.COMPLETE,
                                                           optuna.trial.TrialState.PRUNED)])
    restantes = max(n_trials - hechos, 0)
    pliegues = list(GroupKFold(n_splits=INNER_FOLDS).split(X, y, grupos))
    espacio_efectivo(codigo, espacio)                              # falla pronto si el rango es inválido

    def objetivo(trial: optuna.Trial) -> float:
        params = sugerir_params(trial, codigo, espacio)
        valores: list[float] = []
        for paso, (tr, va) in enumerate(pliegues):
            pipe = construir_pipeline(codigo, params, features_num, features_cat, usar_smote)
            w_ajuste = None if (w_fit is None or usar_smote or not ponderar_ajuste) else w_fit[tr]  # SMOTE no admite pesos
            ajustar_pipeline(pipe, X.iloc[tr], y[tr], w_ajuste, grupos[tr])
            pred = predecir_proba(pipe, X.iloc[va]).argmax(axis=1)
            valores.append(f1_score(y[va], pred, labels=CLASES, average="macro", zero_division=0,
                                    sample_weight=None if w is None else w[va]))
            trial.report(float(np.mean(valores)), paso)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(valores))

    logger.info("Optuna %s: %d ensayos nuevos (%d previos)", codigo, restantes, hechos)
    if restantes:
        estudio.optimize(objetivo, n_trials=restantes, timeout=timeout, catch=(Exception,), gc_after_trial=True,
                         callbacks=callbacks)
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in estudio.trials):
        raise RuntimeError(f"Ningún ensayo completó para {codigo}; revise los registros de error.")
    return estudio


def trials_a_dataframe(estudio: optuna.Study) -> pd.DataFrame:
    """Tabla de ensayos (valor, estado, hiperparámetros) para auditoría."""
    return estudio.trials_dataframe(attrs=("number", "value", "state", "params", "duration"))
