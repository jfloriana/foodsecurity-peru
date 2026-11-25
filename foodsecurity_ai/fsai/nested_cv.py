"""nested_cv.py - Fase 2: validación cruzada ANIDADA exacta dentro del 70 % de Desarrollo.

Protocolo (cada paso ocurre dentro de cada pliegue EXTERNO, 5 en total):
  (a) búsqueda Optuna (TPE) sobre los 3 pliegues INTERNOS de ``outer_train`` (GroupKFold por hogar_uid);
  (b) reentrenamiento con los mejores hiperparámetros sobre ``outer_train`` completo;
  (c) evaluación UNA sola vez sobre ``outer_val``.
Después se agregan los 5 resultados (una observación por pliegue externo): media, DE e IC t al 95 %.

Reparto de los datos (ver ``splits.py``): la CV anidada usa SOLO Desarrollo (70 %). Con el modelo final
(Optuna sobre todo Desarrollo + reentrenamiento en Desarrollo) el 15 % de Validación fija el umbral operativo
de la clase severa y calibra las probabilidades (isotónica uno-contra-resto). El 15 % de Prueba no se toca aquí.

Anti-fuga: imputación, escalado, SMOTE opcional, hiperparámetros y umbral se ajustan solo con filas de entrenamiento
del nivel correspondiente. Cada pliegue deja una auditoría (``auditoria``) verificable.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold

from fsai import config as C
from fsai.metrics_weighted import calcular_metricas, predecir_con_umbral, preparar_pesos_ajuste, umbral_clase_severa
from fsai.models import ajustar_pipeline, construir_pipeline, predecir_proba
from fsai.optuna_optimizer import optimizar_modelo, params_desde_estudio
from fsai.preprocessing import dividir_features, enriquecer_para_escenario, preparar_matriz
from fsai.splits import etiqueta_estrato
from fsai.umbrales import metricas_comparadas, optimizar_multiplicadores

logger = logging.getLogger(__name__)

METRICAS_AGREGADAS = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
                      "roc_auc_ovr", "pr_auc_macro", "mcc", "brier", "recall_severa", "ece"]


class RegistroProgreso:
    """Escribe ``progreso.json`` (estado en vivo para la UI) y ``log.txt`` de forma atómica."""

    def __init__(self, dir_run: Path) -> None:
        self.dir_run = dir_run
        dir_run.mkdir(parents=True, exist_ok=True)
        self.ruta_json = dir_run / "progreso.json"
        self.ruta_log = dir_run / "log.txt"
        self.estado: dict[str, Any] = {"estado": "iniciando", "trials": [], "inicio": datetime.now().isoformat(timespec="seconds")}

    def log(self, mensaje: str) -> None:
        linea = f"{datetime.now():%H:%M:%S} | {mensaje}"
        logger.info(mensaje)
        with self.ruta_log.open("a", encoding="utf-8") as f:
            f.write(linea + "\n")

    def actualizar(self, **kw: Any) -> None:
        self.estado.update(kw)
        self.estado["actualizado"] = datetime.now().isoformat(timespec="seconds")
        tmp = self.ruta_json.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.estado, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(tmp, self.ruta_json)

    def callback_optuna(self, etiqueta: str) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
        """Callback para ``study.optimize``: registra cada ensayo y el mejor valor acumulado."""
        def _cb(estudio: optuna.Study, t: optuna.trial.FrozenTrial) -> None:
            mejor = max((x.value for x in estudio.trials if x.value is not None), default=None)
            self.estado["trials"].append({"etapa": etiqueta, "numero": t.number, "valor": t.value,
                                          "estado": t.state.name, "mejor_hasta_ahora": mejor})
            self.estado["trial_actual"] = t.number + 1
            self.log(f"[{etiqueta}] ensayo {t.number}: valor={t.value if t.value is None else round(t.value, 4)} "
                     f"estado={t.state.name} mejor={None if mejor is None else round(mejor, 4)}")
            self.actualizar()
        return _cb


def ic_t(valores: np.ndarray, nivel: float = 0.95) -> dict[str, float]:
    """Media, DE y IC t sobre las observaciones (una por pliegue externo)."""
    v = np.asarray([x for x in valores if x is not None and not np.isnan(x)], dtype=float)
    n = len(v)
    if n == 0:
        return {"media": float("nan"), "sd": float("nan"), "n": 0, "ic_inf": float("nan"), "ic_sup": float("nan")}
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    if n > 1:
        h = float(stats.t.ppf(0.5 + nivel / 2, n - 1) * sd / np.sqrt(n))
        return {"media": m, "sd": sd, "n": n, "ic_inf": m - h, "ic_sup": m + h}
    return {"media": m, "sd": sd, "n": n, "ic_inf": float("nan"), "ic_sup": float("nan")}


def agregar_pliegues(folds: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """Agrega las 5 observaciones por métrica, ponderada y no ponderada."""
    out: dict[str, dict[str, dict[str, float]]] = {"ponderada": {}, "no_ponderada": {}}
    for clave in out:
        for m in METRICAS_AGREGADAS:
            out[clave][m] = ic_t(np.array([f["metricas"][clave].get(m, np.nan) for f in folds], dtype=float))
    return out


def estratos_dev(df_dev: pd.DataFrame) -> np.ndarray:
    return etiqueta_estrato(df_dev)


def calibrador_isotonico(y: np.ndarray, proba: np.ndarray, pesos: np.ndarray | None) -> dict[str, Any]:
    """Isotónica uno-contra-resto ajustada en Validación. Devuelve umbrales serializables (X_, y_ por clase)."""
    cal: dict[str, Any] = {"metodo": "isotonica_ovr", "clases": {}}
    for k in C.CLASES:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(proba[:, k], (y == k).astype(float), sample_weight=pesos)
        cal["clases"][str(k)] = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    return cal


def aplicar_calibrador(cal: dict[str, Any], proba: np.ndarray) -> np.ndarray:
    """Aplica el calibrador isotónico y renormaliza filas a suma 1."""
    out = np.column_stack([np.interp(proba[:, k], cal["clases"][str(k)]["x"], cal["clases"][str(k)]["y"])
                           for k in C.CLASES])
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s


def ejecutar_modelo(df: pd.DataFrame, part: dict[str, np.ndarray], *, codigo: str, escenario: str, n_trials: int,
                    dir_run: Path, progreso: RegistroProgreso, n_outer: int = C.N_SPLITS_EXTERNOS,
                    n_trials_final: int | None = None, espacio: dict | None = None, semilla: int = C.SEED,
                    ponderar_ajuste: bool = False, usar_smote: bool = False, ingenieria: bool = True,
                    recorte_pesos: float | None = None, normalizar_pesos: bool = False,
                    optimizar_umbrales: bool = False) -> dict[str, Any]:
    """CV anidada + modelo final + umbral/calibración para UN modelo y UN escenario.

    Opciones V2 (todas por defecto desactivadas/sin efecto sobre M1-M5 y los escenarios previos):
    ``ingenieria`` (solo ``socio_terr``: variables derivadas), ``recorte_pesos``/``normalizar_pesos`` (solo con
    ``ponderar_ajuste``: pesos de AJUSTE recortados en ese percentil y/o normalizados a media 1; las métricas usan pesos
    originales) y ``optimizar_umbrales`` (multiplicadores por clase para F1-macro, ajustados SOLO en Validación).
    """
    if C.INNER_FOLDS != 3 or n_outer != 5:
        logger.warning("Configuración no estándar: %d externos x %d internos (el protocolo fija 5 x 3).", n_outer, C.INNER_FOLDS)
    ident = f"{escenario}_{codigo}"
    (dir_run / "modelos").mkdir(parents=True, exist_ok=True)
    df = enriquecer_para_escenario(df, escenario, ingenieria=ingenieria)     # solo socio_terr agrega columnas
    df_dev = df.iloc[part["desarrollo"]].reset_index(drop=True)
    df_val = df.iloc[part["validacion"]].reset_index(drop=True)
    num, cat = dividir_features(df, escenario)
    X_dev, X_val = preparar_matriz(df_dev, num, cat), preparar_matriz(df_val, num, cat)
    y_dev, y_val = df_dev[C.COL_OBJETIVO].to_numpy(), df_val[C.COL_OBJETIVO].to_numpy()
    w_dev, w_val = df_dev[C.COL_PESO].to_numpy(), df_val[C.COL_PESO].to_numpy()
    w_fit_dev = preparar_pesos_ajuste(w_dev, recorte_pesos, normalizar_pesos)   # == w_dev si no se pide recorte/normalización
    g_dev = df_dev[C.COL_GRUPO].to_numpy()
    uid_ajeno = set(df.iloc[np.concatenate([part["validacion"], part["prueba"]])][C.COL_GRUPO])
    assert not (set(g_dev) & uid_ajeno), "Fuga: Desarrollo comparte hogar_uid con Validación/Prueba"

    cv = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=semilla)
    folds: list[dict[str, Any]] = []
    filas_trials: list[dict[str, Any]] = []
    t_modelo = time.time()
    for k, (ot, ov) in enumerate(cv.split(X_dev, estratos_dev(df_dev), g_dev)):
        t0 = time.time()
        progreso.actualizar(fase="cv_externa", fold=k + 1, n_folds=n_outer, trial_actual=0, n_trials=n_trials,
                            mensaje=f"{ident}: pliegue externo {k + 1}/{n_outer} - búsqueda Optuna en 3 pliegues internos")
        progreso.log(f"{ident} externo {k + 1}/{n_outer}: outer_train={len(ot)} outer_val={len(ov)}")
        sol = len(set(g_dev[ot]) & set(g_dev[ov]))
        assert sol == 0, "Fuga: hogar_uid compartido entre outer_train y outer_val"
        # (a) Optuna SOLO con outer_train (CV interna GroupKFold(3) dentro de optimizar_modelo)
        estudio = optimizar_modelo(
            codigo, X_dev.iloc[ot], y_dev[ot], g_dev[ot], pesos=w_dev[ot], n_trials=n_trials, features_num=num,
            features_cat=cat, usar_smote=usar_smote, ponderar_ajuste=ponderar_ajuste, nombre_estudio=None,
            storage_url="", espacio=espacio, semilla=semilla + k, pesos_ajuste=w_fit_dev[ot],
            callbacks=[progreso.callback_optuna(f"{ident} f{k + 1}")])
        mejores = params_desde_estudio(estudio, codigo)
        # (b) reentrenar con los mejores hiperparámetros sobre outer_train
        progreso.actualizar(mensaje=f"{ident}: reentrenando pliegue {k + 1} con mejores hiperparámetros")
        pipe = construir_pipeline(codigo, mejores, num, cat, usar_smote)
        ajustar_pipeline(pipe, X_dev.iloc[ot], y_dev[ot], None if (usar_smote or not ponderar_ajuste) else w_fit_dev[ot], g_dev[ot])
        # (c) evaluar UNA vez sobre outer_val
        proba = predecir_proba(pipe, X_dev.iloc[ov])
        met = {"ponderada": calcular_metricas(y_dev[ov], proba, w_dev[ov]),
               "no_ponderada": calcular_metricas(y_dev[ov], proba, None)}
        joblib.dump(pipe, dir_run / "modelos" / f"outer_{ident}_f{k}.joblib")
        folds.append({"pliegue": k, "n_outer_train": int(len(ot)), "n_outer_val": int(len(ov)),
                      "mejor_valor_interno": float(estudio.best_value), "mejores_params": mejores,
                      "n_trials": len(estudio.trials), "metricas": met, "segundos": round(time.time() - t0, 1),
                      "auditoria": {"inner_n_splits": C.INNER_FOLDS, "optuna_solo_con_outer_train": True,
                                    "evaluaciones_outer_val": 1, "hogar_uid_compartidos_train_val": sol,
                                    "prueba_tocada": False, "n_ensayos_registrados": len(estudio.trials)}})
        for t in estudio.trials:
            filas_trials.append({"pliegue": k, "numero": t.number, "valor": t.value, "estado": t.state.name,
                                 "duracion_s": t.duration.total_seconds() if t.duration else None,
                                 **{f"p_{a}": b for a, b in t.params.items()}})
        progreso.log(f"{ident} externo {k + 1}: F1-macro outer_val (pond.)={met['ponderada']['f1_macro']:.4f}")
    agregado = agregar_pliegues(folds)
    pd.DataFrame(filas_trials).to_csv(dir_run / f"trials_{ident}.csv", index=False)

    # Modelo final: Optuna sobre TODO Desarrollo, reentrenar en Desarrollo; umbral/calibración en Validación
    progreso.actualizar(fase="modelo_final", fold=None, trial_actual=0, n_trials=n_trials_final or n_trials,
                        mensaje=f"{ident}: búsqueda final sobre todo Desarrollo")
    est_f = optimizar_modelo(
        codigo, X_dev, y_dev, g_dev, pesos=w_dev, n_trials=n_trials_final or n_trials, features_num=num, features_cat=cat,
        usar_smote=usar_smote, ponderar_ajuste=ponderar_ajuste, nombre_estudio=None, storage_url="", espacio=espacio,
        semilla=semilla + 99, pesos_ajuste=w_fit_dev, callbacks=[progreso.callback_optuna(f"{ident} final")])
    mejores_f = params_desde_estudio(est_f, codigo)
    pipe_f = construir_pipeline(codigo, mejores_f, num, cat, usar_smote)
    ajustar_pipeline(pipe_f, X_dev, y_dev, None if (usar_smote or not ponderar_ajuste) else w_fit_dev, g_dev)
    joblib.dump(pipe_f, dir_run / "modelos" / f"final_{ident}.joblib")
    progreso.actualizar(fase="validacion", mensaje=f"{ident}: umbral operativo y calibración en Validación (15 %)")
    p_val = predecir_proba(pipe_f, X_val)
    umbral = umbral_clase_severa(y_val, p_val, w_val)
    cal = calibrador_isotonico(y_val, p_val, w_val)
    p_val_cal = aplicar_calibrador(cal, p_val)
    pred_u = predecir_con_umbral(p_val, umbral)
    sev = y_val == C.CLASE_SEVERA
    val = {"n": int(len(y_val)), "metricas_ponderadas": calcular_metricas(y_val, p_val, w_val),
           "metricas_ponderadas_calibradas": calcular_metricas(y_val, p_val_cal, w_val),
           "umbral_severa": float(umbral),
           "recall_severa_umbral": float(np.average(pred_u[sev] == C.CLASE_SEVERA, weights=w_val[sev])) if sev.any() else None,
           "especificidad_severa_umbral": float(np.average(pred_u[~sev] != C.CLASE_SEVERA, weights=w_val[~sev]))}
    umbrales_clase = None
    if optimizar_umbrales:       # multiplicadores por clase: SOLO Validación (probabilidades calibradas); Prueba no se toca
        umbrales_clase = optimizar_multiplicadores(y_val, p_val_cal, w_val)
        umbrales_clase["validacion_ponderada"] = metricas_comparadas(y_val, p_val_cal, w_val, umbrales_clase["multiplicadores"])
        progreso.log(f"{ident}: F1-macro Validación argmax={umbrales_clase['f1_argmax_val']:.4f} -> optimizado={umbrales_clase['f1_optimizado_val']:.4f} "
                     f"(m={np.round(umbrales_clase['multiplicadores'], 2).tolist()})")
    np.savez_compressed(dir_run / f"pred_val_{ident}.npz", uid=df_val[C.COL_GRUPO].to_numpy().astype(str), y=y_val, proba=p_val, w=w_val)
    from fsai.preprocessing import DESCARTADAS_100_NAN
    resultado = {"id": ident, "escenario": escenario, "codigo": codigo, "features_num": num, "features_cat": cat,
                 "features_descartadas_100pct_nan": list(DESCARTADAS_100_NAN.get(escenario, [])),
                 "n_trials": n_trials, "n_trials_final": n_trials_final or n_trials, "n_outer": n_outer,
                 "n_inner": C.INNER_FOLDS, "semilla": semilla, "folds": folds, "agregado": agregado,
                 "final": {"hiperparametros": mejores_f, "mejor_valor_interno": float(est_f.best_value),
                           "modelo": f"modelos/final_{ident}.joblib", "calibrador": cal, "validacion": val,
                           **({"umbrales_clase": umbrales_clase} if umbrales_clase else {})},
                 "opciones_v2": {"ingenieria": bool(ingenieria and escenario == "socio_terr"), "ponderar_ajuste": ponderar_ajuste,
                                 "recorte_pesos_percentil": recorte_pesos, "normalizar_pesos": normalizar_pesos,
                                 "optimizar_umbrales": optimizar_umbrales},
                 "segundos_total": round(time.time() - t_modelo, 1),
                 "nota": ("CV anidada dentro del 70 % de Desarrollo; umbral y calibración con el 15 % de Validación; "
                          "el 15 % de Prueba no se usó.")}
    (dir_run / f"resultado_{ident}.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    progreso.log(f"{ident}: completado en {resultado['segundos_total']} s")
    return resultado


def ejecutar_fase2(df: pd.DataFrame, part: dict[str, np.ndarray], *, modelos: list[str], escenarios: list[str],
                   n_trials: int, dir_run: Path, **kw: Any) -> list[dict[str, Any]]:
    """Recorre escenarios x modelos y escribe el progreso global. Devuelve los resultados."""
    progreso = RegistroProgreso(dir_run)
    resultados = []
    total = len(modelos) * len(escenarios)
    i = 0
    try:
        for esc in escenarios:
            for cod in modelos:
                i += 1
                progreso.actualizar(estado="ejecutando", modelo=cod, escenario=esc, combinacion=i, n_combinaciones=total,
                                    trials=progreso.estado["trials"])
                resultados.append(ejecutar_modelo(df, part, codigo=cod, escenario=esc, n_trials=n_trials,
                                                  dir_run=dir_run, progreso=progreso, **kw))
        progreso.actualizar(estado="completado", fase="fin", mensaje="CV anidada completada. La Prueba (15 %) sigue intacta.")
    except Exception as exc:
        progreso.log(f"ERROR: {type(exc).__name__}: {exc}")
        progreso.actualizar(estado="error", mensaje=f"{type(exc).__name__}: {exc}")
        raise
    return resultados


def curva_aprendizaje(df: pd.DataFrame, part: dict[str, np.ndarray], dir_run: Path, ident: str,
                      fracciones: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0), semilla: int = C.SEED) -> pd.DataFrame:
    """Curva de aprendizaje: F1-macro ponderado en entrenamiento y en VALIDACIÓN (nunca Prueba) según la fracción de Desarrollo.

    Usa los hiperparámetros del modelo final y submuestrea hogares completos (grupos) de Desarrollo.
    """
    from sklearn.metrics import f1_score
    res = json.loads((dir_run / f"resultado_{ident}.json").read_text(encoding="utf-8"))
    df = enriquecer_para_escenario(df, res["escenario"], ingenieria=any(c in res["features_num"] for c in C.FEATURES_INGENIERIA_NUM))
    num, cat, params, codigo = res["features_num"], res["features_cat"], res["final"]["hiperparametros"], res["codigo"]
    dev = df.iloc[part["desarrollo"]].reset_index(drop=True)
    val = df.iloc[part["validacion"]].reset_index(drop=True)
    Xd, Xv = preparar_matriz(dev, num, cat), preparar_matriz(val, num, cat)
    yd, yv = dev[C.COL_OBJETIVO].to_numpy(), val[C.COL_OBJETIVO].to_numpy()
    wd, wv = dev[C.COL_PESO].to_numpy(), val[C.COL_PESO].to_numpy()
    uid = dev[C.COL_GRUPO].to_numpy()
    orden = np.random.default_rng(semilla).permutation(pd.unique(uid))
    filas = []
    for fr in fracciones:
        usar = set(orden[: max(int(len(orden) * fr), 20)])
        m = np.isin(uid, list(usar))
        pipe = construir_pipeline(codigo, params, num, cat, False)
        ajustar_pipeline(pipe, Xd[m], yd[m], None, uid[m])
        f = lambda X, y, w: float(f1_score(y, predecir_proba(pipe, X).argmax(1), labels=C.CLASES, average="macro", zero_division=0, sample_weight=w))  # noqa: E731
        filas.append({"fraccion": fr, "n_entrenamiento": int(m.sum()), "f1_entrenamiento": f(Xd[m], yd[m], wd[m]), "f1_validacion": f(Xv, yv, wv)})
    return pd.DataFrame(filas)
