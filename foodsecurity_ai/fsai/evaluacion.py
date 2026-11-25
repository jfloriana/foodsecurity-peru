"""evaluacion.py - Fase 3: evaluación ÚNICA sobre la Prueba (15 %), métricas con IC, curvas y pruebas estadísticas.

Regla de oro: la Prueba se evalúa UNA sola vez por (corrida, escenario, modelo). La primera evaluación crea un
archivo-candado exclusivo (``prueba_<id>.lock``, modo ``x``) y guarda las predicciones (``pred_prueba_<id>.npz``) y
un registro (``prueba_<id>.json``). Toda tabla, curva o prueba posterior se calcula a partir de esas predicciones
guardadas, sin volver a invocar al modelo sobre la Prueba. Un segundo intento lanza ``PruebaYaEvaluada``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (auc, average_precision_score, confusion_matrix, precision_recall_curve, roc_curve)

from fsai import config as C
from fsai.metrics_weighted import brier_multiclase, calcular_metricas, ece_top_label, predecir_con_umbral
from fsai.nested_cv import aplicar_calibrador
from fsai.preprocessing import preparar_matriz
from fsai.splits import cargar_particion
from fsai.statistical_tests import delong_ovr, friedman_nemenyi, holm, mcnemar_ponderado, wilcoxon_pareado

logger = logging.getLogger(__name__)


class PruebaYaEvaluada(RuntimeError):
    """La Prueba (15 %) ya fue evaluada para este (corrida, escenario, modelo): no se permite repetirla."""


# ------------------------------ corridas ------------------------------
def listar_corridas() -> list[str]:
    """Corridas de Fase 2 con CV anidada completada (más recientes primero)."""
    if not C.DIR_FASES.is_dir():
        return []
    d = [p for p in C.DIR_FASES.iterdir() if p.is_dir() and any(p.glob("resultado_*.json"))]
    return [p.name for p in sorted(d, key=lambda q: q.stat().st_mtime, reverse=True)]


def dir_corrida(nombre: str) -> Path:
    r = (C.DIR_FASES / nombre).resolve()
    if C.DIR_FASES.resolve() not in r.parents or not r.is_dir():
        raise ValueError("Corrida inexistente o fuera de la carpeta de fases.")
    return r


def resultados_corrida(nombre: str) -> dict[str, dict[str, Any]]:
    """{id: resultado_json} de una corrida."""
    out = {}
    for p in sorted(dir_corrida(nombre).glob("resultado_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out[r["id"]] = r
    return out


def config_corrida(nombre: str) -> dict[str, Any]:
    return json.loads((dir_corrida(nombre) / "config.json").read_text(encoding="utf-8"))


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------ evaluación única ------------------------------
def registro_prueba(dir_run: Path, ident: str) -> dict[str, Any] | None:
    p = dir_run / f"prueba_{ident}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def evaluar_prueba_una_vez(df: pd.DataFrame, dir_run: Path, ident: str, usuario: str = "") -> dict[str, Any]:
    """Evalúa el modelo final de ``ident`` sobre la Prueba. Segunda llamada -> ``PruebaYaEvaluada``."""
    resultado = json.loads((dir_run / f"resultado_{ident}.json").read_text(encoding="utf-8"))
    lock = dir_run / f"prueba_{ident}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)       # exclusivo: bloquea la repetición
    except FileExistsError as exc:
        raise PruebaYaEvaluada(f"La Prueba ya fue evaluada para {ident}: {registro_prueba(dir_run, ident)}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps({"inicio": datetime.now().isoformat(timespec="seconds"), "usuario": usuario}))
        part, carga = cargar_particion(df, dir_run / "particion.json")
        te = df.iloc[part["prueba"]].reset_index(drop=True)
        X = preparar_matriz(te, resultado["features_num"], resultado["features_cat"])
        ruta_modelo = dir_run / resultado["final"]["modelo"]
        pipe = joblib.load(ruta_modelo)
        from fsai.models import predecir_proba
        proba = predecir_proba(pipe, X)                                  # ÚNICA predicción sobre la Prueba
        np.savez_compressed(dir_run / f"pred_prueba_{ident}.npz", uid=te[C.COL_GRUPO].to_numpy().astype(str),
                            y=te[C.COL_OBJETIVO].to_numpy(), proba=proba, w=te[C.COL_PESO].to_numpy())
        reg = {"id": ident, "evaluado_en": datetime.now().isoformat(timespec="seconds"), "usuario": usuario,
               "huella_particion": carga["huella"], "sha256_modelo": _sha256(ruta_modelo), "n_prueba": int(len(te)),
               "umbral_severa_validacion": resultado["final"]["validacion"]["umbral_severa"]}
        (dir_run / f"prueba_{ident}.json").write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")
        return reg
    except Exception:
        lock.unlink(missing_ok=True)       # fallo técnico ANTES de guardar predicciones: no hubo evaluación
        raise


def cargar_prediccion_prueba(dir_run: Path, ident: str) -> dict[str, np.ndarray] | None:
    p = dir_run / f"pred_prueba_{ident}.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ------------------------------ métricas con IC ------------------------------
def metricas_con_ic(y, proba, w, n_boot: int = 200, semilla: int = C.SEED) -> pd.DataFrame:
    """Las 11 métricas del contrato, ponderadas y no ponderadas, con IC bootstrap percentil estratificado por clase."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(semilla)
    por_clase = [np.flatnonzero(y == c) for c in np.unique(y)]
    filas = []
    for pond in (True, False):
        wv = w if pond else None
        base = calcular_metricas(y, proba, wv)
        reps: dict[str, list[float]] = {k: [] for k in base}
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in por_clase])
            try:
                m = calcular_metricas(y[idx], proba[idx], None if wv is None else wv[idx])
            except ValueError:
                continue
            for k, v in m.items():
                reps[k].append(v)
        for k, v in base.items():
            r = np.array(reps[k], dtype=float)
            lo, hi = (np.nanpercentile(r, [2.5, 97.5]) if len(r) > 10 else (np.nan, np.nan))
            filas.append({"metrica": k, "ponderada": pond, "valor": v, "ic_inf": float(lo), "ic_sup": float(hi), "n_boot": len(r)})
    return pd.DataFrame(filas)


def matriz_confusion(y, pred, w=None, normalizar: bool = False) -> np.ndarray:
    cm = confusion_matrix(y, pred, labels=C.CLASES, sample_weight=w)
    if normalizar:
        s = cm.sum(axis=1, keepdims=True).astype(float)
        s[s == 0] = 1.0
        return cm / s
    return cm


def foco_severa(y, proba, w, umbral: float) -> dict[str, float]:
    """Sensibilidad/especificidad/VPP de IA severa con argmax y con el umbral operativo de Validación."""
    y = np.asarray(y).astype(int)
    sev = y == C.CLASE_SEVERA
    out: dict[str, float] = {}
    for nombre, pred in (("argmax", proba.argmax(1)), ("umbral_validacion", predecir_con_umbral(proba, umbral))):
        ps = pred == C.CLASE_SEVERA
        out[f"sensibilidad_{nombre}"] = float(np.average(ps[sev], weights=w[sev])) if sev.any() else float("nan")
        out[f"especificidad_{nombre}"] = float(np.average(~ps[~sev], weights=w[~sev]))
        out[f"vpp_{nombre}"] = float(np.average(sev[ps], weights=w[ps])) if ps.any() else float("nan")
    return out


def curvas_roc_pr(y, proba) -> dict[str, Any]:
    """ROC y PR uno-contra-resto por clase + ROC macro (interpolada) y AUC/AP por clase."""
    y = np.asarray(y).astype(int)
    out: dict[str, Any] = {"roc": {}, "pr": {}, "auc": {}, "ap": {}, "prevalencia": {}}
    grid = np.linspace(0, 1, 101)
    tprs = []
    for k in C.CLASES:
        yk = (y == k).astype(int)
        if yk.sum() == 0 or yk.sum() == len(yk):
            continue
        fpr, tpr, _ = roc_curve(yk, proba[:, k])
        pr, rc, _ = precision_recall_curve(yk, proba[:, k])
        out["roc"][k] = (fpr, tpr)
        out["pr"][k] = (rc, pr)
        out["auc"][k] = float(auc(fpr, tpr))
        out["ap"][k] = float(average_precision_score(yk, proba[:, k]))
        out["prevalencia"][k] = float(yk.mean())
        tprs.append(np.interp(grid, fpr, tpr))
    if tprs:
        out["roc_macro"] = (grid, np.mean(tprs, axis=0))
        out["auc_macro"] = float(auc(grid, np.mean(tprs, axis=0)))
    return out


def curva_calibracion(y, proba, clase: int, n_bins: int = 10) -> pd.DataFrame:
    """Confiabilidad uno-contra-resto con bins por cuantiles: probabilidad media predicha vs frecuencia observada."""
    yk = (np.asarray(y).astype(int) == clase).astype(float)
    p = proba[:, clase]
    cortes = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(cortes) < 3:
        return pd.DataFrame(columns=["p_predicha", "frecuencia", "n"])
    b = np.clip(np.searchsorted(cortes, p, side="right") - 1, 0, len(cortes) - 2)
    filas = [{"p_predicha": float(p[b == i].mean()), "frecuencia": float(yk[b == i].mean()), "n": int((b == i).sum())}
             for i in range(len(cortes) - 1) if (b == i).any()]
    return pd.DataFrame(filas)


def calibracion_resumen(y, proba, w, calibrador: dict | None) -> dict[str, float | None]:
    y = np.asarray(y).astype(int)
    out = {"brier": brier_multiclase(y, proba, w), "ece": ece_top_label(y, proba, w), "brier_calibrado": None, "ece_calibrado": None}
    if calibrador:
        pc = aplicar_calibrador(calibrador, proba)
        out["brier_calibrado"], out["ece_calibrado"] = brier_multiclase(y, pc, w), ece_top_label(y, pc, w)
    return out


# ------------------------------ pruebas entre modelos ------------------------------
def pruebas_entre_modelos(dir_run: Path, escenario: str, codigos: list[str]) -> pd.DataFrame:
    """DeLong (severa), McNemar ponderado, Wilcoxon y Friedman+Nemenyi (F1 por pliegue externo); Holm por familia."""
    preds, folds = {}, {}
    for c in codigos:
        ident = f"{escenario}_{c}"
        p = cargar_prediccion_prueba(dir_run, ident)
        if p is not None:
            preds[c] = p
        folds[c] = [f["metricas"]["ponderada"]["f1_macro"] for f in
                    json.loads((dir_run / f"resultado_{ident}.json").read_text(encoding="utf-8"))["folds"]]
    filas: list[dict[str, Any]] = []
    fam: dict[str, list[dict[str, Any]]] = {"delong_severa": [], "mcnemar": [], "wilcoxon_pliegues": []}
    for a, b in combinations(codigos, 2):
        if a in preds and b in preds:
            y, w = preds[a]["y"], preds[a]["w"]
            r = delong_ovr(y, preds[a]["proba"], preds[b]["proba"], C.CLASE_SEVERA)
            fam["delong_severa"].append({"comparacion": f"{a}-{b}", "estadistico": r["z"], "p_valor": r["p_valor"],
                                         "detalle": f"AUC {a}={r['auc1']:.3f}; {b}={r['auc2']:.3f}"})
            r = mcnemar_ponderado(y, preds[a]["proba"].argmax(1), preds[b]["proba"].argmax(1), w)
            fam["mcnemar"].append({"comparacion": f"{a}-{b}", "estadistico": r["estadistico"], "p_valor": r["p_valor"],
                                   "detalle": f"b={r['b']:.1f}; c={r['c']:.1f}"})
        r = wilcoxon_pareado(folds[a], folds[b])
        fam["wilcoxon_pliegues"].append({"comparacion": f"{a}-{b}", "estadistico": r["estadistico"], "p_valor": r["p_valor"],
                                         "detalle": f"n={r['n']} pliegues; p mínimo posible={r['p_minimo_posible']:.4f}"})
    for prueba, lista in fam.items():
        if lista:
            aj = holm([f["p_valor"] for f in lista])
            filas += [{"prueba": prueba, **f, "p_ajustado": float(p)} for f, p in zip(lista, aj)]
    if len(codigos) >= 3:
        M = np.column_stack([folds[c] for c in codigos])
        fr = friedman_nemenyi(M, codigos)
        filas.append({"prueba": "friedman", "comparacion": "global", "estadistico": fr["chi2"], "p_valor": fr["p_valor"],
                      "p_ajustado": fr["p_valor"], "detalle": f"rangos medios={fr['rangos_medios']}"})
        for a, b in combinations(codigos, 2):
            pn = float(fr["nemenyi_p"].loc[a, b])
            filas.append({"prueba": "nemenyi", "comparacion": f"{a}-{b}", "estadistico": None, "p_valor": pn, "p_ajustado": pn, "detalle": ""})
    return pd.DataFrame(filas)
