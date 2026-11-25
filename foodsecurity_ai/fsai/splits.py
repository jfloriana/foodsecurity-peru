"""splits.py - Partición Desarrollo 70 % / Validación 15 % / Prueba 15 % (Fase 2) y validación de grupos (Fase 1).

Cómo encaja con el resto del protocolo (regla fija):
* La validación cruzada ANIDADA (5 externos x 3 internos) se ejecuta SOLO dentro del 70 % de Desarrollo.
* El 15 % de Validación sirve para elegir el umbral operativo de la clase severa, calibrar probabilidades y
  (en M5) la parada temprana, sin tocar la Prueba.
* El 15 % de Prueba NO se usa hasta la Fase 3 y se evalúa UNA sola vez (ver ``evaluacion.py``).

La partición es agrupada (un ``hogar_uid`` nunca cruza particiones; opcionalmente ``conglome``) y
estratificada por ``fies_nivel`` x ``area``. Se genera con ``StratifiedGroupKFold(20)``: 14 pliegues de
~5 % forman Desarrollo, 3 Validación y 3 Prueba. La partición se guarda con una huella SHA-256.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold

from fsai.config import (COL_GRUPO, COL_OBJETIVO, COL_PESO, NOMBRES_PARTICION, PROPORCIONES_PARTICION, SEED)

logger = logging.getLogger(__name__)

N_PLIEGUES_BASE = 20          # 20 pliegues de ~5 %: 14 + 3 + 3 = 70/15/15


def etiqueta_estrato(df: pd.DataFrame) -> np.ndarray:
    """Etiqueta entera fies_nivel x area (U=0, R=1) usada para estratificar."""
    area = (df["area"].astype(str) == "R").astype(int).to_numpy() if "area" in df.columns else np.zeros(len(df), int)
    return df[COL_OBJETIVO].astype(int).to_numpy() * 2 + area


def particionar_70_15_15(df: pd.DataFrame, semilla: int = SEED, agrupar_por: str = COL_GRUPO) -> dict[str, np.ndarray]:
    """Devuelve índices POSICIONALES por partición: {'desarrollo','validacion','prueba'} (disjuntos y exhaustivos)."""
    if agrupar_por not in df.columns:
        raise KeyError(f"No existe la columna de agrupación '{agrupar_por}'.")
    if len(df) < 200:
        raise ValueError("Se requieren al menos 200 hogares para una partición 70/15/15 estratificada.")
    y = etiqueta_estrato(df)
    g = df[agrupar_por].to_numpy()
    cv = StratifiedGroupKFold(n_splits=N_PLIEGUES_BASE, shuffle=True, random_state=semilla)
    pliegues = [te for _, te in cv.split(np.zeros((len(df), 1)), y, g)]
    n_val = round(N_PLIEGUES_BASE * PROPORCIONES_PARTICION[1])
    n_pru = round(N_PLIEGUES_BASE * PROPORCIONES_PARTICION[2])
    n_des = N_PLIEGUES_BASE - n_val - n_pru
    partes = {NOMBRES_PARTICION[0]: pliegues[:n_des], NOMBRES_PARTICION[1]: pliegues[n_des:n_des + n_val],
              NOMBRES_PARTICION[2]: pliegues[n_des + n_val:]}
    return {k: np.sort(np.concatenate(v)) for k, v in partes.items()}


def huella_particion(df: pd.DataFrame, part: dict[str, np.ndarray], semilla: int, agrupar_por: str) -> str:
    """SHA-256 de (semilla, agrupación, hogar_uid ordenados por partición)."""
    h = hashlib.sha256(f"{semilla}|{agrupar_por}".encode())
    uid = df[COL_GRUPO].astype(str).to_numpy()
    for nombre in NOMBRES_PARTICION:
        h.update(nombre.encode())
        h.update("\n".join(sorted(uid[part[nombre]])).encode())
    return h.hexdigest()


def verificar_particion(df: pd.DataFrame, part: dict[str, np.ndarray]) -> dict[str, Any]:
    """Comprobaciones duras: exhaustividad, disjunción de índices y cruces de hogar_uid / conglome.

    ``cruces_hogar_uid`` debe ser 0 siempre. ``cruces_conglome`` es 0 solo si se agrupó por conglome; si se
    agrupó por hogar_uid (V1) hogares del mismo conglomerado pueden caer en particiones distintas (es lo que
    mide V1c como sensibilidad).
    """
    n = len(df)
    todos = np.concatenate([part[k] for k in NOMBRES_PARTICION])
    res: dict[str, Any] = {"n_total": n, "exhaustiva": bool(len(todos) == n and len(np.unique(todos)) == n),
                           "tamanos": {k: int(len(part[k])) for k in NOMBRES_PARTICION},
                           "proporciones": {k: round(len(part[k]) / n, 4) for k in NOMBRES_PARTICION}}
    for col, clave in ((COL_GRUPO, "cruces_hogar_uid"), ("conglome", "cruces_conglome")):
        if col not in df.columns:
            res[clave] = None
            continue
        conj = {k: set(df[col].astype(str).to_numpy()[part[k]]) for k in NOMBRES_PARTICION}
        pares = {}
        a, b, c = NOMBRES_PARTICION
        for x, y in ((a, b), (a, c), (b, c)):
            pares[f"{x}-{y}"] = len(conj[x] & conj[y])
        res[clave] = pares
        res[clave + "_total"] = int(sum(pares.values()))
    return res


def guardar_particion(df: pd.DataFrame, part: dict[str, np.ndarray], ruta: Path, semilla: int, agrupar_por: str,
                      extra: dict | None = None) -> str:
    """Guarda hogar_uid por partición + huella en JSON. Devuelve la huella."""
    huella = huella_particion(df, part, semilla, agrupar_por)
    uid = df[COL_GRUPO].astype(str).to_numpy()
    carga = {"semilla": semilla, "agrupar_por": agrupar_por, "huella": huella,
             "proporciones": list(PROPORCIONES_PARTICION), "verificacion": verificar_particion(df, part),
             "hogar_uid": {k: uid[part[k]].tolist() for k in NOMBRES_PARTICION}, **(extra or {})}
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(carga, ensure_ascii=False), encoding="utf-8")
    return huella


def cargar_particion(df: pd.DataFrame, ruta: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reconstruye los índices a partir de hogar_uid y comprueba la huella (falla si el dataset cambió)."""
    carga = json.loads(ruta.read_text(encoding="utf-8"))
    pos = {u: i for i, u in enumerate(df[COL_GRUPO].astype(str).to_numpy())}
    try:
        part = {k: np.sort(np.array([pos[u] for u in carga["hogar_uid"][k]], dtype=np.int64)) for k in NOMBRES_PARTICION}
    except KeyError as exc:
        raise ValueError("La partición guardada contiene hogares que no están en el dataset actual.") from exc
    if huella_particion(df, part, carga["semilla"], carga["agrupar_por"]) != carga["huella"]:
        raise ValueError("La huella de la partición no coincide: el dataset o la partición cambiaron.")
    return part, carga


# --------------------------------------------------------------------------
# Balance entre particiones (Fase 1: validación de grupos)
# --------------------------------------------------------------------------
def _kish(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w ** 2).sum())


def chi2_ponderado(categoria: pd.Series, particion: pd.Series, pesos: pd.Series | None = None) -> dict[str, float]:
    """Chi-cuadrado de homogeneidad categoría x partición.

    Con ``pesos`` la tabla suma los pesos y se reescala al tamaño efectivo de Kish para no inflar la
    significancia con el factor de expansión (misma idea que ``mcnemar_ponderado``).
    """
    if pesos is None:
        tabla = pd.crosstab(categoria, particion)
    else:
        w = pesos.to_numpy(dtype=float)
        tabla = pd.crosstab(categoria, particion, values=w, aggfunc="sum").fillna(0.0)
        tabla = tabla * (_kish(w) / tabla.to_numpy().sum())
    tabla = tabla.loc[(tabla.sum(axis=1) > 0), (tabla.sum(axis=0) > 0)]
    if tabla.shape[0] < 2 or tabla.shape[1] < 2:
        return {"chi2": float("nan"), "gl": 0, "p_valor": float("nan")}
    chi2, p, gl, _ = stats.chi2_contingency(tabla.to_numpy(), correction=False)
    return {"chi2": float(chi2), "gl": int(gl), "p_valor": float(p)}


def ks_ponderado(x: np.ndarray, y: np.ndarray, wx: np.ndarray | None = None, wy: np.ndarray | None = None) -> dict[str, float]:
    """KS de dos muestras con CDF ponderadas; p-valor asintótico con tamaños efectivos de Kish."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    wx = np.ones(len(x)) if wx is None else np.asarray(wx, float)
    wy = np.ones(len(y)) if wy is None else np.asarray(wy, float)
    mx, my = ~np.isnan(x), ~np.isnan(y)
    x, wx, y, wy = x[mx], wx[mx], y[my], wy[my]
    if len(x) < 2 or len(y) < 2:
        return {"D": float("nan"), "p_valor": float("nan")}
    malla = np.sort(np.concatenate([x, y]))
    ox, oy = np.argsort(x), np.argsort(y)
    cx = np.concatenate([[0], np.cumsum(wx[ox]) / wx.sum()])
    cy = np.concatenate([[0], np.cumsum(wy[oy]) / wy.sum()])
    fx = cx[np.searchsorted(x[ox], malla, side="right")]
    fy = cy[np.searchsorted(y[oy], malla, side="right")]
    d = float(np.max(np.abs(fx - fy)))
    n_e = _kish(wx) * _kish(wy) / (_kish(wx) + _kish(wy))
    p = float(stats.kstwo.sf(d, max(int(round(n_e)), 1)))
    return {"D": d, "p_valor": min(max(p, 0.0), 1.0)}


def balance_entre_particiones(df: pd.DataFrame, part: dict[str, np.ndarray], categoricas: list[str],
                              numericas: list[str], ponderar: bool = True) -> pd.DataFrame:
    """Tabla de balance: chi-cuadrado (categóricas) y KS por pares (numéricas) entre las 3 particiones."""
    etiqueta = np.empty(len(df), dtype=object)
    for k in NOMBRES_PARTICION:
        etiqueta[part[k]] = k
    et = pd.Series(etiqueta, index=df.index)
    pesos = df[COL_PESO] if (ponderar and COL_PESO in df.columns) else None
    filas = []
    for c in categoricas:
        if c in df.columns:
            r = chi2_ponderado(df[c].astype(str), et, pesos)
            filas.append({"variable": c, "prueba": "chi² ponderada (Kish)" if pesos is not None else "chi²", "comparacion": "3 particiones",
                          "estadistico": r["chi2"], "p_valor": r["p_valor"]})
    a, b, c3 = NOMBRES_PARTICION
    for v in numericas:
        if v not in df.columns:
            continue
        x = pd.to_numeric(df[v], errors="coerce").to_numpy()
        w = df[COL_PESO].to_numpy() if pesos is not None else None
        for p1, p2 in ((a, b), (a, c3)):
            r = ks_ponderado(x[part[p1]], x[part[p2]], None if w is None else w[part[p1]], None if w is None else w[part[p2]])
            filas.append({"variable": v, "prueba": "KS ponderada (Kish)" if w is not None else "KS", "comparacion": f"{p1} vs {p2}",
                          "estadistico": r["D"], "p_valor": r["p_valor"]})
    return pd.DataFrame(filas)


def composicion_por_particion(df: pd.DataFrame, part: dict[str, np.ndarray], columna: str) -> pd.DataFrame:
    """Proporción de cada categoría de ``columna`` dentro de cada partición (filas = categoría)."""
    cols = {}
    for k in NOMBRES_PARTICION:
        cols[k] = df[columna].astype(str).iloc[part[k]].value_counts(normalize=True)
    return pd.DataFrame(cols).fillna(0.0).sort_index()
