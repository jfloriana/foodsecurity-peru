"""validation.py - Esquemas de validación de la sección 8.1 con particiones guardadas.

V1  GroupKFold(5) por hogar_uid, repetido N_REPETICIONES veces (semillas SEED+rep) -> 5x4 = 20 bloques
V1c sensibilidad de V1: misma estructura, grupos = conglome
V2  validación geográfica: variante "lodo" (dejar fuera un departamento) y "macro" (costa/sierra/selva)
V3  partición estratificada urbano/rural (5 pliegues) + 2 de transferencia (U->R, R->U); métricas por estrato
V4  temporal (entrenar en años previos, evaluar en el siguiente); requiere 'anio' con >=2 años comparables
V5  ponderación con factor_expansion: NO es una partición; es un modo de métricas (metrics_weighted.py)
    aplicable sobre cualquier esquema. El pipeline reporta siempre la versión ponderada y la no ponderada.

Las particiones se generan UNA vez, se verifican (sin solape) y se guardan en JSON con una huella
SHA-256; los cinco modelos consumen exactamente las mismas particiones. Cada partición es una tupla
``(train, test)`` con los atributos ``rep``, ``fold``, ``estrato`` y ``agregable``.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from fsai.config import (COL_GRUPO, COL_OBJETIVO, ESQUEMAS_PARTICION, N_REPETICIONES, N_SPLITS, SEED,
                         VARIANTES_V2)

logger = logging.getLogger(__name__)

COL_CONGLOME = "conglome"


class Particion(tuple):
    """Par ``(train, test)`` de índices posicionales con metadatos.

    rep: repetición (0 si el esquema no se repite); fold: índice del pliegue dentro de la repetición;
    estrato: etiqueta legible del bloque; agregable: si su test forma parte de la predicción fuera de
    muestra "de referencia" (repetición 0 y no transferencia); transferencia: bloque de transferencia
    (V3: entrenar en un área y evaluar en la otra), que no es comparable con los pliegues regulares.
    Los bloques con ``transferencia=False`` (las 20 combinaciones rep x fold en V1/V1c) son los pares
    que alimentan Wilcoxon y Friedman.
    """

    def __new__(cls, train, test, rep: int = 0, fold: int = 0, estrato: str = "", agregable: bool = True,
                transferencia: bool = False):
        obj = super().__new__(cls, (np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)))
        obj.rep, obj.fold, obj.estrato = int(rep), int(fold), str(estrato)
        obj.transferencia = bool(transferencia)
        obj.agregable = bool(agregable) and not obj.transferencia
        return obj

    def __getnewargs__(self):
        return (self[0], self[1], self.rep, self.fold, self.estrato, self.agregable, self.transferencia)


def _validar_df(df: pd.DataFrame, columnas: list[str]) -> None:
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas para particionar: {faltan}")


def esquema_base(esquema: str) -> str:
    """'V2_lodo' -> 'V2'; 'V1c' -> 'V1c' (código admitido por el CHECK de la base de datos)."""
    return esquema.split("_")[0]


def normalizar_esquemas(esquemas: list[str], variante_v2: str = "ambas") -> tuple[list[str], bool]:
    """Expande los códigos pedidos a claves ejecutables.

    'V2' se expande según ``variante_v2`` ('lodo', 'macro' o 'ambas'); 'V5' no genera particiones:
    se devuelve como indicador (True) y, si es lo único pedido, se aplica sobre V1.
    """
    if variante_v2 not in VARIANTES_V2 + ["ambas"]:
        raise ValueError(f"variante_v2 inválida: {variante_v2}")
    claves: list[str] = []
    pidio_v5 = False
    for e in esquemas:
        if e == "V5":
            pidio_v5 = True
        elif e == "V2":
            claves += [f"V2_{v}" for v in (VARIANTES_V2 if variante_v2 == "ambas" else [variante_v2])]
        elif e in ESQUEMAS_PARTICION:
            claves.append(e)
        else:
            raise ValueError(f"Esquema desconocido: {e}")
    if pidio_v5 and not claves:
        claves = ["V1"]
    return list(dict.fromkeys(claves)), pidio_v5


def _kfold_agrupado(df: pd.DataFrame, columna_grupo: str, n_splits: int, n_repeticiones: int) -> list[Particion]:
    _validar_df(df, [columna_grupo, COL_OBJETIVO])
    y = df[COL_OBJETIVO].to_numpy()
    g = df[columna_grupo].to_numpy()
    X_ficticia = np.zeros((len(df), 1))
    out: list[Particion] = []
    for rep in range(n_repeticiones):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED + rep)
        for k, (tr, te) in enumerate(cv.split(X_ficticia, y, g)):
            out.append(Particion(tr, te, rep=rep, fold=k, estrato=f"rep{rep}_fold{k}", agregable=(rep == 0)))
    return out


def generar_particiones_group_kfold(df: pd.DataFrame, n_splits: int = N_SPLITS,
                                    n_repeticiones: int = N_REPETICIONES) -> list[Particion]:
    """V1: StratifiedGroupKFold(5) por hogar_uid y estratificado por fies_nivel, repetido
    ``n_repeticiones`` veces con semillas SEED+rep (n_splits x n_repeticiones bloques)."""
    return _kfold_agrupado(df, COL_GRUPO, n_splits, n_repeticiones)


def particiones_v1c(df: pd.DataFrame, n_splits: int = N_SPLITS,
                    n_repeticiones: int = N_REPETICIONES) -> list[Particion]:
    """V1c (sensibilidad de V1): misma estructura, pero los grupos son los conglomerados."""
    return _kfold_agrupado(df, COL_CONGLOME, n_splits, n_repeticiones)


def _por_columna(df: pd.DataFrame, columna: str) -> list[Particion]:
    """Deja fuera, en cada pliegue, todos los hogares de una categoría de ``columna``."""
    _validar_df(df, [columna])
    valores = df[columna].astype(str).to_numpy()
    etiquetas = sorted(str(v) for v in pd.Series(df[columna]).dropna().unique())
    parts: list[Particion] = []
    for et in etiquetas:
        test = np.flatnonzero(valores == et)
        train = np.flatnonzero(valores != et)
        if len(test) and len(train):
            parts.append(Particion(train, test, rep=0, fold=len(parts), estrato=et))
    if len(parts) < 2:
        raise ValueError(f"'{columna}' tiene menos de 2 categorías: no se puede validar por bloques.")
    return parts


def particiones_lodo(df: pd.DataFrame) -> list[Particion]:
    """V2 (variante lodo): Leave-One-Department-Out (generalización geográfica)."""
    return _por_columna(df, "departamento")


def particiones_macroregion(df: pd.DataFrame) -> list[Particion]:
    """V2 (variante macro): dejar fuera cada macrorregión natural (costa, sierra, selva)."""
    return _por_columna(df, "region_natural")


def particiones_area(df: pd.DataFrame, n_splits: int = N_SPLITS) -> list[Particion]:
    """V3: 5 pliegues estratificados por área (U/R) x fies_nivel (agrupados por hogar_uid) y 2 de
    transferencia (entrenar en un estrato, evaluar en el otro). Los de transferencia no son agregables."""
    _validar_df(df, ["area", COL_GRUPO, COL_OBJETIVO])
    area = df["area"].astype(str).to_numpy()
    if len(set(area)) < 2:
        raise ValueError("'area' tiene menos de 2 categorías: V3 no es aplicable.")
    estrato_y = pd.Series(area + "_" + df[COL_OBJETIVO].astype(str).to_numpy())
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    parts = [Particion(tr, te, rep=0, fold=k, estrato=f"fold_{k}")
             for k, (tr, te) in enumerate(cv.split(np.zeros((len(df), 1)), estrato_y.to_numpy(),
                                                   df[COL_GRUPO].to_numpy()))]
    for origen, destino in (("U", "R"), ("R", "U")):
        tr, te = np.flatnonzero(area == origen), np.flatnonzero(area == destino)
        if len(tr) and len(te):
            parts.append(Particion(tr, te, rep=0, fold=len(parts), estrato=f"transfer_{origen}_a_{destino}",
                                   transferencia=True))
    return parts


def particiones_temporal(df: pd.DataFrame) -> list[Particion]:
    """V4: expansión temporal (train = años previos, test = año siguiente). Requiere 'anio'."""
    _validar_df(df, ["anio"])
    anios = sorted(df["anio"].dropna().unique())
    if len(anios) < 2:
        raise ValueError("V4 requiere al menos dos años comparables en 'anio'.")
    return [Particion(np.flatnonzero((df["anio"] < a).to_numpy()), np.flatnonzero((df["anio"] == a).to_numpy()),
                      rep=0, fold=i, estrato=str(a)) for i, a in enumerate(anios[1:])]


def etiquetas_particiones(particiones: list[Particion]) -> list[str]:
    """Etiqueta de estrato de cada partición (para reportar métricas por estrato)."""
    return [p.estrato for p in particiones]


def columna_grupo_interna(esquema: str) -> str:
    """Columna que agrupa la CV interna: conglome en V1c, hogar_uid en los demás esquemas."""
    return COL_CONGLOME if esquema_base(esquema) == "V1c" else COL_GRUPO


def generar_particiones(df: pd.DataFrame, esquema: str, variante: str | None = None) -> list[Particion]:
    """Despachador por esquema: 'V1', 'V1c', 'V2_lodo', 'V2_macro' (o 'V2' + ``variante``), 'V3', 'V4'.

    'V5' no es una partición y se rechaza. Verifica la ausencia de solape antes de devolver.
    """
    if esquema == "V2":
        esquema = f"V2_{variante or 'lodo'}"
    if esquema == "V5":
        raise ValueError("V5 (ponderación) no es una partición: se aplica como modo de métricas sobre otro esquema.")
    despacho = {"V1": generar_particiones_group_kfold, "V1c": particiones_v1c,
                "V2_lodo": particiones_lodo, "V2_macro": particiones_macroregion,
                "V3": particiones_area, "V4": particiones_temporal}
    if esquema not in despacho:
        raise ValueError(f"Esquema desconocido: {esquema}")
    particiones = despacho[esquema](df)
    verificar_sin_solape(particiones, df[COL_GRUPO].to_numpy())
    if esquema == "V1c":
        verificar_sin_solape(particiones, df[COL_CONGLOME].to_numpy())
    if esquema in ("V1", "V1c"):
        verificar_cobertura_por_repeticion(particiones, len(df))
    return particiones


def verificar_sin_solape(particiones: list[Particion], grupos: np.ndarray) -> None:
    """Falla si algún grupo (hogar_uid, o conglome en V1c) aparece a la vez en train y test."""
    for i, (tr, te) in enumerate(particiones):
        if set(grupos[tr]) & set(grupos[te]):
            raise AssertionError(f"Pliegue {i}: grupos compartidos entre train y test (fuga).")
        if set(tr.tolist()) & set(te.tolist()):
            raise AssertionError(f"Pliegue {i}: índices solapados.")


def verificar_cobertura_por_repeticion(particiones: list[Particion], n: int) -> None:
    """En V1/V1c cada fila debe ser test exactamente una vez por repetición."""
    for rep in sorted({p.rep for p in particiones}):
        tests = np.concatenate([p[1] for p in particiones if p.rep == rep])
        if len(tests) != n or len(np.unique(tests)) != n:
            raise AssertionError(f"Repetición {rep}: los tests no cubren cada fila exactamente una vez.")


def hash_particiones(particiones: list[Particion]) -> str:
    """Huella SHA-256 (16 hex) de las particiones para comprobar que son idénticas entre modelos."""
    h = hashlib.sha256()
    for tr, te in particiones:
        h.update(np.asarray(tr, dtype=np.int64).tobytes())
        h.update(b"|")
        h.update(np.asarray(te, dtype=np.int64).tobytes())
        h.update(b";")
    return h.hexdigest()[:16]


def guardar_particiones(particiones: list[Particion], ruta: Path, meta: dict | None = None) -> str:
    """Guarda las particiones (índices posicionales, rep/fold/estrato) en JSON y devuelve su huella."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    huella = hash_particiones(particiones)
    carga = {"huella": huella, "meta": meta or {},
             "particiones": [{"rep": p.rep, "fold": p.fold, "estrato": p.estrato, "agregable": p.agregable,
                              "transferencia": p.transferencia, "train": p[0].tolist(), "test": p[1].tolist()} for p in particiones]}
    ruta.write_text(json.dumps(carga), encoding="utf-8")
    logger.info("Particiones guardadas en %s (huella %s)", ruta, huella)
    return huella


def cargar_particiones(ruta: Path) -> list[Particion]:
    """Recupera particiones guardadas y verifica su huella."""
    carga = json.loads(ruta.read_text(encoding="utf-8"))
    parts = [Particion(p["train"], p["test"], p.get("rep", 0), p.get("fold", i), p.get("estrato", ""),
                       p.get("agregable", True), p.get("transferencia", False)) for i, p in enumerate(carga["particiones"])]
    if hash_particiones(parts) != carga["huella"]:
        raise ValueError(f"Huella inconsistente en {ruta}: archivo alterado.")
    return parts
