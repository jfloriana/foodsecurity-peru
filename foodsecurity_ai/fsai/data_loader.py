"""data_loader.py - Lectura de ENAHO (.dta/.csv/.parquet), unión por llaves INEI y calidad.

Los módulos ENAHO se unen por las llaves de hogar (conglome, vivienda, hogar).
Los módulos a nivel de persona deben agregarse a hogar ANTES de la unión; esta
capa lo verifica para impedir duplicación silenciosa de filas (fan-out).
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fsai.config import (CLASES, COL_BINARIA, COL_OBJETIVO, COL_PESO, FEATURES_FIES,
                         LLAVES_INEI, MAPA_COLUMNAS)

logger = logging.getLogger(__name__)

EXTENSIONES = (".dta", ".csv", ".parquet")


def _leer_archivo(ruta: Path) -> pd.DataFrame:
    """Lee un archivo tabular según su extensión; normaliza nombres a minúsculas."""
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo de datos: {ruta}")
    try:
        sufijo = ruta.suffix.lower()
        if sufijo == ".dta":
            df = pd.read_stata(ruta, convert_categoricals=False)
        elif sufijo == ".csv":
            # llaves y ubigeo como texto: conservan los ceros a la izquierda (p. ej. ubigeo '010101')
            df = pd.read_csv(ruta, low_memory=False, encoding="utf-8-sig",
                             dtype={c: str for c in ("ubigeo", "conglome", "vivienda", "hogar")})
        elif sufijo == ".parquet":
            df = pd.read_parquet(ruta)
        else:
            raise ValueError(f"Extensión no soportada: {ruta.suffix} (use {EXTENSIONES})")
    except Exception:
        logger.exception("Fallo al leer %s", ruta)
        raise
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _normalizar_llaves(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte las llaves a texto sin espacios (INEI mezcla enteros y cadenas)."""
    faltan = [c for c in LLAVES_INEI if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan llaves INEI {faltan}; columnas disponibles: {list(df.columns)[:15]}...")
    out = df.copy()
    for c in LLAVES_INEI:
        out[c] = out[c].astype(str).str.strip()
    return out


def renombrar_columnas(df: pd.DataFrame, mapa: dict[str, str]) -> pd.DataFrame:
    """Renombra códigos INEI a nombres analíticos del contrato (solo los presentes)."""
    presentes = {k.lower(): v for k, v in mapa.items() if k.lower() in df.columns}
    return df.rename(columns=presentes)


def unir_modulos(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Unión izquierda por (conglome, vivienda, hogar) tomando el primer módulo como base.

    Exige unicidad de llaves en cada módulo: si una tabla tiene varias filas por
    hogar (nivel persona) debe agregarse antes; de lo contrario se aborta.
    """
    if not dfs:
        raise ValueError("No hay módulos para unir.")
    nombres = list(dfs)
    for nombre in nombres:
        dup = int(dfs[nombre].duplicated(LLAVES_INEI).sum())
        if dup:
            raise ValueError(f"Módulo '{nombre}' tiene {dup} llaves duplicadas: agréguelo a nivel hogar.")
    base = dfs[nombres[0]].copy()
    n_base = len(base)
    for nombre in nombres[1:]:
        extra = dfs[nombre]
        repetidas = [c for c in extra.columns if c in base.columns and c not in LLAVES_INEI]
        if repetidas:
            logger.warning("Columnas repetidas en '%s' se descartan del módulo: %s", nombre, repetidas)
            extra = extra.drop(columns=repetidas)
        base = base.merge(extra, on=LLAVES_INEI, how="left", validate="one_to_one")
        if len(base) != n_base:
            raise RuntimeError(f"La unión con '{nombre}' alteró el número de filas.")
    return base


def cargar_enaho(ruta: str | Path, modulos: list[str] | None = None) -> pd.DataFrame:
    """Carga y une módulos ENAHO a nivel hogar.

    Parámetros
    ----------
    ruta : archivo único (tabla ya a nivel hogar) o directorio con un archivo por módulo.
    modulos : nombres (sin extensión) de los archivos a leer dentro de ``ruta``.
        Se ignora si ``ruta`` es un archivo.
    """
    ruta = Path(ruta)
    if ruta.is_file():
        df = _normalizar_llaves(_leer_archivo(ruta))
        return renombrar_columnas(df, MAPA_COLUMNAS)
    if not ruta.is_dir():
        raise FileNotFoundError(f"Ruta inexistente: {ruta}")
    if not modulos:
        raise ValueError("Para un directorio debe indicar la lista 'modulos'.")
    dfs: dict[str, pd.DataFrame] = {}
    for nombre in modulos:
        existentes = [ruta / f"{nombre}{ext}" for ext in EXTENSIONES if (ruta / f"{nombre}{ext}").exists()]
        if not existentes:
            raise FileNotFoundError(f"Módulo '{nombre}' no encontrado en {ruta} (ext. {EXTENSIONES})")
        dfs[nombre] = _normalizar_llaves(_leer_archivo(existentes[0]))
        logger.info("Módulo %s: %d filas, %d columnas", nombre, *dfs[nombre].shape)
    return renombrar_columnas(unir_modulos(dfs), MAPA_COLUMNAS)


def hash_sha256(texto: str) -> str:
    """SHA-256 hexadecimal (64 caracteres) de un texto."""
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def construir_hogar_uid(df: pd.DataFrame, salt: str) -> pd.Series:
    """Seudónimo del hogar: SHA-256 de 'conglome|vivienda|hogar|SALT' (hex de 64)."""
    if not salt:
        raise ValueError("Se requiere una sal no vacía (variable de entorno FSAI_SALT).")
    llaves = _normalizar_llaves(df)[LLAVES_INEI]
    texto = llaves["conglome"] + "|" + llaves["vivienda"] + "|" + llaves["hogar"] + "|" + salt
    uid = texto.map(hash_sha256)
    uid.name = "hogar_uid"
    return uid


def recodificar_fies_crudo(df: pd.DataFrame) -> pd.DataFrame:
    """Recodifica fies_q1..q8 crudos INEI (1=Sí, 2=No; otros=no sabe/no responde) a 0/1/NaN."""
    out = df.copy()
    for col in FEATURES_FIES[:8]:
        if col not in out.columns:
            raise KeyError(f"Falta la columna FIES cruda {col}")
        out[col] = out[col].map({1: 1.0, 2: 0.0}).astype("float")
    return out


def derivar_grupos_equidad(df: pd.DataFrame) -> pd.DataFrame:
    """Crea grupo_edad_jefe, grupo_educ_jefe y tiene_menor5 (grupos de equidad del contrato)."""
    out = df.copy()
    out["grupo_edad_jefe"] = pd.cut(out["edad_jefe"], bins=[-np.inf, 29, 44, 59, np.inf],
                                    labels=["<30", "30-44", "45-59", "60+"]).astype(str)
    # nivel_educ_jefe 0-11 (INEI): 0-1 sin primaria; 2-3 primaria; 4-5 secundaria; >=6 superior [POR VALIDAR]
    out["grupo_educ_jefe"] = pd.cut(out["nivel_educ_jefe"], bins=[-np.inf, 1, 3, 5, np.inf],
                                    labels=["sin_primaria", "primaria", "secundaria", "superior"]).astype(str)
    out["tiene_menor5"] = (out["n_menores5"] > 0).astype(int)
    return out


def validar_calidad(df: pd.DataFrame) -> dict[str, Any]:
    """Controles de calidad. Devuelve un dict con 'errores' (bloqueantes) y 'advertencias'."""
    errores: list[str] = []
    advertencias: list[str] = []
    rep: dict[str, Any] = {"n_filas": int(len(df)), "n_columnas": int(df.shape[1])}

    if "hogar_uid" in df.columns:
        dup = int(df["hogar_uid"].duplicated().sum())
        rep["hogar_uid_duplicados"] = dup
        if dup:
            errores.append(f"{dup} hogar_uid duplicados")
    if COL_PESO not in df.columns:
        errores.append(f"Falta {COL_PESO}")
    else:
        w = pd.to_numeric(df[COL_PESO], errors="coerce")
        if w.isna().any() or (w <= 0).any():
            errores.append(f"{COL_PESO} debe ser > 0 y sin nulos")
        else:
            rep["suma_factor_expansion"] = float(w.sum())
            rep["n_efectivo_kish"] = float(w.sum() ** 2 / (w ** 2).sum())
    for q in FEATURES_FIES[:8]:
        if q in df.columns and not set(df[q].dropna().unique()) <= {0, 1}:
            errores.append(f"{q} fuera de {{0,1}} (¿falta recodificar el crudo?)")
    if "area" in df.columns and not set(df["area"].dropna().unique()) <= {"U", "R"}:
        errores.append("area debe ser 'U' o 'R'")
    if "region_natural" in df.columns and not set(df["region_natural"].dropna().unique()) <= {"costa", "sierra", "selva"}:
        errores.append("region_natural fuera de {costa, sierra, selva}")
    if "nivel_educ_jefe" in df.columns and not df["nivel_educ_jefe"].dropna().between(0, 11).all():
        errores.append("nivel_educ_jefe fuera de 0-11")
    for c in ("informalidad", "asistencia_escolar"):
        if c in df.columns and not df[c].dropna().between(0, 1).all():
            errores.append(f"{c} fuera de [0,1]")
    for c in ("ingreso_pc", "gasto_pc", "gasto_alim_pc", "hacinamiento"):
        if c in df.columns and (df[c] < 0).any():
            errores.append(f"{c} con valores negativos")

    nulos = df.isna().mean()
    rep["porcentaje_nulos"] = {k: round(float(v), 4) for k, v in nulos[nulos > 0].items()}
    alto = [k for k, v in nulos.items() if v > 0.30]
    if alto:
        advertencias.append(f"Columnas con >30% de nulos: {alto}")
    if COL_OBJETIVO in df.columns:
        dist = df[COL_OBJETIVO].value_counts(normalize=True).sort_index()
        rep["distribucion_objetivo"] = {int(k): round(float(v), 4) for k, v in dist.items()}
        faltan = [c for c in CLASES if c not in dist.index]
        if faltan:
            errores.append(f"Clases ausentes en el objetivo: {faltan}")
        elif dist.min() < 0.01:
            advertencias.append("Clase con <1% de frecuencia: usar pesos de clase/evaluación estratificada")
    if COL_BINARIA in df.columns and COL_OBJETIVO in df.columns:
        if not (df[COL_BINARIA] == (df[COL_OBJETIVO] >= 2).astype(int)).all():
            errores.append("ia_binaria inconsistente con fies_nivel>=2")
    rep["errores"] = errores
    rep["advertencias"] = advertencias
    rep["ok"] = not errores
    for msg in errores:
        logger.error("Calidad: %s", msg)
    for msg in advertencias:
        logger.warning("Calidad: %s", msg)
    return rep
