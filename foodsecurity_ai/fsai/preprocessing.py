"""preprocessing.py - ColumnTransformer sin fuga de información.

El transformador SOLO se ajusta (fit) con datos de entrenamiento: se integra
dentro de un Pipeline, de modo que cada validación cruzada vuelva a estimar
medianas, modas, medias y desviaciones únicamente con el subconjunto de train.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fsai.config import (COLUMNAS_ID_TERRITORIALES, COLUMNAS_PROHIBIDAS, ESCENARIOS_TODOS,
                         FEATURES_FIES, FEATURES_INGENIERIA_CAT, FEATURES_INGENIERIA_NUM, FEATURES_SOCIO_CAT,
                         FEATURES_SOCIO_NUM, FEATURES_SOCIO_TERR_NUM)

logger = logging.getLogger(__name__)

# {escenario: [columnas 100 % NaN descartadas]} de la última llamada a dividir_features (trazabilidad)
DESCARTADAS_100_NAN: dict[str, list[str]] = {}


# Variables fuente permitidas para la ingeniería de variables (SOLO predictores socioeconómicos; jamás objetivo ni FIES).
_FUENTES_INGENIERIA = ("ingreso_pc", "gasto_pc", "gasto_alim_pc", "n_miembros", "n_menores5", "n_menores18",
                       "n_adultos_mayores", "asistencia_escolar", "edad_jefe", "nivel_educ_jefe")


def agregar_ingenieria(df: pd.DataFrame) -> pd.DataFrame:
    """Variables derivadas del escenario ``socio_terr`` (función pura: devuelve una copia; no toca filas ni columnas previas).

    - ``log_ingreso_pc``, ``log_gasto_pc``, ``log_gasto_alim_pc``: log1p de montos muy asimétricos (negativos -> 0).
    - ``log_ingreso_por_adulto_eq``: log1p(ingreso_pc * n_miembros / adultos equivalentes); equivalencia simple
      (adulto = 1, menor de 18 = 0,5, mínimo 1) [POR VALIDAR frente a escalas oficiales].
    - ``razon_dependencia``: (menores de 18 + adultos mayores) / max(miembros - esos, 1).
    - ``asistencia_escolar_faltante``: 1 si ``asistencia_escolar`` es NaN (la falta de dato puede ser informativa:
      hogares sin miembros en edad escolar); el imputador de mediana ya no oculta esa señal.
    - ``tiene_menor5``: 1 si n_menores5 > 0.  ``grupo_edad_jefe`` / ``grupo_educ_jefe``: tramos de edad_jefe / nivel_educ_jefe
      (mismos cortes de ``data_loader.derivar_grupos_equidad``; se conservan si ya existen, 'nan' -> NaN).
    Todas dependen únicamente de ``_FUENTES_INGENIERIA``; no hay fuga (ninguna usa el objetivo ni FIES).
    """
    prohibidas = set(COLUMNAS_PROHIBIDAS) | set(FEATURES_FIES)
    assert not (set(_FUENTES_INGENIERIA) & prohibidas), "la ingeniería no puede depender del objetivo/FIES"
    out = df.copy()
    num = lambda c: pd.to_numeric(out[c], errors="coerce") if c in out.columns else pd.Series(np.nan, index=out.index)  # noqa: E731
    for fuente, nuevo in (("ingreso_pc", "log_ingreso_pc"), ("gasto_pc", "log_gasto_pc"), ("gasto_alim_pc", "log_gasto_alim_pc")):
        out[nuevo] = np.log1p(num(fuente).clip(lower=0))
    n, m18, may, m5 = num("n_miembros"), num("n_menores18"), num("n_adultos_mayores"), num("n_menores5")
    adultos_eq = (n - m18 + 0.5 * m18).clip(lower=1.0)
    out["log_ingreso_por_adulto_eq"] = np.log1p((num("ingreso_pc").clip(lower=0) * n / adultos_eq).clip(lower=0))
    out["razon_dependencia"] = (m18 + may) / (n - m18 - may).clip(lower=1.0)
    out["asistencia_escolar_faltante"] = num("asistencia_escolar").isna().astype(float)
    out["tiene_menor5"] = (m5 > 0).astype(float).where(m5.notna())
    if "grupo_edad_jefe" not in out.columns:
        out["grupo_edad_jefe"] = pd.cut(num("edad_jefe"), bins=[-np.inf, 29, 44, 59, np.inf], labels=["<30", "30-44", "45-59", "60+"]).astype(object)
    if "grupo_educ_jefe" not in out.columns:
        out["grupo_educ_jefe"] = pd.cut(num("nivel_educ_jefe"), bins=[-np.inf, 1, 3, 5, np.inf],
                                        labels=["sin_primaria", "primaria", "secundaria", "superior"]).astype(object)
    for g in FEATURES_INGENIERIA_CAT:
        out[g] = out[g].astype(object).where(out[g].astype(object).astype(str) != "nan", np.nan)
    return out


def enriquecer_para_escenario(df: pd.DataFrame, escenario: str, *, ingenieria: bool = True,
                              tabla_territorial: pd.DataFrame | None = None) -> pd.DataFrame:
    """Prepara el DataFrame para el escenario. Solo ``socio_terr`` agrega columnas (unión territorial + ingeniería opcional);
    los demás escenarios devuelven ``df`` sin cambios (garantiza que ``socio`` / ``fies`` / ``hibrido`` no se alteren)."""
    if escenario != "socio_terr":
        return df
    out = df
    if not all(c in out.columns for c in FEATURES_SOCIO_TERR_NUM):
        from fsai.fuentes_territoriales import unir_territorial   # importación perezosa (evita ciclos y trabajo innecesario)
        out, rep = unir_territorial(out, tabla_territorial)
        logger.info("socio_terr: unión territorial %s", {k: rep[k] for k in ("filas", "filas_sin_unir", "oni")})
    if ingenieria:
        out = agregar_ingenieria(out)
    return out


def dividir_features(df: pd.DataFrame, escenario: str) -> tuple[list[str], list[str]]:
    """Devuelve (numéricas, categóricas) del escenario 'socio' | 'fies' | 'hibrido' | 'socio_terr'.

    - socio_terr: 'socio' + indicadores territoriales (FEATURES_SOCIO_TERR_NUM) + variables de ingeniería presentes en ``df``
      (use ``enriquecer_para_escenario`` antes). La definición de 'socio' NO cambia.

    - socio: solo socioeconómicas (escenario de PREDICCIÓN real).
    - fies: solo los 8 ítems y el puntaje crudo (reconstrucción/clasificación).
    - hibrido: ambas (reconstrucción/clasificación; circular por construcción).
    """
    if escenario not in ESCENARIOS_TODOS:
        raise ValueError(f"Escenario '{escenario}' inválido; use uno de {ESCENARIOS_TODOS}")
    num: list[str] = []
    cat: list[str] = []
    if escenario in ("socio", "hibrido", "socio_terr"):
        num += FEATURES_SOCIO_NUM
        cat += FEATURES_SOCIO_CAT
    if escenario == "socio_terr":
        faltan = [c for c in FEATURES_SOCIO_TERR_NUM if c not in df.columns]
        if faltan:
            raise ValueError(f"socio_terr requiere las columnas territoriales {faltan}: use enriquecer_para_escenario/unir_territorial.")
        num += FEATURES_SOCIO_TERR_NUM + [c for c in FEATURES_INGENIERIA_NUM if c in df.columns]
        cat += [c for c in FEATURES_INGENIERIA_CAT if c in df.columns]
    if escenario in ("fies", "hibrido"):
        num += FEATURES_FIES
    num = [c for c in num if c in df.columns]
    cat = [c for c in cat if c in df.columns]
    # Columnas 100 % NaN (p. ej. informalidad en ENAHO 2025): no aportan señal; se eliminan de los predictores con aviso
    # y se registran en DESCARTADAS_100_NAN (los resultados las guardan como 'features_descartadas_100pct_nan').
    vacias = [c for c in num + cat if df[c].isna().all()]
    DESCARTADAS_100_NAN[escenario] = vacias
    if vacias:
        logger.warning("Escenario %s: se descartan %d predictores 100%% NaN: %s", escenario, len(vacias), vacias)
        num = [c for c in num if c not in vacias]
        cat = [c for c in cat if c not in vacias]
    prohibidas = (set(num) | set(cat)) & (set(COLUMNAS_PROHIBIDAS) | set(COLUMNAS_ID_TERRITORIALES))
    if prohibidas:  # salvaguarda anti-fuga: objetivo e identificadores nunca son predictores
        raise ValueError(f"Variables prohibidas como predictoras: {sorted(prohibidas)}")
    if not num and not cat:
        raise ValueError(f"El dataset no contiene variables del escenario '{escenario}'.")
    logger.info("Escenario %s: %d numéricas, %d categóricas", escenario, len(num), len(cat))
    return num, cat


def preparar_matriz(df: pd.DataFrame, features_num: list[str], features_cat: list[str]) -> pd.DataFrame:
    """Selecciona columnas y homogeniza tipos (float para numéricas; object con np.nan para categóricas)."""
    X = df[features_num + features_cat].copy()
    for c in features_num:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    for c in features_cat:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X


def construir_preprocesador(features_num: list[str], features_cat: list[str]) -> ColumnTransformer:
    """ColumnTransformer: mediana+estandarización, moda+one-hot y rama ordinal para FIES."""
    cols_fies = [c for c in features_num if c in FEATURES_FIES]
    cols_num = [c for c in features_num if c not in FEATURES_FIES]
    transformadores = []
    if cols_num:
        transformadores.append(("num", Pipeline([
            ("imputador", SimpleImputer(strategy="median")),
            ("escalador", StandardScaler()),
        ]), cols_num))
    if cols_fies:
        # Ítems 0/1 y puntaje 0-8 son ordinales: moda para faltantes y mismo escalado
        # (transformación monótona que preserva el orden) para modelos sensibles a la escala.
        transformadores.append(("fies", Pipeline([
            ("imputador", SimpleImputer(strategy="most_frequent")),
            ("escalador", StandardScaler()),
        ]), cols_fies))
    if features_cat:
        transformadores.append(("cat", Pipeline([
            ("imputador", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), features_cat))
    if not transformadores:
        raise ValueError("No hay columnas para preprocesar.")
    return ColumnTransformer(transformadores, remainder="drop", verbose_feature_names_out=True)
