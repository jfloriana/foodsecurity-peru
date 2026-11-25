"""datasets.py - Fase 1: carga de archivos TABULARES (CSV/.dta/.sav/.parquet), BD y editor de metadata.

ADAPTACIÓN TABULAR: el Promptuario habla de "carga de imágenes (dataset público)"; en FoodSecurity AI los datos son
microdatos de hogares (ENAHO/FIES), por lo que se cargan archivos tabulares + diccionario/metadata.

Seguridad: los archivos subidos se guardan en ``DIR_CARGAS`` con nombre saneado, extensión en lista blanca y
tope de tamaño; nunca se ejecuta su contenido. La metadata se persiste como JSON por dataset (``DIR_METADATA``).
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fsai import config as C
from fsai.contrato_esquema import aplicar_mapa_columnas
from fsai.data_loader import _normalizar_llaves, construir_hogar_uid, derivar_grupos_equidad, recodificar_fies_crudo, validar_calidad
from fsai.fies import UMBRALES_RAW, agregar_columnas_fies

logger = logging.getLogger(__name__)

EXTENSIONES_PERMITIDAS = (".csv", ".dta", ".sav", ".parquet")
TOPE_BYTES = 200 * 1024 * 1024
PATRON_NOMBRE = re.compile(r"[^A-Za-z0-9_.\-]")
PREFIJO_DEMO = "DEMO_SINTETICO"


# --------------------------------------------------------------------------
# Archivos
# --------------------------------------------------------------------------
def es_demo(nombre: str) -> bool:
    """True si el nombre de dataset o archivo corresponde al demo sintético (no son resultados del estudio)."""
    n = str(nombre).upper()
    return n.startswith(PREFIJO_DEMO) or "DEMO_SINTETICO" in n


def listar_archivos_datos() -> list[Path]:
    """Archivos tabulares disponibles en ``DIR_DATOS`` y ``DIR_CARGAS`` (orden alfabético)."""
    salida: list[Path] = []
    for d in (C.DIR_DATOS, C.DIR_CARGAS):
        if d.is_dir():
            salida += sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONES_PERMITIDAS)
    return salida


def ruta_permitida(ruta: str | Path) -> Path:
    """Valida que ``ruta`` esté dentro de ``DIR_DATOS``/``DIR_CARGAS`` (evita leer rutas arbitrarias)."""
    r = Path(ruta).resolve()
    for d in (C.DIR_DATOS, C.DIR_CARGAS):
        if d.exists() and (d.resolve() == r.parent or d.resolve() in r.parents):
            if r.suffix.lower() in EXTENSIONES_PERMITIDAS and r.is_file():
                return r
    raise ValueError("Ruta de datos fuera de las carpetas permitidas o con extensión no admitida.")


def guardar_carga(nombre: str, contenido: bytes) -> Path:
    """Guarda un archivo subido con nombre saneado; rechaza extensiones no admitidas o tamaños excesivos."""
    base = PATRON_NOMBRE.sub("_", Path(nombre).name)[:80] or "carga"
    if Path(base).suffix.lower() not in EXTENSIONES_PERMITIDAS:
        raise ValueError(f"Extensión no admitida; use {EXTENSIONES_PERMITIDAS}.")
    if len(contenido) > TOPE_BYTES:
        raise ValueError("El archivo supera el tope de 200 MB.")
    C.DIR_CARGAS.mkdir(parents=True, exist_ok=True)
    destino = C.DIR_CARGAS / base
    destino.write_bytes(contenido)
    return destino


def leer_tabla(ruta: Path) -> pd.DataFrame:
    """Lee CSV/.dta/.sav/.parquet. ``.sav`` requiere ``pyreadstat`` (opcional)."""
    s = ruta.suffix.lower()
    if s == ".csv":
        df = pd.read_csv(ruta, low_memory=False, encoding="utf-8-sig",
                         dtype={c: str for c in ("ubigeo", "conglome", "vivienda", "hogar")})
    elif s == ".dta":
        df = pd.read_stata(ruta, convert_categoricals=False)
    elif s == ".parquet":
        df = pd.read_parquet(ruta)
    elif s == ".sav":
        try:
            df = pd.read_spss(ruta, convert_categoricals=False)
        except ImportError as exc:
            raise ImportError("Leer .sav requiere 'pyreadstat' (pip install pyreadstat).") from exc
    else:
        raise ValueError(f"Extensión no admitida: {s}")
    df.columns = [str(c).strip().lower() for c in df.columns]
    df, _, _ = aplicar_mapa_columnas(df)    # config.MAPA_COLUMNAS (vacío por defecto: no cambia nada)
    return df


# --------------------------------------------------------------------------
# Preparación (misma lógica que train_pipeline.preparar_datos)
# --------------------------------------------------------------------------
def preparar_dataframe(df: pd.DataFrame, *, fies_crudo: bool = False, umbrales: dict | None = None,
                       params_items: dict | None = None) -> pd.DataFrame:
    """Completa hogar_uid, FIES derivado y grupos de equidad; falla si la calidad es insuficiente."""
    df = df.copy()
    if "hogar_uid" not in df.columns:
        if not C.SALT:
            raise ValueError("Falta hogar_uid y no hay FSAI_SALT para construirlo.")
        df = _normalizar_llaves(df)
        df["hogar_uid"] = construir_hogar_uid(df, C.SALT)
    if fies_crudo:
        df = recodificar_fies_crudo(df)
    if C.COL_OBJETIVO not in df.columns or df[C.COL_OBJETIVO].isna().all():
        df = agregar_columnas_fies(df, umbrales or UMBRALES_RAW, params_items)
    df = df[df[C.COL_OBJETIVO].notna()].copy()
    df[C.COL_OBJETIVO] = df[C.COL_OBJETIVO].astype(int)
    df[C.COL_BINARIA] = (df[C.COL_OBJETIVO] >= 2).astype(int)
    df = derivar_grupos_equidad(df)
    informe = validar_calidad(df)
    if not informe["ok"]:
        raise ValueError(f"Calidad de datos insuficiente: {informe['errores']}")
    return df.reset_index(drop=True)


def cargar_hogares_ruta(ruta: str | Path, *, fies_crudo: bool = False) -> pd.DataFrame:
    """Como ``cargar_hogares`` pero SIN restringir la ruta (uso de línea de comandos / scripts de verificación)."""
    return preparar_dataframe(leer_tabla(Path(ruta)), fies_crudo=fies_crudo)


def cargar_hogares(ruta: str | Path, *, fies_crudo: bool = False) -> pd.DataFrame:
    """Archivo analítico a nivel hogar (dentro de las carpetas permitidas) -> DataFrame listo para particionar/modelar."""
    return cargar_hogares_ruta(ruta_permitida(ruta), fies_crudo=fies_crudo)


SQL_HOGARES_BD = """
SELECT h.hogar_uid, h.conglome_hash AS conglome, h.ubigeo, h.departamento, h.provincia, h.distrito, h.area,
       h.region_natural, CAST(h.factor_expansion AS double precision) AS factor_expansion, h.anio,
       r.fies_q1, r.fies_q2, r.fies_q3, r.fies_q4, r.fies_q5, r.fies_q6, r.fies_q7, r.fies_q8,
       r.fies_raw_score, CAST(r.fies_rasch_logit AS double precision) AS fies_rasch_logit, r.fies_nivel, r.ia_binaria,
       CAST(i.gasto_alim_pc AS double precision) AS gasto_alim_pc,
       CAST(i.share_alimentario AS double precision) AS share_alimentario,
       CAST(i.ingreso_pc AS double precision) AS ingreso_pc, CAST(i.gasto_pc AS double precision) AS gasto_pc,
       i.pobreza_monetaria,
       c.n_miembros, c.n_menores5, c.n_menores18, c.n_adultos_mayores, c.edad_jefe, c.sexo_jefe, c.nivel_educ_jefe,
       CAST(c.asistencia_escolar AS double precision) AS asistencia_escolar, c.n_ocupados,
       CAST(c.informalidad AS double precision) AS informalidad, c.desempleo,
       CAST(c.dependencia_economica AS double precision) AS dependencia_economica, c.agua_potable, c.saneamiento,
       CAST(c.hacinamiento AS double precision) AS hacinamiento, c.tipo_vivienda, c.material_vivienda
FROM fsai.hogares h
JOIN fsai.respuestas_fies r ON r.hogar_uid = h.hogar_uid
LEFT JOIN fsai.indicadores_alimentarios i ON i.hogar_uid = h.hogar_uid
LEFT JOIN fsai.condiciones_socioeconomicas c ON c.hogar_uid = h.hogar_uid
WHERE h.dataset_id = :dataset_id
"""


def cargar_desde_bd(engine, dataset_id: int) -> pd.DataFrame:
    """Reconstruye el dataset analítico desde fsai.* (consulta parametrizada). ``conglome`` = conglome_hash."""
    from sqlalchemy import text
    with engine.connect() as conn:
        res = conn.execute(text(SQL_HOGARES_BD), {"dataset_id": int(dataset_id)})
        df = pd.DataFrame([tuple(f) for f in res.fetchall()], columns=list(res.keys()))
    if df.empty:
        raise ValueError(f"El dataset_id={dataset_id} no tiene hogares.")
    return preparar_dataframe(df)


def exportar_para_pipeline(df: pd.DataFrame, nombre: str) -> Path:
    """Escribe un CSV para ``train_pipeline.py`` (que exige llaves INEI) a partir de un dataset de la BD.

    ``vivienda``/``hogar`` se rellenan con marcadores porque la BD solo guarda el hash del conglomerado.
    """
    out = df.copy()
    out["vivienda"] = out.get("vivienda", pd.Series(["NA"] * len(out), index=out.index))
    out["hogar"] = out.get("hogar", pd.Series(["1"] * len(out), index=out.index))
    C.DIR_CARGAS.mkdir(parents=True, exist_ok=True)
    destino = C.DIR_CARGAS / (PATRON_NOMBRE.sub("_", nombre)[:60] + ".csv")
    out.to_csv(destino, index=False, encoding="utf-8")
    return destino


# --------------------------------------------------------------------------
# Metadata (tipo, rol, rango, descripción)
# --------------------------------------------------------------------------
DESCRIPCIONES: dict[str, str] = {
    "conglome": "Conglomerado muestral (llave INEI)", "vivienda": "Número de vivienda (llave INEI)",
    "hogar": "Número de hogar (llave INEI)", "hogar_uid": "Seudónimo SHA-256 del hogar (conglome|vivienda|hogar|SALT)",
    "factor_expansion": "Factor de expansión muestral (factor07)", "ubigeo": "Código de ubicación geográfica (6 dígitos)",
    "departamento": "Departamento", "provincia": "Provincia", "distrito": "Distrito", "area": "Área: U urbano / R rural",
    "region_natural": "Región natural: costa, sierra o selva", "anio": "Año de la encuesta",
    "fies_raw_score": "Puntaje crudo FIES (0-8)", "fies_rasch_logit": "Severidad Rasch (logit) [POR VALIDAR]",
    "fies_nivel": "Nivel FIES 0-3 (objetivo multiclase)", "ia_binaria": "Inseguridad alimentaria moderada/severa (fies_nivel>=2)",
    "ingreso_pc": "Ingreso per cápita", "gasto_pc": "Gasto per cápita", "gasto_alim_pc": "Gasto en alimentos per cápita",
    "share_alimentario": "Proporción del gasto destinada a alimentos", "pobreza_monetaria": "Pobreza monetaria (0/1)",
    "n_ocupados": "Número de ocupados", "informalidad": "Proporción de ocupados informales (0-1)",
    "desempleo": "Algún desocupado en el hogar (0/1)", "dependencia_economica": "Ocupados / miembros",
    "nivel_educ_jefe": "Nivel educativo del jefe (0-11 INEI)", "asistencia_escolar": "Proporción de 6-17 años que asisten (0-1)",
    "n_miembros": "Número de miembros", "n_menores5": "Menores de 5 años", "n_menores18": "Menores de 18 años",
    "edad_jefe": "Edad del jefe de hogar", "sexo_jefe": "Sexo del jefe: H/M", "n_adultos_mayores": "Adultos mayores",
    "agua_potable": "Acceso a agua potable (0/1)", "saneamiento": "Acceso a saneamiento (0/1)",
    "hacinamiento": "Personas por habitación", "tipo_vivienda": "Tipo de vivienda (categórica)",
    "material_vivienda": "Material predominante de la vivienda (categórica)",
    "grupo_edad_jefe": "Grupo de edad del jefe (equidad)", "grupo_educ_jefe": "Grupo educativo del jefe (equidad)",
    "tiene_menor5": "Hogar con menor de 5 años (0/1)",
}
RANGOS_CONTRATO: dict[str, tuple[float, float]] = {
    "fies_raw_score": (0, 8), "fies_nivel": (0, 3), "ia_binaria": (0, 1), "nivel_educ_jefe": (0, 11),
    "informalidad": (0, 1), "asistencia_escolar": (0, 1), "pobreza_monetaria": (0, 1), "desempleo": (0, 1),
    "agua_potable": (0, 1), "saneamiento": (0, 1), **{f"fies_q{i}": (0, 1) for i in range(1, 9)},
}
ROLES_META = ("id", "objetivo", "peso", "territorial", "predictora_socio", "predictora_fies", "grupo_equidad", "otro", "excluida")
TIPOS_META = ("numérica", "categórica", "binaria", "texto/id")


def _rol_de(col: str) -> str:
    if col in C.COLUMNAS_ID_TERRITORIALES:
        return "id" if col in ("conglome", "vivienda", "hogar", "hogar_uid") else "territorial"
    if col in (C.COL_OBJETIVO, C.COL_BINARIA):
        return "objetivo"
    if col == C.COL_PESO:
        return "peso"
    if col in C.FEATURES_FIES:
        return "predictora_fies"
    if col in C.FEATURES_SOCIO:
        return "predictora_socio"
    if col in ("grupo_edad_jefe", "grupo_educ_jefe", "tiene_menor5"):
        return "grupo_equidad"
    if col == "fies_rasch_logit":
        return "excluida"
    return "otro"


def metadata_defecto(df: pd.DataFrame) -> pd.DataFrame:
    """Metadata propuesta por variable: tipo, rol, rango observado/contrato, descripción y % de nulos."""
    filas = []
    for col in df.columns:
        s = df[col]
        numerica = pd.api.types.is_numeric_dtype(s)
        binaria = numerica and set(s.dropna().unique()) <= {0, 1}
        tipo = "binaria" if binaria else ("numérica" if numerica else ("texto/id" if col in ("hogar_uid", "conglome", "vivienda", "hogar") else "categórica"))
        lo, hi = RANGOS_CONTRATO.get(col, (None, None))
        if lo is None and numerica and s.notna().any():
            lo, hi = float(s.min()), float(s.max())
        filas.append({"variable": col, "tipo": tipo, "rol": _rol_de(col), "rango_min": lo, "rango_max": hi,
                      "descripcion": DESCRIPCIONES.get(col, ""), "pct_nulos": round(float(s.isna().mean()) * 100, 2)})
    return pd.DataFrame(filas)


def _ruta_meta(clave: str) -> Path:
    return C.DIR_METADATA / (PATRON_NOMBRE.sub("_", clave)[:100] + ".json")


def guardar_metadata(clave: str, meta: pd.DataFrame, usuario: str = "") -> Path:
    """Persiste la metadata editada como JSON (con fecha y usuario)."""
    validar_metadata(meta)
    C.DIR_METADATA.mkdir(parents=True, exist_ok=True)
    ruta = _ruta_meta(clave)
    registro = {"dataset": clave, "guardado_en": datetime.now().isoformat(timespec="seconds"), "usuario": usuario,
                "variables": json.loads(meta.to_json(orient="records", force_ascii=False))}
    ruta.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta


def leer_metadata(clave: str) -> pd.DataFrame | None:
    ruta = _ruta_meta(clave)
    if not ruta.is_file():
        return None
    try:
        return pd.DataFrame(json.loads(ruta.read_text(encoding="utf-8"))["variables"])
    except (ValueError, KeyError):
        return None


def validar_metadata(meta: pd.DataFrame) -> None:
    """Valida tipos/roles permitidos y rangos coherentes (min<=max)."""
    if not set(meta["tipo"].dropna()) <= set(TIPOS_META):
        raise ValueError(f"Tipo inválido; use {TIPOS_META}.")
    if not set(meta["rol"].dropna()) <= set(ROLES_META):
        raise ValueError(f"Rol inválido; use {ROLES_META}.")
    mm = meta.dropna(subset=["rango_min", "rango_max"])
    if (mm["rango_min"] > mm["rango_max"]).any():
        raise ValueError("Hay variables con rango_min > rango_max.")
    if meta["variable"].duplicated().any():
        raise ValueError("Nombres de variable duplicados.")


def verificar_rangos(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Cuenta valores fuera del rango declarado en la metadata (solo numéricas con rango)."""
    filas = []
    for r in meta.dropna(subset=["rango_min", "rango_max"]).itertuples():
        if r.variable in df.columns and pd.api.types.is_numeric_dtype(df[r.variable]):
            s = df[r.variable].dropna()
            fuera = int(((s < r.rango_min) | (s > r.rango_max)).sum())
            filas.append({"variable": r.variable, "rango_min": r.rango_min, "rango_max": r.rango_max,
                          "n_fuera_de_rango": fuera, "pct_fuera": round(100 * fuera / max(len(s), 1), 2)})
    return pd.DataFrame(filas)


def resumen_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Cifras de cabecera del dataset (todas calculadas de ``df``)."""
    w = df[C.COL_PESO] if C.COL_PESO in df.columns else pd.Series(np.ones(len(df)))
    return {"n_hogares": int(len(df)), "n_columnas": int(df.shape[1]),
            "n_conglomerados": int(df["conglome"].nunique()) if "conglome" in df.columns else None,
            "pct_nulos_total": round(float(df.isna().mean().mean()) * 100, 2),
            "n_efectivo_kish": float(w.sum() ** 2 / (w ** 2).sum())}
