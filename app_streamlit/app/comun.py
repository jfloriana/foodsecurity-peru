"""Utilidades compartidas de FoodSecurity AI: conexión, consultas con caché y auditoría.

Todas las páginas consultan PostgreSQL (esquema ``fsai``) a través de ``consultar``,
que usa consultas parametrizadas con ``sqlalchemy.text`` y nunca interpola valores
del usuario en el texto SQL.
"""
from __future__ import annotations

import json
import logging
import math
import os
import unicodedata
from typing import Any, Mapping

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

import tema  # noqa: F401  (registra la plantilla Plotly común y el estilo de gráficos)


def _asegurar_fsai_en_ruta() -> None:
    """Hace importable ``fsai`` (FSAI_RAIZ_PROYECTO o la disposición V2: <raíz>/foodsecurity_ai) sin exigir fases_comun."""
    import sys
    from pathlib import Path

    candidatos = []
    if os.environ.get("FSAI_RAIZ_PROYECTO"):
        candidatos.append(Path(os.environ["FSAI_RAIZ_PROYECTO"]))
    candidatos += [Path(__file__).resolve().parents[2] / "foodsecurity_ai", Path(__file__).resolve().parents[1]]
    for c in candidatos:
        if (c / "fsai" / "__init__.py").is_file():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return


_asegurar_fsai_en_ruta()

LOGGER = logging.getLogger("fsai.app")

TTL_CACHE_SEG = int(os.environ.get("FSAI_CACHE_TTL", "300"))
TIMEOUT_SENTENCIA_MS = int(os.environ.get("FSAI_STATEMENT_TIMEOUT_MS", "60000"))

class ErrorConsulta(Exception):
    """Error controlado de acceso a datos (sin exponer detalles de conexión)."""


@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    """Crea (una sola vez por proceso) el engine SQLAlchemy 2.0 hacia PostgreSQL."""
    url = os.environ.get("FSAI_DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("Variable de entorno FSAI_DATABASE_URL no definida.")
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        pool_recycle=1800,
        connect_args={"options": f"-c statement_timeout={TIMEOUT_SENTENCIA_MS}"},
    )


def _normalizar_params(params: Mapping[str, Any] | None) -> tuple:
    """Convierte los parámetros en una tupla hashable (requisito de st.cache_data)."""
    return tuple(sorted((str(k), v) for k, v in (params or {}).items()))


def _consulta_directa(sql: str, params: tuple) -> pd.DataFrame:
    with get_engine().connect() as conn:
        resultado = conn.execute(text(sql), dict(params))
        filas = [tuple(fila) for fila in resultado.fetchall()]
        return pd.DataFrame(filas, columns=list(resultado.keys()))


_consulta_cacheada = st.cache_data(ttl=TTL_CACHE_SEG, show_spinner=False, max_entries=256)(
    _consulta_directa
)


def consultar(
    sql: str, params: Mapping[str, Any] | None = None, *, cache: bool = True
) -> pd.DataFrame:
    """Ejecuta una consulta parametrizada y devuelve un DataFrame.

    Los valores deben ser tipos nativos de Python (int, float, str, None); los
    ``numpy.int64`` no son adaptables por psycopg2 y deben convertirse antes.
    """
    clave = _normalizar_params(params)
    try:
        if cache:
            return _consulta_cacheada(sql, clave).copy()
        return _consulta_directa(sql, clave)
    except (SQLAlchemyError, RuntimeError) as exc:
        LOGGER.error("Fallo de consulta: %s", type(exc).__name__)
        raise ErrorConsulta(
            "No fue posible consultar la base de datos. Verifique FSAI_DATABASE_URL y que "
            "el servicio PostgreSQL esté activo."
        ) from exc


def mostrar_error_bd(exc: Exception) -> None:
    st.error(str(exc))


def explicar_error(exc: BaseException, accion: str = "completar la operación") -> str:
    """Mensaje comprensible para investigadores: qué falló, causa probable y detalle técnico entre paréntesis."""
    nombre = type(exc).__name__
    causas = (
        (("FileNotFoundError", "OSError"), "no se encontró un archivo necesario o no se pudo escribir; verifique que existan los artefactos de la corrida y que la carpeta de reportes tenga espacio y permisos"),
        (("KeyError", "ValueError"), "faltan columnas o hay valores inesperados en los datos; verifique que el dataset activo sea el mismo con que se entrenó la corrida"),
        (("MemoryError",), "no hay memoria suficiente; cierre otras aplicaciones o use una submuestra"),
        (("OperationalError", "ErrorConsulta", "SQLAlchemyError"), "no se pudo consultar la base de datos; verifique que PostgreSQL esté activo (iniciar.ps1)"),
        (("PermissionError",), "no hay permiso para escribir en la carpeta de destino; cierre el archivo si está abierto en Word o Excel"),
    )
    causa = next((c for nombres, c in causas if nombre in nombres), "ocurrió un error inesperado; repita la operación y, si persiste, revise los registros de la aplicación")
    return f"No se pudo {accion}: {causa}. Detalle técnico: {nombre}: {str(exc)[:200]}"


def registrar_auditoria(
    accion: str,
    entidad: str | None = None,
    entidad_id: str | None = None,
    detalle: Mapping[str, Any] | None = None,
    usuario: str | None = None,
) -> None:
    """Inserta un registro en fsai.auditoria; nunca interrumpe la interfaz si falla."""
    quien = (usuario or st.session_state.get("auth_usuario") or "anonimo")[:64]
    try:
        with get_engine().begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO fsai.auditoria (usuario, accion, entidad, entidad_id, detalle) "
                    "VALUES (:usuario, :accion, :entidad, :entidad_id, CAST(:detalle AS jsonb))"
                ),
                {
                    "usuario": quien,
                    "accion": accion[:64],
                    "entidad": entidad,
                    "entidad_id": None if entidad_id is None else str(entidad_id),
                    "detalle": json.dumps(dict(detalle or {}), default=str, ensure_ascii=False),
                },
            )
    except Exception as exc:  # noqa: BLE001 - la auditoría no debe romper la UI
        LOGGER.warning("No se pudo registrar auditoría (%s): %s", accion, type(exc).__name__)


def fmt_pct(valor: float | None, decimales: int = 1) -> str:
    """Formatea una proporción (0-1) como porcentaje; '—' si no hay valor."""
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "—"
    return f"{tema.num_es(100 * float(valor), decimales)} %"


def fmt_num(valor: float | None, decimales: int = 3) -> str:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "—"
    return tema.num_es(float(valor), decimales)


def normalizar_texto(valor: object) -> str:
    """Mayúsculas sin tildes ni espacios sobrantes (para cruzar nombres de departamento)."""
    base = unicodedata.normalize("NFKD", str(valor or ""))
    sin_tildes = "".join(c for c in base if not unicodedata.combining(c))
    return " ".join(sin_tildes.upper().split())


def leer_json(valor: Any) -> dict:
    """Devuelve un dict a partir de un valor JSONB (dict, str JSON o None)."""
    if isinstance(valor, dict):
        return valor
    if isinstance(valor, (str, bytes)):
        try:
            leido = json.loads(valor)
        except (TypeError, ValueError):
            return {}
        return leido if isinstance(leido, dict) else {}
    return {}


# ---------------------------------------------------------------- datos territoriales derivados (solo lectura, sin base de datos)
def _dir_datos_derivados():
    from pathlib import Path

    env = os.environ.get("FSAI_DATOS_DERIVADOS", "").strip()
    for c in ([Path(env)] if env else []) + [Path(__file__).resolve().parents[2] / "datos_derivados"]:
        if c.is_dir():
            return c
    return None


# clave -> (etiqueta, unidad corta, familia, ¿divergente?, decimales)
INDICADORES_TERRITORIALES: dict[str, tuple[str, str, str, bool, int]] = {
    "terr_desnutricion_cronica_pct": ("Desnutrición crónica infantil", "%", "Nutrición", False, 1),
    "terr_anemia_ninos_pct": ("Anemia en niños 6-59 m", "%", "Nutrición", False, 1),
    "terr_anemia_mujeres_pct": ("Anemia en mujeres 15-49", "%", "Nutrición", False, 1),
    "terr_mpi_indice": ("Índice de pobreza multidimensional (MPI)", "", "Pobreza", False, 3),
    "terr_mpi_incidencia_pct": ("Incidencia de pobreza multidimensional", "%", "Pobreza", False, 1),
    "terr_hogares_programa_social_pct": ("Hogares con algún programa social", "%", "Cobertura", False, 1),
    "terr_hogares_juntos_pct": ("Hogares en Juntos", "%", "Cobertura", False, 1),
    "terr_hogares_pension65_pct": ("Hogares con Pensión 65", "%", "Cobertura", False, 1),
    "terr_hogares_vaso_leche_pct": ("Hogares con Vaso de Leche", "%", "Cobertura", False, 1),
    "terr_precip_anom_pct": ("Anomalía de precipitación 2025", "%", "Clima", True, 1),
}


@st.cache_data(show_spinner=False)
def cargar_indicadores_territoriales() -> pd.DataFrame | None:
    """Indicadores por departamento (25 filas) de ``datos_derivados``; None si el archivo no está disponible."""
    carpeta = _dir_datos_derivados()
    if carpeta is None or not (carpeta / "indicadores_territoriales_departamento.csv").is_file():
        return None
    df = pd.read_csv(carpeta / "indicadores_territoriales_departamento.csv", dtype={"cod_departamento": str})
    df["cod_departamento"] = df["cod_departamento"].str.zfill(2)
    return df


@st.cache_data(show_spinner=False)
def cargar_json_derivado(nombre: str) -> dict:
    carpeta = _dir_datos_derivados()
    if carpeta is None or not (carpeta / nombre).is_file():
        return {}
    try:
        return json.loads((carpeta / nombre).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@st.cache_data(show_spinner=False)
def cargar_geojson_departamental() -> dict | None:
    """GeoJSON departamental incluido en la app (o el de FSAI_GEOJSON_DEPARTAMENTOS); None si no se puede leer."""
    try:
        from fsai import geo

        return geo.cargar_geojson()
    except Exception:  # noqa: BLE001
        return None
