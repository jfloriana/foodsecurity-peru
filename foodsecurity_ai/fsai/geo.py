"""geo.py - Acceso al GeoJSON de departamentos del Perú (ayuda mínima para la app y los reportes).

Orden de resolución de la ruta:
  1. variable de entorno ``FSAI_GEOJSON_DEPARTAMENTOS`` (si apunta a un archivo existente);
  2. GeoJSON incluido en ``app_streamlit/app/assets/geo/peru_departamental.geojson`` (ver ``LICENCIA_GEO.md`` en esa carpeta).

El GeoJSON incluido tiene 25 features con ``NOMBDEP`` (mayúsculas, sin tildes) y ``FIRST_IDDP`` ("01".."25"). Es un archivo de
datos: solo se lee como JSON (nunca se ejecuta).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fsai import config as C

CLAVE_NOMBRE = "NOMBDEP"
CLAVE_CODIGO = "FIRST_IDDP"


def ruta_geojson() -> Path:
    """Ruta efectiva del GeoJSON departamental (variable de entorno si es válida; si no, el incluido en la app)."""
    env = os.getenv("FSAI_GEOJSON_DEPARTAMENTOS", "").strip()
    if env and Path(env).is_file():
        return Path(env)
    return C.GEOJSON_DEPARTAMENTOS_INCLUIDO


def cargar_geojson(ruta: Path | None = None) -> dict[str, Any]:
    """Carga y valida mínimamente el GeoJSON (FeatureCollection con propiedad de código de 2 dígitos)."""
    r = Path(ruta) if ruta else ruta_geojson()
    if not r.is_file():
        raise FileNotFoundError(f"No se encontró el GeoJSON departamental: {r}")
    g = json.loads(r.read_text(encoding="utf-8"))
    if g.get("type") != "FeatureCollection" or not g.get("features"):
        raise ValueError(f"{r.name}: no es una FeatureCollection con features")
    return g
