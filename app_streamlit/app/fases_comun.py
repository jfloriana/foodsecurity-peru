"""Utilidades compartidas por las páginas de análisis (datasets, entrenamiento, textos interpretativos).

* Hace importable el paquete ``fsai`` (ruta ``FSAI_RAIZ_PROYECTO`` o la ubicación relativa en el repositorio).
* Selector de dataset GLOBAL (barra lateral, persistente en ``session_state``) que usan todos los módulos.
* Presentación de los textos interpretativos (qué muestra / cómo leerlo / limitacion).
* Lanzamiento controlado (lista blanca, sin shell) de ``train_pipeline.py`` como subproceso.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitDuplicateElementKey


def _raiz_fsai() -> Path:
    candidatos = []
    if os.environ.get("FSAI_RAIZ_PROYECTO"):
        candidatos.append(Path(os.environ["FSAI_RAIZ_PROYECTO"]))
    candidatos.append(Path(__file__).resolve().parents[2] / "codigo_verificacion" / "foodsecurity_ai")
    for c in candidatos:
        if (c / "fsai" / "__init__.py").is_file():
            return c
    raise RuntimeError("No se encontró el paquete fsai (defina FSAI_RAIZ_PROYECTO).")


RAIZ_FSAI = _raiz_fsai()
if str(RAIZ_FSAI) not in sys.path:
    sys.path.insert(0, str(RAIZ_FSAI))

from fsai import config as C  # noqa: E402
from fsai import datasets as D  # noqa: E402
from comun import ErrorConsulta, consultar, get_engine, registrar_auditoria  # noqa: E402

SCRIPT_PIPELINE = RAIZ_FSAI / "train_pipeline.py"
PATRON_NOMBRE = re.compile(r"^[A-Za-z0-9_.\-]{3,60}$")
MODELOS = {
    "M1": "M1 · Regresión logística multinomial",
    "M2": "M2 · Random Forest",
    "M3": "M3 · CatBoost",
    "M4": "M4 · Stacking CatBoost + RF + LR",
    "M5": "M5 · Autoencoder + XGBoost",
}
ESCENARIOS = {
    "socio": "socio · solo variables socioeconómicas (predicción real)",
    "fies": "fies · solo ítems FIES (reconstrucción/clasificación; circular)",
    "hibrido": "hibrido · socioeconómicas + FIES (reconstrucción/clasificación; circular)",
}
AVISO_CIRCULARIDAD = (
    "**Circularidad FIES.** `fies_nivel` se deriva de `fies_q1..fies_q8`. Los escenarios **fies** e **hibrido** usan "
    "esos ítems como predictores de su propio objetivo: son tareas de **reconstrucción/clasificación** y su desempeño casi "
    "perfecto NO mide capacidad predictiva. Solo **socio** es la predicción real a partir de condiciones del hogar."
)
AVISO_NO_CAUSAL = "Las explicaciones y asociaciones mostradas **no son causales**."

# ------------------------------ presentación ------------------------------
def interpretacion(texto: dict[str, str] | None, titulo: str = "Interpretación (generada a partir de los datos mostrados)") -> None:
    """Muestra qué muestra / cómo leerlo / limitación en un recuadro (contenedor; válido dentro de expanders y pestañas)."""
    if not texto:
        return
    with st.container(border=True):
        st.markdown(f"##### {titulo}")
        st.markdown(f"**Qué muestra.** {texto['que_muestra']}")
        st.markdown(f"**Cómo leerlo.** {texto['como_leerlo']}")
        st.markdown(f"**Limitación.** {texto['limitacion']}")


# ------------------------------ datasets ------------------------------
CLAVE_DATASET = "dataset_global"


def _es_sintetico(nombre: str) -> bool:
    """Los datasets sintéticos de prueba no se ofrecen como datos de análisis."""
    return "SINTETICO" in str(nombre).upper()


def _opciones_dataset() -> dict[str, dict[str, Any]]:
    """Datasets de PostgreSQL (más recientes primero) y archivos tabulares; los sintéticos de prueba no se ofrecen."""
    ops: dict[str, dict[str, Any]] = {}
    try:
        ds = consultar("SELECT dataset_id, nombre, anio, n_filas FROM fsai.datasets ORDER BY dataset_id DESC LIMIT 50", cache=False)
        for r in ds.itertuples():
            ops[f"bd:{int(r.dataset_id)}"] = {"etiqueta": f"{r.nombre} · {r.anio} · {int(r.n_filas):,} hogares", "origen": "bd",
                                             "dataset_id": int(r.dataset_id), "nombre": str(r.nombre), "sintetico": _es_sintetico(str(r.nombre))}
    except Exception:  # noqa: BLE001 - sin BD solo se ofrecen archivos
        pass
    for p in D.listar_archivos_datos():
        ops[f"archivo:{p.name}"] = {"etiqueta": f"Archivo · {p.name}", "origen": "archivo", "ruta": str(p), "nombre": p.name, "sintetico": _es_sintetico(p.name)}
    reales = {k: v for k, v in ops.items() if not v["sintetico"]}
    return reales or ops


@st.cache_data(show_spinner="Cargando y preparando el dataset...", max_entries=4)
def _cargar_archivo(ruta: str, mtime: float, fies_crudo: bool) -> pd.DataFrame:
    return D.cargar_hogares(ruta, fies_crudo=fies_crudo)


@st.cache_data(show_spinner="Cargando el dataset desde PostgreSQL...", max_entries=4, ttl=600)
def _cargar_bd(dataset_id: int) -> pd.DataFrame:
    return D.cargar_desde_bd(get_engine(), dataset_id)


def selector_lateral() -> dict[str, dict[str, Any]]:
    """Dibuja el selector GLOBAL de dataset en la barra lateral y devuelve las opciones (clave -> info).

    El enrutador lo dibuja una vez por ejecución; las páginas ejecutadas sueltas lo dibujan si falta.
    """
    ops = _opciones_dataset()
    if not ops:
        return ops
    claves = list(ops)
    if st.session_state.get(CLAVE_DATASET) not in ops:
        st.session_state[CLAVE_DATASET] = claves[0]
    try:
        with st.sidebar:
            st.selectbox("Dataset activo", claves, key=CLAVE_DATASET, format_func=lambda k: ops[k]["etiqueta"],
                         help="Todos los módulos trabajan con este dataset. Los datasets de PostgreSQL provienen de la carga de ENAHO.")
            if ops[st.session_state[CLAVE_DATASET]]["origen"] == "archivo":
                st.checkbox("Ítems FIES con codificación INEI cruda (1 = Sí, 2 = No)", value=False, key="fies_crudo_global",
                            help="Marque solo si el archivo trae los ítems FIES sin recodificar.")
    except StreamlitDuplicateElementKey:
        pass  # ya dibujado en esta ejecución (por el enrutador o por otra llamada de la misma página)
    return ops


def _opciones_y_selector() -> dict[str, dict[str, Any]]:
    if st.session_state.get("_fsai_sel_en_router"):
        return _opciones_dataset()
    return selector_lateral()


def dataset_id_activo() -> int | None:
    """``dataset_id`` de PostgreSQL del dataset activo, o None si es un archivo o no hay datasets."""
    ops = _opciones_y_selector()
    info = ops.get(st.session_state.get(CLAVE_DATASET, ""))
    if info is None and ops:
        info = next(iter(ops.values()))
    return info.get("dataset_id") if info and info["origen"] == "bd" else None


def selector_dataset(clave_estado: str = CLAVE_DATASET) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Dataset activo global. Devuelve (df preparado, info). Detiene la página si no hay datos."""
    ops = _opciones_y_selector()
    if not ops:
        st.warning("No hay datasets disponibles. **Siguiente paso:** cargue ENAHO en el módulo «Carga de datos», coloque un archivo tabular en la carpeta de datos o cargue ENAHO en PostgreSQL.")
        st.stop()
    clave = st.session_state.get(CLAVE_DATASET)
    if clave not in ops:
        clave = next(iter(ops))
    info = {**ops[clave], "clave": clave}
    try:
        if info["origen"] == "archivo":
            df = _cargar_archivo(info["ruta"], Path(info["ruta"]).stat().st_mtime, bool(st.session_state.get("fies_crudo_global", False)))
        else:
            df = _cargar_bd(info["dataset_id"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo preparar el dataset: {type(exc).__name__}: {exc}")
        st.stop()
    info["n"] = len(df)
    return df, info


def ruta_para_pipeline(df: pd.DataFrame, info: dict[str, Any]) -> Path:
    """Ruta de archivo que ``train_pipeline.py`` puede leer (exporta los datasets de la BD a CSV)."""
    if info["origen"] == "archivo":
        return D.ruta_permitida(info["ruta"])
    return D.exportar_para_pipeline(df, f"bd_{info['dataset_id']}_{info['nombre']}")


# ------------------------------ subprocesos ------------------------------
def lanzar_pipeline(argumentos: list[str], log: Path) -> subprocess.Popen:
    """Ejecuta ``train_pipeline.py`` con una LISTA de argumentos ya validados (sin shell)."""
    if not SCRIPT_PIPELINE.is_file():
        raise FileNotFoundError("No se encontró train_pipeline.py")
    log.parent.mkdir(parents=True, exist_ok=True)
    comando = [sys.executable, str(SCRIPT_PIPELINE), *argumentos]
    entorno = {**os.environ, "PYTHONUTF8": "1"}
    with log.open("wb") as salida:
        return subprocess.Popen(comando, stdout=salida, stderr=subprocess.STDOUT, cwd=str(RAIZ_FSAI), shell=False, env=entorno)  # noqa: S603


def leer_json(ruta: Path) -> dict[str, Any]:
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def cola_texto(ruta: Path, max_bytes: int = 4000) -> str:
    try:
        with ruta.open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def en_ejecucion(progreso: dict[str, Any], max_inactividad_s: float = 180.0) -> bool:
    """True si progreso.json dice 'ejecutando' y se actualizó hace poco (detecta corridas de otras sesiones)."""
    if progreso.get("estado") != "ejecutando":
        return False
    try:
        t = time.mktime(time.strptime(progreso["actualizado"], "%Y-%m-%dT%H:%M:%S"))
    except (KeyError, ValueError):
        return False
    return time.time() - t < max_inactividad_s


def audit(accion: str, entidad: str | None = None, entidad_id: str | None = None, detalle: dict | None = None) -> None:
    registrar_auditoria(accion, entidad, entidad_id, detalle)


def formatear_ic(m: dict[str, float], d: int = 3) -> str:
    if not m or pd.isna(m.get("media")):
        return "—"
    ic = "" if pd.isna(m.get("ic_inf")) else f" [{m['ic_inf']:.{d}f}; {m['ic_sup']:.{d}f}]"
    sd = "" if pd.isna(m.get("sd")) else f" ± {m['sd']:.{d}f}"
    return f"{m['media']:.{d}f}{sd}{ic}"
