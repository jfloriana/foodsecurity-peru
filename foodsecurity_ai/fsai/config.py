"""config.py - Constantes globales, listas de variables y utilidades de configuración.

Fuente única de verdad para semillas, hiperparámetros del protocolo, nombres de
columnas y rutas. Ningún otro módulo debe redefinir estas constantes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Constantes del protocolo (CONTRATO)
# --------------------------------------------------------------------------
SEED: int = 42                 # misma semilla para los 5 modelos y todas las particiones
N_SPLITS: int = 5              # pliegues externos de GroupKFold
N_REPETICIONES: int = 4        # repeticiones de V1/V1c (semillas SEED+rep) -> 5x4 = 20 bloques pareados
N_TRIALS: int = 100            # ensayos de Optuna por modelo y pliegue externo
INNER_FOLDS: int = 3           # pliegues de la CV interna (GroupKFold por hogar_uid)
N_BOOT: int = 1000             # réplicas bootstrap
ALPHA: float = 0.05            # nivel de significancia
CLASES: list[int] = [0, 1, 2, 3]
CLASE_SEVERA: int = 3
ESPECIFICIDAD_MIN: float = 0.70   # restricción al fijar el umbral de la clase severa
UMBRAL_BRECHA_TPR: float = 0.05   # alerta de equidad: 5 puntos porcentuales

# --------------------------------------------------------------------------
# Nombres de columnas
# --------------------------------------------------------------------------
COL_OBJETIVO: str = "fies_nivel"
COL_BINARIA: str = "ia_binaria"
COL_PESO: str = "factor_expansion"
COL_GRUPO: str = "hogar_uid"
LLAVES_INEI: list[str] = ["conglome", "vivienda", "hogar"]

FEATURES_FIES: list[str] = [f"fies_q{i}" for i in range(1, 9)] + ["fies_raw_score"]

# Variables derivadas del objetivo o identificadores: NUNCA son predictoras.
COLUMNAS_ID_TERRITORIALES: list[str] = [
    "conglome", "vivienda", "hogar", "hogar_uid", "ubigeo", "provincia", "distrito",
]
COLUMNAS_PROHIBIDAS: list[str] = [
    "fies_nivel", "ia_binaria", "fies_rasch_logit", "factor_expansion",
]

# Socioeconómicas numéricas (incluye indicadores 0/1 tratados como numéricos).
FEATURES_SOCIO_NUM: list[str] = [
    "ingreso_pc", "gasto_pc", "gasto_alim_pc", "share_alimentario", "pobreza_monetaria",
    "n_ocupados", "informalidad", "desempleo", "dependencia_economica",
    "nivel_educ_jefe", "asistencia_escolar",
    "n_miembros", "n_menores5", "n_menores18", "edad_jefe", "n_adultos_mayores",
    "agua_potable", "saneamiento", "hacinamiento",
]
FEATURES_SOCIO_CAT: list[str] = [
    "departamento", "area", "region_natural", "sexo_jefe", "tipo_vivienda", "material_vivienda",
]
FEATURES_SOCIO: list[str] = FEATURES_SOCIO_NUM + FEATURES_SOCIO_CAT

ESCENARIOS: list[str] = ["socio", "fies", "hibrido"]

# --- Fase 2 (V2): escenario ADICIONAL 'socio_terr' (no altera la definición de 'socio' ni de 'todos') -------------------
# Indicadores territoriales a nivel DEPARTAMENTO (todos con prefijo ``terr_``; ver fsai.fuentes_territoriales):
# ENDES 2024 (INEI), OPHI MPI (ENDES 2023), NASA POWER (anomalía de precipitación 2025 vs 2015-2024) y ONI de NOAA por mes
# de encuesta. NINGUNO se deriva de ENAHO/FIES ni del objetivo. Desfase temporal declarado: ENDES 2024 / MPI 2023 vs ENAHO 2025.
FEATURES_SOCIO_TERR_NUM: list[str] = [
    "terr_desnutricion_cronica_pct", "terr_anemia_ninos_pct", "terr_anemia_mujeres_pct",
    "terr_hogares_programa_social_pct", "terr_hogares_juntos_pct", "terr_hogares_pension65_pct", "terr_hogares_vaso_leche_pct",
    "terr_mpi_indice", "terr_mpi_incidencia_pct", "terr_mpi_intensidad_pct",
    "terr_precip_anom_pct", "terr_precip_anom_z", "terr_oni_mes",
]
# Variables derivadas (ingeniería de variables, SOLO escenario socio_terr; ver preprocessing.agregar_ingenieria).
# Se calculan únicamente a partir de predictores socioeconómicos (nunca del objetivo ni de FIES).
FEATURES_INGENIERIA_NUM: list[str] = [
    "log_ingreso_pc", "log_gasto_pc", "log_gasto_alim_pc", "log_ingreso_por_adulto_eq",
    "razon_dependencia", "asistencia_escolar_faltante", "tiene_menor5",
]
FEATURES_INGENIERIA_CAT: list[str] = ["grupo_edad_jefe", "grupo_educ_jefe"]
ESCENARIOS_EXTRA: list[str] = ["socio_terr"]
ESCENARIOS_TODOS: list[str] = ESCENARIOS + ESCENARIOS_EXTRA     # válidos para dividir_features/CLI; 'todos' sigue siendo ESCENARIOS
# Esquemas de la sección 8.1: V1 = GroupKFold(5) por hogar_uid (repetido N_REPETICIONES veces);
# V1c = sensibilidad de V1 agrupando por 'conglome'; V2 = validación geográfica (variantes lodo/macro);
# V3 = partición estratificada urbano/rural con métricas por estrato; V4 = temporal 2024->2025
# (solo con columna 'anio' de >=2 años comparables); V5 = ponderación con factor_expansion
# (NO es una partición: modo de métricas aplicable a cualquier esquema, siempre se reporta ponderado y no ponderado).
ESQUEMAS_VALIDACION: list[str] = ["V1", "V1c", "V2", "V3", "V4", "V5"]
VARIANTES_V2: list[str] = ["lodo", "macro"]    # lodo = dejar fuera un departamento; macro = costa/sierra/selva
# Claves ejecutables (particiones reales). V2 se expande en sus dos variantes; V5 no genera particiones.
ESQUEMAS_PARTICION: list[str] = ["V1", "V1c", "V2_lodo", "V2_macro", "V3", "V4"]
MODELOS: list[str] = ["M1", "M2", "M3", "M4", "M5"]
# M6 = HistGradientBoosting (scikit-learn); modelo EXTRA opcional: no forma parte del protocolo M1-M5 ni de 'todos'.
MODELOS_EXTRA: list[str] = ["M6"]
MODELOS_TODOS: list[str] = MODELOS + MODELOS_EXTRA

# Mapa opcional código INEI -> nombre analítico. Se deja vacío a propósito:
# los códigos de variable cambian entre años y deben validarse contra el
# diccionario oficial de la ENAHO del año analizado [POR VALIDAR].
# Para otro año/archivo se llena (sin tocar el código) con un JSON {"codigo_inei": "nombre_analitico"} cuya ruta se
# da en la variable de entorno FSAI_MAPA_COLUMNAS; lo consumen ``contrato_esquema`` y ``datasets.leer_tabla``.
def _cargar_mapa_columnas() -> dict[str, str]:
    ruta = os.getenv("FSAI_MAPA_COLUMNAS", "")
    if not ruta:
        return {}
    try:
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        if not isinstance(datos, dict):
            raise ValueError("el JSON debe ser un objeto {origen: destino}")
        return {str(k): str(v) for k, v in datos.items()}
    except (OSError, ValueError) as exc:
        logging.getLogger("fsai").warning("FSAI_MAPA_COLUMNAS ignorado (%s): %s", ruta, exc)
        return {}


MAPA_COLUMNAS: dict[str, str] = _cargar_mapa_columnas()

# --------------------------------------------------------------------------
# Rutas y base de datos
# --------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = Path(os.getenv("FSAI_DATA_DIR", str(BASE_DIR / "data")))
ARTIFACTS_DIR: Path = Path(os.getenv("FSAI_ARTIFACTS_DIR", str(BASE_DIR / "artifacts")))
# Ejemplo: postgresql+psycopg2://usuario:clave@localhost:5432/fsai_db
DATABASE_URL: str = os.getenv("FSAI_DATABASE_URL", "")
N_HILOS: int = int(os.getenv("FSAI_N_HILOS", "-1"))   # hilos por modelo (RF/CatBoost/XGBoost/PyTorch); -1 = todos (para correr varios modelos en paralelo sin sobresuscribir CPU)
SALT: str = os.getenv("FSAI_SALT", "")   # sal secreta del seudónimo hogar_uid (no se versiona)

# --------------------------------------------------------------------------
# Módulos por fases (Fase 1-7): partición 70/15/15 y carpetas de trabajo
# --------------------------------------------------------------------------
PROPORCIONES_PARTICION: tuple[float, float, float] = (0.70, 0.15, 0.15)   # Desarrollo / Validación / Prueba
NOMBRES_PARTICION: tuple[str, str, str] = ("desarrollo", "validacion", "prueba")
N_SPLITS_EXTERNOS: int = N_SPLITS      # CV anidada: 5 pliegues externos (dentro del 70 % de Desarrollo)
DIR_DATOS: Path = Path(os.getenv("FSAI_DIR_DATOS", str(ARTIFACTS_DIR.parent / "datos")))
DIR_CARGAS: Path = ARTIFACTS_DIR / "cargas"          # archivos tabulares subidos desde la UI (Fase 1)
DIR_FASES: Path = ARTIFACTS_DIR / "fases"            # corridas de Fase 2 (particiones, CV anidada, modelos)
DIR_METADATA: Path = ARTIFACTS_DIR / "metadata"      # editor de metadata (JSON por dataset)
DIR_OPTUNA: Path = ARTIFACTS_DIR / "optuna"          # estudios Optuna persistentes (SQLite) de la Fase 7

# --------------------------------------------------------------------------
# Fuentes públicas adicionales (Fase 2 V2)
# --------------------------------------------------------------------------
REPO_DIR: Path = BASE_DIR.parent
DIR_DATOS_PUBLICOS: Path = Path(os.getenv("FSAI_DIR_DATOS_PUBLICOS", str(REPO_DIR / "datos_publicos")))   # descargas (no versionado)
DIR_DERIVADOS: Path = Path(os.getenv("FSAI_DIR_DERIVADOS", str(REPO_DIR / "datos_derivados")))             # tablas derivadas pequeñas (versionadas)
RUTA_TABLA_TERRITORIAL: Path = DIR_DERIVADOS / "indicadores_territoriales_departamento.csv"
RUTA_TABLA_ONI: Path = DIR_DERIVADOS / "oni_mensual.csv"
GEOJSON_DEPARTAMENTOS_INCLUIDO: Path = REPO_DIR / "app_streamlit" / "app" / "assets" / "geo" / "peru_departamental.geojson"
# Valor efectivo: FSAI_GEOJSON_DEPARTAMENTOS si existe; si no, el GeoJSON incluido en la app (ver fsai.geo.ruta_geojson).
_geo_env = os.getenv("FSAI_GEOJSON_DEPARTAMENTOS", "").strip()
RUTA_GEOJSON_DEPARTAMENTOS: Path = Path(_geo_env) if _geo_env and Path(_geo_env).is_file() else GEOJSON_DEPARTAMENTOS_INCLUIDO

logger = logging.getLogger("fsai")


def configurar_logging(ruta_log: Path | None = None, nivel: int = logging.INFO) -> None:
    """Configura el registro en consola y, opcionalmente, en archivo."""
    formato = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if ruta_log is not None:
        ruta_log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(ruta_log, encoding="utf-8"))
    logging.basicConfig(level=nivel, format=formato, handlers=handlers, force=True)


def hash_configuracion(cfg: dict[str, Any]) -> str:
    """Hash SHA-256 (16 hex) de una configuración serializada de forma canónica."""
    texto = json.dumps(cfg, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16]


def constantes_protocolo() -> dict[str, Any]:
    """Devuelve las constantes que definen el experimento (entran al hash)."""
    return {
        "SEED": SEED, "N_SPLITS": N_SPLITS, "N_REPETICIONES": N_REPETICIONES, "N_TRIALS": N_TRIALS, "INNER_FOLDS": INNER_FOLDS,
        "N_BOOT": N_BOOT, "ALPHA": ALPHA, "CLASES": CLASES, "CLASE_SEVERA": CLASE_SEVERA,
        "ESPECIFICIDAD_MIN": ESPECIFICIDAD_MIN, "UMBRAL_BRECHA_TPR": UMBRAL_BRECHA_TPR,
        "FEATURES_FIES": FEATURES_FIES, "FEATURES_SOCIO": FEATURES_SOCIO,
    }
