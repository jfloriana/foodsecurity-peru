"""fuentes_territoriales.py - Indicadores territoriales por DEPARTAMENTO a partir de fuentes públicas adicionales.

Construye (con funciones puras, sin red) una tabla con una fila por departamento (clave ``cod_departamento`` de 2 dígitos,
la misma de ``fuentes_publicas.DEPARTAMENTOS``) desde los archivos que baja ``scripts/descargar_fuentes_adicionales.py``:

| columna                              | fuente (año)                                   | definición |
|--------------------------------------|------------------------------------------------|------------|
| terr_desnutricion_cronica_pct        | ENDES 2024, RECH6 (HC70) + RECH0 (HV005)       | % niños 0-59 m con talla/edad z < -2 (OMS), solo datos válidos, ponderado HV005 |
| terr_anemia_ninos_pct                | ENDES 2024, RECH6 (HC57) + RECH0               | % niños con anemia (HC57 en 1-3 sobre 1-4), ajustada por altitud (RM 363-2022-MINSA) |
| terr_anemia_mujeres_pct              | ENDES 2024, RECH5 (HA57) + RECH0               | % mujeres 15-49 con anemia (HA57 en 1-3 sobre 1-4) |
| terr_hogares_programa_social_pct     | ENDES 2024, módulo 1641 (QH91..QH106)          | % hogares con algún miembro beneficiario de Beca 18, Trabaja Perú, Juntos, Pensión 65, Vaso de Leche, Comedor o Cuna Más |
| terr_hogares_juntos_pct, ..._pension65_pct, ..._vaso_leche_pct | ENDES 2024, módulo 1641 | % hogares beneficiarios de cada programa |
| terr_mpi_indice / _incidencia_pct / _intensidad_pct | OPHI MPI (ENDES 2023, CC0)       | índice de pobreza multidimensional, incidencia (H) e intensidad (A) |
| terr_precip_anom_pct / _z            | Open-Meteo archive (ERA5), centroide, 2025 vs 2015-2024 | anomalía de la precipitación acumulada de 12 meses (calendario de la encuesta) frente a la climatología |
| terr_oni_mes                         | NOAA CPC ONI (nacional)                        | ONI del trimestre móvil centrado en el mes de la encuesta (columna por (anio, mes), no por departamento) |

Notas honestas
* DESFASE TEMPORAL: ENDES 2024 y MPI (ENDES 2023) describen un año distinto del de ENAHO 2025; son indicadores estructurales
  del departamento, no mediciones de 2025. La anomalía de precipitación y el ONI SÍ corresponden al año/mes de la ENAHO.
* Los indicadores son constantes dentro de un departamento (25 valores distintos): aportan información de contexto,
  pero no pueden mejorar la discriminación dentro de un departamento y se solapan con la variable categórica ``departamento``.
* PRECIPITACIÓN: NASA POWER (MERRA-2) se descargó primero, pero su serie 2025 muestra un salto no verosímil (p. ej. Lima ~166 mm
  vs 27-70 mm en 2015-2024; Ayacucho 757 vs 300-400 mm), de modo que NO se usa para el rasgo (se conserva solo como diagnóstico en el
  informe). Se usa ERA5 vía Open-Meteo, cuya serie no presenta ese salto. ERA5 sobreestima la lluvia absoluta en los Andes/costa
  (celdas de ~25 km): el rasgo es una ANOMALÍA relativa a la climatología propia del punto, no un valor absoluto en mm.
* Anti-fuga: ninguna columna proviene de ENAHO/FIES ni del objetivo; ``verificar_sin_fuga`` lo asserta.
* Lima (15) agrupa Lima Metropolitana y Lima Provincias (UBIGEO[:2]); Callao (07) va aparte; igual que en la ENAHO.
* Todo archivo descargado es NO confiable: aquí solo se lee como datos (CSV/JSON/zip sin extraer a disco).
"""
from __future__ import annotations

import calendar
import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fsai import config as C
from fsai.fuentes_publicas import DEPARTAMENTOS

logger = logging.getLogger(__name__)

ANIO_ENDES = 2024
ANIO_MPI = 2023
ANIO_ENAHO = 2025
CLIMATOLOGIA = (2015, 2024)             # ventana de referencia de la anomalía de precipitación
MAX_BYTES_MIEMBRO_ZIP = 300_000_000     # tope por archivo leído desde un zip (anti zip-bomb)
COL_CLAVE = "cod_departamento"

# Metadatos por columna (fuente/año/definición): se guardan con la tabla y alimentan docs/FUENTES_ADICIONALES.md.
INDICADORES: dict[str, dict[str, str]] = {
    "terr_desnutricion_cronica_pct": {"fuente": "ENDES_2024_RECH6", "anio": "2024", "unidad": "%", "descripcion": "Desnutrición crónica (HC70 < -200) en niños 0-59 m con dato válido, ponderado HV005"},
    "terr_anemia_ninos_pct": {"fuente": "ENDES_2024_RECH6", "anio": "2024", "unidad": "%", "descripcion": "Anemia en niños (HC57 1-3 sobre 1-4, ajustada por altitud), ponderado HV005"},
    "terr_anemia_mujeres_pct": {"fuente": "ENDES_2024_RECH5", "anio": "2024", "unidad": "%", "descripcion": "Anemia en mujeres 15-49 (HA57 1-3 sobre 1-4), ponderado HV005"},
    "terr_hogares_programa_social_pct": {"fuente": "ENDES_2024_PS_HOGAR", "anio": "2024", "unidad": "%", "descripcion": "Hogares con algún programa social (Beca 18, Trabaja Perú, Juntos, Pensión 65, Vaso de Leche, Comedor, Cuna Más)"},
    "terr_hogares_juntos_pct": {"fuente": "ENDES_2024_PS_HOGAR", "anio": "2024", "unidad": "%", "descripcion": "Hogares beneficiarios de Juntos (QH95)"},
    "terr_hogares_pension65_pct": {"fuente": "ENDES_2024_PS_HOGAR", "anio": "2024", "unidad": "%", "descripcion": "Hogares con beneficiario de Pensión 65 (QH99)"},
    "terr_hogares_vaso_leche_pct": {"fuente": "ENDES_2024_PS_HOGAR", "anio": "2024", "unidad": "%", "descripcion": "Hogares con beneficiario del Vaso de Leche (QH101)"},
    "terr_mpi_indice": {"fuente": "OPHI_MPI_HDX", "anio": "2023", "unidad": "índice 0-1", "descripcion": "MPI (OPHI) por departamento, ENDES 2023"},
    "terr_mpi_incidencia_pct": {"fuente": "OPHI_MPI_HDX", "anio": "2023", "unidad": "%", "descripcion": "Incidencia de pobreza multidimensional (Headcount Ratio)"},
    "terr_mpi_intensidad_pct": {"fuente": "OPHI_MPI_HDX", "anio": "2023", "unidad": "%", "descripcion": "Intensidad de las privaciones entre los pobres multidimensionales"},
    "terr_precip_anom_pct": {"fuente": "OPENMETEO_ERA5", "anio": "2025", "unidad": "%", "descripcion": "(P12m - media 2015-2024) / media 2015-2024 * 100, en el centroide del departamento"},
    "terr_precip_anom_z": {"fuente": "OPENMETEO_ERA5", "anio": "2025", "unidad": "desv. est.", "descripcion": "(P12m - media 2015-2024) / DE 2015-2024 (anomalía estandarizada)"},
    "terr_oni_mes": {"fuente": "NOAA_CPC_ONI", "anio": "por mes de encuesta", "unidad": "°C", "descripcion": "ONI (anomalía Niño 3.4, trimestre móvil centrado en el mes), unido por (anio, mes)"},
}
FUENTES_PROHIBIDAS = ("ENAHO", "FIES")   # prefijos de fuente que NUNCA pueden alimentar un indicador territorial


# --------------------------------------------------------------------------
# Anti-fuga
# --------------------------------------------------------------------------
def verificar_sin_fuga(columnas: list[str] | None = None, indicadores: dict[str, dict[str, str]] | None = None) -> None:
    """Asserta que ningún indicador territorial proviene de ENAHO/FIES ni coincide con el objetivo/identificadores."""
    indicadores = INDICADORES if indicadores is None else indicadores
    columnas = list(indicadores) if columnas is None else columnas
    prohibidas = set(C.COLUMNAS_PROHIBIDAS) | set(C.COLUMNAS_ID_TERRITORIALES) | set(C.FEATURES_FIES)
    malas = [c for c in columnas if c in prohibidas or "fies" in c.lower() or c.startswith("ia_")]
    if malas:
        raise ValueError(f"Columnas territoriales prohibidas (objetivo/FIES/identificadores): {malas}")
    for c in columnas:
        if c not in indicadores:
            raise ValueError(f"Columna territorial sin metadatos de fuente: {c}")
        fuente = indicadores[c]["fuente"].upper()
        if fuente.startswith(FUENTES_PROHIBIDAS):
            raise ValueError(f"{c}: la fuente {fuente} está prohibida (derivar de ENAHO/FIES sería fuga)")


# --------------------------------------------------------------------------
# GeoJSON: centroides (para consultar clima por punto)
# --------------------------------------------------------------------------
def _anillo_area_centroide(anillo: list[list[float]]) -> tuple[float, float, float]:
    """Área con signo y centroide plano (shoelace) de un anillo [[lon, lat], ...]."""
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(anillo[:-1], anillo[1:]):
        t = x0 * y1 - x1 * y0
        a += t
        cx += (x0 + x1) * t
        cy += (y0 + y1) * t
    if a == 0:
        xs = [p[0] for p in anillo]
        ys = [p[1] for p in anillo]
        return 0.0, float(np.mean(xs)), float(np.mean(ys))
    return a / 2.0, cx / (6.0 * (a / 2.0)), cy / (6.0 * (a / 2.0))


def centroide_geometria(geom: dict[str, Any]) -> tuple[float, float]:
    """Centroide (lon, lat) ponderado por área de un Polygon/MultiPolygon (anillos exteriores menos huecos)."""
    polis = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    area_t = sx = sy = 0.0
    for poli in polis:
        for i, anillo in enumerate(poli):
            a, cx, cy = _anillo_area_centroide(anillo)
            signo = 1.0 if i == 0 else -1.0
            a = abs(a) * signo
            area_t += a
            sx += a * cx
            sy += a * cy
    if area_t == 0:
        raise ValueError("Geometría sin área")
    return sx / area_t, sy / area_t


def centroides_departamentos(ruta_geojson: Path, clave: str = "FIRST_IDDP") -> dict[str, tuple[float, float]]:
    """{cod_departamento: (lon, lat)} desde el GeoJSON departamental (propiedad ``clave`` = código de 2 dígitos)."""
    g = json.loads(Path(ruta_geojson).read_text(encoding="utf-8"))
    out: dict[str, tuple[float, float]] = {}
    for f in g["features"]:
        cod = str(f["properties"][clave]).zfill(2)
        out[cod] = centroide_geometria(f["geometry"])
    return out


# --------------------------------------------------------------------------
# Lectura segura de CSV dentro de zips (sin extraer a disco)
# --------------------------------------------------------------------------
def leer_csv_zip(ruta_zip: Path, sufijo: str, usecols: list[str] | None = None, **kw: Any) -> pd.DataFrame:
    """Lee (todo como texto) el CSV latin-1 del zip cuyo nombre termina en ``sufijo``; quita el BOM de la primera columna."""
    with zipfile.ZipFile(ruta_zip) as zf:
        miembros = [i for i in zf.infolist() if i.filename.lower().endswith(sufijo.lower())]
        if len(miembros) != 1:
            raise FileNotFoundError(f"{ruta_zip.name}: se esperaba 1 archivo '*{sufijo}', hay {len(miembros)}")
        m = miembros[0]
        if m.file_size > MAX_BYTES_MIEMBRO_ZIP:
            raise ValueError(f"{m.filename} excede el tope de {MAX_BYTES_MIEMBRO_ZIP} bytes")
        with zf.open(m) as fh:
            d = pd.read_csv(io.BytesIO(fh.read(MAX_BYTES_MIEMBRO_ZIP + 1)), encoding="latin-1", dtype=str, low_memory=False, **kw)
    d.columns = [c.lstrip("﻿").strip() for c in d.columns]
    if usecols is not None:
        faltan = [c for c in usecols if c not in d.columns]
        if faltan:
            raise KeyError(f"{m.filename}: faltan columnas {faltan}")
        d = d[usecols]
    return d


def _pct_ponderado(df: pd.DataFrame, flag: str, w: str, por: str = COL_CLAVE) -> pd.Series:
    """% ponderado de ``flag`` (0/1, sin NaN) por departamento."""
    g = df.assign(_fw=df[flag].astype(float) * df[w]).groupby(por)
    return 100.0 * g["_fw"].sum() / g[w].sum()


# --------------------------------------------------------------------------
# ENDES 2024
# --------------------------------------------------------------------------
def _hogares_endes(ruta_m1629: Path) -> pd.DataFrame:
    """HHID -> (cod_departamento, peso). Peso = HV005/1e6 (convención ENDES/DHS)."""
    r0 = leer_csv_zip(ruta_m1629, "RECH0_2024.csv", ["HHID", "UBIGEO", "HV005"])
    r0["HHID"] = r0["HHID"].str.strip()
    r0[COL_CLAVE] = r0["UBIGEO"].str.strip().str.zfill(6).str[:2]
    r0["w"] = pd.to_numeric(r0["HV005"], errors="coerce") / 1e6
    r0 = r0[r0["w"] > 0]
    if r0["HHID"].duplicated().any():
        raise ValueError("RECH0: HHID duplicado")
    return r0[["HHID", COL_CLAVE, "w"]]


def indicadores_endes_talla_anemia(ruta_m1629: Path, ruta_m1638: Path) -> pd.DataFrame:
    """Desnutrición crónica, anemia infantil y anemia de mujeres 15-49 por departamento (ponderados HV005)."""
    hog = _hogares_endes(ruta_m1629)
    r6 = leer_csv_zip(ruta_m1638, "RECH6_2024.csv", ["HHID", "HC1", "HC70", "HC57"])
    r6["HHID"] = r6["HHID"].str.strip()
    for c in ("HC1", "HC70", "HC57"):
        r6[c] = pd.to_numeric(r6[c], errors="coerce")
    r6 = r6.merge(hog, on="HHID", how="inner", validate="m:1")
    r6 = r6[(r6["HC1"] >= 0) & (r6["HC1"] <= 59)]
    # talla/edad: z*100; códigos >= 9990 = faltante/no válido/fuera de rango (DHS); no se imputa
    talla = r6[(r6["HC70"] > -600) & (r6["HC70"] < 600)].copy()
    talla["f"] = (talla["HC70"] < -200).astype(int)
    anem = r6[r6["HC57"].between(1, 4)].copy()
    anem["f"] = (anem["HC57"] <= 3).astype(int)
    r5 = leer_csv_zip(ruta_m1638, "RECH5_2024.csv", ["HHID", "HA1", "HA57"])
    r5["HHID"] = r5["HHID"].str.strip()
    r5["HA57"] = pd.to_numeric(r5["HA57"], errors="coerce")
    r5 = r5.merge(hog, on="HHID", how="inner", validate="m:1")
    anem_m = r5[r5["HA57"].between(1, 4)].copy()
    anem_m["f"] = (anem_m["HA57"] <= 3).astype(int)
    out = pd.DataFrame({
        "terr_desnutricion_cronica_pct": _pct_ponderado(talla, "f", "w"),
        "terr_anemia_ninos_pct": _pct_ponderado(anem, "f", "w"),
        "terr_anemia_mujeres_pct": _pct_ponderado(anem_m, "f", "w"),
        "n_ninos_talla": talla.groupby(COL_CLAVE).size(),
        "n_ninos_anemia": anem.groupby(COL_CLAVE).size(),
        "n_mujeres_anemia": anem_m.groupby(COL_CLAVE).size(),
    })
    logger.info("ENDES talla/anemia: %d departamentos; niños con talla válida %d de %d",
                len(out), len(talla), len(r6))
    return out


# Mapa verificado contra los archivos PS_* del módulo 1641 (n.º de hogares con respuesta 1 == hogares distintos del archivo):
# QH91 Beca 18 (48), QH93 Trabaja Perú (73), QH95 Juntos (5271 = filas con detalle QH96AC), QH99 Pensión 65 (2236),
# QH101 Vaso de Leche (7813), QH103 Comedor Popular (1166), QH106 Cuna Más/Wawa Wasi (705).
PROGRAMAS_PS: dict[str, str] = {"QH91": "beca18", "QH93": "trabaja", "QH95": "juntos", "QH99": "pension65",
                                "QH101": "vaso_leche", "QH103": "comedor", "QH106": "cuna_mas"}


def indicadores_programas_sociales(ruta_m1629: Path, ruta_m1641: Path) -> pd.DataFrame:
    """% de hogares con algún programa social / Juntos / Pensión 65 / Vaso de Leche por departamento (ponderado HV005)."""
    hog = _hogares_endes(ruta_m1629)
    ps = leer_csv_zip(ruta_m1641, "Programas Sociales x Hogar_2024.csv", ["HHID"] + list(PROGRAMAS_PS))
    ps["HHID"] = ps["HHID"].str.strip()
    for q in PROGRAMAS_PS:
        ps[q] = pd.to_numeric(ps[q], errors="coerce")
    ps = ps.merge(hog, on="HHID", how="inner", validate="m:1")
    ps["alguno"] = (ps[list(PROGRAMAS_PS)] == 1).any(axis=1).astype(int)
    # por programa: denominador = hogares con respuesta válida (1 o 2); 'alguno' usa todos los hogares del módulo
    out = pd.DataFrame({"terr_hogares_programa_social_pct": _pct_ponderado(ps, "alguno", "w")})
    for q, nombre, col in (("QH95", "juntos", "terr_hogares_juntos_pct"), ("QH99", "pension65", "terr_hogares_pension65_pct"),
                           ("QH101", "vaso_leche", "terr_hogares_vaso_leche_pct")):
        v = ps[ps[q].isin([1, 2])].copy()
        v["f"] = (v[q] == 1).astype(int)
        out[col] = _pct_ponderado(v, "f", "w")
    out["n_hogares_ps"] = ps.groupby(COL_CLAVE).size()
    return out


# --------------------------------------------------------------------------
# OPHI MPI
# --------------------------------------------------------------------------
def indicadores_mpi(ruta_csv: Path) -> pd.DataFrame:
    d = pd.read_csv(ruta_csv, dtype={"Admin 1 PCode": str})
    d = d[d["Admin 1 PCode"].notna()].copy()
    d[COL_CLAVE] = d["Admin 1 PCode"].str.strip().str.replace("PE", "", regex=False).str.zfill(2)
    out = pd.DataFrame({
        "terr_mpi_indice": pd.to_numeric(d["MPI"], errors="coerce").to_numpy(),
        "terr_mpi_incidencia_pct": pd.to_numeric(d["Headcount Ratio"], errors="coerce").to_numpy(),
        "terr_mpi_intensidad_pct": pd.to_numeric(d["Intensity of Deprivation"], errors="coerce").to_numpy(),
    }, index=d[COL_CLAVE].to_numpy())
    out.index.name = COL_CLAVE
    return out


# --------------------------------------------------------------------------
# NASA POWER: anomalía de precipitación
# --------------------------------------------------------------------------
def precipitacion_mensual_mm(ruta_json: Path) -> pd.Series:
    """Serie mensual (mm/mes) indexada por 'YYYYMM' desde un JSON de NASA POWER (PRECTOTCORR en mm/día). -999 -> NaN."""
    j = json.loads(Path(ruta_json).read_text(encoding="utf-8"))
    p = j["properties"]["parameter"]["PRECTOTCORR"]
    s = pd.Series({k: float(v) for k, v in p.items() if not k.endswith("13")})   # 'YYYY13' = promedio anual (se omite)
    s[s <= -990] = np.nan
    dias = pd.Series([calendar.monthrange(int(k[:4]), int(k[4:]))[1] for k in s.index], index=s.index)
    return s * dias


def precipitacion_mensual_openmeteo(ruta_json: Path) -> pd.Series:
    """Serie mensual (mm/mes) indexada por 'YYYYMM' desde un JSON de Open-Meteo archive (``daily.precipitation_sum``, mm/día)."""
    j = json.loads(Path(ruta_json).read_text(encoding="utf-8"))
    d = pd.Series(j["daily"]["precipitation_sum"], index=pd.to_datetime(j["daily"]["time"]), dtype=float)
    mensual = d.groupby(d.index.strftime("%Y%m")).agg(["sum", "count"])
    n_dias = pd.Series([calendar.monthrange(int(k[:4]), int(k[4:]))[1] for k in mensual.index], index=mensual.index)
    s = mensual["sum"].where(mensual["count"] == n_dias)          # un mes con días faltantes no es válido (NaN)
    return s


def anomalia_precipitacion(serie_mm: pd.Series, anio: int = ANIO_ENAHO, clim: tuple[int, int] = CLIMATOLOGIA) -> dict[str, float]:
    """Acumulado del año ``anio`` frente a la climatología anual ``clim`` (excluye el año objetivo). Exige 12 meses completos."""
    def total(a: int) -> float:
        meses = [f"{a}{m:02d}" for m in range(1, 13)]
        v = serie_mm.reindex(meses)
        return float(v.sum()) if v.notna().all() else float("nan")
    totales = np.array([total(a) for a in range(clim[0], clim[1] + 1) if a != anio])
    totales = totales[~np.isnan(totales)]
    p = total(anio)
    if len(totales) < 5 or np.isnan(p):
        return {"precip_anio_mm": p, "precip_clim_mm": float("nan"), "terr_precip_anom_pct": float("nan"), "terr_precip_anom_z": float("nan")}
    media, de = float(totales.mean()), float(totales.std(ddof=1))
    return {"precip_anio_mm": p, "precip_clim_mm": media,
            "terr_precip_anom_pct": 100.0 * (p - media) / media if media > 0 else float("nan"),
            "terr_precip_anom_z": (p - media) / de if de > 0 else float("nan")}


def indicadores_precipitacion(dir_clima: Path, anio: int = ANIO_ENAHO, fuente: str = "openmeteo") -> pd.DataFrame:
    """Anomalía por departamento. ``fuente``: 'openmeteo' (``<dir>/openmeteo/<cod>.json``) o 'power' (diagnóstico, ver docstring del módulo)."""
    lector = precipitacion_mensual_openmeteo if fuente == "openmeteo" else precipitacion_mensual_mm
    filas = {}
    for cod in DEPARTAMENTOS:
        ruta = Path(dir_clima) / fuente / f"{cod}.json"
        if ruta.exists():
            filas[cod] = anomalia_precipitacion(lector(ruta), anio)
    out = pd.DataFrame.from_dict(filas, orient="index")
    out.index.name = COL_CLAVE
    return out


# --------------------------------------------------------------------------
# NOAA ONI
# --------------------------------------------------------------------------
_MES_CENTRAL = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def tabla_oni(ruta_txt: Path, desde: int = 2015) -> pd.DataFrame:
    """(anio, mes, terr_oni_mes): ONI del trimestre móvil centrado en el mes (DJF -> enero, ..., NDJ -> diciembre)."""
    filas = []
    for linea in Path(ruta_txt).read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        p = linea.split()
        if len(p) == 4 and p[0] in _MES_CENTRAL:
            filas.append({"anio": int(p[1]), "mes": _MES_CENTRAL[p[0]], "terr_oni_mes": float(p[3])})
    d = pd.DataFrame(filas)
    return d[d["anio"] >= desde].sort_values(["anio", "mes"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Ensamblado
# --------------------------------------------------------------------------
def construir_tabla_territorial(dir_publicos: Path | None = None, anio_enaho: int = ANIO_ENAHO) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """(tabla departamento, tabla ONI mensual, informe). Una fuente que falle NO detiene a las demás: queda en el informe."""
    d = Path(dir_publicos or C.DIR_DATOS_PUBLICOS)
    partes: list[pd.DataFrame] = []
    informe: dict[str, Any] = {"fuentes": {}}

    def intentar(nombre: str, f) -> None:
        try:
            df = f()
            partes.append(df)
            informe["fuentes"][nombre] = {"estado": "ok", "departamentos": int(len(df))}
        except Exception as exc:  # noqa: BLE001 (se reporta el motivo exacto)
            informe["fuentes"][nombre] = {"estado": "fallo", "motivo": f"{type(exc).__name__}: {exc}"}
            logger.warning("Fuente %s no procesada: %s", nombre, exc)

    z = d / "endes" / "2024"
    intentar("endes_talla_anemia", lambda: indicadores_endes_talla_anemia(z / "968-Modulo1629.zip", z / "968-Modulo1638.zip"))
    intentar("endes_programas_sociales", lambda: indicadores_programas_sociales(z / "968-Modulo1629.zip", z / "968-Modulo1641.zip"))
    intentar("ophi_mpi", lambda: indicadores_mpi(d / "ophi" / "per_mpi.csv"))
    intentar("openmeteo_era5_precipitacion", lambda: indicadores_precipitacion(d, anio_enaho, "openmeteo"))
    base = pd.DataFrame(index=pd.Index(list(DEPARTAMENTOS), name=COL_CLAVE))
    for p in partes:
        base = base.join(p, how="left")
    tabla = base.reset_index()
    tabla.insert(1, "departamento", tabla[COL_CLAVE].map(DEPARTAMENTOS))
    columnas_feat = [c for c in INDICADORES if c in tabla.columns]
    verificar_sin_fuga(columnas_feat)
    try:
        oni = tabla_oni(d / "noaa" / "oni.ascii.txt")
        informe["fuentes"]["noaa_oni"] = {"estado": "ok", "filas": int(len(oni))}
    except Exception as exc:  # noqa: BLE001
        oni = pd.DataFrame(columns=["anio", "mes", "terr_oni_mes"])
        informe["fuentes"]["noaa_oni"] = {"estado": "fallo", "motivo": f"{type(exc).__name__}: {exc}"}
    informe["columnas_indicador"] = columnas_feat
    informe["cobertura_departamentos"] = {c: int(tabla[c].notna().sum()) for c in columnas_feat}
    informe["rangos"] = {c: [float(tabla[c].min()), float(tabla[c].max())] for c in columnas_feat if tabla[c].notna().any()}
    try:   # diagnóstico: contraste con NASA POWER (no se usa como rasgo)
        pw = indicadores_precipitacion(d, anio_enaho, "power")["terr_precip_anom_pct"]
        er = tabla.set_index(COL_CLAVE)["terr_precip_anom_pct"]
        informe["diagnostico_nasa_power"] = {"mediana_anom_pct_power": float(pw.median()), "mediana_anom_pct_era5": float(er.median()),
                                             "spearman": float(pw.corr(er, method="spearman")),
                                             "decision": "NASA POWER descartada para el rasgo por salto en 2025 (ver docstring); se reporta solo como diagnóstico"}
    except Exception as exc:  # noqa: BLE001
        informe["diagnostico_nasa_power"] = {"estado": "no disponible", "motivo": f"{type(exc).__name__}: {exc}"}
    informe["anio_enaho"] = anio_enaho
    informe["ventana_precipitacion"] = f"12 meses de {anio_enaho} vs media anual {CLIMATOLOGIA[0]}-{CLIMATOLOGIA[1]}"
    return tabla, oni, informe


def verificar_rangos(tabla: pd.DataFrame) -> list[str]:
    """Chequeos de cordura (devuelve la lista de avisos; vacía = todo en rango plausible)."""
    av: list[str] = []
    reglas = {"terr_desnutricion_cronica_pct": (0, 60), "terr_anemia_ninos_pct": (0, 90), "terr_anemia_mujeres_pct": (0, 60),
              "terr_hogares_programa_social_pct": (0, 100), "terr_hogares_juntos_pct": (0, 100), "terr_hogares_pension65_pct": (0, 100),
              "terr_hogares_vaso_leche_pct": (0, 100), "terr_mpi_indice": (0, 1), "terr_mpi_incidencia_pct": (0, 100),
              "terr_mpi_intensidad_pct": (0, 100), "terr_precip_anom_pct": (-100, 500), "terr_precip_anom_z": (-6, 6)}
    for c, (lo, hi) in reglas.items():
        if c in tabla.columns and tabla[c].notna().any() and not tabla[c].dropna().between(lo, hi).all():
            av.append(f"{c} fuera de [{lo}, {hi}]: min={tabla[c].min():.3f} max={tabla[c].max():.3f}")
    if len(tabla) != 25:
        av.append(f"se esperaban 25 departamentos y hay {len(tabla)}")
    return av


def guardar_tabla_territorial(tabla: pd.DataFrame, oni: pd.DataFrame, informe: dict[str, Any], dir_salida: Path | None = None) -> dict[str, Path]:
    """Escribe los dos CSV (pequeños, versionables) y un manifiesto JSON con fuentes/años/definiciones y el informe."""
    out = Path(dir_salida or C.DIR_DERIVADOS)
    out.mkdir(parents=True, exist_ok=True)
    r_tab, r_oni, r_man = out / C.RUTA_TABLA_TERRITORIAL.name, out / C.RUTA_TABLA_ONI.name, out / "indicadores_territoriales_departamento.manifiesto.json"
    t = tabla.copy()
    for c in t.select_dtypes("float").columns:
        t[c] = t[c].round(4)
    t.to_csv(r_tab, index=False, encoding="utf-8", lineterminator="\n")
    oni.to_csv(r_oni, index=False, encoding="utf-8", lineterminator="\n")
    man = {"descripcion": "Indicadores a nivel departamento derivados de fuentes públicas; ver docs/FUENTES_ADICIONALES.md",
           "clave": COL_CLAVE, "indicadores": INDICADORES, "informe": informe,
           "aviso_temporal": "ENDES 2024 y MPI (ENDES 2023) NO son de 2025 (año de la ENAHO); precipitación y ONI sí corresponden a 2025.",
           "aviso_fuga": "Ningún indicador se deriva de ENAHO/FIES ni del objetivo (verificar_sin_fuga)."}
    r_man.write_text(json.dumps(man, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return {"tabla": r_tab, "oni": r_oni, "manifiesto": r_man}


# --------------------------------------------------------------------------
# Unión al dataset analítico
# --------------------------------------------------------------------------
def cargar_tabla_territorial(ruta: Path | None = None) -> pd.DataFrame:
    ruta = Path(ruta or C.RUTA_TABLA_TERRITORIAL)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la tabla territorial {ruta}. Ejecute scripts/descargar_fuentes_adicionales.py --derivar (ver LEEME).")
    return pd.read_csv(ruta, dtype={COL_CLAVE: str})


def _clave_departamento(df: pd.DataFrame) -> pd.Series:
    """Código de 2 dígitos: de ``ubigeo`` si existe; si no, del nombre 'departamento' (slug de DEPARTAMENTOS)."""
    if "ubigeo" in df.columns:
        return df["ubigeo"].astype("string").str.strip().str.zfill(6).str[:2]
    if "departamento" in df.columns:
        inv = {v: k for k, v in DEPARTAMENTOS.items()}
        return df["departamento"].astype("string").str.strip().str.lower().map(inv)
    raise KeyError("El dataset necesita 'ubigeo' o 'departamento' para la unión territorial.")


def unir_territorial(df: pd.DataFrame, tabla: pd.DataFrame | None = None, oni: pd.DataFrame | None = None,
                     columnas: list[str] | None = None, estricto: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Left-merge m:1 por departamento (y por (anio, mes) para el ONI). Devuelve (df_nuevo, reporte).

    Nunca pierde ni duplica filas (``validate='m:1'`` + aserción de longitud) ni altera columnas existentes. Las filas sin
    departamento reconocido quedan con NaN en las nuevas columnas y se cuentan en ``reporte['filas_sin_unir']``.
    Con ``estricto=True`` se lanza ValueError si alguna fila no se une.
    """
    columnas = list(columnas or [c for c in C.FEATURES_SOCIO_TERR_NUM if c != "terr_oni_mes"])
    verificar_sin_fuga(columnas + ["terr_oni_mes"])
    tabla = cargar_tabla_territorial() if tabla is None else tabla
    faltan = [c for c in columnas if c not in tabla.columns]
    if faltan:
        raise KeyError(f"La tabla territorial no tiene las columnas {faltan}")
    n0 = len(df)
    if tabla[COL_CLAVE].duplicated().any():
        raise ValueError("La tabla territorial tiene departamentos duplicados (se exige m:1)")
    ya = [c for c in columnas if c in df.columns]
    base = df.drop(columns=ya) if ya else df
    clave = _clave_departamento(base)
    aux = base.assign(**{COL_CLAVE: clave.to_numpy()})
    out = aux.merge(tabla[[COL_CLAVE] + columnas], on=COL_CLAVE, how="left", validate="m:1")
    out.index = base.index
    if len(out) != n0:
        raise AssertionError(f"La unión territorial cambió el número de filas: {n0} -> {len(out)}")
    sin_unir = out[columnas].isna().all(axis=1) if columnas else pd.Series(False, index=out.index)
    rep: dict[str, Any] = {"filas": n0, "filas_sin_unir": int(sin_unir.sum()), "departamentos_no_reconocidos":
                           sorted({str(v) for v in clave[clave.isna() | ~clave.isin(tabla[COL_CLAVE])].unique()}),
                           "columnas_agregadas": columnas, "columnas_sobrescritas": ya, "oni": "no aplicado"}
    if oni is None and C.RUTA_TABLA_ONI.exists():
        oni = pd.read_csv(C.RUTA_TABLA_ONI)
    if oni is not None and {"anio", "mes"} <= set(out.columns):
        o = oni[["anio", "mes", "terr_oni_mes"]].drop_duplicates(["anio", "mes"]).rename(columns={"anio": "_a", "mes": "_m"})
        o = o.astype({"_a": "Int64", "_m": "Int64"})
        claves = pd.DataFrame({"_a": pd.to_numeric(out["anio"], errors="coerce").astype("Int64"),
                               "_m": pd.to_numeric(out["mes"], errors="coerce").astype("Int64")}, index=out.index)
        unido = claves.merge(o, on=["_a", "_m"], how="left", validate="m:1")
        if len(unido) != n0:
            raise AssertionError("La unión del ONI cambió el número de filas")
        out["terr_oni_mes"] = unido["terr_oni_mes"].to_numpy()
        rep["oni"] = f"aplicado; filas sin ONI: {int(out['terr_oni_mes'].isna().sum())}"
    elif oni is not None:
        rep["oni"] = "no aplicado: el dataset no tiene 'anio'/'mes'"
    if estricto and rep["filas_sin_unir"]:
        raise ValueError(f"{rep['filas_sin_unir']} filas sin unir (departamentos: {rep['departamentos_no_reconocidos']})")
    if rep["filas_sin_unir"]:
        logger.warning("Unión territorial: %d de %d filas sin departamento reconocido (%s)", rep["filas_sin_unir"], n0, rep["departamentos_no_reconocidos"])
    return out, rep
