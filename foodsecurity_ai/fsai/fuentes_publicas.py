"""Fuentes publicas (INEI ENAHO 2025, FAO FIES agregado, Banco Mundial WDI).

Construye el dataset analitico con los nombres EXACTOS de CONTRATO.md a partir de los
microdatos publicos ENAHO 2025 anual (INEI, codigo de encuesta 1031) descargados SIN
cuenta desde https://proyectos.inei.gob.pe/microdatos/ .

Modulos usados (CSV, separador ';', codificacion latin-1, decimales con coma):
  * 1031-Modulo2082 -> Enaho01-2025-130.csv   Inseguridad alimentaria (FIES, P130_1..P130_8)
  * 1031-Modulo01   -> Enaho01-2025-100.csv   Vivienda y hogar
  * 1031-Modulo02   -> Enaho01-2025-200.csv   Miembros del hogar
  * 1031-Modulo03   -> Enaho01A-2025-300.csv  Educacion
  * 1031-Modulo05   -> Enaho01a-2025-500.csv  Empleo e ingresos
  * 1031-Modulo34   -> Sumaria-2025.csv       Variables calculadas (gasto, ingreso, pobreza)

Codificacion INEI observada (cedula CED-01-P130-2025.pdf y Diccionario_2025.pdf del propio
modulo): 1 = Si, 2 = No, 3 = No sabe, 4 = No responde. Aqui: 1->1, 2->0, 3 y 4 -> faltante
(NaN). NO se imputan. Los hogares sin modulo FIES (campos en blanco) quedan con fies_* = NaN.

Todo dato descargado se trata como NO confiable: solo se lee como texto/CSV (nunca se
ejecuta nada) y la extraccion de zip usa nombres planos (sin rutas) y un tope de tamano.
"""
from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from fsai.fies import UMBRALES_RAW, clasificar_fies

CODIGO_ENCUESTA_2025 = 1031
URL_BASE_INEI = "https://proyectos.inei.gob.pe/iinei/srienaho/descarga/CSV"
# modulo -> (archivo CSV esperado dentro del zip, descripcion)
MODULOS_ENAHO_2025: dict[str, tuple[str, str]] = {
    "01": ("Enaho01-2025-100.csv", "Caracteristicas de la vivienda y del hogar"),
    "02": ("Enaho01-2025-200.csv", "Caracteristicas de los miembros del hogar"),
    "03": ("Enaho01A-2025-300.csv", "Educacion"),
    "05": ("Enaho01a-2025-500.csv", "Empleo e ingresos"),
    "34": ("Sumaria-2025.csv", "Sumarias (variables calculadas)"),
    "2082": ("Enaho01-2025-130.csv", "Inseguridad alimentaria (FIES)"),
}
LLAVES = ["CONGLOME", "VIVIENDA", "HOGAR"]
ITEMS_FIES = [f"P130_{i}" for i in range(1, 9)]
# [POR VALIDAR] Umbrales del puntaje crudo (0-8): valores por defecto del proyecto (fsai.fies).
UMBRALES_FIES_DEFECTO: dict[str, float] = dict(UMBRALES_RAW)
SALT_DEFECTO_PUBLICO = "FSAI_SALT_NO_SECRETA_SOLO_DESARROLLO"  # sustituir con FSAI_HOGAR_SALT

DEPARTAMENTOS = {
    "01": "amazonas", "02": "ancash", "03": "apurimac", "04": "arequipa", "05": "ayacucho",
    "06": "cajamarca", "07": "callao", "08": "cusco", "09": "huancavelica", "10": "huanuco",
    "11": "ica", "12": "junin", "13": "la_libertad", "14": "lambayeque", "15": "lima",
    "16": "loreto", "17": "madre_de_dios", "18": "moquegua", "19": "pasco", "20": "piura",
    "21": "puno", "22": "san_martin", "23": "tacna", "24": "tumbes", "25": "ucayali",
}
# DOMINIO ENAHO: 1-3 costa (norte/centro/sur), 4-6 sierra, 7 selva, 8 Lima Metropolitana.
REGION_POR_DOMINIO = {"1": "costa", "2": "costa", "3": "costa", "4": "sierra", "5": "sierra",
                      "6": "sierra", "7": "selva", "8": "costa"}
# Grupo 1 "Alimentos" de la Sumaria (formula de GASHOG2D en Sumaria_2025.pdf), anual deflactado.
COLS_GASTO_ALIM = ["GRU11HD", "GRU12HD1", "GRU12HD2", "GRU13HD1", "GRU13HD2", "GRU13HD3",
                   "GRU14HD", "GRU14HD1", "GRU14HD2", "GRU14HD3", "GRU14HD4", "GRU14HD5", "GRU14HD6"]


def sha256_archivo(ruta: Path, bloque: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(bloque), b""):
            h.update(b)
    return h.hexdigest()


def extraer_zip_seguro(zip_path: Path, destino: Path, max_bytes: int = 2_000_000_000) -> list[Path]:
    """Extrae con nombres planos (descarta rutas) y tope de tamano; no ejecuta nada."""
    destino.mkdir(parents=True, exist_ok=True)
    salida: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            nombre = os.path.basename(info.filename.replace("\\", "/"))
            if info.is_dir() or not nombre or info.file_size > max_bytes:
                continue
            (destino / nombre).write_bytes(zf.read(info))
            salida.append(destino / nombre)
    return salida


def leer_csv_inei(ruta: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Lee un CSV INEI como texto (dtype=str); latin-1, separador ';'."""
    df = pd.read_csv(ruta, sep=";", encoding="latin1", dtype=str, usecols=usecols,
                     keep_default_na=False)
    df.columns = [("ANIO" if c.startswith("A") and c.endswith("O") and len(c) == 3 else c)
                  for c in df.columns]
    for c in LLAVES:
        if c in df.columns:
            df[c] = df[c].str.strip()
    return df


def _num(s: pd.Series) -> pd.Series:
    """Texto INEI -> float (coma decimal; blanco -> NaN)."""
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def construir_hogar_uid(df: pd.DataFrame, salt: str) -> pd.Series:
    """SHA-256 hex de 'conglome|vivienda|hogar|SALT' (CONTRATO)."""
    cad = df["conglome"] + "|" + df["vivienda"] + "|" + df["hogar"] + "|" + salt
    return cad.map(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())


def _agregados_personas(dir_ext: Path) -> pd.DataFrame:
    """Agregados a nivel hogar desde miembros (200), educacion (300) y empleo (500)."""
    def ruta(mod: str) -> Path:
        return dir_ext / f"1031-Modulo{mod}" / MODULOS_ENAHO_2025[mod][0]

    p = leer_csv_inei(ruta("02"), LLAVES + ["CODPERSO", "P203", "P204", "P207", "P208A"])
    p["edad"] = _num(p["P208A"])
    p["parent"] = _num(p["P203"])
    # Miembro del hogar: P204==1 y no trabajador del hogar/pensionista (P203 no en 8,9),
    # criterio de MIEPERHO en la nota de la Sumaria; se contrasta con MIEPERHO mas abajo.
    p = p[(p["P204"] == "1") & (~p["parent"].isin([8, 9]))].copy()
    g = p.groupby(LLAVES)
    out = pd.DataFrame({
        "n_miembros": g.size(),
        "n_menores5": g["edad"].apply(lambda s: int((s < 5).sum())),
        "n_menores18": g["edad"].apply(lambda s: int((s < 18).sum())),
        "n_adultos_mayores": g["edad"].apply(lambda s: int((s >= 65).sum())),
    })
    jefe = p[p["parent"] == 1].drop_duplicates(LLAVES).set_index(LLAVES)
    out["edad_jefe"] = jefe["edad"]
    out["sexo_jefe"] = jefe["P207"].map({"1": "H", "2": "M"})

    # Educacion: nivel del jefe (P301A 1..12 -> 0..11) y asistencia escolar 6-17 (P307).
    e = leer_csv_inei(ruta("03"), LLAVES + ["CODPERSO", "P301A", "P307"])
    e = e.merge(p[LLAVES + ["CODPERSO", "edad", "parent"]], on=LLAVES + ["CODPERSO"], how="inner")
    ej = e[e["parent"] == 1].drop_duplicates(LLAVES).set_index(LLAVES)
    nivel = _num(ej["P301A"])
    out["nivel_educ_jefe"] = (nivel - 1).where(nivel.between(1, 12))
    esc = e[(e["edad"] >= 6) & (e["edad"] <= 17) & e["P307"].isin(["1", "2"])]
    out["asistencia_escolar"] = esc.groupby(LLAVES)["P307"].apply(lambda s: float((s == "1").mean()))

    # Empleo: OCU500 1=ocupado, 2/3=desocupado abierto/oculto, 4=no PEA (verificado vs P501).
    t = leer_csv_inei(ruta("05"), LLAVES + ["CODPERSO", "P204", "OCU500"])
    t = t[t["P204"] == "1"]
    t = t.merge(p[LLAVES + ["CODPERSO"]], on=LLAVES + ["CODPERSO"], how="inner")
    gt = t.groupby(LLAVES)["OCU500"]
    out["n_ocupados"] = gt.apply(lambda s: int((s == "1").sum()))
    out["desempleo"] = gt.apply(lambda s: int(s.isin(["2", "3"]).any()))
    out["dependencia_economica"] = out["n_ocupados"] / out["n_miembros"]
    return out.reset_index()


def construir_dataset_enaho(dir_extraido: Path, salt: str | None = None,
                            umbrales: dict[str, float] | None = None) -> tuple[pd.DataFrame, dict]:
    """Arma el dataset analitico (una fila por hogar) y devuelve (df, informe_calidad).

    ``umbrales``: cortes inferiores del puntaje crudo 0-8 para fies_nivel [POR VALIDAR].
    ``informalidad`` queda en NaN: la base 500 publicada NO trae la variable derivada de
    informalidad (OCUPINF) y no se fabrica una regla propia.
    """
    umbrales = dict(umbrales or UMBRALES_FIES_DEFECTO)
    salt = salt or os.environ.get("FSAI_HOGAR_SALT") or SALT_DEFECTO_PUBLICO
    def ruta(mod: str) -> Path:
        return dir_extraido / f"1031-Modulo{mod}" / MODULOS_ENAHO_2025[mod][0]

    f = leer_csv_inei(ruta("2082"), LLAVES + ["UBIGEO", "DOMINIO", "ESTRATO", "MES", "FACTOR07"] + ITEMS_FIES)
    s = leer_csv_inei(ruta("34"), LLAVES + ["MIEPERHO", "GASHOG2D", "INGHOG2D", "POBREZA"] + COLS_GASTO_ALIM)
    v = leer_csv_inei(ruta("01"), LLAVES + ["P101", "P102", "P104", "P110", "P111A"])
    df = f.merge(s, on=LLAVES, how="left", validate="1:1").merge(v, on=LLAVES, how="left", validate="1:1")
    df = df.merge(_agregados_personas(dir_extraido), on=LLAVES, how="left", validate="1:1")

    o = pd.DataFrame({c.lower(): df[c] for c in LLAVES})
    o["hogar_uid"] = construir_hogar_uid(o, salt)
    o["factor_expansion"] = _num(df["FACTOR07"])
    o["ubigeo"] = df["UBIGEO"].str.strip().str.zfill(6)
    o["departamento"] = o["ubigeo"].str[:2].map(DEPARTAMENTOS)
    o["provincia"] = o["ubigeo"].str[:4]      # codigo (el catalogo de nombres no esta en las bases)
    o["distrito"] = o["ubigeo"]               # codigo
    o["area"] = np.where(_num(df["ESTRATO"]) <= 5, "U", "R")  # estratos 1-5 urbano, 6-8 rural
    o["region_natural"] = df["DOMINIO"].str.strip().map(REGION_POR_DOMINIO)
    o["anio"] = 2025
    o["mes"] = _num(df["MES"]).astype("Int64")
    for i, c in enumerate(ITEMS_FIES, start=1):
        o[f"fies_q{i}"] = df[c].str.strip().map({"1": 1.0, "2": 0.0})  # 3/4/blanco -> NaN
    q = o[[f"fies_q{i}" for i in range(1, 9)]]
    completo = q.notna().all(axis=1)
    o["fies_raw_score"] = q.sum(axis=1).where(completo)
    o["fies_rasch_logit"] = np.nan  # [POR VALIDAR] requiere parametros Rasch calibrados; no se fabrican
    o["fies_nivel"] = clasificar_fies(o["fies_raw_score"], umbrales)
    o["ia_binaria"] = (o["fies_nivel"] >= 2).astype("Int64").where(o["fies_nivel"].notna())

    nm = _num(df["MIEPERHO"])
    o["ingreso_pc"] = _num(df["INGHOG2D"]) / 12.0 / nm   # mensual per capita (Sumaria es anual)
    o["gasto_pc"] = _num(df["GASHOG2D"]) / 12.0 / nm
    alim = sum(_num(df[c]).fillna(0.0) for c in COLS_GASTO_ALIM)
    o["gasto_alim_pc"] = alim / 12.0 / nm
    o["share_alimentario"] = o["gasto_alim_pc"] / o["gasto_pc"].where(o["gasto_pc"] > 0)
    o["pobreza_monetaria"] = _num(df["POBREZA"]).map({1.0: 1, 2.0: 1, 3.0: 0}).astype("Int64")

    for c in ["n_ocupados", "desempleo", "dependencia_economica", "nivel_educ_jefe",
              "asistencia_escolar", "n_miembros", "n_menores5", "n_menores18", "edad_jefe",
              "sexo_jefe", "n_adultos_mayores"]:
        o[c] = df[c]
    o["informalidad"] = np.nan
    # Habitacional. agua_potable = red publica (P110 1,2); saneamiento = desague red publica,
    # letrina con tratamiento, pozo septico/biodigestor (P111A 1-4). Definiciones propias, declaradas.
    o["agua_potable"] = df["P110"].str.strip().map(lambda x: 1 if x in ("1", "2") else (0 if x else np.nan))
    o["saneamiento"] = df["P111A"].str.strip().map(lambda x: 1 if x in ("1", "2", "3", "4") else (0 if x else np.nan))
    hab = _num(df["P104"]).where(lambda x: (x > 0) & (x < 99))
    o["hacinamiento"] = o["n_miembros"] / hab
    o["tipo_vivienda"] = df["P101"].str.strip().replace("", np.nan)
    o["material_vivienda"] = df["P102"].str.strip().replace("", np.nan)  # paredes exteriores (P102)

    edad = o["edad_jefe"]
    o["grupo_edad_jefe"] = pd.cut(edad, [-np.inf, 30, 45, 60, np.inf], right=False,
                                  labels=["<30", "30-44", "45-59", "60+"]).astype(object)
    ne = o["nivel_educ_jefe"]
    o["grupo_educ_jefe"] = pd.cut(ne, [-np.inf, 2, 4, 6, np.inf], right=False,
                                  labels=["sin_primaria", "primaria", "secundaria", "superior"]).astype(object)
    # nivel 0-1 sin nivel/inicial; 2-3 primaria; 4-5 secundaria; >=6 superior (P301A-1)
    o["tiene_menor5"] = (o["n_menores5"] > 0).astype("Int64").where(o["n_menores5"].notna())

    orden = (["conglome", "vivienda", "hogar", "hogar_uid", "factor_expansion", "anio", "mes",
              "ubigeo", "departamento", "provincia", "distrito", "area", "region_natural"]
             + [f"fies_q{i}" for i in range(1, 9)]
             + ["fies_raw_score", "fies_rasch_logit", "fies_nivel", "ia_binaria",
                "ingreso_pc", "gasto_pc", "gasto_alim_pc", "share_alimentario", "pobreza_monetaria",
                "n_ocupados", "informalidad", "desempleo", "dependencia_economica",
                "nivel_educ_jefe", "asistencia_escolar", "n_miembros", "n_menores5", "n_menores18",
                "edad_jefe", "sexo_jefe", "n_adultos_mayores", "agua_potable", "saneamiento",
                "hacinamiento", "tipo_vivienda", "material_vivienda",
                "grupo_edad_jefe", "grupo_educ_jefe", "tiene_menor5"])
    o = o[orden].reset_index(drop=True)
    return o, informe_calidad(o, df, umbrales, salt != SALT_DEFECTO_PUBLICO)


def informe_calidad(o: pd.DataFrame, crudo: pd.DataFrame, umbrales: dict, salt_propia: bool) -> dict:
    """Metricas de calidad verificables (sin afirmar calibracion peruana)."""
    n = len(o)
    ok = o["fies_nivel"].notna()
    w = o.loc[ok, "factor_expansion"]
    dist = o.loc[ok, "fies_nivel"].value_counts().sort_index()
    raw = o.loc[ok, "fies_raw_score"].value_counts().sort_index()
    ia = o.loc[ok, "ia_binaria"].astype(float)
    nm = _num(crudo["MIEPERHO"])
    return {
        "n_filas": int(n),
        "n_hogares_unicos_llave": int(o[["conglome", "vivienda", "hogar"]].drop_duplicates().shape[0]),
        "n_hogares_uid_unicos": int(o["hogar_uid"].nunique()),
        "fies_items_en_blanco_hogares": int((crudo[ITEMS_FIES].apply(lambda c: c.str.strip() == "").all(axis=1)).sum()),
        "fies_con_ns_nr_en_algun_item": int((crudo[ITEMS_FIES].isin(["3", "4"]).any(axis=1)).sum()),
        "fies_nivel_valido": int(ok.sum()),
        "fies_nivel_conteo": {int(k): int(v) for k, v in dist.items()},
        "fies_raw_conteo": {int(k): int(v) for k, v in raw.items()},
        "prevalencia_hogares_ia_moderada_severa_ponderada_FACTOR07": float(np.average(ia, weights=w)),
        "prevalencia_hogares_ia_moderada_severa_sin_ponderar": float(ia.mean()),
        "n_miembros_igual_MIEPERHO": float((o["n_miembros"] == nm).mean()),
        "factor_expansion_min_max": [float(o["factor_expansion"].min()), float(o["factor_expansion"].max())],
        "umbrales_fies_raw_POR_VALIDAR": umbrales,
        "salt_hogar_uid_propia": bool(salt_propia),
        "nulos_por_columna": {c: int(v) for c, v in o.isna().sum().items() if v},
    }
