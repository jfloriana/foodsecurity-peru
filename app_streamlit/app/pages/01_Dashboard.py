"""Panorama de inseguridad alimentaria: mapa coroplético por defecto (GeoJSON incluido), prevalencia FIES (si hay dataset) e
indicadores territoriales derivados (ENDES, OPHI, ERA5), ranking y múltiplos pequeños.

El mapa y los indicadores territoriales NO requieren base de datos; la prevalencia FIES sí (necesita hogares con factor de expansión).
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from auth import barra_lateral, requerir_acceso
import tema
from fases_comun import dataset_id_activo
from tema import config_pagina, encabezado
from comun import (
    INDICADORES_TERRITORIALES,
    ErrorConsulta,
    cargar_geojson_departamental,
    cargar_indicadores_territoriales,
    consultar,
    fmt_pct,
    mostrar_error_bd,
    normalizar_texto,
)

config_pagina("Dashboard")
requerir_acceso("dashboard")
barra_lateral()

encabezado("Panorama de inseguridad alimentaria",
    "Dónde se concentra la vulnerabilidad: prevalencia FIES del dataset activo e indicadores territoriales de nutrición, pobreza, cobertura social y clima.",
    "Resultados y despliegue",
    pasos=["Elija el indicador del mapa: la prevalencia FIES (si hay dataset cargado) o un indicador territorial.",
           "Compare el mapa con el ranking y con los múltiplos pequeños de los demás indicadores.",
           "Si hay dataset, filtre por año y área en la barra lateral y revise la distribución de niveles FIES."],
    produce="Mapa coroplético, ranking por departamento y KPIs de prevalencia ponderada (sin intervalos: use Reportes para estimaciones con IC).",
)

ETIQUETAS_NIVEL = {0: "N0 · Seguridad alimentaria", 1: "N1 · Leve", 2: "N2 · Moderada", 3: "N3 · Severa"}
ETIQUETAS_AREA = {"U": "Urbano", "R": "Rural"}
N_MINIMO_HOGARES = 30

BASE_FROM = """
FROM fsai.hogares h
JOIN fsai.respuestas_fies r ON r.hogar_uid = h.hogar_uid
WHERE h.dataset_id = CAST(:dataset AS integer)
  AND (CAST(:anio AS integer) IS NULL OR h.anio = CAST(:anio AS integer))
  AND (CAST(:area AS text) IS NULL OR h.area = CAST(:area AS text))
"""

PREV_IA = (
    "CAST(SUM(h.factor_expansion * CAST(r.ia_binaria AS integer)) "
    "/ NULLIF(SUM(h.factor_expansion), 0) AS double precision)"
)

SQL_ANIOS = "SELECT DISTINCT anio FROM fsai.hogares WHERE anio IS NOT NULL AND dataset_id = :dataset ORDER BY anio DESC"

SQL_KPI = (
    "SELECT COUNT(*) AS n_hogares, "
    "CAST(SUM(h.factor_expansion) AS double precision) AS hogares_expandidos, "
    f"{PREV_IA} AS prev_ia, "
    "CAST(SUM(CASE WHEN r.fies_nivel = 3 THEN h.factor_expansion ELSE 0 END) "
    "/ NULLIF(SUM(h.factor_expansion), 0) AS double precision) AS prev_severa, "
    "CAST(SUM(h.factor_expansion * r.fies_raw_score) "
    "/ NULLIF(SUM(h.factor_expansion), 0) AS double precision) AS raw_medio "
    + BASE_FROM
)

SQL_DEPARTAMENTO = (
    "SELECT h.departamento, COUNT(*) AS n_hogares, "
    f"{PREV_IA} AS prevalencia "
    + BASE_FROM
    + " AND h.departamento IS NOT NULL GROUP BY h.departamento ORDER BY prevalencia DESC NULLS LAST"
)

SQL_AREA = (
    "SELECT h.area, COUNT(*) AS n_hogares, "
    f"{PREV_IA} AS prevalencia "
    + BASE_FROM
    + " GROUP BY h.area ORDER BY h.area"
)

SQL_NIVELES = (
    "SELECT r.fies_nivel AS nivel, COUNT(*) AS n_hogares, "
    "CAST(SUM(h.factor_expansion) / NULLIF(SUM(SUM(h.factor_expansion)) OVER (), 0) "
    "AS double precision) AS proporcion "
    + BASE_FROM
    + " GROUP BY r.fies_nivel ORDER BY r.fies_nivel"
)


def numero(valor: object) -> float:
    return float(valor) if valor is not None and not pd.isna(valor) else float("nan")


def coma(x: float, d: int = 1) -> str:
    return f"{x:,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")


# ------------------------------------------------------------------ 1) datos FIES (opcionales)
terr = cargar_indicadores_territoriales()
geo = cargar_geojson_departamental()
DATASET_ID = dataset_id_activo()
motivo_sin_fies = ""
kpi = df_dep = df_area = df_niv = None
if DATASET_ID is None:
    motivo_sin_fies = "El dataset activo no es un dataset de PostgreSQL (o no hay conexión): la prevalencia FIES necesita hogares cargados."
else:
    try:
        anios_df = consultar(SQL_ANIOS, {"dataset": DATASET_ID})
        if anios_df.empty:
            motivo_sin_fies = "El dataset activo no tiene hogares con año registrado."
    except ErrorConsulta:
        motivo_sin_fies = "No hay conexión con PostgreSQL: la prevalencia FIES necesita el dataset cargado."
    except Exception:  # noqa: BLE001
        motivo_sin_fies = "El esquema «fsai» no está disponible: la prevalencia FIES necesita el dataset cargado."
if not motivo_sin_fies:
    anios = [int(a) for a in anios_df["anio"]]
    with st.sidebar:
        st.header("Filtros")
        anio_sel = st.selectbox("Año", options=[None, *anios], format_func=lambda a: "Todos" if a is None else str(a))
        area_sel = st.selectbox("Área", options=[None, "U", "R"], format_func=lambda a: "Todas" if a is None else ETIQUETAS_AREA[a])
    params = {"dataset": DATASET_ID, "anio": anio_sel, "area": area_sel}
    try:
        kpi = consultar(SQL_KPI, params).iloc[0]
        df_dep = consultar(SQL_DEPARTAMENTO, params)
        df_area = consultar(SQL_AREA, params)
        df_niv = consultar(SQL_NIVELES, params)
        if int(kpi["n_hogares"]) == 0:
            motivo_sin_fies = "No hay hogares que cumplan los filtros seleccionados."
            kpi = None
    except ErrorConsulta as exc:
        mostrar_error_bd(exc)
        motivo_sin_fies = "No se pudo consultar la prevalencia FIES."

# ------------------------------------------------------------------ 2) KPIs
if kpi is not None:
    tema.kpi_tarjetas([
        {"etiqueta": "Hogares en la muestra", "valor": f"{int(kpi['n_hogares']):,}", "nota": f"≈ {numero(kpi['hogares_expandidos']):,.0f} hogares expandidos"},
        {"etiqueta": "IA moderada o severa", "valor": fmt_pct(numero(kpi["prev_ia"])), "nota": "Prevalencia ponderada (niveles FIES 2-3)"},
        {"etiqueta": "IA severa", "valor": fmt_pct(numero(kpi["prev_severa"])), "nota": "Prevalencia ponderada (nivel FIES 3)"},
        {"etiqueta": "Puntaje FIES crudo medio", "valor": f"{numero(kpi['raw_medio']):.2f}", "unidad": "/ 8", "nota": "Promedio ponderado del puntaje 0-8"},
    ])
    st.caption("Prevalencias ponderadas con factor_expansion, calculadas sobre **hogares** (no personas); no son directamente comparables con las cifras poblacionales "
               "del documento (30,5 % nacional; 35,4 % rural; 29,3 % urbano en 2025). Los intervalos de confianza se calculan en el módulo Reportes.")
else:
    tema.vacio(motivo_sin_fies + " Mientras tanto se muestran los indicadores territoriales reales (ENDES 2024, OPHI 2023, ERA5 2025).",
               "Cargue el dataset en «Carga de datos»", "pages/02_Importacion_ENAHO.py", titulo="Prevalencia FIES no disponible")

# ------------------------------------------------------------------ 3) mapa + ranking
tema.seccion("Mapa departamental", "Elija qué indicador pintar. El contorno grueso marca los cinco departamentos con el valor más alto.")
if terr is None or geo is None:
    tema.vacio("No se encontraron los indicadores territoriales (datos_derivados) o el GeoJSON departamental incluido en la app.",
               "Verifique datos_derivados y app/assets/geo", titulo="Mapa no disponible")
    tema.pie_pagina()
    st.stop()

cod_por_nombre = {normalizar_texto(f["properties"].get("NOMBDEP", "")): f["properties"].get("FIRST_IDDP") for f in geo["features"]}
familias: dict[str, list[str]] = {}
for k, (_et, _u, fam, _d, _n) in INDICADORES_TERRITORIALES.items():
    familias.setdefault(fam, []).append(k)
opciones_fam = (["Prevalencia FIES"] if kpi is not None else []) + list(familias)
fam = tema.pildoras("Familia de indicadores", opciones_fam, "fam")
if fam == "Prevalencia FIES":
    d = df_dep.assign(prevalencia_pct=100 * df_dep["prevalencia"].astype(float))
    d["cod_departamento"] = d["departamento"].map(lambda x: cod_por_nombre.get(normalizar_texto(x)))
    sin_cruce = d.loc[d["cod_departamento"].isna(), "departamento"].tolist()
    d = d.dropna(subset=["cod_departamento"])
    columna, titulo_ind, sufijo, fmt, diverg = "prevalencia_pct", "IA moderada o severa (%)", " %", ".1f", False
    fuente = "Prevalencia ponderada del dataset activo (PostgreSQL)"
    if sin_cruce:
        st.warning("Departamentos de la base sin correspondencia en el GeoJSON (no se dibujan): " + ", ".join(sorted(map(str, sin_cruce))))
else:
    ks = familias[fam]
    clave = ks[0] if len(ks) == 1 else tema.pildoras("Indicador", ks, f"ind_{fam}", formato=lambda k: INDICADORES_TERRITORIALES[k][0])
    et, u, _f, diverg, dec = INDICADORES_TERRITORIALES[clave]
    d = terr
    columna, titulo_ind, sufijo, fmt = clave, et + (f" ({u})" if u else ""), (f" {u}" if u else ""), f".{dec}f"

    fuente = et

col_m, col_r = st.columns([3, 2], gap="medium")
with col_m:
    st.plotly_chart(tema.mapa_coropletico(d, columna, titulo_ind, geo, divergente=diverg, formato=fmt, sufijo=sufijo, alto=560), key="dash_mapa")
with col_r:
    st.plotly_chart(tema.barras_ranking(d, columna, titulo_ind, formato=fmt, sufijo=sufijo, alto=560), key="dash_rank")
st.markdown(
    '<p class="fs-nota-mapa">El color codifica el valor y el contorno grueso, el quintil superior; el ranking de la derecha (con cifras) es el equivalente textual del mapa. '
    "Límites departamentales: GeoJSON incluido en la app, <b>licencia no verificada</b> (fuente: juaneladio/peru-geojson; verificar antes de redistribuir). "
    "Indicadores de ENDES 2024 y OPHI/MPI 2023 (no coinciden con el año de la ENAHO 2025); la anomalía de lluvia es 2025 (ERA5).</p>", unsafe_allow_html=True)
if fam != "Prevalencia FIES":
    tema.llamada("etica", "Asociación, no causalidad", "Un indicador territorial describe al departamento, no a sus hogares: no justifica por sí solo ninguna decisión sobre un hogar.")
if fam == "Prevalencia FIES":
    pocos = df_dep.loc[df_dep["n_hogares"] < N_MINIMO_HOGARES, "departamento"].tolist()
    if pocos:
        st.warning(f"Departamentos con menos de {N_MINIMO_HOGARES} hogares muestrales (estimaciones inestables): {', '.join(pocos)}.")

# ------------------------------------------------------------------ 4) múltiplos pequeños
tema.seccion("Múltiplos pequeños", "Los mismos 25 departamentos, un mapa por indicador, para ver coincidencias a simple vista.")
chicos = ["terr_anemia_ninos_pct", "terr_desnutricion_cronica_pct", "terr_mpi_incidencia_pct", "terr_hogares_programa_social_pct", "terr_anemia_mujeres_pct", "terr_precip_anom_pct"]
filas_mm = [st.columns(3, gap="small"), st.columns(3, gap="small")]
for i, k in enumerate(chicos):
    et, u, _f, dv, dec = INDICADORES_TERRITORIALES[k]
    with filas_mm[i // 3][i % 3]:
        st.markdown(f'<p class="fs-nota-mapa" style="margin:0"><b>{tema._e(et)}</b></p>', unsafe_allow_html=True)
        st.plotly_chart(tema.mapa_coropletico(terr, k, et, geo, divergente=dv, formato=f".{dec}f", sufijo=(f" {u}" if u else ""), alto=250, barra=False, marcar_top=3),
                        key=f"mm_{k}")

# ------------------------------------------------------------------ 5) distribución FIES y área (si hay datos)
if kpi is not None:
    tema.seccion("Niveles FIES y brecha urbano-rural", "Hogares ponderados por nivel de inseguridad y prevalencia por área.")
    c_n, c_a = st.columns([3, 2], gap="medium")
    with c_n:
        if df_niv is not None and not df_niv.empty:
            df_n = df_niv.assign(nombre=df_niv["nivel"].map(ETIQUETAS_NIVEL), pct=100 * df_niv["proporcion"].astype(float))
            fig_n = go.Figure(go.Bar(
                x=df_n["pct"], y=df_n["nombre"], orientation="h", marker_color=[tema.color_nivel(int(n)) for n in df_n["nivel"]],
                text=[f"{v:.1f} %" for v in df_n["pct"]], textposition="outside", cliponaxis=False,
                customdata=df_n["n_hogares"], hovertemplate="<b>%{y}</b><br>%{x:.1f} % de hogares ponderados<br>%{customdata} hogares muestrales<extra></extra>"))
            fig_n.update_layout(height=300, margin=dict(l=0, r=40, t=8, b=36), xaxis_title="Hogares ponderados (%)", yaxis=dict(autorange="reversed", showgrid=False, showline=False, ticks=""))
            fig_n.update_xaxes(range=[0, max(60, float(df_n["pct"].max()) + 12)])
            st.plotly_chart(fig_n, key="dash_niv")
            tema.leyenda_fies()
        else:
            tema.vacio("Sin datos de niveles FIES para los filtros elegidos.", titulo="Sin niveles")
    with c_a:
        if df_area is not None and not df_area.empty:
            df_a = df_area.assign(nombre=df_area["area"].map(ETIQUETAS_AREA), pct=100 * df_area["prevalencia"].astype(float))
            fig_a = go.Figure(go.Bar(x=df_a["nombre"], y=df_a["pct"], marker_color=tema.TOKENS["acento"], text=[f"{v:.1f} %" for v in df_a["pct"]],
                                     textposition="outside", cliponaxis=False, customdata=df_a["n_hogares"],
                                     hovertemplate="<b>%{x}</b><br>%{y:.1f} %<br>%{customdata} hogares muestrales<extra></extra>"))
            fig_a.update_layout(height=300, margin=dict(l=0, r=8, t=8, b=36), yaxis_title="IA moderada o severa (%)")
            fig_a.update_yaxes(range=[0, 100])
            st.plotly_chart(fig_a, key="dash_area")
    st.caption("Los umbrales de clasificación de fies_nivel están marcados como [POR VALIDAR] en el protocolo; no se afirma calibración peruana de la escala.")
tema.pie_pagina()
