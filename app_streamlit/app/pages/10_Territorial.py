"""Análisis territorial: indicadores departamentales reales (ENDES, OPHI, ERA5) con el mapa incluido en la app, dispersión entre indicadores
(asociación, no causalidad), cobertura por indicador y, si hay dataset, prevalencia ponderada de inseguridad alimentaria por departamento, área y región.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import tema
from auth import barra_lateral, requerir_acceso
from comun import (
    INDICADORES_TERRITORIALES,
    cargar_geojson_departamental,
    cargar_indicadores_territoriales,
    cargar_json_derivado,
    normalizar_texto,
)
from fases_comun import C, _opciones_y_selector, interpretacion, selector_dataset
from tema import config_pagina, encabezado

from fsai import interpretacion as I  # noqa: E402

config_pagina("Territorial")
usuario, rol = requerir_acceso("territorial")
barra_lateral()
encabezado("Análisis territorial",
    "Indicadores por departamento (nutrición, pobreza, cobertura social, clima) y, con dataset, prevalencia ponderada de inseguridad alimentaria.",
    "Evaluación e interpretación",
    pasos=["Elija un indicador territorial para ver el mapa y el ranking.",
           "Cruce dos indicadores en la dispersión; lea la nota: es asociación entre 25 departamentos, no causalidad.",
           "Si hay dataset cargado, revise la prevalencia FIES por departamento, área o región natural con su intervalo de confianza."],
    produce="Mapa, ranking, dispersión, cobertura por indicador y (con dataset) prevalencia ponderada con IC.",
)

terr = cargar_indicadores_territoriales()
geo = cargar_geojson_departamental()
manifiesto = cargar_json_derivado("indicadores_territoriales_departamento.manifiesto.json")
cod_por_nombre = {normalizar_texto(f["properties"].get("NOMBDEP", "")): f["properties"].get("FIRST_IDDP") for f in geo["features"]} if geo else {}
N_COL = {"terr_desnutricion_cronica_pct": "n_ninos_talla", "terr_anemia_ninos_pct": "n_ninos_anemia", "terr_anemia_mujeres_pct": "n_mujeres_anemia",
         "terr_hogares_programa_social_pct": "n_hogares_ps", "terr_hogares_juntos_pct": "n_hogares_ps", "terr_hogares_pension65_pct": "n_hogares_ps",
         "terr_hogares_vaso_leche_pct": "n_hogares_ps"}

# ------------------------------------------------------------------ dataset (opcional): prevalencia FIES por departamento
hay_dataset = bool(_opciones_y_selector())
df = info = None
prev_dep = None
UMBRAL = 2
if hay_dataset:
    df, info = selector_dataset()
    if "departamento" in df.columns:
        _filas = []
        for g, s in df.groupby("departamento"):
            w = s[C.COL_PESO].to_numpy()
            _filas.append({"departamento": str(g), "prevalencia_ia_pct": 100 * float(np.average((s[C.COL_OBJETIVO].to_numpy() >= UMBRAL).astype(float), weights=w))})
        prev_dep = pd.DataFrame(_filas)
        prev_dep["cod_departamento"] = prev_dep["departamento"].map(lambda x: cod_por_nombre.get(normalizar_texto(x)))
        prev_dep = prev_dep.dropna(subset=["cod_departamento"])

# ------------------------------------------------------------------ 1) mapa y ranking de indicadores territoriales
if terr is None or geo is None:
    tema.vacio("No se encontraron los indicadores territoriales (datos_derivados) o el GeoJSON incluido en la app.", "Verifique datos_derivados y app/assets/geo",
               titulo="Indicadores territoriales no disponibles")
else:
    tema.seccion("Indicadores territoriales", "25 departamentos · fuentes públicas · el contorno grueso marca los cinco valores más altos.")
    ks = list(INDICADORES_TERRITORIALES)
    k = st.selectbox("Indicador", ks, format_func=lambda x: INDICADORES_TERRITORIALES[x][0] + (f" ({INDICADORES_TERRITORIALES[x][1]})" if INDICADORES_TERRITORIALES[x][1] else ""),
                     key="terr_ind")
    et, u, fam, dv, dec = INDICADORES_TERRITORIALES[k]
    meta = (manifiesto.get("indicadores") or {}).get(k, {})
    tema.tira_estadisticos([(f"{terr[k].min():.{dec}f}".replace(".", ",") + (f" {u}" if u else ""), "mínimo (departamento)"),
                            (f"{terr[k].median():.{dec}f}".replace(".", ",") + (f" {u}" if u else ""), "mediana de los 25"),
                            (f"{terr[k].max():.{dec}f}".replace(".", ",") + (f" {u}" if u else ""), "máximo (departamento)"),
                            (f"{meta.get('fuente', '—')} · {meta.get('anio', '—')}", "fuente y año")])
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        st.plotly_chart(tema.mapa_coropletico(terr, k, et + (f" ({u})" if u else ""), geo, divergente=dv, formato=f".{dec}f", sufijo=(f" {u}" if u else ""), alto=540), key="terr_mapa")
    with c2:
        st.plotly_chart(tema.barras_ranking(terr, k, et + (f" ({u})" if u else ""), formato=f".{dec}f", sufijo=(f" {u}" if u else ""), alto=540), key="terr_rank")
    st.markdown('<p class="fs-nota-mapa">Límites: GeoJSON incluido en la app (fuente juaneladio/peru-geojson), <b>licencia no verificada</b>; verifique antes de redistribuir. '
                "El ranking con cifras es el equivalente textual del mapa.</p>", unsafe_allow_html=True)

    # -------------------------------------------------------------- 2) dispersión entre indicadores
    tema.seccion("Dispersión entre indicadores", "Cada punto es un departamento. Útil para ver si dos fenómenos coinciden, no para explicar por qué.")
    opciones_y = ks + (["__prev"] if prev_dep is not None and len(prev_dep) else [])
    nombre = lambda x: ("Prevalencia IA moderada o severa (dataset activo, %)" if x == "__prev" else INDICADORES_TERRITORIALES[x][0])  # noqa: E731
    cx, cy = st.columns(2)
    x_k = cx.selectbox("Eje X", ks, index=ks.index("terr_mpi_incidencia_pct"), format_func=nombre, key="sc_x")
    y_k = cy.selectbox("Eje Y", opciones_y, index=opciones_y.index("__prev") if "__prev" in opciones_y else ks.index("terr_anemia_ninos_pct"), format_func=nombre, key="sc_y")
    d = terr.copy()
    if y_k == "__prev":
        d = d.merge(prev_dep[["cod_departamento", "prevalencia_ia_pct"]], on="cod_departamento", how="inner")
        y_col = "prevalencia_ia_pct"
    else:
        y_col = y_k
    d = d[[x_k, y_col, "departamento"]].dropna() if x_k != y_col else d
    if len(d) >= 5 and x_k != y_col:
        rho = float(d[x_k].corr(d[y_col], method="spearman"))
        coef = np.polyfit(d[x_k], d[y_col], 1)
        xs = np.linspace(d[x_k].min(), d[x_k].max(), 20)
        t = tema.TOKENS
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=np.polyval(coef, xs), mode="lines", line=dict(color=t["linea-fuerte"], width=2, dash="dash"), name="Ajuste lineal (orientativo)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=d[x_k], y=d[y_col], mode="markers+text", text=d["departamento"].str.title(), textposition="top center", textfont=dict(size=10, color=t["suave"]),
                                 marker=dict(size=11, color=t["acento"], line=dict(color=t["superficie"], width=1.5), symbol="circle"), name="Departamentos",
                                 hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>"))
        fig.update_layout(height=520, xaxis_title=nombre(x_k), yaxis_title=nombre(y_k), showlegend=False, margin=dict(l=60, r=24, t=16, b=56))
        st.plotly_chart(fig, key="terr_scatter")
        tema.tira_estadisticos([(f"{rho:+.2f}".replace(".", ","), "Spearman ρ"), (str(len(d)), "departamentos (n)"), ("25", "valores distintos por indicador")])
        tema.llamada("etica", "Asociación, no causalidad",
                     "Con 25 departamentos, una correlación alta puede deberse a un tercer factor o al azar; el ajuste lineal es solo orientativo y no hay intervalo de confianza. "
                     "ENDES 2024 y MPI 2023 no coinciden con el año de la ENAHO (2025).")
    else:
        tema.vacio("Elija dos indicadores distintos con al menos 5 departamentos en común.", titulo="Dispersión no disponible")

    # -------------------------------------------------------------- 3) cobertura por indicador
    tema.seccion("Cobertura y fuentes por indicador", "Cuántas fuentes, años y observaciones respaldan cada cifra.")
    filas = []
    for kk, (ee, uu, ff, _dd, dd) in INDICADORES_TERRITORIALES.items():
        m = (manifiesto.get("indicadores") or {}).get(kk, {})
        ncol = N_COL.get(kk)
        rango_n = f"{int(terr[ncol].min()):,}–{int(terr[ncol].max()):,}" if ncol and ncol in terr else "n/d"
        filas.append({"ind": ee, "fuente": m.get("fuente", "—"), "anio": str(m.get("anio", "—")), "n_fuentes": "1", "n_obs": rango_n,
                      "dist": str(int(terr[kk].nunique())), "rango": f"{terr[kk].min():.{dd}f}–{terr[kk].max():.{dd}f} {uu}".strip()})
    tema.tabla_html(filas, [("ind", "Indicador"), ("fuente", "Fuente"), ("anio", "Año"), ("n_fuentes", "N.º de fuentes"), ("n_obs", "Obs. por departamento"),
                            ("dist", "Valores distintos"), ("rango", "Rango")], "Un valor por departamento: 25 observaciones por indicador, un solo año de referencia cada uno.",
                    ("n_fuentes", "n_obs", "dist"))
    aviso = manifiesto.get("aviso_temporal")
    if aviso:
        tema.llamada("aviso", "Desfase temporal", str(aviso))

# ------------------------------------------------------------------ 4) prevalencia FIES del dataset
tema.seccion("Prevalencia de inseguridad alimentaria", "Estimación ponderada del dataset activo, con intervalo de confianza aproximado.")
if not hay_dataset or df is None:
    tema.vacio("La prevalencia FIES por territorio necesita hogares cargados (dataset en PostgreSQL o archivo tabular). Los indicadores de arriba no dependen de ello.",
               "Cargue ENAHO en «Carga de datos»", "pages/02_Importacion_ENAHO.py", titulo="Sin dataset de hogares")
    tema.pie_pagina()
    st.stop()

unidades = [c for c in ["departamento", "area", "region_natural"] if c in df.columns]
if not unidades:
    st.warning("El dataset no tiene variables territoriales (departamento, area, region_natural).")
    st.stop()
c1, c2 = st.columns(2)
unidad = c1.selectbox("Unidad territorial", unidades)
medida = c2.selectbox("Medida", ["IA moderada o severa (niveles 2-3)", "IA severa (nivel 3)"])
umbral_nivel = 2 if medida.startswith("IA moderada") else 3


def estimar(d: pd.DataFrame, por: str) -> pd.DataFrame:
    filas = []
    for g, s in d.groupby(por):
        w = s[C.COL_PESO].to_numpy()
        y = (s[C.COL_OBJETIVO].to_numpy() >= umbral_nivel).astype(float)
        p = float(np.average(y, weights=w))
        n_ef = float(w.sum() ** 2 / (w ** 2).sum())
        se = float(np.sqrt(max(p * (1 - p), 0) / n_ef))
        filas.append({por: g, "n_hogares": len(s), "n_efectivo": round(n_ef, 1), "prevalencia_ia": p,
                      "ic_inf": max(0.0, p - 1.96 * se), "ic_sup": min(1.0, p + 1.96 * se)})
    return pd.DataFrame(filas)


tabla = estimar(df, unidad)
st.dataframe(tabla.sort_values("prevalencia_ia", ascending=False).style.format({"prevalencia_ia": "{:.1%}", "ic_inf": "{:.1%}", "ic_sup": "{:.1%}"}), hide_index=True)
t = tabla.sort_values("prevalencia_ia")
fig = go.Figure(go.Bar(x=t["prevalencia_ia"], y=t[unidad].astype(str), orientation="h", marker_color=tema.TOKENS["acento"],
                       error_x={"type": "data", "symmetric": False, "array": t["ic_sup"] - t["prevalencia_ia"], "arrayminus": t["prevalencia_ia"] - t["ic_inf"]}))
fig.update_layout(xaxis_tickformat=".0%", xaxis_title=f"Prevalencia ponderada · {medida}", height=max(320, 22 * len(t) + 100), margin={"t": 20})
st.plotly_chart(fig, key="terr_barras")
interpretacion(I.i_territorial(tabla, unidad))

if unidad == "departamento" and geo is not None:
    tema.seccion("Mapa de prevalencia", "Cruce por nombre normalizado con el GeoJSON incluido (licencia no verificada).")
    tt = tabla.assign(cod_departamento=tabla["departamento"].map(lambda x: cod_por_nombre.get(normalizar_texto(x))), pct=100 * tabla["prevalencia_ia"])
    sin = tt.loc[tt["cod_departamento"].isna(), "departamento"].tolist()
    st.caption(f"{len(tt) - len(sin)} de {len(tt)} departamentos del dataset coinciden con el GeoJSON.")
    if sin:
        st.warning("Sin correspondencia en el GeoJSON (no se dibujan): " + ", ".join(map(str, sin)))
    st.plotly_chart(tema.mapa_coropletico(tt.dropna(subset=["cod_departamento"]), "pct", medida + " (%)", geo, sufijo=" %", alto=520), key="terr_mapa_prev")

if len(unidades) >= 2 and "area" in df.columns and unidad != "area":
    tema.seccion(f"{unidad} × área")
    cruce = df.groupby([unidad, "area"]).apply(lambda s: float(np.average((s[C.COL_OBJETIVO] >= umbral_nivel).astype(float), weights=s[C.COL_PESO])), include_groups=False).unstack("area")
    st.dataframe(cruce.style.format("{:.1%}"))
tema.pie_pagina()
