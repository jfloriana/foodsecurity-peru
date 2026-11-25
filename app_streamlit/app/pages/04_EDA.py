"""Análisis exploratorio (EDA): visualización del dataset (tabla paginada, filtros, resumen) y estadística descriptiva con tablas y gráficos interpretativos."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from auth import ROLES_ESCRITURA, barra_lateral, requerir_acceso
from tema import config_pagina, encabezado
from fases_comun import C, D, audit, interpretacion, selector_dataset

from fsai import etiquetas as E  # noqa: E402
from fsai import interpretacion as I  # noqa: E402
from fsai import plots as P  # noqa: E402
from fsai import splits as S  # noqa: E402
config_pagina("EDA")
usuario, rol = requerir_acceso("eda")
barra_lateral()
encabezado("Exploración de datos",
    "Distribuciones, asociaciones y calidad de los datos antes del modelado.",
    "Datos",
    pasos=["En «Visualización» filtre y revise la tabla de hogares.",
           "En «Estadística descriptiva» elija una variable para ver su distribución y su relación con el nivel FIES.",
           "Revise los faltantes y las asociaciones antes de entrenar."],
    produce="Tablas y gráficos descriptivos con una lectura interpretada de cada uno.",
)
df, info = selector_dataset()

t_vis, t_desc = st.tabs(["1 · Visualización del dataset", "2 · Estadística descriptiva"])

# ------------------------------------------------------------------ 3 · Visualización
with t_vis:
    f1, f2, f3, f4 = st.columns(4)
    sel_area = f1.multiselect("Área", sorted(df["area"].dropna().unique()), default=[]) if "area" in df else []
    sel_reg = f2.multiselect("Región natural", sorted(df["region_natural"].dropna().unique()), default=[]) if "region_natural" in df else []
    sel_niv = f3.multiselect("Nivel FIES", sorted(df["fies_nivel"].unique()), default=[])
    sel_dep = f4.multiselect("Departamento", sorted(df["departamento"].dropna().unique()), default=[]) if "departamento" in df else []
    cols_vis = st.multiselect("Columnas a mostrar", list(df.columns), format_func=E.etq, help="Se muestra el nombre legible; el código original de cada columna está en el anexo «Diccionario de variables» del informe.", default=[c for c in ["hogar_uid", "departamento", "area", "fies_nivel", "ingreso_pc", "n_miembros", "edad_jefe"] if c in df.columns])
    m = pd.Series(True, index=df.index)
    for col, sel in (("area", sel_area), ("region_natural", sel_reg), ("fies_nivel", sel_niv), ("departamento", sel_dep)):
        if sel:
            m &= df[col].isin(sel)
    vista = df.loc[m, cols_vis] if cols_vis else df.loc[m]
    p1, p2 = st.columns(2)
    tam = p1.selectbox("Filas por página", [10, 25, 50, 100], index=1)
    n_pag = max(1, int(np.ceil(len(vista) / tam)))
    pag = p2.number_input(f"Página (1-{n_pag})", min_value=1, max_value=n_pag, value=1, step=1)
    st.caption(f"{len(vista):,} de {len(df):,} hogares cumplen los filtros.")
    st.dataframe(E.renombrar_columnas(vista.iloc[(pag - 1) * tam: pag * tam]), hide_index=True)
    st.markdown("**Resumen estadístico de las columnas numéricas mostradas**")
    num_v = vista.select_dtypes("number")
    if not num_v.empty:
        st.dataframe(num_v.describe().T.rename(index=E.etq).round(3))
    interpretacion({"que_muestra": f"La vista contiene {len(vista):,} hogares ({len(vista) / max(len(df), 1):.1%} del dataset) y {vista.shape[1]} columnas.",
                    "como_leerlo": "Use los filtros para inspeccionar subgrupos; el resumen estadístico describe solo las filas filtradas.",
                    "limitacion": "Una vista filtrada no es una muestra representativa: no extrapole sus cifras a la población."} if len(vista) else I.vacio("Los filtros no dejan ningún hogar."))

# ------------------------------------------------------------------ 5 · Descriptiva
with t_desc:
    st.markdown("**Variable objetivo**")
    st.plotly_chart(P.fig_objetivo(df))
    interpretacion(I.i_objetivo(df))
    st.markdown("**Valores faltantes por variable**")
    nulos = (df.isna().mean() * 100).round(2).rename("% nulos").reset_index().rename(columns={"index": "variable"}).sort_values("% nulos", ascending=False)
    st.dataframe(E.presentar_tabla(nulos.head(15)), hide_index=True)
    interpretacion(I.i_nulos(df))
    excluidas = set(C.COLUMNAS_ID_TERRITORIALES) | {"hogar_uid", "factor_expansion"}
    numericas = [c for c in df.select_dtypes("number").columns if c not in excluidas]
    categoricas = [c for c in df.columns if c not in numericas and c not in excluidas and df[c].nunique() <= 60]
    st.markdown("**Variable numérica**")
    vn = st.selectbox("Variable numérica", numericas, index=numericas.index("ingreso_pc") if "ingreso_pc" in numericas else 0, format_func=lambda c: E.etq(c, True), help=E.ayuda(numericas[0]) if numericas else None)
    color = st.selectbox("Colorear por", ["(ninguno)", "fies_nivel", "area"], index=0, format_func=lambda c: c if c.startswith("(") else E.etq(c))
    df_g = df.assign(fies_nivel=df["fies_nivel"].astype(str)) if color == "fies_nivel" else df
    st.plotly_chart(P.fig_hist(df_g, vn, None if color == "(ninguno)" else color))
    interpretacion(I.i_numerica(df, vn))
    interpretacion(I.i_bivariada(df, vn), "Relación con el nivel FIES (asociación, no causalidad)")
    if categoricas:
        st.markdown("**Variable categórica**")
        vc = st.selectbox("Variable categórica", categoricas, format_func=E.etq)
        st.plotly_chart(P.fig_barras_cat(df, vc))
        interpretacion(I.i_categorica(df, vc))
    st.markdown("**Correlaciones (Spearman)**")
    sel_corr = st.multiselect("Variables", numericas, default=numericas[:12], format_func=E.etq)
    if len(sel_corr) >= 2:
        corr = df[sel_corr].corr(method="spearman")
        st.plotly_chart(P.fig_corr(corr))
        interpretacion(I.i_correlacion(corr))
