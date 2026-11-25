"""Procesamiento de FIES: puntaje crudo, medida Rasch y niveles de inseguridad alimentaria (usa fsai.fies) + validación de calidad.

Los cortes sobre el puntaje crudo son parametrizables y están marcados [POR VALIDAR]; los parámetros Rasch para el Perú no se inventan:
sin dificultades de ítem inyectadas, la medida Rasch queda vacía. Esta página NO modifica la base de datos (solo recalcula en pantalla).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from auth import barra_lateral, requerir_acceso
from tema import config_pagina, encabezado
from fases_comun import C, audit, interpretacion, selector_dataset

from fsai import etiquetas as E  # noqa: E402
from fsai import interpretacion as I  # noqa: E402
from fsai.data_loader import validar_calidad  # noqa: E402
from fsai.fies import ITEMS, UMBRALES_RAW, calcular_raw_score, calcular_rasch_logit, clasificar_fies  # noqa: E402

config_pagina("Procesamiento FIES")
usuario, rol = requerir_acceso("procesamiento_fies")
barra_lateral()
encabezado("Procesamiento FIES",
    "Puntaje crudo, escala Rasch, niveles de inseguridad y control de calidad de la escala FIES.",
    "Datos",
    pasos=["Revise el puntaje crudo (0-8) y su distribución.",
           "Ajuste, si lo necesita, los cortes que definen los niveles 0-3 (marcados [POR VALIDAR]).",
           "Compare con la medida Rasch y revise los indicadores de calidad de la escala."],
    produce="Distribución de niveles FIES, concordancia con la escala Rasch y diagnóstico de calidad (no modifica la base de datos).",
)
df, info = selector_dataset()

faltan = [q for q in ITEMS if q not in df.columns]
if faltan:
    st.error(f"El dataset no tiene los ítems FIES {faltan}.")
    st.stop()

t_raw, t_rasch, t_calidad = st.tabs(["1 · Puntaje crudo y niveles", "2 · Medida Rasch", "3 · Validación de calidad"])

with t_raw:
    st.markdown("**Cortes inferiores sobre el puntaje crudo (0-8)** [POR VALIDAR]")
    c1, c2, c3 = st.columns(3)
    leve = c1.number_input("leve ≥", 1, 7, int(UMBRALES_RAW["leve"]))
    moderada = c2.number_input("moderada ≥", 2, 8, int(UMBRALES_RAW["moderada"]))
    severa = c3.number_input("severa ≥", 3, 8, int(UMBRALES_RAW["severa"]))
    umbrales = {"leve": leve, "moderada": moderada, "severa": severa}
    if not leve < moderada < severa:
        st.error("Los cortes deben ser estrictamente crecientes (leve < moderada < severa).")
        st.stop()
    raw = calcular_raw_score(df)
    nivel = clasificar_fies(raw, umbrales)
    ok = raw.notna()
    concordancia = float((nivel[ok].astype(int) == df.loc[ok, C.COL_OBJETIVO].astype(int)).mean()) if C.COL_OBJETIVO in df.columns else None
    dist = raw.value_counts().sort_index()
    f = go.Figure(go.Bar(x=dist.index.astype(int), y=dist.values / dist.sum(), marker_color="#1b6ca8"))
    for k, v in umbrales.items():
        f.add_vline(x=float(v) - 0.5, line_dash="dash", line_color="#c0392b", annotation_text=k)
    f.update_layout(xaxis_title="Puntaje crudo FIES (respuestas afirmativas)", yaxis_title="Proporción de hogares", yaxis_tickformat=".0%", margin={"t": 20})
    st.plotly_chart(f, key="fies_raw")
    interpretacion(I.i_fies(raw.dropna(), concordancia, umbrales, int((~ok).sum())))
    w = df[C.COL_PESO]
    tabla = pd.DataFrame({"nivel": [0, 1, 2, 3], "hogares (recalculado)": [int((nivel == k).sum()) for k in range(4)],
                          "% sin ponderar": [float((nivel == k).mean()) * 100 for k in range(4)],
                          "% ponderado": [float(w[nivel == k].sum() / w[ok].sum()) * 100 for k in range(4)]})
    st.dataframe(tabla.round(2), hide_index=True)
    interpretacion(I.i_objetivo(df.assign(fies_nivel=nivel.fillna(0).astype(int))), "Distribución de niveles con los cortes elegidos")
    st.caption("Esta vista recalcula en pantalla; el nivel que usan Entrenamiento y Evaluación es el almacenado/derivado con los cortes por defecto del contrato (leve 1, moderada 4, severa 7).")
    audit("procesamiento_fies_consulta", "dataset", info["nombre"], {"umbrales": {k: int(v) for k, v in umbrales.items()}})

with t_rasch:
    st.markdown("**Dificultades de ítem (parámetros Rasch)** — no se proveen valores por defecto: deben venir de una calibración validada [POR VALIDAR].")
    texto = st.text_area("JSON {\"fies_q1\": b1, ..., \"fies_q8\": b8}", value="", height=120, key="rasch_json")
    if texto.strip():
        try:
            params = json.loads(texto)
            logit = calcular_rasch_logit(df, params)
            st.metric("Hogares con medida Rasch", f"{int(logit.notna().sum()):,}")
            st.plotly_chart(go.Figure(go.Histogram(x=logit.dropna(), nbinsx=30, marker_color="#1b6ca8")).update_layout(xaxis_title="Severidad (logit)", margin={"t": 20}), key="rasch_hist")
            interpretacion({"que_muestra": f"La medida Rasch toma {logit.nunique()} valores distintos (uno por puntaje crudo), de {logit.min():.2f} a {logit.max():.2f}.",
                            "como_leerlo": "Valores mayores indican mayor severidad de inseguridad alimentaria; puntajes extremos (0 y 8) usan la corrección 0,5 / 7,5.",
                            "limitacion": "Las dificultades ingresadas no se validan aquí; sin calibración peruana la medida no es comparable con otras fuentes."})
        except (ValueError, KeyError) as exc:
            st.error(f"Parámetros inválidos: {exc}")
    else:
        st.info("Sin parámetros Rasch la medida queda vacía (NaN); los niveles se calculan con el puntaje crudo.")

with t_calidad:
    informe = validar_calidad(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Hogares", f"{informe['n_filas']:,}")
    c2.metric("Errores bloqueantes", len(informe["errores"]))
    c3.metric("Advertencias", len(informe["advertencias"]))
    (st.success if informe["ok"] else st.error)("Calidad: " + ("sin errores bloqueantes." if informe["ok"] else "; ".join(informe["errores"])))
    for a in informe["advertencias"]:
        st.warning(a)
    nulos = pd.DataFrame({"variable": list(informe["porcentaje_nulos"]), "% nulos": [v * 100 for v in informe["porcentaje_nulos"].values()]})
    if len(nulos):
        st.dataframe(E.presentar_tabla(nulos.sort_values("% nulos", ascending=False).round(2)), hide_index=True)
    interpretacion(I.i_nulos(df))
    if "n_efectivo_kish" in informe:
        st.caption(f"Suma del factor de expansión: {informe['suma_factor_expansion']:,.0f} · n efectivo de Kish: {informe['n_efectivo_kish']:,.0f}")
