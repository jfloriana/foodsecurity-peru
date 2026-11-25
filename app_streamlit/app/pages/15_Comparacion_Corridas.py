"""Comparación de corridas: métricas agregadas (F1-macro ponderado, AUC, Brier) de los experimentos, con avisos de comparabilidad.

Fuente: ``fsai.experimentos`` (PostgreSQL) si hay conexión y filas; si no, los JSON agregados de ``resultados_agregados/``
(carpeta resuelta por ``FSAI_RESULTADOS_AGREGADOS`` o relativa al repositorio). Solo lectura. La lógica vive en ``fsai.comparacion_corridas``.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import modulos
from auth import barra_lateral, requerir_acceso
from comun import consultar
import tema
from tema import config_pagina, encabezado

modulos.registrar_accesos()
config_pagina("Comparación de corridas")
usuario, rol = requerir_acceso("comparacion_corridas")
barra_lateral()

from fsai import comparacion_corridas as CC  # noqa: E402

encabezado(
    "Comparación de corridas",
    "Compare experimentos entre corridas, escenarios y modelos, con intervalos de confianza y avisos cuando la comparación no es válida.",
    "Evaluación e interpretación",
    pasos=["Elija las corridas, escenarios y modelos que quiere comparar.",
           "Lea los avisos de comparabilidad antes de interpretar las diferencias.",
           "Revise la tabla y el gráfico de barras agrupadas (las barras de error son el IC de 95 %, cuando existe)."],
    produce="Tabla y gráfico comparativos de F1-macro ponderado, AUC y Brier; no ejecuta entrenamientos.",
)


@st.cache_data(ttl=300, show_spinner=False)
def _cargar() -> tuple[pd.DataFrame, str, list[str]]:
    tabla, origen = CC.cargar_corridas(lambda sql: consultar(sql, cache=False))
    return tabla, origen, list(tabla.attrs.get("problemas", []))


tabla, origen, problemas = _cargar()
if tabla.empty:
    tema.vacio("No hay corridas para comparar: no se encontraron experimentos en PostgreSQL ni archivos resultado_*.json en «resultados_agregados».",
               "entrene una corrida en «Entrenamiento» o defina FSAI_RESULTADOS_AGREGADOS con la carpeta de resultados", "pages/06_Entrenamiento.py", titulo="No hay corridas para comparar")
    for p in problemas:
        st.caption(p)
    st.stop()

st.caption("Origen de los datos: " + ("PostgreSQL (fsai.experimentos)" if origen == "bd" else "archivos JSON agregados (resultados_agregados)") + f" · {len(tabla)} filas.")
for p in problemas:
    st.warning(f"Archivo omitido: {p}")

c1, c2, c3 = st.columns(3)
corridas = c1.multiselect("Corridas", sorted(tabla["corrida"].dropna().unique()), default=sorted(tabla["corrida"].dropna().unique()))
escenarios = c2.multiselect("Escenarios", sorted(tabla["escenario"].dropna().unique()), default=sorted(tabla["escenario"].dropna().unique()),
                            help="«fies» e «hibrido» reconstruyen el objetivo (circulares); «socio» es la predicción real.")
modelos = c3.multiselect("Modelos", sorted(tabla["modelo"].dropna().unique()), default=sorted(tabla["modelo"].dropna().unique()))
sel = CC.filtrar(tabla, corridas, escenarios, modelos)
if sel.empty:
    st.info("La selección no contiene filas. Amplíe los filtros.")
    st.stop()

tema.tira_estadisticos([(str(len(sel)), "filas seleccionadas"), (str(sel["corrida"].nunique()), "corridas"), (str(sel["escenario"].nunique()), "escenarios"), (str(sel["modelo"].nunique()), "modelos")])
for aviso in CC.advertencias_comparabilidad(sel):
    st.warning(aviso, icon=":material/warning:")

tema.seccion("Tabla comparativa", "F1-macro ponderado, AUC y Brier por modelo, escenario y corrida.")
vista = sel.rename(columns={"corrida": "Corrida", "esquema": "Esquema", "escenario": "Escenario", "modelo": "Modelo", "n_trials": "n_trials",
                            "f1_macro": "F1-macro pond.", "auc": "AUC (OVR)", "brier": "Brier"})
st.dataframe(vista[["Corrida", "Esquema", "Escenario", "Modelo", "F1-macro pond.", "AUC (OVR)", "Brier", "n_trials"]],
             hide_index=True, use_container_width=True,
             column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ("F1-macro pond.", "AUC (OVR)", "Brier")})
st.download_button("Descargar tabla (CSV)", sel.to_csv(index=False).encode("utf-8-sig"), file_name="comparacion_corridas.csv", mime="text/csv")

tema.seccion("Gráfico", "Barras agrupadas con IC de 95 % entre pliegues externos.")
etiquetas = {"f1_macro": "F1-macro ponderado", "auc": "AUC (uno contra el resto)", "brier": "Brier (menor es mejor)"}
metrica = st.radio("Métrica", list(etiquetas), format_func=etiquetas.get, horizontal=True)
fig = go.Figure()
sel = sel.assign(serie=sel["corrida"].astype(str) + " · " + sel["escenario"].astype(str))
for serie, grupo in sel.groupby("serie", sort=True):
    inf, sup = grupo[f"{metrica}_ic_inf"], grupo[f"{metrica}_ic_sup"]
    hay_ic = inf.notna().all() and sup.notna().all()
    fig.add_bar(x=grupo["modelo"], y=grupo[metrica], name=serie,
                error_y=dict(type="data", symmetric=False, array=(sup - grupo[metrica]).clip(lower=0), arrayminus=(grupo[metrica] - inf).clip(lower=0)) if hay_ic else None)
fig.update_layout(barmode="group", xaxis_title="Modelo", yaxis_title=etiquetas[metrica], legend_title_text="Corrida · escenario")
st.plotly_chart(fig, use_container_width=True)
st.caption("Las barras de error muestran el intervalo de confianza de 95 % entre pliegues externos; si una serie no lo informa, se dibuja sin barras. "
           "Una diferencia entre barras no es significativa solo por ser visible: use «Comparación estadística» para contrastarla.")
