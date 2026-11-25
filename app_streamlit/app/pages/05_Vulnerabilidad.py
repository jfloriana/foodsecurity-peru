"""Vulnerabilidad: (1) perfil descriptivo de hogares vulnerables (menores de 5 años, adultos mayores, pobreza monetaria) y (2) indicadores derivados del hogar por nivel FIES y área.

El perfil es descriptivo (no es un índice ponderado ni validado) y no incluye desenlaces nutricionales directos. Los umbrales de las «banderas» de la segunda pestaña
son ajustables y están marcados [POR VALIDAR]: no se afirma una definición oficial. Todo se pondera con el factor de expansión.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from auth import barra_lateral, requerir_acceso
from comun import explicar_error
from tema import config_pagina, encabezado
from fases_comun import C, interpretacion, selector_dataset

from fsai import interpretacion as I  # noqa: E402
from fsai import perfil_vulnerabilidad as PV  # noqa: E402
from fsai import plots as P  # noqa: E402

config_pagina("Vulnerabilidad")
usuario, rol = requerir_acceso("vulnerabilidad")
barra_lateral()
encabezado("Vulnerabilidad de los hogares",
    "Perfil descriptivo de hogares vulnerables e indicadores de vulnerabilidad del hogar por nivel FIES.",
    "Datos",
    pasos=["En «Perfil descriptivo» revise la prevalencia de inseguridad alimentaria según menores de 5 años, adultos mayores y pobreza, y según el recuento de estas características.",
           "En «Indicadores por nivel FIES» ajuste los umbrales de las banderas (marcados [POR VALIDAR]) y compare niveles FIES y áreas.",
           "Descargue las tablas o genere el informe del perfil en PDF, Word y Excel."],
    produce="Prevalencias ponderadas con IC95 por subgrupo y por recuento, cruces con FIES e informe descargable.",
)
df, info = selector_dataset()
tab_perfil, tab_ind = st.tabs(["Perfil descriptivo de hogares vulnerables", "Indicadores por nivel FIES"])

MIMES = {"PDF": "application/pdf", "Word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
         "Excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}


@st.cache_data(show_spinner="Calculando el perfil...", max_entries=4, ttl=600)
def _perfil(clave: str, _df: pd.DataFrame) -> dict:
    """Perfil descriptivo (por dataset; ``clave`` identifica el dataset)."""
    return PV.perfil(_df)


def _excel_perfil(pf: dict) -> bytes:
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"aviso": [PV.ALCANCE, PV.METODO_IC]}).to_excel(xw, sheet_name="Alcance", index=False)
        pf["definiciones"].to_excel(xw, sheet_name="Definiciones", index=False)
        pf["subgrupos"].drop(columns=["clave"]).to_excel(xw, sheet_name="Subgrupos", index=False)
        pf["diferencias"].to_excel(xw, sheet_name="Diferencias", index=False)
        pf["recuento"].to_excel(xw, sheet_name="Recuento", index=False)
        pf["fies_recuento"].to_excel(xw, sheet_name="FIES_por_recuento", index=False)
        pf["fies_caracteristica"].to_excel(xw, sheet_name="FIES_por_caracteristica", index=False)
    return buf.getvalue()


with tab_perfil:
    st.info(PV.ALCANCE)
    if C.COL_OBJETIVO not in df.columns or C.COL_PESO not in df.columns:
        st.warning("El dataset activo no tiene nivel FIES o factor de expansión: el perfil no se puede calcular. Cargue y procese un dataset ENAHO con FIES en «Carga de datos».")
    else:
        pf = _perfil(f"{info.get('origen')}:{info.get('dataset_id', info.get('nombre'))}:{len(df)}", df)
        if pf["n_con_fies"] == 0:
            st.warning("El dataset activo no tiene hogares con nivel FIES calculado: ejecute «Procesamiento FIES» antes de ver el perfil.")
        else:
            tot = pf["total"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Hogares con FIES", f"{pf['n_con_fies']:,}")
            m2.metric("Prevalencia total (IA moderada o severa)", f"{100 * tot['prevalencia']:.1f} %".replace(".", ","),
                      help=f"IC95 {100 * tot['ic_inf']:.1f}-{100 * tot['ic_sup']:.1f} %. {PV.METODO_IC}")
            m3.metric("Hogares fuera del recuento", f"{pf['n_excluidos_recuento']:,}", help="Hogares con dato faltante en alguna de las tres características.")
            with st.expander("Definición de las características (variables del dataset)", expanded=False):
                st.dataframe(pf["definiciones"], hide_index=True)
                st.caption("Los elementos marcados [POR VALIDAR] son convenciones o significados de código que no se contrastaron con el protocolo ni con el diccionario oficial.")
            sub, dif, rec, fr = pf["subgrupos"], pf["diferencias"], pf["recuento"], pf["fies_recuento"]
            fmt_pct = {"% de hogares (ponderado)": "{:.1%}", "prevalencia": "{:.1%}", "IC95 inf": "{:.1%}", "IC95 sup": "{:.1%}", "n": "{:,}"}
            ren = {"pct_hogares": "% de hogares (ponderado)", "ic_inf": "IC95 inf", "ic_sup": "IC95 sup"}
            st.subheader("1. Prevalencia por subgrupo (con y sin la característica)")
            st.dataframe(sub.drop(columns=["clave"]).rename(columns=ren).style.format(fmt_pct), hide_index=True)
            st.plotly_chart(P.fig_perfil_subgrupos(sub), key="perfil_sub")
            st.dataframe(dif.style.format({"prevalencia con": "{:.1%}", "prevalencia sin": "{:.1%}", "diferencia (pp)": "{:+.1f}"}), hide_index=True)
            interpretacion(PV.i_subgrupos(sub, dif))
            st.subheader("2. Prevalencia según el recuento de características (0 a 3)")
            st.dataframe(rec.rename(columns=ren).style.format(fmt_pct), hide_index=True)
            st.plotly_chart(P.fig_perfil_recuento(rec, tot["prevalencia"]), key="perfil_rec")
            interpretacion(PV.i_recuento(rec))
            st.subheader("3. Cruce con el nivel FIES")
            st.dataframe(fr.style.format({**{c: "{:.1%}" for c in PV.NIVELES_FIES.values()}, "n": "{:,}"}), hide_index=True)
            st.plotly_chart(P.fig_perfil_fies(fr), key="perfil_fies")
            interpretacion(PV.i_fies(fr))
            with st.expander("Nivel FIES con y sin cada característica"):
                st.dataframe(pf["fies_caracteristica"].style.format({**{c: "{:.1%}" for c in PV.NIVELES_FIES.values()}, "n": "{:,}"}), hide_index=True)
            st.subheader("Descargas")
            d1, d2 = st.columns(2)
            d1.download_button("Descargar tablas del perfil (Excel)", data=_excel_perfil(pf), file_name="perfil_vulnerabilidad_tablas.xlsx",
                               mime=MIMES["Excel"], key="dl_perfil_xlsx", use_container_width=True)
            if info.get("origen") == "bd":
                if d2.button("Generar informe del perfil (PDF, Word y Excel con figuras)", key="gen_perfil", use_container_width=True):
                    from comun import get_engine
                    from fsai import reportes_fases as R
                    try:
                        with st.spinner("Generando figuras a 600 DPI y documentos..."):
                            salida = C.ARTIFACTS_DIR / "reportes" / f"perfil_vulnerabilidad_{time.strftime('%Y%m%d_%H%M%S')}"
                            st.session_state["perfil_salida"] = R.generar_reporte(get_engine(), int(info["dataset_id"]), df, None, "socio", [], ["perfil"], salida,
                                                                                  ["PDF", "Word", "Excel"], None, "Perfil descriptivo de hogares vulnerables")
                    except Exception as exc:  # noqa: BLE001
                        st.error(explicar_error(exc, "generar el informe del perfil"))
                salida_p = st.session_state.get("perfil_salida")
                if salida_p:
                    cols = st.columns(len(salida_p["archivos"]))
                    for col, (fmt, ruta) in zip(cols, salida_p["archivos"].items()):
                        col.download_button(f"Descargar {fmt}", data=ruta.read_bytes(), file_name=ruta.name, mime=MIMES[fmt], key=f"dl_perfil_{fmt}", use_container_width=True)
            else:
                d2.caption("El informe con figuras requiere un dataset registrado en PostgreSQL; con un archivo tabular use la descarga en Excel.")

with tab_ind:
    st.markdown("**Umbrales de las banderas de vulnerabilidad** [POR VALIDAR] (ajustables)")
    c1, c2, c3, c4 = st.columns(4)
    u_share = c1.slider("Carga alimentaria alta: share_alimentario >", 0.2, 0.9, 0.5, 0.05)
    u_hac = c2.slider("Hacinamiento: personas por habitación >", 1.0, 6.0, 3.0, 0.5)
    u_dep = c3.slider("Baja relación ocupados/miembros ≤", 0.0, 0.8, 0.25, 0.05)
    u_inf = c4.slider("Informalidad alta: proporción >", 0.2, 1.0, 0.8, 0.05)

    banderas: dict[str, pd.Series] = {}
    if "share_alimentario" in df:
        banderas[f"Carga alimentaria alta (share > {u_share})"] = df["share_alimentario"] > u_share
    if "hacinamiento" in df:
        banderas[f"Hacinamiento (> {u_hac} pers./hab.)"] = df["hacinamiento"] > u_hac
    if "dependencia_economica" in df:
        banderas[f"Baja relación ocupados/miembros (≤ {u_dep})"] = df["dependencia_economica"] <= u_dep
    if "informalidad" in df:
        banderas[f"Informalidad alta (> {u_inf})"] = df["informalidad"] > u_inf
    for col, etiqueta in (("agua_potable", "Sin agua potable"), ("saneamiento", "Sin saneamiento")):
        if col in df:
            banderas[etiqueta] = df[col] == 0
    for col, etiqueta in (("pobreza_monetaria", "Pobreza monetaria"), ("desempleo", "Algún desocupado")):
        if col in df:
            banderas[etiqueta] = df[col] == 1
    if "tiene_menor5" in df:
        banderas["Hogar con menor de 5 años"] = df["tiene_menor5"] == 1
    if "nivel_educ_jefe" in df:
        banderas["Jefe sin primaria completa (nivel ≤ 1)"] = df["nivel_educ_jefe"] <= 1
    if not banderas:
        st.warning("El dataset no contiene variables para derivar indicadores de vulnerabilidad.")
        st.stop()

    w = df[C.COL_PESO].to_numpy()
    niveles = sorted(df[C.COL_OBJETIVO].dropna().unique())

    def prevalencia_ponderada(mask: pd.Series, grupo: np.ndarray) -> float:
        return float(w[grupo & mask.to_numpy()].sum() / w[grupo].sum()) if w[grupo].sum() > 0 else float("nan")

    tabla = pd.DataFrame({n: {k: prevalencia_ponderada(b, (df[C.COL_OBJETIVO] == n).to_numpy()) for k, b in banderas.items()} for n in niveles})
    tabla["Total"] = {k: prevalencia_ponderada(b, np.ones(len(df), bool)) for k, b in banderas.items()}

    st.subheader("Prevalencia ponderada de cada bandera por nivel FIES")
    st.dataframe(tabla.rename(columns=lambda c: f"Nivel {c}" if c != "Total" else c).style.format("{:.1%}"))
    fig = go.Figure([go.Bar(name=f"Nivel {n}", x=list(tabla.index), y=tabla[n]) for n in niveles])
    fig.update_layout(barmode="group", yaxis_tickformat=".0%", yaxis_title="% de hogares (ponderado)", xaxis_tickangle=-30, height=460, margin={"t": 20})
    st.plotly_chart(fig, key="vuln_barras")
    interpretacion(I.i_vulnerabilidad(tabla.drop(columns="Total")))

    st.subheader("Indicadores continuos: media ponderada por nivel FIES")
    continuas = [c for c in ["share_alimentario", "hacinamiento", "dependencia_economica", "informalidad", "gasto_alim_pc", "ingreso_pc", "n_miembros"] if c in df]
    medias = pd.DataFrame({n: {c: float(np.average(df.loc[df[C.COL_OBJETIVO] == n, c].fillna(df[c].median()), weights=df.loc[df[C.COL_OBJETIVO] == n, C.COL_PESO]))
                               for c in continuas} for n in niveles})
    st.dataframe(medias.rename(columns=lambda c: f"Nivel {c}").round(3))
    interpretacion(I.i_vulnerabilidad(medias))

    if "area" in df:
        st.subheader("Banderas por área (urbano / rural)")
        por_area = pd.DataFrame({a: {k: prevalencia_ponderada(b, (df["area"] == a).to_numpy()) for k, b in banderas.items()} for a in sorted(df["area"].dropna().unique())})
        st.dataframe(por_area.style.format("{:.1%}"))
        if {"U", "R"} <= set(por_area.columns):
            dif_a = (por_area["R"] - por_area["U"]).sort_values(ascending=False)
            interpretacion({"que_muestra": f"La mayor brecha rural − urbana es «{dif_a.index[0]}» ({dif_a.iloc[0] * 100:+.1f} puntos porcentuales).",
                            "como_leerlo": "Una diferencia positiva indica que la condición es más frecuente en hogares rurales (prevalencia ponderada).",
                            "limitacion": "Asociación descriptiva sin intervalos de confianza; no es causal."})
