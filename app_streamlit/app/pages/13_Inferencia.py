"""Inferencia: predicción individual de un hogar y por lote, con incertidumbre del ensamble y explicación local.

Método (declarado): ensamble = 5 modelos de los pliegues externos de la CV anidada + modelo final; la probabilidad es el promedio y el
intervalo un bootstrap percentil sobre los miembros (mide variabilidad del modelo, no un IC frecuentista). Umbral operativo: Validación.
Análisis locales (fsai.analisis_local): LIME frente a SHAP, sensibilidad (curvas ICE y tornado), contrafactual con variables accionables,
comparación con los demás modelos de la corrida y advertencia de extrapolación. Todo describe al MODELO (asociación predictiva): no es causal.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

import tema
from auth import barra_lateral, requerir_acceso
from tema import config_pagina, encabezado
from comun import explicar_error
from fases_comun import AVISO_CIRCULARIDAD, C, audit, interpretacion, selector_dataset

from fsai import catalogo_campos as K  # noqa: E402
from fsai import etiquetas as E  # noqa: E402
from fsai import evaluacion as EV  # noqa: E402
from fsai import inference as INF  # noqa: E402
from fsai import plots as P  # noqa: E402
from fsai import analisis_local as AL  # noqa: E402
from fsai import reporte_inferencia as RI  # noqa: E402
from fsai import perfil_vulnerabilidad as PV  # noqa: E402
from fsai.splits import cargar_particion  # noqa: E402

config_pagina("Inferencia")
usuario, rol = requerir_acceso("inferencia")
barra_lateral()
encabezado("Inferencia",
    "Predicción de inseguridad alimentaria para un hogar o un lote de hogares, con intervalo y explicación local.",
    "Resultados y despliegue",
    pasos=["Elija la corrida y el escenario del modelo final.",
           "Ingrese las variables del hogar en el formulario o suba un CSV por lote.",
           "Revise las probabilidades por nivel FIES con su intervalo y la explicación local."],
    produce="Probabilidades por nivel FIES (IC del ensamble), percentil de riesgo y contribuciones locales.",
)
df, info = selector_dataset()

corridas = EV.listar_corridas()
if not corridas:
    tema.vacio("No hay corridas de entrenamiento completadas.", "Entrene un modelo en «Entrenamiento»", "pages/06_Entrenamiento.py")
    st.stop()
c1, c2 = st.columns(2)
nombre = c1.selectbox("Corrida", corridas)
dir_run = EV.dir_corrida(nombre)
resultados = EV.resultados_corrida(nombre)
ident = c2.selectbox("Modelo / escenario", list(resultados), format_func=lambda k: f"{k.split('_')[1]} · {k.split('_')[0]}")
res = resultados[ident]
if res["escenario"] != "socio":
    st.warning(AVISO_CIRCULARIDAD + " Al usar los ítems FIES como entrada, la predicción reconstruye el nivel FIES, no lo predice.")


@st.cache_resource(show_spinner="Cargando el ensamble de modelos...")
def _ensamble(ruta: str, ident_: str):
    return INF.cargar_ensamble(__import__("pathlib").Path(ruta), ident_)


ens = _ensamble(str(dir_run), ident)
umbral = ens["umbral"]
try:
    part, _ = cargar_particion(df, dir_run / "particion.json")
except (ValueError, OSError) as exc:
    st.error(f"El dataset activo no coincide con la corrida ({exc}).")
    st.stop()
dev = df.iloc[part["desarrollo"]].reset_index(drop=True)
val = df.iloc[part["validacion"]].reset_index(drop=True)
st.caption(f"Ensamble: {len(ens['miembros'])} modelos ({len(ens['miembros']) - 1} de CV externa + final) · umbral operativo de severa (Validación) = {umbral:.3f} · "
           "intervalo: bootstrap percentil 95 % sobre los miembros (B=1000), mide variabilidad del modelo.")
# ------------------------------------------------------------------ presentación desde el snapshot (lo mismo que recibe el reporte)
def _botones_reporte(sn: dict, clave: str) -> None:
    """Botones Word / Excel / PDF del reporte de ESTA inferencia. Los archivos (figuras a 600 DPI, Word, Excel y PDF) se generan una sola vez
    por snapshot y SOLO cuando el usuario lo pide, para que la pantalla de resultados no se bloquee."""
    k = f"inf_archivos_{sn['id']}"
    st.markdown("**Reporte de esta inferencia**")
    if k not in st.session_state:
        st.caption("La generación del reporte (figuras a 600 DPI, Word, Excel y PDF) tarda de 20 a 60 segundos según el tipo de inferencia; la pantalla de resultados no espera por ella.")
        if st.button("Preparar reporte (Word, Excel y PDF)", key=f"prep_{clave}_{sn['id']}"):
            with st.spinner("Preparando el reporte (figuras a 600 DPI, Word, Excel y PDF)..."):
                try:
                    st.session_state[k] = RI.generar_archivos(sn)
                except Exception as exc:  # noqa: BLE001
                    st.session_state[k] = exc
            st.rerun()
        return
    arch = st.session_state[k]
    if isinstance(arch, Exception):
        st.error(explicar_error(arch, "generar el reporte de la inferencia"))
        if st.button("Reintentar", key=f"reint_{clave}_{sn['id']}"):
            del st.session_state[k]
            st.rerun()
        return
    cols = st.columns(3)
    for col, fmt in zip(cols, ("word", "excel", "pdf")):
        ext, mime, etiqueta = RI.FORMATOS[fmt]
        col.download_button(f"Descargar {etiqueta}", data=arch[fmt], file_name=RI.nombre_archivo(sn, fmt), mime=mime, key=f"dl_{clave}_{fmt}_{sn['id']}",
                            on_click=audit, args=("reporte_inferencia_descargado", "prediccion", f"{sn['modelo']['corrida']}/{sn['modelo']['id']}",
                                                  {"id_inferencia": sn["id"], "formato": ext, "bytes": len(arch[fmt])}))
    st.caption(f"Incluye los datos ingresados, el modelo y la corrida, el resultado, la explicación (SHAP y LIME), la sensibilidad, el contrafactual, la comparación con los demás modelos, la validez de las entradas, "
               f"todos los gráficos (PNG a 600 DPI) con sus tablas, la interpretación, la trazabilidad (id {sn['id']}) y las advertencias.")


def _analisis_locales(sn: dict) -> None:
    """LIME, sensibilidad, contrafactual, otros modelos y validez de entradas (todo viene del snapshot congelado)."""
    err = sn.get("extra_err") or {}
    st.markdown("**Análisis adicionales del modelo para este hogar**")
    st.caption("Describen cómo responde el MODELO (asociaciones predictivas aprendidas de otros hogares). No son efectos causales ni recomendaciones de política, y la capacidad predictiva del modelo es modesta.")
    t_l, t_s, t_c, t_o, t_v = st.tabs(["LIME y SHAP", "Sensibilidad", "Contrafactual", "Otros modelos", "Validez de las entradas"])
    with t_l:
        L = sn.get("lime")
        if L is None:
            st.warning(f"No se pudo calcular LIME: {err.get('lime', 'sin detalle')}")
        else:
            st.plotly_chart(P.fig_lime(L["tabla"], L["clase"]), key="lime_ind")
            interpretacion(sn["interp"]["lime"])
            c = sn.get("lime_shap")
            if c and c.get("disponible"):
                st.plotly_chart(P.fig_lime_shap(c["tabla"]), key="lime_shap_ind")
                interpretacion(sn["interp"]["lime_shap"])
                with st.expander("Tabla LIME frente a SHAP"):
                    st.dataframe(c["tabla"].drop(columns=["abs LIME", "abs SHAP"]).round(4), hide_index=True)
            st.caption(f"LIME: {L['n_muestras']:,} perturbaciones, semilla {L['semilla']}, fondo de {L['n_fondo']:,} hogares de Desarrollo, R² local {L['r2_local']:.2f}. Aproximación local del modelo final.")
    with t_s:
        S = sn.get("sens")
        if S is None:
            st.warning(f"No se pudo calcular la sensibilidad: {err.get('sens', 'sin detalle')}")
        else:
            st.markdown("**Curvas de sensibilidad (una variable a la vez, el resto del hogar fijo)**")
            for i in range(0, len(S["principales"]), 2):
                cols = st.columns(2)
                for col, v in zip(cols, S["principales"][i:i + 2]):
                    with col:
                        st.plotly_chart(P.fig_sens_curva(S["curvas"], v, S["tipos"][v]), key=f"sens_{v}")
            interpretacion(sn["interp"]["sens"])
            with st.expander("Tabla completa de las curvas"):
                st.dataframe(S["curvas"].drop(columns=["rol"]).round(4), hide_index=True)
            st.markdown("**Tornado: efecto de mover cada variable entre su percentil 10 y su percentil 90**")
            st.plotly_chart(P.fig_tornado(S["tornado"]), key="tornado_ind")
            interpretacion(sn["interp"]["tornado"])
            with st.expander("Tabla del tornado"):
                st.dataframe(S["tornado"].drop(columns=["variable"]).round(4), hide_index=True)
    with t_c:
        cf = sn.get("contra")
        if cf is None:
            st.warning(f"No se pudo calcular el contrafactual: {err.get('contra', 'sin detalle')}")
        elif not cf.get("aplica", False):
            st.info("El contrafactual no aplica a este modelo. " + cf.get("motivo", ""))
        else:
            st.warning("Asociación predictiva del modelo, NO un efecto causal ni una recomendación de política. La lista de variables accionables está [POR VALIDAR].")
            st.caption(f"Objetivo: P(moderada o severa) < {cf['objetivo']:.1%} (reducción relativa del {AL.CF_REDUCCION_RELATIVA:.0%} desde {cf['p_mod_sev_base']:.1%}); hasta {cf['max_cambios']} cambios por escenario, valores entre el actual y el percentil 90 (o 10) de Desarrollo.")
            if cf["escenarios"]:
                st.plotly_chart(P.fig_contrafactual(cf), key="cf_ind")
                filas = [{"Escenario": e["n"], "Cambios": RI._txt_cambios(e), "Distancia": round(e["distancia"], 3), "P(mod.+sev.) nueva": round(e["p_mod_sev"], 4), "Reducción (pp)": round(e["reduccion_pp"], 2),
                          "¿Alcanza el objetivo?": "Sí" if e["alcanza_objetivo"] else "No", "Miembros que reducen P": f"{e['miembros_que_reducen']} de {e['n_miembros']}"} for e in cf["escenarios"]]
                st.dataframe(pd.DataFrame(filas), hide_index=True)
            interpretacion(sn["interp"]["contra"])
            with st.expander("Variables accionables y no accionables (decisión y justificación)"):
                st.dataframe(E.accionables_df(), hide_index=True)
    with t_o:
        t = sn.get("otros")
        if t is None or len(t) == 0:
            st.warning(f"No se pudo comparar con los demás modelos: {err.get('otros', 'sin detalle')}")
        else:
            st.plotly_chart(P.fig_otros_modelos(t), key="otros_ind")
            st.dataframe(t.drop(columns=["ident", "codigo"]).round(4), hide_index=True)
            interpretacion(sn["interp"]["otros"])
    with t_v:
        v = sn.get("validez")
        if v is None:
            st.warning(f"No se pudo evaluar la validez de las entradas: {err.get('validez', 'sin detalle')}")
        else:
            fuera = v[v["Estado"] != "dentro"]
            (st.warning if len(fuera) else st.success)(sn["interp"]["validez"]["que_muestra"])
            st.dataframe(v, hide_index=True)
    st.caption("Tiempos de cálculo (s): " + ", ".join(f"{k} {x}" for k, x in (sn.get("tiempos") or {}).items()))


def _perfil_hogar(sn: dict) -> None:
    """Grupos del perfil de vulnerabilidad a los que pertenece el hogar y prevalencia observada del grupo (contexto, no predicción)."""
    ctx = sn.get("perfil_ctx") or {}
    st.markdown("**Perfil de vulnerabilidad del hogar**")
    if sn.get("perfil_error"):
        st.warning(f"No se pudo calcular la prevalencia de referencia del perfil: {sn['perfil_error']}")
    for t in PV.texto_contexto_hogar(ctx):
        st.markdown(f"- {t}")
    st.dataframe(PV.tabla_contexto_hogar(ctx), hide_index=True)
    st.caption("Perfil descriptivo: no es un índice ponderado ni validado ni incluye desenlaces nutricionales directos. La prevalencia del grupo describe al conjunto analítico (IC95 aproximado) y no es la probabilidad de este hogar.")


def _mostrar_individual(sn: dict) -> None:
    m = sn["modelo"]
    st.divider()
    st.caption(f"Inferencia {sn['id']} · {sn['fecha'].replace('T', ' ')} · corrida {m['corrida']} · modelo {m['codigo']} ({m['escenario']}). "
               "El resultado queda congelado: no cambia si modifica los controles; vuelva a pulsar «Predecir» para una nueva inferencia.")
    if (m["corrida"], m["id"]) != (nombre, ident):
        st.info(f"Este resultado corresponde a {m['corrida']} / {m['id']}, distinto de la selección actual.")
    media, lo, hi, nivel, p3, umbral_s = sn["media"], sn["ic_inf"], sn["ic_sup"], sn["nivel"], sn["p_severa"], sn["umbral"]
    a, b, c, d = st.columns(4)
    a.metric("Nivel predicho (0-3)", nivel)
    b.metric("P(IA severa)", f"{p3:.1%}", f"IC [{lo[3]:.1%}; {hi[3]:.1%}]", delta_color="off")
    c.metric("¿Supera el umbral operativo?", "Sí" if sn["supera"] else "No", f"umbral {umbral_s:.3f}", delta_color="off")
    d.metric("Percentil poblacional de P(severa)", f"{sn['percentil']:.0%}")
    st.plotly_chart(P.fig_probabilidades(media, lo, hi), key="prob_ind")
    tabla = pd.DataFrame({"nivel": C.CLASES, "probabilidad": media, "IC inf": lo, "IC sup": hi, "mín. miembros": sn["min"], "máx. miembros": sn["max"]})
    st.dataframe(tabla.round(4), hide_index=True)
    interpretacion(sn["interp"]["prediccion"])
    st.markdown("**Explicación local (SHAP del modelo final)**")
    if sn["shap"]:
        st.plotly_chart(P.fig_contribuciones(sn["shap"]["nombres"], np.array(sn["shap"]["valores"])), key="shap_ind")
        interpretacion(sn["interp"]["shap"])
    else:
        st.warning(f"No se pudo calcular la explicación SHAP: {sn['shap_error']}")
    st.markdown("**Comparación con la distribución poblacional (Desarrollo, ponderada)**")
    if len(sn["comparacion"]):
        st.dataframe(sn["comparacion"].drop(columns=["código"]).round(3), hide_index=True)
        interpretacion(sn["interp"]["comparacion"])
    _perfil_hogar(sn)
    _analisis_locales(sn)
    st.caption(f"Predicción registrada en fsai.auditoria (huella de entrada {sn['huella'][:16]}; no se guardan los valores del hogar).")
    _botones_reporte(sn, "ind")


def _mostrar_lote(sn: dict) -> None:
    import plotly.graph_objects as go
    m, out = sn["modelo"], sn["salida"]
    st.divider()
    st.caption(f"Lote {sn['id']} · {sn['fecha'].replace('T', ' ')} · {sn['archivo'] or 'CSV'} · corrida {m['corrida']} · modelo {m['codigo']} ({m['escenario']}).")
    if (m["corrida"], m["id"]) != (nombre, ident):
        st.info(f"Este resultado corresponde a {m['corrida']} / {m['id']}, distinto de la selección actual.")
    st.dataframe(out.round(4).head(200), hide_index=True)
    st.download_button("Descargar predicciones", data=pd.concat([sn["entradas"], out], axis=1).to_csv(index=False), file_name="predicciones_lote.csv", mime="text/csv", key=f"csv_{sn['id']}")
    pred = sn["conteo_clases"] / max(sn["n"], 1)
    f = go.Figure([go.Bar(name="Lote (nivel predicho)", x=[f"Nivel {k}" for k in C.CLASES], y=100 * pred, marker_color=P.OKABE_ITO[0]),
                   go.Bar(name="Prevalencia de referencia (Desarrollo)", x=[f"Nivel {k}" for k in C.CLASES], y=100 * sn["prevalencia"], marker_color=P.OKABE_ITO[1])])
    f.update_layout(barmode="group", yaxis_title="Hogares (%)", margin={"t": 20})
    st.plotly_chart(f, key="lote_clases")
    h = np.histogram(out["p_clase_3"].to_numpy(), bins=10)
    f2 = go.Figure(go.Bar(x=[f"{a:.3f}-{b:.3f}" for a, b in zip(h[1][:-1], h[1][1:])], y=h[0], marker_color=P.OKABE_ITO[0]))
    f2.update_layout(xaxis_title="P(IA severa)", yaxis_title="Hogares", margin={"t": 20})
    st.plotly_chart(f2, key="lote_psevera")
    interpretacion(sn["interp"]["lote"])
    pl = sn.get("perfil_lote")
    st.markdown("**Perfil de vulnerabilidad de los hogares del lote**")
    if pl is None:
        st.info("El CSV no trae las columnas del perfil (n_menores5, n_adultos_mayores, pobreza_monetaria); no se calculó la distribución por recuento.")
    else:
        st.dataframe(pl["por_caracteristica"], hide_index=True)
        st.dataframe(pl["por_recuento"], hide_index=True)
        st.caption("Distribución del lote por características y recuento (0 a 3) con la prevalencia observada del conjunto de referencia (contexto, no predicción). "
                   "Perfil descriptivo: no es un índice ponderado ni validado." + (f" {pl['n_sin_recuento']:,} hogares sin dato en alguna característica quedan fuera del recuento." if pl["n_sin_recuento"] else ""))
    if sn.get("sens_lote") is not None:
        st.markdown(f"**Sensibilidad agregada (muestra de {sn['sens_lote']['n_muestra']:,} hogares del lote): efecto medio de mover cada variable entre su percentil 10 y su percentil 90**")
        t = sn["sens_lote"]["tabla"].rename(columns={"valor bajo (p10 / 0)": "valor bajo", "valor alto (p90 / 1)": "valor alto", "efecto medio (pp)": "efecto (pp)"}).copy()
        for c in ("valor bajo", "valor alto"):
            t[c] = t[c].map(lambda x: x if isinstance(x, str) else f"{float(x):.4g}")
        st.plotly_chart(P.fig_tornado(t), key="tornado_lote")
    if sn.get("validez_lote") is not None:
        v = sn["validez_lote"]
        n_f = int((v["Hogares fuera de rango"] > 0).sum())
        (st.warning if n_f else st.success)(f"{n_f} de {len(v)} variables tienen hogares fuera del rango observado en Desarrollo (advertencia de extrapolación)." if n_f else "Ningún hogar del lote cae fuera del rango observado en Desarrollo.")
        with st.expander("Validez de las entradas del lote"):
            st.dataframe(v, hide_index=True)
    st.caption("Por lote no se calculan LIME, contrafactual ni la comparación hogar a hogar con los demás modelos; use la predicción individual.")
    _botones_reporte(sn, "lote")


tab_i, tab_l = st.tabs(["Predicción individual", "Predicción por lote (CSV)"])
FEATURES = res["features_num"] + res["features_cat"]


def _opciones_formulario(clave: str, spec_por_var: dict) -> dict:
    """Opciones {valor del modelo: etiqueta oficial} de una entrada, limitadas a lo que el modelo vio en Desarrollo."""
    obs = spec_por_var.get(clave, {}).get("opciones")
    if clave == "nivel_educ_jefe":      # numérica en el modelo: se ofrecen los niveles observados
        obs = [v for v in K.NIVEL_EDUC_MODELO if v in set(pd.to_numeric(dev[clave], errors="coerce").dropna().astype(int))]
    elif clave == "pobreza_monetaria":
        obs = [v for v in K.POBREZA if v in set(pd.to_numeric(dev[clave], errors="coerce").dropna().astype(int))]
    return K.opciones_de(clave, obs)


def _campo(col, clave: str, ini, opciones: dict | None, prefijo: str):
    """Dibuja UN campo del catálogo: pregunta clara, una línea de ayuda y detalle en el tooltip; devuelve el valor ingresado."""
    e = K.ENTRADAS[clave]
    k = f"{prefijo}_{clave}"
    det = e["detalle"] or None
    if k not in st.session_state:            # valor inicial por session_state (sin «value»/«index» para no chocar con el botón «Cargar un ejemplo»)
        if e["tipo"] == "entero":
            ini = int(min(max(ini, e["min"]), e["max"]))
        elif e["tipo"] == "decimal":
            ini = float(min(max(ini, e["min"]), e["max"]))
        elif e["tipo"] == "opciones" and ini not in opciones:
            ini = list(opciones)[0]
        st.session_state[k] = ini
    if e["tipo"] == "si_no":
        v = col.radio(e["etiqueta"], [1, 0], key=k, horizontal=True, format_func=lambda o: "Sí" if o == 1 else "No", help=det)
    elif e["tipo"] == "opciones":
        v = col.selectbox(e["etiqueta"], list(opciones), key=k, format_func=lambda o, d=opciones: d[o], help=det)
    elif e["tipo"] == "entero":
        v = col.number_input(e["etiqueta"], int(e["min"]), int(e["max"]), step=1, key=k, format="%d", help=det)
    else:
        v = col.number_input(e["etiqueta"], float(e["min"]), float(e["max"]), step=float(e["paso"]), key=k, format="%.2f", help=det)
    col.caption(e["ayuda"])
    return v


def _derivadas_seccion(sec: str, entradas: dict, spec_por_var: dict) -> None:
    for var in FEATURES:
        d = K.DERIVADOS.get(var)
        if d is None or d["seccion"] != sec or any(x not in entradas for x in d["entradas"]):
            continue
        val = d["fn"](entradas)
        txt = "no aplica (dato faltante, igual que en el entrenamiento)" if pd.isna(val) else f"{val:.2f}".replace(".", ",")
        st.markdown(f"**{d['etiqueta']}:** {txt} · _{d['formula']}_", help=d.get("detalle"))
        st.caption(d["ayuda"])


with tab_i:
    spec = INF.especificacion_formulario(res, dev)
    spec_por_var = {c["variable"]: c for c in spec}
    claves = K.entradas_para(FEATURES)
    with st.expander("Opcional: partir de un hogar real de la muestra de Validación"):
        pre = st.number_input(f"Número de hogar (0 a {len(val) - 1})", 0, len(val) - 1, 0, help="Solo para explorar: rellena el formulario con un hogar de Validación; puede editar cada valor. "
                              "Las habitaciones y los escolares se reconstruyen de forma aproximada a partir de los cocientes guardados.")
    prefijo = f"inf_{ident}_{int(pre)}"
    base = K.entradas_desde_fila(val.iloc[int(pre)])
    ops = {k: _opciones_formulario(k, spec_por_var) for k in claves if K.ENTRADAS[k]["tipo"] == "opciones"}

    def _cargar_ejemplo() -> None:
        for k in claves:
            if k in K.EJEMPLO and (k not in ops or K.EJEMPLO[k] in ops[k]):
                st.session_state[f"{prefijo}_{k}"] = K.EJEMPLO[k]
        st.session_state["inf_ejemplo_activo"] = prefijo

    cA, cB = st.columns([1, 3])
    cA.button("Cargar un ejemplo", on_click=_cargar_ejemplo, help="Llena todo el formulario con un hogar ilustrativo inventado para ver cómo se responde.")
    cB.caption("Responda cada pregunta con los datos del hogar. Las variables que son cocientes se calculan solas. Cada campo explica qué es y en qué unidad se escribe.")
    if st.session_state.get("inf_ejemplo_activo") == prefijo:
        st.info(K.EJEMPLO_AVISO + " Puede editar cualquier campo.")
    entradas: dict = {}
    for sec, titulo, intro in K.SECCIONES:
        ks = [k for k in claves if K.ENTRADAS[k]["seccion"] == sec]
        if not ks:
            continue
        st.markdown(f"#### {titulo}")
        st.caption(intro)
        if sec == "fies":
            st.caption(K.FIES_LIMITACION)
        for i, k in enumerate(ks):
            if i % 3 == 0:            # una fila de columnas por cada tres campos: en el teléfono se apilan en el orden correcto
                cols = st.columns(3)
            entradas[k] = _campo(cols[i % 3], k, base.get(k, K.ENTRADAS[k]["defecto"]), ops.get(k), prefijo)
        _derivadas_seccion(sec, entradas, spec_por_var)
    errores = K.validar_entradas(entradas)
    valores = K.valores_modelo(entradas, FEATURES)
    for e_ in errores:
        st.error(e_)
    fuera = []
    for v, x in valores.items():
        c = spec_por_var[v]
        if c["tipo"] != "categorica" and not pd.isna(x) and not (c["min"] <= float(x) <= c["max"]):
            fuera.append(f"«{E.etq(v)}» = {float(x):.4g}, fuera del rango observado en Desarrollo ({c['min']:.4g} a {c['max']:.4g}).")
    if fuera and not errores:
        st.warning("Extrapolación: el modelo no vio valores así en los hogares con que se entrenó, por lo que la predicción es menos fiable. " + " ".join(fuera))
    valores = {v: (str(x) if spec_por_var[v]["tipo"] == "categorica" else (np.nan if pd.isna(x) else float(x))) for v, x in valores.items()}
    enviar = st.button("Predecir", type="primary", disabled=bool(errores), help="Corrija los datos marcados en rojo para poder predecir." if errores else None)
    if enviar:
        with st.spinner("Calculando la inferencia (ensamble, SHAP, LIME, sensibilidad, contrafactual y comparación con los demás modelos)..."):
            sn = RI.ejecutar_inferencia_individual(nombre_corrida=nombre, ident=ident, dir_run=dir_run, res=res, ens=ens, dev=dev, valores=valores,
                                                   usuario=usuario, dataset={"nombre": info.get("nombre", "dataset activo"), "n": len(df)}, df_perfil=df)
        st.session_state["inf_ind"] = sn          # snapshot congelado: la pantalla y el reporte leen de aquí, no de los controles
        audit("inferencia_individual", "prediccion", f"{nombre}/{ident}",
              {"huella_entrada": sn["huella"][:16], "id_inferencia": sn["id"], "nivel_predicho": sn["nivel"], "p_severa": round(sn["p_severa"], 4),
               "supera_umbral": bool(sn["supera"]), "metodo": "ensamble_bootstrap_miembros"})
    sn = st.session_state.get("inf_ind")
    if sn is not None:
        _mostrar_individual(sn)


# ------------------------------------------------------------------ lote
with tab_l:
    req = FEATURES
    obs_cat = {c["variable"]: c["opciones"] for c in spec if c["tipo"] == "categorica"}
    rangos = {c["variable"]: (c["min"], c["max"]) for c in spec if c["tipo"] != "categorica"}
    st.write(f"El CSV debe traer estas **{len(req)} columnas** con el **nombre de columna** de la primera columna de la tabla como encabezado (máx. {INF.MAX_FILAS_LOTE} filas). "
             "Las variables marcadas «derivado» son cocientes: escriba el valor ya calculado con la fórmula indicada.")
    st.dataframe(K.catalogo_df(req, obs_cat, rangos), hide_index=True)
    if any(v.startswith("fies_") for v in req):
        st.caption(K.FIES_LIMITACION)
    ej = K.valores_modelo(K.EJEMPLO, req)
    plantilla = ",".join(req) + "\n"
    ejemplo_csv = ",".join(req) + "\n" + ",".join("" if pd.isna(ej[c]) else (str(ej[c]) if isinstance(ej[c], str) else f"{float(ej[c]):.6g}") for c in req) + "\n"
    d1, d2 = st.columns(2)
    d1.download_button("Descargar plantilla CSV (solo encabezados)", data=plantilla, file_name="plantilla_lote.csv", mime="text/csv")
    d2.download_button("Descargar plantilla con una fila de ejemplo", data=ejemplo_csv, file_name="plantilla_lote_ejemplo.csv", mime="text/csv",
                       help=K.EJEMPLO_AVISO + " Borre o reemplace la fila antes de subir sus datos.")
    subida = st.file_uploader("CSV de hogares", type=["csv"], key="lote_csv")

    if subida is not None and st.button("Predecir lote"):
        try:
            dfl = pd.read_csv(io.BytesIO(subida.getvalue()))
            with st.spinner("Calculando el lote..."):
                sn_l = RI.ejecutar_inferencia_lote(nombre_corrida=nombre, ident=ident, dir_run=dir_run, res=res, ens=ens, dev=dev, dfl=dfl, bytes_csv=subida.getvalue(),
                                                   nombre_archivo=subida.name, usuario=usuario, dataset={"nombre": info.get("nombre", "dataset activo"), "n": len(df)}, df_perfil=df)
            st.session_state["inf_lote"] = sn_l
            audit("inferencia_lote", "prediccion", f"{nombre}/{ident}", {"n_filas": sn_l["n"], "alertas": sn_l["n_alertas"], "id_inferencia": sn_l["id"], "huella_archivo": sn_l["huella"][:16]})
        except (ValueError, pd.errors.ParserError) as exc:
            st.error(f"El CSV no se pudo leer: {exc}. Verifique que sea un CSV con coma como separador y use la plantilla descargable (encabezados con el nombre de columna indicado en la tabla).")
    sn_l = st.session_state.get("inf_lote")
    if sn_l is not None:
        _mostrar_lote(sn_l)
