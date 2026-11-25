"""Interpretabilidad SHAP: explicaciones almacenadas (PostgreSQL) + explicabilidad de los modelos de la CV anidada
(SHAP, LIME, permutación, PDP/ALE, mapas de calor de |SHAP| y máscaras de variables).

Sección 1 -

Esquema JSONB esperado en fsai.explicaciones.contribuciones (convención de esta aplicación;
debe producirlo explainability.contribuciones_a_jsonb):

* tipo = 'shap_global' (hogar_uid NULL):
  {"version": 1, "base_value": float, "n_muestra": int,
   "importancia": {variable: media_abs_shap},
   "por_clase": {"<clase>": {variable: media_abs_shap}},          # opcional
   "beeswarm": {"variables": [nombres], "shap": [filas x variables],
               "valor_normalizado": [filas x variables]}}                # opcional
* tipo = 'shap_local' (hogar_uid informado):
  {"version": 1, "clase_explicada": int, "base_value": float, "prediccion": float,
   "contribuciones": {variable: shap}, "valores": {variable: valor_observado}}
"""
from __future__ import annotations

import re
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import tema
from auth import barra_lateral, requerir_acceso
from tema import config_pagina, encabezado
from comun import (
    ErrorConsulta,
    consultar,
    fmt_num,
    leer_json,
    mostrar_error_bd,
    registrar_auditoria,
)

import joblib
import pandas as pd

from fases_comun import AVISO_CIRCULARIDAD, C, audit, dataset_id_activo, interpretacion, selector_dataset

from fsai import etiquetas as E  # noqa: E402
from fsai import evaluacion as EV  # noqa: E402
from fsai import explainability as X  # noqa: E402
from fsai import interpretacion as I  # noqa: E402
from fsai import plots as P  # noqa: E402
from fsai.preprocessing import preparar_matriz  # noqa: E402
from fsai.splits import cargar_particion  # noqa: E402

config_pagina("SHAP")
requerir_acceso("shap")
barra_lateral()

encabezado("Interpretabilidad",
    "Qué variables explican las predicciones del modelo: SHAP, LIME, PDP/ALE, mapas de calor y máscaras de variables.",
    "Evaluación e interpretación",
    pasos=["Elija una corrida, un modelo y la clase a explicar (por defecto, inseguridad severa).",
           "Revise la importancia global y el mapa de calor de |SHAP|; use LIME o SHAP local para un hogar concreto.",
           "Valide la explicación con las curvas de eliminación/inserción y revise PDP/ALE de las variables principales."],
    produce="Importancias globales y locales con su interpretación (asociaciones predictivas, no efectos causales).",
)
st.caption(
    "Los valores SHAP describen cuánto contribuye cada variable a la salida del modelo "
    "(Lundberg y Lee, 2017). Son asociaciones predictivas, no efectos causales."
)

SEMILLA = 42
PATRON_UID = re.compile(r"^[0-9a-f]{64}$")

SQL_MODELOS = """
SELECT mo.modelo_id, mo.codigo, e.nombre, e.escenario, e.esquema_validacion
FROM fsai.modelos mo
JOIN fsai.experimentos e ON e.experimento_id = mo.experimento_id
WHERE e.dataset_id = :dataset AND EXISTS (
  SELECT 1 FROM fsai.explicaciones x
  WHERE x.modelo_id = mo.modelo_id AND x.tipo IN ('shap_global', 'shap_local')
)
ORDER BY mo.creado_en DESC LIMIT 200
"""
SQL_GLOBAL = """
SELECT contribuciones FROM fsai.explicaciones
WHERE modelo_id = :modelo_id AND tipo = 'shap_global' AND hogar_uid IS NULL
ORDER BY creado_en DESC LIMIT 1
"""
SQL_HOGARES_LOCAL = """
SELECT hogar_uid FROM fsai.explicaciones
WHERE modelo_id = :modelo_id AND tipo = 'shap_local' AND hogar_uid IS NOT NULL
ORDER BY creado_en DESC LIMIT 200
"""
SQL_LOCAL = """
SELECT contribuciones FROM fsai.explicaciones
WHERE modelo_id = :modelo_id AND tipo = 'shap_local' AND hogar_uid = :hogar_uid
ORDER BY creado_en DESC LIMIT 1
"""
SQL_CONTEXTO = """
SELECT h.departamento, h.area, h.region_natural, r.fies_nivel
FROM fsai.hogares h LEFT JOIN fsai.respuestas_fies r ON r.hogar_uid = h.hogar_uid
WHERE h.hogar_uid = :hogar_uid
"""


def grafico_importancia(importancia: dict[str, float], top_n: int, titulo: str) -> go.Figure:
    pares = sorted(
        ((str(k), float(v)) for k, v in importancia.items() if v is not None),
        key=lambda p: p[1], reverse=True,
    )[:top_n]
    pares.reverse()
    fig = go.Figure(
        go.Bar(x=[p[1] for p in pares], y=[E.etq(p[0]) for p in pares], orientation="h",
               marker_color="#1b6ca8")
    )
    fig.update_layout(
        title=titulo, xaxis_title="Media del |valor SHAP|", height=max(320, 26 * len(pares)),
        margin={"l": 0, "r": 10, "t": 40, "b": 0},
    )
    return fig


def grafico_beeswarm(bee: dict, top_n: int, max_puntos: int, clase: int | None = None) -> go.Figure | None:
    """Beeswarm: un punto por hogar; posición = SHAP firmado (empuje hacia/contra la clase), color = valor normalizado de la variable
    (azul bajo, rojo alto). Las variables sin valor ordenable (categóricas) se dibujan en gris. Jitter vertical con semilla fija."""
    variables = [str(v) for v in bee.get("variables", [])]
    try:
        shap_vals = np.asarray(bee["shap"], dtype=float)
        colores = np.asarray(bee["valor_normalizado"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if shap_vals.ndim != 2 or shap_vals.shape != colores.shape or shap_vals.shape[1] != len(variables):
        return None
    rng = np.random.default_rng(SEMILLA)
    if shap_vals.shape[0] > max_puntos:
        filas = rng.choice(shap_vals.shape[0], size=max_puntos, replace=False)
        shap_vals, colores = shap_vals[filas], colores[filas]
    orden = np.argsort(np.nanmean(np.abs(shap_vals), axis=0))[::-1][:top_n][::-1]
    x, y, c = [], [], []
    for pos, j in enumerate(orden):
        x.append(shap_vals[:, j])
        y.append(pos + rng.uniform(-0.3, 0.3, size=shap_vals.shape[0]))
        c.append(colores[:, j])
    x, y, c = np.concatenate(x), np.concatenate(y), np.concatenate(c)
    gris = np.isnan(c)
    fig = go.Figure()
    if gris.any():
        fig.add_trace(go.Scattergl(x=x[gris], y=y[gris], mode="markers", marker={"size": 5, "color": "#9AA3AB", "opacity": 0.6}, hoverinfo="x", name="Sin escala de valor"))
    fig.add_trace(go.Scattergl(
        x=x[~gris], y=y[~gris], mode="markers", name="Valor de la variable", hoverinfo="x", showlegend=False,
        marker={"size": 5, "color": c[~gris], "colorscale": "RdBu_r", "cmin": 0, "cmax": 1, "opacity": 0.7,
                "colorbar": {"title": {"text": "Valor de la variable", "side": "top"}, "tickvals": [0, 1], "ticktext": ["bajo", "alto"], "orientation": "h",
                             "len": 0.5, "thickness": 12, "y": -0.2, "yanchor": "top", "x": 0.5}}))
    fig.update_layout(
        xaxis_title="Valor SHAP" + (f" (derecha: empuja hacia el nivel {clase})" if clase is not None else ""),
        height=max(460, 32 * len(orden) + 190), showlegend=bool(gris.any()), legend={"orientation": "h", "y": -0.42, "x": 0.5, "xanchor": "center"},
        yaxis={"automargin": True, "tickmode": "array", "tickvals": list(range(len(orden))), "ticktext": [E.etq(variables[j]) for j in orden]},
        margin={"l": 0, "r": 10, "t": 10, "b": 0},
    )
    fig.add_vline(x=0, line_dash="dot", line_color="#888888")
    return fig


def _seg(x: float) -> str:
    return "menos de 1 s" if x < 1 else f"{x:.1f} s"


LECTURA_BEESWARM = (
    "**Cómo leerlo:** cada punto es un hogar. Su posición horizontal indica cuánto empuja esa variable la predicción del modelo hacia el nivel "
    "elegido (derecha) o en contra (izquierda); el color indica si el valor de la variable en ese hogar es bajo (azul) o alto (rojo) respecto "
    "de los demás hogares de la muestra. Las variables en gris son categorías sin orden (no tienen «alto» ni «bajo»). Describe cómo razona el "
    "modelo: son asociaciones predictivas, no causas.")


@st.cache_resource(show_spinner=False)
def _cargar_modelo(ruta: str, mtime: float):
    return joblib.load(ruta)


@st.cache_data(show_spinner=False, max_entries=8)
def _beeswarm_bajo_demanda(_df: pd.DataFrame, ruta_corrida: str, ident: str, fuente: str, n: int, clase: int):
    """SHAP firmado de una submuestra estratificada por nivel FIES (Validación o Prueba ya evaluada; nunca el entrenamiento)."""
    from pathlib import Path
    dir_run = Path(ruta_corrida)
    res = EV.resultados_corrida(dir_run.name)[ident]
    pipe = _cargar_modelo(str(dir_run / res["final"]["modelo"]), (dir_run / res["final"]["modelo"]).stat().st_mtime)
    part, _ = cargar_particion(_df, dir_run / "particion.json")
    dsel = _df.iloc[part["validacion"] if fuente.startswith("Validación") else part["prueba"]].reset_index(drop=True)
    Xs = preparar_matriz(dsel, res["features_num"], res["features_cat"])
    t0 = time.perf_counter()
    sm = X.shap_matriz(pipe, Xs, res["codigo"], n, estrato=dsel[C.COL_OBJETIVO].to_numpy())
    bee = X.beeswarm_datos(sm["vals"][C.CLASES.index(clase)], Xs.loc[sm["indice"]], sm["nombres"], clase=clase, split=fuente, top=60, max_filas=n)
    niveles = dsel.loc[sm["indice"], C.COL_OBJETIVO].value_counts().sort_index()
    return {"bee": bee, "segundos": time.perf_counter() - t0, "n": int(len(sm["indice"])),
            "por_nivel": {int(k): int(v) for k, v in niveles.items()}}


def mostrar_beeswarm(r: dict, clase: int, top_n: int, fuente: str, codigo: str) -> None:
    """Dibuja el beeswarm calculado por ``_beeswarm_bajo_demanda`` con su lectura y la nota ética."""
    fig = grafico_beeswarm(r["bee"], top_n, 5000, clase)
    if fig is None:
        st.warning("El beeswarm calculado no tiene la estructura esperada.")
        return
    st.plotly_chart(fig)
    st.markdown(LECTURA_BEESWARM)
    st.caption(
        f"Muestra de {r['n']} hogares de {fuente} con reparto parejo entre niveles FIES (hogares por nivel: "
        + ", ".join(f"{k}: {v}" for k, v in r["por_nivel"].items())
        + f"), semilla 42; no representa la prevalencia real. SHAP de la clase {clase} del modelo {codigo}, con las categorías de cada variable sumadas. "
        f"Calculado en {_seg(r['segundos'])}. Nunca se usa el entrenamiento.")


def _beeswarm_faltante(fila) -> None:
    """Este modelo guardó solo importancias promedio: explica qué falta y ofrece calcular el beeswarm AHORA con el modelo guardado."""
    tema.vacio("Este modelo guardó solo la importancia promedio de cada variable, no los puntos por hogar que necesita el beeswarm. "
               "Puede calcularlos ahora con el modelo guardado.", "Elija la corrida y pulse «Calcular el beeswarm»", titulo="Beeswarm: se calcula bajo demanda")
    df_, _info = st.session_state["_df_activo"]
    corridas = EV.listar_corridas()
    if not corridas:
        st.caption("No hay corridas con modelos guardados en este equipo, por lo que no se puede calcular.")
        return
    ident = f"{fila['escenario']}_{fila['codigo']}"
    c1, c2, c3 = st.columns(3)
    nombre = c1.selectbox("Corrida con el modelo guardado", corridas, key="bs_alm_corrida")
    if ident not in EV.resultados_corrida(nombre):
        st.warning(f"La corrida «{nombre}» no contiene el modelo {ident}; elija otra corrida.")
        return
    fuente = c2.selectbox("Hogares", ["Validación (15 %)"], key="bs_alm_fuente")
    n_max = 60 if fila["codigo"] == "M4" else 300
    n = c3.slider("Hogares (rendimiento)", 20, n_max, min(100, n_max), step=10, key="bs_alm_n",
                  help="Más hogares = más puntos pero más tiempo. En una prueba, el modelo M2 con 150 hogares tardó unos 3 minutos; M1 es casi inmediato.")
    if fila["codigo"] == "M4":
        st.caption("M4 usa KernelSHAP, que es lento: se limita a 60 hogares.")
    if st.button("Calcular el beeswarm", key="bs_alm_btn", type="primary"):
        with st.status(f"Calculando SHAP de {ident} sobre {n} hogares...", expanded=True) as estado:
            try:
                st.session_state["bs_alm_res"] = (ident, nombre, _beeswarm_bajo_demanda(df_, str(EV.dir_corrida(nombre)), ident, fuente, n, 3))
                estado.update(label="Beeswarm listo", state="complete", expanded=False)
            except Exception as exc:  # noqa: BLE001
                estado.update(label="No se pudo calcular", state="error")
                st.error(f"No se pudo calcular el beeswarm de {ident}: {type(exc).__name__}: {exc}")
    r = st.session_state.get("bs_alm_res")
    if r and r[0] == ident:
        mostrar_beeswarm(r[2], 3, 15, "Validación", fila["codigo"])


def seccion_almacenada() -> None:
    """Explicaciones SHAP almacenadas en fsai.explicaciones (global y local por hogar)."""
    try:
        _ds = dataset_id_activo()
        modelos = consultar(SQL_MODELOS, {"dataset": _ds if _ds is not None else -1})
    except ErrorConsulta as exc:
        mostrar_error_bd(exc)
        return
    except Exception:  # noqa: BLE001
        st.error("El esquema «fsai» de la base de datos no está disponible. Inicie PostgreSQL con iniciar.ps1 (o cree el esquema con sql/01_ddl_fsai.sql) y recargue la página.")
        return

    if modelos.empty:
        tema.vacio("Ninguna corrida guardó explicaciones SHAP en la base de datos para este dataset (se guardan al entrenar con explicabilidad habilitada).",
                   "Entrene con explicabilidad en «Entrenamiento», o abra la otra pestaña, que calcula SHAP con los modelos ya guardados", "pages/06_Entrenamiento.py",
                   titulo="Aún no hay explicaciones almacenadas")
        return

    etiquetas = {
        int(r.modelo_id): f"#{int(r.modelo_id)} · {r.codigo} · {r.nombre} · {r.escenario} · {r.esquema_validacion}"
        for r in modelos.itertuples()
    }
    modelo_id = int(st.selectbox("Modelo", options=list(etiquetas), format_func=etiquetas.get))
    tab_global, tab_local = st.tabs(["Global", "Local por hogar"])

    # ------------------------------------------------------------------ global
    with tab_global:
        try:
            glob = consultar(SQL_GLOBAL, {"modelo_id": modelo_id})
        except ErrorConsulta as exc:
            mostrar_error_bd(exc)
            glob = None
        if glob is None or glob.empty:
            tema.vacio("Este modelo no tiene guardada su importancia SHAP global en la base de datos.",
                       "Abra la pestaña «Modelos de la CV anidada»: allí se calcula con el modelo guardado", titulo="Falta la explicación global de este modelo")
        else:
            contenido = leer_json(glob.iloc[0]["contribuciones"])
            importancia = contenido.get("importancia") or {}
            por_clase = contenido.get("por_clase") or {}
            st.write(f"Observaciones usadas para la explicación (muestreo): {contenido.get('n_muestra', 'no registrado')}")
            col1, col2 = st.columns(2)
            top_n = col1.slider("Variables a mostrar", 5, 30, 15)
            clases = ["Todas (promedio)", *sorted(por_clase)] if por_clase else ["Todas (promedio)"]
            clase = col2.selectbox("Clase FIES", options=clases)
            datos_imp = por_clase.get(clase, {}) if clase != "Todas (promedio)" else importancia
            if not datos_imp:
                tema.vacio(f"El modelo no guardó importancias para «{clase}».", "Elija «Todas (promedio)» u otra clase FIES en el selector de arriba", titulo="Sin importancias para esta clase")
            else:
                st.plotly_chart(grafico_importancia(datos_imp, top_n, "Importancia global SHAP"))
            bee = contenido.get("beeswarm")
            if bee:
                max_puntos = st.slider("Puntos máximos del beeswarm (muestreo)", 200, 5000, 1500, step=100)
                figura = grafico_beeswarm(bee, top_n, max_puntos)
                if figura is None:
                    st.warning("El bloque 'beeswarm' almacenado no tiene la estructura esperada.")
                else:
                    st.plotly_chart(figura)
                    st.caption(
                        "Beeswarm aproximado con jitter vertical aleatorio (semilla 42); el color "
                        "indica el valor normalizado de la variable (azul bajo, rojo alto)."
                    )
            else:
                _beeswarm_faltante(modelos.set_index("modelo_id").loc[modelo_id])

    # ------------------------------------------------------------------ local
    with tab_local:
        try:
            candidatos = consultar(SQL_HOGARES_LOCAL, {"modelo_id": modelo_id})
        except ErrorConsulta as exc:
            mostrar_error_bd(exc)
            candidatos = None
        lista = [] if candidatos is None else [str(u) for u in candidatos["hogar_uid"]]
        if not lista:
            tema.vacio("Las explicaciones por hogar solo se guardan si se pidieron al entrenar; este modelo no tiene ninguna.",
                       "Para explicar un hogar ahora, use «SHAP local y LIME» en la pestaña «Modelos de la CV anidada»", titulo="Sin explicaciones locales guardadas")
        else:
            elegido = st.selectbox("Hogares con explicación local (más recientes)", options=lista,
                                   format_func=lambda u: f"{u[:12]}…")
            manual = st.text_input("O ingrese un hogar_uid (64 caracteres hexadecimales)", max_chars=64).strip().lower()
            if manual and not PATRON_UID.fullmatch(manual):
                st.warning("El hogar_uid debe tener 64 caracteres hexadecimales.")
                manual = ""
            hogar_uid = manual or elegido
            try:
                local = consultar(SQL_LOCAL, {"modelo_id": modelo_id, "hogar_uid": hogar_uid})
                contexto = consultar(SQL_CONTEXTO, {"hogar_uid": hogar_uid})
            except ErrorConsulta as exc:
                mostrar_error_bd(exc)
                local, contexto = None, None
            if local is None or local.empty:
                tema.vacio("No hay una explicación guardada para ese hogar con este modelo.", "Elija otro hogar de la lista, o calcule la explicación en «SHAP local y LIME» (otra pestaña)",
                           titulo="Ese hogar no tiene explicación guardada")
            else:
                registrar_auditoria("consulta_shap_local", "hogar", hogar_uid, {"modelo_id": modelo_id})
                datos = leer_json(local.iloc[0]["contribuciones"])
                contrib = {str(k): float(v) for k, v in (datos.get("contribuciones") or {}).items()}
                valores = datos.get("valores") or {}
                if not contrib:
                    tema.vacio("El registro existe pero no trae contribuciones por variable (se guardó incompleto).", "Calcule la explicación de este hogar en «SHAP local y LIME» (otra pestaña)",
                               titulo="Explicación guardada incompleta")
                else:
                    if contexto is not None and not contexto.empty:
                        ctx = contexto.iloc[0]
                        st.write(
                            f"Departamento: {ctx['departamento']} · Área: {ctx['area']} · "
                            f"Región natural: {ctx['region_natural']} · Nivel FIES observado: {ctx['fies_nivel']}"
                        )
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Clase explicada", str(datos.get("clase_explicada", "—")))
                    m2.metric("Valor base", fmt_num(datos.get("base_value")))
                    m3.metric("Salida del modelo", fmt_num(datos.get("prediccion")))
                    top_k = st.slider("Variables a mostrar (local)", 5, 25, 12)
                    pares = sorted(contrib.items(), key=lambda p: abs(p[1]), reverse=True)[:top_k][::-1]
                    etiquetas_y = [f"{k} = {valores.get(k, '—')}" for k, _ in pares]
                    fig = go.Figure(
                        go.Bar(
                            x=[v for _, v in pares], y=etiquetas_y, orientation="h",
                            marker_color=["#d95f02" if v > 0 else "#1b6ca8" for _, v in pares],
                        )
                    )
                    fig.update_layout(
                        xaxis_title="Contribución SHAP (naranja: aumenta la salida; azul: la reduce)",
                        height=max(320, 28 * len(pares)), margin={"l": 0, "r": 10, "t": 10, "b": 0},
                    )
                    st.plotly_chart(fig)
                    st.caption(
                        "La explicación local describe el comportamiento del modelo para este hogar; "
                        "no debe usarse para decidir su acceso a programas sociales."
                    )



def seccion_modelos() -> None:
    """Explicabilidad de los modelos entrenados con la CV anidada (Validación/Prueba ya evaluada)."""
    df, info = st.session_state["_df_activo"]
    corridas = EV.listar_corridas()
    if not corridas:
        tema.vacio("No hay corridas de entrenamiento completadas.", "Entrene modelos con validación cruzada anidada", "pages/06_Entrenamiento.py")
        return
    c1, c2, c3 = st.columns(3)
    nombre = c1.selectbox("Corrida", corridas)
    dir_run = EV.dir_corrida(nombre)
    resultados = EV.resultados_corrida(nombre)
    ident = c2.selectbox("Modelo / escenario", list(resultados), format_func=lambda k: f"{k.split('_')[1]} · {k.split('_')[0]}")
    res = resultados[ident]
    codigo, escenario = res["codigo"], res["escenario"]
    prueba_ok = EV.registro_prueba(dir_run, ident) is not None
    fuente = c3.selectbox("Hogares a explicar", ["Validación (15 %)"] + (["Prueba (ya evaluada)"] if prueba_ok else []))
    if escenario != "socio":
        st.warning(AVISO_CIRCULARIDAD)

    n_max = 60 if codigo == "M4" else 300
    n_muestra = st.slider("Hogares en la submuestra (rendimiento)", 20, n_max, min(120, n_max), step=10,
                          help="SHAP se calcula sobre una submuestra reproducible (SEED=42). M4 usa KernelExplainer (lento): máximo 60.")
    clase = st.selectbox("Clase a explicar", C.CLASES, index=3, format_func=lambda k: f"Nivel {k}" + (" (IA severa)" if k == 3 else ""))

    try:
        part, _ = cargar_particion(df, dir_run / "particion.json")
    except (ValueError, OSError) as exc:
        st.error(f"El dataset activo no coincide con la corrida ({exc}). Seleccione el mismo dataset de Entrenamiento.")
        return


    @st.cache_resource(show_spinner="Cargando el modelo final...")
    def _modelo(ruta: str, mtime: float):
        return joblib.load(ruta)


    pipe = _modelo(str(dir_run / res["final"]["modelo"]), (dir_run / res["final"]["modelo"]).stat().st_mtime)
    idx_part = part["validacion"] if fuente.startswith("Validación") else part["prueba"]
    dsel = df.iloc[idx_part].reset_index(drop=True)
    Xs = preparar_matriz(dsel, res["features_num"], res["features_cat"])
    ys, ws = dsel[C.COL_OBJETIVO].to_numpy(), dsel[C.COL_PESO].to_numpy()
    dev = df.iloc[part["desarrollo"]].reset_index(drop=True)
    Xdev = preparar_matriz(dev, res["features_num"], res["features_cat"])


    @st.cache_data(show_spinner=False, max_entries=6)
    def _shap(ruta_modelo: str, fuente_: str, n: int, ident_: str):
        t0 = time.perf_counter()
        r = X.shap_matriz(pipe, Xs, codigo, n, estrato=ys)        # submuestra con reparto parejo entre niveles FIES
        r["segundos"] = time.perf_counter() - t0
        return r


    if codigo == "M4":
        st.info("M4 usa KernelSHAP, que es lento (puede tardar varios minutos aun con el tope de 60 hogares).")
    try:
        with st.status(f"Calculando SHAP de {ident} sobre {n_muestra} hogares de {fuente}...", expanded=False) as estado:
            sm = _shap(str(dir_run), fuente, n_muestra, ident)
            estado.update(label=f"SHAP de {ident}: {len(sm['indice'])} hogares (calculado en {_seg(sm['segundos'])})", state="complete")
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo calcular SHAP para {codigo}: {type(exc).__name__}: {exc}")
        return
    audit("shap_modelo_anidado", "modelo", f"{nombre}/{ident}", {"n_muestra": n_muestra, "fuente": fuente})
    agr = sm["agregado_crudo"]                       # clases x muestras x columnas crudas
    cols = sm["columnas_crudas"]
    imp_global = pd.DataFrame({"variable": cols, "importancia": agr.mean(axis=(0, 1))})
    imp_clase = pd.DataFrame({"variable": cols, "importancia": agr[C.CLASES.index(clase)].mean(axis=0)})

    t1, tb, t2, t3, t4, t5 = st.tabs(["SHAP global", "Beeswarm por hogar", "Mapas de calor |SHAP|", "Máscaras de variables", "SHAP local y LIME", "Permutación, PDP y ALE"])

    with t1:
        st.markdown("**Importancia global: media de |SHAP| por variable (suma de sus dummies)**")
        ver_clase = st.toggle(f"Mostrar solo la clase {clase}", value=False)
        imp = imp_clase if ver_clase else imp_global
        st.plotly_chart(P.fig_importancia(imp, top=20), key=f"shap_g_{ident}_{ver_clase}_{clase}")
        interpretacion(I.i_importancia(imp))
        st.dataframe(E.presentar_tabla(imp.sort_values("importancia", ascending=False).round(5)), hide_index=True)

    with tb:
        st.markdown(f"**Beeswarm: cómo empuja cada variable la predicción del nivel {clase}, hogar por hogar**")
        top_b = st.slider("Variables a mostrar", 5, min(25, len(cols)), min(15, len(cols)), key="bs_top")
        bee = X.beeswarm_datos(sm["vals"][C.CLASES.index(clase)], Xs.loc[sm["indice"]], sm["nombres"], clase=clase, split=fuente, top=len(cols), max_filas=len(sm["indice"]))
        niv = dsel.loc[sm["indice"], C.COL_OBJETIVO].value_counts().sort_index()
        mostrar_beeswarm({"bee": bee, "segundos": sm["segundos"], "n": len(sm["indice"]), "por_nivel": {int(k): int(v) for k, v in niv.items()}}, clase, top_b, fuente, codigo)

    with t2:
        st.markdown("**Mapa de calor de |SHAP| (el análogo tabular de un «mapa de atención»)**")
        modo = st.radio("Vista", ["Variables × hogares (ordenados por influencia total)", "Hogares agrupados por grupo", "Promedio por grupo (variables × grupos)"], horizontal=False)
        top_v = st.slider("Variables mostradas", 5, min(40, len(cols)), min(20, len(cols)))
        orden_v = imp_clase.sort_values("importancia", ascending=False).head(top_v)["variable"].tolist()
        iv = [cols.index(v) for v in orden_v]
        A = agr[C.CLASES.index(clase)]                 # muestras x variables
        grupo_col = None
        if modo != "Variables × hogares (ordenados por influencia total)":
            grupo_col = st.selectbox("Grupo", [g for g in ["area", "region_natural", "sexo_jefe", "grupo_edad_jefe"] if g in dsel.columns], key="grp_hm")
            gl = dsel.loc[sm["indice"], grupo_col].astype(str).to_numpy()
        if modo.startswith("Variables ×"):
            orden_h = np.argsort(-A[:, iv].sum(axis=1))
            st.plotly_chart(P.fig_heatmap(A[orden_h][:, iv].T, orden_v, f"|SHAP| clase {clase}"), key=f"hm1_{ident}_{clase}_{top_v}")
            interpretacion(I.i_heatmap(A[orden_h][:, iv].T))
        elif modo.startswith("Hogares agrupados"):
            orden_h = np.lexsort((-A[:, iv].sum(axis=1), gl))
            st.plotly_chart(P.fig_heatmap(A[orden_h][:, iv].T, orden_v, f"|SHAP| clase {clase}, hogares agrupados por {grupo_col} ({', '.join(sorted(set(gl)))})", "Hogares (agrupados)"), key=f"hm2_{ident}_{clase}_{grupo_col}")
            interpretacion(I.i_heatmap(A[orden_h][:, iv].T))
        else:
            prom = pd.DataFrame(A[:, iv], columns=orden_v).groupby(gl).mean().T
            f = P.go.Figure(P.go.Heatmap(z=prom.values, x=prom.columns.astype(str), y=prom.index, colorscale="YlOrRd", colorbar={"title": "|SHAP| medio"}))
            f.update_layout(height=max(360, 20 * len(prom) + 120), yaxis_autorange="reversed", margin={"t": 20})
            st.plotly_chart(f, key=f"hm3_{ident}_{clase}_{grupo_col}")
            dif = (prom.max(axis=1) - prom.min(axis=1)).sort_values(ascending=False)
            interpretacion({"que_muestra": f"|SHAP| medio por variable para los grupos de «{grupo_col}» ({', '.join(prom.columns.astype(str))}); la variable cuya influencia más difiere entre grupos es «{dif.index[0]}» (diferencia {dif.iloc[0]:.4f}).",
                            "como_leerlo": "Compare columnas: una variable más intensa en un grupo pesa más en las predicciones de ese grupo.",
                            "limitacion": f"Submuestra de {len(A)} hogares; grupos pequeños dan promedios inestables."})

    with t3:
        st.markdown("**Máscara de variables: top-k por hogar** (variables que más sostienen la predicción de la clase predicha)")
        k = st.slider("k (variables en la máscara)", 1, min(15, len(cols)), 5)
        pred_cls = sm["proba"].argmax(1)
        A_pred = np.stack([agr[C.CLASES.index(int(c)), i, :] for i, c in enumerate(pred_cls)])
        M = X.mascara_topk(A_pred, k)
        orden_v_m = np.argsort(-M.sum(axis=0))[:min(25, len(cols))]
        st.plotly_chart(P.fig_mascara(M[:, orden_v_m].T, [cols[i] for i in orden_v_m]), key=f"mask_{ident}_{k}")
        top_global = set(np.argsort(-A_pred.mean(axis=0))[:k])
        estab = float(np.mean([len(set(np.argsort(-A_pred[i])[:k]) & top_global) / k for i in range(len(A_pred))]))
        interpretacion(I.i_mascara(k, len(cols), estab))
        st.markdown("**Curvas de eliminación / inserción por permutación** (validan que el orden de importancia es fiel al modelo)")
        orden_cols = imp_global.sort_values("importancia", ascending=False)["variable"].tolist()
        max_k = st.slider("Variables a recorrer", 3, len(orden_cols), min(12, len(orden_cols)))
        if st.button("Calcular curvas de eliminación e inserción"):
            with st.spinner("Permutando variables..."):
                curva = X.curva_eliminacion_insercion(pipe, Xs, ys, orden_cols[:max_k], ws, n_muestra=min(400, len(Xs)))
            st.plotly_chart(P.fig_eliminacion(curva), key=f"del_{ident}_{max_k}")
            interpretacion(I.i_eliminacion(curva))
            st.dataframe(curva.round(4), hide_index=True)

    with t4:
        n_idx = len(sm["indice"])
        pos = st.number_input(f"Hogar de la submuestra (0-{n_idx - 1})", 0, n_idx - 1, 0)
        fila = Xs.loc[[sm["indice"][pos]]]
        uid = dsel.loc[sm["indice"][pos], C.COL_GRUPO]
        st.caption(f"hogar_uid `{str(uid)[:16]}…` · nivel FIES real {int(ys[sm['indice'][pos]])} · probabilidades del modelo: " + ", ".join(f"{k}: {v:.1%}" for k, v in zip(C.CLASES, sm["proba"][pos])))
        sl = X.shap_local(pipe, fila, codigo, Xdev, clase)
        st.plotly_chart(P.fig_contribuciones(sl["nombres"], np.array(sl["valores_shap"])), key=f"shl_{ident}_{pos}_{clase}")
        top = pd.DataFrame({"variable": sl["nombres"], "shap": sl["valores_shap"]}).reindex(pd.Series(sl["valores_shap"]).abs().sort_values(ascending=False).index).head(3)
        interpretacion({"que_muestra": f"Para la clase {clase}, las variables que más mueven la predicción de este hogar son {', '.join(f'«{E.etq(v)}» ({s:+.3f})' for v, s in zip(top['variable'], top['shap']))}; valor base {sl['valor_base']:.3f}.",
                        "como_leerlo": "Rojo (positivo) empuja la probabilidad de la clase hacia arriba; azul (negativo), hacia abajo. La suma de contribuciones más el valor base reconstruye la salida del modelo.",
                        "limitacion": "Explicación local de un solo hogar para un modelo concreto; no es un efecto causal ni generaliza a otros hogares."})
        if st.button("Calcular explicación LIME de este hogar"):
            try:
                with st.spinner("LIME..."):
                    lm = X.lime_local(pipe, Xdev.sample(n=min(500, len(Xdev)), random_state=C.SEED), fila, clase)
                tabla = pd.DataFrame(lm["pesos"], columns=["regla LIME", "peso"])
                st.dataframe(tabla, hide_index=True)
                st.plotly_chart(P.fig_contribuciones(tabla["regla LIME"].tolist(), tabla["peso"].to_numpy()), key=f"lime_{ident}_{pos}_{clase}")
                interpretacion({"que_muestra": f"LIME ajusta un modelo lineal local (R² local = {lm['r2_local']:.3f}); la regla más influyente es «{tabla.iloc[0]['regla LIME']}».",
                                "como_leerlo": "Cada peso indica cuánto aporta esa condición a la probabilidad de la clase en el vecindario del hogar.",
                                "limitacion": "LIME depende del muestreo del vecindario (semilla fija) y puede discrepar de SHAP; un R² local bajo indica una explicación poco fiable."})
            except Exception as exc:  # noqa: BLE001
                st.error(f"LIME no se pudo calcular: {type(exc).__name__}: {exc}")

    with t5:
        if st.button("Calcular importancia por permutación (F1-macro ponderado)"):
            with st.spinner("Permutando..."):
                imp_p = X.importancia_permutacion(pipe, Xs, ys, ws, n_repeticiones=5)
            st.session_state["f4_perm"] = imp_p
        imp_p = st.session_state.get("f4_perm")
        if imp_p is not None:
            st.plotly_chart(P.fig_importancia(imp_p, error="desv_est"), key="perm_f4")
            interpretacion(I.i_importancia(imp_p))
        num_cols = [c for c in res["features_num"] if Xs[c].nunique() > 3]
        if num_cols:
            v = st.selectbox("Variable numérica (PDP y ALE)", num_cols, format_func=lambda c: E.etq(c, True), help="El código original de la variable se conserva en el anexo del informe y en la ayuda de cada campo.")
            try:
                pdp = X.dependencia_parcial(pipe, Xs, v, clase, pesos=ws)
                ale = X.ale_1d(pipe, Xs, v, clase)
                cA, cB = st.columns(2)
                cA.plotly_chart(P.fig_linea_xy(pdp["rejilla"], pdp["prob_media"], E.etq(v, True), f"P(nivel {clase}) media", "PDP"), key=f"pdp_{v}_{clase}")
                cB.plotly_chart(P.fig_linea_xy(ale["bordes"], ale["ale"], E.etq(v, True), "Efecto acumulado centrado", "ALE"), key=f"ale_{v}_{clase}")
                sube = pdp["prob_media"][-1] > pdp["prob_media"][0]
                interpretacion({"que_muestra": f"Al pasar «{E.etq(v)}» de {pdp['rejilla'][0]:.3g} a {pdp['rejilla'][-1]:.3g}, la probabilidad media del nivel {clase} {'sube' if sube else 'baja'} de {pdp['prob_media'][0]:.3f} a {pdp['prob_media'][-1]:.3f} (PDP).",
                                "como_leerlo": "PDP promedia sobre las demás variables; ALE corrige el sesgo cuando hay variables correlacionadas, por lo que es preferible si ambas difieren.",
                                "limitacion": "Curvas marginales: ocultan interacciones y extrapolan combinaciones poco realistas (PDP). No son efectos causales."})
            except Exception as exc:  # noqa: BLE001
                st.warning(f"No se pudo calcular PDP/ALE para «{v}»: {exc}")



st.session_state["_df_activo"] = selector_dataset()      # un solo selector de dataset para ambas pestañas
t_alm, t_mod = st.tabs(["Explicaciones almacenadas (PostgreSQL)", "Modelos de la CV anidada: SHAP, LIME, mapas de calor y máscaras"])
with t_alm:
    seccion_almacenada()
with t_mod:
    seccion_modelos()
