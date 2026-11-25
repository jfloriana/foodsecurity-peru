"""Comparación estadística: métricas con IC, matrices de confusión, ROC/PR, pruebas estadísticas y calibración.

La Prueba (15 %) se evalúa UNA sola vez por (corrida, escenario, modelo): el candado es un archivo creado con modo
exclusivo (fsai.evaluacion). Todo lo que se muestra después sale de las predicciones guardadas en esa única evaluación.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

import tema
from auth import ROLES_ESCRITURA, barra_lateral, requerir_acceso
from tema import config_pagina, encabezado
from comun import fmt_num
from fases_comun import AVISO_CIRCULARIDAD, C, MODELOS, audit, interpretacion, selector_dataset

from fsai import evaluacion as EV  # noqa: E402
from fsai import interpretacion as I  # noqa: E402
from fsai import plots as P  # noqa: E402
from fsai.metrics_weighted import predecir_con_umbral  # noqa: E402
from fsai.nested_cv import aplicar_calibrador  # noqa: E402

config_pagina("Comparación estadística")
usuario, rol = requerir_acceso("comparacion")
barra_lateral()
encabezado("Comparación estadística",
    "Evalúe la Prueba (15 %) una sola vez, compare curvas y modelos con pruebas estadísticas y revise la calibración.",
    "Evaluación e interpretación",
    pasos=["Elija una corrida de entrenamiento y evalúe cada modelo en la Prueba (solo se permite una vez por modelo).",
           "Revise métricas con IC, matrices de confusión, curvas ROC/PR y calibración.",
           "Compare modelos con DeLong, McNemar, Wilcoxon y Friedman/Nemenyi (p ajustados por Holm)."],
    produce="Tablas y figuras de desempeño final con su interpretación; las predicciones de la Prueba quedan guardadas.",
)
df, info = selector_dataset()

corridas = EV.listar_corridas()
if not corridas:
    tema.vacio("No hay corridas de entrenamiento completadas.", "Entrene modelos con validación cruzada anidada", "pages/06_Entrenamiento.py")
    st.stop()
nombre = st.selectbox("Corrida de Entrenamiento", corridas)
dir_run = EV.dir_corrida(nombre)
cfg = EV.config_corrida(nombre)
resultados = EV.resultados_corrida(nombre)
escenarios = sorted({r["escenario"] for r in resultados.values()})
escenario = st.radio("Escenario", escenarios, horizontal=True)
if escenario != "socio":
    st.warning(AVISO_CIRCULARIDAD)
codigos = sorted(r["codigo"] for r in resultados.values() if r["escenario"] == escenario)

# ------------------------------------------------------------------ estado y evaluación única
st.subheader("A · Registro de la evaluación única")
filas = []
for c in codigos:
    reg = EV.registro_prueba(dir_run, f"{escenario}_{c}")
    filas.append({"modelo": c, "estado": "EVALUADO (bloqueado)" if reg else "pendiente", "evaluado_en": reg["evaluado_en"] if reg else "—",
                  "usuario": reg["usuario"] if reg else "—", "n_prueba": reg["n_prueba"] if reg else "—"})
st.dataframe(pd.DataFrame(filas), hide_index=True)
pendientes = [f["modelo"] for f in filas if f["estado"] == "pendiente"]
if pendientes:
    st.warning("**Aviso:** la Prueba solo puede evaluarse **una vez** por modelo. Tras pulsar el botón no habrá forma de repetirla en esta corrida "
               "(si ajusta el modelo después de ver la Prueba, el resultado deja de ser una estimación honesta).")
    sel = st.multiselect("Modelos a evaluar ahora", pendientes, default=pendientes)
    conf = st.checkbox("Confirmo que el modelo final y el umbral de Validación están definitivos y deseo consumir la Prueba.")
    if st.button("Evaluar la Prueba (una sola vez)", disabled=rol not in ROLES_ESCRITURA or not conf or not sel):
        for c in sel:
            try:
                reg = EV.evaluar_prueba_una_vez(df, dir_run, f"{escenario}_{c}", usuario)
                audit("comparacion_prueba_evaluada", "modelo", f"{nombre}/{escenario}_{c}", {"n_prueba": reg["n_prueba"], "sha256_modelo": reg["sha256_modelo"][:16]})
                st.success(f"{c}: Prueba evaluada ({reg['n_prueba']} hogares).")
            except EV.PruebaYaEvaluada as exc:
                st.error(f"{c}: bloqueado. {exc}")
            except (ValueError, OSError) as exc:
                st.error(f"{c}: no se pudo evaluar ({exc}). Verifique que el dataset activo coincide con el de la corrida.")
        st.rerun()
    if rol not in ROLES_ESCRITURA:
        st.info("Su rol (lector) puede consultar resultados pero no consumir la Prueba.")
evaluados = [c for c in codigos if EV.registro_prueba(dir_run, f"{escenario}_{c}")]
if not evaluados:
    tema.vacio("La comparación estadística usa las predicciones de la Prueba, y la Prueba solo se evalúa una vez por modelo; todavía no se evaluó ninguno de este escenario.",
               "Marque los modelos y pulse «Evaluar la Prueba (una sola vez)» en la sección anterior (requiere rol con permiso de escritura)", titulo="Aún no hay modelos evaluados en la Prueba")
    st.stop()
preds = {c: EV.cargar_prediccion_prueba(dir_run, f"{escenario}_{c}") for c in evaluados}
res_c = {c: resultados[f"{escenario}_{c}"] for c in evaluados}

# ------------------------------------------------------------------ B · métricas con IC
st.subheader("B · Métricas con intervalo de confianza bootstrap")
n_boot = st.slider("Réplicas bootstrap", 100, 1000, 200, step=100, help="Contrato: N_BOOT=1000; use 200 para explorar.")
pond = st.toggle("Ponderadas con factor de expansión", value=True, key="pond_f3")


@st.cache_data(show_spinner="Calculando IC bootstrap...", max_entries=16)
def _mets(nombre_run: str, ident: str, nb: int) -> pd.DataFrame:
    p = EV.cargar_prediccion_prueba(EV.dir_corrida(nombre_run), ident)
    return EV.metricas_con_ic(p["y"], p["proba"], p["w"], nb)


tablas, wide = {}, []
for c in evaluados:
    m = _mets(nombre, f"{escenario}_{c}", n_boot)
    tablas[c] = m
    sub = m[m["ponderada"] == pond].set_index("metrica")
    wide.append({"modelo": c, **{k: f"{r.valor:.3f} [{r.ic_inf:.3f}; {r.ic_sup:.3f}]" for k, r in sub.iterrows()}})
st.dataframe(pd.DataFrame(wide).set_index("modelo"))
f1s = pd.DataFrame([{"modelo": c, "f1_macro": float(tablas[c].query("metrica=='f1_macro' and ponderada==@pond")["valor"].iloc[0])} for c in evaluados])
interpretacion(I.i_metricas(f1s))

# ------------------------------------------------------------------ C · confusión
st.subheader("C · Matrices de confusión")
mod = st.selectbox("Modelo", evaluados, key="mod_f3")
p = preds[mod]
umbral = res_c[mod]["final"]["validacion"]["umbral_severa"]
regla = st.radio("Regla de decisión", ["argmax", "umbral operativo de Validación para la clase severa"], horizontal=True)
pred = p["proba"].argmax(1) if regla == "argmax" else predecir_con_umbral(p["proba"], umbral)
ca, cb = st.columns(2)
cm = EV.matriz_confusion(p["y"], pred)
ca.markdown("Conteos")
ca.plotly_chart(P.fig_confusion(cm), key=f"cm_{mod}_{regla}")
cb.markdown("Normalizada por fila (sensibilidad por clase)")
cb.plotly_chart(P.fig_confusion(EV.matriz_confusion(p["y"], pred, normalizar=True), True), key=f"cmn_{mod}_{regla}")
interpretacion(I.i_confusion(cm))
fs = EV.foco_severa(p["y"], p["proba"], p["w"], umbral)
st.markdown("**Foco en IA severa (ponderado)**")
st.dataframe(pd.DataFrame([{"regla": k.split("_", 1)[1], "medida": k.split("_")[0], "valor": round(v, 4)} for k, v in fs.items()]).pivot(index="regla", columns="medida", values="valor"))

# ------------------------------------------------------------------ D · ROC / PR
st.subheader("D · Curvas ROC y PR (uno-contra-resto)")
cv = EV.curvas_roc_pr(p["y"], p["proba"])
cc, cd = st.columns(2)
cc.plotly_chart(P.fig_roc(cv), key=f"roc_{mod}")
cd.plotly_chart(P.fig_pr(cv), key=f"pr_{mod}")
interpretacion(I.i_roc_pr(cv["auc"], cv["ap"], cv["prevalencia"]))

# ------------------------------------------------------------------ E · calibración
st.subheader("E · Calibración")
clase_cal = st.selectbox("Clase", C.CLASES, index=3, format_func=lambda k: f"Nivel {k}")
curvas = {"Sin calibrar": EV.curva_calibracion(p["y"], p["proba"], clase_cal)}
cal = res_c[mod]["final"]["calibrador"]
curvas["Calibrada (isotónica, ajustada en Validación)"] = EV.curva_calibracion(p["y"], aplicar_calibrador(cal, p["proba"]), clase_cal)
st.plotly_chart(P.fig_calibracion(curvas), key=f"cal_{mod}_{clase_cal}")
cr = EV.calibracion_resumen(p["y"], p["proba"], p["w"], cal)
st.dataframe(pd.DataFrame([cr]).rename(columns={"brier": "Brier", "ece": "ECE", "brier_calibrado": "Brier calibrado", "ece_calibrado": "ECE calibrado"}).round(4), hide_index=True)
interpretacion(I.i_calibracion(cr["brier"], cr["ece"], cr["brier_calibrado"], cr["ece_calibrado"]))

# ------------------------------------------------------------------ F · pruebas estadísticas
st.subheader("F · Pruebas estadísticas entre modelos")
if len(codigos) < 2:
    st.info("Se necesitan al menos 2 modelos en la corrida para comparar.")
else:
    pr = EV.pruebas_entre_modelos(dir_run, escenario, codigos)
    st.dataframe(pr.round(5), hide_index=True)
    interpretacion(I.i_pruebas(pr))
    st.caption("DeLong y McNemar usan las predicciones de la Prueba de los modelos ya evaluados; Wilcoxon y Friedman+Nemenyi usan el F1 de los 5 pliegues "
               "externos de la CV anidada (Desarrollo). Holm por familia de pruebas.")
