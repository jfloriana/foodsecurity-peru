"""Optimización Optuna: configuración de la búsqueda, monitoreo de ensayos en vivo y visualización de resultados.

La búsqueda corre en un subproceso (train_pipeline.py --solo-optuna) sobre el 70 % de Desarrollo, con TPE y pruner de la mediana.
El estudio se persiste en SQLite (artifacts/optuna/estudios.db) y es reanudable: PostgreSQL no es viable con el rol de la app
(fsai_app no tiene CREATE en el esquema public, donde Optuna crea sus tablas).
"""
from __future__ import annotations

import json
import time

import numpy as np
import optuna
import pandas as pd
import streamlit as st

from auth import barra_lateral, requerir_acceso
from tema import config_pagina, encabezado
from comun import consultar
from fases_comun import (AVISO_CIRCULARIDAD, C, ESCENARIOS, MODELOS, PATRON_NOMBRE, audit, cola_texto, en_ejecucion,
                         interpretacion, lanzar_pipeline, leer_json, ruta_para_pipeline, selector_dataset)

from fsai import interpretacion as I  # noqa: E402
from fsai import plots as P  # noqa: E402
from fsai.optuna_optimizer import ESPACIO_DEFECTO, PRUNERS, SAMPLERS  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
config_pagina("Optuna")
usuario, rol = requerir_acceso("optuna")
barra_lateral()
encabezado("Optimización de hiperparámetros",
    "Búsqueda automática de hiperparámetros con seguimiento de ensayos sobre el 70 % de Desarrollo.",
    "Modelado",
    pasos=["Elija el modelo, el escenario y el número de ensayos (opcionalmente restrinja el espacio de búsqueda).",
           "Lance el estudio y siga los ensayos en vivo; puede reanudarlo después.",
           "Revise la curva de optimización, la importancia de hiperparámetros y el mejor conjunto."],
    produce="Un estudio de optimización guardado y reanudable (paso opcional: el entrenamiento ya optimiza sus propios hiperparámetros).",
)
df, info = selector_dataset()

DB = C.DIR_OPTUNA / "estudios.db"
URL = "sqlite:///" + DB.as_posix()


def estudios_existentes() -> list[str]:
    if not DB.is_file():
        return []
    try:
        return sorted(s.study_name for s in optuna.get_all_study_summaries(storage=URL))
    except Exception:  # noqa: BLE001
        return []


try:
    pg_ok = bool(consultar("SELECT has_schema_privilege(current_user, 'public', 'CREATE') AS ok", cache=False).iloc[0]["ok"])
except Exception:  # noqa: BLE001
    pg_ok = False
st.caption("Persistencia: **SQLite local** `artifacts/optuna/estudios.db` " + ("(PostgreSQL también sería viable con este rol)." if pg_ok else
           "(PostgreSQL no es viable: el rol de la app no puede crear las tablas de Optuna en `public`)."))

# ------------------------------------------------------------------ A · configuración
st.subheader("A · Configuración de la búsqueda")
existentes = estudios_existentes()
c1, c2, c3 = st.columns(3)
modelo = c1.selectbox("Modelo", list(MODELOS), format_func=MODELOS.get)
escenario = c2.selectbox("Escenario", list(ESCENARIOS), format_func=ESCENARIOS.get)
reanudar = c3.selectbox("Reanudar estudio existente", ["(nuevo)"] + existentes)
if escenario != "socio":
    st.warning(AVISO_CIRCULARIDAD)
st.markdown("**Espacio de búsqueda** (editable solo dentro de los rangos válidos; cualquier valor fuera se recorta)")
espacio: dict[str, dict] = {}
cols_e = st.columns(3)
for i, (p, s) in enumerate(ESPACIO_DEFECTO[modelo].items()):
    w = cols_e[i % 3]
    if s["tipo"] == "cat":
        sel = w.multiselect(p, s["opciones"], default=s["opciones"], key=f"esp_{modelo}_{p}")
        if sel and sel != s["opciones"]:
            espacio[p] = {"opciones": sel}
    else:
        lo = w.number_input(f"{p} · mín. (≥ {s['lo']})", value=float(s["lo"]), min_value=float(s["lo"]), max_value=float(s["hi"]), key=f"lo_{modelo}_{p}", format="%.5g")
        hi = w.number_input(f"{p} · máx. (≤ {s['hi']})", value=float(s["hi"]), min_value=float(s["lo"]), max_value=float(s["hi"]), key=f"hi_{modelo}_{p}", format="%.5g")
        if lo > hi:
            w.error(f"{p}: el mínimo no puede superar al máximo.")
        elif (lo, hi) != (float(s["lo"]), float(s["hi"])):
            espacio[p] = {"lo": lo, "hi": hi}
with st.form("form_optuna"):
    d1, d2, d3, d4 = st.columns(4)
    n_trials = d1.number_input("n_trials", 1, 500, 20, step=5)
    timeout = d2.number_input("timeout (s; 0 = sin límite)", 0, 86400, 0, step=60)
    semilla = d3.number_input("Semilla", 0, 10**6, C.SEED)
    pruner = d4.selectbox("Pruner", list(PRUNERS), format_func=lambda x: {"mediana": "Mediana (MedianPruner)", "ninguno": "Ninguno"}[x])
    e1, e2 = st.columns(2)
    e1.selectbox("Sampler", list(SAMPLERS), format_func=lambda x: "TPE (fijado por el protocolo)", disabled=True)
    nombre = e2.text_input("Nombre del estudio", value=(reanudar if reanudar != "(nuevo)" else time.strftime("optuna_%Y%m%d_%H%M")), max_chars=60)
    conf = st.checkbox("Entiendo que la búsqueda usa solo Desarrollo (70 %); Validación y Prueba no se tocan.")
    lanzar = st.form_submit_button("Iniciar / reanudar búsqueda")

proceso = st.session_state.get("f7_proceso")
if lanzar:
    if proceso is not None and proceso.poll() is None:
        st.warning("Ya hay una búsqueda en ejecución en esta sesión.")
    elif not conf:
        st.warning("Debe confirmar la nota.")
    elif not PATRON_NOMBRE.fullmatch(nombre or ""):
        st.warning("Nombre inválido.")
    elif nombre in existentes and reanudar != nombre:
        st.warning("Ese nombre ya existe; elíjalo en «Reanudar estudio existente» o use otro.")
    else:
        try:
            C.DIR_OPTUNA.mkdir(parents=True, exist_ok=True)
            args = ["--ruta-datos", str(ruta_para_pipeline(df, info)), "--solo-optuna", "--estudio-nombre", nombre, "--modelos", modelo,
                    "--escenario", escenario, "--trials", str(int(n_trials)), "--semilla-optuna", str(int(semilla)), "--pruner", pruner, "--almacenamiento", "sqlite"]
            if timeout:
                args += ["--timeout", str(float(timeout))]
            if espacio:
                ruta_esp = C.DIR_OPTUNA / f"{nombre}.espacio.json"
                ruta_esp.write_text(json.dumps(espacio, ensure_ascii=False), encoding="utf-8")
                args += ["--espacio-json", str(ruta_esp)]
            proceso = lanzar_pipeline(args, C.DIR_OPTUNA / f"{nombre}.stdout.log")
        except (OSError, ValueError) as exc:
            st.error(f"No se pudo lanzar: {exc}")
        else:
            st.session_state.update(f7_proceso=proceso, f7_nombre=nombre)
            audit("optuna_lanzada", "estudio", nombre, {"modelo": modelo, "escenario": escenario, "n_trials": int(n_trials), "timeout": int(timeout),
                                                              "semilla": int(semilla), "pruner": pruner, "espacio_editado": sorted(espacio), "reanuda": reanudar != "(nuevo)"})
            st.success(f"Búsqueda «{nombre}» lanzada (PID {proceso.pid}).")


def cargar_estudio(nombre_e: str):
    return optuna.load_study(study_name=nombre_e, storage=URL)


def tabla_trials(estudio) -> pd.DataFrame:
    t = estudio.trials_dataframe(attrs=("number", "state", "value", "duration", "params"))
    t.columns = [c.replace("params_", "p_") for c in t.columns]
    return t


# ------------------------------------------------------------------ B · monitoreo
st.subheader("B · Monitoreo de ensayos en vivo")


def panel_monitoreo() -> None:
    nombre_m = st.session_state.get("f7_nombre")
    if not nombre_m:
        st.info("Inicie una búsqueda para monitorearla aquí. Los estudios anteriores se ven en la sección C.")
        return
    prog = leer_json(C.DIR_OPTUNA / nombre_m / "progreso.json")
    pr = st.session_state.get("f7_proceso")
    codigo = None if pr is None else pr.poll()
    st.write(f"Estudio **{nombre_m}** · estado: {prog.get('estado', 'iniciando')} · {prog.get('mensaje', '')}")
    if nombre_m in estudios_existentes():
        est = cargar_estudio(nombre_m)
        t = tabla_trials(est)
        completos = int((t["state"] == "COMPLETE").sum())
        a, b, c_, d = st.columns(4)
        a.metric("Ensayos", len(t))
        b.metric("Completos", completos)
        c_.metric("Podados", int((t["state"] == "PRUNED").sum()))
        d.metric("Mejor valor (F1-macro interno)", f"{est.best_value:.4f}" if completos else "—")
        total = int(prog.get("n_trials", max(len(t), 1)))
        st.progress(min(1.0, len(t) / max(total, 1)), text=f"{len(t)} de {total} ensayos")
        if completos:
            st.plotly_chart(optuna.visualization.plot_optimization_history(est), key=f"hist_live_{nombre_m}_{len(t)}")
            interpretacion(I.i_curva_optuna(t.sort_values("number")["value"].tolist()))
        st.dataframe(t.sort_values("number", ascending=False).head(15), hide_index=True)
    else:
        st.write("Esperando el primer ensayo (cargando datos y generando la partición)...")
    st.code(cola_texto(C.DIR_OPTUNA / nombre_m / "log.txt") or cola_texto(C.DIR_OPTUNA / f"{nombre_m}.stdout.log") or "(sin registro)", language="text")
    if codigo is None and pr is not None:
        if st.button("Cancelar búsqueda", key="cancel_f7"):
            pr.terminate()
            audit("optuna_cancelada", "estudio", nombre_m)
            st.rerun()
    elif codigo is not None and st.session_state.get("f7_audit") != nombre_m:
        st.session_state["f7_audit"] = nombre_m
        audit("optuna_finalizada", "estudio", nombre_m, {"codigo_salida": codigo})
        st.rerun()


p0 = st.session_state.get("f7_proceso")
if p0 is not None and p0.poll() is None:
    st.fragment(run_every="2s")(panel_monitoreo)()
else:
    panel_monitoreo()

# ------------------------------------------------------------------ C · resultados
st.subheader("C · Visualización de resultados")
existentes = estudios_existentes()
if not existentes:
    st.info("Aún no hay estudios guardados. Configure y lance una búsqueda arriba; el estudio quedará disponible aquí para reanudarlo.")
    st.stop()
sel = st.selectbox("Estudio", existentes, index=existentes.index(st.session_state["f7_nombre"]) if st.session_state.get("f7_nombre") in existentes else 0)
est = cargar_estudio(sel)
t = tabla_trials(est)
completos = t[t["state"] == "COMPLETE"]
cfg = leer_json(C.DIR_OPTUNA / sel / "config.json")
if cfg:
    st.caption(f"Modelo {cfg.get('modelo')} · escenario {cfg.get('escenario')} · sampler TPE · pruner {cfg.get('pruner')} · semilla {cfg.get('semilla')} · partición `{str(cfg.get('huella_particion', ''))[:12]}…`")
st.dataframe(t, hide_index=True)
if len(completos) < 2:
    st.info("Se necesitan al menos 2 ensayos completos para las gráficas de importancia, coordenadas paralelas, slice y contour.")
    st.stop()
imps: dict | None = None
try:
    imps = optuna.importance.get_param_importances(est)
except Exception:  # noqa: BLE001
    imps = None
interpretacion(I.i_optuna_resultados(t, imps))
g1, g2 = st.columns(2)
g1.plotly_chart(optuna.visualization.plot_optimization_history(est), key=f"oh_{sel}")
if imps:
    g2.plotly_chart(optuna.visualization.plot_param_importances(est), key=f"pi_{sel}")
else:
    g2.info("No se pudieron estimar importancias (pocos ensayos o parámetros constantes).")
st.markdown("**Coordenadas paralelas**")
st.plotly_chart(optuna.visualization.plot_parallel_coordinate(est), key=f"pc_{sel}")
parametros = [c[2:] for c in t.columns if c.startswith("p_")]
st.markdown("**Slice**")
st.plotly_chart(optuna.visualization.plot_slice(est), key=f"sl_{sel}")
if len(parametros) >= 2:
    st.markdown("**Contour**")
    cc1, cc2 = st.columns(2)
    px_ = cc1.selectbox("Hiperparámetro X", parametros, index=0)
    py_ = cc2.selectbox("Hiperparámetro Y", parametros, index=1)
    if px_ != py_:
        st.plotly_chart(optuna.visualization.plot_contour(est, params=[px_, py_]), key=f"ct_{sel}_{px_}_{py_}")
mejor = est.best_trial
st.success(f"Mejor ensayo #{mejor.number}: F1-macro interno {mejor.value:.4f} con {mejor.params}")
st.download_button("Descargar ensayos (CSV)", data=t.to_csv(index=False), file_name=f"{sel}_ensayos.csv", mime="text/csv")
