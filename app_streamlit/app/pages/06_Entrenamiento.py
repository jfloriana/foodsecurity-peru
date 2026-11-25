"""Módulo Entrenamiento: validación cruzada anidada 5x3 con partición 70/15/15 (recomendada) y comparación de esquemas V1-V5.

* La CV ANIDADA (5 externos x 3 internos) se corre SOLO dentro del 70 % de Desarrollo.
* El 15 % de Validación fija el umbral operativo de la clase severa y la calibración, sin tocar la Prueba.
* El 15 % de Prueba NO se usa hasta el módulo «Comparación estadística» (una sola evaluación, con candado).
El entrenamiento corre en un subproceso (train_pipeline.py) con argumentos validados y sin shell.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from auth import barra_lateral, requerir_acceso
from tema import config_pagina, encabezado
from comun import ErrorConsulta, consultar, fmt_num, mostrar_error_bd, registrar_auditoria
from fases_comun import (AVISO_CIRCULARIDAD, C, D, audit, cola_texto, en_ejecucion, formatear_ic, interpretacion,
                         leer_json, ruta_para_pipeline, selector_dataset)
from fases_comun import ESCENARIOS as ESCENARIOS_F
from fases_comun import MODELOS as MODELOS_F
from fases_comun import PATRON_NOMBRE as PATRON_NOMBRE_F
from fases_comun import lanzar_pipeline as lanzar_subproceso

from fsai import evaluacion as EV  # noqa: E402
from fsai import interpretacion as I  # noqa: E402
from fsai import nested_cv as N  # noqa: E402
from fsai import plots as P  # noqa: E402
from fsai.optuna_optimizer import ESPACIO_DEFECTO  # noqa: E402

config_pagina("Entrenamiento")
usuario, rol = requerir_acceso("entrenamiento")
barra_lateral()

encabezado("Entrenamiento de modelos",
    "Entrene los modelos M1-M5 con validación cruzada anidada o compare esquemas de validación (V1-V5) en PostgreSQL.",
    "Modelado",
    pasos=["Elija el método (pestañas): la validación cruzada anidada es la recomendada para resultados publicables.",
           "Seleccione modelos, escenario y número de ensayos de optimización, y lance el entrenamiento; el progreso se ve en vivo.",
           "Revise las métricas con intervalos de confianza; luego evalúe la Prueba en «Comparación estadística»."],
    produce="Corridas con métricas por pliegue, modelo final, umbral operativo y calibración (CV anidada) o métricas V1-V5 guardadas en PostgreSQL.",
)

MODELOS = {
    "M1": "M1 · Regresión logística ordinal (individual)",
    "M2": "M2 · Random Forest (individual)",
    "M3": "M3 · CatBoost (individual)",
    "M4": "M4 · Stacking CatBoost + Random Forest + Regresión logística (híbrido)",
    "M5": "M5 · Autoencoder + XGBoost (híbrido)",
}
ESCENARIOS = {
    "socio": "socio · solo variables socioeconómicas (escenario de predicción)",
    "fies": "fies · solo ítems FIES",
    "hibrido": "hibrido · socioeconómicas + FIES (escenario de reconstrucción/clasificación)",
}
ESQUEMAS = {
    "V1": "V1 · GroupKFold(5) por hogar, 4 repeticiones (20 bloques)",
    "V1c": "V1c · Sensibilidad de V1: grupos por conglomerado",
    "V2": "V2 · Validación geográfica (LODO y macrorregiones)",
    "V2_lodo": "V2_lodo · Dejar fuera un departamento",
    "V2_macro": "V2_macro · Dejar fuera una macrorregión (costa/sierra/selva)",
    "V3": "V3 · Partición estratificada urbano/rural (métricas por estrato)",
    "V4": "V4 · Temporal 2024→2025 (solo con ≥2 años comparables) [POR VALIDAR]",
    "V5": "V5 · Métricas ponderadas con factor de expansión (no es partición; se aplica sobre V1)",
}
ETAPAS = {"cv_externa": "validación externa", "modelo_final": "modelo final", "validacion": "umbral y calibración"}
N_TRIALS_DEFECTO = 100
N_TRIALS_MAX = 500
PATRON_NOMBRE = re.compile(r"^[A-Za-z0-9_.\-]{3,60}$")

RAIZ = Path(os.environ.get("FSAI_RAIZ_PROYECTO", Path(__file__).resolve().parents[2]))
SCRIPT = RAIZ / "train_pipeline.py"
DIR_LOGS = Path(os.environ.get("FSAI_ARTIFACTS_DIR", RAIZ / "artifacts")) / "logs"

SQL_EXPERIMENTOS = (
    "SELECT experimento_id, nombre, escenario, esquema_validacion, estado, iniciado_en "
    "FROM fsai.experimentos ORDER BY iniciado_en DESC LIMIT 100"
)
SQL_PROGRESO = (
    "SELECT e.experimento_id, e.estado, e.iniciado_en, e.finalizado_en, "
    "(SELECT COUNT(*) FROM fsai.modelos m WHERE m.experimento_id = e.experimento_id) AS n_modelos "
    "FROM fsai.experimentos e WHERE e.nombre LIKE :nombre || '%' ORDER BY e.iniciado_en DESC LIMIT 1"
)
SQL_METRICAS = (
    "SELECT mo.codigo AS modelo, me.nombre_metrica, "
    "CAST(me.valor AS double precision) AS valor, "
    "CAST(me.ic_inf AS double precision) AS ic_inf, CAST(me.ic_sup AS double precision) AS ic_sup "
    "FROM fsai.metricas me JOIN fsai.modelos mo ON mo.modelo_id = me.modelo_id "
    "WHERE mo.experimento_id = :experimento_id AND me.fold IS NULL AND me.estrato IS NULL "
    "AND me.ponderada = :ponderada ORDER BY mo.codigo, me.nombre_metrica"
)


def lanzar_pipeline(
    modelos: list[str], escenario: str, esquema: str, n_trials: int, ruta_datos: str, nombre: str
) -> tuple[subprocess.Popen, Path]:
    """Lanza train_pipeline.py como proceso hijo con argumentos validados (sin shell)."""
    if not set(modelos) <= set(MODELOS) or not modelos:
        raise ValueError("Selección de modelos inválida.")
    if escenario not in ESCENARIOS or esquema not in ESQUEMAS:
        raise ValueError("Escenario o esquema de validación inválido.")
    if not 1 <= n_trials <= N_TRIALS_MAX or not PATRON_NOMBRE.fullmatch(nombre):
        raise ValueError("n_trials o nombre de experimento inválido.")
    if not SCRIPT.is_file():
        raise FileNotFoundError(f"No se encontró train_pipeline.py en {RAIZ}.")
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    log = DIR_LOGS / f"{nombre}_{int(time.time())}.log"
    if not ruta_datos:
        raise ValueError("No hay un archivo analítico de hogares para entrenar: elija un dataset en la barra lateral.")
    comando = [
        sys.executable, str(SCRIPT),
        "--ruta-datos", ruta_datos,
        "--modelos", *modelos,
        "--escenario", escenario,
        "--esquemas", esquema,
        "--trials", str(n_trials),
        "--nombre-experimento", nombre,
    ]
    with log.open("wb") as salida:
        proceso = subprocess.Popen(  # noqa: S603 - lista de argumentos validada, sin shell
            comando, stdout=salida, stderr=subprocess.STDOUT, cwd=str(RAIZ), shell=False
        )
    return proceso, log


def cola_log(ruta: Path, max_bytes: int = 4000) -> str:
    try:
        with ruta.open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


JUSTIFICACION = {
    "C": "Inverso de la regularización; escala log porque actúa multiplicativamente (1e-3 = muy regularizado, 1e2 = casi sin regularizar).",
    "penalty": "l2 es estable con variables correlacionadas; l1 puede anular coeficientes (selección de variables).",
    "class_weight": "«balanced» compensa el desbalance de la clase severa; «ninguno» conserva las frecuencias originales.",
    "n_estimators": "Más árboles reducen la varianza con rendimientos decrecientes; el rango equilibra costo y estabilidad.",
    "max_depth": "Controla la complejidad: poco profundo subajusta, muy profundo sobreajusta.",
    "min_samples_leaf": "Hojas mínimas más grandes suavizan el árbol; escala log porque el efecto es relativo.",
    "depth": "Profundidad de los árboles de CatBoost (simétricos); 4-10 cubre desde modelos simples a expresivos.",
    "learning_rate": "Tasa de aprendizaje en escala log: valores bajos requieren más iteraciones.",
    "iterations": "Número de árboles del boosting; interactúa con learning_rate.",
    "cb_depth": "Profundidad del CatBoost base del stacking (acotada por costo).", "cb_learning_rate": "Tasa de aprendizaje del CatBoost base.",
    "cb_iterations": "Árboles del CatBoost base (acotado por costo).", "rf_n_estimators": "Árboles del RF base.",
    "rf_max_depth": "Profundidad del RF base.", "rf_min_samples_leaf": "Hoja mínima del RF base.",
    "lr_C": "Regularización de la logística base.", "C_meta": "Regularización del metaaprendiz del stacking.",
    "capas_cfg": "Arquitectura del autoencoder (capas ocultas simétricas).", "dim_latente": "Tamaño del código latente z.",
    "dropout": "Regularización del autoencoder.", "lr_ae": "Tasa de aprendizaje del autoencoder (parada temprana interna sobre una fracción de train).",
    "modo_entrada": "XGBoost recibe [x||z] o solo el latente z.", "subsample": "Fracción de filas por árbol (reduce varianza).",
    "colsample_bytree": "Fracción de columnas por árbol.",
}



def seccion_clasica() -> None:
    """Entrenamiento clásico M1-M5 con los esquemas V1-V5 y métricas de fsai.metricas."""
    # ------------------------------------------------------------------ configuración
    st.subheader("Configuración del experimento")
    st.caption("Entrena los modelos con uno de los esquemas de validación del protocolo y guarda las métricas por pliegue y por estrato en PostgreSQL. "
               "Usa el dataset activo de la barra lateral y la misma semilla y particiones para todos los modelos.")
    df_ds, info_ds = selector_dataset()
    with st.form("form_entrenamiento"):
        modelos_sel = st.multiselect(
            "Modelos", options=list(MODELOS), default=list(MODELOS), format_func=MODELOS.get,
            help="M1-M3 son modelos individuales; M4 y M5 son híbridos (stacking y autoencoder + XGBoost).",
        )
        col1, col2, col3 = st.columns(3)
        escenario = col1.radio(
            "Escenario", options=list(ESCENARIOS), format_func=ESCENARIOS.get,
            help="«socio» predice con condiciones del hogar. «fies» e «hibrido» usan los ítems con los que se define el objetivo: reconstruyen el nivel FIES (circular).",
        )
        esquema = col2.selectbox("Esquema de validación", options=list(ESQUEMAS), format_func=ESQUEMAS.get,
                                 help="V1 es el esquema base; V2 valida por territorio, V3 por área urbana/rural y V4 en el tiempo.")
        n_trials = col3.number_input(
            "Ensayos de optimización por modelo", min_value=1, max_value=N_TRIALS_MAX, value=N_TRIALS_DEFECTO, step=10,
            help="El protocolo prevé 100; valores menores sirven para pruebas rápidas.",
        )
        nombre = st.text_input(
            "Nombre del experimento", value=time.strftime("exp_%Y%m%d_%H%M"), max_chars=60,
            help="Letras, números, punto, guion y guion bajo (3 a 60 caracteres).",
        )
        lanzar = st.form_submit_button("Lanzar entrenamiento", disabled=rol not in ("admin", "analista"))

    en_curso = st.session_state.get("entr_proceso")
    if lanzar:
        if en_curso is not None and en_curso.poll() is None:
            st.warning("Ya hay un entrenamiento en ejecución en esta sesión.")
        elif not modelos_sel:
            st.warning("Seleccione al menos un modelo.")
        elif not PATRON_NOMBRE.fullmatch(nombre or ""):
            st.warning("Nombre de experimento inválido.")
        else:
            if escenario != "socio":
                st.warning(AVISO_CIRCULARIDAD)
            try:
                ruta_datos = os.environ.get("FSAI_RUTA_DATOS", "").strip() or str(ruta_para_pipeline(df_ds, info_ds))
                proceso, log = lanzar_pipeline(modelos_sel, escenario, esquema, int(n_trials), ruta_datos, nombre)
            except (OSError, ValueError) as exc:
                st.error(f"No se pudo lanzar el entrenamiento: {exc}")
            else:
                st.session_state.update(
                    entr_proceso=proceso, entr_log=log, entr_nombre=nombre,
                    entr_total=len(modelos_sel), entr_en_curso=True,
                )
                registrar_auditoria(
                    "entrenamiento_lanzado", "experimento", nombre,
                    {"modelos": modelos_sel, "escenario": escenario, "esquema": esquema,
                     "n_trials": int(n_trials), "dataset": info_ds["nombre"], "pid": proceso.pid},
                )
                st.success(f"Entrenamiento lanzado (PID {proceso.pid}).")

    # ------------------------------------------------------------------ progreso
    st.subheader("Progreso")


    def panel_progreso() -> None:
        proceso = st.session_state.get("entr_proceso")
        if proceso is None:
            st.info("No hay entrenamientos lanzados desde esta sesión. Lance uno arriba para ver aquí su progreso.")
            return
        codigo = proceso.poll()
        nombre_exp = st.session_state["entr_nombre"]
        total = max(1, int(st.session_state.get("entr_total", 1)))
        try:
            prog = consultar(SQL_PROGRESO, {"nombre": nombre_exp}, cache=False)
        except ErrorConsulta as exc:
            mostrar_error_bd(exc)
            prog = pd.DataFrame()
        n_modelos = int(prog.iloc[0]["n_modelos"]) if not prog.empty else 0
        estado_bd = prog.iloc[0]["estado"] if not prog.empty else "sin registro todavía"
        st.progress(min(1.0, n_modelos / total), text=f"Modelos registrados: {n_modelos} de {total}")
        estado_proc = "en ejecución" if codigo is None else f"finalizado (código de salida {codigo})"
        st.write(f"Experimento **{nombre_exp}** · proceso: {estado_proc} · estado en BD: {estado_bd}")
        registro = cola_log(st.session_state["entr_log"])
        st.code(registro or "(el registro aún no contiene salida)", language="text")
        if codigo is None:
            if st.button("Cancelar ejecución"):
                proceso.terminate()
                registrar_auditoria("entrenamiento_cancelado", "experimento", nombre_exp)
                st.rerun()
        elif st.session_state.get("entr_en_curso"):
            st.session_state["entr_en_curso"] = False
            registrar_auditoria(
                "entrenamiento_finalizado", "experimento", nombre_exp, {"codigo_salida": codigo}
            )
            st.rerun()


    proceso_actual = st.session_state.get("entr_proceso")
    if proceso_actual is not None and proceso_actual.poll() is None:
        st.fragment(run_every="5s")(panel_progreso)()
    else:
        panel_progreso()

    # ------------------------------------------------------------------ métricas
    st.subheader("Métricas de desempeño (fsai.metricas)")
    try:
        experimentos = consultar(SQL_EXPERIMENTOS, cache=False)
    except ErrorConsulta as exc:
        mostrar_error_bd(exc)
        return

    if experimentos.empty:
        st.info("Aún no hay experimentos registrados. Lance un entrenamiento arriba: las métricas aparecerán aquí al terminar.")
        return

    etiquetas_exp = {
        int(r.experimento_id): f"#{int(r.experimento_id)} · {r.nombre} · {r.escenario} · {r.esquema_validacion} · {r.estado}"
        for r in experimentos.itertuples()
    }
    c_a, c_b = st.columns([3, 1])
    experimento_id = c_a.selectbox("Experimento", options=list(etiquetas_exp), format_func=etiquetas_exp.get)
    ponderada = c_b.toggle("Métricas ponderadas", value=True)

    try:
        metricas = consultar(SQL_METRICAS, {"experimento_id": int(experimento_id), "ponderada": bool(ponderada)})
    except ErrorConsulta as exc:
        mostrar_error_bd(exc)
        return

    if metricas.empty:
        st.info("Este experimento no tiene métricas agregadas registradas para la opción elegida.")
        return


    def celda(fila: pd.Series) -> str:
        if pd.isna(fila["valor"]):
            return "—"
        if pd.isna(fila["ic_inf"]) or pd.isna(fila["ic_sup"]):
            return fmt_num(fila["valor"])
        return f"{fmt_num(fila['valor'])} [{fmt_num(fila['ic_inf'])}; {fmt_num(fila['ic_sup'])}]"


    metricas = metricas.assign(texto=metricas.apply(celda, axis=1))
    tabla = metricas.pivot_table(
        index="modelo", columns="nombre_metrica", values="texto", aggfunc="first"
    ).fillna("—")
    st.dataframe(tabla)
    st.caption("Cada celda muestra valor [IC inferior; IC superior] según lo almacenado en fsai.metricas.")

    metrica_graf = st.selectbox("Métrica a graficar", options=sorted(metricas["nombre_metrica"].unique()))
    sub = metricas[metricas["nombre_metrica"] == metrica_graf].dropna(subset=["valor"])
    if not sub.empty:
        con_ic = sub[["ic_inf", "ic_sup"]].notna().all(axis=1)
        fig = go.Figure(
            go.Bar(
                x=sub["modelo"], y=sub["valor"], marker_color="#1b6ca8",
                error_y={
                    "type": "data", "symmetric": False,
                    "array": (sub["ic_sup"] - sub["valor"]).where(con_ic, 0.0),
                    "arrayminus": (sub["valor"] - sub["ic_inf"]).where(con_ic, 0.0),
                },
            )
        )
        fig.update_layout(xaxis_title="Modelo", yaxis_title=metrica_graf, margin={"t": 10})
        st.plotly_chart(fig)



def _exportar_importar_config(modelos_sel, escenario, n_trials, nombre, agrupar, df, info) -> None:
    """Exporta la configuración del formulario a JSON versionado o valida un JSON subido. NUNCA lanza un entrenamiento."""
    import json as _json

    from fsai import config_experimento as CE  # noqa: E402

    st.caption("Guarde la configuración de esta corrida en un archivo JSON para reproducirla o compartirla, o cargue una para revisarla. "
               "Importar solo valida y muestra el contenido: no inicia ningún entrenamiento ni cambia el formulario.")
    st.markdown("**Exportar** (valores actuales del formulario)")
    hash_ds = st.text_input("Hash SHA-256 del dataset", value="", key="cfg_hash_ds", max_chars=128,
                            help="Identifica el archivo de datos usado. Péguelo o use el botón para calcularlo desde el dataset activo (si es un archivo).")
    es_archivo = info.get("origen") == "archivo"

    def _calcular_hash() -> None:
        try:
            st.session_state["cfg_hash_ds"] = CE.hash_archivo(ruta_para_pipeline(df, info))
        except (OSError, ValueError) as exc:
            st.session_state["cfg_hash_err"] = str(exc)

    st.button("Calcular hash del dataset activo", key="cfg_calc_hash", on_click=_calcular_hash, disabled=not es_archivo,
              help="Solo disponible si el dataset activo es un archivo; los datasets de PostgreSQL no tienen un archivo de origen único.")
    if st.session_state.pop("cfg_hash_err", None):
        st.warning("No se pudo calcular el hash del dataset activo.")
    if escenario == "todos":
        st.info("La configuración exportable admite un solo escenario: elija «socio», «fies» o «hibrido» en el formulario.")
    else:
        try:
            cfg = CE.exportar_config(run_id=nombre, escenario=escenario, modelos=list(modelos_sel), n_trials=int(n_trials),
                                     hash_dataset=hash_ds.strip(), agrupar_por=agrupar)
        except CE.ConfigInvalida as exc:
            st.warning("Complete lo siguiente para poder exportar: " + "; ".join(exc.problemas))
        except (KeyError, ValueError) as exc:
            st.warning(f"No se pudo construir la configuración: {exc}")
        else:
            st.download_button("Descargar configuración (JSON)", _json.dumps(cfg, ensure_ascii=False, indent=2).encode("utf-8"),
                               file_name=f"config_{nombre}.json", mime="application/json", key="cfg_descarga")
    st.markdown("**Importar** (solo validar)")
    subido = st.file_uploader("Archivo de configuración (.json)", type=["json"], key="cfg_subida")
    if subido is not None:
        try:
            datos = _json.loads(subido.getvalue().decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            st.error(f"El archivo no es un JSON válido: {exc}")
            return
        problemas = CE.validar_config(datos)
        if problemas:
            st.error("La configuración no es válida:")
            for p in problemas:
                st.markdown(f"- {p}")
            return
        cfg_imp = CE.importar_config(datos)
        st.success("Configuración válida. No se ejecutó nada.")
        if cfg_imp.escenario != "socio":
            st.warning(AVISO_CIRCULARIDAD)
        if cfg_imp.n_trials != int(n_trials):
            st.warning(f"La configuración usa n_trials = {cfg_imp.n_trials}, distinto del formulario ({int(n_trials)}).")
        if es_archivo:
            try:
                if not CE.verificar_hash_dataset(cfg_imp, ruta_para_pipeline(df, info)):
                    st.warning("El hash del dataset activo NO coincide con el de la configuración: los resultados pueden no ser reproducibles.")
            except (OSError, ValueError):
                st.caption("No se pudo verificar el hash contra el dataset activo (archivo no accesible).")
        else:
            st.caption("El dataset activo viene de PostgreSQL: no se compara el hash del archivo.")
        st.json(cfg_imp.a_dict(), expanded=False)


def seccion_anidada() -> None:
    """Partición 70/15/15 y validación cruzada anidada 5x3 (solo Desarrollo)."""
    df, info = selector_dataset()
    # ------------------------------------------------------------------ A · Cómo encaja
    st.subheader("A · Cómo encajan las particiones")
    st.markdown(
        "| Partición | Uso | Cuándo |\n|---|---|---|\n"
        "| **Desarrollo 70 %** | CV anidada 5 externos × 3 internos (Optuna solo en los internos) y modelo final | Entrenamiento (esta sección) |\n"
        "| **Validación 15 %** | Umbral operativo de la clase severa, calibración (isotónica) y parada temprana de M5 | Entrenamiento, tras el modelo final |\n"
        "| **Prueba 15 %** | **Una sola evaluación** por modelo, bloqueada tras su uso | Comparación estadística |\n")
    st.caption("Anti-fuga: imputación, escalado y SMOTE se ajustan dentro de cada pliegue de entrenamiento; la partición se genera una vez, "
               "se guarda con huella SHA-256 y es la misma para todos los modelos (semilla SEED=42).")

    # ------------------------------------------------------------------ B · Configuración
    st.subheader("B · Configuración del experimento")
    with st.form("form_cv_anidada"):
        c1, c2 = st.columns(2)
        modelos_sel = c1.multiselect("Modelos", list(MODELOS_F), default=["M1", "M2"], format_func=MODELOS_F.get)
        escenario = c2.selectbox("Escenario", list(ESCENARIOS_F) + ["todos"], format_func=lambda k: ESCENARIOS_F.get(k, "todos · socio + fies + hibrido"))
        if escenario != "socio":
            st.warning(AVISO_CIRCULARIDAD)
        c3, c4, c5, c6 = st.columns(4)
        agrupar = c3.selectbox("Agrupar por", ["hogar_uid", "conglome"], help="hogar_uid = V1; conglome evita compartir conglomerados entre particiones.")
        n_trials = c4.number_input("Ensayos Optuna por pliegue externo", 1, N_TRIALS_MAX, 20, step=5,
                                   help="Contrato: N_TRIALS=100 para el estudio real; 20 es un valor de exploración (el costo crece con los modelos).")
        n_final = c5.number_input("Ensayos de la búsqueda final", 1, N_TRIALS_MAX, 20, step=5)
        ponderar = c6.checkbox("Ponderar el ajuste con factor_expansion", value=False,
                               help="Por defecto los pesos solo ponderan el F1 de validación (criterio Optuna), no el ajuste (contrato).")
        nombre = st.text_input("Nombre de la corrida", value=time.strftime("cv_%Y%m%d_%H%M"), max_chars=60)
        lanzar = st.form_submit_button("Lanzar CV anidada", disabled=rol not in ("admin", "analista"))

    with st.expander("Hiperparámetros por defecto asumidos y su justificación (espacio de la sección 7 del protocolo)"):
        for cod in modelos_sel or ["M1"]:
            st.markdown(f"**{MODELOS_F[cod]}**")
            filas = []
            for p, s in ESPACIO_DEFECTO[cod].items():
                rango = f"{s['lo']} … {s['hi']}" + (" (log)" if s.get("log") else "") + (f", paso {s['step']}" if s.get("step", 1) > 1 else "") if s["tipo"] != "cat" else ", ".join(map(str, s["opciones"]))
                filas.append({"hiperparámetro": p, "rango / opciones": rango, "justificación": JUSTIFICACION.get(p, "")})
            st.dataframe(pd.DataFrame(filas), hide_index=True)
        st.caption("Muestreador TPE (semilla distinta por pliegue, derivada de SEED), pruner de la mediana, criterio = F1-macro ponderado en CV interna "
                   "GroupKFold(3) por hogar. Para editar rangos use el módulo Optimización Optuna.")

    with st.expander("Exportar / importar configuración"):
        _exportar_importar_config(modelos_sel, escenario, int(n_trials), nombre, agrupar, df, info)

    proceso = st.session_state.get("f2_proceso")
    if lanzar:
        if proceso is not None and proceso.poll() is None:
            st.warning("Ya hay un entrenamiento en ejecución en esta sesión.")
        elif not modelos_sel:
            st.warning("Seleccione al menos un modelo.")
        elif not PATRON_NOMBRE_F.fullmatch(nombre or ""):
            st.warning("Nombre inválido (3-60 caracteres: letras, números, punto, guion, guion bajo).")
        elif (C.DIR_FASES / nombre).exists():
            st.warning("Ya existe una corrida con ese nombre; elija otro.")
        else:
            try:
                ruta = ruta_para_pipeline(df, info)
                args = ["--ruta-datos", str(ruta), "--modelos", *modelos_sel, "--escenario", escenario, "--trials", str(int(n_trials)),
                        "--trials-final", str(int(n_final)), "--cv-anidada", "--agrupar-por", agrupar, "--nombre-experimento", nombre]
                if ponderar:
                    args.append("--ponderar-ajuste")
                proceso = lanzar_subproceso(args, C.DIR_FASES / f"{nombre}.stdout.log")
            except (OSError, ValueError) as exc:
                st.error(f"No se pudo lanzar: {exc}")
            else:
                st.session_state.update(f2_proceso=proceso, f2_nombre=nombre)
                audit("entrenamiento_cv_anidada_lanzada", "corrida", nombre, {"modelos": modelos_sel, "escenario": escenario, "trials": int(n_trials),
                                                                      "trials_final": int(n_final), "agrupar_por": agrupar, "dataset": info["nombre"], "pid": proceso.pid})
                st.success(f"Corrida «{nombre}» lanzada (PID {proceso.pid}).")

    # ------------------------------------------------------------------ C · Monitoreo
    st.subheader("C · Monitoreo en tiempo real")


    def panel_progreso() -> None:
        nombre_c = st.session_state.get("f2_nombre")
        if not nombre_c:
            st.info("No hay corridas lanzadas desde esta sesión. Las corridas anteriores se ven en la sección D.")
            return
        d = C.DIR_FASES / nombre_c
        prog = leer_json(d / "progreso.json")
        pr = st.session_state.get("f2_proceso")
        codigo = None if pr is None else pr.poll()
        if not prog:
            st.write(f"Corrida «{nombre_c}»: iniciando (cargando datos y generando la partición)...")
            st.code(cola_texto(C.DIR_FASES / f"{nombre_c}.stdout.log") or "(sin salida todavía)", language="text")
            return
        n_comb = max(int(prog.get("n_combinaciones", 1)), 1)
        comb = int(prog.get("combinacion", 1))
        n_tr = max(int(prog.get("n_trials", 1)), 1)
        fase_f = (((int(prog.get("fold") or 1) - 1) + int(prog.get("trial_actual", 0)) / n_tr) / 5 * 0.8) if prog.get("fase") == "cv_externa" else (
            0.8 + 0.2 * int(prog.get("trial_actual", 0)) / n_tr if prog.get("fase") == "modelo_final" else (0.99 if prog.get("fase") == "validacion" else 1.0))
        total = min(1.0, ((comb - 1) + fase_f) / n_comb) if prog.get("estado") != "completado" else 1.0
        st.progress(total, text=f"{prog.get('estado', '?')} · {prog.get('modelo', '')} / {prog.get('escenario', '')} · combinación {comb}/{n_comb}")
        a, b = st.columns(2)
        a.progress(min(1.0, (int(prog.get("fold") or 0)) / 5), text=f"Pliegue externo {prog.get('fold') or '—'}/5 ({ETAPAS.get(prog.get('fase', ''), prog.get('fase', ''))})")
        b.progress(min(1.0, int(prog.get("trial_actual", 0)) / n_tr), text=f"Ensayo Optuna {prog.get('trial_actual', 0)}/{n_tr}")
        st.write(prog.get("mensaje", ""))
        if prog.get("trials"):
            st.plotly_chart(P.fig_curva_optuna(prog["trials"]), key=f"curva_{nombre_c}_{len(prog['trials'])}")
            interpretacion(I.i_curva_optuna([t["valor"] for t in prog["trials"] if t.get("valor") is not None]), "Interpretación de la curva de ensayos")
        st.code(cola_texto(d / "log.txt") or "(sin registro)", language="text")
        if codigo is None and pr is not None:
            if st.button("Cancelar ejecución", key="cancelar_f2"):
                pr.terminate()
                audit("entrenamiento_cv_anidada_cancelada", "corrida", nombre_c)
                st.rerun()
        elif codigo is not None and st.session_state.get("f2_audit_fin") != nombre_c:
            st.session_state["f2_audit_fin"] = nombre_c
            audit("entrenamiento_cv_anidada_finalizada", "corrida", nombre_c, {"codigo_salida": codigo, "estado": prog.get("estado")})
            st.rerun()


    pr0 = st.session_state.get("f2_proceso")
    if pr0 is not None and pr0.poll() is None:
        st.fragment(run_every="3s")(panel_progreso)()
    else:
        panel_progreso()

    # ------------------------------------------------------------------ D · Resultados
    st.subheader("D · Resultados de la validación cruzada anidada")
    corridas = EV.listar_corridas()
    if not corridas:
        st.info("Aún no hay corridas completadas. Lance una arriba; con pocos ensayos tarda unos minutos.")
        return
    nombre_sel = st.selectbox("Corrida", corridas)
    dir_run = EV.dir_corrida(nombre_sel)
    cfg = EV.config_corrida(nombre_sel)
    resultados = EV.resultados_corrida(nombre_sel)
    ver = cfg.get("verificacion_particion", {})
    st.caption(f"Datos: `{cfg.get('ruta_datos', '').split(chr(92))[-1]}` · huella de partición `{cfg.get('huella_particion', '')[:16]}…` · agrupada por {cfg.get('agrupar_por')}")
    interpretacion(I.i_particion(ver), "Partición utilizada")
    ident = st.selectbox("Modelo / escenario", list(resultados), format_func=lambda k: f"{k.split('_')[1]} · {k.split('_')[0]}")
    res = resultados[ident]
    if res["escenario"] != "socio":
        st.warning(AVISO_CIRCULARIDAD)
    bloqueada = (dir_run / f"prueba_{ident}.lock").exists()
    st.caption("Estado de la Prueba (15 %): " + ("**ya evaluada** (bloqueada)" if bloqueada else "**intacta** (se evalúa en Comparación estadística)"))

    pond = st.toggle("Métricas ponderadas (factor de expansión)", value=True)
    clave = "ponderada" if pond else "no_ponderada"
    tabla = pd.DataFrame([{"métrica": m, "media ± DE [IC t 95 %]": formatear_ic(v), "n pliegues": int(v["n"])} for m, v in res["agregado"][clave].items()])
    st.dataframe(tabla, hide_index=True)
    interpretacion(I.i_nested(res["agregado"]))

    met_graf = st.selectbox("Métrica por pliegue", N.METRICAS_AGREGADAS, index=N.METRICAS_AGREGADAS.index("f1_macro"))
    vals = [f["metricas"][clave].get(met_graf, np.nan) for f in res["folds"]]
    st.plotly_chart(P.fig_nested(vals, met_graf, res["agregado"][clave][met_graf]), key=f"nested_{ident}_{met_graf}_{clave}")
    pliegues = pd.DataFrame([{"pliegue": f["pliegue"] + 1, "outer_train": f["n_outer_train"], "outer_val": f["n_outer_val"], "F1 interno (Optuna)": round(f["mejor_valor_interno"], 4),
                              "F1-macro outer_val": round(f["metricas"][clave]["f1_macro"], 4), "recall severa": round(f["metricas"][clave].get("recall_severa", np.nan), 4),
                              "ensayos": f["n_trials"], "seg.": f["segundos"], "mejores hiperparámetros": str(f["mejores_params"])} for f in res["folds"]])
    st.dataframe(pliegues, hide_index=True)
    interpretacion(I.i_estabilidad_hp(res["folds"]), "Estabilidad de hiperparámetros entre pliegues")

    with st.expander("Auditoría del diseño anidado (verificable por pliegue)"):
        aud = pd.DataFrame([{"pliegue": f["pliegue"] + 1, **f["auditoria"]} for f in res["folds"]])
        st.dataframe(aud, hide_index=True)
        st.caption("Optuna solo vio outer_train (CV interna de 3 pliegues); outer_val se evaluó una vez; la Prueba no se tocó.")

    trials_csv = dir_run / f"trials_{ident}.csv"
    if trials_csv.is_file():
        tr = pd.read_csv(trials_csv)
        with st.expander("Ensayos Optuna e importancia (asociación) de hiperparámetros"):
            st.dataframe(tr, hide_index=True)
            pcols = [c for c in tr.columns if c.startswith("p_") and pd.api.types.is_numeric_dtype(tr[c])]
            asoc = pd.DataFrame([{"hiperparámetro": c[2:], "importancia": abs(tr[c].corr(tr["valor"], method="spearman"))} for c in pcols if tr[c].nunique() > 1]).dropna()
            if len(asoc) and len(tr) >= 8:
                st.plotly_chart(P.fig_importancia(asoc, "hiperparámetro", "importancia"), key=f"imp_hp_{ident}")
                interpretacion({"que_muestra": f"Entre {len(asoc)} hiperparámetros numéricos, el más asociado al F1 interno es «{asoc.sort_values('importancia').iloc[-1]['hiperparámetro']}» (|rho de Spearman| {asoc['importancia'].max():.2f}).",
                                "como_leerlo": "Barras largas = el valor objetivo cambia de forma monótona con ese hiperparámetro en los ensayos realizados.",
                                "limitacion": "Con pocos ensayos la asociación es ruidosa; para importancias fANOVA use el módulo Optimización Optuna."})
            else:
                st.info("Se necesitan al menos 8 ensayos con hiperparámetros numéricos variables para estimar asociaciones.")

    fin = res["final"]
    st.markdown("**Modelo final, umbral operativo y calibración (Validación 15 %)**")
    val = fin["validacion"]
    cA, cB, cC = st.columns(3)
    cA.metric("Umbral P(severa)", f"{val['umbral_severa']:.3f}")
    cB.metric("Sensibilidad severa (Validación)", f"{val['recall_severa_umbral']:.3f}" if val["recall_severa_umbral"] is not None else "—")
    cC.metric("Especificidad severa (Validación)", f"{val['especificidad_severa_umbral']:.3f}")
    st.caption(f"Umbral = máxima sensibilidad con especificidad ≥ {C.ESPECIFICIDAD_MIN} calculada SOLO en Validación (n={val['n']}). Hiperparámetros finales: {fin['hiperparametros']}")

    if st.button("Calcular curva de aprendizaje (entrenamiento vs Validación)"):
        with st.spinner("Entrenando con fracciones crecientes de Desarrollo..."):
            from fsai.splits import cargar_particion
            part, _ = cargar_particion(df, dir_run / "particion.json")
            ca = N.curva_aprendizaje(df, part, dir_run, ident)
        fig = P.fig_linea_xy(ca["n_entrenamiento"], ca["f1_validacion"], "Hogares de entrenamiento", "F1-macro ponderado", "Validación")
        fig.add_scatter(x=ca["n_entrenamiento"], y=ca["f1_entrenamiento"], mode="lines+markers", name="Entrenamiento")
        st.plotly_chart(fig, key=f"aprend_{ident}")
        brecha = float(ca["f1_entrenamiento"].iloc[-1] - ca["f1_validacion"].iloc[-1])
        interpretacion({"que_muestra": f"Con todo Desarrollo ({int(ca['n_entrenamiento'].iloc[-1]):,} hogares) el F1 es {ca['f1_entrenamiento'].iloc[-1]:.3f} en entrenamiento y {ca['f1_validacion'].iloc[-1]:.3f} en Validación (brecha {brecha:.3f}).",
                        "como_leerlo": "Una brecha grande indica sobreajuste; si la curva de Validación aún sube al final, más datos ayudarían.",
                        "limitacion": "Una sola semilla de submuestreo; usa Validación (no Prueba)."})



with st.container(border=True):
    st.markdown("**¿Qué método usar?**")
    st.markdown(
        "| | Validación cruzada anidada (recomendada) | Comparación de esquemas V1-V5 |\n|---|---|---|\n"
        "| **Para qué sirve** | Estimar el desempeño sin sesgo optimista y dejar un modelo final listo | Probar la robustez del modelo (por hogar, conglomerado, territorio, área, tiempo) |\n"
        "| **Datos que usa** | Desarrollo 70 % (la Prueba 15 % queda intacta) | Todo el dataset por esquema |\n"
        "| **Dónde quedan los resultados** | Archivos de la corrida (modelo, partición, predicciones) | Tablas de métricas en PostgreSQL |\n"
        "| **Siguiente paso** | Comparación estadística, Interpretabilidad, Reportes | Reportes (resultados por esquema y estrato) |"
    )
t_anidada, t_clasico = st.tabs(["Validación cruzada anidada (recomendada)", "Comparación de esquemas de validación (V1-V5)"])
with t_anidada:
    seccion_anidada()
with t_clasico:
    seccion_clasica()
