"""Reportes: informe científico (PDF, Word, Excel) con figuras a 600 DPI, generado desde PostgreSQL y los artefactos de entrenamiento.

Reutiliza fsai.reportes_fases (construcción del informe), fsai.report_generator (exportadores) y fsai.plots (Matplotlib:
kaleido no está instalado). Nunca evalúa la Prueba: solo usa sus predicciones si ya fueron guardadas.
"""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

import tema
from auth import barra_lateral, requerir_acceso
from comun import explicar_error, get_engine
from fases_comun import C, audit, selector_dataset
from tema import config_pagina, encabezado

from fsai import explainability as X  # noqa: E402
from fsai import reportes_fases as R  # noqa: E402
from fsai.preprocessing import preparar_matriz  # noqa: E402
from fsai.splits import cargar_particion  # noqa: E402

config_pagina("Reportes")
usuario, rol = requerir_acceso("reportes")
barra_lateral()
encabezado(
    "Reportes",
    "Genere un informe científico completo (PDF, Word y Excel) con tablas, figuras a 600 DPI y su interpretación.",
    "Resultados y despliegue",
    pasos=["Elija la corrida de entrenamiento, el escenario y los modelos (o ninguna corrida para un informe solo de datos).",
           "Marque las secciones y los formatos que desea; el informe siempre incluye «Limitaciones y uso responsable».",
           "Pulse «Generar informe» y descargue los archivos; las secciones cuyo insumo falta lo indican con un aviso."],
    produce="Informe con portada, resumen ejecutivo, datos, métodos, resultados, comparación estadística, interpretabilidad, equidad, limitaciones y reproducibilidad.",
)
df, info = selector_dataset()
if info["origen"] != "bd":
    tema.vacio("El informe se construye desde PostgreSQL y el dataset activo es un archivo.", "Elija un dataset de PostgreSQL en la barra lateral o cárguelo en «Carga de datos»",
               "pages/02_Importacion_ENAHO.py")
    st.stop()

MIMES = {"PDF": "application/pdf", "Word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
         "Excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
corridas = R.corridas_disponibles()
SIN_CORRIDA = "(sin corrida: solo datos y diseño)"

st.subheader("1. Qué incluir")
c1, c2 = st.columns(2)
nombres = {str(p): f"{p.name}" + (" · datos reales" if p.parent.parent.name == "real" else "") for p in corridas}
clave_run = c1.selectbox("Corrida de entrenamiento", [SIN_CORRIDA, *nombres], index=1 if nombres else 0, format_func=lambda k: k if k == SIN_CORRIDA else nombres[k],
                         help="Las corridas se crean en el módulo «Entrenamiento». Sin corrida solo se describen los datos.")
dir_run = None if clave_run == SIN_CORRIDA else Path(clave_run)
escenario, sel_cod = "socio", []
if dir_run is not None:
    pares = [tuple(p.stem.split("_", 2)[1:]) for p in dir_run.glob("resultado_*.json")]
    escenarios = sorted({e for e, _c in pares})
    escenario = c2.selectbox("Escenario", escenarios, help="«socio» es la predicción real; «fies» e «hibrido» son reconstrucciones circulares del objetivo.")
    codigos = sorted(c for e, c in pares if e == escenario)
    sel_cod = st.multiselect("Modelos", codigos, default=codigos, help="Modelos de la corrida que se incluirán en las comparaciones.")
else:
    c2.caption("Sin corrida no hay secciones de desempeño, comparación, interpretabilidad ni equidad.")

c3, c4 = st.columns(2)
ETIQ_CORTA = {"datos": "1. Datos", "perfil": "1b. Perfil de vulnerabilidad", "metodos": "2. Métodos", "resultados": "3. Resultados", "comparacion": "4. Comparación", "interpretabilidad": "5. Interpretabilidad",
             "equidad": "6. Equidad", "reproducibilidad": "8. Reproducibilidad"}
incl = c3.multiselect("Secciones", R.SECCIONES, default=R.SECCIONES, format_func=ETIQ_CORTA.get,
                      help="«Limitaciones y uso responsable» (sección 7) se incluye siempre.")
formatos = c4.multiselect("Formatos", ["PDF", "Word", "Excel"], default=["PDF", "Word", "Excel"],
                          help="El PDF y el Word son el manuscrito; el Excel trae una hoja por sección con todas las filas.")
c5, c6 = st.columns(2)
n_boot = c5.select_slider("Réplicas bootstrap (IC de Prueba y equidad)", options=[100, 200, 500, 1000], value=200,
                          help="Más réplicas dan IC más estables pero tardan más.")
con_shap = c6.checkbox("Recalcular SHAP con más detalle (más lento)", value=False,
                       disabled=dir_run is None or not sel_cod,
                       help="El informe SIEMPRE incluye PDP/ALE de las 8 variables principales (mejor modelo, submuestra ponderada de Validación de 2 000 hogares). "
                            "Si marca esta casilla, además se recalcula SHAP con el primer modelo seleccionado y PDP/ALE usan una submuestra de 5 000 hogares. Sin marcarla se usan las explicaciones guardadas.")
titulo = st.text_input("Título del informe (opcional)", value="", max_chars=120)
st.caption("Las figuras se exportan como PNG a **600 DPI** (paleta apta para daltonismo, columna simple 8,5 cm o doble 17,8 cm) con Matplotlib.")


def _payload_shap() -> dict | None:
    """SHAP firmado (clase IA severa) e importancia global; marca ``detalle`` para PDP/ALE con submuestra mayor (el informe calcula PDP/ALE siempre)."""
    import joblib
    import numpy as np
    ident = f"{escenario}_{sel_cod[0]}"
    res = R._json(dir_run / f"resultado_{ident}.json")  # noqa: SLF001
    part, _ = cargar_particion(df, dir_run / "particion.json")
    pipe = joblib.load(dir_run / res["final"]["modelo"])
    Xv = preparar_matriz(df.iloc[part["validacion"]].reset_index(drop=True), res["features_num"], res["features_cat"])
    sm = X.shap_matriz(pipe, Xv, res["codigo"], n_muestra=60 if res["codigo"] == "M4" else 100)
    crudas = X.agrupar_a_columna_cruda(sm["nombres"], list(Xv.columns))
    cols = sm["columnas_crudas"]
    firmado = np.stack([sm["vals"][C.CLASE_SEVERA][:, [i for i, c in enumerate(crudas) if c == col]].sum(axis=1) for col in cols], axis=1)
    imp = pd.DataFrame({"variable": cols, "importancia": sm["agregado_crudo"].mean(axis=(0, 1))})
    muestra = Xv.loc[sm["indice"], cols].reset_index(drop=True)
    return {"importancia": imp, "firmado": firmado, "X": muestra, "detalle": True,
            "fuente": f"SHAP calculado sobre {len(muestra)} hogares de Validación con el modelo {res['codigo']} (clase IA severa)"}


if st.button("Generar informe", type="primary", disabled=not incl or not formatos, icon=":material/description:"):
    try:
        with st.spinner("Generando tablas, figuras a 600 DPI y documentos (puede tardar uno o dos minutos)..."):
            shap = None
            if con_shap and dir_run is not None and sel_cod:
                try:
                    shap = _payload_shap()
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"No se pudo calcular SHAP nuevo ({type(exc).__name__}: {str(exc)[:120]}), así que el informe usará las explicaciones guardadas si existen. "
                               "Para evitar este paso, desmarque «Recalcular SHAP con más detalle» y genere el informe de nuevo; revise «Interpretabilidad» para ver si el modelo carga.")
            salida = C.ARTIFACTS_DIR / "reportes" / f"{(dir_run.name if dir_run else 'datos')}_{escenario}_{time.strftime('%Y%m%d_%H%M%S')}"
            out = R.generar_reporte(get_engine(), int(info["dataset_id"]), df, dir_run, escenario, sel_cod, incl, salida, formatos, shap,
                                    titulo.strip() or None, version_app=tema.VERSION_APP, n_boot=int(n_boot))
        st.session_state["reporte_salida"] = out
        audit("reportes_generado", "corrida", dir_run.name if dir_run else "sin_corrida",
              {"escenario": escenario, "modelos": sel_cod, "formatos": formatos, "secciones": incl, "dataset_id": int(info["dataset_id"])})
        st.success(f"Informe generado: {len(out['figuras'])} figuras y {len(out['archivos'])} documento(s).")
    except Exception as exc:  # noqa: BLE001
        st.error(explicar_error(exc, "generar el informe"))

out = st.session_state.get("reporte_salida")
if out:
    st.subheader("2. Descargas")
    cols = st.columns(len(out["archivos"]) + 1)
    for col, (fmt, ruta) in zip(cols, out["archivos"].items()):
        col.download_button(f"Descargar {fmt}", data=ruta.read_bytes(), file_name=ruta.name, mime=MIMES[fmt], key=f"dl_{fmt}", use_container_width=True)
    if out["figuras"]:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            for f in out["figuras"]:
                z.write(f, f.name)
        cols[-1].download_button("Descargar figuras (.zip)", data=buf.getvalue(), file_name="figuras_600dpi.zip", mime="application/zip", key="dl_figs", use_container_width=True)
    st.markdown("**Secciones del informe:** " + " · ".join(out["secciones"]))
    if out["figuras"]:
        st.subheader("3. Figuras exportadas")
        st.dataframe(pd.DataFrame([{"archivo": f.name, "DPI": str(R.dpi_png(f)), "KB": round(f.stat().st_size / 1024)} for f in out["figuras"]]), hide_index=True)
        for f in out["figuras"][:4]:
            st.image(str(f), caption=f"{f.name} · {R.dpi_png(f)} DPI", width=420)
else:
    st.info("Todavía no se generó ningún informe. Elija las opciones y pulse «Generar informe».")
