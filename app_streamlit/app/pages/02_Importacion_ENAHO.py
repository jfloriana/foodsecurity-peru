"""Importación de ENAHO: carga de archivos tabulares (CSV/.dta/.sav/.parquet) o datasets de PostgreSQL, editor de metadata y validación de grupos.

Acceso: administrador."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from auth import ROLES_ESCRITURA, barra_lateral, requerir_acceso
from tema import config_pagina, encabezado
from fases_comun import C, D, audit, interpretacion, selector_dataset

from fsai import interpretacion as I  # noqa: E402
from fsai import plots as P  # noqa: E402
from fsai import splits as S  # noqa: E402
config_pagina("Importación ENAHO")
usuario, rol = requerir_acceso("importacion_enaho")
barra_lateral()
encabezado("Carga de datos (ENAHO)",
    "Cargue microdatos de la ENAHO, edite el diccionario de variables y valide los grupos de análisis.",
    "Datos",
    pasos=["Suba un archivo tabular (CSV, .dta, .sav o .parquet) a nivel hogar o elija un dataset ya registrado en la barra lateral.",
           "Revise y edite el diccionario de variables (tipo, rol y rango válido).",
           "Valide que ningún hogar quede repetido entre grupos de análisis."],
    produce="Un dataset disponible para todos los módulos y un diccionario de variables guardado.",
)
df, info = selector_dataset()
puede_escribir = rol in ROLES_ESCRITURA

t_carga, t_meta, t_grupos = st.tabs(["1 · Carga de datasets", "2 · Editor de metadata", "3 · Validación de grupos (partición 70/15/15)"])

# ------------------------------------------------------------------ 1 · Carga
with t_carga:
    r = D.resumen_dataset(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hogares", f"{r['n_hogares']:,}")
    c2.metric("Variables", r["n_columnas"])
    c3.metric("Conglomerados", f"{r['n_conglomerados']:,}" if r["n_conglomerados"] else "—")
    c4.metric("n efectivo (Kish)", f"{r['n_efectivo_kish']:,.0f}")
    st.caption(f"Dataset activo: **{info['etiqueta']}** · nulos totales {r['pct_nulos_total']} %. "
               "El tamaño efectivo de Kish descuenta el efecto del diseño muestral sobre la precisión.")
    st.markdown("**Archivos tabulares disponibles**")
    arch = D.listar_archivos_datos()
    st.dataframe(pd.DataFrame({"archivo": [p.name for p in arch], "carpeta": [p.parent.name for p in arch],
                               "tamaño (KB)": [round(p.stat().st_size / 1024, 1) for p in arch]}), hide_index=True)
    if puede_escribir:
        st.markdown("**Subir un archivo tabular** (CSV, .dta, .sav o .parquet; máx. 200 MB) y, opcionalmente, su diccionario")
        subida = st.file_uploader("Archivo de hogares", type=["csv", "dta", "sav", "parquet"], key="subida_datos")
        dicc = st.file_uploader("Diccionario / metadata (CSV con columnas variable, tipo, rol, rango_min, rango_max, descripcion)",
                                type=["csv"], key="subida_dicc")
        if subida is not None and st.button("Guardar archivo en la carpeta de cargas"):
            try:
                destino = D.guardar_carga(subida.name, subida.getvalue())
                if dicc is not None:
                    meta_up = pd.read_csv(io.BytesIO(dicc.getvalue()))
                    faltan = {"variable", "tipo", "rol"} - set(meta_up.columns)
                    if faltan:
                        raise ValueError(f"El diccionario no tiene las columnas {sorted(faltan)}.")
                    for c in ("rango_min", "rango_max", "descripcion", "pct_nulos"):
                        if c not in meta_up.columns:
                            meta_up[c] = np.nan if c != "descripcion" else ""
                    D.guardar_metadata(destino.name, meta_up, usuario)
                audit("importacion_carga_archivo", "dataset", destino.name, {"bytes": len(subida.getvalue())})
                st.success(f"Archivo guardado como {destino.name}. Selecciónelo en «Dataset activo» (barra lateral).")
            except (ValueError, ImportError) as exc:
                st.error(f"No se pudo guardar: {exc}")
    else:
        st.info("Su rol (lector) puede consultar los datos pero no subir archivos.")

# ------------------------------------------------------------------ 2 · Metadata
with t_meta:
    clave_meta = info["nombre"]
    guardada = D.leer_metadata(clave_meta)
    meta = guardada if guardada is not None else D.metadata_defecto(df)
    st.caption("Metadata guardada: " + ("sí (JSON persistido)" if guardada is not None else "no; se muestra la propuesta por defecto derivada del contrato y de los datos."))
    editada = st.data_editor(
        meta, hide_index=True, num_rows="fixed", disabled=not puede_escribir or ["variable", "pct_nulos"], key=f"meta_{clave_meta}",
        column_config={"tipo": st.column_config.SelectboxColumn("tipo", options=list(D.TIPOS_META), required=True),
                       "rol": st.column_config.SelectboxColumn("rol", options=list(D.ROLES_META), required=True),
                       "rango_min": st.column_config.NumberColumn("rango_min"), "rango_max": st.column_config.NumberColumn("rango_max"),
                       "descripcion": st.column_config.TextColumn("descripcion", width="large")})
    cA, cB = st.columns(2)
    if cA.button("Guardar metadata (JSON)", disabled=not puede_escribir):
        try:
            ruta = D.guardar_metadata(clave_meta, editada, usuario)
            audit("importacion_metadata_guardada", "dataset", clave_meta, {"variables": len(editada)})
            st.success(f"Metadata guardada en {ruta.name}.")
        except ValueError as exc:
            st.error(f"Metadata inválida: {exc}")
    cB.download_button("Descargar metadata (JSON)", data=editada.to_json(orient="records", force_ascii=False, indent=1),
                       file_name=f"metadata_{clave_meta}.json", mime="application/json")
    rangos = D.verificar_rangos(df, editada)
    if not rangos.empty:
        st.markdown("**Verificación de rangos declarados**")
        st.dataframe(rangos, hide_index=True)
        fuera = int(rangos["n_fuera_de_rango"].sum())
        interpretacion({"que_muestra": f"{fuera} valores quedan fuera de los rangos declarados en {int((rangos['n_fuera_de_rango'] > 0).sum())} de {len(rangos)} variables con rango.",
                        "como_leerlo": "Un rango declarado incorrecto o un valor fuera de rango señalan errores de captura o de recodificación (p. ej. ítems FIES sin recodificar).",
                        "limitacion": "Los rangos del contrato son los de ENAHO/INEI; los editados por el usuario son su responsabilidad."})

# ------------------------------------------------------------------ 4 · Validación de grupos
with t_grupos:
    st.markdown("**Vista previa de la partición 70/15/15** (la partición definitiva, con huella, la guarda el módulo Entrenamiento).")
    agrupar = st.radio("Agrupar por (un grupo no cruza particiones)", ["hogar_uid", "conglome"], horizontal=True,
                       help="hogar_uid = esquema V1 del contrato; conglome = sensibilidad V1c (evita compartir conglomerados).")
    if agrupar not in df.columns:
        st.warning(f"El dataset no tiene la columna «{agrupar}».")
        st.stop()

    @st.cache_data(show_spinner="Generando partición...", max_entries=6)
    def _particion(clave: str, n: int, ag: str):
        p = S.particionar_70_15_15(df, C.SEED, ag)
        return p, S.verificar_particion(df, p)

    part, ver = _particion(info["clave"], len(df), agrupar)
    ok_uid = ver["cruces_hogar_uid_total"] == 0 and ver["exhaustiva"]
    (st.success if ok_uid else st.error)(
        f"hogar_uid compartidos entre particiones: {ver['cruces_hogar_uid_total']} · partición exhaustiva: {ver['exhaustiva']} · "
        f"conglome compartidos: {ver.get('cruces_conglome_total')}")
    tam_tab = pd.DataFrame({"partición": list(ver["tamanos"]), "hogares": list(ver["tamanos"].values()),
                            "proporción": [f"{v:.1%}" for v in ver["proporciones"].values()]})
    st.dataframe(tam_tab, hide_index=True)
    interpretacion(I.i_particion(ver))
    st.markdown("**Balance de clases y estratos entre particiones**")
    var_comp = st.selectbox("Variable a comparar", [c for c in ["fies_nivel", "area", "region_natural", "departamento"] if c in df.columns])
    comp = S.composicion_por_particion(df, part, var_comp)
    st.plotly_chart(P.fig_composicion(comp, f"Composición de «{var_comp}» por partición"))
    cats = [c for c in ["fies_nivel", "area", "region_natural", "departamento", "sexo_jefe"] if c in df.columns]
    nums = [c for c in ["ingreso_pc", "gasto_pc", "edad_jefe", "n_miembros", "hacinamiento", "factor_expansion"] if c in df.columns]
    bal = S.balance_entre_particiones(df, part, cats, nums, ponderar=st.toggle("Ponderar con factor de expansión (tamaño efectivo de Kish)", value=True))
    st.dataframe(bal.round(4), hide_index=True)
    interpretacion(I.i_balance(bal))
