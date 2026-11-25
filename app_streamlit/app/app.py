"""FoodSecurity AI: enrutador de la aplicación Streamlit.

Ejecución: ``streamlit run app/app.py``. Antes de autenticarse solo existe la pantalla de acceso (sin barra
lateral); después, ``st.navigation`` ofrece solo los módulos permitidos al rol, agrupados por sección. La barra
lateral incluye el selector GLOBAL de dataset (lo usan todos los módulos), y el
glosario de siglas. El Inicio es un resumen ejecutivo con cifras reales (indicadores territoriales y corridas agregadas),
la «Ruta de análisis» como tubería visual con el estado de cada paso calculado desde PostgreSQL y los artefactos.
Cada página sigue aplicando su propio control de acceso mediante ``auth.requerir_acceso``.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import tema
from auth import ACCESO_PAGINAS, barra_lateral, formulario_login, sesion_activa
from comun import cargar_indicadores_territoriales, consultar

_AUTENTICADO = sesion_activa()
st.set_page_config(
    page_title="FoodSecurity AI · Inseguridad alimentaria y vulnerabilidad nutricional",
    page_icon=str(tema.DIR_ASSETS / "logo_icono.svg"),
    layout="wide",
    initial_sidebar_state="auto" if _AUTENTICADO else "collapsed",
)

import modulos  # noqa: E402
import ruta  # noqa: E402
from fases_comun import dataset_id_activo, selector_lateral  # noqa: E402

modulos.registrar_accesos()  # roles de los módulos V2 (Glosario, Comparación de corridas, Bitácora, Fuentes de datos)
MODULOS = modulos.MODULOS

SQL_RESUMEN = """
SELECT
  (SELECT COUNT(*) FROM fsai.datasets)    AS n_datasets,
  (SELECT COUNT(*) FROM fsai.hogares)     AS n_hogares,
  (SELECT COUNT(*) FROM fsai.experimentos) AS n_experimentos,
  (SELECT COUNT(*) FROM fsai.modelos)     AS n_modelos
"""


def pagina_acceso() -> None:
    formulario_login()


def _pct(x: float, d: int = 1) -> str:
    return tema.num_es(x, d)


def _datos_resumen() -> dict:
    """Cifras reales para el resumen ejecutivo (indicadores territoriales y corridas agregadas); vacío si no hay archivos."""
    out: dict = {}
    terr = cargar_indicadores_territoriales()
    if terr is not None and len(terr):
        out["terr"] = terr
    try:
        from fsai import comparacion_corridas as CC

        tabla, origen = CC.cargar_corridas(lambda sql: consultar(sql, cache=False))
        if not tabla.empty:
            out["corridas"], out["origen_corridas"] = tabla, origen
    except Exception:  # noqa: BLE001 - sin carpeta de resultados: se omite el bloque
        pass
    return out


def _resumen_ejecutivo(datos: dict) -> None:
    """Héroe + KPIs con datos reales (nunca inventados): si faltan, se explica qué cargar."""
    terr = datos.get("terr")
    corridas = datos.get("corridas")
    if terr is None:
        tema.heroe("Aún no hay <em>indicadores territoriales</em> cargados.",
                   "Coloque la carpeta datos_derivados (o defina FSAI_DATOS_DERIVADOS) para ver aquí el panorama nacional.", "Resumen ejecutivo")
        return
    n40 = int((terr["terr_anemia_ninos_pct"] > 40).sum())
    n = len(terr)
    n_ind = sum(1 for c in terr.columns if c.startswith("terr_"))
    n_corr = 0 if corridas is None else int(corridas[["corrida", "escenario", "modelo"]].drop_duplicates().shape[0])
    tema.heroe(f"En <em>{n40} de {n}</em> departamentos, más de 4 de cada 10 niños tienen anemia.",
               "FoodSecurity AI cruza la escala FIES de la ENAHO con indicadores territoriales públicos (ENDES, OPHI, ERA5) "
               "para anticipar dónde se concentra la vulnerabilidad nutricional de los hogares.",
               "Resumen ejecutivo · ENDES 2024",
               [(str(n), "departamentos"), (str(n_ind), "indicadores territoriales"), ("5", "modelos M1-M5"), (str(n_corr) if n_corr else "—", "corridas agregadas")])

    items = []
    for col, et in (("terr_anemia_ninos_pct", "Anemia infantil"), ("terr_desnutricion_cronica_pct", "Desnutrición crónica"),
                    ("terr_mpi_incidencia_pct", "Pobreza multidimensional")):
        mx, mn = terr.loc[terr[col].idxmax()], terr.loc[terr[col].idxmin()]
        items.append({"etiqueta": et, "valor": _pct(float(mx[col])), "unidad": "% máx.", "delta": f"Más alto: {str(mx['departamento']).title()}",
                      "tendencia": "sube", "nota": f"Más bajo: {str(mn['departamento']).title()}, {_pct(float(mn[col]))} %. Curva: los 25 departamentos ordenados (no es una serie en el tiempo)",
                      "spark": sorted(terr[col].tolist()), "spark_etiqueta": f"{et}: 25 departamentos ordenados de menor a mayor"})
    if corridas is not None and "socio" in set(corridas["escenario"]):
        s = corridas[corridas["escenario"] == "socio"].sort_values("auc", ascending=False).iloc[0]
        ic = "" if s["auc_ic_inf"] != s["auc_ic_inf"] else f" · IC95 {_pct(s['auc_ic_inf'], 2)}–{_pct(s['auc_ic_sup'], 2)}"
        items.append({"etiqueta": "Mejor AUC · socioeconómico", "valor": _pct(float(s["auc"]), 2), "delta": f"{s['modelo']} · {s['corrida']}",
                      "tendencia": "igual", "nota": f"Predicción real, sin ítems FIES{ic}"})
    tema.kpi_tarjetas(items)
    tema.llamada("etica", "Lectura responsable",
                 "Los indicadores territoriales son asociaciones entre departamentos (n = 25), no causas. ENDES 2024 y MPI 2023 no coinciden con el año de la ENAHO (2025).")


MAX_FILAS_CORRIDAS = 8


def _corridas_recientes(datos: dict) -> None:
    corridas = datos.get("corridas")
    tema.seccion("Corridas registradas", "Resultados agregados de la validación cruzada anidada (solo lectura).")
    if corridas is None:
        tema.vacio("No hay corridas en PostgreSQL ni archivos resultado_*.json en «resultados_agregados».", "Entrene M1-M5 en «Entrenamiento»",
                   modulos.archivo("entrenamiento"), titulo="Sin corridas todavía")
        return
    sel = corridas[corridas["escenario"] == "socio"].sort_values("auc", ascending=False)
    total = len(sel)
    sel = sel.head(MAX_FILAS_CORRIDAS)
    filas = [{"modelo": r.modelo, "escenario": "socioeconómico (predicción real)", "corrida": r.corrida, "f1": _pct(r.f1_macro, 3),
              "auc": _pct(r.auc, 3), "brier": _pct(r.brier, 3)} for r in sel.itertuples()]
    tema.tabla_html(filas, [("modelo", "Modelo"), ("escenario", "Escenario"), ("corrida", "Corrida"), ("f1", "F1-macro"), ("auc", "AUC"), ("brier", "Brier")],
                    "Escenario «socio»: los escenarios fies/híbrido reconstruyen el objetivo y no miden predicción."
                    + (f" Se muestran las {len(sel)} mejores por AUC de {tema.num_es(total)}; el resto está en «Comparación de corridas»." if total > len(sel) else ""), ("f1", "auc", "brier"), apilar=True)
    st.page_link(modulos.archivo("comparacion_corridas"), label="Comparar corridas con intervalos", icon=":material/difference:")


def _ruta_de_analisis(rol: str) -> None:
    st.subheader("Ruta de análisis")
    st.caption("Orden recomendado de trabajo. El estado de cada paso se calcula desde PostgreSQL y los artefactos de entrenamiento.")
    estados = ruta.estado_ruta(dataset_id_activo(), st.session_state.get("auth_usuario"))
    sig = ruta.siguiente_paso(estados, ACCESO_PAGINAS, rol)
    if sig:
        if st.button(f"Continuar con el paso siguiente: {modulos.titulo(sig)}", type="primary", icon=":material/arrow_forward:",
                     help="Abre el primer paso de la ruta que ya puede ejecutarse con los datos actuales."):
            st.switch_page(modulos.archivo(sig))
    else:
        tema.llamada("ok", "Sin pasos pendientes", "No hay pasos pendientes con su rol. Revise los resultados en Reportes o Equidad y sesgos.")
    tema.ruta_analisis(estados, rol, ACCESO_PAGINAS)


def pagina_inicio() -> None:
    rol = st.session_state["auth_rol"]
    tema.encabezado(
        "Inicio",
        "",
        "Plataforma",
        pasos=[
            "Elija el dataset activo en la barra lateral (todos los módulos lo usan).",
            "Siga la ruta de análisis de izquierda a derecha, o use el botón «Continuar con el paso siguiente».",
            "Genere el informe científico en el módulo «Reportes» cuando tenga resultados.",
        ],
        produce="Una visión del avance del proyecto y acceso directo a cada módulo.",
        compacto=True,
    )
    datos = _datos_resumen()
    _resumen_ejecutivo(datos)
    tema.seccion("Estado de la base de datos", "Conteos reales de PostgreSQL; sin conexión se indica qué falta.")
    try:
        fila = consultar(SQL_RESUMEN, cache=False).iloc[0]
        tema.tira_estadisticos([(tema.num_es(fila["n_datasets"]), "datasets"), (tema.num_es(fila["n_hogares"]), "hogares"),
                                (tema.num_es(fila["n_experimentos"]), "experimentos"), (tema.num_es(fila["n_modelos"]), "modelos")])
        if int(fila["n_hogares"]) == 0:
            tema.vacio("La base de datos no contiene hogares todavía; la prevalencia FIES necesita el dataset cargado.",
                       "Importe ENAHO en «Carga de datos»", modulos.archivo("importacion_enaho"), titulo="Base sin hogares")
    except Exception:  # noqa: BLE001 - sin conexión (ErrorConsulta) o esquema fsai no creado
        tema.vacio("No hay conexión con PostgreSQL o el esquema «fsai» no existe. Los mapas territoriales y las corridas agregadas siguen disponibles; "
                   "la prevalencia FIES requiere el dataset cargado.", "Inicie PostgreSQL con iniciar.ps1 (o cree el esquema con sql/01_ddl_fsai.sql) y recargue",
                   titulo="Base de datos no disponible")
    _ruta_de_analisis(rol)
    _corridas_recientes(datos)
    tema.pie_pagina()


def _barra_lateral_router() -> None:
    st.session_state.pop("_fsai_barra_en_router", None)
    st.session_state.pop("_fsai_sel_en_router", None)
    barra_lateral()  # usuario/rol y cierre de sesión (una vez por ejecución)
    st.session_state["_fsai_barra_en_router"] = True
    selector_lateral()  # dataset activo global
    st.session_state["_fsai_sel_en_router"] = True
    tema.glosario_lateral()


def main() -> None:
    tema.aplicar_estilos(ocultar_lateral=not _AUTENTICADO)
    if not _AUTENTICADO:
        st.session_state.pop("_fsai_barra_en_router", None)
        pagina = st.navigation([st.Page(pagina_acceso, title="Acceso · FoodSecurity AI", url_path="acceso", default=True)])
        pagina.run()
        return

    rol = st.session_state["auth_rol"]
    st.logo(tema.logo_marca_data_uri(), size="large", icon_image=tema.logo_marca_data_uri(True))
    secciones: dict[str, list] = {"Plataforma": [st.Page(pagina_inicio, title="Inicio", url_path="inicio", default=True, icon=":material/home:")]}
    for seccion in tema.SECCIONES:
        paginas = [
            st.Page(archivo, title=titulo, icon=icono, url_path=Path(archivo).stem.split("_", 1)[1].lower())
            for clave, sec, titulo, _d, archivo, icono in MODULOS
            if sec == seccion and rol in ACCESO_PAGINAS[clave]
        ]
        if paginas:
            secciones[seccion] = paginas
    navegacion = st.navigation(secciones, expanded=True)
    _barra_lateral_router()
    navegacion.run()


main()
