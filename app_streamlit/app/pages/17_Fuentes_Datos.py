"""Fuentes de datos: procedencia, licencia, granularidad, año, estado y huella (sha256 corto) de cada fuente pública usada, más
las fuentes V1 (ENAHO, FAOSTAT, WDI), las limitaciones conocidas y las fuentes que fallaron o se descartaron. Solo lectura.

Lee ``datos_derivados/MANIFIESTO_ADICIONALES.json``; no consulta la base de datos ni la red.
"""
from __future__ import annotations

import streamlit as st

import modulos
import tema
from auth import barra_lateral, requerir_acceso
from comun import cargar_json_derivado
from tema import config_pagina, encabezado

modulos.registrar_accesos()
config_pagina("Fuentes de datos")
usuario, rol = requerir_acceso("fuentes_datos")
barra_lateral()

encabezado(
    "Fuentes de datos",
    "De dónde sale cada cifra: origen, licencia, granularidad, año y estado de las fuentes públicas, con sus límites.",
    "Datos",
    pasos=["Filtre por familia de fuente o por estado.",
           "Verifique la licencia y la huella sha256 antes de redistribuir cualquier archivo.",
           "Lea las limitaciones: el desfase temporal y los 25 valores por indicador condicionan las conclusiones."],
    produce="Tabla de procedencia de datos; no descarga ni modifica nada.",
    clave="fuentes_datos",
)

manifiesto = cargar_json_derivado("MANIFIESTO_ADICIONALES.json")
indic = cargar_json_derivado("indicadores_territoriales_departamento.manifiesto.json")
fuentes = manifiesto.get("fuentes", [])


def familia(i: str) -> str:
    for pref, nombre in (("endes", "ENDES (INEI)"), ("nasa_power", "NASA POWER (descartada)"), ("openmeteo", "Open-Meteo / ERA5"),
                         ("geojson", "Límites (GeoJSON)"), ("ophi", "OPHI · MPI"), ("noaa", "NOAA · ONI")):
        if i.startswith(pref):
            return nombre
    return "Otras"


def licencia_clara(txt: str) -> bool:
    t = txt.lower()
    return not ("no declarada" in t or "sin licencia" in t)


if not fuentes:
    tema.vacio("No se encontró datos_derivados/MANIFIESTO_ADICIONALES.json (o no es legible). La procedencia de los indicadores no puede mostrarse sin él.",
               "Genere el manifiesto con scripts/descargar_fuentes_publicas.py o defina FSAI_DATOS_DERIVADOS", titulo="Manifiesto de fuentes no disponible")
else:
    fam_n = {}
    for f in fuentes:
        fam_n.setdefault(familia(f["id"]), []).append(f)
    n_ok = sum(1 for f in fuentes if f.get("estado") == "ok")
    n_fallo = len(fuentes) - n_ok
    n_sin_lic = sum(1 for f in fuentes if not licencia_clara(str(f.get("licencia", ""))))
    tema.kpi_tarjetas([
        {"etiqueta": "Fuentes registradas", "valor": str(len(fuentes)), "nota": f"{len(fam_n)} familias · manifiesto del {str(manifiesto.get('generado', ''))[:10]}"},
        {"etiqueta": "Descargadas (ok)", "valor": str(n_ok), "nota": "Con huella sha256 verificable"},
        {"etiqueta": "Con fallo en la última corrida", "valor": str(n_fallo), "nota": "Ver también «Fuentes descartadas» abajo"},
        {"etiqueta": "Licencia no declarada", "valor": str(n_sin_lic), "nota": "Citar la fuente y verificar antes de redistribuir"},
    ])

    tema.seccion("Fuentes del proyecto", "Cada fila del manifiesto de descargas; los archivos se tratan como datos no confiables (solo lectura).")
    opciones = ["Todas"] + sorted(fam_n)
    col_a, col_b = st.columns(2)
    sel_fam = col_a.selectbox("Familia", opciones, key="fuentes_fam")
    sel_est = col_b.selectbox("Estado", ["Todos", "ok", "fallo"], key="fuentes_est")
    filas = []
    for f in fuentes:
        if sel_fam != "Todas" and familia(f["id"]) != sel_fam:
            continue
        if sel_est != "Todos" and f.get("estado") != sel_est:
            continue
        filas.append({"id": f["id"], "fam": familia(f["id"]), "url": f.get("url", ""), "lic": f.get("licencia", "") or "—", "gran": f.get("granularidad", "") or "—",
                      "anio": str(f.get("anios", "—")), "est": ("OK" if f.get("estado") == "ok" else "FALLO") + (f" ({f.get('motivo')})" if f.get("estado") != "ok" and f.get("motivo") else ""),
                      "sha": str(f.get("sha256", ""))[:8] or "—"})
    if filas:
        tema.tabla_html(filas, [("id", "Fuente"), ("fam", "Familia"), ("url", "URL"), ("lic", "Licencia"), ("gran", "Granularidad"), ("anio", "Año"), ("est", "Estado"), ("sha", "sha256")],
                        f"{len(filas)} de {len(fuentes)} fuentes · la licencia se transcribe tal como la declara el manifiesto, sin interpretarla.", enlaces=("url",), mono=("sha", "id"))
    else:
        tema.vacio("Ninguna fuente coincide con el filtro.", "Amplíe el filtro de familia o estado", titulo="Sin coincidencias")

# ------------------------------------------------------------------ fuentes V1
tema.seccion("Fuentes de la versión 1", "Base del análisis FIES; su licencia no se verifica en este repositorio.")
tema.tabla_html(
    [
        {"n": "ENAHO 2025 (INEI)", "d": "Microdatos de hogares: módulo FIES (P130_1..8), vivienda, empleo, Sumaria", "g": "Hogar; anual 2025", "l": "INEI datos abiertos; citar la fuente (verificar términos vigentes)", "e": "Fuente del objetivo FIES"},
        {"n": "FAO · FAOSTAT / FIES agregado", "d": "Estimaciones agregadas de inseguridad alimentaria para referencia y contraste", "g": "País; series anuales", "l": "No verificada aquí; consultar los términos de FAOSTAT", "e": "Solo referencia (V1)"},
        {"n": "Banco Mundial · WDI", "d": "Indicadores de desarrollo a nivel país como contexto", "g": "País; series anuales", "l": "No verificada aquí; consultar los términos de WDI", "e": "Solo referencia (V1)"},
    ],
    [("n", "Fuente"), ("d", "Contenido"), ("g", "Granularidad"), ("l", "Licencia"), ("e", "Uso")], "Fuentes heredadas de V1; se listan sin afirmar licencias no comprobadas.")

# ------------------------------------------------------------------ limitaciones
tema.seccion("Limitaciones que condicionan la lectura")
aviso_t = indic.get("aviso_temporal") or "ENDES 2024 y MPI (ENDES 2023) NO son de 2025 (año de la ENAHO); precipitación y ONI sí corresponden a 2025."
tema.llamada("aviso", "Desfase temporal", str(aviso_t))
tema.llamada("aviso", "Solo 25 valores distintos por indicador",
             "Cada indicador territorial aporta un valor por departamento: al unirlo a los hogares, todos los hogares de un departamento comparten el mismo valor. "
             "Eso limita lo que un modelo puede aprender de él y obliga a validar dejando fuera departamentos completos (LODO).")
tema.llamada("info", "Sin fuga del objetivo", str(indic.get("aviso_fuga") or "Ningún indicador territorial se deriva de ENAHO/FIES ni del objetivo."))
tema.llamada("etica", "Uso responsable", "Las asociaciones entre indicadores no son causales ni justifican negar ayudas o programas sociales a un hogar.")

tema.seccion("Fuentes descartadas o fallidas", "Se listan para no reintentarlas ni afirmarlas.")
tema.tabla_html(
    [
        {"f": "API CKAN de datosabiertos.gob.pe", "m": "Intento fallido; no se usa"}, {"f": "MIDIS REDinforma", "m": "Intento fallido; no se usa"},
        {"f": "UNICEF JME", "m": "Intento fallido; no se usa"}, {"f": "IMF (xls de commodities)", "m": "Intento fallido; no se usa"},
        {"f": "CHIRPS", "m": "Intento fallido; no se usa"}, {"f": "SENAMHI", "m": "Intento fallido; no se usa"},
        {"f": "Precios WFP", "m": "Solo Lima y terminan en 2022; descartada"},
        {"f": "NASA POWER (precipitación)", "m": "Descargada 25/25 puntos pero su serie de 2025 muestra un salto no verosímil y no se correlaciona con ERA5 (Spearman ≈ -0,01): solo diagnóstico, no alimenta ningún rasgo"},
    ],
    [("f", "Fuente"), ("m", "Motivo")], "Según docs/FUENTES_ADICIONALES.md. El manifiesto de descargas solo registra las que se descargaron.")
tema.pie_pagina()
