"""Módulo 11: Equidad y sesgos (brechas TPR/FPR/error por grupo con IC bootstrap)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from auth import barra_lateral, requerir_acceso
import tema
from fases_comun import dataset_id_activo
from tema import config_pagina, encabezado
from comun import (
    ErrorConsulta,
    consultar,
    mostrar_error_bd,
    registrar_auditoria,
)

config_pagina("Equidad")
requerir_acceso("equidad")
barra_lateral()

encabezado("Equidad y sesgos",
    "Brechas de clasificación entre grupos de hogares (TPR, FPR y error ponderado) con intervalos bootstrap.",
    "Evaluación e interpretación",
    pasos=["Elija un modelo con predicciones guardadas y la variable que define los grupos.",
           "Elija la clase a evaluar (IA moderada/severa o solo severa) y el número de réplicas bootstrap.",
           "Revise las alertas: brechas de sensibilidad mayores de 5 puntos porcentuales respecto de la población evaluada."],
    produce="Tabla y gráfico de brechas TPR/FPR/error por grupo con IC95 y alertas.",
)

SEMILLA = 42
ALFA = 0.05
UMBRAL_BRECHA_TPR = 0.05
MIN_POSITIVOS = 30
LIMITE_FILAS = 200_000

GRUPOS = {
    "sexo_jefe": "Sexo del jefe del hogar",
    "grupo_edad_jefe": "Edad del jefe del hogar",
    "grupo_educ_jefe": "Educación del jefe del hogar",
    "area": "Área de residencia (urbano / rural)",
    "region_natural": "Región natural",
    "pobreza_monetaria": "Pobreza monetaria",
    "tiene_menor5": "Presencia de menores de 5 años",
}
OBJETIVOS = {
    "ia_binaria": ("Inseguridad moderada o severa (clase >= 2, ia_binaria)", 2),
    "severa": ("Inseguridad severa (clase 3)", 3),
}
COLOR_ESTADO = {"ALERTA": "#d95f02", "sin alerta": "#1b6ca8", "n insuficiente": "#9e9e9e"}

SQL_MODELOS = """
SELECT mo.modelo_id, mo.codigo, e.nombre, e.escenario, e.esquema_validacion
FROM fsai.modelos mo
JOIN fsai.experimentos e ON e.experimento_id = mo.experimento_id
WHERE e.dataset_id = :dataset AND EXISTS (SELECT 1 FROM fsai.predicciones p WHERE p.modelo_id = mo.modelo_id)
ORDER BY mo.creado_en DESC LIMIT 200
"""

# Los grupos derivados replican las definiciones del contrato; el mapeo de nivel_educ_jefe
# (0-11, INEI) a grupos es [POR VALIDAR] frente al diccionario oficial de la ENAHO.
SQL_PREDICCIONES = """
SELECT p.hogar_uid, p.y_real, p.y_pred,
  CAST(h.factor_expansion AS double precision) AS factor_expansion,
  h.area, h.region_natural, c.sexo_jefe,
  CASE WHEN c.edad_jefe IS NULL THEN NULL
       WHEN c.edad_jefe < 30 THEN '<30' WHEN c.edad_jefe < 45 THEN '30-44'
       WHEN c.edad_jefe < 60 THEN '45-59' ELSE '60+' END AS grupo_edad_jefe,
  CASE WHEN c.nivel_educ_jefe IS NULL THEN NULL
       WHEN c.nivel_educ_jefe < 2 THEN 'sin_primaria' WHEN c.nivel_educ_jefe < 4 THEN 'primaria'
       WHEN c.nivel_educ_jefe < 6 THEN 'secundaria' ELSE 'superior' END AS grupo_educ_jefe,
  CASE WHEN c.n_menores5 IS NULL THEN NULL
       WHEN c.n_menores5 > 0 THEN 'con_menor5' ELSE 'sin_menor5' END AS tiene_menor5,
  CASE WHEN i.pobreza_monetaria IS NULL THEN NULL
       WHEN CAST(i.pobreza_monetaria AS integer) = 1 THEN 'pobre' ELSE 'no_pobre' END AS pobreza_monetaria
FROM fsai.predicciones p
JOIN fsai.hogares h ON h.hogar_uid = p.hogar_uid
LEFT JOIN fsai.condiciones_socioeconomicas c ON c.hogar_uid = p.hogar_uid
LEFT JOIN fsai.indicadores_alimentarios i ON i.hogar_uid = p.hogar_uid
WHERE p.modelo_id = :modelo_id
ORDER BY p.hogar_uid
LIMIT :limite
"""


def calcular_brechas(
    df: pd.DataFrame, col_grupo: str, umbral_clase: int, n_boot: int, semilla: int = SEMILLA
) -> pd.DataFrame:
    """TPR, FPR y tasa de error ponderadas por grupo, con IC bootstrap percentil.

    brecha_tpr = TPR del grupo - TPR de la población evaluada (ambas ponderadas). El bootstrap
    remuestrea hogares con reemplazo e ignora estrato/conglomerado (IC aproximados).
    """
    sub = df.loc[df[col_grupo].notna()].reset_index(drop=True)
    if sub.empty:
        return pd.DataFrame()
    codigos, etiquetas = pd.factorize(sub[col_grupo].astype(str), sort=True)
    k = len(etiquetas)
    y = (sub["y_real"].to_numpy() >= umbral_clase).astype(float)
    p = (sub["y_pred"].to_numpy() >= umbral_clase).astype(float)
    w = sub["factor_expansion"].to_numpy(dtype=float)
    error = (y != p).astype(float)
    unos = np.ones_like(y)

    def razon(idx: np.ndarray, num: np.ndarray, den: np.ndarray) -> np.ndarray:
        g, ww = codigos[idx], w[idx]
        n_ = np.bincount(g, weights=ww * num[idx], minlength=k)
        d_ = np.bincount(g, weights=ww * den[idx], minlength=k)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(d_ > 0, n_ / d_, np.nan)

    def global_tpr(idx: np.ndarray) -> float:
        den = float(np.sum(w[idx] * y[idx]))
        return float(np.sum(w[idx] * y[idx] * p[idx]) / den) if den > 0 else float("nan")

    def estimar(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tpr = razon(idx, y * p, y)
        fpr = razon(idx, (1 - y) * p, 1 - y)
        err = razon(idx, error, unos)
        return tpr, fpr, err, tpr - global_tpr(idx)

    todos = np.arange(len(sub))
    tpr, fpr, err, brecha = estimar(todos)
    rng = np.random.default_rng(semilla)
    muestras = [estimar(rng.integers(0, len(sub), size=len(sub))) for _ in range(n_boot)]
    pila = [np.vstack([m[i] for m in muestras]) for i in range(4)]

    def ic(matriz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return (np.nanpercentile(matriz, 100 * ALFA / 2, axis=0),
                    np.nanpercentile(matriz, 100 * (1 - ALFA / 2), axis=0))

    (t_inf, t_sup), (f_inf, f_sup), (e_inf, e_sup), (b_inf, b_sup) = (ic(m) for m in pila)
    n_hogares = np.bincount(codigos, minlength=k)
    n_pos = np.bincount(codigos, weights=y, minlength=k).astype(int)
    estado = np.where(
        n_pos < MIN_POSITIVOS, "n insuficiente",
        np.where(np.abs(brecha) > UMBRAL_BRECHA_TPR, "ALERTA", "sin alerta"),
    )
    return pd.DataFrame(
        {
            "grupo": list(etiquetas), "n_hogares": n_hogares, "n_positivos": n_pos,
            "tpr": tpr, "tpr_ic_inf": t_inf, "tpr_ic_sup": t_sup,
            "fpr": fpr, "fpr_ic_inf": f_inf, "fpr_ic_sup": f_sup,
            "error": err, "error_ic_inf": e_inf, "error_ic_sup": e_sup,
            "brecha_tpr": brecha, "brecha_ic_inf": b_inf, "brecha_ic_sup": b_sup,
            "ic_excluye_cero": (b_inf > 0) | (b_sup < 0), "estado": estado,
        }
    )


try:
    modelos = consultar(SQL_MODELOS, {"dataset": dataset_id_activo() if dataset_id_activo() is not None else -1})
except ErrorConsulta as exc:
    mostrar_error_bd(exc)
    st.stop()
except Exception:  # noqa: BLE001
    st.error("El esquema «fsai» de la base de datos no está disponible. Inicie PostgreSQL con iniciar.ps1 (o cree el esquema con sql/01_ddl_fsai.sql) y recargue la página.")
    st.stop()

if modelos.empty:
    tema.vacio("No hay predicciones guardadas en PostgreSQL para el dataset activo.", "Entrene modelos en el módulo «Entrenamiento»", "pages/06_Entrenamiento.py")
    st.stop()

etiquetas_mod = {
    int(r.modelo_id): f"#{int(r.modelo_id)} · {r.codigo} · {r.nombre} · {r.escenario} · {r.esquema_validacion}"
    for r in modelos.itertuples()
}
c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
modelo_id = int(c1.selectbox("Modelo", options=list(etiquetas_mod), format_func=etiquetas_mod.get))
clave_obj = c2.selectbox("Evento positivo", options=list(OBJETIVOS), format_func=lambda k: OBJETIVOS[k][0])
dimension = c3.selectbox("Dimensión de equidad", options=list(GRUPOS), format_func=GRUPOS.get)
n_boot = c4.slider("Réplicas bootstrap", 100, 1000, 200, step=100)
umbral_clase = OBJETIVOS[clave_obj][1]

try:
    datos = consultar(SQL_PREDICCIONES, {"modelo_id": modelo_id, "limite": LIMITE_FILAS})
except ErrorConsulta as exc:
    mostrar_error_bd(exc)
    st.stop()

if datos.empty:
    tema.vacio("El modelo elegido no guardó predicciones por hogar, y la equidad se calcula comparando esas predicciones entre grupos.",
               "Elija otro modelo en la lista o entrene guardando las predicciones en «Entrenamiento»", "pages/06_Entrenamiento.py", titulo="Este modelo no tiene predicciones guardadas")
    st.stop()
if len(datos) >= LIMITE_FILAS:
    st.warning(f"Se alcanzó el límite de {LIMITE_FILAS:,} filas; el análisis puede estar truncado.")
if datos["hogar_uid"].duplicated().any():
    st.warning("Hay hogares con más de una predicción; se conserva la primera por hogar_uid.")
    datos = datos.drop_duplicates("hogar_uid", keep="first")

resultado = calcular_brechas(datos, dimension, umbral_clase, n_boot)
# Presentación: valores de grupo legibles (U/R -> Urbano/Rural, H/M -> Hombre/Mujer); los cálculos ya terminaron con los códigos originales
from fsai import etiquetas as E  # noqa: E402
resultado = resultado.assign(grupo=[E.valor_etq(dimension, g) for g in resultado["grupo"]])
if resultado.empty:
    tema.vacio("Ningún grupo de esta dimensión tiene hogares suficientes con predicción para calcular brechas.",
               "Elija otra «Dimensión de equidad» en el selector de arriba", titulo="Sin grupos para comparar")
    st.stop()
registrar_auditoria(
    "consulta_equidad", "modelo", str(modelo_id),
    {"dimension": dimension, "evento": clave_obj, "n_boot": n_boot},
)

alertas = resultado[resultado["estado"] == "ALERTA"]
if alertas.empty:
    st.success(f"Ningún grupo supera la brecha de {100 * UMBRAL_BRECHA_TPR:.0f} pp en TPR (con n suficiente).")
else:
    st.error(
        f"{len(alertas)} grupo(s) con |brecha TPR| > {100 * UMBRAL_BRECHA_TPR:.0f} pp: "
        + ", ".join(alertas["grupo"])
        + ". Una alerta indica una diferencia en el error de clasificación del modelo; "
        "no establece su causa."
    )

st.subheader("Brecha de TPR respecto de la población evaluada")
fig = go.Figure(
    go.Bar(
        x=resultado["grupo"], y=100 * resultado["brecha_tpr"],
        marker_color=[COLOR_ESTADO[e] for e in resultado["estado"]],
        error_y={
            "type": "data", "symmetric": False,
            "array": (100 * (resultado["brecha_ic_sup"] - resultado["brecha_tpr"])).fillna(0),
            "arrayminus": (100 * (resultado["brecha_tpr"] - resultado["brecha_ic_inf"])).fillna(0),
        },
    )
)
for nivel in (100 * UMBRAL_BRECHA_TPR, -100 * UMBRAL_BRECHA_TPR):
    fig.add_hline(y=nivel, line_dash="dash", line_color="#555555")
fig.update_layout(xaxis_title=GRUPOS[dimension], yaxis_title="Brecha TPR (puntos porcentuales)",
                  margin={"t": 10})
st.plotly_chart(fig)
st.caption(
    "Líneas discontinuas: ±5 pp. Naranja: alerta; azul: sin alerta; gris: menos de "
    f"{MIN_POSITIVOS} positivos muestrales (no concluyente). Barras de error: IC bootstrap del 95 %."
)

st.subheader("TPR, FPR y tasa de error por grupo")
fig2 = go.Figure()
for col, nombre, color in (("tpr", "TPR (sensibilidad)", "#1b6ca8"), ("fpr", "FPR", "#d95f02"),
                           ("error", "Tasa de error", "#6c8ea4")):
    fig2.add_bar(
        name=nombre, x=resultado["grupo"], y=100 * resultado[col], marker_color=color,
        error_y={"type": "data", "symmetric": False,
                 "array": (100 * (resultado[f"{col}_ic_sup"] - resultado[col])).fillna(0),
                 "arrayminus": (100 * (resultado[col] - resultado[f"{col}_ic_inf"])).fillna(0)},
    )
fig2.update_layout(barmode="group", yaxis_title="Porcentaje ponderado", xaxis_title=GRUPOS[dimension],
                   margin={"t": 10})
st.plotly_chart(fig2)

st.subheader("Tabla de resultados")
vista = resultado.copy()
for col, prefijo in (("tpr", "tpr"), ("fpr", "fpr"), ("error", "error"), ("brecha_tpr", "brecha")):
    vista[col] = (100 * vista[col]).round(2)
    vista[f"{col}_ic"] = [
        "—" if pd.isna(a) or pd.isna(b) else f"[{100 * a:.2f}; {100 * b:.2f}]"
        for a, b in zip(resultado[f"{prefijo}_ic_inf"], resultado[f"{prefijo}_ic_sup"])
    ]
st.dataframe(
    vista[["grupo", "n_hogares", "n_positivos", "tpr", "tpr_ic", "fpr", "fpr_ic", "error",
           "error_ic", "brecha_tpr", "brecha_tpr_ic", "ic_excluye_cero", "estado"]],
    hide_index=True,
)
st.caption(
    "Valores en % (brecha en pp). Las tasas están ponderadas por factor_expansion y los IC "
    "ignoran estrato y conglomerado, por lo que son aproximados. La definición de grupos "
    "educativos es [POR VALIDAR] frente al diccionario ENAHO."
)
