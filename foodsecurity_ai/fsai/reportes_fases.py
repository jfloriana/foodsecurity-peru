"""reportes_fases.py - Informe de nivel de manuscrito (PDF, Word, Excel + figuras PNG a 600 DPI) de una corrida de FoodSecurity AI.

Estructura (numeración fija; una sección deseleccionada simplemente no aparece):
  1 Datos y diseño · 2 Métodos · 3 Resultados de desempeño · 4 Comparación estadística · 5 Interpretabilidad ·
  6 Equidad · 7 Limitaciones y uso responsable (siempre incluida) · 8 Reproducibilidad.

Las cifras salen SIEMPRE de PostgreSQL (hogares, FIES, experimentos, métricas, predicciones, explicaciones) y de los
artefactos de la corrida (JSON de CV anidada y predicciones de Validación/Prueba). Nada se inventa: si falta un
insumo, la sección dice «sin datos disponibles». Este módulo NUNCA evalúa la Prueba (15 %): solo usa las
predicciones de Prueba si ya fueron generadas en el módulo de Comparación estadística; de lo contrario usa las de
Validación y lo declara en cada tabla y figura.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import logging
import platform
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

from fsai import config as C
from fsai import etiquetas as E
from fsai import explainability as XP
from fsai import evaluacion as EV
from fsai import interpretacion as I
from fsai import perfil_vulnerabilidad as PV
from fsai import plots as P
from fsai import statistical_tests as ST
from fsai.fairness import brechas_por_grupo
from fsai.nested_cv import METRICAS_AGREGADAS
from fsai.preprocessing import preparar_matriz
from fsai.report_generator import exportar_excel, exportar_pdf, exportar_word
from fsai.splits import cargar_particion

logger = logging.getLogger(__name__)

SECCIONES = ["datos", "perfil", "metodos", "resultados", "comparacion", "interpretabilidad", "equidad", "reproducibilidad"]
ETIQUETAS_SECCION = {
    "datos": "1. Datos y diseño muestral",
    "perfil": "1b. Perfil descriptivo de hogares vulnerables",
    "metodos": "2. Métodos",
    "resultados": "3. Resultados de desempeño",
    "comparacion": "4. Comparación estadística",
    "interpretabilidad": "5. Interpretabilidad",
    "equidad": "6. Equidad",
    "reproducibilidad": "8. Reproducibilidad",
}
HOJAS = {"datos": "1_Datos", "perfil": "1b_Perfil", "metodos": "2_Metodos", "resultados": "3_Resultados", "comparacion": "4_Comparacion",
         "interpretabilidad": "5_Interpretabilidad", "equidad": "6_Equidad", "limitaciones": "7_Limitaciones", "reproducibilidad": "8_Reproducibilidad",
         "anexo": "9_Diccionario"}
N_DEP_VARIABLES = 8          # variables principales con PDP/ALE en el informe
N_DEP_MUESTRA = 2000         # tamaño de la submuestra ponderada para PDP/ALE (modo estándar)
N_DEP_MUESTRA_DETALLE = 5000  # tamaño con «recalcular con más detalle»
MODELOS_NOMBRE = {"M1": "Regresión logística multinomial", "M2": "Random Forest", "M3": "CatBoost",
                  "M4": "Stacking CatBoost + RF + LR", "M5": "Autoencoder + XGBoost"}
ESQUEMAS_TEXTO = {
    "V1": "GroupKFold(5) por hogar_uid, repetido con semillas SEED+rep (bloques pareados)",
    "V1c": "sensibilidad de V1 agrupando por conglomerado (conglome)",
    "V2": "validación geográfica: dejar fuera un departamento (lodo) o una macrorregión (macro)",
    "V3": "partición estratificada urbano/rural con métricas por estrato",
    "V4": "temporal 2024 -> 2025 (requiere columna anio con >= 2 años comparables)",
    "V5": "ponderación con factor_expansion (modo de métricas aplicable a cualquier esquema; se reporta ponderado y no ponderado)",
}
METRICAS_CLAVE = ["f1_macro", "balanced_accuracy", "roc_auc_ovr", "pr_auc_macro", "mcc", "recall_severa", "brier", "ece"]
ETQ_METRICA = {"f1_macro": "F1-macro", "balanced_accuracy": "Exactitud balanceada", "roc_auc_ovr": "AUC ROC (OvR)", "pr_auc_macro": "AUC PR macro",
               "mcc": "MCC", "recall_severa": "Recall IA severa", "brier": "Brier", "ece": "ECE", "accuracy": "Exactitud",
               "precision_macro": "Precisión macro", "recall_macro": "Recall macro"}
PAQUETES = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "catboost", "optuna", "shap", "lime", "torch", "statsmodels",
            "imbalanced-learn", "SQLAlchemy", "psycopg2-binary", "matplotlib", "plotly", "streamlit", "python-docx", "openpyxl", "reportlab"]
Q_NEMENYI_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


# ------------------------------ utilidades ------------------------------
def dpi_png(ruta: Path) -> tuple[float, float] | None:
    """DPI declarado en el PNG (chunk pHYs), para verificar los 600 DPI."""
    from PIL import Image
    with Image.open(ruta) as im:
        d = im.info.get("dpi")
    return (round(d[0]), round(d[1])) if d else None


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with Path(ruta).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _json(ruta: Path) -> dict[str, Any]:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def corridas_disponibles() -> list[Path]:
    """Directorios de corrida con resultados (artefactos ``fases`` y ``real/fases``), más recientes primero."""
    raices = [C.DIR_FASES, C.ARTIFACTS_DIR / "real" / "fases"]
    out = []
    for raiz in raices:
        if raiz.is_dir():
            out += [p for p in raiz.iterdir() if p.is_dir() and any(p.glob("resultado_*.json"))]
    return sorted(out, key=lambda q: q.stat().st_mtime, reverse=True)


def _vacio(texto: str) -> dict[str, Any]:
    return {"tipo": "vacio", "texto": texto}


def _p(texto: str) -> dict[str, Any]:
    return {"tipo": "p", "texto": texto}


def _presentar(df: pd.DataFrame) -> pd.DataFrame:
    """Capa de presentación: etiquetas legibles (con el código original entre corchetes) en las columnas ``variable``/``grupo``.

    Devuelve una COPIA; nunca altera las columnas de entrenamiento.
    """
    if df is None or not isinstance(df, pd.DataFrame) or "variable" not in df.columns:
        return df
    out = df.copy()
    orig = out["variable"].astype(str)
    if "grupo" in out.columns:
        out["grupo"] = [E.valor_etq(v, g) if v in E.DICCIONARIO else g for v, g in zip(orig, out["grupo"])]
    out["variable"] = [E.etq_codigo(v) for v in orig]
    return out


def _tabla(titulo: str, df: pd.DataFrame, pie: Any = None, max_filas: int | None = None) -> dict[str, Any]:
    return {"tipo": "tabla", "titulo": titulo, "df": _presentar(df), "pie": pie, "max_filas": max_filas}


class _Ctx:
    """Estado compartido durante la construcción del informe."""

    def __init__(self, engine, dataset_id, df, dir_run, escenario, codigos, dir_salida, n_boot, shap):
        self.engine, self.dataset_id, self.df = engine, dataset_id, df
        self.dir_run, self.escenario, self.codigos = dir_run, escenario, codigos
        self.dir_salida, self.n_boot, self.shap = dir_salida, n_boot, shap
        self.cfg: dict[str, Any] = {}
        self.res: dict[str, dict[str, Any]] = {}
        self.figuras: list[Path] = []
        self.hechos: dict[str, Any] = {}     # cifras para el resumen ejecutivo
        self.preds: dict[str, tuple[dict[str, np.ndarray], str]] = {}

    def figura(self, nombre: str, fig, titulo: str, pie: Any, ancho: str, datos: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """``datos`` (opcional): tablas EXACTAS que generan la figura, para el libro Excel (ver ``fsai.excel_graficos``)."""
        ruta = P.guardar_mpl(fig, self.dir_salida / f"{nombre}.png")
        self.figuras.append(ruta)
        return {"tipo": "figura", "titulo": titulo, "ruta": ruta, "pie": pie, "ancho": ancho, "datos": datos or []}

    def sql(self, consulta: str, **params) -> pd.DataFrame:
        from sqlalchemy import text
        with self.engine.connect() as conn:
            r = conn.execute(text(consulta), params)
            return pd.DataFrame([tuple(f) for f in r.fetchall()], columns=list(r.keys()))


# ------------------------------ datos de las figuras para el Excel ------------------------------
def _d(titulo: str, df: pd.DataFrame, chart: dict[str, Any] | None = None, color: str | None = None) -> dict[str, Any]:
    """Tabla de datos de una figura (para el Excel): título, DataFrame exacto, especificación del gráfico nativo y formato."""
    return {"titulo": titulo, "df": df, "chart": chart, "color": color}


def _datos_ic(t: pd.DataFrame, col_valor: str, lo: str, hi: str) -> pd.DataFrame:
    """Copia de ``t`` con las columnas auxiliares de barras de error (semiancho superior e inferior)."""
    out = t.copy()
    out["semiancho sup. (aux. gráfico)"] = (out[hi] - out[col_valor]).clip(lower=0)
    out["semiancho inf. (aux. gráfico)"] = (out[col_valor] - out[lo]).clip(lower=0)
    return out


def _serie_ic(col: str, nombre: str | None = None) -> dict[str, Any]:
    return {"col": col, "nombre": nombre or col, "err_mas": "semiancho sup. (aux. gráfico)", "err_menos": "semiancho inf. (aux. gráfico)"}


def _datos_prevalencia(t: pd.DataFrame, col: str, etq: str) -> list[dict[str, Any]]:
    tp = t.rename(columns={"ic_inf": "IC95 inf", "ic_sup": "IC95 sup"}).copy()
    tp["grupo"] = [E.valor_etq(col, g) for g in tp["grupo"]]
    tp = _datos_ic(tp, "prevalencia", "IC95 inf", "IC95 sup")
    return [_d(f"Prevalencia ponderada de IA moderada/severa por {etq.lower()}", tp,
               {"tipo": "barra_h" if len(tp) > 8 else "barra", "titulo": f"Prevalencia por {etq.lower()} (IC95)", "cat": "grupo", "series": [_serie_ic("prevalencia", "Prevalencia")],
                "y_titulo": "Prevalencia", "alto_cm": 9 if len(tp) <= 8 else 14})]


def _datos_desempeno(sel: pd.DataFrame, metricas: list[str]) -> list[dict[str, Any]]:
    largo = sel.assign(metrica_legible=[ETQ_METRICA.get(m, m) for m in sel["metrica"]]).rename(columns={"metrica": "métrica (código)", "metrica_legible": "métrica", "ic_inf": "IC95 inf", "ic_sup": "IC95 sup"})
    largo = largo[["modelo", "métrica", "métrica (código)", "media", "DE", "IC95 inf", "IC95 sup", "n"]]
    filas, modelos = [], sorted(sel["modelo"].unique())
    for m in metricas:
        fila: dict[str, Any] = {"métrica": ETQ_METRICA.get(m, m)}
        for c in modelos:
            r = sel[(sel["modelo"] == c) & (sel["metrica"] == m)]
            if len(r):
                r = r.iloc[0]
                fila.update({f"{c} media": r["media"], f"{c} +IC95": max(r["ic_sup"] - r["media"], 0.0), f"{c} −IC95": max(r["media"] - r["ic_inf"], 0.0)})
        filas.append(fila)
    ancho = pd.DataFrame(filas)
    series = [{"col": f"{c} media", "nombre": c, "err_mas": f"{c} +IC95", "err_menos": f"{c} −IC95"} for c in modelos]
    return [_d("Datos exactos de la figura (CV anidada, métricas ponderadas)", largo),
            _d("Tabla derivada para el gráfico: media y semianchos del IC95 por modelo", ancho,
               {"tipo": "barra", "titulo": "Desempeño por modelo (media e IC95)", "cat": "métrica", "series": series, "y_titulo": "Valor de la métrica", "ancho_cm": 20})]


def _matriz_df(cm: np.ndarray, normalizada: bool) -> pd.DataFrame:
    m = np.asarray(cm, dtype=float)
    if normalizada:
        s = m.sum(axis=1, keepdims=True)
        m = m / np.where(s == 0, 1, s)
    else:
        m = m.astype(int)
    out = pd.DataFrame(m, columns=[f"predicho {k}" for k in range(m.shape[1])])
    out.insert(0, "nivel FIES real \\ predicho", [f"real {k}" for k in range(m.shape[0])])
    return out


def _datos_curvas(curvas: dict[str, tuple], x_nombre: str, y_nombre: str, titulo: str, diagonal: bool) -> list[dict[str, Any]]:
    """Puntos exactos de curvas ROC/PR: dos columnas (x, y) por modelo. ``curvas[modelo] = (x, y, resumen)``."""
    cols: dict[str, pd.Series] = {}
    series = []
    for c, (xv, yv, r) in curvas.items():
        cols[f"{c} {x_nombre}"] = pd.Series(np.asarray(xv, dtype=float))
        cols[f"{c} {y_nombre}"] = pd.Series(np.asarray(yv, dtype=float))
        series.append({"x": f"{c} {x_nombre}", "col": f"{c} {y_nombre}", "nombre": f"{c} ({r:.3f})"})
    if diagonal:
        cols["referencia x"], cols["referencia y"] = pd.Series([0.0, 1.0]), pd.Series([0.0, 1.0])
        series.append({"x": "referencia x", "col": "referencia y", "nombre": "azar", "punteada": True})
    return [_d(titulo, pd.DataFrame(cols),
               {"tipo": "dispersion", "titulo": titulo, "x_titulo": x_nombre, "y_titulo": y_nombre, "series": series, "ylim": (0, 1.02)})]


def _datos_calibracion(curvas: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    cols: dict[str, pd.Series] = {}
    series = []
    for c, d in curvas.items():
        if len(d):
            cols[f"{c} prob. predicha"] = d["p_predicha"].reset_index(drop=True)
            cols[f"{c} frecuencia observada"] = d["frecuencia"].reset_index(drop=True)
            cols[f"{c} n hogares"] = d["n"].reset_index(drop=True)
            series.append({"x": f"{c} prob. predicha", "col": f"{c} frecuencia observada", "nombre": c, "marcador": True})
    cols["referencia x"], cols["referencia y"] = pd.Series([0.0, 1.0]), pd.Series([0.0, 1.0])
    series.append({"x": "referencia x", "col": "referencia y", "nombre": "calibración perfecta", "punteada": True})
    return [_d("Curvas de calibración de la clase IA severa (10 cuantiles de probabilidad)", pd.DataFrame(cols),
               {"tipo": "dispersion", "titulo": "Calibración de P(IA severa)", "x_titulo": "Probabilidad predicha", "y_titulo": "Frecuencia observada", "series": series})]


def _datos_shap(imp: pd.DataFrame) -> list[dict[str, Any]]:
    tot = float(imp["importancia"].sum()) or 1.0
    d = imp.assign(etiqueta=[E.etq(v) for v in imp["variable"]]).rename(columns={"variable": "variable (código)", "etiqueta": "variable"})
    d["participación (%)"] = 100 * d["importancia"] / tot
    d["acumulada (%)"] = d["participación (%)"].cumsum()
    d = d[["variable", "variable (código)", "importancia", "participación (%)", "acumulada (%)"]]
    return [_d("|SHAP| medio por variable (orden descendente)", d,
               {"tipo": "barra_h", "titulo": "Importancia SHAP: 15 variables principales", "cat": "variable", "series": [{"col": "importancia", "nombre": "|SHAP| medio"}], "max_filas": 15, "alto_cm": 11})]


def _datos_dependencia(curvas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    t = XP.tabla_dependencia(curvas)
    t["valor"] = [E.valor_etq(v, k) if isinstance(k, str) else float(k) for v, k in zip(t["variable"], t["valor"])]
    t["variable"] = [E.etq_codigo(str(v)) for v in t["variable"]]
    return [_d("Valores numéricos de PDP y ALE de las variables de la figura", t)]


def _datos_escenarios(te: pd.DataFrame) -> list[dict[str, Any]]:
    d = te.rename(columns={"ic_inf": "IC95 inf", "ic_sup": "IC95 sup"}).copy()
    d.insert(0, "escenario · modelo", [f"{a} · {b}" for a, b in zip(d["escenario"], d["modelo"])])
    d = _datos_ic(d, "media", "IC95 inf", "IC95 sup")
    return [_d("F1-macro ponderado por escenario y modelo (CV anidada)", d,
               {"tipo": "barra", "titulo": "F1-macro ponderado por escenario y modelo (IC95)", "cat": "escenario · modelo", "series": [_serie_ic("media", "F1-macro")], "y_titulo": "F1-macro", "ancho_cm": 20})]


def _datos_dc(fr: dict[str, Any]) -> list[dict[str, Any]]:
    rangos = pd.DataFrame({"modelo": list(dict(fr["rangos_medios"])), "rango medio": [float(v) for v in dict(fr["rangos_medios"]).values()]}).sort_values("rango medio")
    dc = pd.DataFrame([{"concepto": "diferencia crítica (Nemenyi, α = 0,05)", "valor": float(fr["diferencia_critica"])}])
    return [_d("Rangos medios de Friedman (1 = mejor)", rangos, {"tipo": "barra", "titulo": "Rango medio por modelo (menor es mejor)", "cat": "modelo", "series": [{"col": "rango medio", "nombre": "Rango medio"}], "y_titulo": "Rango medio"}),
            _d("Diferencia crítica", dc)]


def _datos_equidad(t: pd.DataFrame) -> list[dict[str, Any]]:
    d = _presentar(t.rename(columns={"grupo_col": "variable", "ic_inf": "IC95 inf", "ic_sup": "IC95 sup"}))
    obj = d[d["objetivo"] == "ia_binaria"] if "objetivo" in d.columns else d
    obj = obj if len(obj) else d
    g = _datos_ic(obj.assign(**{"etiqueta": [f"{v} · {gr}" for v, gr in zip(obj["variable"], obj["grupo"])]}), "brecha_tpr", "IC95 inf", "IC95 sup")
    g["semiancho sup. (aux. gráfico)"] = g["semiancho sup. (aux. gráfico)"].fillna(0.0)
    g["semiancho inf. (aux. gráfico)"] = g["semiancho inf. (aux. gráfico)"].fillna(0.0)
    return [_d("Brechas de TPR, FPR y error por grupo (tabla completa de la figura)", d.drop(columns=["es_referencia"], errors="ignore")),
            _d("Tabla derivada para el gráfico: brecha de TPR por grupo", g[["etiqueta", "n", "brecha_tpr", "IC95 inf", "IC95 sup", "semiancho sup. (aux. gráfico)", "semiancho inf. (aux. gráfico)"]],
               {"tipo": "barra_h", "titulo": "Brecha de TPR frente al grupo de referencia (IC95)", "cat": "etiqueta", "series": [_serie_ic("brecha_tpr", "Brecha de TPR")], "alto_cm": 15})]


# ------------------------------ 1. Datos y diseño ------------------------------
def prevalencia_ic(df: pd.DataFrame, mascara: pd.Series | None = None) -> tuple[float, float, float, int]:
    """Prevalencia ponderada de IA moderada/severa (nivel >= 2) con IC95 por linealización de Taylor.

    Trata los conglomerados (``conglome``) como unidades primarias de muestreo y NO incorpora la estratificación del
    diseño: el IC es aproximado. Devuelve (p, ic_inf, ic_sup, n).
    """
    d = df if mascara is None else df[mascara]
    d = d[d["fies_nivel"].notna() & d["factor_expansion"].notna()]
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    w = d["factor_expansion"].to_numpy(float)
    y = (d["fies_nivel"].to_numpy(float) >= 2).astype(float)
    sw = w.sum()
    p = float((w * y).sum() / sw)
    z = w * (y - p) / sw
    if "conglome" in d.columns and d["conglome"].nunique() > 1:
        zc = pd.Series(z).groupby(d["conglome"].to_numpy()).sum().to_numpy()
        m = len(zc)
        var = m / (m - 1) * float((zc ** 2).sum())
    else:
        var = n / (n - 1) * float((z ** 2).sum()) if n > 1 else float("nan")
    se = float(np.sqrt(var)) if var == var else float("nan")
    return p, max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se), n


def _tabla_prevalencia(df: pd.DataFrame, col: str, min_n: int = 30) -> pd.DataFrame:
    filas = []
    for g, sub in df.groupby(col, dropna=True):
        p, lo, hi, n = prevalencia_ic(sub)
        filas.append({"grupo": str(g), "n": n, "prevalencia": p, "ic_inf": lo, "ic_sup": hi,
                      "nota": "n < 30: inestable" if n < min_n else ""})
    return pd.DataFrame(filas).sort_values("prevalencia", ascending=False).reset_index(drop=True) if filas else pd.DataFrame()


def _seccion_datos(x: _Ctx) -> list[dict[str, Any]]:
    df = x.df
    b: list[dict[str, Any]] = []
    meta = x.sql("SELECT dataset_id, nombre, fuente, anio, version, hash_archivo, n_filas, creado_en FROM fsai.datasets WHERE dataset_id = :d", d=int(x.dataset_id))
    cnt = x.sql("SELECT COUNT(*) AS total, COUNT(r.fies_nivel) AS con_fies FROM fsai.hogares h LEFT JOIN fsai.respuestas_fies r ON r.hogar_uid = h.hogar_uid WHERE h.dataset_id = :d", d=int(x.dataset_id))
    total, con_fies = (int(cnt.iloc[0]["total"]), int(cnt.iloc[0]["con_fies"])) if len(cnt) else (len(df), len(df))
    n_conglo = int(df["conglome"].nunique()) if "conglome" in df else None
    if len(meta):
        m = meta.iloc[0]
        b.append(_p(f"Fuente: {m['fuente']} (dataset «{m['nombre']}», año {m['anio']}, versión {m['version'] or 'sin versión'}). "
                    f"Registrado en PostgreSQL (fsai.datasets #{int(m['dataset_id'])}) con hash de archivo {str(m['hash_archivo'])[:16]}…."))
    else:
        b.append(_vacio("Sin datos disponibles: el dataset activo no está registrado en fsai.datasets (se usó un archivo tabular)."))
    x.hechos.update(n=len(df), n_total=total, n_conglo=n_conglo)
    fies = df["fies_nivel"].dropna()
    excl = pd.DataFrame([
        {"concepto": "Hogares registrados en fsai.hogares", "n": total},
        {"concepto": "Hogares con nivel FIES calculado", "n": con_fies},
        {"concepto": "Hogares sin nivel FIES (excluidos del análisis)", "n": max(total - con_fies, 0)},
        {"concepto": "Hogares en el dataset analítico", "n": len(df)},
        {"concepto": "Conglomerados distintos", "n": n_conglo if n_conglo is not None else "sin datos"},
    ])
    b.append(_tabla("Tamaño muestral y exclusiones", excl, {"que_muestra": f"El dataset analítico tiene {len(df):,} hogares; {max(total - con_fies, 0):,} hogares registrados no tienen nivel FIES y quedan excluidos.",
                                                           "como_leerlo": "Las exclusiones son hogares sin respuesta completa a los ocho ítems FIES.", "limitacion": "No se imputó el objetivo; los excluidos podrían diferir sistemáticamente de los incluidos."}))
    p, lo, hi, n = prevalencia_ic(df)
    x.hechos.update(prev=p, prev_lo=lo, prev_hi=hi)
    b.append(_p(f"Prevalencia nacional ponderada de inseguridad alimentaria moderada o severa (FIES nivel 2-3): {100 * p:.1f} % (IC95 {100 * lo:.1f}-{100 * hi:.1f} %), n = {n:,}. "
                "Método: proporción ponderada con el factor de expansión; varianza por linealización de Taylor con los conglomerados como unidades primarias de muestreo, sin incorporar la estratificación del diseño (IC aproximado)."))
    metodo_ic = "IC95 por linealización con conglomerados (sin estratos): aproximado."
    for col, etq, nombre in (("area", "Área", "area"), ("region_natural", "Región natural", "region"), ("departamento", "Departamento", "departamento")):
        if col not in df.columns or df[col].dropna().empty:
            b.append(_vacio(f"Sin datos disponibles para la prevalencia por {etq.lower()}."))
            continue
        t = _tabla_prevalencia(df, col)
        if t.empty:
            continue
        if col == "area":
            x.hechos["prev_area"] = t
        t_show = t.assign(prevalencia=lambda d: d["prevalencia"], **{"IC95 inf": t["ic_inf"], "IC95 sup": t["ic_sup"]}).drop(columns=["ic_inf", "ic_sup"])
        b.append(_tabla(f"Prevalencia ponderada de IA moderada/severa por {etq.lower()}", t_show, None, max_filas=30))
        b.append(x.figura(f"fig_prevalencia_{nombre}", P.mpl_prevalencia_grupos(t, etq), f"Prevalencia ponderada de IA moderada/severa por {etq.lower()} (IC95)",
                          I.i_prevalencia(t, etq.lower(), metodo_ic), "doble", _datos_prevalencia(t, col, etq)))
    sin = fies.value_counts(normalize=True).sort_index()
    w = df.loc[fies.index].groupby("fies_nivel")["factor_expansion"].sum()
    con = (w / w.sum()).sort_index()
    dist = pd.DataFrame({"nivel": sin.index.astype(int), "hogares": fies.value_counts().sort_index().to_numpy(), "sin_ponderar": sin.to_numpy(), "ponderada": con.reindex(sin.index).to_numpy()})
    b.append(_tabla("Distribución de los niveles FIES (0 = seguridad, 3 = inseguridad severa)", dist, I.i_objetivo(df)))
    b.append(x.figura("fig_distribucion_fies", P.mpl_distribucion_fies(dist), "Distribución del nivel FIES (sin ponderar y ponderada)", I.i_objetivo(df), "simple",
                          [_d("Distribución de los niveles FIES", dist, {"tipo": "barra", "titulo": "Distribución del nivel FIES", "cat": "nivel", "series": [{"col": "sin_ponderar", "nombre": "Sin ponderar"}, {"col": "ponderada", "nombre": "Ponderada"}], "x_titulo": "Nivel FIES", "y_titulo": "Proporción"})]))
    num_cols = C.FEATURES_SOCIO + C.FEATURES_FIES + [C.COL_OBJETIVO]
    rol = {**{c: "predictora socioeconómica" for c in C.FEATURES_SOCIO}, **{c: "ítem FIES (escenarios fies/híbrido)" for c in C.FEATURES_FIES}, C.COL_OBJETIVO: "objetivo"}
    filas = []
    for c in num_cols:
        if c not in df.columns:
            continue
        s = df[c]
        rango = (f"{s.min():.3g} a {s.max():.3g}" if pd.api.types.is_numeric_dtype(s) and s.notna().any() else
                 (str(s.mode().iloc[0]) + " (moda)" if s.notna().any() else "sin datos"))
        filas.append({"variable": c, "rol": rol[c], "tipo": "numérica" if pd.api.types.is_numeric_dtype(s) else "categórica",
                      "% faltante": round(100 * float(s.isna().mean()), 2), "valores distintos": int(s.nunique()), "rango / moda": rango})
    if filas:
        dicc = pd.DataFrame(filas)
        b.append(_tabla("Diccionario resumido de variables analíticas", dicc, {"que_muestra": f"{len(dicc)} variables analíticas con su rol, tipo, porcentaje de faltantes y rango.",
                                                                              "como_leerlo": "Las variables con 100 % de faltantes se descartan automáticamente de los predictores.",
                                                                              "limitacion": "El diccionario oficial de la ENAHO del año analizado debe validarse [POR VALIDAR] para los códigos de variable."}, max_filas=60))
    return b


# ------------------------------ 1b. Perfil descriptivo de hogares vulnerables ------------------------------
def _datos_perfil_subgrupos(sub: pd.DataFrame) -> list[dict[str, Any]]:
    cars = list(dict.fromkeys(sub["característica"]))
    filas = []
    for c in cars:
        con = sub[(sub["característica"] == c) & sub["grupo"].str.startswith("Con")].iloc[0]
        sin = sub[(sub["característica"] == c) & sub["grupo"].str.startswith("Sin")].iloc[0]
        filas.append({"característica": c, "prevalencia con": con["prevalencia"], "prevalencia sin": sin["prevalencia"], "n con": int(con["n"]), "n sin": int(sin["n"]),
                      "IC95 inf con": con["ic_inf"], "IC95 sup con": con["ic_sup"], "IC95 inf sin": sin["ic_inf"], "IC95 sup sin": sin["ic_sup"],
                      "semiancho sup. con (aux. gráfico)": max(con["ic_sup"] - con["prevalencia"], 0.0), "semiancho inf. con (aux. gráfico)": max(con["prevalencia"] - con["ic_inf"], 0.0),
                      "semiancho sup. sin (aux. gráfico)": max(sin["ic_sup"] - sin["prevalencia"], 0.0), "semiancho inf. sin (aux. gráfico)": max(sin["prevalencia"] - sin["ic_inf"], 0.0)})
    t = pd.DataFrame(filas)
    ser = [{"col": "prevalencia con", "nombre": "Con la característica", "color": "D55E00", "err_mas": "semiancho sup. con (aux. gráfico)", "err_menos": "semiancho inf. con (aux. gráfico)"},
           {"col": "prevalencia sin", "nombre": "Sin la característica", "color": "56B4E9", "err_mas": "semiancho sup. sin (aux. gráfico)", "err_menos": "semiancho inf. sin (aux. gráfico)"}]
    return [_d("Prevalencia ponderada de IA moderada/severa con y sin cada característica (IC95)", t,
               {"tipo": "barra", "titulo": "Prevalencia por característica (IC95)", "cat": "característica", "series": ser, "y_titulo": "Prevalencia", "alto_cm": 9})]


def _datos_perfil_recuento(rec: pd.DataFrame) -> list[dict[str, Any]]:
    t = rec[rec["n"] > 0].copy()
    t["recuento"] = t["recuento"].astype(int).astype(str)
    t = t.rename(columns={"ic_inf": "IC95 inf", "ic_sup": "IC95 sup"})
    t = _datos_ic(t, "prevalencia", "IC95 inf", "IC95 sup")
    return [_d("Prevalencia ponderada de IA moderada/severa por recuento de características (IC95)", t,
               {"tipo": "barra", "titulo": "Prevalencia por recuento (IC95)", "cat": "recuento", "series": [_serie_ic("prevalencia", "Prevalencia")],
                "x_titulo": "Recuento de características", "y_titulo": "Prevalencia", "alto_cm": 9})]


def _datos_perfil_fies(fr: pd.DataFrame) -> list[dict[str, Any]]:
    t = fr[fr["n"] > 0].copy()
    t["recuento"] = t["recuento"].astype(int).astype(str)
    niveles = [c for c in t.columns if c not in ("recuento", "n")]
    colores = ["9ECAE1", "F0E442", "E69F00", "D55E00"]
    return [_d("Distribución ponderada del nivel FIES por recuento de características", t,
               {"tipo": "barra", "titulo": "Nivel FIES por recuento", "cat": "recuento", "series": [{"col": c, "nombre": c, "color": k} for c, k in zip(niveles, colores)],
                "x_titulo": "Recuento de características", "y_titulo": "Proporción de hogares", "alto_cm": 9})]


def _seccion_perfil(x: _Ctx) -> list[dict[str, Any]]:
    """Perfil DESCRIPTIVO de hogares vulnerables: no es un índice ponderado ni validado; sin desenlaces nutricionales directos."""
    pf = PV.perfil(x.df)
    b: list[dict[str, Any]] = []
    if pf["n_con_fies"] == 0:
        return [_vacio("Sin datos disponibles: el dataset no tiene hogares con nivel FIES para construir el perfil.")]
    x.hechos["perfil"] = pf
    tot = pf["total"]
    b.append(_p(PV.ALCANCE))
    prev_txt = f"{100 * tot['prevalencia']:.1f}".replace(".", ",")
    n_txt = f"{tot['n']:,}".replace(",", " ")
    b.append(_p(f"Medida: prevalencia ponderada de inseguridad alimentaria moderada o severa (FIES nivel 2-3); total {prev_txt} % (n = {n_txt}). "
                f"Tres características, definidas con variables reales del dataset (misma definición que la descripción de subgrupos del artículo). {PV.METODO_IC}"))
    b.append(_tabla("Definición de las características del perfil", pf["definiciones"],
                    {"que_muestra": "Variable del dataset y regla con que se marca cada característica.", "como_leerlo": "Un hogar pertenece al grupo si cumple la regla; el recuento suma las características presentes (0 a 3).",
                     "limitacion": "Los umbrales y significados marcados [POR VALIDAR] no se verificaron contra el protocolo ni el diccionario oficial."}))
    sub, dif, rec, fr = pf["subgrupos"], pf["diferencias"], pf["recuento"], pf["fies_recuento"]
    niv = tuple(PV.NIVELES_FIES.values())
    t_sub = PV.formatear_tabla(sub.drop(columns=["clave"]).rename(columns={"ic_inf": "IC95 inf", "ic_sup": "IC95 sup", "pct_hogares": "% de hogares (ponderado)", "prevalencia": "prevalencia IA mod./sev."}),
                               pct=("% de hogares (ponderado)", "prevalencia IA mod./sev.", "IC95 inf", "IC95 sup"), enteros=("n",))
    b.append(_tabla("Prevalencia ponderada de IA moderada/severa por subgrupo", t_sub, {
        "que_muestra": "Prevalencia ponderada, número de hogares de la muestra (n) e IC95 de cada subgrupo, con y sin cada característica.",
        "como_leerlo": "El % de hogares está ponderado con el factor de expansión; la interpretación comparativa está bajo la Figura correspondiente.",
        "limitacion": "Subgrupos con n < 30 se marcan como inestables. " + PV.METODO_IC}))
    b.append(x.figura("fig_perfil_subgrupos", P.mpl_perfil_subgrupos(sub), "Prevalencia ponderada de IA moderada/severa con y sin cada característica (IC95)",
                      PV.i_subgrupos(sub, dif), "doble", _datos_perfil_subgrupos(sub)))
    b.append(_tabla("Diferencia de prevalencia entre hogares con y sin la característica", PV.formatear_tabla(dif, pct=("prevalencia con", "prevalencia sin"), pp=("diferencia (pp)",)), {
        "que_muestra": "Diferencia en puntos porcentuales (con menos sin) y si los IC95 de ambos subgrupos se solapan.",
        "como_leerlo": "IC solapados: la diferencia no es concluyente. No se realizaron pruebas de hipótesis.", "limitacion": "Descriptivo y asociativo; no causal."}))
    ex = f" {pf['n_excluidos_recuento']:,} hogares sin dato en alguna característica quedan fuera del recuento." if pf["n_excluidos_recuento"] else ""
    t_rec = PV.formatear_tabla(rec.rename(columns={"ic_inf": "IC95 inf", "ic_sup": "IC95 sup", "pct_hogares": "% de hogares (ponderado)", "prevalencia": "prevalencia IA mod./sev."}),
                               pct=("% de hogares (ponderado)", "prevalencia IA mod./sev.", "IC95 inf", "IC95 sup"), enteros=("n",))
    b.append(_tabla("Prevalencia ponderada de IA moderada/severa por recuento de características (0-3)", t_rec, {
        "que_muestra": "Prevalencia ponderada, n e IC95 de los hogares según cuántas de las tres características cumplen.",
        "como_leerlo": "Recuento 0 = ninguna característica; 3 = las tres. La interpretación está bajo la Figura correspondiente." + ex,
        "limitacion": "Recuentos con pocos hogares dan IC amplios. " + PV.METODO_IC}))
    b.append(x.figura("fig_perfil_recuento", P.mpl_perfil_recuento(rec, tot["prevalencia"]), "Prevalencia ponderada de IA moderada/severa por recuento de características (IC95)",
                      {**PV.i_recuento(rec), "limitacion": PV.i_recuento(rec)["limitacion"] + ex}, "simple", _datos_perfil_recuento(rec)))
    b.append(_tabla("Distribución ponderada del nivel FIES por recuento de características (% de hogares)", PV.formatear_tabla(fr, pct=niv, enteros=("n",)), {
        "que_muestra": "Reparto ponderado de los hogares con cada recuento entre los cuatro niveles FIES (cada fila suma 100 %).",
        "como_leerlo": "La interpretación está bajo la Figura correspondiente.", "limitacion": "Proporciones puntuales ponderadas, sin intervalos."}))
    b.append(x.figura("fig_perfil_fies", P.mpl_perfil_fies(fr), "Nivel FIES por recuento de características (% ponderado de hogares)", PV.i_fies(fr), "simple", _datos_perfil_fies(fr)))
    b.append(_tabla("Distribución ponderada del nivel FIES con y sin cada característica (% de hogares)", PV.formatear_tabla(pf["fies_caracteristica"], pct=niv, enteros=("n",)), {
        "que_muestra": "Reparto de los hogares entre los cuatro niveles FIES dentro de cada subgrupo (cada fila suma 1).", "como_leerlo": "Compare la proporción de inseguridad moderada y severa entre filas «con» y «sin».",
        "limitacion": PV.ALCANCE}))
    return b


# ------------------------------ 2. Métodos ------------------------------
def _versiones() -> pd.DataFrame:
    filas = [{"componente": "Python", "versión": platform.python_version()}, {"componente": "Sistema", "versión": f"{platform.system()} {platform.release()}"}]
    for pkg in PAQUETES:
        try:
            filas.append({"componente": pkg, "versión": md.version(pkg)})
        except md.PackageNotFoundError:
            continue
    return pd.DataFrame(filas)


def _plano(d: Any) -> str:
    if isinstance(d, dict):
        return json.dumps({k: _plano(v) if isinstance(v, dict) else (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}, ensure_ascii=False, default=str)
    return f"{d:.6g}" if isinstance(d, float) else str(d)


def _seccion_metodos(x: _Ctx) -> list[dict[str, Any]]:
    b: list[dict[str, Any]] = []
    cfg = x.cfg
    if x.codigos:
        mod = pd.DataFrame([{"código": c, "modelo": MODELOS_NOMBRE.get(c, c), "escenario": x.escenario} for c in x.codigos])
    else:
        mod = pd.DataFrame([{"código": c, "modelo": n, "escenario": "—"} for c, n in MODELOS_NOMBRE.items()])
    b.append(_tabla("Modelos comparados (M1-M5) y escenario analizado", mod,
                    {"que_muestra": "Los cinco modelos del protocolo; solo los seleccionados tienen resultados en este informe.", "como_leerlo": "El escenario «socio» usa solo variables socioeconómicas (predicción real); «fies» e «hibrido» incluyen los ítems del objetivo (circulares).", "limitacion": "El desempeño de los escenarios fies/hibrido no mide capacidad predictiva."}))
    ver = cfg.get("verificacion_particion", {})
    if ver:
        tam, prop = ver.get("tamanos", {}), ver.get("proporciones", {})
        part = pd.DataFrame({"partición": list(tam), "hogares": list(tam.values()), "proporción": [prop.get(k) for k in tam]})
        b.append(_p(f"Partición {int(100 * C.PROPORCIONES_PARTICION[0])}/{int(100 * C.PROPORCIONES_PARTICION[1])}/{int(100 * C.PROPORCIONES_PARTICION[2])} agrupada por «{cfg.get('agrupar_por', 'hogar_uid')}»: "
                    f"Desarrollo (CV anidada y modelo final), Validación (umbral operativo y calibración) y Prueba (una sola evaluación). Huella SHA-256 de la partición: {cfg.get('huella_particion', 'sin datos')}."))
        b.append(_tabla("Tamaños de la partición Desarrollo / Validación / Prueba", part, I.i_particion(ver)))
    else:
        b.append(_vacio("Sin datos disponibles: la corrida no registró la verificación de la partición."))
    esq = pd.DataFrame([{"esquema": k, "descripción": v} for k, v in ESQUEMAS_TEXTO.items()])
    b.append(_tabla("Esquemas de validación del protocolo (V1 a V5 y V1c)", esq, {"que_muestra": "Los esquemas posibles del protocolo.", "como_leerlo": "V5 no es una partición sino el modo de métricas ponderadas.", "limitacion": "Solo los esquemas ejecutados tienen resultados en la sección 3."}))
    b.append(_p(f"CV anidada: {cfg.get('n_outer', C.N_SPLITS_EXTERNOS)} pliegues externos × {cfg.get('n_inner', C.INNER_FOLDS)} internos dentro de Desarrollo; repeticiones de V1/V1c: {C.N_REPETICIONES}; "
                f"semilla {cfg.get('seed', C.SEED)}; bootstrap {C.N_BOOT} réplicas; nivel de significancia {C.ALPHA}."))
    if x.res:
        hp, pres = [], []
        for c, r in x.res.items():
            fin = r.get("final", {})
            for k, v in (fin.get("hiperparametros") or {}).items():
                hp.append({"modelo": r["codigo"], "hiperparámetro": k, "valor óptimo": _plano(v)})
            pres.append({"modelo": r["codigo"], "ensayos por pliegue": r.get("n_trials"), "ensayos búsqueda final": r.get("n_trials_final"),
                         "mejor valor interno (F1-macro)": fin.get("mejor_valor_interno"), "segundos totales": r.get("segundos_total")})
        b.append(_tabla("Hiperparámetros óptimos del modelo final por modelo", pd.DataFrame(hp), {"que_muestra": "Hiperparámetros del modelo final, elegidos por Optuna sobre todo Desarrollo.", "como_leerlo": "Cada fila es un hiperparámetro del modelo indicado.", "limitacion": "Con presupuestos de ensayos pequeños el óptimo puede estar lejos del óptimo real."}, max_filas=60))
        pres_df = pd.DataFrame(pres)
        bajo = [r["modelo"] for r in pres if (r["ensayos por pliegue"] or 0) < C.N_TRIALS]
        b.append(_tabla("Presupuesto de ensayos de optimización", pres_df, {"que_muestra": "Número de ensayos de Optuna usados por modelo.", "como_leerlo": f"El protocolo prevé {C.N_TRIALS} ensayos por modelo y pliegue externo.",
                                                                          "limitacion": ("Presupuesto inferior al del protocolo en: " + ", ".join(bajo) + "; los resultados son preliminares.") if bajo else "Presupuesto conforme al protocolo."}))
        x.hechos["trials_bajos"] = bool(bajo)
    else:
        b.append(_vacio("Sin datos disponibles de hiperparámetros ni presupuesto de ensayos: no hay una corrida de CV anidada seleccionada."))
    b.append(_tabla("Versiones de software (importlib.metadata)", _versiones(), {"que_muestra": "Versiones exactas de las librerías instaladas al generar el informe.", "como_leerlo": "Necesarias para reproducir los resultados.", "limitacion": "Describen el entorno del informe, no necesariamente el de la corrida de entrenamiento."}, max_filas=40))
    return b


# ------------------------------ predicciones / agregados ------------------------------
def _cargar_pred(x: _Ctx, c: str) -> tuple[dict[str, np.ndarray] | None, str | None]:
    if x.dir_run is None:
        return None, None
    ident = f"{x.escenario}_{c}"
    p = EV.cargar_prediccion_prueba(x.dir_run, ident)
    if p is not None:
        return p, "Prueba"
    f = x.dir_run / f"pred_val_{ident}.npz"
    if f.is_file():
        z = np.load(f, allow_pickle=False)
        return {k: z[k] for k in z.files}, "Validación"
    return None, None


def _agregado_run(x: _Ctx, clave: str) -> pd.DataFrame:
    filas = []
    for c, r in x.res.items():
        for m in METRICAS_AGREGADAS:
            a = (r.get("agregado", {}).get(clave) or {}).get(m)
            if a:
                filas.append({"modelo": r["codigo"], "metrica": m, "media": a["media"], "DE": a["sd"], "ic_inf": a["ic_inf"], "ic_sup": a["ic_sup"], "n": a["n"]})
    return pd.DataFrame(filas)


def _agregado_bd(x: _Ctx, ponderada: bool) -> pd.DataFrame:
    """Respaldo: métricas por pliegue de experimentos en PostgreSQL (sin estrato), agregadas como media, DE e IC95 t."""
    q = x.sql("""SELECT mo.codigo AS modelo, m.nombre_metrica AS metrica, m.valor FROM fsai.metricas m
                 JOIN fsai.modelos mo ON mo.modelo_id = m.modelo_id JOIN fsai.experimentos e ON e.experimento_id = mo.experimento_id
                 WHERE e.dataset_id = :d AND m.ponderada = :p AND m.estrato IS NULL AND m.fold IS NOT NULL""", d=int(x.dataset_id), p=bool(ponderada))
    filas = []
    for (mod, met), g in q.groupby(["modelo", "metrica"]):
        v = g["valor"].dropna().to_numpy(float)
        if len(v) == 0:
            continue
        sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        h = float(stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        filas.append({"modelo": mod, "metrica": met, "media": float(v.mean()), "DE": sd, "ic_inf": float(v.mean()) - h, "ic_sup": float(v.mean()) + h, "n": len(v)})
    return pd.DataFrame(filas)


def _tabla_agregada(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"modelo": "modelo", "metrica": "métrica", "media": "media", "ic_inf": "IC95 inf", "ic_sup": "IC95 sup"})[["modelo", "métrica", "media", "DE", "IC95 inf", "IC95 sup", "n"]]


# ------------------------------ 3. Resultados ------------------------------
def _seccion_resultados(x: _Ctx) -> list[dict[str, Any]]:
    b: list[dict[str, Any]] = []
    fuente_bd = False
    agp, agn = _agregado_run(x, "ponderada"), _agregado_run(x, "no_ponderada")
    if agp.empty:
        agp, agn = _agregado_bd(x, True), _agregado_bd(x, False)
        fuente_bd = not agp.empty
    if agp.empty:
        return [_vacio("Sin datos disponibles: no hay una corrida de CV anidada seleccionada ni métricas de experimentos en PostgreSQL para este dataset. "
                       "Entrene los modelos en el módulo de Entrenamiento y vuelva a generar el informe.")]
    origen = "métricas por pliegue de experimentos en PostgreSQL" if fuente_bd else f"CV anidada {x.cfg.get('n_outer', C.N_SPLITS_EXTERNOS)}×{x.cfg.get('n_inner', C.INNER_FOLDS)} de la corrida «{x.dir_run.name}»"
    b.append(_p(f"Fuente de las cifras: {origen}. Media, desviación estándar (DE) e intervalo de confianza del 95 % (t de Student sobre los pliegues externos). "
                "Las métricas ponderadas usan el factor de expansión; las no ponderadas describen la muestra."))
    pie_tab = {"que_muestra": "Desempeño de cada modelo en los pliegues externos de la CV anidada.", "como_leerlo": "Valores altos son mejores, salvo Brier y ECE (menores son mejores).",
               "limitacion": "El IC con pocos pliegues es amplio; es validación cruzada interna, no la Prueba independiente."}
    b.append(_tabla("Desempeño por modelo y métrica (ponderado con factor de expansión)", _tabla_agregada(agp), pie_tab, max_filas=80))
    b.append(_tabla("Desempeño por modelo y métrica (no ponderado)", _tabla_agregada(agn), pie_tab, max_filas=80))
    x.hechos["agregado"] = agp
    sel = agp[agp["metrica"].isin(["f1_macro", "balanced_accuracy", "roc_auc_ovr", "recall_severa"])]
    if not sel.empty:
        b.append(x.figura("fig_desempeno_modelos", P.mpl_metricas_modelos(sel, ["f1_macro", "balanced_accuracy", "roc_auc_ovr", "recall_severa"], ETQ_METRICA),
                          "Desempeño por modelo en la CV anidada (media e IC95; métricas ponderadas)", I.i_rendimiento(agp), "doble",
                          _datos_desempeno(sel, ["f1_macro", "balanced_accuracy", "roc_auc_ovr", "recall_severa"])))
    sev = []
    for c, r in x.res.items():
        a_p = (r.get("agregado", {}).get("ponderada") or {}).get("recall_severa")
        a_n = (r.get("agregado", {}).get("no_ponderada") or {}).get("recall_severa")
        v = (r.get("final") or {}).get("validacion", {})
        sev.append({"modelo": c, "recall IA severa (pond.)": a_p and a_p["media"], "IC95 inf": a_p and a_p["ic_inf"], "IC95 sup": a_p and a_p["ic_sup"],
                    "recall IA severa (no pond.)": a_n and a_n["media"], "umbral operativo (Validación)": v.get("umbral_severa"),
                    "recall con umbral (Validación)": v.get("recall_severa_umbral"), "especificidad con umbral": v.get("especificidad_severa_umbral")})
    if sev:
        b.append(_tabla("Recall de inseguridad alimentaria severa", pd.DataFrame(sev), {"que_muestra": "Sensibilidad para detectar hogares con IA severa, con argmax (CV) y con el umbral operativo fijado en Validación.",
                                                                                        "como_leerlo": "El umbral se eligió exigiendo una especificidad mínima de 0,70; mayor recall con el umbral implica más falsos positivos.", "limitacion": "El recall de la clase minoritaria varía mucho entre pliegues; mire su IC."}))
    # curvas y matrices
    curvas_roc, curvas_pr, curvas_cal, filas_cal = {}, {}, {}, []
    fuentes = {}
    for c in x.codigos:
        p, fuente = _cargar_pred(x, c)
        if p is None:
            continue
        x.preds[c] = (p, fuente)
        fuentes[c] = fuente
        y, proba, w = p["y"].astype(int), p["proba"], p["w"]
        cv = EV.curvas_roc_pr(y, proba)
        if C.CLASE_SEVERA in cv["roc"]:
            curvas_roc[c] = (*cv["roc"][C.CLASE_SEVERA], cv["auc"][C.CLASE_SEVERA])
            curvas_pr[c] = (*cv["pr"][C.CLASE_SEVERA], cv["ap"][C.CLASE_SEVERA])
        curvas_cal[c] = EV.curva_calibracion(y, proba, C.CLASE_SEVERA)
        cal = (x.res.get(f"{x.escenario}_{c}", {}).get("final") or {}).get("calibrador")
        resumen = EV.calibracion_resumen(y, proba, w, cal if fuente == "Prueba" else None)
        filas_cal.append({"modelo": c, "fuente": fuente, "AUC IA severa": cv["auc"].get(C.CLASE_SEVERA), "AUC macro": cv.get("auc_macro"), "AP IA severa": cv["ap"].get(C.CLASE_SEVERA),
                          "Brier (sin calibrar)": resumen["brier"], "ECE (sin calibrar)": resumen["ece"], "Brier calibrado": resumen["brier_calibrado"], "ECE calibrado": resumen["ece_calibrado"]})
        cm = EV.matriz_confusion(y, proba.argmax(1))
        pie_cm = I.i_confusion(cm)
        pie_cm["limitacion"] += f" Predicciones de {fuente} ({len(y):,} hogares), sin ponderar."
        b.append(x.figura(f"fig_confusion_{c}", P.mpl_confusion_doble(cm, f"{c} · {fuente}"), f"Matriz de confusión del modelo {c} ({fuente}): conteos y normalizada por fila", pie_cm, "doble",
                          [_d(f"Modelo {c} · {fuente}: conteos (filas = nivel real, columnas = nivel predicho por argmax)", _matriz_df(cm, False), color="escala"),
                           _d(f"Modelo {c} · {fuente}: normalizada por fila (sensibilidad por nivel)", _matriz_df(cm, True), color="escala")]))
    if filas_cal:
        usadas = ", ".join(f"{k}: {v}" for k, v in fuentes.items())
        b.append(_p(f"Curvas, matrices y calibración calculadas sobre las predicciones guardadas ({usadas}). La Prueba solo se usa si ya fue evaluada en el módulo de Comparación estadística; "
                    "en caso contrario se usa la Validación (15 %), cuyo umbral y calibrador ya se ajustaron allí, por lo que su Brier/ECE calibrados no se reportan."))
        tb = pd.DataFrame(filas_cal)
        resumen_pie = {"que_muestra": "Discriminación (AUC, AP) y calibración (Brier, ECE) por modelo sobre las predicciones guardadas.", "como_leerlo": "AUC y AP altos son mejores; Brier y ECE bajos son mejores.",
                       "limitacion": "AUC de la clase IA severa (uno-contra-resto); Brier multiclase. Sobre Validación las cifras calibradas no se reportan (el calibrador se ajustó allí)."}
        b.append(_tabla("Discriminación y calibración por modelo", tb, resumen_pie))
        if curvas_roc:
            prev_sev = float(next(iter(x.preds.values()))[0]["y"].astype(int).__eq__(C.CLASE_SEVERA).mean())
            pie_roc = I.i_roc_pr_modelos({k: v[2] for k, v in curvas_roc.items()}, {k: v[2] for k, v in curvas_pr.items()}, prev_sev, "roc")
            pie_pr = I.i_roc_pr_modelos({k: v[2] for k, v in curvas_roc.items()}, {k: v[2] for k, v in curvas_pr.items()}, prev_sev, "pr")
            b.append(x.figura("fig_roc_severa", P.mpl_curvas_multi(curvas_roc, "roc"), "Curvas ROC de la clase IA severa (uno-contra-resto) por modelo", pie_roc, "simple",
                              _datos_curvas(curvas_roc, "FPR", "TPR", "Curvas ROC de la clase IA severa", True)))
            b.append(x.figura("fig_pr_severa", P.mpl_curvas_multi(curvas_pr, "pr"), "Curvas precisión-recall de la clase IA severa por modelo", pie_pr, "simple",
                              _datos_curvas(curvas_pr, "recall", "precisión", "Curvas precisión-recall de la clase IA severa", False)))
        b.append(x.figura("fig_calibracion_severa", P.mpl_calibracion_multi(curvas_cal), "Curvas de calibración de la clase IA severa por modelo",
                          I.i_calibracion(float(tb["Brier (sin calibrar)"].mean()), float(tb["ECE (sin calibrar)"].mean())), "simple", _datos_calibracion(curvas_cal)))
    else:
        b.append(_vacio("Sin datos disponibles: la corrida no tiene predicciones guardadas (Validación o Prueba) para matrices de confusión, ROC, PR ni calibración."))
    evaluados = [c for c in x.codigos if x.preds.get(c, (None, None))[1] == "Prueba"]
    if evaluados:
        for c in evaluados:
            p = x.preds[c][0]
            met = EV.metricas_con_ic(p["y"], p["proba"], p["w"], n_boot=x.n_boot)
            b.append(_tabla(f"Métricas en la Prueba con IC95 bootstrap · {c}", met[met.ponderada][["metrica", "valor", "ic_inf", "ic_sup"]].rename(columns={"metrica": "métrica", "ic_inf": "IC95 inf", "ic_sup": "IC95 sup"}),
                            {"que_muestra": f"Métricas ponderadas de {c} en la evaluación única de la Prueba.", "como_leerlo": "El IC es bootstrap percentil estratificado por clase.", "limitacion": "El bootstrap ignora estrato y conglomerado."}))
    else:
        b.append(_p("La Prueba (15 %) aún no fue evaluada para ninguno de los modelos seleccionados: este informe no la consume. Evalúela una sola vez en el módulo de Comparación estadística."))
    x.hechos["n_prueba"] = len(evaluados)
    # por esquema y estrato (PostgreSQL)
    try:
        q = x.sql("""SELECT e.esquema_validacion AS esquema, mo.codigo AS modelo, m.ponderada, m.nombre_metrica AS metrica, m.estrato, m.valor
                     FROM fsai.metricas m JOIN fsai.modelos mo ON mo.modelo_id = m.modelo_id JOIN fsai.experimentos e ON e.experimento_id = mo.experimento_id
                     WHERE e.dataset_id = :d AND m.nombre_metrica IN ('f1_macro','roc_auc_ovr','recall_severa','balanced_accuracy')""", d=int(x.dataset_id))
    except Exception:  # noqa: BLE001
        q = pd.DataFrame()
    if q.empty:
        b.append(_vacio("Sin datos disponibles de resultados por esquema de validación ni por estrato: no hay experimentos registrados en PostgreSQL para este dataset."))
    else:
        for etq, mask in (("por esquema de validación (sin estratos)", q["estrato"].isna()), ("por estrato", q["estrato"].notna())):
            sub = q[mask & q["ponderada"]]
            if sub.empty:
                b.append(_vacio(f"Sin datos disponibles de resultados {etq}."))
                continue
            claves = ["esquema", "modelo", "metrica"] + (["estrato"] if etq == "por estrato" else [])
            g = sub.groupby(claves)["valor"].agg(media="mean", DE="std", pliegues="count").reset_index().rename(columns={"metrica": "métrica"})
            b.append(_tabla(f"Resultados {etq} (ponderados; media de pliegues)", g, {"que_muestra": f"Media y DE de pliegues por esquema{' y estrato' if 'estrato' in claves else ''}, leídas de fsai.metricas.", "como_leerlo": "Compare esquemas: V2 (geográfico) menor que V1 indica dependencia territorial.",
                                                                                      "limitacion": "Los experimentos de PostgreSQL pueden haberse corrido con presupuestos de ensayos distintos a los de la corrida principal."}, max_filas=60))
    return b


# ------------------------------ 4. Comparación estadística ------------------------------
def _folds_f1(x: _Ctx) -> dict[str, list[float]]:
    return {c: [f["metricas"]["ponderada"]["f1_macro"] for f in x.res[f"{x.escenario}_{c}"]["folds"]] for c in x.codigos if f"{x.escenario}_{c}" in x.res}


def _seccion_comparacion(x: _Ctx) -> list[dict[str, Any]]:
    cods = [c for c in x.codigos if f"{x.escenario}_{c}" in x.res]
    if len(cods) < 2:
        return [_vacio("Sin datos disponibles: la comparación estadística requiere al menos dos modelos con CV anidada completada en la misma corrida y escenario.")]
    b: list[dict[str, Any]] = []
    folds = _folds_f1(x)
    n_pl = min(len(v) for v in folds.values())
    fuentes = {x.preds[c][1] for c in cods if c in x.preds}
    comun = fuentes.pop() if len(fuentes) == 1 and all(c in x.preds for c in cods) else None
    filas: list[dict[str, Any]] = []
    fam: dict[str, list[dict[str, Any]]] = {"delong_severa": [], "mcnemar": [], "wilcoxon_pliegues": []}
    efecto: list[dict[str, Any]] = []
    for a, bb in combinations(cods, 2):
        fa, fb = np.array(folds[a][:n_pl]), np.array(folds[bb][:n_pl])
        d = fa - fb
        r = ST.wilcoxon_pareado(fa, fb)
        fam["wilcoxon_pliegues"].append({"comparacion": f"{a}-{bb}", "estadistico": r["estadistico"], "p_valor": r["p_valor"], "detalle": f"n={r['n']} pliegues; p mínimo posible={r['p_minimo_posible']:.4f}"})
        sd = float(d.std(ddof=1)) if n_pl > 1 else float("nan")
        # sesgo rango-biserial: (W+ - W-)/(n(n+1)/2)
        rangos = stats.rankdata(np.abs(d[d != 0])) if (d != 0).any() else np.array([])
        sig = np.sign(d[d != 0])
        rb = float((rangos[sig > 0].sum() - rangos[sig < 0].sum()) / rangos.sum()) if rangos.size else float("nan")
        fila = {"comparación": f"{a} - {bb}", "ΔF1-macro medio (pliegues)": float(d.mean()), "d de Cohen pareado": float(d.mean() / sd) if sd and sd == sd else float("nan"), "correlación rango-biserial": rb,
                "ΔAUC IA severa": None, "razón b/c McNemar": None}
        if comun is not None:
            pa, pb = x.preds[a][0], x.preds[bb][0]
            dl = ST.delong_ovr(pa["y"], pa["proba"], pb["proba"], C.CLASE_SEVERA)
            fam["delong_severa"].append({"comparacion": f"{a}-{bb}", "estadistico": dl["z"], "p_valor": dl["p_valor"], "detalle": f"AUC {a}={dl['auc1']:.3f}; {bb}={dl['auc2']:.3f}"})
            mc = ST.mcnemar_ponderado(pa["y"], pa["proba"].argmax(1), pb["proba"].argmax(1), pa["w"])
            fam["mcnemar"].append({"comparacion": f"{a}-{bb}", "estadistico": mc["estadistico"], "p_valor": mc["p_valor"], "detalle": f"b={mc['b']:.1f}; c={mc['c']:.1f}"})
            fila["ΔAUC IA severa"] = dl["auc1"] - dl["auc2"]
            fila["razón b/c McNemar"] = float(mc["b"] / mc["c"]) if mc["c"] else float("nan")
        efecto.append(fila)
    for prueba, lista in fam.items():
        if lista:
            aj = ST.holm([f["p_valor"] for f in lista])
            filas += [{"prueba": prueba, **f, "p_ajustado": float(p)} for f, p in zip(lista, aj)]
    fr = None
    if len(cods) >= 3 and n_pl >= 2:
        M = np.column_stack([np.array(folds[c][:n_pl]) for c in cods])
        fr = ST.friedman_nemenyi(M, cods)
        filas.append({"prueba": "friedman", "comparacion": "global", "estadistico": fr["chi2"], "p_valor": fr["p_valor"], "p_ajustado": fr["p_valor"], "detalle": f"rangos medios={dict(fr['rangos_medios'])}"})
        for a, bb in combinations(cods, 2):
            pn = float(fr["nemenyi_p"].loc[a, bb])
            filas.append({"prueba": "nemenyi", "comparacion": f"{a}-{bb}", "estadistico": None, "p_valor": pn, "p_ajustado": pn, "detalle": ""})
    tabla = pd.DataFrame(filas)
    x.hechos["pruebas"] = tabla
    if comun is None:
        b.append(_p("DeLong y McNemar requieren predicciones guardadas de todos los modelos en el mismo conjunto (Prueba o Validación); no están disponibles, por lo que solo se reportan las pruebas sobre pliegues (Wilcoxon, Friedman/Nemenyi)."))
    else:
        b.append(_p(f"DeLong (AUC de la clase IA severa) y McNemar ponderado se calcularon sobre las predicciones de {comun}. Wilcoxon y Friedman/Nemenyi usan el F1-macro ponderado de los {n_pl} pliegues externos. "
                    "Los p-valores se ajustaron con Holm dentro de cada familia de pruebas."))
    pie_t = {"que_muestra": f"{len(tabla)} contrastes entre modelos con su estadístico, p-valor y p ajustado por Holm.", "como_leerlo": "p ajustado < 0,05 indica diferencia estadísticamente significativa; un p grande no demuestra igualdad.",
             "limitacion": "DeLong y McNemar asumen observaciones independientes; Wilcoxon/Friedman con pocos pliegues tienen muy baja potencia."}
    b.append(_tabla("Pruebas estadísticas entre modelos (p ajustados por Holm)", tabla, pie_t, max_filas=60))
    if efecto:
        b.append(_tabla("Tamaños de efecto entre modelos", pd.DataFrame(efecto), {"que_muestra": "Diferencias medias de F1-macro entre pares, d de Cohen pareado, correlación rango-biserial (Wilcoxon) y, si hay predicciones, ΔAUC y razón b/c de McNemar.",
                                                                                "como_leerlo": "|d| ≈ 0,2 pequeño, 0,5 mediano, 0,8 grande. Una razón b/c > 1 indica que el primer modelo acierta más donde el segundo falla.", "limitacion": "Con pocos pliegues los tamaños de efecto son inestables."}))
    if fr is not None:
        fig = P.mpl_diagrama_dc(dict(fr["rangos_medios"]), float(fr["diferencia_critica"]))
        b.append(x.figura("fig_diferencia_critica", fig, "Diagrama de diferencia crítica (Friedman + Nemenyi, F1-macro por pliegue)",
                          {"que_muestra": f"Rangos medios de {len(cods)} modelos en {n_pl} pliegues; DC = {fr['diferencia_critica']:.2f}.", "como_leerlo": "Modelos unidos por una barra negra no difieren de forma significativa (diferencia de rangos < DC). El rango 1 es el mejor.",
                           "limitacion": "Con pocos pliegues la DC es grande y casi nada resulta distinguible."}, "doble", _datos_dc(fr)))
    b.append({"tipo": "h2", "texto": "Lectura en lenguaje claro"})
    b.append({"tipo": "lista", "items": I.lectura_pruebas(tabla, n_pl)})
    return b


# ------------------------------ 5. Interpretabilidad ------------------------------
def _shap_desde_bd(x: _Ctx, modelo: str | None = None, escenario: str | None = None) -> pd.DataFrame | None:
    """|SHAP| medio almacenado en PostgreSQL (fsai.explicaciones) para el dataset.

    Mejora #16: sin filtros promedia TODAS las explicaciones del dataset (mezcla de modelos y escenarios).
    ``modelo`` (código M1..M5) y ``escenario`` (socio/fies/hibrido) restringen el promedio a un único modelo y/o
    escenario; sin ellos el comportamiento es el histórico (compatibilidad hacia atrás).
    """
    consulta = """SELECT ex.contribuciones FROM fsai.explicaciones ex JOIN fsai.modelos mo ON mo.modelo_id = ex.modelo_id
                  JOIN fsai.experimentos e ON e.experimento_id = mo.experimento_id WHERE e.dataset_id = :d AND ex.tipo ILIKE '%shap%'"""
    params: dict[str, Any] = {"d": int(x.dataset_id)}
    if modelo:
        consulta += " AND mo.codigo = :m"
        params["m"] = str(modelo)
    if escenario:
        consulta += " AND e.escenario = :e"
        params["e"] = str(escenario)
    try:
        q = x.sql(consulta + " LIMIT 500", **params)
    except Exception:  # noqa: BLE001
        return None
    acum: dict[str, list[float]] = {}
    for v in q["contribuciones"]:
        v = v if isinstance(v, dict) else json.loads(v)
        for k, val in (v.get("importancia") or v.get("features") or {}).items():
            acum.setdefault(k, []).append(abs(float(val)))
    if not acum:
        return None
    return pd.DataFrame({"variable": list(acum), "importancia": [float(np.mean(v)) for v in acum.values()]})


def _shap_modelo_guardado(x: _Ctx) -> tuple[pd.DataFrame | None, str]:
    """|SHAP| medio guardado del MEJOR modelo de la corrida (analisis/shap_<escenario>_<modelo>.json), con su muestra declarada."""
    cod = _mejor_modelo(x)
    if cod is None or x.dir_run is None:
        return None, ""
    f = x.dir_run / "analisis" / f"shap_{x.escenario}_{cod}.json"
    if not f.is_file():
        return None, ""
    try:
        d = _json(f)
        imp = pd.DataFrame(list((d.get("importancia") or {}).items()), columns=["variable", "importancia"])
        imp["importancia"] = imp["importancia"].astype(float)
    except (OSError, ValueError, KeyError, TypeError):
        return None, ""
    if imp.empty:
        return None, ""
    n = d.get("n_muestra")
    return imp, (f"|SHAP| medio guardado del modelo evaluado {cod} (escenario {x.escenario}; {d.get('explainer', 'SHAP')}"
                 f"{f', muestra de {int(n)} hogares' if n else ''}; archivo {f.name})")


def _seccion_interpretabilidad(x: _Ctx) -> list[dict[str, Any]]:
    b: list[dict[str, Any]] = []
    sh = x.shap or {}
    imp: pd.DataFrame | None = sh.get("importancia")
    origen = sh.get("fuente", "SHAP calculado sobre Validación")
    if imp is None or imp.empty:
        imp, origen = _shap_modelo_guardado(x)   # SHAP guardado del modelo evaluado (el mismo que usan equidad y PDP/ALE)
    if imp is None or imp.empty:
        cod = _mejor_modelo(x)
        imp = _shap_desde_bd(x, modelo=cod, escenario=x.escenario) if cod else None   # #16: primero el mejor modelo y el escenario evaluado
        if imp is not None and not imp.empty:
            origen = (f"promedio de las explicaciones SHAP almacenadas en PostgreSQL del modelo {cod} y el escenario {x.escenario} "
                      "(fsai.explicaciones, hasta 500 filas)")
        else:
            imp = _shap_desde_bd(x)   # último recurso: sin filtro
            origen = ("promedio de TODAS las explicaciones SHAP almacenadas en PostgreSQL para el dataset (fsai.explicaciones, hasta 500 filas, "
                      "mezcla de modelos y escenarios; no corresponde a un único modelo)")
    if imp is None or imp.empty:
        b.append(_vacio("Sin datos disponibles de SHAP: no hay explicaciones almacenadas para este dataset. Active «Calcular SHAP» al generar el informe o use el módulo de Interpretabilidad."))
    else:
        imp = imp.sort_values("importancia", ascending=False).reset_index(drop=True)
        tot = float(imp["importancia"].sum()) or 1.0
        top = imp.head(15).assign(**{"participación (%)": lambda d: 100 * d["importancia"] / tot, "acumulada (%)": lambda d: 100 * d["importancia"].cumsum() / tot})
        b.append(_p(f"Fuente: {origen}. La importancia global es el |SHAP| medio por variable. SHAP describe asociaciones del modelo con la predicción, no relaciones causales."))
        pie_g = I.i_importancia(imp.assign(variable=[E.etq(v) for v in imp["variable"]]))
        b.append(x.figura("fig_shap_barras", P.mpl_shap_barras(imp), "Importancia global SHAP (|SHAP| medio) de las variables principales", pie_g, "simple", _datos_shap(imp)))
        b.append(_tabla("Variables más importantes (top 15) y participación en la importancia total", top, pie_g))
        if sh.get("firmado") is not None and sh.get("X") is not None:
            b.append(x.figura("fig_shap_beeswarm", P.mpl_shap_beeswarm(np.asarray(sh["firmado"]), sh["X"]), "SHAP por hogar para la clase IA severa (beeswarm aproximado)",
                              {"que_muestra": "Cada punto es un hogar de la muestra; su posición horizontal es el efecto de la variable sobre la clase IA severa.", "como_leerlo": "Puntos a la derecha empujan hacia IA severa; el color indica si el valor de la variable es bajo o alto (categóricas en gris).",
                               "limitacion": "Beeswarm aproximado sobre una submuestra; las dummies de una variable categórica se suman."}, "simple"))
        fies = imp[imp["variable"].astype(str).str.split("__").str[-1].str.startswith("fies_")]["importancia"].sum() / tot
        b.append(_tabla("Comparación de la importancia: variables FIES frente a socioeconómicas", pd.DataFrame([{"grupo de variables": "Ítems y puntaje FIES", "participación": float(fies)}, {"grupo de variables": "Socioeconómicas y territoriales", "participación": float(1 - fies)}]),
                        I.i_comparacion_escenarios_shap(float(fies), float(1 - fies))))
        b.append({"tipo": "h2", "texto": "Lectura guiada"})
        top3 = ", ".join(f"«{E.etq(v)}»" for v in imp.head(3)["variable"])
        b.append({"tipo": "lista", "items": [f"Las tres variables más influyentes en el modelo son {top3}.",
                                             f"Las diez primeras variables acumulan {100 * float(imp.head(10)['importancia'].sum() / tot):.1f} % de la importancia total.",
                                             "Un |SHAP| alto significa que la variable cambia mucho la predicción del modelo; no que cause la inseguridad alimentaria.",
                                             "Si los ítems FIES dominan la importancia, el modelo reconstruye el objetivo (escenarios fies/hibrido) en lugar de predecirlo."]})
    b += _bloque_dependencia(x, imp)
    esc = []
    if x.dir_run is not None:
        for p in sorted(x.dir_run.glob("resultado_*.json")):
            r = _json(p)
            a = (r.get("agregado", {}).get("ponderada") or {}).get("f1_macro")
            if a:
                esc.append({"escenario": r["escenario"], "modelo": r["codigo"], "media": a["media"], "ic_inf": a["ic_inf"], "ic_sup": a["ic_sup"]})
    if esc:
        te = pd.DataFrame(esc)
        b.append(x.figura("fig_escenarios", P.mpl_escenarios(te), "F1-macro ponderado por escenario y modelo (CV anidada, IC95)", I.i_escenarios(te), "simple", _datos_escenarios(te)))
        b.append(_tabla("F1-macro ponderado por escenario (FIES frente a socioeconómicas)", te.rename(columns={"ic_inf": "IC95 inf", "ic_sup": "IC95 sup"}), I.i_escenarios(te)))
    return b


def _mejor_modelo(x: _Ctx) -> str | None:
    """Código del modelo con mayor F1-macro ponderado de la CV anidada entre los de la corrida (None si no hay resultados)."""
    mejor, mejor_f1 = None, -1.0
    for c in x.codigos:
        r = x.res.get(f"{x.escenario}_{c}")
        f1 = ((r or {}).get("agregado", {}).get("ponderada") or {}).get("f1_macro", {}).get("media") if r else None
        if f1 is not None and f1 > mejor_f1:
            mejor, mejor_f1 = c, float(f1)
    return mejor


def _orden_variables(x: _Ctx, cod: str, res: dict[str, Any], imp: pd.DataFrame | None) -> tuple[list[str], str]:
    """Variables crudas del modelo ordenadas por importancia (SHAP agregado a la columna cruda; si no, permutación). Devuelve (orden, fuente)."""
    cols = list(res["features_num"]) + list(res["features_cat"])

    def agregar(df_imp: pd.DataFrame) -> pd.Series:
        crudas = XP.agrupar_a_columna_cruda([str(v) for v in df_imp["variable"]], cols)
        return pd.Series(df_imp["importancia"].to_numpy(float), index=crudas).groupby(level=0).sum().sort_values(ascending=False)

    f = x.dir_run / "analisis" / f"shap_{x.escenario}_{cod}.json" if x.dir_run is not None else None
    if f is not None:
        if f.is_file():
            try:
                d = pd.DataFrame(list(_json(f)["importancia"].items()), columns=["variable", "importancia"])
                s = agregar(d)
                s = s[s.index.isin(cols)]
                if len(s):
                    return s.index.tolist(), f"|SHAP| medio guardado de la corrida ({f.name})"
            except (OSError, ValueError, KeyError):
                pass
    if imp is not None and not imp.empty:
        s = agregar(imp)
        s = s[s.index.isin(cols)]
        if len(s):
            return s.index.tolist(), "importancia SHAP global del informe (todas las explicaciones almacenadas)"
    if x.dir_run is not None:
        f = x.dir_run / "analisis" / f"permutacion_{x.escenario}_{cod}.csv"
        if f.is_file():
            d = pd.read_csv(f).sort_values("importancia", ascending=False)
            v = [c for c in d["variable"].astype(str) if c in cols]
            if v:
                return v, f"importancia por permutación guardada de la corrida ({f.name})"
    return cols, "orden de las variables del modelo (no hay importancias guardadas)"


def _calcular_dependencia(x: _Ctx, imp: pd.DataFrame | None) -> dict[str, Any]:
    """PDP/ALE del mejor modelo sobre una submuestra ponderada de la VALIDACIÓN (15 %). Nunca toca la Prueba bloqueada."""
    import joblib
    if x.dir_run is None:
        raise ValueError("no hay una corrida de entrenamiento seleccionada")
    cod = _mejor_modelo(x)
    if cod is None:
        raise ValueError("la corrida no tiene resultados del escenario seleccionado")
    res = x.res[f"{x.escenario}_{cod}"]
    part, _ = cargar_particion(x.df, x.dir_run / "particion.json")
    pipe = joblib.load(x.dir_run / res["final"]["modelo"])
    val = x.df.iloc[part["validacion"]].reset_index(drop=True)
    Xv = preparar_matriz(val, res["features_num"], res["features_cat"])
    orden, fuente = _orden_variables(x, cod, res, imp)
    n_muestra = N_DEP_MUESTRA_DETALLE if (x.shap or {}).get("detalle") else N_DEP_MUESTRA
    dep = XP.pdp_ale_principales(pipe, Xv, val[C.COL_PESO].to_numpy(float), orden[:N_DEP_VARIABLES], C.CLASE_SEVERA, n_muestra=n_muestra)
    dep.update({"modelo": cod, "fuente_orden": fuente, "conjunto": "Validación (15 %)"})
    return dep


def _fmt_valor(t: pd.DataFrame) -> pd.DataFrame:
    """Valores legibles de la tabla PDP/ALE: categorías con etiqueta y números a 4 cifras significativas (copia)."""
    out = t.copy()
    out["valor"] = [E.valor_etq(v, k) if isinstance(k, str) else f"{k:.4g}" for v, k in zip(out["variable"], out["valor"])]
    return out


def _bloque_dependencia(x: _Ctx, imp: pd.DataFrame | None) -> list[dict[str, Any]]:
    """PDP y ALE de las variables principales: SIEMPRE se calculan (no dependen de la casilla «Calcular SHAP y ALE»)."""
    b: list[dict[str, Any]] = [{"tipo": "h2", "texto": "Dependencia parcial (PDP) y efectos locales acumulados (ALE)"}]
    dep = (x.shap or {}).get("dependencia")
    try:
        if not dep:
            dep = _calcular_dependencia(x, imp)
    except Exception as exc:  # noqa: BLE001 - se declara el motivo; no se oculta
        logger.exception("PDP/ALE")
        b.append(_vacio(f"Sin datos disponibles de PDP/ALE: no fue posible calcularlos ({type(exc).__name__}: {exc})."))
        return b
    curvas = dep["curvas"]
    if not curvas:
        b.append(_vacio("Sin datos disponibles de PDP/ALE: ninguna de las variables principales admitió el cálculo. Omitidas: "
                        + "; ".join(f"{o['variable']} ({o['motivo']})" for o in dep.get("omitidas", []))))
        return b
    b.append(_p(f"Se calcularon PDP y ALE para las {len(curvas)} variables más importantes del modelo {dep.get('modelo', '')} (mayor F1-macro de la CV anidada; orden por {dep.get('fuente_orden', 'importancia')}) "
                f"sobre P(IA severa). Conjunto: {dep.get('conjunto', 'Validación (15 %)')} (el modelo no se entrenó con él y la Prueba bloqueada no se usó). "
                f"Tamaño de muestra: {dep['n_muestra']:,} hogares de {dep['n_base']:,} (submuestra aleatoria ponderada por el factor de expansión, sin reemplazo, semilla {C.SEED}). "
                "En variables categóricas solo se reporta PDP por categoría; el ALE no aplica."))
    if dep.get("omitidas"):
        b.append(_p("Variables omitidas por no admitir el cálculo: " + "; ".join(f"{E.etq(o['variable'])} [{o['variable']}] ({o['motivo']})" for o in dep["omitidas"]) + "."))
    pie = {"que_muestra": "Fila superior: probabilidad media predicha de IA severa al fijar la variable en cada valor (PDP). Fila inferior: efecto local acumulado centrado (ALE).",
           "como_leerlo": "Pendiente positiva: al subir la variable el modelo predice mayor probabilidad de IA severa. El ALE se lee respecto del promedio (0) y es más robusto cuando las variables están correlacionadas.",
           "limitacion": f"Describen al modelo, no causalidad. PDP asume independencia entre variables; ALE depende de los intervalos. Submuestra ponderada de {dep['n_muestra']:,} hogares de Validación; los extremos de cada variable tienen pocos hogares."}
    for i in range(0, len(curvas), 4):
        grupo = curvas[i:i + 4]
        nombre = "fig_pdp_ale" if i == 0 else f"fig_pdp_ale_{i // 4 + 1}"
        b.append(x.figura(nombre, P.mpl_dependencia_doble(grupo), f"PDP y ALE de la probabilidad de IA severa ({', '.join(E.etq(c['variable']) for c in grupo)})", pie, "doble", _datos_dependencia(grupo)))
    tabla = XP.tabla_dependencia(curvas)
    tabla["variable"] = tabla["variable"].astype(str)
    resumen = []
    for c in curvas:
        if c["tipo"] == "numerica":
            pr = np.asarray(c["pdp"]["prob_media"], dtype=float)
            sent = "creciente" if pr[-1] - pr[0] > 0.005 else "decreciente" if pr[-1] - pr[0] < -0.005 else "casi plana"
            resumen.append({"variable": c["variable"], "tipo": "numérica", "PDP al mínimo (p5)": float(pr[0]), "PDP al máximo (p95)": float(pr[-1]), "rango PDP": float(pr.max() - pr.min()), "tendencia global": sent})
        else:
            pr = np.asarray(c["prob_media"], dtype=float)
            resumen.append({"variable": c["variable"], "tipo": "categórica", "PDP al mínimo (p5)": float(pr.min()), "PDP al máximo (p95)": float(pr.max()), "rango PDP": float(pr.max() - pr.min()), "tendencia global": "ver categorías"})
    b.append(_tabla("Resumen de PDP por variable principal (probabilidad media de IA severa)", pd.DataFrame(resumen), pie))
    b.append(_tabla("Valores numéricos de PDP y ALE (para reproducir las figuras)", tabla.assign(valor=lambda d: d["valor"].astype(str)), None, max_filas=400))
    mayor = max(resumen, key=lambda r: r["rango PDP"])
    b.append({"tipo": "lista", "items": [
        f"La variable con mayor variación de la probabilidad media de IA severa es «{E.etq(mayor['variable'])}» (rango PDP {mayor['rango PDP']:.3f}).",
        "Un rango PDP pequeño indica que, en promedio, el modelo casi no cambia su predicción al variar esa variable por sí sola; puede ser relevante en interacción con otras.",
        "Estas curvas describen al modelo y no demuestran que cambiar la variable cambie la inseguridad alimentaria."]})
    x.hechos["dependencia_n"] = dep["n_muestra"]
    return b


# ------------------------------ 6. Equidad ------------------------------
def _grupos_equidad(df: pd.DataFrame) -> list[str]:
    """Prepara (in situ, sobre la copia ``df``) las columnas de grupo de equidad y devuelve las disponibles.

    Los grupos de edad y educación se definen UNA sola vez, con los cortes del contrato de datos (fuentes_publicas.py / data_loader.py):
    edad <30, 30-44, 45-59, 60+; nivel_educ_jefe 0-1 sin_primaria, 2-3 primaria, 4-5 secundaria, >=6 superior. Si el dataset ya trae la
    columna del contrato, se respeta tal cual (solo se normalizan 'nan'/'None' a nulo); NUNCA se redefine con otros cortes, porque el
    informe dejaría de coincidir con equidad.csv, el módulo Equidad y la tabla de datos."""
    d = df
    for g in ("grupo_edad_jefe", "grupo_educ_jefe"):
        if g in d.columns:
            d[g] = d[g].astype(object).where(~d[g].astype(str).isin(["nan", "None", "<NA>", ""]), np.nan)
    if "grupo_edad_jefe" not in d.columns and "edad_jefe" in d.columns:
        d["grupo_edad_jefe"] = pd.cut(d["edad_jefe"], CORTES_EDAD, right=False, labels=ETIQ_EDAD).astype(object)
    if "grupo_educ_jefe" not in d.columns and "nivel_educ_jefe" in d.columns:
        d["grupo_educ_jefe"] = pd.cut(d["nivel_educ_jefe"], CORTES_EDUC, right=False, labels=ETIQ_EDUC).astype(object)
    if "pobreza_monetaria" in d:
        d["pobreza"] = d["pobreza_monetaria"].map({1: "pobre", 0: "no_pobre", 1.0: "pobre", 0.0: "no_pobre"})
    return [g for g in ("sexo_jefe", "area", "region_natural", "grupo_edad_jefe", "grupo_educ_jefe", "pobreza") if g in d.columns]


# Cortes del contrato (intervalos [a, b) sobre valores enteros), idénticos a fsai.fuentes_publicas.
CORTES_EDAD = [-np.inf, 30, 45, 60, np.inf]
ETIQ_EDAD = ["<30", "30-44", "45-59", "60+"]
CORTES_EDUC = [-np.inf, 2, 4, 6, np.inf]
ETIQ_EDUC = ["sin_primaria", "primaria", "secundaria", "superior"]


def conteos_equidad_tabla(t: pd.DataFrame) -> dict[str, dict[str, int]]:
    """{variable_de_grupo: {grupo: n}} leído de la tabla de brechas (una sola fila por grupo; se ignoran los dos objetivos repetidos)."""
    out: dict[str, dict[str, int]] = {}
    for g, sub in t.groupby("grupo_col"):
        out[str(g)] = {str(r["grupo"]): int(r["n"]) for _, r in sub.iterrows()}
    return out


def _seccion_equidad(x: _Ctx) -> list[dict[str, Any]]:
    disp = [c for c in x.codigos if c in x.preds]
    if not disp:
        return [_vacio("Sin datos disponibles: la equidad requiere predicciones por hogar (Validación o Prueba) de al menos un modelo.")]
    agp = x.hechos.get("agregado")
    mejor = disp[0]
    if agp is not None and not agp.empty:
        f1 = agp[(agp["metrica"] == "f1_macro") & agp["modelo"].isin(disp)].sort_values("media", ascending=False)
        if len(f1):
            mejor = f1.iloc[0]["modelo"]
    p, fuente = x.preds[mejor]
    base = x.df.drop_duplicates("hogar_uid").set_index("hogar_uid")
    uid = pd.Series(p["uid"].astype(str))
    ok = uid.isin(base.index).to_numpy()
    if ok.sum() < 100:
        return [_vacio("Sin datos disponibles: las predicciones guardadas no se pudieron enlazar con los hogares del dataset activo (¿otro dataset?).")]
    d = base.loc[uid[ok]].reset_index()   # copia: _grupos_equidad no modifica x.df
    y, pred = p["y"].astype(int)[ok], p["proba"].argmax(1)[ok]
    grupos = _grupos_equidad(d)
    tablas = []
    for g in grupos:
        try:
            t = brechas_por_grupo(d.dropna(subset=[g]).reset_index(drop=True), y[d[g].notna().to_numpy()], pred[d[g].notna().to_numpy()], g, [2, 3], n_boot=x.n_boot)
            tablas.append(t)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Equidad omitida para %s: %s", g, exc)
    if not tablas:
        return [_vacio("Sin datos disponibles: no fue posible calcular brechas por grupo con los datos actuales.")]
    t = pd.concat(tablas, ignore_index=True)
    # Autoverificación: los n de la tabla deben ser el recuento directo de los hogares enlazados con las predicciones.
    cont = conteos_equidad_tabla(t)
    for g, por_grupo in cont.items():
        directo = d[g].dropna().astype(str).value_counts().to_dict()
        if {k: int(v) for k, v in directo.items()} != por_grupo:
            raise ValueError(f"Conteos de equidad inconsistentes para {g}: tabla={por_grupo} recuento={directo}")
    x.hechos["equidad_conteos"] = cont
    x.hechos["equidad_alertas"] = int(t["alerta"].sum())
    x.hechos["equidad_n"] = int((~t["es_referencia"]).sum())
    b: list[dict[str, Any]] = [_p(f"Modelo evaluado: {mejor} (mayor F1-macro entre los seleccionados), predicciones de {fuente} ({ok.sum():,} hogares). Clase positiva: IA moderada o severa (niveles 2-3); "
                                  f"brecha = TPR del grupo menos TPR del grupo de referencia (el más numeroso); IC95 bootstrap percentil con {x.n_boot} réplicas; alerta si |brecha| > {int(100 * C.UMBRAL_BRECHA_TPR)} pp.")]
    pie = I.i_equidad(t, 100 * C.UMBRAL_BRECHA_TPR)
    b.append(_tabla("Brechas de TPR, FPR y error por grupo (IA moderada/severa)", t.rename(columns={"grupo_col": "variable", "ic_inf": "IC95 inf", "ic_sup": "IC95 sup"}).drop(columns=["es_referencia"]), pie, max_filas=60))
    b.append(x.figura("fig_equidad", P.mpl_equidad(t, C.UMBRAL_BRECHA_TPR), "Brecha de sensibilidad (TPR) por grupo frente a su referencia, con IC95", pie, "doble", _datos_equidad(t)))
    return b


# ------------------------------ 7. Limitaciones y 8. Reproducibilidad ------------------------------
def _seccion_limitaciones(x: _Ctx) -> list[dict[str, Any]]:
    b: list[dict[str, Any]] = [{"tipo": "lista", "items": [
        "Uso responsable: esta herramienta es de apoyo al análisis y a la investigación. Sus resultados no deben usarse para negar, restringir ni condicionar ayudas sociales a ningún hogar.",
        "Las explicaciones (SHAP, importancias, PDP/ALE, brechas de equidad) son asociaciones predictivas, no evidencia de causalidad.",
        "Las cifras deben leerse junto con sus intervalos de confianza y las limitaciones del diseño muestral: los IC de prevalencia incorporan los conglomerados pero no la estratificación, y los IC bootstrap de métricas y de equidad ignoran estrato y conglomerado (son aproximados).",
    ]}]
    tec = ["Los cortes de los niveles FIES sobre el puntaje crudo y los parámetros Rasch para el Perú están marcados [POR VALIDAR]; no se afirma calibración peruana de la escala.",
           "Las cifras de CV anidada provienen de pocos pliegues (5 externos): los IC son amplios y las pruebas por pliegue tienen baja potencia."]
    esc = {x.escenario} | ({r["escenario"] for r in x.res.values()} if x.res else set())
    if esc & {"fies", "hibrido"}:
        tec.append("Los escenarios «fies» e «hibrido» usan los ocho ítems FIES como predictores del nivel que se deriva de ellos: son tareas de reconstrucción por circularidad y su desempeño casi perfecto NO mide capacidad predictiva. Solo «socio» es predicción real.")
    df = x.df
    if "informalidad" in df.columns and df["informalidad"].isna().all():
        tec.append("La variable de informalidad laboral no está disponible (100 % faltante) en este dataset y se descartó de los predictores.")
    if "fies_rasch_logit" in df.columns and df["fies_rasch_logit"].isna().all():
        tec.append("No hay medida Rasch (fies_rasch_logit) disponible en este dataset: la escala se resume con el puntaje crudo y sus niveles.")
    if x.hechos.get("trials_bajos"):
        tec.append("El presupuesto de ensayos de optimización es inferior al del protocolo (100 por modelo y pliegue): los hiperparámetros y el desempeño son preliminares.")
    if not x.hechos.get("n_prueba"):
        tec.append("La Prueba independiente (15 %) no fue evaluada para los modelos de este informe: las cifras de predicciones provienen de Validación y no deben tomarse como desempeño final.")
    tec.append("La partición agrupa por hogar_uid; hogares del mismo conglomerado pueden aparecer en Desarrollo y Prueba, lo que puede inflar ligeramente el desempeño frente a una validación por conglomerado (esquema V1c).")
    b.append(_p("Limitaciones técnicas de este análisis:"))
    b.append({"tipo": "lista", "items": tec})
    return b


def _seccion_reproducibilidad(x: _Ctx) -> list[dict[str, Any]]:
    filas = [{"elemento": "Semilla global", "valor": x.cfg.get("seed", C.SEED)},
             {"elemento": "Hash de constantes del protocolo (SHA-256, 16 hex)", "valor": C.hash_configuracion(C.constantes_protocolo())}]
    ds = x.sql("SELECT hash_archivo FROM fsai.datasets WHERE dataset_id = :d", d=int(x.dataset_id))
    if len(ds):
        filas.append({"elemento": "Hash del archivo de datos del dataset activo (fsai.datasets)", "valor": str(ds.iloc[0]["hash_archivo"])})
    if x.dir_run is not None:
        cfgp = x.dir_run / "config.json"
        if x.cfg.get("hash_datos"):
            filas.append({"elemento": "Hash de datos usado por la corrida (config.json)", "valor": x.cfg["hash_datos"]})
        if x.cfg.get("huella_particion"):
            filas.append({"elemento": "Huella SHA-256 de la partición", "valor": x.cfg["huella_particion"]})
        if cfgp.is_file():
            filas.append({"elemento": "SHA-256 de config.json", "valor": _sha256(cfgp)})
        for c in x.codigos:
            for nombre in (f"resultado_{x.escenario}_{c}.json",):
                f = x.dir_run / nombre
                if f.is_file():
                    filas.append({"elemento": f"SHA-256 de {nombre}", "valor": _sha256(f)})
            mj = (x.res.get(f"{x.escenario}_{c}", {}).get("final") or {}).get("modelo")
            if mj and (x.dir_run / mj).is_file():
                filas.append({"elemento": f"SHA-256 del modelo final {c}", "valor": _sha256(x.dir_run / mj)})
    b: list[dict[str, Any]] = [_tabla("Hashes de datos y artefactos", pd.DataFrame(filas), {"que_muestra": "Huellas que identifican de forma única los datos, la partición y los artefactos que producen estas cifras.", "como_leerlo": "Si un hash cambia, el artefacto no es el mismo que el de este informe.", "limitacion": "Las huellas de modelos dependen del serializador y de las versiones de librerías."}, max_filas=60)]
    cmd = ["# 1) Cargar el dataset en PostgreSQL y entrenar con CV anidada (ver despliegue_local/LEEME.md)",
           "cd protocolo/codigo_verificacion/foodsecurity_ai",
           f"python train_pipeline.py --ruta-datos <ruta_al_archivo_analitico> --escenario {x.escenario} --modelos {' '.join(x.codigos) or 'M1 M2 M3 M4 M5'} "
           f"--trials {x.cfg.get('trials', C.N_TRIALS)} --cv-anidada --nombre-experimento <nombre>",
           "# 2) Regenerar este informe desde la aplicación: módulo «Reportes» con la misma corrida, escenario y modelos.",
           "# 3) Verificar el entorno: python -m pip freeze  (compare con la tabla de versiones de la sección 2)"]
    b.append(_p("Comandos para reproducir (sustituya los marcadores <...>; no incluya credenciales en la línea de comandos):"))
    b.append({"tipo": "codigo", "texto": "\n".join(cmd)})
    return b


def _seccion_anexo(x: _Ctx) -> list[dict[str, Any]]:
    """Anexo A: diccionario de variables (etiqueta legible, código original, unidad, tipo, descripción y avisos por validar)."""
    usadas = [c for c in x.df.columns if c in E.DICCIONARIO]
    resto = [c for c in x.df.columns if c not in E.DICCIONARIO]
    pred = set()
    for r in x.res.values():
        pred |= set(r.get("features_num", [])) | set(r.get("features_cat", []))
    orden = [c for c in usadas if c in pred] + [c for c in usadas if c not in pred]
    b: list[dict[str, Any]] = [_p("En todo el informe las variables se muestran con su nombre legible; el código original del dataset se conserva entre corchetes en las tablas y aquí. "
                                  "Las descripciones salen del código que construye el dataset a partir de la ENAHO 2025; lo que no pudo verificarse está marcado [POR VALIDAR]. "
                                  "Las variables que son predictoras de los modelos de esta corrida aparecen primero.")]
    t = E.diccionario_df(orden)
    t.insert(2, "Predictora en esta corrida", ["sí" if c in pred else "no" for c in orden])
    b.append(_tabla("Diccionario de variables del dataset analítico", t, {"que_muestra": f"{len(t)} variables del dataset activo con nombre legible, código original, unidad, tipo y descripción.",
                                                                         "como_leerlo": "Use el código original para localizar la columna en los archivos y la base de datos; las etiquetas solo se aplican al mostrar resultados.",
                                                                         "limitacion": "Las definiciones marcadas [POR VALIDAR] no pudieron verificarse con el código ni los datos disponibles."}, max_filas=100))
    if resto:
        b.append(_p("Columnas del dataset sin entrada en el diccionario (se muestran con su código): " + ", ".join(resto) + "."))
    return b


# ------------------------------ Excel: pronóstico, diccionario ampliado y metodología ------------------------------
NIVELES_FIES = {0: "seguridad", 1: "leve", 2: "moderada", 3: "severa"}
N_VARIABLES_PRONOSTICO = 10   # variables explicativas principales que acompañan a cada predicción


def _modelo_pronostico(x: _Ctx) -> str | None:
    """Mismo criterio que la sección de equidad: el modelo con mayor F1-macro ponderado entre los que tienen predicciones guardadas."""
    disp = [c for c in x.codigos if c in x.preds]
    if not disp:
        return None
    agp = x.hechos.get("agregado")
    if agp is not None and not agp.empty:
        f1 = agp[(agp["metrica"] == "f1_macro") & agp["modelo"].isin(disp)].sort_values("media", ascending=False)
        if len(f1):
            return str(f1.iloc[0]["modelo"])
    return disp[0]


def _pronostico(x: _Ctx) -> dict[str, Any] | None:
    """Hoja «Pronóstico»: predicciones por hogar del conjunto evaluado (Prueba si fue evaluada; si no, Validación)."""
    mejor = _modelo_pronostico(x)
    if mejor is None:
        return None
    p, fuente = x.preds[mejor]
    uid = pd.Series(p["uid"].astype(str))
    y, proba, w = p["y"].astype(int), np.asarray(p["proba"], dtype=float), np.asarray(p["w"], dtype=float)
    n = len(y)
    res = x.res.get(f"{x.escenario}_{mejor}")
    cols_modelo = (list(res["features_num"]) + list(res["features_cat"])) if res else []
    orden: list[str] = []
    if res:
        orden, _ = _orden_variables(x, mejor, res, None)
    variables = [v for v in orden if v in x.df.columns][:N_VARIABLES_PRONOSTICO] or [v for v in cols_modelo if v in x.df.columns][:N_VARIABLES_PRONOSTICO]
    base = x.df.drop_duplicates("hogar_uid").set_index("hogar_uid")
    base.index = base.index.astype(str)
    sub = base.reindex(uid)[variables].reset_index(drop=True)
    pref = "T" if fuente == "Prueba" else "V"
    pred = proba.argmax(1)
    df = pd.DataFrame({"id": [f"{pref}{i + 1:05d}" for i in range(n)], "conjunto": fuente})
    for k in range(proba.shape[1]):
        df[f"proba_{k}"] = proba[:, k]
    df["suma_proba"] = proba.sum(axis=1)
    df["clase_predicha"], df["clase_observada"] = pred, y
    df["acierto"] = np.where(pred == y, "Sí", "No")
    df["peso"] = w
    for v in variables:
        col = sub[v]
        if col.dtype == object or str(col.dtype).startswith("string"):
            df[v] = [E.valor_etq(v, a) if isinstance(a, str) else None for a in col]
        else:
            df[v] = col.to_numpy()
    k = proba.shape[1]
    legible = (["ID interno del hogar (número de orden)", "Conjunto evaluado"]
               + [f"P(nivel FIES {j} · {NIVELES_FIES.get(j, j)})" for j in range(k)]
               + ["Suma de probabilidades", "Nivel FIES predicho (argmax)", "Nivel FIES observado", "¿Acierto?", "Peso muestral (factor de expansión)"]
               + [E.etq(v, unidad=True) for v in variables])
    codigo = (["id_hogar_interno", "conjunto"] + [f"proba_{j}" for j in range(k)]
              + ["suma_proba", "clase_predicha", "clase_observada", "acierto", "peso"] + [f"[{v}]" for v in variables])
    exactitud = float((pred == y).mean())
    nota = (f"Modelo {mejor} ({MODELOS_NOMBRE.get(mejor, mejor)}), escenario «{x.escenario}»; predicciones guardadas de {fuente} ({n:,} hogares); mayor F1-macro ponderado entre los modelos con predicciones. "
            f"Exactitud sin ponderar con argmax: {100 * exactitud:.1f} %. El identificador es un número de orden en el archivo de predicciones: no es un identificador de la ENAHO ni contiene datos personales. "
            f"Las {len(variables)} variables explicativas son las de mayor |SHAP| del modelo; el nombre legible está en la fila 5 y el código original en la fila 6. "
            "Las filas con predicción distinta de la observada aparecen sombreadas.")
    return {"df": df, "cab_legible": legible, "cab_codigo": codigo, "col_prob": list(range(2, 2 + k)), "col_clase": [3 + k, 4 + k], "col_acierto": 5 + k, "col_peso": 6 + k,
            "titulo": f"Pronóstico por hogar · modelo {mejor} · {fuente}", "nota": nota, "modelo": mejor, "fuente": fuente, "variables": variables, "exactitud": exactitud}


def _diccionario_extra(pron: dict[str, Any] | None) -> pd.DataFrame:
    filas = []
    if pron:
        desc = {"id_hogar_interno": "Número de orden del hogar en el archivo de predicciones; sirve para filtrar y no identifica al hogar.",
                "conjunto": "Conjunto de datos del que provienen las predicciones (Prueba o Validación).",
                "suma_proba": "Suma de las probabilidades de los cuatro niveles FIES; debe ser 1 salvo redondeo.",
                "clase_predicha": "Nivel FIES con mayor probabilidad (argmax).", "clase_observada": "Nivel FIES observado (0 a 3).",
                "acierto": "Sí si el nivel predicho coincide con el observado.", "peso": "Factor de expansión de la ENAHO usado para ponderar las métricas."}
        for leg, cod in zip(pron["cab_legible"], pron["cab_codigo"]):
            d = desc.get(cod) or (f"Probabilidad que el modelo asigna al nivel {cod.split('_')[1]} de la escala FIES." if cod.startswith("proba_") else
                                   "Variable explicativa del hogar; véase su definición en la tabla superior del diccionario.")
            filas.append({"Campo": leg, "Código": cod.strip("[]"), "Hoja": "Pronóstico", "Descripción": d})
    filas.append({"Campo": "semiancho sup. / inf. (aux. gráfico)", "Código": "", "Hoja": "Hojas Fig_*", "Descripción": "Distancia entre la estimación y el límite superior (o inferior) del IC95; alimenta las barras de error del gráfico nativo."})
    filas.append({"Campo": "IC95 inf / IC95 sup", "Código": "", "Hoja": "Hojas Fig_*", "Descripción": "Límites del intervalo de confianza del 95 % tal como los calcula el informe (método indicado en cada pie de figura)."})
    return pd.DataFrame(filas)


def _metodologia(x: _Ctx, secciones: list[dict[str, Any]], pron: dict[str, Any] | None) -> list[tuple[str, list[str]]]:
    h, cfg = x.hechos, x.cfg
    alcance = [f"Datos: {h.get('n', len(x.df)):,} hogares de la ENAHO con nivel FIES calculado; el objetivo es el nivel FIES de 0 (seguridad) a 3 (inseguridad severa).",
               f"Escenario «{x.escenario}»: " + ("solo variables socioeconómicas y territoriales; es predicción real." if x.escenario == "socio" else "incluye los ítems FIES; es una tarea de reconstrucción, no de predicción."),
               f"Validación: CV anidada con {cfg.get('n_outer', C.N_SPLITS_EXTERNOS)} pliegues externos y {cfg.get('n_inner', C.INNER_FOLDS)} internos dentro de Desarrollo; partición agrupada por «{cfg.get('agrupar_por', 'hogar_uid')}»; semilla {cfg.get('seed', C.SEED)}.",
               "Las métricas ponderadas usan el factor de expansión; los IC de las métricas de CV son t de Student sobre los pliegues externos.",
               "Toda cifra de este libro sale de PostgreSQL y de los artefactos de la corrida; el libro no contiene fórmulas de cálculo (las hojas con gráficos nativos enlazan a las tablas de datos de la misma hoja)."]
    if pron:
        alcance.append(f"Hoja Pronóstico: predicciones de {pron['fuente']} del modelo {pron['modelo']} ({len(pron['df']):,} hogares); la Prueba solo se usa cuando ya fue evaluada en el módulo de Comparación estadística.")
    lectura = []
    ag = h.get("agregado")
    if ag is not None and not ag.empty:
        mm = ag[ag["metrica"] == "f1_macro"].sort_values("media", ascending=False)
        if len(mm):
            m = mm.iloc[0]
            txt = f"Desempeño modesto: el mejor F1-macro ponderado de la CV anidada es {m['media']:.3f} (IC95 {m['ic_inf']:.3f}-{m['ic_sup']:.3f}, modelo {m['modelo']})"
            for met, nom in (("roc_auc_ovr", "AUC ROC (OvR)"), ("recall_severa", "recall de IA severa")):
                f = ag[(ag["modelo"] == m["modelo"]) & (ag["metrica"] == met)]
                if len(f):
                    txt += f"; {nom} {f.iloc[0]['media']:.3f}"
            lectura.append(txt + ".")
    if pron:
        lectura.append(f"En la hoja Pronóstico, el nivel predicho coincide con el observado en {100 * pron['exactitud']:.1f} % de los {len(pron['df']):,} hogares: el modelo distingue algo mejor que el azar, pero se equivoca con frecuencia.")
    lectura.append("Las predicciones por hogar describen al modelo y no deben usarse para decidir sobre ningún hogar concreto ni para condicionar ayudas sociales.")
    lim: list[str] = []
    for s in secciones:
        if s["titulo"].startswith("7."):
            for b in s["bloques"]:
                if b["tipo"] == "lista":
                    lim += list(b["items"])
    return [("Alcance y origen de las cifras", alcance), ("Cómo leer el desempeño", lectura), ("Advertencias y limitaciones", lim),
            ("Cómo usar este libro", ["Empiece por el Índice: cada figura del informe tiene una hoja Fig_* con la imagen, la tabla exacta que la genera y, cuando corresponde, un gráfico nativo de Excel enlazado a esa tabla.",
                                      "Las cifras marcadas [POR VALIDAR] (cortes FIES, parámetros Rasch, definiciones de variables) no deben citarse como validadas para el Perú."])]


# ------------------------------ resumen y ensamblado ------------------------------
def _resumen(x: _Ctx) -> list[str]:
    h = x.hechos
    out = []
    if "prev" in h and h["prev"] == h["prev"]:
        out.append(f"El análisis cubre {h['n']:,} hogares (ENAHO) con nivel FIES calculado. La prevalencia ponderada de inseguridad alimentaria moderada o severa es {100 * h['prev']:.1f} % "
                   f"(IC95 aproximado {100 * h['prev_lo']:.1f}-{100 * h['prev_hi']:.1f} %).")
        pa = h.get("prev_area")
        if pa is not None and len(pa) >= 2:
            a, z = pa.iloc[0], pa.iloc[-1]
            out.append(f"Por área, la prevalencia es máxima en «{a['grupo']}» ({100 * a['prevalencia']:.1f} %) y mínima en «{z['grupo']}» ({100 * z['prevalencia']:.1f} %).")
    else:
        out.append("Sin datos disponibles para estimar la prevalencia del dataset activo.")
    pf = h.get("perfil")
    if pf is not None:
        sub, rec = pf["subgrupos"], pf["recuento"]
        r = rec[rec["n"] > 0]
        f = lambda k, g: sub[(sub["clave"] == k) & sub["grupo"].str.startswith(g)].iloc[0]  # noqa: E731
        po, mc, am = f("pobreza", "Con"), f("menor5", "Con"), f("adulto_mayor", "Con")
        c = lambda v: f"{100 * v:.1f}".replace(".", ",")  # noqa: E731
        out.append(f"Perfil descriptivo (no es un índice validado): la prevalencia fue {c(po['prevalencia'])} % en hogares en pobreza monetaria, {c(mc['prevalencia'])} % con menores de 5 años y "
                   f"{c(am['prevalencia'])} % con adultos mayores; según el recuento simple de estas características fue {c(r.iloc[0]['prevalencia'])} % con ninguna y {c(r.iloc[-1]['prevalencia'])} % con {int(r.iloc[-1]['recuento'])} (n = {int(r.iloc[-1]['n'])}). "
                   "Asociación descriptiva, sin desenlaces nutricionales directos.")
    ag = h.get("agregado")
    if ag is not None and not ag.empty:
        f1 = ag[ag["metrica"] == "f1_macro"].sort_values("media", ascending=False)
        rs = ag[ag["metrica"] == "recall_severa"].set_index("modelo")
        if len(f1):
            m = f1.iloc[0]
            txt = f"En la CV anidada (escenario «{x.escenario}»), el mejor F1-macro ponderado corresponde a {m['modelo']} ({m['media']:.3f}; IC95 {m['ic_inf']:.3f}-{m['ic_sup']:.3f})"
            if m["modelo"] in rs.index:
                txt += f", con recall de IA severa {rs.loc[m['modelo'], 'media']:.3f}"
            out.append(txt + ".")
    else:
        out.append("Sin datos disponibles de desempeño de modelos.")
    pr = h.get("pruebas")
    if pr is not None and len(pr):
        col = "p_ajustado" if "p_ajustado" in pr else "p_valor"
        out.append(f"De {len(pr)} contrastes entre modelos, {int((pr[col] < C.ALPHA).sum())} son significativos (p ajustado < {C.ALPHA}).")
    if "equidad_n" in h:
        out.append(f"En equidad, {h['equidad_alertas']} de {h['equidad_n']} grupos superan la alerta de {int(100 * C.UMBRAL_BRECHA_TPR)} pp de brecha de sensibilidad.")
    out.append("La Prueba independiente fue evaluada en " + (f"{h['n_prueba']} modelo(s)." if h.get("n_prueba") else "ningún modelo: las cifras de predicciones provienen de Validación y son preliminares."))
    out.append("Consulte la sección 7 para las limitaciones y el uso responsable de estos resultados.")
    return out


def generar_reporte(engine, dataset_id: int, df: pd.DataFrame, dir_run: Path | None, escenario: str, codigos: list[str], incluir: list[str],
                    dir_salida: Path, formatos: list[str], shap: dict[str, Any] | None = None, titulo: str | None = None,
                    version_app: str = "", n_boot: int = 200) -> dict[str, Any]:
    """Genera figuras (PNG 600 DPI) y documentos. Devuelve {'figuras', 'archivos', 'secciones'}.

    ``shap`` (opcional): {"importancia": DataFrame[variable, importancia], "firmado": ndarray (n, p), "X": DataFrame, "curvas": [dict PDP/ALE], "fuente": str}.
    """
    dir_salida.mkdir(parents=True, exist_ok=True)
    x = _Ctx(engine, dataset_id, df.copy(), Path(dir_run) if dir_run else None, escenario, list(codigos), dir_salida, n_boot, shap)
    if x.dir_run is not None:
        try:
            x.cfg = _json(x.dir_run / "config.json")
        except (OSError, ValueError):
            x.cfg = {}
        for c in x.codigos:
            f = x.dir_run / f"resultado_{escenario}_{c}.json"
            if f.is_file():
                x.res[f"{escenario}_{c}"] = _json(f)
        x.codigos = [c for c in x.codigos if f"{escenario}_{c}" in x.res] or x.codigos
    constructores: dict[str, Callable[[_Ctx], list[dict[str, Any]]]] = {
        "datos": _seccion_datos, "perfil": _seccion_perfil, "metodos": _seccion_metodos, "resultados": _seccion_resultados, "comparacion": _seccion_comparacion,
        "interpretabilidad": _seccion_interpretabilidad, "equidad": _seccion_equidad, "reproducibilidad": _seccion_reproducibilidad}
    orden = ["datos", "perfil", "metodos", "resultados", "comparacion", "interpretabilidad", "equidad", "limitaciones", "reproducibilidad", "anexo"]
    secciones = []
    for clave in orden:
        if clave not in ("limitaciones", "anexo") and clave not in incluir:
            continue
        try:
            bloques = (_seccion_limitaciones if clave == "limitaciones" else _seccion_anexo if clave == "anexo" else constructores[clave])(x)
        except Exception as exc:  # noqa: BLE001 - una sección con error no debe tumbar el informe
            logger.exception("Sección %s", clave)
            bloques = [_vacio(f"Sin datos disponibles: no fue posible construir esta sección ({type(exc).__name__}: {exc}).")]
        etq = "7. Limitaciones y uso responsable" if clave == "limitaciones" else "Anexo A. Diccionario de variables" if clave == "anexo" else ETIQUETAS_SECCION[clave]
        secciones.append({"titulo": etq, "hoja": HOJAS[clave], "bloques": bloques})
    ds = x.sql("SELECT nombre, anio FROM fsai.datasets WHERE dataset_id = :d", d=int(dataset_id))
    nombre_ds = f"{ds.iloc[0]['nombre']} ({ds.iloc[0]['anio']})" if len(ds) else "archivo tabular"
    hash_cfg = C.hash_configuracion({**C.constantes_protocolo(), "escenario": escenario, "modelos": x.codigos, "corrida": x.dir_run.name if x.dir_run else None})
    ahora = datetime.now()
    titulo = titulo or f"Informe técnico FoodSecurity AI · inseguridad alimentaria y vulnerabilidad nutricional · escenario {escenario}"
    portada = [("Dataset", nombre_ds), ("Hogares analizados", f"{len(df):,}"), ("Corrida de modelos", x.dir_run.name if x.dir_run else "sin corrida seleccionada"),
               ("Escenario / modelos", f"{escenario} / {', '.join(x.codigos) or '—'}"), ("Fecha de generación", ahora.strftime("%Y-%m-%d %H:%M")),
               ("Versión de la aplicación", version_app or "—"), ("Hash de configuración", hash_cfg), ("Semilla", x.cfg.get("seed", C.SEED))]
    pron = None
    try:
        pron = _pronostico(x)
    except Exception:  # noqa: BLE001 - la hoja Pronóstico no debe tumbar el informe
        logger.exception("Pronóstico")
    doc = {"titulo": titulo, "subtitulo": "FoodSecurity AI · Predicción de inseguridad alimentaria y vulnerabilidad nutricional de los hogares mediante inteligencia artificial explicable", "encabezado": f"FoodSecurity AI · {nombre_ds}",
           "portada": portada, "resumen": _resumen(x), "secciones": secciones,
           "excel_extra": {"pronostico": pron, "metodologia": _metodologia(x, secciones, pron), "diccionario_extra": _diccionario_extra(pron)}}
    base = dir_salida / f"informe_foodsecurity_ai_{escenario}_{ahora:%Y%m%d_%H%M%S}"
    archivos: dict[str, Path] = {}
    if "PDF" in formatos:
        archivos["PDF"] = exportar_pdf(doc, base.with_suffix(".pdf"))
    if "Word" in formatos:
        archivos["Word"] = exportar_word(doc, base.with_suffix(".docx"))
    if "Excel" in formatos:
        archivos["Excel"] = exportar_excel(doc, base.with_suffix(".xlsx"), {"corrida": x.dir_run.name if x.dir_run else "", "escenario": escenario, "dataset_id": dataset_id,
                                                                           "hash_configuracion": hash_cfg, "dpi_figuras": P.DPI_EXPORTACION, "version_app": version_app})
    return {"figuras": x.figuras, "archivos": archivos, "secciones": [s["titulo"] for s in secciones]}
