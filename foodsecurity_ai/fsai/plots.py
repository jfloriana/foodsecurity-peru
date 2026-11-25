"""plots.py - Figuras Plotly (interfaz) y Matplotlib (exportación a 600 DPI, estilo de revista) de FoodSecurity AI.

Las funciones solo dibujan lo que reciben; no calculan métricas. ``guardar_mpl`` fija 600 DPI (kaleido no está
instalado, por lo que la exportación de alta resolución usa Matplotlib). Las figuras de exportación usan la paleta
Okabe-Ito (segura para daltonismo) y dos anchos de revista: columna simple (8,5 cm) y doble (17,8 cm).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from fsai.etiquetas import etq, valor_etq  # noqa: E402  (solo presentación: nunca renombra columnas de entrenamiento)

# Paleta Okabe-Ito (colorblind-safe); la misma en pantalla y en exportación.
AZUL, ROJO, VERDE, GRIS = "#0072B2", "#D55E00", "#009E73", "#6b7280"
PALETA = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
COLOR_MODELO = {"M1": "#0072B2", "M2": "#E69F00", "M3": "#009E73", "M4": "#CC79A7", "M5": "#D55E00"}
ANCHO_SIMPLE = 3.35   # pulgadas (8,5 cm): columna simple de revista
ANCHO_DOBLE = 7.0     # pulgadas (17,8 cm): doble columna
DPI_EXPORTACION = 600


# ------------------------------ Plotly (UI) ------------------------------
def fig_objetivo(df: pd.DataFrame, col: str = "fies_nivel", peso: str = "factor_expansion") -> go.Figure:
    sin = df[col].value_counts(normalize=True).sort_index()
    con = (df.groupby(col)[peso].sum() / df[peso].sum()).sort_index()
    f = go.Figure([go.Bar(x=sin.index.astype(str), y=sin.values, name="Sin ponderar", marker_color=AZUL),
                   go.Bar(x=con.index.astype(str), y=con.values, name="Ponderada (factor de expansión)", marker_color="#e9a23b")])
    f.update_layout(barmode="group", xaxis_title="Nivel FIES (0-3)", yaxis_title="Proporción de hogares", yaxis_tickformat=".0%", margin={"t": 20})
    return f


def fig_hist(df: pd.DataFrame, var: str, color: str | None = None) -> go.Figure:
    f = px.histogram(df, x=var, color=color, nbins=40, marginal="box", opacity=0.8, color_discrete_sequence=PALETA, labels={var: etq(var, True)})
    f.update_layout(margin={"t": 20}, yaxis_title="Hogares")
    return f


def fig_barras_cat(df: pd.DataFrame, var: str) -> go.Figure:
    s = df[var].astype(str).value_counts(normalize=True).head(25)
    f = go.Figure(go.Bar(x=[valor_etq(var, i) for i in s.index], y=s.values, marker_color=AZUL))
    f.update_layout(yaxis_tickformat=".0%", yaxis_title="Proporción", xaxis_title=etq(var), margin={"t": 20})
    return f


def fig_corr(corr: pd.DataFrame) -> go.Figure:
    f = go.Figure(go.Heatmap(z=corr.values, x=[etq(c) for c in corr.columns], y=[etq(c) for c in corr.index], zmin=-1, zmax=1, colorscale="RdBu_r", colorbar={"title": "r"}))
    f.update_layout(height=620, margin={"t": 20})
    return f


def fig_composicion(tabla: pd.DataFrame, titulo: str = "") -> go.Figure:
    f = go.Figure([go.Bar(name=str(c), x=tabla.index.astype(str), y=tabla[c].values) for c in tabla.columns])
    f.update_layout(barmode="group", yaxis_tickformat=".0%", yaxis_title="Proporción dentro de la partición", title=titulo, margin={"t": 40})
    return f


def fig_nested(folds_valores: list[float], metrica: str, ic: Mapping[str, float] | None = None) -> go.Figure:
    f = go.Figure(go.Scatter(x=[f"Pliegue {i + 1}" for i in range(len(folds_valores))], y=folds_valores, mode="markers+lines",
                             marker={"size": 11, "color": AZUL}, name=metrica))
    if ic and not np.isnan(ic.get("media", np.nan)):
        f.add_hline(y=ic["media"], line_dash="dash", line_color=ROJO, annotation_text=f"media {ic['media']:.3f}")
        if not np.isnan(ic.get("ic_inf", np.nan)):
            f.add_hrect(y0=ic["ic_inf"], y1=ic["ic_sup"], fillcolor=ROJO, opacity=0.1, line_width=0)
    f.update_layout(yaxis_title=metrica, xaxis_title="Pliegue externo (una evaluación en outer_val por pliegue)", margin={"t": 20})
    return f


def fig_curva_optuna(trials: list[Mapping[str, Any]]) -> go.Figure:
    v = [t["valor"] for t in trials if t.get("valor") is not None]
    f = go.Figure()
    if v:
        f.add_trace(go.Scatter(x=list(range(len(v))), y=v, mode="markers", name="Ensayo", marker={"color": GRIS}))
        f.add_trace(go.Scatter(x=list(range(len(v))), y=np.maximum.accumulate(v), mode="lines", name="Mejor acumulado", line={"color": ROJO}))
    f.update_layout(xaxis_title="Ensayo (acumulado en la corrida)", yaxis_title="F1-macro interno ponderado", margin={"t": 20})
    return f


def fig_confusion(cm: np.ndarray, normalizada: bool = False) -> go.Figure:
    etiquetas = [f"Nivel {i}" for i in range(cm.shape[0])]
    texto = np.vectorize(lambda x: f"{x:.1%}" if normalizada else f"{x:,.0f}")(cm)
    f = go.Figure(go.Heatmap(z=cm, x=etiquetas, y=etiquetas, text=texto, texttemplate="%{text}", colorscale="Blues", showscale=False))
    f.update_layout(xaxis_title="Predicho", yaxis_title="Real", yaxis_autorange="reversed", margin={"t": 20}, height=420)
    return f


def fig_roc(cv: Mapping[str, Any]) -> go.Figure:
    f = go.Figure()
    for k, (fpr, tpr) in cv["roc"].items():
        f.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Clase {k} (AUC {cv['auc'][k]:.3f})",
                               line={"width": 4 if k == 3 else 2, "color": PALETA[k % 4]}))
    if "roc_macro" in cv:
        f.add_trace(go.Scatter(x=cv["roc_macro"][0], y=cv["roc_macro"][1], mode="lines", name=f"Macro (AUC {cv['auc_macro']:.3f})", line={"dash": "dash", "color": "black"}))
    f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line={"dash": "dot", "color": GRIS}))
    f.update_layout(xaxis_title="1 - especificidad (FPR)", yaxis_title="Sensibilidad (TPR)", margin={"t": 20})
    return f


def fig_pr(cv: Mapping[str, Any]) -> go.Figure:
    f = go.Figure()
    for k, (rc, pr) in cv["pr"].items():
        f.add_trace(go.Scatter(x=rc, y=pr, mode="lines", name=f"Clase {k} (AP {cv['ap'][k]:.3f}; prev. {cv['prevalencia'][k]:.1%})",
                               line={"width": 4 if k == 3 else 2, "color": PALETA[k % 4]}))
    f.update_layout(xaxis_title="Recall", yaxis_title="Precisión", margin={"t": 20})
    return f


def fig_calibracion(curvas: Mapping[str, pd.DataFrame]) -> go.Figure:
    f = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfecta", line={"dash": "dot", "color": GRIS}))
    for nombre, c in curvas.items():
        if len(c):
            f.add_trace(go.Scatter(x=c["p_predicha"], y=c["frecuencia"], mode="lines+markers", name=nombre))
    f.update_layout(xaxis_title="Probabilidad predicha", yaxis_title="Frecuencia observada", margin={"t": 20})
    return f


def fig_importancia(df: pd.DataFrame, col_var: str = "variable", col_val: str = "importancia", top: int = 20, error: str | None = None) -> go.Figure:
    d = df.sort_values(col_val, ascending=False).head(top).iloc[::-1]
    f = go.Figure(go.Bar(x=d[col_val], y=[etq(v) for v in d[col_var]], orientation="h", marker_color=AZUL, error_x={"type": "data", "array": d[error]} if error and error in d else None))
    f.update_layout(xaxis_title=col_val, height=max(300, 24 * len(d) + 80), margin={"t": 20})
    return f


def fig_heatmap(M: np.ndarray, filas: list[str], titulo: str = "", x_titulo: str = "Hogares (ordenados)") -> go.Figure:
    f = go.Figure(go.Heatmap(z=M, y=[etq(c) for c in filas], colorscale="YlOrRd", colorbar={"title": "|SHAP|"}))
    f.update_layout(xaxis_title=x_titulo, title=titulo, height=max(360, 18 * len(filas) + 120), margin={"t": 40}, yaxis_autorange="reversed")
    return f


def fig_mascara(M: np.ndarray, filas: list[str]) -> go.Figure:
    f = go.Figure(go.Heatmap(z=M, y=[etq(c) for c in filas], colorscale=[[0, "#f3f4f6"], [1, "#1f2937"]], showscale=False))
    f.update_layout(xaxis_title="Hogares", height=max(360, 18 * len(filas) + 120), margin={"t": 20}, yaxis_autorange="reversed")
    return f


def fig_eliminacion(curva: pd.DataFrame) -> go.Figure:
    f = go.Figure([go.Scatter(x=curva["k"], y=curva["f1_eliminacion"], mode="lines+markers", name="Eliminación (top-k permutadas)", line={"color": ROJO}),
                   go.Scatter(x=curva["k"], y=curva["f1_insercion"], mode="lines+markers", name="Inserción (top-k restauradas)", line={"color": VERDE})])
    f.update_layout(xaxis_title="k variables (por orden de importancia)", yaxis_title="F1-macro", margin={"t": 20})
    return f


def fig_linea_xy(x, y, xt: str, yt: str, nombre: str = "") -> go.Figure:
    f = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers", name=nombre, line={"color": AZUL}))
    f.update_layout(xaxis_title=xt, yaxis_title=yt, margin={"t": 20})
    return f


def fig_probabilidades(media: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> go.Figure:
    f = go.Figure(go.Bar(x=[f"Nivel {k}" for k in range(len(media))], y=media, marker_color=AZUL,
                         error_y={"type": "data", "symmetric": False, "array": hi - media, "arrayminus": media - lo}))
    f.update_layout(yaxis_title="Probabilidad (media del ensamble, IC bootstrap)", yaxis_tickformat=".0%", margin={"t": 20})
    return f


def fig_contribuciones(nombres: list[str], valores: np.ndarray, top: int = 15) -> go.Figure:
    d = pd.DataFrame({"v": nombres, "x": valores}).reindex(pd.Series(valores).abs().sort_values(ascending=False).index).head(top).iloc[::-1]
    f = go.Figure(go.Bar(x=d["x"], y=[etq(v) for v in d["v"]], orientation="h", marker_color=[ROJO if x > 0 else AZUL for x in d["x"]]))
    f.update_layout(xaxis_title="Contribución (rojo: aumenta la probabilidad de la clase)", height=max(320, 24 * len(d) + 80), margin={"t": 20})
    return f


# ------------------------------ Matplotlib (exportación 600 DPI) ------------------------------
def guardar_mpl(fig, ruta: Path, dpi: int = DPI_EXPORTACION) -> Path:
    """Guarda la figura en PNG a 600 DPI (metadatos pHYs incluidos)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if not getattr(fig, "_fsai_layout_fijo", False):      # algunas figuras (leyenda/título de figura) fijan su propio layout
        fig.tight_layout()
    fig.savefig(ruta, dpi=dpi, format="png", metadata={"Software": "FoodSecurity AI"})
    plt.close(fig)
    return ruta


def mpl_confusion(cm: np.ndarray, normalizada: bool, titulo: str):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:.1%}" if normalizada else f"{v:,.0f}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=9)
    ax.set_xticks(range(cm.shape[1]), [f"Nivel {i}" for i in range(cm.shape[1])])
    ax.set_yticks(range(cm.shape[0]), [f"Nivel {i}" for i in range(cm.shape[0])])
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(titulo, fontsize=10)
    return fig


def mpl_roc(cv: Mapping[str, Any], titulo: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    for k, (fpr, tpr) in cv["roc"].items():
        ax.plot(fpr, tpr, lw=2.4 if k == 3 else 1.4, color=PALETA[k % 4], label=f"Clase {k} (AUC {cv['auc'][k]:.3f})")
    if "roc_macro" in cv:
        ax.plot(*cv["roc_macro"], "k--", label=f"Macro (AUC {cv['auc_macro']:.3f})")
    ax.plot([0, 1], [0, 1], ":", color=GRIS)
    ax.set_xlabel("1 - especificidad")
    ax.set_ylabel("Sensibilidad")
    ax.set_title(titulo, fontsize=10)
    ax.legend(fontsize=7)
    return fig


def mpl_calibracion(curvas: Mapping[str, pd.DataFrame], titulo: str):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.plot([0, 1], [0, 1], ":", color=GRIS, label="Perfecta")
    for n, c in curvas.items():
        if len(c):
            ax.plot(c["p_predicha"], c["frecuencia"], "o-", label=n)
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia observada")
    ax.set_title(titulo, fontsize=10)
    ax.legend(fontsize=7)
    return fig


def mpl_importancia(df: pd.DataFrame, col_var: str, col_val: str, titulo: str, top: int = 15):
    d = df.sort_values(col_val, ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.4, 0.28 * len(d) + 1.4))
    ax.barh([etq(v) for v in d[col_var]], d[col_val], color=AZUL)
    ax.set_xlabel(col_val)
    ax.set_title(titulo, fontsize=10)
    return fig


def mpl_nested(folds_valores: list[float], metrica: str, titulo: str):
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.plot(range(1, len(folds_valores) + 1), folds_valores, "o-", color=AZUL)
    if folds_valores:
        ax.axhline(float(np.mean(folds_valores)), color=ROJO, ls="--", label=f"media {np.mean(folds_valores):.3f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Pliegue externo")
    ax.set_ylabel(metrica)
    ax.set_title(titulo, fontsize=10)
    return fig


def mpl_eliminacion(curva: pd.DataFrame, titulo: str):
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.plot(curva["k"], curva["f1_eliminacion"], "o-", color=ROJO, label="Eliminación")
    ax.plot(curva["k"], curva["f1_insercion"], "o-", color=VERDE, label="Inserción")
    ax.set_xlabel("k variables (por importancia)")
    ax.set_ylabel("F1-macro")
    ax.set_title(titulo, fontsize=10)
    ax.legend(fontsize=8)
    return fig


def mpl_objetivo(df: pd.DataFrame, titulo: str, col: str = "fies_nivel"):
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    s = df[col].value_counts(normalize=True).sort_index()
    ax.bar(s.index.astype(str), s.values, color=AZUL)
    ax.set_xlabel("Nivel FIES")
    ax.set_ylabel("Proporción de hogares")
    ax.set_title(titulo, fontsize=10)
    return fig


# ------------------------------ Matplotlib de revista (informe) ------------------------------
RC_REVISTA = {
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.7, "lines.linewidth": 1.4, "axes.grid": True,
    "grid.color": "#E3E8ED", "grid.linewidth": 0.5, "figure.facecolor": "white", "axes.facecolor": "white",
}


def _col_modelo(nombre: str, i: int = 0) -> str:
    return COLOR_MODELO.get(str(nombre).split("_")[-1], OKABE_ITO[i % len(OKABE_ITO)])


def _figura(ancho: float, alto: float, **kw):
    with plt.rc_context(RC_REVISTA):
        return plt.subplots(figsize=(ancho, alto), **kw)


def mpl_prevalencia_grupos(tabla: pd.DataFrame, etiqueta: str, titulo: str = "", ancho: float = ANCHO_DOBLE):
    """Barras horizontales de prevalencia ponderada con IC. Columnas: grupo, prevalencia, ic_inf, ic_sup."""
    t = tabla.sort_values("prevalencia").reset_index(drop=True)
    fig, ax = _figura(ancho, max(1.8, 0.2 * len(t) + 0.9))
    y = np.arange(len(t))
    err = np.vstack([(t["prevalencia"] - t["ic_inf"]).clip(lower=0).fillna(0), (t["ic_sup"] - t["prevalencia"]).clip(lower=0).fillna(0)])
    ax.barh(y, t["prevalencia"] * 100, color=OKABE_ITO[0], height=0.62, xerr=err * 100,
            error_kw={"elinewidth": 0.8, "capsize": 1.5, "ecolor": "#333333"})
    ax.set_yticks(y, t["grupo"].astype(str))
    ax.set_xlabel("Prevalencia ponderada de inseguridad alimentaria moderada o severa (%)")
    ax.set_ylabel(etiqueta)
    ax.grid(axis="y", visible=False)
    if titulo:
        ax.set_title(titulo)
    return fig


def _gradiente_recuento(k: int) -> list[str]:
    """Gradiente secuencial (claro -> oscuro, legible en escala de grises) para el recuento 0..k-1."""
    base = ["#C6DBEF", "#6BAED6", "#2171B5", "#08306B", "#041F4A"]
    return base[:k] if k <= len(base) else base


def mpl_perfil_subgrupos(sub: pd.DataFrame, titulo: str = ""):
    """Barras agrupadas de prevalencia ponderada con IC95 por característica, con/sin ella.
    Columnas: característica, grupo ('Con...'/'Sin...'), prevalencia, ic_inf, ic_sup, n."""
    cars = list(dict.fromkeys(sub["característica"]))
    fig, ax = _figura(ANCHO_DOBLE, 2.6)
    x = np.arange(len(cars))
    for j, (pref, color, etq) in enumerate((("Con", OKABE_ITO[5], "Con la característica"), ("Sin", OKABE_ITO[4], "Sin la característica"))):
        t = sub[sub["grupo"].str.startswith(pref)].set_index("característica").reindex(cars)
        err = np.vstack([(t["prevalencia"] - t["ic_inf"]).clip(lower=0).fillna(0), (t["ic_sup"] - t["prevalencia"]).clip(lower=0).fillna(0)])
        pos = x + (j - 0.5) * 0.36
        ax.bar(pos, t["prevalencia"] * 100, 0.34, color=color, label=etq, yerr=err * 100, error_kw={"elinewidth": 0.8, "capsize": 1.5, "ecolor": "#333333"})
        for xi, (p, n) in zip(pos, zip(t["prevalencia"], t["n"])):
            ax.text(xi, 1.0, f"n = {int(n):,}".replace(",", " "), ha="center", va="bottom", fontsize=5.8, color="white" if p > 0.08 else "#333333", rotation=90)
    ax.set_xticks(x, cars)
    ax.set_ylabel("Prevalencia ponderada (%)")
    ax.set_ylim(0, 100 * float(np.nanmax(sub["ic_sup"])) * 1.12)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)
    if titulo:
        ax.set_title(titulo)
    return fig


def mpl_perfil_recuento(rec: pd.DataFrame, total: float | None = None, titulo: str = ""):
    """Barras de prevalencia ponderada con IC95 por recuento 0-3, con gradiente de color por recuento y n sobre cada barra.
    Columnas: recuento, n, prevalencia, ic_inf, ic_sup."""
    t = rec[rec["n"] > 0].reset_index(drop=True)
    fig, ax = _figura(ANCHO_SIMPLE, 2.6)
    cols = _gradiente_recuento(len(rec))
    x = np.arange(len(t))
    err = np.vstack([(t["prevalencia"] - t["ic_inf"]).clip(lower=0).fillna(0), (t["ic_sup"] - t["prevalencia"]).clip(lower=0).fillna(0)])
    ax.bar(x, t["prevalencia"] * 100, 0.62, color=[cols[int(r)] for r in t["recuento"]], edgecolor="#333333", linewidth=0.4, yerr=err * 100,
           error_kw={"elinewidth": 0.8, "capsize": 1.5, "ecolor": "#333333"})
    for xi, p, hi, n in zip(x, t["prevalencia"], t["ic_sup"], t["n"]):
        ax.text(xi, hi * 100 + 1.0, f"{100 * p:.1f} %\nn = {int(n):,}".replace(",", " ").replace(".", ","), ha="center", va="bottom", fontsize=6, zorder=5,
                bbox={"fc": "white", "ec": "none", "pad": 0.8, "alpha": 0.9})
    if total is not None and total == total:
        ax.axhline(total * 100, color="#555555", linestyle="--", linewidth=0.8, label=f"Prevalencia total ({100 * total:.1f} %)".replace(".", ","))
        ax.legend(frameon=False, loc="upper left", fontsize=6)
    ax.set_xticks(x, [str(int(r)) for r in t["recuento"]])
    ax.set_xlabel("Recuento de características (0 a 3)")
    ax.set_ylabel("Prevalencia ponderada (%)")
    ax.set_ylim(0, 100 * float(np.nanmax(t["ic_sup"])) * 1.3)
    ax.grid(axis="x", visible=False)
    if titulo:
        ax.set_title(titulo)
    return fig


def mpl_perfil_fies(fr: pd.DataFrame, titulo: str = ""):
    """Barras apiladas (100 %) del nivel FIES por recuento. Columnas: recuento, n y una columna por nivel FIES."""
    niveles = [c for c in fr.columns if c not in ("recuento", "n")]
    t = fr[fr["n"] > 0].reset_index(drop=True)
    fig, ax = _figura(ANCHO_SIMPLE, 2.6)
    colores = ["#9ECAE1", "#F0E442", "#E69F00", "#D55E00"]
    x = np.arange(len(t))
    base = np.zeros(len(t))
    for c, col in zip(niveles, colores):
        v = t[c].to_numpy(float) * 100
        ax.bar(x, v, 0.62, bottom=base, color=col, edgecolor="white", linewidth=0.4, label=c)
        for xi, b0, vv in zip(x, base, v):
            if vv >= 6:
                ax.text(xi, b0 + vv / 2, f"{vv:.0f}".replace(".", ","), ha="center", va="center", fontsize=5.8)
        base += v
    ax.set_xticks(x, [f"{int(r)}\nn={int(n):,}".replace(",", " ") for r, n in zip(t["recuento"], t["n"])])
    ax.set_xlabel("Recuento de características")
    ax.set_ylabel("Hogares (% ponderado)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.42), fontsize=5.8)
    ax.grid(axis="x", visible=False)
    if titulo:
        ax.set_title(titulo)
    return fig


def fig_perfil_subgrupos(sub: pd.DataFrame) -> go.Figure:
    """Versión interactiva (app) de las barras con/sin característica con IC95."""
    cars = list(dict.fromkeys(sub["característica"]))
    fig = go.Figure()
    for pref, color, etq in (("Con", "#D55E00", "Con la característica"), ("Sin", "#56B4E9", "Sin la característica")):
        t = sub[sub["grupo"].str.startswith(pref)].set_index("característica").reindex(cars)
        fig.add_trace(go.Bar(name=etq, x=cars, y=t["prevalencia"], marker_color=color, customdata=t["n"],
                             error_y={"type": "data", "symmetric": False, "array": t["ic_sup"] - t["prevalencia"], "arrayminus": t["prevalencia"] - t["ic_inf"]},
                             hovertemplate="%{x}<br>Prevalencia %{y:.1%}<br>n = %{customdata:,}<extra>" + etq + "</extra>"))
    fig.update_layout(barmode="group", yaxis_tickformat=".0%", yaxis_title="Prevalencia ponderada de IA moderada o severa", height=420, margin={"t": 20}, legend={"orientation": "h", "y": -0.2})
    return fig


def fig_perfil_recuento(rec: pd.DataFrame, total: float | None = None) -> go.Figure:
    """Versión interactiva (app) del gradiente por recuento con IC95 y n."""
    t = rec[rec["n"] > 0]
    cols = _gradiente_recuento(len(rec))
    fig = go.Figure(go.Bar(x=[str(int(r)) for r in t["recuento"]], y=t["prevalencia"], marker_color=[cols[int(r)] for r in t["recuento"]],
                           marker_line_color="#333333", marker_line_width=0.6, customdata=t["n"],
                           text=[f"{p:.1%}<br>n = {int(n):,}" for p, n in zip(t["prevalencia"], t["n"])], textposition="outside",
                           error_y={"type": "data", "symmetric": False, "array": t["ic_sup"] - t["prevalencia"], "arrayminus": t["prevalencia"] - t["ic_inf"]},
                           hovertemplate="Recuento %{x}<br>Prevalencia %{y:.1%}<br>n = %{customdata:,}<extra></extra>"))
    if total is not None and total == total:
        fig.add_hline(y=total, line_dash="dash", line_color="#555555", annotation_text="total")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Prevalencia ponderada de IA moderada o severa", xaxis_title="Recuento de características", height=420, margin={"t": 40},
                      yaxis_range=[0, float(t["ic_sup"].max()) * 1.25])
    return fig


def fig_perfil_fies(fr: pd.DataFrame) -> go.Figure:
    """Versión interactiva (app) del nivel FIES por recuento (100 % apilado)."""
    niveles = [c for c in fr.columns if c not in ("recuento", "n")]
    t = fr[fr["n"] > 0]
    colores = ["#9ECAE1", "#F0E442", "#E69F00", "#D55E00"]
    fig = go.Figure([go.Bar(name=c, x=[f"{int(r)} (n = {int(n):,})" for r, n in zip(t["recuento"], t["n"])], y=t[c], marker_color=col, text=[f"{v:.0%}" for v in t[c]], textposition="inside")
                     for c, col in zip(niveles, colores)])
    fig.update_layout(barmode="stack", yaxis_tickformat=".0%", yaxis_title="Hogares (% ponderado)", xaxis_title="Recuento de características", height=420, margin={"t": 20}, legend={"orientation": "h", "y": -0.25})
    return fig


def mpl_distribucion_fies(tabla: pd.DataFrame, titulo: str = ""):
    """Distribución del nivel FIES. Columnas: nivel, sin_ponderar, ponderada (proporciones)."""
    fig, ax = _figura(ANCHO_SIMPLE, 2.5)
    x = np.arange(len(tabla))
    ax.bar(x - 0.2, tabla["sin_ponderar"] * 100, 0.4, color=OKABE_ITO[0], label="Sin ponderar")
    ax.bar(x + 0.2, tabla["ponderada"] * 100, 0.4, color=OKABE_ITO[1], label="Ponderada")
    ax.set_xticks(x, [f"Nivel {int(n)}" for n in tabla["nivel"]])
    ax.set_ylabel("Hogares (%)")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    if titulo:
        ax.set_title(titulo)
    return fig


def mpl_metricas_modelos(tabla: pd.DataFrame, metricas: list[str], etiquetas: Mapping[str, str] | None = None, titulo: str = ""):
    """Puntos con IC95 por modelo y métrica. Columnas: modelo, metrica, media, ic_inf, ic_sup."""
    etiquetas = etiquetas or {}
    fig, axes = _figura(ANCHO_DOBLE, 2.5, ncols=len(metricas))
    axes = np.atleast_1d(axes)
    modelos = list(dict.fromkeys(tabla["modelo"]))
    for ax, m in zip(axes, metricas):
        d = tabla[tabla["metrica"] == m].set_index("modelo").reindex(modelos)
        x = np.arange(len(modelos))
        lo = (d["media"] - d["ic_inf"]).clip(lower=0).fillna(0).to_numpy()
        hi = (d["ic_sup"] - d["media"]).clip(lower=0).fillna(0).to_numpy()
        for i, mod in enumerate(modelos):
            ax.errorbar(x[i], d["media"].iloc[i], yerr=[[lo[i]], [hi[i]]], fmt="o", color=_col_modelo(mod, i), capsize=2.5, ms=4.5, elinewidth=1)
        ax.set_xticks(x, modelos)
        ax.set_title(etiquetas.get(m, m))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Media e IC95 (pliegues externos)")
    if titulo:
        fig.suptitle(titulo, fontsize=8)
    return fig


def mpl_confusion_doble(cm: np.ndarray, titulo: str = ""):
    """Matriz de confusión: conteos (izquierda) y normalizada por fila (derecha)."""
    fig, axes = _figura(ANCHO_DOBLE, 3.0, ncols=2)
    suma = cm.sum(axis=1, keepdims=True)
    norm = cm / np.where(suma == 0, 1, suma)
    for ax, M, fmt, sub in ((axes[0], cm, lambda v: f"{v:,.0f}", "Conteos"),
                            (axes[1], norm, lambda v: f"{v:.1%}", "Normalizada por fila (sensibilidad)")):
        ax.imshow(M, cmap="Blues", vmin=0)
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7, color="white" if v > M.max() / 2 else "black")
        ax.set_xticks(range(M.shape[1]), [str(i) for i in range(M.shape[1])])
        ax.set_yticks(range(M.shape[0]), [str(i) for i in range(M.shape[0])])
        ax.set_xlabel("Nivel FIES predicho")
        ax.set_ylabel("Nivel FIES real")
        ax.set_title(sub)
        ax.grid(False)
    if titulo:
        fig.suptitle(titulo, fontsize=8)
    return fig


def mpl_curvas_multi(curvas: Mapping[str, tuple], tipo: str, titulo: str = ""):
    """ROC o PR superpuestas por modelo. ``curvas[modelo] = (x, y, resumen)`` con resumen = AUC (roc) o AP (pr)."""
    fig, ax = _figura(ANCHO_SIMPLE, 3.0)
    for i, (mod, (x, y, r)) in enumerate(curvas.items()):
        ax.plot(x, y, color=_col_modelo(mod, i), label=f"{mod} ({'AUC' if tipo == 'roc' else 'AP'} {r:.3f})")
    if tipo == "roc":
        ax.plot([0, 1], [0, 1], ":", color=GRIS, lw=0.9)
        ax.set_xlabel("1 - especificidad (FPR)")
        ax.set_ylabel("Sensibilidad (TPR)")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precisión")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right" if tipo == "roc" else "upper right")
    if titulo:
        ax.set_title(titulo)
    return fig


def mpl_calibracion_multi(curvas: Mapping[str, pd.DataFrame], titulo: str = ""):
    fig, ax = _figura(ANCHO_SIMPLE, 3.0)
    ax.plot([0, 1], [0, 1], ":", color=GRIS, label="Calibración perfecta")
    for i, (mod, c) in enumerate(curvas.items()):
        if len(c):
            ax.plot(c["p_predicha"], c["frecuencia"], "o-", ms=3, color=_col_modelo(mod, i), label=mod)
    ax.set_xlabel("Probabilidad predicha (IA severa)")
    ax.set_ylabel("Frecuencia observada")
    ax.legend(frameon=False)
    if titulo:
        ax.set_title(titulo)
    return fig


def mpl_diagrama_dc(rangos: Mapping[str, float], dc: float, titulo: str = ""):
    """Diagrama de diferencia crítica (Demšar): rangos medios; las barras unen modelos con |dif| < DC."""
    nombres = sorted(rangos, key=lambda k: rangos[k])
    r = np.array([rangos[k] for k in nombres], dtype=float)
    k = len(nombres)
    fig, ax = _figura(ANCHO_DOBLE, 1.3 + 0.3 * k)
    ax.set_xlim(k + 0.3, 0.7)             # el mejor rango (1) a la derecha
    ax.set_ylim(-(k + 3.2), 1.6)
    ax.axis("off")
    ax.plot([k, 1], [0, 0], color="black", lw=0.9)
    for t in range(1, k + 1):
        ax.plot([t, t], [0, 0.18], color="black", lw=0.9)
        ax.text(t, 0.3, str(t), ha="center", va="bottom", fontsize=7)
    ax.plot([1, 1 + dc], [1.1, 1.1], color="black", lw=1.2)
    ax.text(1 + dc / 2, 1.18, f"DC = {dc:.2f}", ha="center", va="bottom", fontsize=6.5)
    for i, (n, v) in enumerate(zip(nombres, r)):
        y = -(1 + i)
        ax.plot([v, v], [0, y], color=_col_modelo(n, i), lw=1)
        ax.text(v, y - 0.15, f"{n} ({v:.2f})", ha="center", va="top", fontsize=7, color=_col_modelo(n, i))
    y0 = -(k + 1.6)
    ultimo = -1
    for i in range(k):
        j = i
        while j + 1 < k and r[j + 1] - r[i] < dc:
            j += 1
        if j > i and j > ultimo:
            ax.plot([r[i] - 0.03, r[j] + 0.03], [y0, y0], color="black", lw=2.2)
            y0 -= 0.35
            ultimo = j
    if titulo:
        ax.set_title(titulo, loc="left")
    return fig


def mpl_shap_barras(imp: pd.DataFrame, top: int = 15, titulo: str = ""):
    """|SHAP| medio por variable (barras horizontales). Columnas: variable, importancia."""
    d = imp.sort_values("importancia", ascending=False).head(top).iloc[::-1]
    fig, ax = _figura(ANCHO_SIMPLE, max(2.0, 0.19 * len(d) + 0.8))
    ax.barh([etq(v) for v in d["variable"]], d["importancia"], color=OKABE_ITO[0], height=0.65)
    ax.set_xlabel("|SHAP| medio")
    ax.grid(axis="y", visible=False)
    if titulo:
        ax.set_title(titulo)
    return fig


def mpl_shap_beeswarm(shap_vals: np.ndarray, X: pd.DataFrame, top: int = 12, titulo: str = ""):
    """Beeswarm aproximado: SHAP firmado por hogar coloreado por el valor relativo de la variable (categóricas en gris)."""
    imp = np.abs(shap_vals).mean(axis=0)
    orden = np.argsort(-imp)[:top][::-1]
    fig, ax = _figura(ANCHO_SIMPLE + 0.6, max(2.2, 0.26 * len(orden) + 0.9))
    rng = np.random.default_rng(42)
    sc = None
    for fila, j in enumerate(orden):
        v = shap_vals[:, j]
        col = X.iloc[:, j]
        y = fila + rng.uniform(-0.28, 0.28, size=len(v))
        if pd.api.types.is_numeric_dtype(col):
            q = col.rank(pct=True).to_numpy() if col.nunique() > 2 else col.to_numpy(float)
            sc = ax.scatter(v, y, c=q, cmap="viridis", s=4, alpha=0.8, linewidths=0)
        else:
            ax.scatter(v, y, c="#9AA3AB", s=4, alpha=0.8, linewidths=0)
    ax.set_yticks(range(len(orden)), [etq(X.columns[j]) for j in orden])
    ax.axvline(0, color="#555555", lw=0.6)
    ax.set_xlabel("SHAP (impacto sobre la clase IA severa)")
    ax.grid(axis="y", visible=False)
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["bajo", "alto"])
        cb.set_label("Valor de la variable", fontsize=6.5)
        cb.ax.tick_params(labelsize=6.5)
    if titulo:
        ax.set_title(titulo)
    return fig


def mpl_dependencia(curvas: list[Mapping[str, Any]], titulo: str = ""):
    """Paneles de PDP (``rejilla``/``prob_media``) o ALE (``bordes``/``ale``) para varias variables."""
    fig, axes = _figura(ANCHO_DOBLE, 2.3, ncols=max(1, len(curvas)))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, curvas):
        if "rejilla" in c:
            ax.plot(c["rejilla"], c["prob_media"], "o-", ms=3, color=OKABE_ITO[0])
            ax.set_ylabel("Probabilidad media (PDP)")
        else:
            b = np.asarray(c["bordes"], dtype=float)
            a = np.asarray(c["ale"], dtype=float)
            xs = (b[:-1] + b[1:]) / 2 if len(b) == len(a) + 1 else b[: len(a)]
            ax.plot(xs, a, "o-", ms=3, color=OKABE_ITO[2])
            ax.axhline(0, color="#555555", lw=0.6)
            ax.set_ylabel("Efecto ALE")
        ax.set_xlabel(etq(c["variable"], True))
        if c.get("n_muestra"):
            ax.set_title(f"n = {int(c['n_muestra']):,}", fontsize=6.5)
    if titulo:
        fig.suptitle(titulo, fontsize=8)
    return fig


def mpl_equidad(tabla: pd.DataFrame, umbral: float = 0.05, titulo: str = ""):
    """Brecha TPR (pp) por grupo con IC. Columnas: grupo_col, grupo, es_referencia, brecha_tpr, ic_inf, ic_sup, alerta."""
    t = tabla[~tabla["es_referencia"]].copy()
    t["etq"] = [etq(g) + ": " + valor_etq(g, v) for g, v in zip(t["grupo_col"].astype(str), t["grupo"].astype(str))]
    t = t.iloc[::-1].reset_index(drop=True)
    fig, ax = _figura(ANCHO_DOBLE, max(1.8, 0.2 * len(t) + 0.9))
    y = np.arange(len(t))
    colores = [OKABE_ITO[5] if a else OKABE_ITO[0] for a in t["alerta"]]
    ax.axvspan(-umbral * 100, umbral * 100, color="#E8EEF4", zorder=0)
    ax.scatter(t["brecha_tpr"] * 100, y, c=colores, s=14, zorder=3)
    lo = (t["brecha_tpr"] - t["ic_inf"]).fillna(0).clip(lower=0).to_numpy() * 100
    hi = (t["ic_sup"] - t["brecha_tpr"]).fillna(0).clip(lower=0).to_numpy() * 100
    ax.errorbar(t["brecha_tpr"] * 100, y, xerr=np.vstack([lo, hi]), fmt="none", ecolor="#444444", elinewidth=0.8, capsize=1.5, zorder=2)
    ax.axvline(0, color="#555555", lw=0.6)
    ax.set_yticks(y, t["etq"])
    ax.set_xlabel("Brecha de sensibilidad (TPR) frente al grupo de referencia (puntos porcentuales)")
    ax.grid(axis="y", visible=False)
    if titulo:
        ax.set_title(titulo)
    return fig


def mpl_escenarios(tabla: pd.DataFrame, metrica_etq: str = "F1-macro ponderado", titulo: str = ""):
    """Compara escenarios por modelo. Columnas: escenario, modelo, media, ic_inf, ic_sup."""
    fig, ax = _figura(ANCHO_SIMPLE + 0.8, 2.6)
    escenarios = list(dict.fromkeys(tabla["escenario"]))
    modelos = list(dict.fromkeys(tabla["modelo"]))
    ancho = 0.8 / max(1, len(modelos))
    for i, mod in enumerate(modelos):
        d = tabla[tabla["modelo"] == mod].set_index("escenario").reindex(escenarios)
        x = np.arange(len(escenarios)) + (i - (len(modelos) - 1) / 2) * ancho
        err = np.vstack([(d["media"] - d["ic_inf"]).clip(lower=0).fillna(0), (d["ic_sup"] - d["media"]).clip(lower=0).fillna(0)])
        ax.bar(x, d["media"], ancho * 0.9, yerr=err, color=_col_modelo(mod, i), label=mod, error_kw={"elinewidth": 0.7, "capsize": 1.5})
    ax.set_xticks(range(len(escenarios)), escenarios)
    ax.set_ylabel(metrica_etq)
    ax.legend(frameon=False, ncol=min(5, len(modelos)), loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7)
    ax.grid(axis="x", visible=False)
    if titulo:
        ax.set_title(titulo)
    return fig


def mpl_dependencia_doble(items: list[Mapping[str, Any]], titulo: str = ""):
    """Rejilla de 2 filas: PDP (arriba) y ALE (abajo) por variable. ``items`` = salida de ``explainability.pdp_ale_principales``['curvas'].

    Numéricas: curva PDP y curva ALE. Categóricas: barras de PDP por categoría y «ALE no aplica». Etiquetas legibles (presentación).
    """
    n = max(1, len(items))
    fig, axes = _figura(ANCHO_DOBLE, 3.9, nrows=2, ncols=n, squeeze=False)
    for j, c in enumerate(items):
        a1, a2 = axes[0][j], axes[1][j]
        nombre = chr(10).join(__import__("textwrap").wrap(etq(c["variable"], True), 34))
        if c["tipo"] == "categorica":
            cats = [valor_etq(c["variable"], k) for k in c["categorias"]]
            a1.barh(range(len(cats)), c["prob_media"], color=OKABE_ITO[0], height=0.65)
            a1.set_yticks(range(len(cats)), cats)
            a1.invert_yaxis()
            a1.set_xlabel("Prob. media (PDP)", fontsize=5.5)
            a1.set_title(etq(c["variable"]), fontsize=6.5)
            a2.text(0.5, 0.5, "ALE no aplica\n(variable categórica)", ha="center", va="center", transform=a2.transAxes, fontsize=6.5, color="#555555")
            a2.set_axis_off()
            continue
        p = c["pdp"]
        a1.plot(p["rejilla"], p["prob_media"], "o-", ms=2.5, color=OKABE_ITO[0])
        a1.set_title(etq(c["variable"]), fontsize=6.5)
        a1.set_xlabel(nombre, fontsize=5.5)
        if j == 0:
            a1.set_ylabel("Prob. media de IA severa (PDP)")
        al = c.get("ale")
        if al:
            b = np.asarray(al["bordes"], dtype=float)
            a2.plot(b, al["ale"], "o-", ms=2.5, color=OKABE_ITO[2])
            a2.axhline(0, color="#555555", lw=0.6)
            a2.set_xlabel(nombre, fontsize=5.5)
            if j == 0:
                a2.set_ylabel("Efecto ALE (centrado)")
        else:
            a2.text(0.5, 0.5, "ALE no disponible\n(pocos valores distintos)", ha="center", va="center", transform=a2.transAxes, fontsize=6.5, color="#555555")
            a2.set_axis_off()
    if titulo:
        fig.suptitle(titulo, fontsize=8)
    fig.tight_layout()
    return fig


# ------------------------------ Inferencia: LIME, sensibilidad, tornado, contrafactual y otros modelos ------------------------------
NOMBRE_NIVEL_CORTO = {0: "Seguridad", 1: "Leve", 2: "Moderada", 3: "Severa"}


def _recorta(t: str, n: int = 46) -> str:
    return t if len(t) <= n else t[: n - 1] + "…"


def mpl_lime(tabla: pd.DataFrame, clase: int, top: int = 12):
    d = tabla.head(top).iloc[::-1]
    with plt.rc_context(RC_REVISTA):
        fig, ax = plt.subplots(figsize=(ANCHO_DOBLE, max(2.4, 0.26 * len(d) + 1.0)))
        ax.barh([_recorta(r) for r in d["regla"]], d["peso"], color=[ROJO if v > 0 else AZUL for v in d["peso"]], height=0.65, zorder=2)
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set_xlabel(f"Peso LIME sobre la probabilidad del nivel {clase} (bermellón: la aumenta; azul: la reduce)")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"Explicación local LIME (aproximación local): {len(d)} condiciones con mayor peso", loc="left")
        return fig


def mpl_lime_shap(tab: pd.DataFrame, top: int = 10):
    d = tab.copy()
    d["l"] = d["LIME (peso)"] / max(d["abs LIME"].max(), 1e-12)
    d["s"] = d["SHAP agregado"] / max(d["abs SHAP"].max(), 1e-12)
    d["orden"] = np.maximum(d["l"].abs(), d["s"].abs())
    d = d.sort_values("orden", ascending=False).head(top).iloc[::-1]
    with plt.rc_context(RC_REVISTA):
        fig, ax = plt.subplots(figsize=(ANCHO_DOBLE, max(2.6, 0.36 * len(d) + 1.0)))
        y = np.arange(len(d))
        ax.barh(y + 0.19, d["l"], height=0.36, color=OKABE_ITO[0], label="LIME", zorder=2)
        ax.barh(y - 0.19, d["s"], height=0.36, color=OKABE_ITO[1], label="SHAP (suma de las columnas de cada variable)", zorder=2)
        ax.set_yticks(y, [etq(v) for v in d["variable"]])
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set_xlabel("Efecto normalizado por el mayor |efecto| de cada método (−1 a +1)")
        ax.set_xlim(-1.1, 1.1)
        ax.grid(axis="y", visible=False)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        ax.set_title("LIME frente a SHAP: signo y orden de las variables principales", loc="left")
        return fig


def mpl_sensibilidad(curvas: pd.DataFrame, principales: list[str], tipos: dict[str, str]):
    n = len(principales)
    nc = 4 if n > 4 else max(n, 1)
    nf = -(-n // nc)
    with plt.rc_context(RC_REVISTA):
        fig, axs = plt.subplots(nf, nc, figsize=(ANCHO_DOBLE, 2.3 * nf + 1.3), squeeze=False)
        for ax in axs.ravel():
            ax.set_visible(False)
        for i, v in enumerate(principales):
            ax = axs[i // nc][i % nc]
            ax.set_visible(True)
            d = curvas[curvas["variable"] == v]
            if tipos.get(v) == "categorica":
                d = d.sort_values("p_mod_sev")
                if len(d) > 9:                       # categorías extremas y la del hogar (la tabla trae todas)
                    act = d[d["es_valor_actual"]]
                    d = pd.concat([d.head(4), d.tail(4), act]).drop_duplicates("valor").sort_values("p_mod_sev")
                ax.barh([_recorta(valor_etq(v, c), 16) for c in d["valor"]], d["p_mod_sev"] * 100, height=0.6,
                        color=[ROJO if a else OKABE_ITO[0] for a in d["es_valor_actual"]], zorder=2)
                ax.set_xlabel("P(moderada o severa) (%)", fontsize=6.5)
                ax.tick_params(axis="y", labelsize=6)
            else:
                x = d["valor"].astype(float).to_numpy()
                for k in range(4):
                    ax.plot(x, d[f"p{k}"] * 100, color=PALETA[k], lw=1.1, marker="o", ms=2, label=NOMBRE_NIVEL_CORTO[k])
                act = d[d["es_valor_actual"]]
                if len(act):
                    ax.axvline(float(act["valor"].iloc[0]), color="#555555", lw=0.8, ls="--")
                ax.set_ylabel("Probabilidad (%)", fontsize=6.5)
                ax.set_xlabel(_recorta(etq(v), 22), fontsize=6.5)
            ax.set_title(_recorta(etq(v), 21), fontsize=7, loc="left")
            ax.tick_params(labelsize=6)
            ax.grid(axis="x" if tipos.get(v) != "categorica" else "y", visible=False)
        h = [plt.Line2D([0], [0], color=PALETA[k], lw=1.4) for k in range(4)]
        fig.legend(h, [NOMBRE_NIVEL_CORTO[k] for k in range(4)], ncol=4, loc="lower center", frameon=False, fontsize=7)
        fig.suptitle("Sensibilidad variable a variable (curvas ICE locales)\nLínea punteada: valor actual del hogar · categóricas: barra bermellón = categoría actual", x=0.01, ha="left", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.08, 1, 0.9), w_pad=1.2, h_pad=1.4)
        fig._fsai_layout_fijo = True
        return fig


def mpl_tornado(tor: pd.DataFrame, p_base: float, top: int = 12):
    d = tor.head(top).iloc[::-1]
    with plt.rc_context(RC_REVISTA):
        fig, ax = plt.subplots(figsize=(ANCHO_DOBLE, max(2.4, 0.27 * len(d) + 1.0)))
        ax.barh([_recorta(f"{a} · {b} → {c}", 62) for a, b, c in zip(d["Variable"], d["valor bajo"], d["valor alto"])], d["efecto (pp)"],
                color=[ROJO if v > 0 else AZUL for v in d["efecto (pp)"]], height=0.65, zorder=2)
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set_xlabel(f"Cambio de P(mod. o severa) en puntos porcentuales (valor bajo → alto)\nreferencia: P(moderada o severa) = {100 * p_base:.1f} %", fontsize=6.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_title("Tornado de sensibilidad (percentil 10 → percentil 90)", loc="left")
        return fig


def mpl_contrafactual(cf: dict, nombres_nivel: dict[int, str] | None = None):
    esc = cf["escenarios"]
    nombres = ["Actual"] + [f"Escenario {e['n']}" for e in esc]
    P = [cf["p_base"]] + [e["p"] for e in esc]
    with plt.rc_context(RC_REVISTA):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(ANCHO_DOBLE, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})
        w = 0.8 / len(nombres)
        for i, (n, p) in enumerate(zip(nombres, P)):
            a1.bar(np.arange(4) + (i - (len(nombres) - 1) / 2) * w, np.array(p) * 100, w * 0.92, color=OKABE_ITO[[7, 0, 1, 3][i % 4]] if i else "#888888", label=n, zorder=2)
        a1.set_xticks(range(4), [NOMBRE_NIVEL_CORTO[k] for k in range(4)])
        a1.set_ylabel("Probabilidad (%)")
        a1.legend(frameon=False, fontsize=6.5)
        a1.grid(axis="x", visible=False)
        a1.set_title("Probabilidad por nivel FIES", loc="left", fontsize=8)
        pm = [cf["p_mod_sev_base"]] + [e["p_mod_sev"] for e in esc]
        a2.bar(range(len(nombres)), np.array(pm) * 100, color=["#888888"] + [OKABE_ITO[2] if e["alcanza_objetivo"] else OKABE_ITO[1] for e in esc], zorder=2)
        a2.axhline(cf["objetivo"] * 100, color=ROJO, ls="--", lw=1.2, label=f"Objetivo ({cf['objetivo'] * 100:.1f} %)")
        for i, v in enumerate(pm):
            a2.text(i, v * 100 + 0.6, f"{v * 100:.1f}", ha="center", fontsize=7)
        a2.set_xticks(range(len(nombres)), [n.replace("Escenario ", "Esc. ") for n in nombres])
        a2.set_ylabel("P(moderada o severa) (%)")
        a2.set_ylim(0, max(pm + [cf["objetivo"]]) * 100 * 1.2)
        a2.legend(frameon=False, fontsize=6.5, loc="upper right")
        a2.grid(axis="x", visible=False)
        a2.set_title("P(mod. o severa) y objetivo", loc="left", fontsize=8)
        fig.suptitle("Escenarios contrafactuales según el modelo (asociación predictiva, no causal) · verde: alcanza el objetivo", x=0.01, ha="left", fontsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig._fsai_layout_fijo = True
        return fig


def mpl_otros_modelos(t: pd.DataFrame):
    with plt.rc_context(RC_REVISTA):
        fig, ax = plt.subplots(figsize=(ANCHO_DOBLE, 3.0))
        n = len(t)
        w = 0.8 / max(n, 1)
        for i, r in enumerate(t.itertuples()):
            ax.bar(np.arange(4) + (i - (n - 1) / 2) * w, [t.iloc[i][f"P(nivel {k})"] * 100 for k in range(4)], w * 0.92,
                   color=COLOR_MODELO.get(r.codigo, OKABE_ITO[i % 8]), label=r.codigo + (" ★" if r.Seleccionado else ""), zorder=2)
        ax.set_xticks(range(4), [f"Nivel {k}\n{NOMBRE_NIVEL_CORTO[k]}" for k in range(4)])
        ax.set_ylabel("Probabilidad (%)")
        ax.grid(axis="x", visible=False)
        ax.legend(ncol=n, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
        ax.set_title("Mismo hogar, todos los modelos de la corrida (media de cada ensamble; ★ = modelo seleccionado)", loc="left")
        return fig


# --- Plotly (interfaz)
def fig_lime(tabla: pd.DataFrame, clase: int, top: int = 12) -> go.Figure:
    d = tabla.head(top).iloc[::-1]
    f = go.Figure(go.Bar(x=d["peso"], y=[_recorta(r, 60) for r in d["regla"]], orientation="h", marker_color=[ROJO if v > 0 else AZUL for v in d["peso"]]))
    f.update_layout(xaxis_title=f"Peso LIME sobre P(nivel {clase}) (rojo: la aumenta)", height=max(320, 26 * len(d) + 90), margin={"t": 20})
    return f


def fig_lime_shap(tab: pd.DataFrame, top: int = 10) -> go.Figure:
    d = tab.copy()
    d["l"], d["s"] = d["LIME (peso)"] / max(d["abs LIME"].max(), 1e-12), d["SHAP agregado"] / max(d["abs SHAP"].max(), 1e-12)
    d = d.assign(o=np.maximum(d["l"].abs(), d["s"].abs())).sort_values("o", ascending=False).head(top).iloc[::-1]
    f = go.Figure([go.Bar(name="LIME", x=d["l"], y=[etq(v) for v in d["variable"]], orientation="h", marker_color=OKABE_ITO[0]),
                   go.Bar(name="SHAP (agregado)", x=d["s"], y=[etq(v) for v in d["variable"]], orientation="h", marker_color=OKABE_ITO[1])])
    f.update_layout(barmode="group", xaxis_title="Efecto normalizado por el mayor |efecto| de cada método", height=max(340, 38 * len(d) + 100), margin={"t": 20})
    return f


def fig_sens_curva(curvas: pd.DataFrame, variable: str, tipo: str) -> go.Figure:
    d = curvas[curvas["variable"] == variable]
    f = go.Figure()
    if tipo == "categorica":
        d = d.sort_values("p_mod_sev")
        f.add_trace(go.Bar(x=d["p_mod_sev"], y=[valor_etq(variable, c) for c in d["valor"]], orientation="h", marker_color=[ROJO if a else OKABE_ITO[0] for a in d["es_valor_actual"]]))
        f.update_layout(xaxis_title="P(moderada o severa) (barra roja: categoría actual)", xaxis_tickformat=".0%", height=max(300, 22 * len(d) + 90), margin={"t": 30})
    else:
        for k in range(4):
            f.add_trace(go.Scatter(x=d["valor"], y=d[f"p{k}"], mode="lines+markers", name=NOMBRE_NIVEL_CORTO[k], line={"color": PALETA[k]}))
        act = d[d["es_valor_actual"]]
        if len(act):
            f.add_vline(x=float(act["valor"].iloc[0]), line_dash="dash", line_color="#555555", annotation_text="valor actual", annotation_position="top")
        f.update_layout(xaxis_title=etq(variable, True), yaxis_title="Probabilidad", yaxis_tickformat=".0%", margin={"t": 30})
    f.update_layout(title={"text": etq(variable), "font": {"size": 13}})
    return f


def fig_tornado(tor: pd.DataFrame, top: int = 12) -> go.Figure:
    d = tor.head(top).iloc[::-1]
    f = go.Figure(go.Bar(x=d["efecto (pp)"], y=[f"{a} · {b} → {c}" for a, b, c in zip(d["Variable"], d["valor bajo"], d["valor alto"])], orientation="h",
                         marker_color=[ROJO if v > 0 else AZUL for v in d["efecto (pp)"]]))
    f.update_layout(xaxis_title="Cambio de P(moderada o severa) en pp (valor bajo → valor alto)", height=max(340, 28 * len(d) + 100), margin={"t": 20})
    return f


def fig_contrafactual(cf: dict) -> go.Figure:
    esc = cf["escenarios"]
    nombres = ["Actual"] + [f"Escenario {e['n']}" for e in esc]
    pm = [cf["p_mod_sev_base"]] + [e["p_mod_sev"] for e in esc]
    f = go.Figure(go.Bar(x=nombres, y=pm, marker_color=["#888888"] + [OKABE_ITO[2] if e["alcanza_objetivo"] else OKABE_ITO[1] for e in esc]))
    f.add_hline(y=cf["objetivo"], line_dash="dash", line_color=ROJO, annotation_text=f"objetivo {cf['objetivo']:.1%}")
    f.update_layout(yaxis_title="P(moderada o severa)", yaxis_tickformat=".0%", margin={"t": 20})
    return f


def fig_otros_modelos(t: pd.DataFrame) -> go.Figure:
    f = go.Figure([go.Bar(name=r["codigo"] + (" ★" if r["Seleccionado"] else ""), x=[f"Nivel {k}" for k in range(4)], y=[r[f"P(nivel {k})"] for k in range(4)],
                          marker_color=COLOR_MODELO.get(r["codigo"], OKABE_ITO[0])) for _, r in t.iterrows()])
    f.update_layout(barmode="group", yaxis_title="Probabilidad", yaxis_tickformat=".0%", margin={"t": 20})
    return f
