"""analisis_local.py - Análisis locales de UNA inferencia: LIME, sensibilidad, contrafactual, otros modelos y validez de las entradas.

Todo se calcula con los modelos REALES de la corrida (ensamble de ``fsai.inference``) y con la muestra de Desarrollo como referencia
(nunca la Prueba). No tiene dependencia de Streamlit. Semilla fija ``config.SEED``.

Alcance declarado (se repite en la interfaz y en los reportes):

* LIME es una aproximación LOCAL (un modelo lineal ponderado ajustado a perturbaciones del hogar); se compara con SHAP por signos y
  por las variables principales, no por magnitud (unidades distintas).
* La sensibilidad y el contrafactual describen cómo responde el MODELO a cambios en sus entradas. Son asociaciones predictivas aprendidas
  de otros hogares: NO son efectos causales ni una recomendación de política. La capacidad predictiva del modelo es modesta.
* Las variables que se pueden mover en el contrafactual salen de ``etiquetas.ACCIONABLES`` ([POR VALIDAR]); nunca los ítems FIES.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fsai import config as C
from fsai import etiquetas as E
from fsai import inference as INF
from fsai.explainability import agrupar_a_columna_cruda
from fsai.models import predecir_proba
from fsai.preprocessing import preparar_matriz

logger = logging.getLogger(__name__)

SEMILLA = C.SEED
LIME_N_MUESTRAS = 2000
LIME_MAX_FONDO = 5000
SENS_N_VARIABLES = 8
SENS_N_PUNTOS = 10
CF_MAX_CAMBIOS = 3
CF_N_ESCENARIOS = 3
CF_N_PUNTOS = 12
CF_REDUCCION_RELATIVA = 0.20
_CACHE_ENSAMBLES: dict[tuple[str, str], dict[str, Any]] = {}


# ============================================================ utilidades
def cuantil_ponderado(x: np.ndarray, w: np.ndarray, q: float | np.ndarray) -> np.ndarray:
    """Cuantil(es) ponderado(s) por el factor de expansión (interpolación lineal sobre la distribución acumulada)."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    orden = np.argsort(x, kind="mergesort")
    x, w = x[orden], w[orden]
    acum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(np.atleast_1d(q), acum, x)


def _es_entera(s: pd.Series) -> bool:
    v = pd.to_numeric(s, errors="coerce").dropna()
    return len(v) > 0 and bool(np.all(np.isclose(v, np.round(v))))


def _proba_media(miembros: list, X: pd.DataFrame) -> np.ndarray:
    return np.mean([predecir_proba(m, X) for m in miembros], axis=0)


def _lote_filas(base: dict[str, Any], cambios: list[dict[str, Any]], res: dict[str, Any]) -> pd.DataFrame:
    """DataFrame con una fila por variante (copia de ``base`` con los cambios dados), con los tipos del entrenamiento."""
    filas = []
    for ch in cambios:
        f = dict(base)
        f.update(ch)
        filas.append(f)
    return preparar_matriz(pd.DataFrame(filas), res["features_num"], res["features_cat"])


def p_mod_sev(P: np.ndarray) -> np.ndarray:
    """P(inseguridad moderada o severa) = P(nivel 2) + P(nivel 3)."""
    return P[..., 2] + P[..., 3]


def cargar_modelos_corrida(dir_run: Path, escenario: str) -> dict[str, dict[str, Any]]:
    """Ensambles de TODOS los modelos de la corrida y el escenario (con caché de proceso: la carga de joblib es lo más lento)."""
    out: dict[str, dict[str, Any]] = {}
    for f in sorted(Path(dir_run).glob(f"resultado_{escenario}_*.json")):
        ident = f.stem.replace("resultado_", "")
        clave = (str(dir_run), ident)
        if clave not in _CACHE_ENSAMBLES:
            try:
                _CACHE_ENSAMBLES[clave] = INF.cargar_ensamble(Path(dir_run), ident)
            except Exception:  # noqa: BLE001
                logger.exception("No se pudo cargar el ensamble %s", ident)
                continue
        out[ident] = _CACHE_ENSAMBLES[clave]
    return out


# ============================================================ LIME
def lime_local(modelo, dev: pd.DataFrame, valores: dict[str, Any], res: dict[str, Any], clase: int, n_muestras: int = LIME_N_MUESTRAS) -> dict[str, Any]:
    """LIME tabular del ``modelo`` final para ``clase``, sobre las variables CRUDAS del sistema.

    Las perturbaciones se generan en el espacio de las variables originales (cuantiles de Desarrollo para las numéricas, frecuencias para
    las categóricas y 0/1) y cada perturbación pasa por el pipeline completo del sistema (mismo imputador, escalador y codificador del
    entrenamiento). Fondo: hasta 5 000 hogares de Desarrollo (sin ponderar por factor de expansión). Semilla fija.
    """
    from lime.lime_tabular import LimeTabularExplainer

    num, cat = list(res["features_num"]), list(res["features_cat"])
    cols = num + cat
    Xd = preparar_matriz(dev, num, cat)
    if len(Xd) > LIME_MAX_FONDO:
        Xd = Xd.sample(n=LIME_MAX_FONDO, random_state=SEMILLA)
    binarias = [c for c in num if set(pd.to_numeric(Xd[c], errors="coerce").dropna().unique()) <= {0.0, 1.0}]
    categoricas = cat + binarias
    niveles: dict[str, list[Any]] = {}
    for c in cat:
        niveles[c] = sorted(Xd[c].dropna().astype(str).unique().tolist())
    for c in binarias:
        niveles[c] = [0.0, 1.0]
    fondo = np.zeros((len(Xd), len(cols)))
    for j, c in enumerate(cols):
        if c in cat:
            moda = Xd[c].dropna().astype(str).mode().iloc[0]
            fondo[:, j] = [niveles[c].index(str(v)) if pd.notna(v) else niveles[c].index(moda) for v in Xd[c]]
        elif c in binarias:
            s = pd.to_numeric(Xd[c], errors="coerce")
            fondo[:, j] = s.fillna(s.mode().iloc[0]).to_numpy()
        else:
            s = pd.to_numeric(Xd[c], errors="coerce")
            fondo[:, j] = s.fillna(s.median()).to_numpy()
    idx_cat = [cols.index(c) for c in categoricas]
    nombres_cat = {cols.index(c): ([E.valor_etq(c, v) for v in niveles[c]] if c in cat else ["No", "Sí"]) for c in categoricas}

    def predecir(Z: np.ndarray) -> np.ndarray:
        d = {}
        for j, c in enumerate(cols):
            if c in cat:
                d[c] = [niveles[c][int(round(v))] for v in Z[:, j]]
            else:
                d[c] = Z[:, j]
        X = preparar_matriz(pd.DataFrame(d), num, cat)
        return predecir_proba(modelo, X)

    fila = np.zeros(len(cols))
    for j, c in enumerate(cols):
        v = valores[c]
        fila[j] = niveles[c].index(str(v)) if c in cat else float(v)
    expl = LimeTabularExplainer(fondo, feature_names=cols, categorical_features=idx_cat, categorical_names=nombres_cat, class_names=[str(k) for k in C.CLASES],
                                mode="classification", discretize_continuous=True, random_state=SEMILLA)
    exp = expl.explain_instance(fila, predecir, labels=[clase], num_features=len(cols), num_samples=n_muestras)
    p_modelo = float(predecir(fila[None, :])[0, clase])
    mapa = dict(exp.local_exp[clase])
    reglas = {int(k): t for k, t in zip([i for i, _ in exp.as_map()[clase]], [t for t, _ in exp.as_list(label=clase)])}
    filas = []
    for j, w in mapa.items():
        c = cols[j]
        regla = reglas.get(j, c)
        regla = re.sub(rf"\b{re.escape(c)}\b", E.etq(c), regla)
        filas.append({"variable": c, "regla": regla, "peso": float(w)})
    d = pd.DataFrame(filas)
    d["abs"] = d["peso"].abs()
    d = d.sort_values("abs", ascending=False).reset_index(drop=True)
    return {"clase": int(clase), "tabla": d, "r2_local": float(exp.score), "prob_modelo": p_modelo, "prob_local": float(exp.local_pred[0]) if np.ndim(exp.local_pred) else float(exp.local_pred),
            "intercepto": float(exp.intercept[clase]), "n_muestras": int(n_muestras), "n_fondo": int(len(Xd)), "semilla": SEMILLA, "n_variables": len(cols)}


def comparar_lime_shap(lime: dict[str, Any], shap_d: dict[str, Any] | None, features: list[str], k: int = 5) -> dict[str, Any]:
    """Compara LIME (por variable cruda) con SHAP agregado por variable cruda: signos, coincidencia de las k principales y Spearman."""
    if not shap_d:
        return {"disponible": False, "motivo": "No hay explicación SHAP para este modelo/hogar."}
    crudas = agrupar_a_columna_cruda(list(shap_d["nombres"]), features)
    s = pd.Series(shap_d["valores"], index=crudas).groupby(level=0).sum()
    t = lime["tabla"].set_index("variable")
    comun = [v for v in t.index if v in s.index]
    tab = pd.DataFrame({"variable": comun, "LIME (peso)": [t.loc[v, "peso"] for v in comun], "SHAP agregado": [float(s[v]) for v in comun]})
    tab["abs LIME"], tab["abs SHAP"] = tab["LIME (peso)"].abs(), tab["SHAP agregado"].abs()
    con_efecto = tab[(tab["abs LIME"] > 1e-9) & (tab["abs SHAP"] > 1e-9)]
    mismo_signo = np.sign(con_efecto["LIME (peso)"]) == np.sign(con_efecto["SHAP agregado"])
    top_l = set(tab.sort_values("abs LIME", ascending=False)["variable"].head(k))
    top_s = set(tab.sort_values("abs SHAP", ascending=False)["variable"].head(k))
    rho = float(tab["abs LIME"].rank().corr(tab["abs SHAP"].rank())) if len(tab) > 2 else float("nan")
    # signos entre las variables que aparecen en el top-k de LIME o de SHAP
    union = tab[tab["variable"].isin(top_l | top_s)]
    ms_union = np.sign(union["LIME (peso)"]) == np.sign(union["SHAP agregado"])
    tab = tab.sort_values("abs SHAP", ascending=False).reset_index(drop=True)
    tab["mismo signo"] = np.where(np.sign(tab["LIME (peso)"]) == np.sign(tab["SHAP agregado"]), "sí", "no")
    return {"disponible": True, "tabla": tab, "k": k, "coincidencia_top_k": len(top_l & top_s), "signos_coinciden": int(mismo_signo.sum()), "signos_total": int(len(con_efecto)),
            "signos_top_coinciden": int(ms_union.sum()), "signos_top_total": int(len(union)), "spearman_abs": rho,
            "top_lime": sorted(top_l), "top_shap": sorted(top_s)}


# ============================================================ validez de las entradas (extrapolación)
def validez_entradas(spec: list[dict[str, Any]], valores: dict[str, Any], dev: pd.DataFrame) -> pd.DataFrame:
    """Posición de cada entrada frente al rango observado en Desarrollo: dentro, extremo (fuera de p1-p99) o fuera del rango observado."""
    filas = []
    for c in spec:
        v = c["variable"]
        if c["tipo"] == "categorica":
            vistas = set(dev[v].dropna().astype(str))
            frec = float((dev[v].astype(str) == str(valores[v])).mean())
            estado = "fuera del rango observado" if str(valores[v]) not in vistas else ("extremo (categoría rara: < 1 % de hogares)" if frec < 0.01 else "dentro")
            filas.append({"Variable": E.etq(v), "Código": v, "Valor": E.valor_etq(v, valores[v]), "Mín. observado": "", "P1": "", "P99": "", "Máx. observado": "",
                          "Frecuencia de la categoría en Desarrollo": frec, "Estado": estado})
            continue
        x = pd.to_numeric(dev[v], errors="coerce")
        w = pd.to_numeric(dev[C.COL_PESO], errors="coerce")
        p1, p99 = [float(a) for a in cuantil_ponderado(x.to_numpy(), w.to_numpy(), [0.01, 0.99])]
        mn, mx, val = float(x.min()), float(x.max()), float(valores[v])
        if c["tipo"] == "binaria":
            estado = "dentro"
        elif val < mn or val > mx:
            estado = "fuera del rango observado"
        elif val < p1 or val > p99:
            estado = "extremo (fuera de P1-P99)"
        else:
            estado = "dentro"
        filas.append({"Variable": E.etq(v), "Código": v, "Valor": f"{val:.6g}", "Mín. observado": mn, "P1": p1, "P99": p99, "Máx. observado": mx,
                      "Frecuencia de la categoría en Desarrollo": None, "Estado": estado})
    return pd.DataFrame(filas)


def validez_lote(dfl: pd.DataFrame, res: dict[str, Any], dev: pd.DataFrame) -> pd.DataFrame:
    """Por variable del lote: n de hogares fuera del rango observado en Desarrollo (numéricas) o con categoría no vista (categóricas)."""
    filas = []
    for c in res["features_num"]:
        x, ref = pd.to_numeric(dfl[c], errors="coerce"), pd.to_numeric(dev[c], errors="coerce")
        fuera = int(((x < ref.min()) | (x > ref.max())).sum())
        filas.append({"Variable": E.etq(c), "Código": c, "Rango observado en Desarrollo": f"{ref.min():.6g} a {ref.max():.6g}", "Hogares fuera de rango": fuera, "% del lote": 100 * fuera / max(len(dfl), 1)})
    for c in res["features_cat"]:
        vistas = set(dev[c].dropna().astype(str))
        fuera = int((~dfl[c].astype(str).isin(vistas) & dfl[c].notna()).sum())
        filas.append({"Variable": E.etq(c), "Código": c, "Rango observado en Desarrollo": f"{len(vistas)} categorías", "Hogares fuera de rango": fuera, "% del lote": 100 * fuera / max(len(dfl), 1)})
    return pd.DataFrame(filas)


# ============================================================ sensibilidad
def _rejilla(var: str, spec_c: dict[str, Any], dev: pd.DataFrame, actual: Any, n: int = SENS_N_PUNTOS) -> list[Any]:
    if spec_c["tipo"] == "categorica":
        return sorted(dev[var].dropna().astype(str).unique().tolist())
    if spec_c["tipo"] == "binaria":
        return [0.0, 1.0]
    x = pd.to_numeric(dev[var], errors="coerce")
    w = pd.to_numeric(dev[C.COL_PESO], errors="coerce")
    q = cuantil_ponderado(x.to_numpy(), w.to_numpy(), np.linspace(0.05, 0.95, n))
    if _es_entera(x):
        q = np.round(q)
    vals = sorted(set(float(v) for v in q) | {float(actual)})
    return vals


def sensibilidad(ens: dict[str, Any], res: dict[str, Any], dev: pd.DataFrame, spec: list[dict[str, Any]], valores: dict[str, Any],
                 orden_importancia: list[str] | None = None, n_variables: int = SENS_N_VARIABLES) -> dict[str, Any]:
    """Sensibilidad variable a variable (el resto del hogar queda fijo).

    * ``curvas``: para las ``n_variables`` principales, la probabilidad por nivel al recorrer una rejilla de valores plausibles (cuantiles
      ponderados p5-p95 de Desarrollo; todas las categorías para las categóricas) — curva ICE local.
    * ``tornado``: para TODAS las variables, el efecto sobre P(moderada o severa) de moverla entre su percentil 10 y su percentil 90 de
      Desarrollo (binarias: 0 y 1; categóricas: de la categoría de menor a la de mayor probabilidad).
    Las probabilidades son la media del ensamble.
    """
    miembros = ens["miembros"]
    por_var = {c["variable"]: c for c in spec}
    base = dict(valores)
    p_base = _proba_media(miembros, _lote_filas(base, [{}], res))[0]
    # variantes
    plan: list[tuple[str, str, Any]] = []        # (variable, rol, valor)
    cambios: list[dict[str, Any]] = []
    todas = [c["variable"] for c in spec]
    principales = [v for v in (orden_importancia or []) if v in por_var][:n_variables]
    resto = [v for v in todas if v not in principales]
    # curvas
    for v in principales:
        for val in _rejilla(v, por_var[v], dev, base[v]):
            plan.append((v, "curva", val))
            cambios.append({v: val})
    # tornado
    rangos: dict[str, tuple[Any, Any]] = {}
    for v in todas:
        c = por_var[v]
        if c["tipo"] in ("categorica", "binaria"):
            continue
        x, w = pd.to_numeric(dev[v], errors="coerce").to_numpy(), pd.to_numeric(dev[C.COL_PESO], errors="coerce").to_numpy()
        lo, hi = [float(a) for a in cuantil_ponderado(x, w, [0.10, 0.90])]
        rangos[v] = (lo, hi)
        plan += [(v, "p10", lo), (v, "p90", hi)]
        cambios += [{v: lo}, {v: hi}]
    for v in todas:
        c = por_var[v]
        if c["tipo"] == "binaria":
            plan += [(v, "bin0", 0.0), (v, "bin1", 1.0)]
            cambios += [{v: 0.0}, {v: 1.0}]
        elif c["tipo"] == "categorica" and v in resto:           # las categóricas principales ya tienen todas sus categorías en las curvas
            for cat in _rejilla(v, c, dev, base[v]):
                plan.append((v, "cat", cat))
                cambios.append({v: cat})
    P = _proba_media(miembros, _lote_filas(base, cambios, res))
    filas = pd.DataFrame({"variable": [p[0] for p in plan], "rol": [p[1] for p in plan], "valor": [p[2] for p in plan]})
    for k in C.CLASES:
        filas[f"p{k}"] = P[:, k]
    filas["p_mod_sev"] = p_mod_sev(P)
    # categóricas principales: sus categorías vienen de las curvas
    curvas = filas[filas["rol"] == "curva"].copy()
    curvas["es_valor_actual"] = [(str(r.valor) == str(base[r.variable])) if por_var[r.variable]["tipo"] == "categorica" else bool(np.isclose(float(r.valor), float(base[r.variable]))) for r in curvas.itertuples()]
    # tornado
    tor = []
    for v in todas:
        c = por_var[v]
        if c["tipo"] == "numerica":
            a = filas[(filas.variable == v) & (filas.rol == "p10")].iloc[0]
            b = filas[(filas.variable == v) & (filas.rol == "p90")].iloc[0]
            etq_lo, etq_hi = f"p10 = {a.valor:.4g}", f"p90 = {b.valor:.4g}"
        elif c["tipo"] == "binaria":
            a = filas[(filas.variable == v) & (filas.rol == "bin0")].iloc[0]
            b = filas[(filas.variable == v) & (filas.rol == "bin1")].iloc[0]
            etq_lo, etq_hi = "No (0)", "Sí (1)"
        else:
            sub = filas[(filas.variable == v) & (filas.rol.isin(["cat", "curva"]))]
            a, b = sub.loc[sub["p_mod_sev"].idxmin()], sub.loc[sub["p_mod_sev"].idxmax()]
            etq_lo, etq_hi = f"{E.valor_etq(v, a.valor)} (menor P)", f"{E.valor_etq(v, b.valor)} (mayor P)"
        tor.append({"variable": v, "Variable": E.etq(v), "valor bajo": etq_lo, "valor alto": etq_hi, "P(mod.+sev.) con valor bajo": float(a.p_mod_sev),
                    "P(mod.+sev.) con valor alto": float(b.p_mod_sev), "efecto (pp)": 100 * float(b.p_mod_sev - a.p_mod_sev)})
    tor = pd.DataFrame(tor)
    tor["|efecto| (pp)"] = tor["efecto (pp)"].abs()
    tor = tor.sort_values("|efecto| (pp)", ascending=False).reset_index(drop=True)
    tipos = {v: por_var[v]["tipo"] for v in principales}
    return {"curvas": curvas.reset_index(drop=True), "tornado": tor, "principales": principales, "tipos": tipos, "p_base": p_base, "p_mod_sev_base": float(p_mod_sev(p_base)),
            "n_variantes": int(len(cambios)), "rangos_p10_p90": rangos, "semilla": SEMILLA}


def sensibilidad_lote(ens: dict[str, Any], res: dict[str, Any], dev: pd.DataFrame, dfl: pd.DataFrame, n_max: int = 500) -> dict[str, Any]:
    """Sensibilidad AGREGADA del lote: efecto medio sobre P(moderada o severa) de llevar cada variable numérica/binaria a su p10 y a su p90
    de Desarrollo en todos los hogares de una submuestra reproducible (máx. ``n_max``) del lote."""
    muestra = dfl.sample(n=min(n_max, len(dfl)), random_state=SEMILLA) if len(dfl) > n_max else dfl
    X0 = preparar_matriz(muestra.reset_index(drop=True), res["features_num"], res["features_cat"])
    p0 = p_mod_sev(_proba_media(ens["miembros"], X0))
    filas = []
    for v in res["features_num"]:
        x, w = pd.to_numeric(dev[v], errors="coerce").to_numpy(), pd.to_numeric(dev[C.COL_PESO], errors="coerce").to_numpy()
        if set(np.unique(x[~np.isnan(x)])) <= {0.0, 1.0}:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = [float(a) for a in cuantil_ponderado(x, w, [0.10, 0.90])]
        r = {}
        for nombre, val in (("bajo", lo), ("alto", hi)):
            X1 = X0.copy()
            X1[v] = val
            r[nombre] = float(p_mod_sev(_proba_media(ens["miembros"], X1)).mean())
        filas.append({"variable": v, "Variable": E.etq(v), "valor bajo (p10 / 0)": lo, "valor alto (p90 / 1)": hi, "P(mod.+sev.) media con valor bajo": r["bajo"],
                      "P(mod.+sev.) media con valor alto": r["alto"], "efecto medio (pp)": 100 * (r["alto"] - r["bajo"])})
    for v in res["features_cat"]:
        cats = sorted(dev[v].dropna().astype(str).unique())
        medias = []
        for cat in cats:
            X1 = X0.copy()
            X1[v] = cat
            medias.append(float(p_mod_sev(_proba_media(ens["miembros"], X1)).mean()))
        i0, i1 = int(np.argmin(medias)), int(np.argmax(medias))
        filas.append({"variable": v, "Variable": E.etq(v), "valor bajo (p10 / 0)": E.valor_etq(v, cats[i0]), "valor alto (p90 / 1)": E.valor_etq(v, cats[i1]),
                      "P(mod.+sev.) media con valor bajo": medias[i0], "P(mod.+sev.) media con valor alto": medias[i1], "efecto medio (pp)": 100 * (medias[i1] - medias[i0])})
    t = pd.DataFrame(filas)
    t["|efecto medio| (pp)"] = t["efecto medio (pp)"].abs()
    return {"tabla": t.sort_values("|efecto medio| (pp)", ascending=False).reset_index(drop=True), "n_muestra": int(len(X0)), "p_mod_sev_media": float(p0.mean()), "semilla": SEMILLA}


# ============================================================ contrafactual
def _candidatos(v: str, spec_c: dict[str, Any], dev: pd.DataFrame, valores: dict[str, Any]) -> list[float]:
    """Valores candidatos PLAUSIBLES para la variable accionable ``v`` en el sentido de mejora (sin salir del rango p10/p90 de Desarrollo)."""
    a = E.ACCIONABLES[v]
    actual = valores.get(v)
    if actual is None or (isinstance(actual, float) and np.isnan(actual)):
        return []
    actual = float(actual)
    if a["tipo"] == "binaria":
        if a["direccion"] == "reducir":
            return [0.0] if actual == 1.0 else []
        return [1.0] if actual == 0.0 else []
    x, w = pd.to_numeric(dev[v], errors="coerce").to_numpy(), pd.to_numeric(dev[C.COL_PESO], errors="coerce").to_numpy()
    p10, p90 = [float(t) for t in cuantil_ponderado(x, w, [0.10, 0.90])]
    entero = a["tipo"] == "entero"
    if a["direccion"] == "aumentar":
        tope = p90
        if v == "n_ocupados" and "n_miembros" in valores:
            tope = min(tope, float(valores["n_miembros"]))
        if v in ("asistencia_escolar",):
            tope = min(tope, 1.0)
        if actual >= tope:
            return []
        pts = np.linspace(actual, tope, CF_N_PUNTOS + 1)[1:]
    else:
        piso = p10
        if actual <= piso:
            return []
        pts = np.linspace(actual, piso, CF_N_PUNTOS + 1)[1:]
    if entero:
        pts = np.round(pts)
        pts = pts[(pts != actual)]
    return sorted(set(float(p) for p in pts), reverse=(a["direccion"] == "reducir"))


def _distancia(v: str, actual: float, nuevo: float, rangos: dict[str, tuple[float, float]]) -> float:
    if E.ACCIONABLES[v]["tipo"] == "binaria":
        return 1.0
    lo, hi = rangos.get(v, (0.0, 1.0))
    return abs(nuevo - actual) / max(hi - lo, 1e-9)


def contrafactual(ens: dict[str, Any], res: dict[str, Any], dev: pd.DataFrame, spec: list[dict[str, Any]], valores: dict[str, Any], objetivo: float | None = None,
                  max_cambios: int = CF_MAX_CAMBIOS, n_escenarios: int = CF_N_ESCENARIOS, prevalencia_modsev: float | None = None) -> dict[str, Any]:
    """Búsqueda de cambios mínimos y plausibles en variables ACCIONABLES para bajar P(moderada o severa) según el propio ensamble.

    Método (voraz y reproducible, sin aleatoriedad): para cada variable accionable se evalúan valores candidatos entre el valor actual y su
    percentil 90 (o 10) de Desarrollo; se aplica en cada paso el cambio de menor distancia que ya alcanza el objetivo o, si ninguno lo
    alcanza, el de mayor reducción de probabilidad, hasta cruzar el objetivo o llegar a ``max_cambios`` variables; después se
    reduce cada cambio al valor más pequeño de la rejilla que mantiene el objetivo. El escenario 2 y el 3 excluyen la primera variable
    de los anteriores para ofrecer alternativas. Distancia = suma de |Δ| / (p90 − p10) (binarias: 1 por cambio).
    El objetivo por defecto es P(mod.+sev.) < (1 − 20 %) × P actual; ``prevalencia_modsev`` solo se informa como referencia.
    """
    if res["escenario"] != "socio":
        return {"aplica": False, "motivo": "El escenario del modelo usa ítems FIES como entrada: el nivel FIES se reconstruye a partir de ellos (circularidad). Un contrafactual sobre esas "
                                          "variables no tiene sentido y las variables derivadas del FIES no pueden ser accionables."}
    por_var = {c["variable"]: c for c in spec}
    miembros = ens["miembros"]
    Pb_m = np.stack([predecir_proba(m, _lote_filas(valores, [{}], res)) for m in miembros])[:, 0, :]
    p_base = Pb_m.mean(axis=0)
    pm_base = float(p_mod_sev(p_base))
    if objetivo is None:
        objetivo = (1 - CF_REDUCCION_RELATIVA) * pm_base
    acc_vars = [v for v in E.ACCIONABLES if v in por_var and E.es_accionable(v) and v in res["features_num"] + res["features_cat"]]
    rangos = {}
    for v in acc_vars:
        x, w = pd.to_numeric(dev[v], errors="coerce").to_numpy(), pd.to_numeric(dev[C.COL_PESO], errors="coerce").to_numpy()
        rangos[v] = tuple(float(t) for t in cuantil_ponderado(x, w, [0.10, 0.90]))
    cand = {v: _candidatos(v, por_var[v], dev, valores) for v in acc_vars}
    omitidas = [{"Variable": E.etq(v), "Código": v, "Motivo": "sin margen de cambio plausible (ya está en el extremo favorable de Desarrollo o sin dato)"} for v in acc_vars if not cand[v]]

    def evaluar(estado: dict[str, float], extra: list[dict[str, float]]) -> np.ndarray:
        return p_mod_sev(_proba_media(miembros, _lote_filas({**valores, **estado}, extra, res)))

    def dist_total(estado: dict[str, float]) -> float:
        return float(sum(_distancia(v, float(valores[v]), nv, rangos) for v, nv in estado.items()))

    escenarios: list[dict[str, Any]] = []
    excluidas: set[str] = set()
    n_evaluaciones = 0
    for _ in range(n_escenarios):
        estado: dict[str, float] = {}
        pm = pm_base
        primera = None
        while len(estado) < max_cambios and pm >= objetivo:
            opciones = [(v, val) for v in acc_vars if v not in estado and v not in excluidas for val in cand[v]]
            if not opciones:
                break
            pr = evaluar(estado, [{v: val} for v, val in opciones])
            n_evaluaciones += len(opciones)
            mejoras = pm - pr
            alcanzan = [i for i, (v, val) in enumerate(opciones) if pr[i] < objetivo and mejoras[i] > 0]
            if alcanzan:
                i = min(alcanzan, key=lambda j: _distancia(opciones[j][0], float(valores[opciones[j][0]]), opciones[j][1], rangos))
            else:                                       # ningún cambio solo alcanza el objetivo: el de mayor reducción dentro del rango plausible
                valido = [i for i in range(len(opciones)) if mejoras[i] > 1e-4]
                if not valido:
                    break
                i = max(valido, key=lambda j: mejoras[j])
            v, val = opciones[i]
            estado[v] = val
            pm = float(pr[i])
            primera = primera or v
        if not estado:
            break
        # reducir cada cambio al valor mínimo de la rejilla que mantiene el objetivo (si se alcanzó)
        if pm < objetivo:
            for v in list(estado):
                for val in cand[v]:                                    # rejilla ordenada de menor a mayor cambio
                    if val == estado[v]:
                        break
                    otro = {k: x for k, x in estado.items() if k != v}
                    p = float(evaluar({**otro, v: val}, [{}])[0])
                    n_evaluaciones += 1
                    if p < objetivo:
                        estado[v] = val
                        break
            pm = float(evaluar(estado, [{}])[0])
        escenarios.append({"estado": dict(estado), "primera": primera})
        excluidas.add(primera)
        if all(len(cand[v]) == 0 or v in excluidas for v in acc_vars):
            break
    # evaluar y documentar cada escenario con el ensamble completo
    salida = []
    for k, esc in enumerate(escenarios, start=1):
        est = esc["estado"]
        fila = _lote_filas({**valores, **est}, [{}], res)
        Pm = np.stack([predecir_proba(m, fila) for m in miembros])[:, 0, :]
        P = Pm.mean(axis=0)
        pm = float(p_mod_sev(P))
        cambios = [{"variable": v, "etiqueta": E.etq(v), "unidad": E.DICCIONARIO.get(v, {}).get("unidad", ""), "valor_actual": float(valores[v]), "valor_nuevo": float(nv),
                    "distancia": _distancia(v, float(valores[v]), float(nv), rangos)} for v, nv in est.items()]
        salida.append({"n": k, "cambios": cambios, "p": [float(a) for a in P], "p_mod_sev": pm, "reduccion_pp": 100 * (pm_base - pm), "nivel": int(P.argmax()), "distancia": dist_total(est),
                       "alcanza_objetivo": bool(pm < objetivo), "miembros_que_reducen": int((p_mod_sev(Pm) < p_mod_sev(Pb_m)).sum()), "n_miembros": int(len(miembros)),
                       "p_mod_sev_por_miembro": [float(a) for a in p_mod_sev(Pm)]})
    return {"aplica": True, "objetivo": float(objetivo), "p_base": [float(a) for a in p_base], "p_mod_sev_base": pm_base, "nivel_base": int(p_base.argmax()), "escenarios": salida,
            "accionables": acc_vars, "omitidas": omitidas, "n_evaluaciones": int(n_evaluaciones), "max_cambios": max_cambios, "rangos_p10_p90": rangos,
            "alcanzado": any(e["alcanza_objetivo"] for e in salida), "ya_cumple": bool(pm_base < objetivo), "prevalencia_modsev_ref": prevalencia_modsev, "semilla": SEMILLA}


# ============================================================ comparación con los demás modelos de la corrida
def comparar_modelos(dir_run: Path, escenario: str, valores: dict[str, Any], res: dict[str, Any], ident_seleccionado: str, umbrales: bool = True) -> pd.DataFrame:
    """Predicción del MISMO hogar con todos los modelos de la corrida y el escenario (media de cada ensamble)."""
    from fsai.reportes_fases import MODELOS_NOMBRE

    filas = []
    for ident, ens in cargar_modelos_corrida(dir_run, escenario).items():
        rs = ens["resultado"]
        fila = _lote_filas(valores, [{}], rs)
        r = INF.predecir_ensamble(ens["miembros"], fila, n_boot=300)
        P = r["media"][0]
        filas.append({"ident": ident, "Modelo": f"{rs['codigo']} · {MODELOS_NOMBRE.get(rs['codigo'], rs['codigo'])}", "codigo": rs["codigo"], **{f"P(nivel {k})": float(P[k]) for k in C.CLASES},
                      "Nivel predicho": int(P.argmax()), "P(mod.+sev.)": float(p_mod_sev(P)), "P(severa) IC95 inf": float(r["ic_inf"][0, 3]), "P(severa) IC95 sup": float(r["ic_sup"][0, 3]),
                      "Supera el umbral de severa": "Sí" if P[3] >= ens["umbral"] else "No", "Seleccionado": "★" if ident == ident_seleccionado else ""})
    return pd.DataFrame(filas)


def acuerdo_modelos(t: pd.DataFrame) -> dict[str, Any]:
    """Resumen del acuerdo entre modelos: nivel más votado, número de modelos que coinciden y rango de P(mod.+sev.)."""
    if len(t) == 0:
        return {"n": 0}
    votos = t["Nivel predicho"].value_counts()
    return {"n": int(len(t)), "nivel_modal": int(votos.index[0]), "n_modal": int(votos.iloc[0]), "p_min": float(t["P(mod.+sev.)"].min()), "p_max": float(t["P(mod.+sev.)"].max()),
            "todos_coinciden": bool(len(votos) == 1)}
