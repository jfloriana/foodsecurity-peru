"""explainability.py - SHAP, LIME, importancia por permutación y dependencia parcial.

Todas las funciones reciben el PIPELINE ya ajustado (pasos 'pre' y 'clf') y datos crudos.
Las explicaciones deben calcularse sobre datos de prueba (fuera de muestra) del pliegue
cuyo modelo se explica. Los resultados se serializan a estructuras JSON-compatibles
para la columna JSONB ``fsai.explicaciones.contribuciones``.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from fsai.config import CLASES, SEED
from fsai.models import AutoencoderXGB

logger = logging.getLogger(__name__)


def _matriz_modelo(modelo, X: pd.DataFrame) -> tuple[np.ndarray, list[str], Any]:
    """Transforma X con el preprocesador y devuelve (matriz que ve el clasificador, nombres, clf)."""
    pre, clf = modelo.named_steps["pre"], modelo.named_steps["clf"]
    Xt = np.asarray(pre.transform(X), dtype=float)
    nombres = [str(n) for n in pre.get_feature_names_out()]
    if isinstance(clf, AutoencoderXGB):          # SHAP explica la parte XGBoost sobre [x || z]
        return clf.transformar_entrada(Xt), clf.nombres_entrada(nombres), clf.xgb_
    return Xt, nombres, clf


def _normalizar_shap(valores, n_clases: int) -> np.ndarray:
    """Unifica el formato de salida de SHAP a un arreglo (clases, muestras, variables)."""
    if isinstance(valores, list):
        return np.stack([np.asarray(v) for v in valores], axis=0)
    v = np.asarray(valores)
    if v.ndim == 3:                               # (muestras, variables, clases)
        return np.transpose(v, (2, 0, 1))
    if v.ndim == 2:                               # una sola salida
        return v[None, ...]
    raise ValueError(f"Forma de SHAP inesperada: {v.shape}")


def _explicador(codigo: str, clf, fondo: np.ndarray):
    """TreeExplainer para RF/CatBoost/XGBoost; Linear para M1; Kernel (lento) para M4."""
    import shap
    if codigo in ("M2", "M3", "M5"):
        return shap.TreeExplainer(clf)
    if codigo == "M1":
        return shap.LinearExplainer(clf, fondo)
    resumen = shap.kmeans(fondo, min(30, len(fondo)))
    return shap.KernelExplainer(clf.predict_proba, resumen)


def _shap_catboost(clf, M: np.ndarray) -> np.ndarray:
    """TreeSHAP nativo de CatBoost (ShapValues) -> (clases, muestras, variables). Equivale a TreeExplainer pero evita el
    convertidor de shap, que no termina en tiempo razonable con árboles simétricos profundos (depth 10, 4 clases)."""
    from catboost import Pool
    sv = np.asarray(clf.get_feature_importance(Pool(M), type="ShapValues"))      # (muestras, clases, variables + sesgo)
    return np.transpose(sv[:, :, :-1], (1, 0, 2))


MAX_FILAS_BEESWARM = 500        # filas máximas guardadas en el beeswarm (100 para M4, KernelSHAP)
MAX_FILAS_BEESWARM_M4 = 100
MAX_VARIABLES_BEESWARM = 30     # variables máximas guardadas (las de mayor |SHAP| medio)


def shap_global(modelo, X: pd.DataFrame, codigo: str, n_muestra: int = 500, split: str | None = None) -> dict[str, Any]:
    """Importancia global: media de |SHAP| por variable y clase, sobre una muestra de X.

    Además devuelve ``beeswarm`` (SHAP FIRMADO de la clase IA severa por hogar, con las dummies sumadas por variable cruda y el valor
    normalizado de cada variable; ver ``beeswarm_datos``) para dibujar el beeswarm sin recalcular. ``split`` (p. ej. «prueba») solo se
    registra para trazabilidad. Es retrocompatible: las claves anteriores no cambian."""
    try:
        muestra = X.sample(n=min(n_muestra, len(X)), random_state=SEED)
        M, nombres, clf = _matriz_modelo(modelo, muestra)
        if codigo == "M3":
            vals = _shap_catboost(clf, M)
        else:
            expl = _explicador(codigo, clf, M)
            kw = {"nsamples": 200} if codigo == "M4" else {}
            if codigo in ("M2", "M3", "M5"):
                kw["check_additivity"] = False
            vals = _normalizar_shap(expl.shap_values(M, **kw), len(CLASES))
    except Exception:
        logger.exception("Fallo en shap_global (%s)", codigo)
        raise
    media_abs = np.abs(vals).mean(axis=1)                      # (clases, variables)
    out = {"nombres": nombres, "n_muestra": int(len(muestra)),
           "importancia_por_clase": {int(CLASES[k]): media_abs[k].tolist() for k in range(media_abs.shape[0])},
           "importancia_media": media_abs.mean(axis=0).tolist()}
    try:
        from fsai.config import CLASE_SEVERA
        k_sev = list(CLASES).index(CLASE_SEVERA) if vals.shape[0] > 1 else 0
        out["beeswarm"] = beeswarm_datos(vals[k_sev], muestra, nombres, clase=int(CLASE_SEVERA), split=split,
                                         max_filas=MAX_FILAS_BEESWARM_M4 if codigo == "M4" else MAX_FILAS_BEESWARM)
    except Exception:   # el beeswarm es un complemento: nunca debe impedir guardar la importancia global
        logger.exception("No se pudo armar el beeswarm (%s)", codigo)
        out["beeswarm"] = None
    return out


def shap_local(modelo, X_fila: pd.DataFrame, codigo: str, X_fondo: pd.DataFrame, clase: int) -> dict[str, Any]:
    """Contribuciones SHAP de un hogar para una clase (X_fila: DataFrame de 1 fila)."""
    if len(X_fila) != 1:
        raise ValueError("shap_local explica exactamente un hogar (DataFrame de 1 fila).")
    fondo = X_fondo.sample(n=min(100, len(X_fondo)), random_state=SEED)
    Mf, _, clf = _matriz_modelo(modelo, fondo)
    M1_, nombres, _ = _matriz_modelo(modelo, X_fila)
    if codigo == "M3":     # TreeSHAP nativo de CatBoost: el convertidor de shap puede colgarse o cerrar el proceso con árboles simétricos profundos
        from catboost import Pool
        sv = np.asarray(clf.get_feature_importance(Pool(M1_), type="ShapValues"))      # (1, clases, variables + sesgo)
        return {"nombres": nombres, "valores_shap": sv[0, clase, :-1].tolist(), "valores_variable": M1_[0].tolist(),
                "valor_base": float(sv[0, clase, -1]), "clase": int(clase)}
    expl = _explicador(codigo, clf, Mf)
    kw = {"check_additivity": False} if codigo in ("M2", "M3", "M5") else ({"nsamples": 200} if codigo == "M4" else {})
    vals = _normalizar_shap(expl.shap_values(M1_, **kw), len(CLASES))[clase, 0]
    base = np.atleast_1d(expl.expected_value)
    return {"nombres": nombres, "valores_shap": vals.tolist(), "valores_variable": M1_[0].tolist(),
            "valor_base": float(base[clase] if len(base) > clase else base[0]), "clase": int(clase)}


def lime_local(modelo, X_entrenamiento: pd.DataFrame, X_fila: pd.DataFrame, clase: int,
               n_variables: int = 10, n_muestras: int = 2000) -> dict[str, Any]:
    """Explicación LIME tabular sobre el espacio preprocesado (el fondo es solo TRAIN)."""
    from lime.lime_tabular import LimeTabularExplainer
    pre, clf = modelo.named_steps["pre"], modelo.named_steps["clf"]
    Mtr = np.asarray(pre.transform(X_entrenamiento), dtype=float)
    Mf = np.asarray(pre.transform(X_fila), dtype=float)
    nombres = [str(n) for n in pre.get_feature_names_out()]
    exp = LimeTabularExplainer(Mtr, feature_names=nombres, class_names=[str(c) for c in CLASES],
                               mode="classification", discretize_continuous=True, random_state=SEED)
    res = exp.explain_instance(Mf[0], clf.predict_proba, labels=[clase], num_features=n_variables,
                               num_samples=n_muestras)
    return {"clase": int(clase), "pesos": [(str(k), float(v)) for k, v in res.as_list(label=clase)],
            "r2_local": float(res.score)}


def importancia_permutacion(modelo, X: pd.DataFrame, y, pesos=None, n_repeticiones: int = 10) -> pd.DataFrame:
    """Importancia por permutación de columnas CRUDAS sobre datos de prueba (F1-macro ponderado)."""
    res = permutation_importance(modelo, X, np.asarray(y).astype(int), scoring="f1_macro",
                                 n_repeats=n_repeticiones, random_state=SEED, n_jobs=1,
                                 sample_weight=None if pesos is None else np.asarray(pesos, dtype=float))
    return (pd.DataFrame({"variable": X.columns, "importancia": res.importances_mean,
                          "desv_est": res.importances_std})
            .sort_values("importancia", ascending=False).reset_index(drop=True))


def dependencia_parcial(modelo, X: pd.DataFrame, variable: str, clase: int,
                        n_puntos: int = 20, pesos=None) -> dict[str, Any]:
    """PDP unidimensional de una variable numérica sobre la probabilidad de ``clase``."""
    if variable not in X.columns or not pd.api.types.is_numeric_dtype(X[variable]):
        raise ValueError(f"'{variable}' no es una columna numérica de X.")
    rejilla = np.unique(np.quantile(X[variable].dropna(), np.linspace(0.05, 0.95, n_puntos)))
    medias = []
    for v in rejilla:
        Xv = X.copy()
        Xv[variable] = v
        p = modelo.predict_proba(Xv)
        cols = list(modelo.named_steps["clf"].classes_)
        medias.append(float(np.average(p[:, cols.index(clase)], weights=pesos)) if clase in cols else 0.0)
    return {"variable": variable, "clase": int(clase), "rejilla": rejilla.tolist(), "prob_media": medias}


def _limpio(v: Any) -> Any:
    """Convierte a tipos JSON válidos (sin NaN/Inf ni tipos NumPy)."""
    if isinstance(v, dict):
        return {str(k): _limpio(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_limpio(x) for x in v]
    if isinstance(v, np.ndarray):
        return _limpio(v.tolist())
    if isinstance(v, (np.floating, float)):
        return None if (math.isnan(float(v)) or math.isinf(float(v))) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def contribuciones_a_jsonb(nombres: list[str], valores, extra: dict | None = None) -> dict[str, Any]:
    """Estructura estándar para fsai.explicaciones.contribuciones (JSONB)."""
    valores = np.asarray(valores, dtype=float).ravel()
    if len(nombres) != len(valores):
        raise ValueError("nombres y valores deben tener la misma longitud.")
    ordenado = sorted(zip(nombres, valores), key=lambda t: -abs(t[1]))
    # "features" es la clave exigida por el CHECK del DDL (fsai.explicaciones); "importancia" es la que lee la
    # página de SHAP; "variables" conserva el orden por magnitud.
    mapa = {n: v for n, v in ordenado}
    return _limpio({"features": mapa, "importancia": mapa,
                    "variables": [{"nombre": n, "valor": v} for n, v in ordenado], **(extra or {})})



# --------------------------------------------------------------------------
# Fase 4: mapas de calor de |SHAP|, máscaras de variables, curvas de eliminación/inserción y ALE
# (adaptación tabular: Grad-CAM++ / mapas de atención / máscaras de imagen NO aplican a datos tabulares)
# --------------------------------------------------------------------------
def agrupar_a_columna_cruda(nombres: list[str], columnas: list[str]) -> list[str]:
    """Asigna cada variable transformada (p. ej. 'cat__area_U', 'num__ingreso_pc') a su columna cruda de origen."""
    ordenadas = sorted(columnas, key=len, reverse=True)
    salida = []
    for n in nombres:
        resto = n.split("__", 1)[-1]
        salida.append(next((c for c in ordenadas if resto == c or resto.startswith(c + "_")), resto))
    return salida


def submuestra_estratificada(estrato, n: int, semilla: int = SEED) -> np.ndarray:
    """Posiciones (reproducibles) de una submuestra de ``n`` elementos con reparto PAREJO entre los valores de ``estrato``
    (p. ej. el nivel FIES): cada nivel aporta ``n/k`` hogares o todos los que tenga, y el sobrante se reparte entre los demás.
    No reproduce la prevalencia real: sirve para que las clases raras (IA severa) estén representadas."""
    est = np.asarray(estrato)
    rng = np.random.default_rng(semilla)
    grupos = {g: np.flatnonzero(est == g) for g in pd.unique(est[~pd.isna(est)])}
    n = int(min(n, sum(len(v) for v in grupos.values())))
    cupo = {g: 0 for g in grupos}
    restante = n
    while restante > 0:
        libres = [g for g in grupos if cupo[g] < len(grupos[g])]
        if not libres:
            break
        parte = max(1, restante // len(libres))
        for g in libres:
            t = min(parte, len(grupos[g]) - cupo[g], restante)
            cupo[g] += t
            restante -= t
            if restante == 0:
                break
    pos = np.concatenate([rng.choice(grupos[g], size=c, replace=False) for g, c in cupo.items() if c > 0]) if n else np.array([], int)
    return np.sort(pos)


def valor_normalizado(col: pd.Series) -> np.ndarray:
    """Valor de una variable normalizado a 0-1 para colorear el beeswarm: rango percentil (robusto a valores extremos) en numéricas
    continuas, 0/1 en binarias y NaN (se dibuja en gris) en categóricas, constantes o faltantes."""
    if not pd.api.types.is_numeric_dtype(col):
        return np.full(len(col), np.nan)
    x = col.astype(float)
    nun = x.nunique()
    if nun < 2:
        return np.full(len(col), np.nan)
    if nun == 2:
        return ((x - x.min()) / (x.max() - x.min())).to_numpy()
    return x.rank(pct=True, method="average").to_numpy()


def beeswarm_datos(vals_clase: np.ndarray, muestra: pd.DataFrame, nombres: list[str], clase: int | None = None, split: str | None = None,
                   top: int = MAX_VARIABLES_BEESWARM, max_filas: int = MAX_FILAS_BEESWARM) -> dict[str, Any]:
    """Arma el beeswarm de una clase: SHAP FIRMADO (hogares x variables CRUDAS, sumando las dummies de cada variable) y valor normalizado.

    ``vals_clase``: (hogares x variables transformadas) de ``shap_matriz``/``shap_global``; ``muestra``: las filas crudas correspondientes
    (mismo orden); ``nombres``: nombres transformados. Devuelve listas JSON (NaN -> None) con a lo sumo ``top`` variables (mayor |SHAP|
    medio, de mayor a menor) y ``max_filas`` hogares (submuestra reproducible). SHAP es aditivo: sumar las dummies de una variable
    equivale a su contribución conjunta."""
    vals_clase = np.asarray(vals_clase, dtype=float)
    if vals_clase.ndim != 2 or vals_clase.shape[0] != len(muestra) or vals_clase.shape[1] != len(nombres):
        raise ValueError("vals_clase debe ser (hogares x variables transformadas) alineada con muestra y nombres.")
    crudas = agrupar_a_columna_cruda(nombres, list(muestra.columns))
    cols = list(dict.fromkeys(crudas))
    shap_c = np.stack([vals_clase[:, [i for i, c in enumerate(crudas) if c == col]].sum(axis=1) for col in cols], axis=1)
    filas = np.arange(len(muestra))
    if len(filas) > max_filas:
        filas = np.sort(np.random.default_rng(SEED).choice(filas, size=max_filas, replace=False))
    orden = np.argsort(-np.abs(shap_c[filas]).mean(axis=0))[:top]
    sel = [cols[j] for j in orden]
    norm = np.stack([valor_normalizado(muestra[c]) if c in muestra.columns else np.full(len(muestra), np.nan) for c in sel], axis=1)
    return _limpio({"variables": sel, "shap": np.round(shap_c[filas][:, orden], 5), "valor_normalizado": np.round(norm[filas], 4),
                    "sin_color": [c for j, c in enumerate(sel) if np.isnan(norm[:, j]).all()],
                    "indice": [str(i) for i in muestra.index[filas]], "clase": clase, "split": split, "n_filas": int(len(filas)),
                    "normalizacion": "rango percentil (continuas), 0/1 (binarias); categóricas sin color"})


def shap_matriz(modelo, X: pd.DataFrame, codigo: str, n_muestra: int = 150, estrato=None) -> dict[str, Any]:
    """Valores SHAP completos (clases x muestras x variables) sobre una submuestra reproducible de X.

    ``estrato`` (opcional, alineado con X): si se da, la submuestra reparte los hogares por igual entre sus valores (p. ej. nivel FIES).

    ``agregado_crudo[c]`` suma |SHAP| de las dummies de cada columna cruda (clases x muestras x columnas crudas).
    """
    if estrato is not None:
        muestra = X.iloc[submuestra_estratificada(estrato, n_muestra)]
    else:
        muestra = X.sample(n=min(n_muestra, len(X)), random_state=SEED)
    M, nombres, clf = _matriz_modelo(modelo, muestra)
    if codigo == "M3":
        vals = _shap_catboost(clf, M)
    else:
        expl = _explicador(codigo, clf, M)
        kw = {"nsamples": 100} if codigo == "M4" else ({"check_additivity": False} if codigo in ("M2", "M3", "M5") else {})
        vals = _normalizar_shap(expl.shap_values(M, **kw), len(CLASES))
    crudas = agrupar_a_columna_cruda(nombres, list(X.columns))
    cols = list(dict.fromkeys(crudas))
    agr = np.stack([np.abs(vals[:, :, [i for i, c in enumerate(crudas) if c == col]]).sum(axis=2) for col in cols], axis=2)
    return {"indice": muestra.index.to_numpy(), "nombres": nombres, "vals": vals, "columnas_crudas": cols,
            "agregado_crudo": agr, "proba": modelo.predict_proba(muestra)}


def mascara_topk(agregado_clase: np.ndarray, k: int) -> np.ndarray:
    """Máscara binaria (muestras x columnas): 1 donde la columna está en el top-k de |SHAP| del hogar."""
    k = max(1, min(k, agregado_clase.shape[1]))
    orden = np.argsort(-agregado_clase, axis=1)[:, :k]
    m = np.zeros_like(agregado_clase, dtype=int)
    np.put_along_axis(m, orden, 1, axis=1)
    return m


def curva_eliminacion_insercion(modelo, X: pd.DataFrame, y, orden_columnas: list[str], pesos=None,
                                n_muestra: int = 400, semilla: int = SEED) -> pd.DataFrame:
    """Deletion/insertion curves por permutación (F1-macro).

    eliminación: se permutan (rompen) las k columnas más importantes y se mide el F1; insertión: partiendo de
    TODAS las columnas permutadas, se restauran las k más importantes. Ambas con la misma submuestra.
    """
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(semilla)
    idx = rng.choice(len(X), size=min(n_muestra, len(X)), replace=False)
    Xs = X.iloc[idx].reset_index(drop=True)
    ys = np.asarray(y).astype(int)[idx]
    ws = None if pesos is None else np.asarray(pesos, dtype=float)[idx]
    perm = {c: Xs[c].to_numpy()[rng.permutation(len(Xs))] for c in orden_columnas}

    def f1(Xm: pd.DataFrame) -> float:
        return float(f1_score(ys, modelo.predict(Xm), labels=CLASES, average="macro", zero_division=0, sample_weight=ws))

    filas = []
    for k in range(len(orden_columnas) + 1):
        Xe, Xi = Xs.copy(), Xs.copy()
        for c in orden_columnas[:k]:
            Xe[c] = perm[c]
        for c in orden_columnas[k:]:
            Xi[c] = perm[c]
        filas.append({"k": k, "variable_k": orden_columnas[k - 1] if k else "(ninguna)",
                      "f1_eliminacion": f1(Xe), "f1_insercion": f1(Xi)})
    return pd.DataFrame(filas)


def ale_1d(modelo, X: pd.DataFrame, variable: str, clase: int, n_bins: int = 10) -> dict[str, Any]:
    """Efectos locales acumulados (ALE, Apley y Zhao) de una variable numérica sobre P(clase), centrados."""
    if variable not in X.columns or not pd.api.types.is_numeric_dtype(X[variable]):
        raise ValueError(f"'{variable}' no es una columna numérica de X.")
    x = X[variable].astype(float)
    bordes = np.unique(np.quantile(x.dropna(), np.linspace(0, 1, n_bins + 1)))
    if len(bordes) < 3:
        raise ValueError(f"'{variable}' tiene muy pocos valores distintos para ALE.")
    cols = list(modelo.named_steps["clf"].classes_)
    if clase not in cols:
        raise ValueError(f"La clase {clase} no está en el modelo.")
    ci = cols.index(clase)
    cont = np.clip(np.searchsorted(bordes, x.fillna(x.median()), side="right") - 1, 0, len(bordes) - 2)
    efectos, n_por = [], []
    for b in range(len(bordes) - 1):
        m = cont == b
        n_por.append(int(m.sum()))
        if not m.any():
            efectos.append(0.0)
            continue
        Xi, Xs = X[m].copy(), X[m].copy()
        Xi[variable], Xs[variable] = bordes[b], bordes[b + 1]
        efectos.append(float(np.mean(modelo.predict_proba(Xs)[:, ci] - modelo.predict_proba(Xi)[:, ci])))
    ale = np.concatenate([[0.0], np.cumsum(efectos)])
    centro = float(np.average((ale[:-1] + ale[1:]) / 2, weights=np.maximum(n_por, 1)))
    return {"variable": variable, "clase": int(clase), "bordes": bordes.tolist(), "ale": (ale - centro).tolist(), "n_por_bin": n_por}


def muestra_ponderada(X: pd.DataFrame, pesos, n: int, semilla: int = SEED) -> tuple[pd.DataFrame, np.ndarray]:
    """Submuestra aleatoria ponderada por ``pesos`` (sin reemplazo). Devuelve (X_muestra, índices posicionales)."""
    if len(X) <= n:
        return X.reset_index(drop=True), np.arange(len(X))
    w = np.nan_to_num(np.asarray(pesos, dtype=float), nan=0.0)
    p = w / w.sum() if w.sum() > 0 else None
    idx = np.sort(np.random.default_rng(semilla).choice(len(X), size=n, replace=False, p=p))
    return X.iloc[idx].reset_index(drop=True), idx


def pdp_ale_principales(modelo, X: pd.DataFrame, pesos, variables: list[str], clase: int, n_muestra: int = 2000,
                        n_puntos: int = 15, n_bins: int = 10, max_categorias: int = 10) -> dict[str, Any]:
    """PDP y ALE de las ``variables`` (en el orden dado) sobre P(clase) con una submuestra ponderada de ``X``.

    Numéricas: PDP (rejilla por cuantiles 5-95 %) y ALE (bins por cuantiles). Categóricas: PDP por categoría (las
    ``max_categorias`` más frecuentes); el ALE no aplica. Una variable que falla se omite y queda registrada en
    ``omitidas``. Devuelve {"curvas", "omitidas", "n_muestra", "n_base"}.
    """
    Xs, _ = muestra_ponderada(X, pesos, n_muestra)
    cols = list(modelo.named_steps["clf"].classes_)
    ci = cols.index(clase) if clase in cols else None
    curvas, omitidas = [], []
    for v in variables:
        try:
            if v not in Xs.columns:
                raise ValueError("la variable no está en la matriz")
            if pd.api.types.is_numeric_dtype(Xs[v]):
                pdp = dependencia_parcial(modelo, Xs, v, clase, n_puntos=n_puntos)
                try:
                    ale = ale_1d(modelo, Xs, v, clase, n_bins=n_bins)
                except ValueError:
                    ale = None   # muy pocos valores distintos (p. ej. binaria): solo PDP
                curvas.append({"variable": v, "tipo": "numerica", "clase": int(clase), "pdp": pdp, "ale": ale, "n_muestra": len(Xs)})
            else:
                cats = Xs[v].astype(object).dropna().astype(str).value_counts().index[:max_categorias].tolist()
                medias = []
                for c in cats:
                    Xv = Xs.copy()
                    Xv[v] = c
                    medias.append(float(modelo.predict_proba(Xv)[:, ci].mean()) if ci is not None else 0.0)
                curvas.append({"variable": v, "tipo": "categorica", "clase": int(clase), "categorias": cats, "prob_media": medias, "ale": None, "n_muestra": len(Xs)})
        except Exception as exc:  # noqa: BLE001 - una variable problemática no detiene el informe
            logger.warning("PDP/ALE omitido para %s: %s", v, exc)
            omitidas.append({"variable": v, "motivo": f"{type(exc).__name__}: {exc}"})
    return {"curvas": curvas, "omitidas": omitidas, "n_muestra": int(len(Xs)), "n_base": int(len(X))}


def tabla_dependencia(curvas: list[dict[str, Any]]) -> pd.DataFrame:
    """Valores numéricos de las curvas en formato largo: variable, tipo de efecto, valor/categoría, estimación."""
    filas = []
    for c in curvas:
        v = c["variable"]
        if c["tipo"] == "categorica":
            filas += [{"variable": v, "efecto": "PDP (prob. media)", "valor": k, "estimación": m} for k, m in zip(c["categorias"], c["prob_media"])]
            continue
        filas += [{"variable": v, "efecto": "PDP (prob. media)", "valor": float(g), "estimación": float(m)} for g, m in zip(c["pdp"]["rejilla"], c["pdp"]["prob_media"])]
        if c.get("ale"):
            b = np.asarray(c["ale"]["bordes"], dtype=float)
            filas += [{"variable": v, "efecto": "ALE (centrado)", "valor": float(x), "estimación": float(a)} for x, a in zip(b, c["ale"]["ale"])]
    return pd.DataFrame(filas)
