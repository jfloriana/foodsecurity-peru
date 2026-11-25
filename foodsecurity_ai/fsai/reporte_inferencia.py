"""reporte_inferencia.py - Reporte descargable (Word, Excel y PDF) de UNA inferencia (individual o por lote) de FoodSecurity AI.

Dos capas, sin dependencia de Streamlit:

1. ``ejecutar_inferencia_individual`` / ``ejecutar_inferencia_lote`` calculan la inferencia y devuelven un *snapshot* congelado
   (entradas, modelo y corrida, resultados, explicación, comparaciones, interpretaciones y trazabilidad). La página de Inferencia
   muestra EXACTAMENTE ese snapshot, de modo que el reporte coincide con lo que se vio en pantalla y no cambia si el usuario
   modifica después los controles.
2. ``generar_archivos`` regenera las figuras a 600 DPI (Matplotlib, estilo y paleta Okabe-Ito del sistema), construye el modelo de
   documento de ``fsai.report_generator`` y lo exporta a .docx, .xlsx y .pdf reutilizando esos exportadores.

Solo vuelca resultados ya calculados y archivos reales de la corrida (métricas, hiperparámetros, hashes); no inventa cifras.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import logging
import platform
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fsai import config as C  # noqa: E402
from fsai import etiquetas as E  # noqa: E402
from fsai import explainability as X  # noqa: E402
from fsai import inference as INF  # noqa: E402
from fsai import interpretacion as I  # noqa: E402
from fsai import perfil_vulnerabilidad as PV
from fsai import plots as P  # noqa: E402
from fsai import analisis_local as AL  # noqa: E402
import time  # noqa: E402
from fsai.preprocessing import preparar_matriz  # noqa: E402
from fsai.report_generator import exportar_excel, exportar_pdf, exportar_word  # noqa: E402
from fsai.reportes_fases import ETQ_METRICA, MODELOS_NOMBRE, PAQUETES  # noqa: E402

logger = logging.getLogger(__name__)

VERSION_REPORTE = "1.1"
N_BOOT_INDIVIDUAL = 1000
N_BOOT_LOTE = 300
NOMBRE_NIVEL = {0: "Seguridad alimentaria", 1: "Inseguridad leve", 2: "Inseguridad moderada", 3: "Inseguridad severa"}
FORMATOS = {
    "word": ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "Word"),
    "excel": ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "Excel"),
    "pdf": ("pdf", "application/pdf", "PDF"),
}

# Hojas del libro Excel para las figuras de este reporte (ver fsai.excel_graficos.GRUPOS_FIGURA)
from fsai import excel_graficos as _EG  # noqa: E402

_EG.GRUPOS_FIGURA.update({
    "fig_inf_probabilidades": ("Fig_Probabilidades", "Probabilidad por nivel FIES (media del ensamble) con IC95 bootstrap sobre los miembros y comparación con la prevalencia de referencia; gráfico nativo de barras."),
    "fig_inf_miembros": ("Fig_Miembros", "Probabilidad que asigna cada miembro del ensamble a cada nivel FIES (variabilidad del modelo)."),
    "fig_inf_shap_local": ("Fig_SHAP_local", "Contribuciones SHAP locales del modelo final para este hogar, ordenadas por magnitud; gráfico nativo de barras (15 principales)."),
    "fig_inf_percentiles": ("Fig_Percentiles", "Percentil del hogar en cada variable numérica frente a la distribución poblacional ponderada de Desarrollo; gráfico nativo de barras."),
    "fig_lote_clases": ("Fig_Lote_Clases", "Distribución de los niveles FIES predichos para el lote frente a la prevalencia de referencia; gráfico nativo de barras."),
    "fig_lote_psevera": ("Fig_Lote_P_severa", "Histograma de P(IA severa) de los hogares del lote y umbral operativo; gráfico nativo de barras."),
    "fig_lote_shap_global": ("Fig_SHAP_global", "Importancia global SHAP guardada del modelo; gráfico nativo de barras (15 principales)."),
    "fig_lote_probabilidades": ("Fig_Lote_Probabilidades", "Distribución (percentiles) de la probabilidad asignada a cada nivel FIES en el lote."),
    "fig_inf_lime": ("Fig_LIME", "Pesos LIME locales (aproximación local del modelo final) para este hogar; gráfico nativo de barras (12 principales)."),
    "fig_inf_lime_shap": ("Fig_LIME_vs_SHAP", "Comparación de LIME con SHAP agregado por variable: efecto normalizado por método (signo y orden), gráfico nativo de barras."),
    "fig_inf_sensibilidad": ("Fig_Sensibilidad", "Curvas de sensibilidad (ICE local) de las variables principales: probabilidad por nivel FIES al variar una variable y dejar fijo el resto del hogar; gráficos nativos."),
    "fig_inf_tornado": ("Fig_Tornado", "Tornado de sensibilidad: cambio de P(moderada o severa) al mover cada variable entre su percentil 10 y su percentil 90; gráfico nativo de barras."),
    "fig_inf_contrafactual": ("Fig_Contrafactual", "Escenarios contrafactuales (asociación predictiva del modelo, no efecto causal): probabilidad actual y de cada escenario frente al objetivo; gráfico nativo."),
    "fig_inf_otros_modelos": ("Fig_Otros_modelos", "Probabilidad por nivel FIES del mismo hogar con todos los modelos de la corrida; gráfico nativo de barras agrupadas."),
    "fig_lote_tornado": ("Fig_Lote_Tornado", "Sensibilidad agregada del lote: efecto medio sobre P(moderada o severa) de mover cada variable entre p10 y p90 (muestra del lote); gráfico nativo."),
})


# ============================================================ utilidades
def _pct(x: float, d: int = 1) -> str:
    return f"{100 * float(x):.{d}f} %"


def _n3(x: float) -> str:
    return f"{float(x):.3f}"


def _sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with Path(ruta).open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _json(ruta: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(ruta).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _versiones() -> pd.DataFrame:
    filas = [{"componente": "Python", "versión": platform.python_version()}, {"componente": "Sistema operativo", "versión": platform.platform()}]
    for p in PAQUETES:
        try:
            filas.append({"componente": p, "versión": md.version(p)})
        except md.PackageNotFoundError:
            continue
    return pd.DataFrame(filas)


def prevalencia_referencia(ref: pd.DataFrame) -> np.ndarray:
    """Distribución ponderada de los niveles FIES (0-3) en la muestra de referencia (Desarrollo)."""
    y, w = pd.to_numeric(ref[C.COL_OBJETIVO], errors="coerce"), pd.to_numeric(ref[C.COL_PESO], errors="coerce")
    m = y.notna() & w.notna()
    s = w[m].groupby(y[m].astype(int)).sum().reindex(C.CLASES, fill_value=0.0)
    return (s / s.sum()).to_numpy(dtype=float)


def info_modelo(dir_run: Path, nombre: str, ident: str, res: dict[str, Any], umbral: float, n_miembros: int) -> dict[str, Any]:
    """Modelo y corrida seleccionados: identificación, hiperparámetros y métricas REALES leídas de los artefactos de la corrida."""
    cfg = _json(dir_run / "config.json")
    prueba = _json(dir_run / f"prueba_{ident}.json")
    ruta_modelo = dir_run / res["final"]["modelo"]
    sha = prueba.get("sha256_modelo") or (_sha256_archivo(ruta_modelo) if ruta_modelo.is_file() else "no disponible")
    met_prueba = None
    f_met = dir_run / "analisis" / f"metricas_prueba_{ident}.csv"
    if f_met.is_file():
        met_prueba = pd.read_csv(f_met)
    codigo = res["codigo"]
    return {
        "corrida": nombre, "id": ident, "codigo": codigo, "algoritmo": MODELOS_NOMBRE.get(codigo, codigo), "escenario": res["escenario"],
        "hiperparametros": dict(res["final"].get("hiperparametros") or {}), "features_num": list(res["features_num"]),
        "features_cat": list(res["features_cat"]), "n_trials": res.get("n_trials_final", res.get("n_trials")), "n_outer": res.get("n_outer"),
        "n_inner": res.get("n_inner"), "semilla": res.get("semilla", C.SEED), "umbral": float(umbral), "n_miembros": int(n_miembros),
        "sha256_modelo": sha, "hash_datos": cfg.get("hash_datos", ""), "huella_particion": cfg.get("huella_particion", ""),
        "fecha_corrida": prueba.get("evaluado_en", ""), "n_prueba": prueba.get("n_prueba"),
        "tamanos_particion": (cfg.get("verificacion_particion") or {}).get("tamanos", {}),
        "metricas_prueba": met_prueba, "metricas_validacion": (res["final"].get("validacion") or {}).get("metricas_ponderadas") or {},
        "metricas_cv": (res.get("agregado") or {}).get("ponderada") or {},
        "foco_umbral": {k: res["final"]["validacion"].get(k) for k in ("recall_severa_umbral", "especificidad_severa_umbral")},
    }


def _tabla_metricas(m: dict[str, Any]) -> pd.DataFrame:
    """Tabla de métricas del modelo: Prueba (valor e IC95 bootstrap), Validación y CV anidada (media e IC95)."""
    filas = []
    pr = m.get("metricas_prueba")
    prp = pr[pr["ponderada"].astype(str).str.lower() == "true"].set_index("metrica") if pr is not None and len(pr) else None
    claves = [k for k in ETQ_METRICA if k in (m["metricas_cv"] or {}) or k in (m["metricas_validacion"] or {}) or (prp is not None and k in prp.index)]
    for k in claves:
        f: dict[str, Any] = {"Métrica": ETQ_METRICA[k]}
        if prp is not None:
            if k in prp.index:
                r = prp.loc[k]
                f.update({"Prueba (valor)": float(r["valor"]), "Prueba IC95 inf": float(r["ic_inf"]), "Prueba IC95 sup": float(r["ic_sup"])})
            else:
                f.update({"Prueba (valor)": None, "Prueba IC95 inf": None, "Prueba IC95 sup": None})
        f["Validación (valor)"] = (m["metricas_validacion"] or {}).get(k)
        cv = (m["metricas_cv"] or {}).get(k)
        f.update({"CV anidada (media)": cv.get("media") if cv else None, "CV IC95 inf": cv.get("ic_inf") if cv else None, "CV IC95 sup": cv.get("ic_sup") if cv else None})
        filas.append(f)
    return pd.DataFrame(filas)


def _valor_legible(campo: dict[str, Any], v: Any) -> tuple[str, str]:
    var = campo["variable"]
    if campo["tipo"] == "categorica":
        return E.valor_etq(var, v), str(v)
    if campo["tipo"] == "binaria":
        return ("Sí" if float(v) == 1 else "No"), str(int(float(v)))
    return f"{float(v):.4g}", f"{float(v):.6g}"


def _tabla_entradas(spec: list[dict[str, Any]], valores: dict[str, Any]) -> pd.DataFrame:
    filas = []
    for c in spec:
        var = c["variable"]
        leg, orig = _valor_legible(c, valores[var])
        if c["tipo"] == "categorica":
            rango = f"{len(c['opciones'])} categorías"
        elif c["tipo"] == "binaria":
            rango = "0 / 1"
        else:
            rango = f"{c['min']:.4g} a {c['max']:.4g}"
        filas.append({"Variable": E.etq(var), "Código": var, "Unidad / valores": E.DICCIONARIO.get(var, {}).get("unidad", ""),
                      "Valor": leg, "Original": orig, "Rango en Desarrollo": rango})
    return pd.DataFrame(filas)


def _nivel_texto(k: int) -> str:
    return f"{k} · {NOMBRE_NIVEL[k]}"


def _advertencias(m: dict[str, Any]) -> list[str]:
    """Advertencias sobrias derivadas de las métricas reales del modelo y del escenario."""
    a = []
    pr = m.get("metricas_prueba")
    base = None
    if pr is not None and len(pr):
        t = pr[pr["ponderada"].astype(str).str.lower() == "true"].set_index("metrica")
        if {"f1_macro", "roc_auc_ovr"} <= set(t.index):
            base = f"F1-macro ponderado {_n3(t.loc['f1_macro', 'valor'])} (IC95 {_n3(t.loc['f1_macro', 'ic_inf'])}-{_n3(t.loc['f1_macro', 'ic_sup'])}) y AUC ROC OvR {_n3(t.loc['roc_auc_ovr', 'valor'])} en la Prueba"
    if base is None:
        cv = m.get("metricas_cv") or {}
        if "f1_macro" in cv and "roc_auc_ovr" in cv:
            base = f"F1-macro ponderado medio {_n3(cv['f1_macro']['media'])} y AUC ROC OvR medio {_n3(cv['roc_auc_ovr']['media'])} en la validación cruzada anidada (la Prueba no se evaluó para este modelo)"
    if m["escenario"] == "socio":
        if base:
            a.append(f"Capacidad predictiva modesta: el modelo alcanza {base}. Una parte importante de los hogares se clasifica en un nivel distinto del real; la probabilidad es una estimación del modelo, no una certeza.")
        else:
            a.append("No se encontraron métricas del modelo en la corrida: no se puede cuantificar su capacidad predictiva y no debe asumirse que sea alta.")
    else:
        a.append("Este escenario usa los ítems FIES como variables de entrada y el nivel FIES se deriva de esos mismos ítems: el resultado es una RECONSTRUCCIÓN del nivel a partir de sus componentes (circularidad), no una predicción. Su desempeño casi perfecto no mide capacidad predictiva.")
    if m["escenario"] == "socio":
        a.append("Los escenarios «fies» e «hibrido» del sistema usan los ítems FIES como entrada y son una reconstrucción del nivel FIES, no una predicción; este modelo usa solo variables socioeconómicas.")
    a.append("Los umbrales que definen los niveles FIES sobre el puntaje crudo (leve >= 1, moderada >= 4, severa >= 7) y la escala Rasch están marcados [POR VALIDAR]; no se afirma calibración peruana de la escala ni coincidencia con la clasificación oficial.")
    a.append("Este reporte no es un diagnóstico individual ni sustituye la medición directa con la escala FIES aplicada al hogar. No debe usarse para negar, restringir ni condicionar ayudas sociales.")
    a.append("La probabilidad es el promedio de un ensamble de modelos; el intervalo mide la variabilidad entre los miembros del ensamble, no es un intervalo de confianza frecuentista de la probabilidad verdadera ni incorpora la incertidumbre de muestreo de la encuesta.")
    a.append("Las explicaciones (SHAP) describen cómo el modelo final usa las variables; son asociaciones estadísticas aprendidas de otros hogares y no efectos causales.")
    return a


def _resumen_metricas_texto(m: dict[str, Any]) -> str:
    pr = m.get("metricas_prueba")
    if pr is not None and len(pr):
        t = pr[pr["ponderada"].astype(str).str.lower() == "true"].set_index("metrica")
        partes = [f"{ETQ_METRICA[k]} {_n3(t.loc[k, 'valor'])} [{_n3(t.loc[k, 'ic_inf'])}; {_n3(t.loc[k, 'ic_sup'])}]" for k in ("f1_macro", "balanced_accuracy", "roc_auc_ovr", "recall_severa") if k in t.index]
        n = f" (n = {int(m['n_prueba']):,} hogares)" if m.get("n_prueba") else ""
        return "Desempeño del modelo en la Prueba independiente" + n + ", métricas ponderadas, IC95 bootstrap: " + "; ".join(partes) + "."
    cv = m.get("metricas_cv") or {}
    partes = [f"{ETQ_METRICA[k]} {_n3(cv[k]['media'])} [{_n3(cv[k]['ic_inf'])}; {_n3(cv[k]['ic_sup'])}]" for k in ("f1_macro", "balanced_accuracy", "roc_auc_ovr", "recall_severa") if k in cv]
    return "La Prueba no se evaluó para este modelo; desempeño en la validación cruzada anidada (media e IC95 entre pliegues): " + "; ".join(partes) + "." if partes else "No hay métricas disponibles en la corrida."


# ============================================================ inferencia individual
def ejecutar_inferencia_individual(*, nombre_corrida: str, ident: str, dir_run: Path, res: dict[str, Any], ens: dict[str, Any], dev: pd.DataFrame,
                                   valores: dict[str, Any], usuario: str = "", dataset: dict[str, Any] | None = None,
                                   df_perfil: pd.DataFrame | None = None) -> dict[str, Any]:
    """Calcula la inferencia individual y devuelve el snapshot congelado (todo lo que se muestra y se reporta)."""
    spec = INF.especificacion_formulario(res, dev)
    valores = {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) else v) for k, v in valores.items()}
    fila = INF.fila_desde_formulario(valores, res)
    r = INF.predecir_ensamble(ens["miembros"], fila, n_boot=N_BOOT_INDIVIDUAL)
    s = INF.severidad(r["media"], ens["umbral"])
    media, lo, hi = r["media"][0], r["ic_inf"][0], r["ic_sup"][0]
    nivel, p3 = int(s["nivel_argmax"][0]), float(s["p_severa"][0])
    pred_val = np.load(dir_run / f"pred_val_{ident}.npz", allow_pickle=False)
    perc = INF.percentil_poblacional(p3, pred_val["proba"][:, C.CLASE_SEVERA], pred_val["w"])
    ahora = datetime.now()
    huella = hashlib.sha256(json.dumps(valores, sort_keys=True, default=str).encode()).hexdigest()
    # explicación local (SHAP del modelo final)
    shap_d, shap_err = None, ""
    try:
        Xdev = preparar_matriz(dev, res["features_num"], res["features_cat"])
        sl = X.shap_local(ens["final"], fila, res["codigo"], Xdev, nivel)
        shap_d = {"nombres": list(sl["nombres"]), "valores": [float(v) for v in sl["valores_shap"]], "valores_variable": [float(v) for v in sl["valores_variable"]],
                  "valor_base": float(sl["valor_base"]), "clase": int(sl["clase"])}
    except Exception as exc:  # noqa: BLE001
        shap_err = f"{type(exc).__name__}: {exc}"
    # comparación con la población
    comp = []
    for campo in spec:
        v = campo["variable"]
        if campo["tipo"] == "numerica":
            x = pd.to_numeric(dev[v], errors="coerce")
            m = x.notna()
            comp.append({"variable": E.etq(v, True), "código": v, "valor del hogar": float(valores[v]), "mediana poblacional": float(np.median(x[m])),
                         "percentil del hogar": INF.percentil_poblacional(float(valores[v]), x[m].to_numpy(), dev.loc[m, C.COL_PESO].to_numpy())})
    comp_df = pd.DataFrame(comp)
    prev = prevalencia_referencia(dev)
    modelo = info_modelo(dir_run, nombre_corrida, ident, res, ens["umbral"], len(ens["miembros"]))
    extra = _analisis_extra(dir_run=dir_run, ident=ident, res=res, ens=ens, dev=dev, spec=spec, valores=valores, nivel=nivel, shap_d=shap_d, prev=prev)
    interp = {"prediccion": I.i_prediccion({k: float(media[k]) for k in C.CLASES}, nivel, p3, ens["umbral"], perc)}
    if shap_d:
        sv = pd.Series(shap_d["valores"], index=shap_d["nombres"])
        top = sv.reindex(sv.abs().sort_values(ascending=False).index).head(3)
        interp["shap"] = {"que_muestra": f"Para el nivel {nivel}, las variables que más mueven la predicción de este hogar son " + ", ".join(f"«{E.etq(v)}» ({x:+.3f})" for v, x in top.items()) + ".",
                          "como_leerlo": "Rojo: aumenta la probabilidad de ese nivel; azul: la reduce (SHAP, Lundberg y Lee 2017).",
                          "limitacion": "Explica al modelo final, no a todo el ensamble, y no es causal."}
    if len(comp_df):
        mx = comp_df.iloc[(comp_df["percentil del hogar"] - 0.5).abs().argmax()]
        interp["comparacion"] = {"que_muestra": f"La variable en que este hogar más se aleja del centro poblacional es «{mx['variable']}» (percentil {mx['percentil del hogar']:.0%}).",
                                 "como_leerlo": "Un percentil cercano a 0 % o 100 % indica un valor poco común; cercano a 50 %, típico.",
                                 "limitacion": "La población de referencia es la muestra de Desarrollo, no la población peruana."}
    interp.update(_interpretaciones_extra(extra, modelo, nivel))
    perfil_ctx, perfil_err = None, ""
    try:
        ref_perfil = df_perfil if df_perfil is not None else dev
        perfil_ctx = PV.contexto_hogar(valores, PV.perfil(ref_perfil))
        perfil_ctx["n_referencia"] = int(len(ref_perfil))
        perfil_ctx["referencia"] = "conjunto analítico del dataset activo" if df_perfil is not None else "muestra de Desarrollo"
    except Exception as exc:  # noqa: BLE001 - el contexto del perfil no debe tumbar la inferencia
        logger.exception("Perfil de vulnerabilidad")
        perfil_err = f"{type(exc).__name__}: {exc}"
        perfil_ctx = {**PV.pertenencia_hogar(valores), "grupos": [], "recuento_fila": None, "disponible": False}
    return {
        **extra, "perfil_ctx": perfil_ctx, "perfil_error": perfil_err,
        "tipo": "individual", "id": f"INF-{ahora:%Y%m%d-%H%M%S}-{huella[:8]}", "fecha": ahora.isoformat(timespec="seconds"), "usuario": usuario or "",
        "dataset": dict(dataset or {}), "modelo": modelo, "spec": spec, "valores": valores, "entradas": _tabla_entradas(spec, valores), "huella": huella,
        "media": np.asarray(media), "ic_inf": np.asarray(lo), "ic_sup": np.asarray(hi), "min": np.asarray(r["min"][0]), "max": np.asarray(r["max"][0]),
        "por_miembro": np.asarray(r["por_miembro"][:, 0, :]), "nivel": nivel, "p_severa": p3, "umbral": float(ens["umbral"]), "supera": bool(p3 >= ens["umbral"]),
        "p_severa_ic": (float(lo[C.CLASE_SEVERA]), float(hi[C.CLASE_SEVERA])), "percentil": float(perc), "prevalencia": prev, "shap": shap_d, "shap_error": shap_err,
        "comparacion": comp_df, "interp": interp, "n_ref_desarrollo": int(len(dev)),
    }



def _analisis_extra(*, dir_run: Path, ident: str, res: dict[str, Any], ens: dict[str, Any], dev: pd.DataFrame, spec: list[dict[str, Any]], valores: dict[str, Any],
                    nivel: int, shap_d: dict[str, Any] | None, prev: np.ndarray) -> dict[str, Any]:
    """LIME, comparación con SHAP, sensibilidad, contrafactual, otros modelos y validez de entradas. Cada análisis se aísla: si uno falla, el resto sigue y
    el error queda declarado en ``extra_err`` (nunca se inventa un resultado)."""
    out: dict[str, Any] = {"lime": None, "lime_shap": None, "sens": None, "contra": None, "otros": None, "validez": None}
    err: dict[str, str] = {}
    tiempos: dict[str, float] = {}
    feats = list(res["features_num"]) + list(res["features_cat"])

    def paso(nombre: str, fn):
        t0 = time.perf_counter()
        try:
            out[nombre] = fn()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Falló el análisis %s", nombre)
            err[nombre] = f"{type(exc).__name__}: {exc}"
        tiempos[nombre] = round(time.perf_counter() - t0, 2)

    paso("validez", lambda: AL.validez_entradas(spec, valores, dev))
    paso("lime", lambda: AL.lime_local(ens["final"], dev, valores, res, nivel))
    if out["lime"] is not None:
        paso("lime_shap", lambda: AL.comparar_lime_shap(out["lime"], shap_d, feats))
    if out["lime_shap"] is not None and out["lime_shap"].get("disponible"):
        orden = list(out["lime_shap"]["tabla"]["variable"])
    elif out["lime"] is not None:
        orden = list(out["lime"]["tabla"]["variable"])
    else:
        orden = feats
    paso("sens", lambda: AL.sensibilidad(ens, res, dev, spec, valores, orden))
    paso("contra", lambda: AL.contrafactual(ens, res, dev, spec, valores, prevalencia_modsev=float(prev[2] + prev[3])))
    paso("otros", lambda: AL.comparar_modelos(dir_run, res["escenario"], valores, res, ident))
    out["extra_err"] = err
    out["tiempos"] = tiempos
    return out


def _interpretaciones_extra(extra: dict[str, Any], modelo: dict[str, Any], nivel: int) -> dict[str, Any]:
    ip: dict[str, Any] = {}
    if extra.get("lime") is not None:
        L = extra["lime"]
        ip["lime"] = I.i_lime(L["tabla"], nivel, L["r2_local"], L["prob_modelo"], L["prob_local"])
    if extra.get("lime_shap") is not None:
        ip["lime_shap"] = I.i_lime_shap(extra["lime_shap"])
    if extra.get("sens") is not None:
        S = extra["sens"]
        ip["sens"] = I.i_sensibilidad(S["curvas"], S["principales"], S["p_mod_sev_base"])
        ip["tornado"] = I.i_tornado(S["tornado"], S["p_mod_sev_base"])
    if extra.get("contra") is not None:
        ip["contra"] = I.i_contrafactual(extra["contra"], _resumen_metricas_texto(modelo))
    if extra.get("otros") is not None:
        ip["otros"] = I.i_otros_modelos(extra["otros"], f"{modelo['codigo']}")
    if extra.get("validez") is not None:
        ip["validez"] = I.i_validez(extra["validez"])
    return ip


# ============================================================ inferencia por lote
def ejecutar_inferencia_lote(*, nombre_corrida: str, ident: str, dir_run: Path, res: dict[str, Any], ens: dict[str, Any], dev: pd.DataFrame,
                             dfl: pd.DataFrame, bytes_csv: bytes, nombre_archivo: str = "", usuario: str = "", dataset: dict[str, Any] | None = None,
                             df_perfil: pd.DataFrame | None = None) -> dict[str, Any]:
    """Predicción por lote y snapshot congelado con resumen agregado, distribución de clases y tabla de resultados."""
    out = INF.predecir_lote(ens, dfl)
    req = res["features_num"] + res["features_cat"]
    ahora = datetime.now()
    huella = hashlib.sha256(bytes_csv).hexdigest()
    prev = prevalencia_referencia(dev)
    n = len(out)
    cont = out["nivel_predicho"].value_counts().reindex(C.CLASES, fill_value=0)
    medias = np.array([out[f"p_clase_{k}"].mean() for k in C.CLASES])
    # resumen por variable de entrada
    filas = []
    for c in req:
        s = dfl[c]
        if c in res["features_num"]:
            x = pd.to_numeric(s, errors="coerce")
            filas.append({"Variable": E.etq(c), "Código": c, "Unidad / valores": E.DICCIONARIO.get(c, {}).get("unidad", ""), "n válidos": int(x.notna().sum()),
                          "% faltante": 100 * float(x.isna().mean()), "mínimo": x.min(), "mediana": x.median(), "media": x.mean(), "máximo": x.max(), "categoría más frecuente": ""})
        else:
            mod = s.astype(str).mode()
            filas.append({"Variable": E.etq(c), "Código": c, "Unidad / valores": E.DICCIONARIO.get(c, {}).get("unidad", ""), "n válidos": int(s.notna().sum()),
                          "% faltante": 100 * float(s.isna().mean()), "mínimo": None, "mediana": None, "media": None, "máximo": None,
                          "categoría más frecuente": E.valor_etq(c, mod.iloc[0]) if len(mod) else ""})
    entradas = dfl[req].reset_index(drop=True)
    modelo = info_modelo(dir_run, nombre_corrida, ident, res, ens["umbral"], len(ens["miembros"]))
    pser = out["p_clase_3"].to_numpy()
    imp = None
    extra_l: dict[str, Any] = {"sens_lote": None, "validez_lote": None}
    err_l: dict[str, str] = {}
    for clave, fn in (("sens_lote", lambda: AL.sensibilidad_lote(ens, res, dev, dfl)), ("validez_lote", lambda: AL.validez_lote(dfl, res, dev))):
        try:
            extra_l[clave] = fn()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Falló el análisis %s del lote", clave)
            err_l[clave] = f"{type(exc).__name__}: {exc}"
    perfil_l = None
    try:
        ref_perfil = df_perfil if df_perfil is not None else dev
        perfil_l = PV.perfil_lote(dfl, PV.perfil(ref_perfil))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Perfil de vulnerabilidad del lote")
        err_l["perfil"] = f"{type(exc).__name__}: {exc}"
    f_shap = dir_run / "analisis" / f"shap_{ident}.json"
    if f_shap.is_file():
        d = _json(f_shap)
        imp = {"importancia": {str(k): float(v) for k, v in (d.get("importancia") or {}).items()}, "explainer": d.get("explainer", "SHAP"), "n_muestra": d.get("n_muestra"), "archivo": f_shap.name}
    return {
        **extra_l, "extra_err": err_l, "perfil_lote": perfil_l,
        "tipo": "lote", "id": f"LOT-{ahora:%Y%m%d-%H%M%S}-{huella[:8]}", "fecha": ahora.isoformat(timespec="seconds"), "usuario": usuario or "", "dataset": dict(dataset or {}),
        "modelo": modelo, "archivo": nombre_archivo, "huella": huella, "n": int(n), "salida": out, "entradas": entradas, "resumen_entradas": pd.DataFrame(filas),
        "conteo_clases": cont.to_numpy(dtype=int), "prob_media": medias, "prevalencia": prev, "umbral": float(ens["umbral"]), "n_alertas": int(out["alerta_severa_umbral"].sum()),
        "p_severa_media": float(pser.mean()), "p_severa_mediana": float(np.median(pser)), "shap_global": imp, "n_ref_desarrollo": int(len(dev)),
        "interp": {"lote": {"que_muestra": f"{n} hogares: {int(out['alerta_severa_umbral'].sum())} superan el umbral operativo de IA severa ({ens['umbral']:.3f}); P(severa) media {pser.mean():.1%}.",
                            "como_leerlo": "Cada fila trae probabilidades por clase e IC de P(severa) entre los miembros del ensamble.",
                            "limitacion": "Resultados individuales no deben usarse para negar ayudas; son estimaciones del modelo."}},
    }


# ============================================================ figuras (Matplotlib, 600 DPI)
def _fig_probabilidades(sn: dict[str, Any]):
    with plt.rc_context(P.RC_REVISTA):
        fig, ax = plt.subplots(figsize=(P.ANCHO_DOBLE, 3.1))
        x = np.arange(4)
        media, lo, hi = sn["media"], sn["ic_inf"], sn["ic_sup"]
        cols = [P.OKABE_ITO[5] if k == sn["nivel"] else P.OKABE_ITO[0] for k in range(4)]
        ax.bar(x, media * 100, color=cols, width=0.55, yerr=np.vstack([(media - lo) * 100, (hi - media) * 100]), error_kw={"elinewidth": 0.9, "capsize": 3, "ecolor": "#222222"}, zorder=2)
        ax.scatter(x, sn["prevalencia"] * 100, marker="D", s=22, color="#000000", zorder=4, label="Prevalencia de referencia (Desarrollo, ponderada)")
        for k in range(4):
            ax.text(k, hi[k] * 100 + 1.2, f"{media[k] * 100:.1f} %", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x, [f"Nivel {k}\n{NOMBRE_NIVEL[k]}" for k in range(4)])
        ax.set_ylabel("Probabilidad (%)")
        ax.set_ylim(0, max(float(hi.max()), float(sn["prevalencia"].max())) * 100 * 1.25 + 2)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right", frameon=False)
        ax.set_title("Probabilidad por nivel FIES: media del ensamble e IC95 bootstrap (barras); nivel predicho en color bermellón", loc="left")
        return fig


def _fig_miembros(sn: dict[str, Any]):
    with plt.rc_context(P.RC_REVISTA):
        fig, ax = plt.subplots(figsize=(P.ANCHO_DOBLE, 3.0))
        pm = sn["por_miembro"]
        m = pm.shape[0]
        for i in range(m):
            es_final = i == m - 1
            ax.scatter(np.arange(4) + (i - (m - 1) / 2) * 0.08, pm[i] * 100, s=24 if es_final else 18, marker="D" if es_final else "o",
                       color=P.OKABE_ITO[5] if es_final else P.OKABE_ITO[i % 5], edgecolor="white", linewidth=0.4, zorder=3,
                       label="Modelo final" if es_final else f"CV externa, pliegue {i + 1}")
        ax.scatter(np.arange(4), sn["media"] * 100, color="#000000", marker="_", s=900, linewidths=1.4, zorder=2, label="Media del ensamble")
        ax.set_xticks(range(4), [f"Nivel {k}\n{NOMBRE_NIVEL[k]}" for k in range(4)])
        ax.set_ylabel("Probabilidad asignada (%)")
        ax.grid(axis="x", visible=False)
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
        ax.set_title("Probabilidad que asigna cada miembro del ensamble (variabilidad del modelo)", loc="left")
        return fig


def _serie_shap(sn: dict[str, Any]) -> pd.DataFrame:
    sh = sn["shap"]
    d = pd.DataFrame({"nombre": sh["nombres"], "shap": sh["valores"]})
    d["abs"] = d["shap"].abs()
    return d.sort_values("abs", ascending=False).reset_index(drop=True)


def _fig_shap(sn: dict[str, Any], top: int = 15):
    d = _serie_shap(sn).head(top).iloc[::-1]
    with plt.rc_context(P.RC_REVISTA):
        fig, ax = plt.subplots(figsize=(P.ANCHO_DOBLE, max(2.2, 0.24 * len(d) + 0.9)))
        ax.barh([E.etq(v) for v in d["nombre"]], d["shap"], color=[P.ROJO if v > 0 else P.AZUL for v in d["shap"]], height=0.65, zorder=2)
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set_xlabel(f"Contribución SHAP a la probabilidad del nivel {sn['shap']['clase']} (bermellón: la aumenta; azul: la reduce)")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"Explicación local (SHAP del modelo final): {len(d)} variables con mayor contribución", loc="left")
        return fig


def _fig_percentiles(sn: dict[str, Any]):
    c = sn["comparacion"].iloc[::-1]
    with plt.rc_context(P.RC_REVISTA):
        fig, ax = plt.subplots(figsize=(P.ANCHO_DOBLE, max(2.2, 0.24 * len(c) + 0.9)))
        ax.barh(c["variable"], c["percentil del hogar"] * 100, color=P.OKABE_ITO[0], height=0.62, zorder=2)
        ax.axvline(50, color="#333333", lw=0.8, ls="--")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentil del hogar en la distribución poblacional ponderada de Desarrollo (%) · línea punteada: mediana")
        ax.grid(axis="y", visible=False)
        ax.set_title("Posición del hogar en cada variable numérica", loc="left")
        return fig


def _fig_lote_clases(sn: dict[str, Any]):
    with plt.rc_context(P.RC_REVISTA):
        fig, ax = plt.subplots(figsize=(P.ANCHO_DOBLE, 3.0))
        x = np.arange(4)
        pred = sn["conteo_clases"] / max(sn["n"], 1)
        ax.bar(x - 0.19, pred * 100, width=0.36, color=P.OKABE_ITO[0], label="Lote: nivel predicho (argmax)", zorder=2)
        ax.bar(x + 0.19, sn["prevalencia"] * 100, width=0.36, color=P.OKABE_ITO[1], label="Prevalencia de referencia (Desarrollo, ponderada)", zorder=2)
        for k in range(4):
            ax.text(k - 0.19, pred[k] * 100 + 0.8, f"{pred[k] * 100:.1f}", ha="center", fontsize=7)
            ax.text(k + 0.19, sn["prevalencia"][k] * 100 + 0.8, f"{sn['prevalencia'][k] * 100:.1f}", ha="center", fontsize=7)
        ax.set_xticks(x, [f"Nivel {k}\n{NOMBRE_NIVEL[k]}" for k in range(4)])
        ax.set_ylabel("Hogares (%)")
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, loc="upper right")
        ax.set_title(f"Distribución de niveles FIES predichos en el lote (n = {sn['n']:,})", loc="left")
        return fig


def _hist_psevera(sn: dict[str, Any]) -> pd.DataFrame:
    p = sn["salida"]["p_clase_3"].to_numpy()
    cortes = np.linspace(0, max(float(p.max()), 1e-6), 11) if p.max() > 0 else np.linspace(0, 1, 11)
    cont, bordes = np.histogram(p, bins=cortes)
    return pd.DataFrame({"intervalo": [f"{a:.3f} a {b:.3f}" for a, b in zip(bordes[:-1], bordes[1:])], "desde": bordes[:-1], "hasta": bordes[1:], "hogares": cont.astype(int)})


def _fig_lote_psevera(sn: dict[str, Any]):
    h = _hist_psevera(sn)
    with plt.rc_context(P.RC_REVISTA):
        fig, ax = plt.subplots(figsize=(P.ANCHO_DOBLE, 3.0))
        anch = (h["hasta"] - h["desde"]).to_numpy()
        ax.bar(h["desde"], h["hogares"], width=anch, align="edge", color=P.OKABE_ITO[0], edgecolor="white", linewidth=0.5, zorder=2)
        ax.axvline(sn["umbral"], color=P.OKABE_ITO[5], lw=1.4, ls="--", label=f"Umbral operativo ({sn['umbral']:.3f})")
        ax.set_xlabel("P(inseguridad alimentaria severa)")
        ax.set_ylabel("Hogares")
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False)
        ax.set_title("Distribución de P(IA severa) en el lote", loc="left")
        return fig


def _tabla_percentiles_lote(sn: dict[str, Any]) -> pd.DataFrame:
    filas = []
    for k in C.CLASES:
        q = np.percentile(sn["salida"][f"p_clase_{k}"].to_numpy(), [5, 25, 50, 75, 95])
        filas.append({"Nivel FIES": _nivel_texto(k), "mínimo": float(sn["salida"][f"p_clase_{k}"].min()), "P5": q[0], "P25": q[1], "mediana": q[2], "P75": q[3], "P95": q[4],
                      "máximo": float(sn["salida"][f"p_clase_{k}"].max()), "media": float(sn["prob_media"][k])})
    return pd.DataFrame(filas)


def _fig_lote_prob(sn: dict[str, Any]):
    with plt.rc_context(P.RC_REVISTA):
        fig, ax = plt.subplots(figsize=(P.ANCHO_DOBLE, 3.0))
        datos = [sn["salida"][f"p_clase_{k}"].to_numpy() * 100 for k in C.CLASES]
        bp = ax.boxplot(datos, tick_labels=[f"Nivel {k}\n{NOMBRE_NIVEL[k]}" for k in C.CLASES], patch_artist=True, widths=0.5, showfliers=False,
                        medianprops={"color": "#000000"}, boxprops={"linewidth": 0.7}, whiskerprops={"linewidth": 0.7}, capprops={"linewidth": 0.7})
        for caja, col in zip(bp["boxes"], P.OKABE_ITO):
            caja.set_facecolor(col)
            caja.set_alpha(0.85)
        ax.set_ylabel("Probabilidad asignada (%)")
        ax.grid(axis="x", visible=False)
        ax.set_title("Distribución de las probabilidades asignadas por nivel FIES (caja: P25-P75; bigotes: P5-P95 aprox.)", loc="left")
        return fig


# ============================================================ bloques del documento
def _p(t: str) -> dict[str, Any]:
    return {"tipo": "p", "texto": t}


def _tabla(titulo: str, df: pd.DataFrame, pie: Any = None, max_filas: int | None = None) -> dict[str, Any]:
    return {"tipo": "tabla", "titulo": titulo, "df": df, "pie": pie, "max_filas": max_filas}


def _aux_ic(df: pd.DataFrame, v: str, lo: str, hi: str) -> pd.DataFrame:
    out = df.copy()
    out["semiancho sup. (aux. gráfico)"] = (out[hi] - out[v]).clip(lower=0)
    out["semiancho inf. (aux. gráfico)"] = (out[v] - out[lo]).clip(lower=0)
    return out


def _datos_prob(sn: dict[str, Any]) -> pd.DataFrame:
    d = pd.DataFrame({"Nivel FIES": [_nivel_texto(k) for k in C.CLASES], "Probabilidad (media del ensamble)": sn["media"], "IC95 inf": sn["ic_inf"], "IC95 sup": sn["ic_sup"],
                      "Mín. entre miembros": sn["min"], "Máx. entre miembros": sn["max"], "Prevalencia de referencia": sn["prevalencia"]})
    d["Razón probabilidad / prevalencia"] = d["Probabilidad (media del ensamble)"] / d["Prevalencia de referencia"].replace(0, np.nan)
    return d


def _datos_miembros(sn: dict[str, Any]) -> pd.DataFrame:
    m = sn["por_miembro"].shape[0]
    nombres = [f"CV externa, pliegue {i + 1}" for i in range(m - 1)] + ["Modelo final"]
    d = pd.DataFrame(sn["por_miembro"], columns=[f"P(nivel {k})" for k in C.CLASES])
    d.insert(0, "Miembro del ensamble", nombres)
    d["Suma"] = d[[f"P(nivel {k})" for k in C.CLASES]].sum(axis=1)
    return d


def _datos_shap(sn: dict[str, Any]) -> pd.DataFrame:
    sh = sn["shap"]
    d = _serie_shap(sn)
    base = pd.DataFrame({"Rango": np.arange(1, len(d) + 1), "Variable": [E.etq(v) for v in d["nombre"]],
                         "Código de la columna del modelo": d["nombre"], "Contribución SHAP": d["shap"], "|SHAP|": d["abs"],
                         "Efecto sobre la probabilidad": np.where(d["shap"] > 0, "la aumenta", np.where(d["shap"] < 0, "la reduce", "sin efecto"))})
    del sh
    return base


def _bloques_figura(nombre: str, fig, dir_fig: Path, titulo: str, pie: Any, datos: list[dict[str, Any]]) -> dict[str, Any]:
    ruta = P.guardar_mpl(fig, dir_fig / f"{nombre}.png")
    return {"tipo": "figura", "titulo": titulo, "ruta": ruta, "pie": pie, "ancho": "doble", "datos": datos}


def _d(titulo: str, df: pd.DataFrame, chart: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"titulo": titulo, "df": df, "chart": chart, "color": None}



# ============================================================ bloques de los análisis adicionales (individual)
def _vacio_err(sn: dict[str, Any], clave: str, texto: str) -> dict[str, Any]:
    e = (sn.get("extra_err") or {}).get(clave)
    return {"tipo": "vacio", "texto": texto + (f" ({e})" if e else "")}


def _legible(v: str, x: Any) -> str:
    return E.valor_etq(v, x) if isinstance(x, str) else f"{float(x):.4g}"


def _bloques_lime(sn: dict[str, Any], dir_fig: Path) -> list[dict[str, Any]]:
    L = sn.get("lime")
    m = sn["modelo"]
    if L is None:
        return [_vacio_err(sn, "lime", "No se pudo calcular LIME para esta inferencia")]
    t = L["tabla"]
    d_l = pd.DataFrame({"Rango": np.arange(1, len(t) + 1), "Variable": [E.etq(v) for v in t["variable"]], "Condición LIME": t["regla"], "Código": t["variable"],
                        "Peso LIME": t["peso"], "|peso|": t["abs"], "Efecto sobre la probabilidad": np.where(t["peso"] > 0, "la aumenta", np.where(t["peso"] < 0, "la reduce", "sin efecto"))})
    bl: list[dict[str, Any]] = [
        _p(f"LIME (Ribeiro et al., 2016) del modelo final ({m['codigo']}) para el nivel {L['clase']}: {L['n_muestras']:,} perturbaciones del hogar (semilla {L['semilla']}) generadas en el espacio de las variables originales "
           f"a partir de {L['n_fondo']:,} hogares de Desarrollo; cada perturbación pasa por el mismo preprocesador y modelo del sistema. APROXIMACIÓN LOCAL: un modelo lineal ponderado ajustado en torno a este hogar "
           f"(R² local {L['r2_local']:.2f}; probabilidad local {_pct(L['prob_local'])} frente a {_pct(L['prob_modelo'])} del modelo)."),
        _bloques_figura("fig_inf_lime", P.mpl_lime(t, L["clase"]), dir_fig, "Pesos LIME locales de las condiciones con mayor peso", sn["interp"].get("lime"), [
            _d("Pesos LIME de todas las variables (orden descendente por magnitud)", d_l, {"tipo": "barra_h", "titulo": "Peso LIME local: 12 condiciones principales", "cat": "Condición LIME",
                                                                                         "series": [{"col": "Peso LIME", "nombre": "Peso LIME"}], "max_filas": 12, "alto_cm": 11})]),
        _tabla("Tabla de pesos LIME por variable", d_l, {"que_muestra": "Peso local de cada variable (una condición por variable, discretizada por cuartiles de Desarrollo).", "como_leerlo": "Positivo: aumenta la probabilidad del nivel; negativo: la reduce.",
                                                       "limitacion": "Aproximación local con perturbaciones aleatorias; no es causal."}, max_filas=30)]
    c = sn.get("lime_shap")
    if c and c.get("disponible"):
        tb = c["tabla"].copy()
        tb["LIME normalizado"] = tb["LIME (peso)"] / max(tb["abs LIME"].max(), 1e-12)
        tb["SHAP normalizado"] = tb["SHAP agregado"] / max(tb["abs SHAP"].max(), 1e-12)
        tb["orden"] = np.maximum(tb["LIME normalizado"].abs(), tb["SHAP normalizado"].abs())
        tb = tb.sort_values("orden", ascending=False).reset_index(drop=True)
        d_c = pd.DataFrame({"Variable": [E.etq(v) for v in tb["variable"]], "Código": tb["variable"], "LIME (peso)": tb["LIME (peso)"], "SHAP agregado": tb["SHAP agregado"],
                            "LIME normalizado": tb["LIME normalizado"], "SHAP normalizado": tb["SHAP normalizado"], "¿Mismo signo?": tb["mismo signo"]})
        rho = c["spearman_abs"]
        res_c = pd.DataFrame([("Variables principales (top-%d) que coinciden entre LIME y SHAP" % c["k"], f"{c['coincidencia_top_k']} de {c['k']}"),
                              ("Variables con el mismo signo (efecto distinto de cero en ambos)", f"{c['signos_coinciden']} de {c['signos_total']}"),
                              ("Mismo signo entre las variables principales de cualquiera de los dos", f"{c['signos_top_coinciden']} de {c['signos_top_total']}"),
                              ("Correlación de rangos (Spearman) de |efecto|", "no calculable" if rho != rho else f"{rho:.2f}")], columns=["Indicador", "Valor"])
        bl.append(_p("Comparación con SHAP: SHAP se agrega por variable original (suma de las columnas de cada variable) y se compara con LIME por signo y orden de importancia; las magnitudes no son comparables (unidades distintas)."))
        bl.append(_bloques_figura("fig_inf_lime_shap", P.mpl_lime_shap(c["tabla"]), dir_fig, "LIME frente a SHAP por variable (efecto normalizado)", sn["interp"].get("lime_shap"), [
            _d("Efectos LIME y SHAP por variable (normalizados por el mayor |efecto| de cada método)", d_c, {"tipo": "barra_h", "titulo": "LIME frente a SHAP (efecto normalizado): 10 variables principales", "cat": "Variable",
                                                                                                         "series": [{"col": "LIME normalizado", "nombre": "LIME"}, {"col": "SHAP normalizado", "nombre": "SHAP agregado"}], "max_filas": 10, "alto_cm": 11})]))
        bl.append(_tabla("Concordancia entre LIME y SHAP", res_c, {"que_muestra": "Resumen de la coincidencia entre ambos métodos para este hogar.", "como_leerlo": "Más coincidencias = explicación más robusta al método.", "limitacion": "Con pocas variables, la correlación de rangos es inestable."}))
        bl.append(_tabla("Efectos LIME y SHAP por variable", d_c.round(4), max_filas=30))
    else:
        bl.append({"tipo": "vacio", "texto": "No se pudo comparar LIME con SHAP porque no hay explicación SHAP para esta inferencia."})
    return bl


def _bloques_validez(sn: dict[str, Any]) -> list[dict[str, Any]]:
    v = sn.get("validez")
    if v is None:
        return [_vacio_err(sn, "validez", "No se pudo evaluar la validez de las entradas")]
    ip = sn["interp"].get("validez")
    return [_p(ip["que_muestra"]), _tabla("Validez de las entradas frente al rango observado en Desarrollo (advertencia de extrapolación)", v, ip, max_filas=60)]


def _fila_curva(sn: dict[str, Any], d: pd.DataFrame, v: str) -> pd.DataFrame:
    out = pd.DataFrame({"Variable": E.etq(v), "Valor": [_legible(v, x) for x in d["valor"]]})
    for k in C.CLASES:
        out[f"P(nivel {k})"] = d[f"p{k}"].to_numpy()
    out["P(mod.+sev.)"] = d["p_mod_sev"].to_numpy()
    out["Valor actual del hogar"] = np.where(d["es_valor_actual"].to_numpy(), "sí", "")
    return out


def _bloques_sensibilidad(sn: dict[str, Any], dir_fig: Path) -> list[dict[str, Any]]:
    S = sn.get("sens")
    if S is None:
        return [_vacio_err(sn, "sens", "No se pudo calcular el análisis de sensibilidad")]
    cur, tor = S["curvas"], S["tornado"]
    datos: list[dict[str, Any]] = []
    largo = []
    for v in S["principales"]:
        d = cur[cur["variable"] == v]
        if S["tipos"][v] == "categorica":
            dd = d.sort_values("p_mod_sev", ascending=False)
            tabla = pd.DataFrame({"Categoría": [E.valor_etq(v, x) for x in dd["valor"]], "P(mod.+sev.)": dd["p_mod_sev"].to_numpy(), "Categoría actual del hogar": np.where(dd["es_valor_actual"], "sí", "")})
            datos.append(_d(f"{E.etq(v)}: P(moderada o severa) por categoría", tabla, {"tipo": "barra_h", "titulo": f"{E.etq(v)}: P(moderada o severa) por categoría", "cat": "Categoría",
                                                                                  "series": [{"col": "P(mod.+sev.)", "nombre": "P(mod.+sev.)"}], "max_filas": 15, "alto_cm": 10}))
        else:
            dd = d.sort_values("valor", key=lambda s_: s_.astype(float))
            tabla = pd.DataFrame({"Valor": dd["valor"].astype(float).to_numpy(), **{f"P(nivel {k})": dd[f"p{k}"].to_numpy() for k in C.CLASES}, "P(mod.+sev.)": dd["p_mod_sev"].to_numpy(),
                                  "Valor actual del hogar": np.where(dd["es_valor_actual"], "sí", "")})
            xs = dd["valor"].astype(float)
            datos.append(_d(f"{E.etq(v)}: probabilidad por nivel FIES", tabla, {"tipo": "dispersion", "titulo": E.etq(v), "x_titulo": E.etq(v, True), "y_titulo": "Probabilidad",
                                                                                "series": [{"x": "Valor", "col": f"P(nivel {k})", "nombre": f"Nivel {k} · {NOMBRE_NIVEL[k]}", "marcador": True} for k in C.CLASES],
                                                                                "xlim": (float(xs.min()), float(xs.max()) if xs.max() > xs.min() else float(xs.min()) + 1), "ylim": (0, 1), "alto_cm": 8.5, "ancho_cm": 15}))
        largo.append(_fila_curva(sn, d, v))
    largo_df = pd.concat(largo, ignore_index=True) if largo else pd.DataFrame()
    d_t = tor[["Variable", "valor bajo", "valor alto", "P(mod.+sev.) con valor bajo", "P(mod.+sev.) con valor alto", "efecto (pp)", "|efecto| (pp)"]]
    bl: list[dict[str, Any]] = [
        _p(f"Método: se varía UNA variable a la vez y el resto del hogar queda fijo; la probabilidad es la media del ensamble. Curvas (ICE local) para las {len(S['principales'])} variables con mayor contribución SHAP/LIME: valores plausibles "
           f"tomados de los cuantiles ponderados p5-p95 de Desarrollo (no inventados) y todas las categorías en las categóricas. Tornado: las {len(tor)} variables, de su percentil 10 a su percentil 90 de Desarrollo. "
           f"{S['n_variantes']} variantes evaluadas; probabilidad actual de IA moderada o severa {_pct(S['p_mod_sev_base'])}. Es una asociación predictiva del modelo, no un efecto causal."),
        _bloques_figura("fig_inf_sensibilidad", P.mpl_sensibilidad(cur, S["principales"], S["tipos"]), dir_fig, "Curvas de sensibilidad de las variables principales", sn["interp"].get("sens"), datos),
        _tabla("Probabilidades por valor explorado (todas las filas de las curvas)", largo_df, {"que_muestra": "Cada fila es una variante del hogar con una sola variable modificada.", "como_leerlo": "Compare con la fila marcada «sí» (valor actual).",
                                                                                           "limitacion": "El resto del hogar no se ajusta de forma coherente al cambio."}, max_filas=220),
        _bloques_figura("fig_inf_tornado", P.mpl_tornado(tor, S["p_mod_sev_base"]), dir_fig, "Tornado de sensibilidad (p10 → p90)", sn["interp"].get("tornado"), [
            _d("Efecto de mover cada variable entre p10 y p90 sobre P(moderada o severa)", d_t, {"tipo": "barra_h", "titulo": "Efecto sobre P(moderada o severa) (pp): 12 variables principales", "cat": "Variable",
                                                                                              "series": [{"col": "efecto (pp)", "nombre": "Efecto (pp)"}], "max_filas": 12, "alto_cm": 11})]),
        _tabla("Tornado: efecto de cada variable (orden descendente por |efecto|)", d_t, {"que_muestra": "P(moderada o severa) con el valor bajo y con el valor alto de cada variable y su diferencia en puntos porcentuales.",
                                                                                         "como_leerlo": "Positivo: el valor alto aumenta la probabilidad respecto del bajo.", "limitacion": "Una variable a la vez; categóricas: de la categoría de menor a la de mayor probabilidad."}, max_filas=40)]
    return bl


def _txt_cambios(e: dict[str, Any]) -> str:
    return "; ".join(f"{c['etiqueta']}: {c['valor_actual']:.4g} → {c['valor_nuevo']:.4g}" for c in e["cambios"])


def _bloques_contrafactual(sn: dict[str, Any], dir_fig: Path) -> list[dict[str, Any]]:
    cf = sn.get("contra")
    m = sn["modelo"]
    if cf is None:
        return [_vacio_err(sn, "contra", "No se pudo calcular el análisis contrafactual")]
    if not cf.get("aplica", False):
        return [{"tipo": "vacio", "texto": "El análisis contrafactual no aplica a este modelo. " + cf.get("motivo", "")}]
    esc = cf["escenarios"]
    aviso = (f"ASOCIACIÓN PREDICTIVA, NO EFECTO CAUSAL NI RECOMENDACIÓN DE POLÍTICA. Muestra qué entradas, al cambiar dentro de rangos plausibles, harían que ESTE modelo asigne menor probabilidad a este hogar; no demuestra que hacer ese cambio "
             f"reduzca la inseguridad alimentaria. {_resumen_metricas_texto(m)} La capacidad predictiva es modesta: las diferencias pueden estar dentro del error del modelo.")
    bl: list[dict[str, Any]] = [
        _p(aviso),
        _p(f"Objetivo: P(IA moderada o severa) < {_pct(cf['objetivo'])} (una reducción relativa de {_pct(AL.CF_REDUCCION_RELATIVA, 0)} respecto de la actual, {_pct(cf['p_mod_sev_base'])}); la prevalencia de referencia de Desarrollo es {_pct(cf['prevalencia_modsev_ref'])}. "
           f"Método: búsqueda voraz y determinista sobre las variables accionables ({', '.join(E.etq(v) for v in cf['accionables'])}), con hasta {cf['max_cambios']} cambios por escenario, valores entre el actual y el percentil 90 (o 10) de Desarrollo, "
           f"y hasta {AL.CF_N_ESCENARIOS} escenarios alternativos; {cf['n_evaluaciones']} evaluaciones del ensamble. Las variables derivadas (gasto, pobreza, proporción del gasto en alimentos) y los ítems FIES se mantienen fijos."),
    ]
    if not esc:
        bl.append({"tipo": "vacio", "texto": ("La probabilidad actual ya es menor que el objetivo." if cf.get("ya_cumple") else
                                              "No se encontró ningún cambio plausible en las variables accionables que reduzca la probabilidad según el modelo.")})
    else:
        t_esc = pd.DataFrame([{"Escenario": f"Escenario {e['n']}", "Cambios": _txt_cambios(e), "Distancia": e["distancia"], "P(mod.+sev.) nueva": e["p_mod_sev"], "Reducción (pp)": e["reduccion_pp"],
                               "Nivel predicho": f"{e['nivel']} · {NOMBRE_NIVEL[e['nivel']]}", "¿Alcanza el objetivo?": "Sí" if e["alcanza_objetivo"] else "No",
                               "Miembros del ensamble que reducen P": f"{e['miembros_que_reducen']} de {e['n_miembros']}"} for e in esc])
        t_cam = pd.DataFrame([{"Escenario": f"Escenario {e['n']}", "Variable": c["etiqueta"], "Código": c["variable"], "Unidad": c["unidad"], "Valor actual": c["valor_actual"], "Valor nuevo": c["valor_nuevo"],
                               "Cambio": c["valor_nuevo"] - c["valor_actual"], "Distancia": c["distancia"]} for e in esc for c in e["cambios"]])
        filas_p = [{"Situación": "Actual", **{f"P(nivel {k})": cf["p_base"][k] for k in C.CLASES}, "P(mod.+sev.)": cf["p_mod_sev_base"], "Objetivo": cf["objetivo"]}]
        filas_p += [{"Situación": f"Escenario {e['n']}", **{f"P(nivel {k})": e["p"][k] for k in C.CLASES}, "P(mod.+sev.)": e["p_mod_sev"], "Objetivo": cf["objetivo"]} for e in esc]
        t_p = pd.DataFrame(filas_p)
        bl.append(_tabla("Escenarios contrafactuales", t_esc, {"que_muestra": sn["interp"]["contra"]["que_muestra"], "como_leerlo": sn["interp"]["contra"]["como_leerlo"], "limitacion": sn["interp"]["contra"]["limitacion"]}))
        bl.append(_tabla("Cambios de cada escenario", t_cam, {"que_muestra": "Variable accionable, valor actual y valor propuesto en cada escenario.", "como_leerlo": "La distancia es el cambio relativo al rango p10-p90 de Desarrollo (binarias: 1).",
                                                              "limitacion": "El perfil modificado puede ser incoherente (p. ej. más ingreso con el mismo gasto y pobreza)."}))
        bl.append(_bloques_figura("fig_inf_contrafactual", P.mpl_contrafactual(cf), dir_fig, "Probabilidades actuales y de cada escenario contrafactual", sn["interp"]["contra"], [
            _d("Probabilidad por nivel FIES y P(moderada o severa): actual y escenarios", t_p, {"tipo": "barra", "titulo": "P(moderada o severa): actual, escenarios y objetivo", "cat": "Situación",
                                                                                           "series": [{"col": "P(mod.+sev.)", "nombre": "P(moderada o severa)"}, {"col": "Objetivo", "nombre": "Objetivo"}], "y_titulo": "Probabilidad (0-1)"})]))
        bl.append(_tabla("Probabilidades por nivel FIES: actual y escenarios", t_p, max_filas=10))
    if cf.get("omitidas"):
        bl.append(_tabla("Variables accionables sin margen de cambio en este hogar", pd.DataFrame(cf["omitidas"])))
    bl.append(_tabla("Variables accionables y no accionables: decisión y justificación [POR VALIDAR]", E.accionables_df(),
                     {"que_muestra": "Qué variables se pueden mover en el contrafactual y por qué se excluyen las demás.", "como_leerlo": "«Accionable» significa modificable en principio por un hogar o un programa, no que su modificación cause un cambio.",
                      "limitacion": "La lista es una propuesta técnica y su accionabilidad está [POR VALIDAR] con profesionales de nutrición y política social."}, max_filas=40))
    return bl


def _bloques_otros_modelos(sn: dict[str, Any], dir_fig: Path) -> list[dict[str, Any]]:
    t = sn.get("otros")
    if t is None or len(t) == 0:
        return [_vacio_err(sn, "otros", "No se pudo comparar con los demás modelos de la corrida")]
    ag = AL.acuerdo_modelos(t)
    ver = t.drop(columns=["ident", "codigo"])
    chart_df = t[["Modelo"] + [f"P(nivel {k})" for k in C.CLASES]].copy()
    chart_df["Modelo"] = t["codigo"] + np.where(t["Seleccionado"] != "", " (seleccionado)", "")
    return [_p(f"El mismo hogar se predijo con los {ag['n']} modelos de la corrida y el escenario «{sn['modelo']['escenario']}» (media del ensamble de cada uno). "
               f"{ag['n_modal']} de {ag['n']} modelos coinciden en el nivel {ag['nivel_modal']}; P(moderada o severa) entre {_pct(ag['p_min'])} y {_pct(ag['p_max'])}."),
            _bloques_figura("fig_inf_otros_modelos", P.mpl_otros_modelos(t), dir_fig, "El mismo hogar con todos los modelos de la corrida", sn["interp"].get("otros"), [
                _d("Probabilidad por nivel FIES según cada modelo", chart_df, {"tipo": "barra", "titulo": "Probabilidad por modelo", "cat": "Modelo",
                                                                              "series": [{"col": f"P(nivel {k})", "nombre": f"Nivel {k} · {NOMBRE_NIVEL[k]}"} for k in C.CLASES], "y_titulo": "Probabilidad (0-1)"})]),
            _tabla("Predicción del mismo hogar con cada modelo", ver, sn["interp"].get("otros"))]


def _bloques_lote_sens(sn: dict[str, Any], dir_fig: Path) -> list[dict[str, Any]]:
    S = sn.get("sens_lote")
    if S is None:
        return [_vacio_err(sn, "sens_lote", "No se pudo calcular la sensibilidad agregada del lote")]
    t = S["tabla"]
    tor = t.rename(columns={"valor bajo (p10 / 0)": "valor bajo", "valor alto (p90 / 1)": "valor alto", "efecto medio (pp)": "efecto (pp)"})
    tor["valor bajo"] = tor["valor bajo"].map(lambda x: x if isinstance(x, str) else f"{float(x):.4g}")
    tor["valor alto"] = tor["valor alto"].map(lambda x: x if isinstance(x, str) else f"{float(x):.4g}")
    d_t = t[["Variable", "valor bajo (p10 / 0)", "valor alto (p90 / 1)", "P(mod.+sev.) media con valor bajo", "P(mod.+sev.) media con valor alto", "efecto medio (pp)", "|efecto medio| (pp)"]].copy()
    for c in ("valor bajo (p10 / 0)", "valor alto (p90 / 1)"):
        d_t[c] = d_t[c].map(lambda x: x if isinstance(x, str) else f"{float(x):.4g}")
    ip = {"que_muestra": f"Sobre una muestra reproducible de {S['n_muestra']:,} hogares del lote (P(moderada o severa) media actual {_pct(S['p_mod_sev_media'])}), la variable cuyo cambio entre p10 y p90 de Desarrollo más mueve esa probabilidad media es «{t.iloc[0]['Variable']}» ({t.iloc[0]['efecto medio (pp)']:+.1f} pp).",
          "como_leerlo": "Cada barra es el cambio medio de P(moderada o severa) al llevar a TODOS los hogares de la muestra de su valor bajo (p10; 0; categoría de menor P) al alto (p90; 1; categoría de mayor P).",
          "limitacion": "Una variable a la vez, con el resto de cada hogar fijo; es una asociación predictiva del modelo, no un efecto causal ni una recomendación de política."}
    return [_p(ip["que_muestra"]),
            _bloques_figura("fig_lote_tornado", P.mpl_tornado(tor.assign(**{"Variable": tor["Variable"]}), S["p_mod_sev_media"]), dir_fig, "Sensibilidad agregada del lote (efecto medio, p10 → p90)", ip, [
                _d("Efecto medio de mover cada variable entre p10 y p90", d_t, {"tipo": "barra_h", "titulo": "Efecto medio sobre P(moderada o severa) (pp): 12 variables principales", "cat": "Variable",
                                                                            "series": [{"col": "efecto medio (pp)", "nombre": "Efecto medio (pp)"}], "max_filas": 12, "alto_cm": 11})]),
            _tabla("Sensibilidad agregada del lote por variable", d_t, ip, max_filas=40)]


def _bloques_lote_validez(sn: dict[str, Any]) -> list[dict[str, Any]]:
    v = sn.get("validez_lote")
    if v is None:
        return [_vacio_err(sn, "validez_lote", "No se pudo evaluar la validez de las entradas del lote")]
    n_fuera = int((v["Hogares fuera de rango"] > 0).sum())
    ip = {"que_muestra": (f"{n_fuera} de {len(v)} variables tienen al menos un hogar fuera del rango observado en Desarrollo." if n_fuera else "Ningún hogar del lote cae fuera del rango observado en Desarrollo en ninguna variable."),
          "como_leerlo": "Un hogar fuera de rango obliga al modelo a extrapolar: su predicción es menos confiable.", "limitacion": "Estar dentro del rango de cada variable por separado no garantiza que la combinación de valores sea frecuente."}
    return [_p(("ADVERTENCIA DE EXTRAPOLACIÓN. " if n_fuera else "") + ip["que_muestra"]), _tabla("Hogares del lote fuera del rango observado en Desarrollo, por variable", v, ip, max_filas=40)]


def _seccion_modelo(sn: dict[str, Any]) -> list[dict[str, Any]]:
    m = sn["modelo"]
    ident = pd.DataFrame([
        ("Corrida", m["corrida"]), ("Identificador del modelo", m["id"]), ("Algoritmo", f"{m['codigo']} · {m['algoritmo']}"),
        ("Escenario de variables", {"socio": "socio · variables socioeconómicas (predicción)", "fies": "fies · ítems FIES (reconstrucción)", "hibrido": "hibrido · socioeconómicas + ítems FIES (reconstrucción)"}.get(m["escenario"], m["escenario"])),
        ("Variables de entrada", f"{len(m['features_num'])} numéricas/binarias y {len(m['features_cat'])} categóricas"),
        ("Ensamble", f"{m['n_miembros']} modelos: {m['n_miembros'] - 1} de la validación cruzada externa + el modelo final"),
        ("Validación cruzada anidada", f"{m['n_outer']} pliegues externos × {m['n_inner']} internos; {m['n_trials']} ensayos de optimización por pliegue"),
        ("Umbral operativo de IA severa (Validación)", f"{m['umbral']:.4f}"),
        ("Semilla", m["semilla"]), ("Fecha de evaluación de la Prueba", m["fecha_corrida"] or "no evaluada"),
        ("Tamaños de partición (hogares)", ", ".join(f"{k} {v:,}" for k, v in m["tamanos_particion"].items()) or "no disponible"),
    ], columns=["Elemento", "Valor"])
    b: list[dict[str, Any]] = [_tabla("Modelo y corrida seleccionados", ident, {"que_muestra": "Identificación del modelo que produjo esta inferencia y de la corrida de entrenamiento de donde proviene.",
                                                                           "como_leerlo": "El ensamble promedia las probabilidades de todos sus miembros.", "limitacion": ""})]
    hp = pd.DataFrame([(k, str(v)) for k, v in m["hiperparametros"].items()], columns=["Hiperparámetro", "Valor"])
    if len(hp):
        b.append(_tabla("Hiperparámetros del modelo final (seleccionados por la validación cruzada interna)", hp))
    tm = _tabla_metricas(m)
    if len(tm):
        b.append(_p(_resumen_metricas_texto(m)))
        b.append(_tabla("Métricas del modelo (ponderadas por factor de expansión)", tm,
                        {"que_muestra": "Desempeño del modelo en la Prueba independiente (valor e IC95 bootstrap), en Validación y en la validación cruzada anidada (media e IC95 entre pliegues).",
                         "como_leerlo": "Brier y ECE: menor es mejor; las demás métricas: mayor es mejor. El azar en AUC es 0.5.", "limitacion": "Son métricas del modelo sobre otros hogares, no de esta inferencia."}))
    else:
        b.append({"tipo": "vacio", "texto": "La corrida no contiene métricas de desempeño para este modelo."})
    return b


def _seccion_metodologia(sn: dict[str, Any]) -> list[dict[str, Any]]:
    items = [
        f"Ensamble: la probabilidad por nivel FIES (0-3) es el promedio de {sn['modelo']['n_miembros']} modelos (los {sn['modelo']['n_miembros'] - 1} entrenados en los pliegues externos de la validación cruzada anidada, cada uno con sus propios hiperparámetros, más el modelo final entrenado con todo Desarrollo).",
        f"Intervalo: bootstrap percentil 95 % (B = {N_BOOT_INDIVIDUAL if sn['tipo'] == 'individual' else N_BOOT_LOTE}, semilla {C.SEED}) sobre los miembros del ensamble. Mide la variabilidad del modelo, no la incertidumbre de muestreo de la encuesta.",
        "Nivel predicho: el de mayor probabilidad (argmax). Umbral operativo: la clase severa se señala además cuando P(IA severa) supera el umbral fijado en Validación; el umbral no altera las probabilidades.",
        "Percentil poblacional de P(IA severa): fracción ponderada (factor de expansión) de hogares de Validación con probabilidad menor o igual a la de este hogar, según las predicciones del mismo modelo.",
        "Prevalencia de referencia: distribución ponderada de los niveles FIES en la muestra de Desarrollo de la ENAHO 2025 (no es la población peruana ni una estimación oficial).",
        "Niveles FIES: derivados del puntaje crudo de 8 ítems con cortes leve >= 1, moderada >= 4, severa >= 7 [POR VALIDAR]; no es la clasificación Rasch oficial.",
    ]
    if sn["tipo"] == "individual":
        items.append(f"LIME: aproximación local (modelo lineal ponderado sobre {AL.LIME_N_MUESTRAS:,} perturbaciones del hogar, semilla {C.SEED}, fondo de hasta {AL.LIME_MAX_FONDO:,} hogares de Desarrollo, discretización por cuartiles) del modelo final con el preprocesador del sistema; se compara con SHAP por signo y orden.")
        items.append(f"Sensibilidad: una variable a la vez, rejilla de cuantiles ponderados p5-p95 de Desarrollo ({AL.SENS_N_PUNTOS} puntos; todas las categorías) para las {AL.SENS_N_VARIABLES} variables principales y tornado p10-p90 para todas; probabilidades = media del ensamble.")
        items.append(f"Contrafactual: búsqueda voraz determinista sobre variables accionables ([POR VALIDAR], ver Anexo), hasta {AL.CF_MAX_CAMBIOS} cambios y {AL.CF_N_ESCENARIOS} escenarios, objetivo de reducción relativa del {AL.CF_REDUCCION_RELATIVA:.0%} de P(moderada o severa), valores acotados por p10/p90 de Desarrollo. Sin aleatoriedad (semilla {C.SEED} solo en muestreos auxiliares).")
        items.append("Comparación con los demás modelos: el mismo hogar se predice con el ensamble de cada modelo de la corrida y el escenario; advertencia de extrapolación según percentiles 1/99 y rango observado de Desarrollo.")
        items.append("Explicación local: valores SHAP (Lundberg y Lee, 2017) del modelo final para el nivel predicho, con un fondo de hasta 100 hogares de Desarrollo (semilla 42). Suman el desvío de la salida del modelo respecto de su valor base; no son efectos causales.")
        items.append("Comparación con la población: percentil ponderado del valor de cada variable numérica del hogar en Desarrollo.")
    return [{"tipo": "lista", "items": items}]


def _seccion_limitaciones(sn: dict[str, Any]) -> list[dict[str, Any]]:
    items = _advertencias(sn["modelo"])
    items.append("El reporte describe una sola inferencia y no sustituye la validación externa del modelo ni el juicio de profesionales.")
    if sn["tipo"] == "individual":
        items.append("LIME es una aproximación local y estocástica (semilla fija); con un R² local bajo explica poco del comportamiento del modelo. La sensibilidad y el contrafactual describen cómo responde el MODELO a cambios en sus entradas: son asociaciones predictivas, NO efectos causales ni una recomendación de política.")
        items.append("La lista de variables accionables es una propuesta técnica [POR VALIDAR]; el contrafactual mueve una o pocas variables con el resto del hogar fijo (incluidas las derivadas del ingreso y del gasto), por lo que el perfil resultante puede ser incoherente; la búsqueda voraz no garantiza el cambio mínimo global.")
        items.append("La capacidad predictiva del modelo es modesta (ver métricas): una diferencia de probabilidad entre escenarios puede quedar dentro del error del modelo.")
    else:
        items.append("Por lote no se calculan LIME, contrafactual ni la comparación hogar a hogar con los demás modelos (son análisis de un hogar y su costo crece con el tamaño del lote): use la predicción individual. La sensibilidad agregada usa una submuestra reproducible de hasta 500 hogares.")
    if sn["modelo"].get("n_prueba") is None:
        items.append("La Prueba independiente (15 %) no fue evaluada para este modelo: sus métricas de Validación y validación cruzada son preliminares.")
    items.append("La partición agrupa por hogar_uid; hogares del mismo conglomerado pueden aparecer en Desarrollo y Prueba, lo que puede inflar ligeramente el desempeño frente a una validación por conglomerado.")
    return [{"tipo": "lista", "items": items}]


def _seccion_traza(sn: dict[str, Any]) -> list[dict[str, Any]]:
    m = sn["modelo"]
    filas = [("Identificador de la inferencia", sn["id"]), ("Fecha y hora de la inferencia", sn["fecha"]), ("Usuario", sn["usuario"] or "sin sesión"),
             ("Huella SHA-256 de las entradas" if sn["tipo"] == "individual" else "Huella SHA-256 del archivo CSV del lote", sn["huella"]),
             ("Corrida / modelo", f"{m['corrida']} / {m['id']}"), ("SHA-256 del modelo final", m["sha256_modelo"]), ("Hash de los datos de entrenamiento (config.json)", m["hash_datos"] or "no disponible"),
             ("Huella de la partición", m["huella_particion"] or "no disponible"), ("Semilla (modelos y bootstrap)", f"{m['semilla']} / {C.SEED}"),
             ("Dataset activo", f"{sn['dataset'].get('nombre', 'no informado')} ({sn['dataset'].get('n', 'n/d')} hogares)" if sn["dataset"] else "no informado"),
             ("Versión del generador de reportes", VERSION_REPORTE), ("Generado el", datetime.now().isoformat(timespec="seconds"))]
    if sn["tipo"] == "individual":
        filas += [("Parámetros LIME", f"{AL.LIME_N_MUESTRAS:,} perturbaciones, semilla {AL.SEMILLA}, fondo ≤ {AL.LIME_MAX_FONDO:,} hogares de Desarrollo"),
                  ("Parámetros de sensibilidad", f"{AL.SENS_N_VARIABLES} variables principales, {AL.SENS_N_PUNTOS} cuantiles p5-p95 + valor actual; tornado p10-p90"),
                  ("Parámetros del contrafactual", f"≤ {AL.CF_MAX_CAMBIOS} cambios, ≤ {AL.CF_N_ESCENARIOS} escenarios, reducción relativa objetivo {AL.CF_REDUCCION_RELATIVA:.0%}, {AL.CF_N_PUNTOS} puntos por variable (determinista)"),
                  ("Tiempo de cálculo de los análisis (s)", ", ".join(f"{k} {v}" for k, v in (sn.get("tiempos") or {}).items()) or "no registrado")]
        if sn.get("extra_err"):
            filas.append(("Análisis que fallaron", "; ".join(f"{k}: {v}" for k, v in sn["extra_err"].items())))
    return [_tabla("Trazabilidad de la inferencia", pd.DataFrame(filas, columns=["Elemento", "Valor"]),
                   {"que_muestra": "Identificadores y huellas que permiten verificar qué entradas, modelo y datos produjeron este resultado.", "como_leerlo": "Si el hash de las entradas o del modelo cambia, el resultado no es reproducible a partir de este reporte.", "limitacion": ""}),
            _tabla("Versiones del entorno de cálculo", _versiones())]


def _seccion_anexo(sn: dict[str, Any]) -> list[dict[str, Any]]:
    m = sn["modelo"]
    d = E.diccionario_df(m["features_num"] + m["features_cat"])
    return [_p("Variables usadas por el modelo (nombre legible, código original, unidad, tipo y descripción). Las marcadas [POR VALIDAR] requieren verificación de su definición."),
            _tabla("Diccionario de variables usadas por el modelo", d, max_filas=80)]


def _metodologia_excel(sn: dict[str, Any]) -> list[tuple[str, list[str]]]:
    return [("Método", _seccion_metodologia(sn)[0]["items"]), ("Limitaciones y advertencias", _seccion_limitaciones(sn)[0]["items"])]


def _doc_comun(sn: dict[str, Any], titulo: str) -> dict[str, Any]:
    m = sn["modelo"]
    portada = [("Tipo de reporte", titulo), ("Identificador", sn["id"]), ("Fecha y hora", sn["fecha"]), ("Usuario", sn["usuario"] or "sin sesión"),
               ("Corrida de modelos", m["corrida"]), ("Modelo", f"{m['codigo']} · {m['algoritmo']} ({m['id']})"), ("Escenario", m["escenario"]),
               ("Huella SHA-256", sn["huella"]), ("Semilla", m["semilla"])]
    return {"titulo": titulo, "subtitulo": "FoodSecurity AI · Predicción de inseguridad alimentaria y vulnerabilidad nutricional de los hogares mediante inteligencia artificial explicable", "encabezado": f"FoodSecurity AI · {titulo} · {sn['id']}",
            "portada": portada, "resumen": [], "secciones": []}


# ------------------------------------------------------------ documento individual
def _seccion_perfil_individual(sn: dict[str, Any]) -> list[dict[str, Any]]:
    ctx = sn.get("perfil_ctx") or {}
    bl: list[dict[str, Any]] = [_p(PV.ALCANCE)]
    if sn.get("perfil_error"):
        bl.append({"tipo": "vacio", "texto": f"No se pudo calcular la prevalencia de referencia del perfil ({sn['perfil_error']}); solo se muestra la pertenencia a los grupos."})
    for t in PV.texto_contexto_hogar(ctx):
        bl.append(_p(t))
    if ctx.get("n_referencia"):
        bl.append(_p(f"Referencia del perfil: {ctx['referencia']} (n = {ctx['n_referencia']:,} hogares)."))
    bl.append(_tabla("Pertenencia del hogar a los grupos del perfil y prevalencia observada de cada grupo", PV.tabla_contexto_hogar(ctx), {
        "que_muestra": "Grupos del perfil (menor de 5 años, adulto mayor, pobreza monetaria) a los que pertenece el hogar evaluado y la prevalencia observada de inseguridad alimentaria moderada o severa en esos grupos.",
        "como_leerlo": "La prevalencia es de un grupo de hogares del conjunto analítico (ponderada, IC95 por linealización con conglomerados, sin estratos); no es la probabilidad de este hogar.",
        "limitacion": "Contexto descriptivo, no una predicción; el perfil no es un índice ponderado ni validado y no incluye desenlaces nutricionales directos."}))
    return bl


def _seccion_perfil_lote(sn: dict[str, Any]) -> list[dict[str, Any]]:
    pl = sn.get("perfil_lote")
    err = (sn.get("extra_err") or {}).get("perfil")
    if pl is None:
        return [{"tipo": "vacio", "texto": "El CSV no trae las columnas del perfil (n_menores5, n_adultos_mayores, pobreza_monetaria); no se calculó la distribución por recuento."
                 + (f" ({err})" if err else "")}]
    exc = f" {pl['n_sin_recuento']:,} hogares con dato faltante en alguna característica no tienen recuento." if pl["n_sin_recuento"] else ""
    return [_p(PV.ALCANCE),
            _p(f"Distribución de los {pl['n']:,} hogares del lote según las características del perfil y el recuento simple (0 a 3).{exc} La prevalencia de la columna de referencia es la observada en el conjunto de referencia (contexto, no una predicción de los hogares del lote)."),
            _tabla("Hogares del lote con cada característica del perfil", pl["por_caracteristica"], {
                "que_muestra": "Cantidad y porcentaje de hogares del lote que cumplen cada característica.", "como_leerlo": "Los porcentajes se calculan sobre todos los hogares del lote.",
                "limitacion": "Un lote no aleatorio no representa a la población."}),
            _tabla("Hogares del lote por recuento de características y prevalencia observada de referencia", pl["por_recuento"], {
                "que_muestra": "Distribución del lote por recuento (0 a 3) junto con la prevalencia observada de inseguridad alimentaria moderada o severa en los hogares del conjunto de referencia con ese recuento.",
                "como_leerlo": "Compare la composición del lote con la de referencia; la prevalencia de referencia no es la del lote.",
                "limitacion": "Descriptivo y asociativo; el perfil no es un índice ponderado ni validado."})]


def construir_doc_individual(sn: dict[str, Any], dir_fig: Path) -> dict[str, Any]:
    m = sn["modelo"]
    doc = _doc_comun(sn, "Reporte de inferencia individual")
    nivel, p3 = sn["nivel"], sn["p_severa"]
    prev3 = float(sn["prevalencia"][C.CLASE_SEVERA])
    razon = p3 / prev3 if prev3 > 0 else float("nan")
    doc["resumen"] = [
        f"Este reporte documenta una inferencia individual realizada el {sn['fecha'].replace('T', ' ')} con el modelo {m['codigo']} ({m['algoritmo']}), escenario «{m['escenario']}», de la corrida «{m['corrida']}». "
        f"Se ingresaron {len(sn['entradas'])} variables del hogar (sección 1).",
        f"Resultado: el ensamble asigna la mayor probabilidad al nivel {nivel} ({NOMBRE_NIVEL[nivel].lower()}, {_pct(sn['media'][nivel])}). La probabilidad de inseguridad alimentaria severa es {_pct(p3)} "
        f"(IC95 del ensamble {_pct(sn['p_severa_ic'][0])} a {_pct(sn['p_severa_ic'][1])}), que {'supera' if sn['supera'] else 'no supera'} el umbral operativo de {m['umbral']:.3f}. "
        f"Su probabilidad de IA severa es mayor que la de {_pct(sn['percentil'], 0)} de los hogares de Validación. Frente a la prevalencia de referencia de IA severa en Desarrollo ({_pct(prev3)}), la probabilidad estimada es "
        + (f"{razon:.2f} veces esa prevalencia." if not np.isnan(razon) else "no comparable."),
        _resumen_metricas_texto(m),
        _advertencias(m)[0],
    ]
    for k in ("lime_shap", "tornado", "contra", "otros", "validez"):
        if sn["interp"].get(k):
            doc["resumen"].append({"lime_shap": "LIME frente a SHAP: ", "tornado": "Sensibilidad: ", "contra": "Contrafactual (asociación predictiva, no causal): ", "otros": "Otros modelos: ", "validez": "Validez de las entradas: "}[k] + sn["interp"][k]["que_muestra"])
    pctx = sn.get("perfil_ctx") or {}
    if pctx.get("disponible"):
        doc["resumen"].append("Perfil de vulnerabilidad (contexto descriptivo, no una predicción): " + " ".join(PV.texto_contexto_hogar(pctx)[:2]))
    doc["resumen"].append("Este reporte no es un diagnóstico individual ni sustituye la medición con la escala FIES; los umbrales de los niveles FIES están [POR VALIDAR].")
    prob = _datos_prob(sn)
    prob_x = _aux_ic(prob, "Probabilidad (media del ensamble)", "IC95 inf", "IC95 sup")
    sec: list[dict[str, Any]] = []
    # 1. datos de entrada
    sec.append({"titulo": "1. Datos de entrada", "hoja": "1_Entradas", "bloques": [
        _p("Valores ingresados o seleccionados para el hogar, con el nombre legible de cada variable, su unidad y el código original (el que usa el modelo)."),
        _tabla("Datos ingresados del hogar", sn["entradas"], {"que_muestra": "Los valores exactos con que se calculó la inferencia, congelados al pulsar «Predecir».", "como_leerlo": "La columna «Original» es el valor que recibió el modelo; el rango corresponde a la muestra de Desarrollo.", "limitacion": "Los valores no se almacenan en la base de datos: solo su huella SHA-256."}, max_filas=60)]})
    # 2. modelo
    sec.append({"titulo": "2. Modelo y corrida", "hoja": "2_Modelo", "bloques": _seccion_modelo(sn)})
    # 3. resultado
    res_t = pd.DataFrame({"Nivel FIES": [_nivel_texto(k) for k in C.CLASES], "Probabilidad": [_pct(v, 2) for v in sn["media"]],
                          "IC95 del ensamble": [f"{_pct(a, 2)} a {_pct(b, 2)}" for a, b in zip(sn["ic_inf"], sn["ic_sup"])],
                          "Mín.-máx. entre miembros": [f"{_pct(a, 2)} a {_pct(b, 2)}" for a, b in zip(sn["min"], sn["max"])],
                          "Prevalencia de referencia": [_pct(v, 2) for v in sn["prevalencia"]]})
    clave = pd.DataFrame([("Nivel predicho (argmax)", f"{nivel} · {NOMBRE_NIVEL[nivel]}"), ("P(IA severa)", f"{_pct(p3, 2)} (IC95 {_pct(sn['p_severa_ic'][0], 2)} a {_pct(sn['p_severa_ic'][1], 2)})"),
                          ("Umbral operativo de IA severa", f"{m['umbral']:.4f}"), ("¿Supera el umbral?", "Sí" if sn["supera"] else "No"),
                          ("Percentil poblacional de P(IA severa)", _pct(sn["percentil"], 1)), ("Suma de las probabilidades por nivel", f"{float(np.sum(sn['media'])):.4f}")], columns=["Indicador", "Valor"])
    fig1 = _bloques_figura("fig_inf_probabilidades", _fig_probabilidades(sn), dir_fig, "Probabilidad por nivel FIES con IC95 y prevalencia de referencia", sn["interp"]["prediccion"], [
        _d("Datos exactos de la figura", prob_x, {"tipo": "barra", "titulo": "Probabilidad por nivel FIES (media e IC95)", "cat": "Nivel FIES",
                                                  "series": [{"col": "Probabilidad (media del ensamble)", "nombre": "Probabilidad", "err_mas": "semiancho sup. (aux. gráfico)", "err_menos": "semiancho inf. (aux. gráfico)"},
                                                             {"col": "Prevalencia de referencia", "nombre": "Prevalencia de referencia"}], "y_titulo": "Probabilidad (0-1)"})])
    fig2 = _bloques_figura("fig_inf_miembros", _fig_miembros(sn), dir_fig, "Probabilidad asignada por cada miembro del ensamble",
                           {"que_muestra": "Cada punto es la probabilidad que un modelo del ensamble asigna a un nivel; la marca negra horizontal es el promedio.", "como_leerlo": "Puntos muy dispersos indican desacuerdo entre los modelos y una estimación menos estable.",
                            "limitacion": "Los miembros comparten datos de entrenamiento: su acuerdo no garantiza que la estimación sea correcta."}, [_d("Probabilidad por miembro del ensamble", _datos_miembros(sn))])
    sec.append({"titulo": "3. Resultado de la inferencia", "hoja": "3_Resultado", "bloques": [
        _tabla("Resultado principal", clave), _tabla("Probabilidades por nivel FIES (media del ensamble, IC95 y prevalencia de referencia)", res_t,
                                                      {"que_muestra": "Probabilidad estimada para cada nivel de inseguridad alimentaria y su comparación con la prevalencia ponderada de Desarrollo.",
                                                       "como_leerlo": "Las cuatro probabilidades suman 1. El IC95 refleja desacuerdo entre los modelos del ensamble.", "limitacion": "La prevalencia de referencia es de la muestra de Desarrollo, no de la población peruana."}),
        fig1, fig2]})
    # 4. explicabilidad
    bl4: list[dict[str, Any]] = []
    if sn["shap"]:
        d_sh = _datos_shap(sn)
        bl4.append(_p(f"Explicación local del modelo final ({m['codigo']}) para el nivel {sn['shap']['clase']}. Valor base del modelo para ese nivel: {sn['shap']['valor_base']:.4f}."))
        bl4.append(_bloques_figura("fig_inf_shap_local", _fig_shap(sn), dir_fig, "Contribuciones SHAP locales de las variables principales", sn["interp"]["shap"], [
            _d("Contribuciones SHAP de todas las variables (orden descendente por magnitud)", d_sh, {"tipo": "barra_h", "titulo": "Contribución SHAP local: 15 variables principales", "cat": "Variable",
                                                                                                       "series": [{"col": "Contribución SHAP", "nombre": "Contribución SHAP"}], "max_filas": 15, "alto_cm": 11})]))
        bl4.append(_tabla("Ranking de variables por contribución local (|SHAP|)", d_sh, {"que_muestra": "Todas las columnas del modelo ordenadas por la magnitud de su contribución en este hogar.",
                                                                                         "como_leerlo": "Las variables categóricas aparecen como una columna por categoría.", "limitacion": "Describe al modelo, no causas."}, max_filas=25))
    else:
        bl4.append({"tipo": "vacio", "texto": f"No se pudo calcular la explicación SHAP de esta inferencia ({sn['shap_error']})."})
    bl4.append({"tipo": "h2", "texto": "Explicación local con LIME y comparación con SHAP"})
    bl4 += _bloques_lime(sn, dir_fig)
    sec.append({"titulo": "4. Explicabilidad", "hoja": "4_Explicabilidad", "bloques": bl4})
    # 5. comparación con la población
    bl5: list[dict[str, Any]] = [_p(f"Prevalencia de referencia (Desarrollo, n = {sn['n_ref_desarrollo']:,} hogares, ponderada): " + "; ".join(f"nivel {k} {_pct(sn['prevalencia'][k])}" for k in C.CLASES) + ".")]
    if len(sn["comparacion"]):
        c = sn["comparacion"]
        bl5.append(_bloques_figura("fig_inf_percentiles", _fig_percentiles(sn), dir_fig, "Percentil del hogar en cada variable numérica", sn["interp"]["comparacion"],
                                   [_d("Valor del hogar, mediana y percentil poblacional por variable", c.assign(**{"percentil del hogar (%)": 100 * c["percentil del hogar"]}).drop(columns=["percentil del hogar"]),
                                       {"tipo": "barra_h", "titulo": "Percentil del hogar (%)", "cat": "variable", "series": [{"col": "percentil del hogar (%)", "nombre": "Percentil (%)"}], "alto_cm": 11})]))
        bl5.append(_tabla("Comparación con la distribución poblacional (Desarrollo, ponderada)", c.rename(columns={"variable": "Variable", "código": "Código", "valor del hogar": "Valor del hogar", "mediana poblacional": "Mediana poblacional", "percentil del hogar": "Percentil del hogar"}).round(4), max_filas=40))
    bl5.append({"tipo": "h2", "texto": "Validez de las entradas y advertencia de extrapolación"})
    bl5 += _bloques_validez(sn)
    sec.append({"titulo": "5. Comparación con la población de referencia y validez de las entradas", "hoja": "5_Poblacion", "bloques": bl5})
    sec.append({"titulo": "5b. Perfil de vulnerabilidad del hogar", "hoja": "5b_Perfil", "bloques": _seccion_perfil_individual(sn)})
    # 6. interpretación en lenguaje claro
    ic = (_pct(sn["p_severa_ic"][0]), _pct(sn["p_severa_ic"][1]))
    clar = [
        f"Con los datos ingresados, el modelo considera que el nivel más probable es «{NOMBRE_NIVEL[nivel].lower()}», con una probabilidad de {_pct(sn['media'][nivel])}. Esto significa que, entre los cuatro niveles posibles, es el que el modelo prefiere, pero no que el hogar tenga ese nivel con certeza: las demás probabilidades suman {_pct(1 - sn['media'][nivel])}.",
        f"La probabilidad de inseguridad alimentaria severa es {_pct(p3)}, con un intervalo entre {ic[0]} y {ic[1]} según el desacuerdo entre los modelos del ensamble. "
        f"Esa cifra {'supera' if sn['supera'] else 'no supera'} el umbral operativo ({m['umbral']:.3f}), fijado con los datos de Validación para la clase severa; superarlo no equivale a afirmar que el hogar esté en inseguridad severa.",
        f"Para ponerla en contexto: en la muestra de Desarrollo el {_pct(prev3)} de los hogares (ponderado) está en inseguridad severa, y la probabilidad de este hogar es mayor que la de {_pct(sn['percentil'], 0)} de los hogares de Validación.",
    ]
    if sn["shap"]:
        clar.append(sn["interp"]["shap"]["que_muestra"] + " Las barras bermellón empujan la probabilidad del nivel hacia arriba; las azules, hacia abajo.")
    if len(sn["comparacion"]):
        clar.append(sn["interp"]["comparacion"]["que_muestra"])
    if (sn.get("perfil_ctx") or {}).get("disponible"):
        clar.append(PV.texto_contexto_hogar(sn["perfil_ctx"])[0])
    if sn["interp"].get("lime"):
        clar.append("Con LIME, una aproximación local del modelo: " + sn["interp"]["lime"]["que_muestra"])
    if sn["interp"].get("lime_shap"):
        clar.append("Comparación con SHAP: " + sn["interp"]["lime_shap"]["que_muestra"])
    if sn["interp"].get("tornado"):
        clar.append("Sensibilidad: " + sn["interp"]["tornado"]["que_muestra"])
    if sn["interp"].get("contra"):
        clar.append("Contrafactual (solo asociación predictiva del modelo, no causal ni recomendación): " + sn["interp"]["contra"]["que_muestra"])
    if sn["interp"].get("otros"):
        clar.append("Otros modelos de la corrida: " + sn["interp"]["otros"]["que_muestra"])
    if sn["interp"].get("validez"):
        clar.append("Validez de las entradas: " + sn["interp"]["validez"]["que_muestra"])
    clar.append(_advertencias(m)[0])
    sec.append({"titulo": "6. Interpretación en lenguaje claro", "hoja": "6_Interpretacion", "bloques": [_p(t) for t in clar]})
    sec.append({"titulo": "7. Análisis de sensibilidad", "hoja": "7_Sensibilidad", "bloques": _bloques_sensibilidad(sn, dir_fig)})
    sec.append({"titulo": "8. Análisis contrafactual (qué tendría que cambiar para bajar el riesgo)", "hoja": "8_Contrafactual", "bloques": _bloques_contrafactual(sn, dir_fig)})
    sec.append({"titulo": "9. Comparación con los demás modelos de la corrida", "hoja": "9_Otros_modelos", "bloques": _bloques_otros_modelos(sn, dir_fig)})
    sec.append({"titulo": "10. Metodología", "hoja": "91_Metodologia", "bloques": _seccion_metodologia(sn)})
    sec.append({"titulo": "11. Limitaciones y advertencias", "hoja": "92_Limitaciones", "bloques": _seccion_limitaciones(sn)})
    sec.append({"titulo": "12. Trazabilidad", "hoja": "93_Trazabilidad", "bloques": _seccion_traza(sn)})
    sec.append({"titulo": "Anexo A. Diccionario de variables usadas", "hoja": "9_Diccionario", "bloques": _seccion_anexo(sn)})
    doc["secciones"] = sec
    doc["excel_extra"] = {"metodologia": _metodologia_excel(sn)}
    return doc


# ------------------------------------------------------------ documento por lote
def construir_doc_lote(sn: dict[str, Any], dir_fig: Path, formato: str = "word") -> dict[str, Any]:
    m = sn["modelo"]
    out = sn["salida"]
    doc = _doc_comun(sn, "Reporte de inferencia por lote")
    doc["portada"].insert(5, ("Archivo del lote", f"{sn['archivo'] or 'CSV cargado'} · {sn['n']:,} hogares"))
    k_mod = int(np.argmax(sn["conteo_clases"]))
    doc["resumen"] = [
        f"Este reporte documenta una inferencia por lote realizada el {sn['fecha'].replace('T', ' ')} sobre {sn['n']:,} hogares (archivo «{sn['archivo'] or 'CSV cargado'}», SHA-256 {sn['huella'][:16]}...) con el modelo {m['codigo']} ({m['algoritmo']}), escenario «{m['escenario']}», corrida «{m['corrida']}».",
        f"Resultado agregado: el nivel predicho más frecuente es el {k_mod} ({NOMBRE_NIVEL[k_mod].lower()}, {_pct(sn['conteo_clases'][k_mod] / sn['n'])} de los hogares). "
        f"{sn['n_alertas']:,} hogares ({_pct(sn['n_alertas'] / sn['n'])}) superan el umbral operativo de IA severa ({m['umbral']:.3f}); la P(IA severa) media del lote es {_pct(sn['p_severa_media'])} (mediana {_pct(sn['p_severa_mediana'])}), "
        f"frente a una prevalencia de referencia de IA severa de {_pct(sn['prevalencia'][C.CLASE_SEVERA])} en Desarrollo.",
        _resumen_metricas_texto(m), _advertencias(m)[0],
        "Este reporte no es un diagnóstico individual ni sustituye la medición con la escala FIES; los umbrales de los niveles FIES están [POR VALIDAR]. Los resultados por hogar no deben usarse para negar ayudas.",
    ]
    cls = pd.DataFrame({"Nivel FIES": [_nivel_texto(k) for k in C.CLASES], "Hogares predichos": sn["conteo_clases"], "% del lote": 100 * sn["conteo_clases"] / sn["n"],
                        "P media del lote (%)": 100 * sn["prob_media"], "Prevalencia de referencia (%)": 100 * sn["prevalencia"]})
    hist = _hist_psevera(sn)
    pcts = _tabla_percentiles_lote(sn)
    sec: list[dict[str, Any]] = []
    ent_bl: list[dict[str, Any]] = [
        _p(f"Archivo: {sn['archivo'] or 'CSV cargado'} · {sn['n']:,} hogares · SHA-256 {sn['huella']}. Se usaron las {len(m['features_num']) + len(m['features_cat'])} columnas requeridas por el modelo, con su código original como encabezado."),
        _tabla("Resumen de las variables de entrada del lote", sn["resumen_entradas"].round(4), {"que_muestra": "Estadísticos descriptivos de cada columna del CSV usada por el modelo.", "como_leerlo": "Revise % faltante y rangos antes de interpretar los resultados.", "limitacion": "Los valores individuales de los hogares solo se incluyen en el libro Excel."}, max_filas=40)]
    if formato == "excel":
        ent = sn["entradas"].copy()
        ent.columns = [f"{E.etq(c)} [{c}]" for c in ent.columns]
        ent.insert(0, "Fila del CSV", np.arange(1, len(ent) + 1))
        ent_bl.append(_tabla("Datos de entrada por hogar (todas las filas)", ent))
    else:
        ent_bl.append(_p("(La tabla con los valores de entrada de cada hogar está en el libro Excel de este reporte.)"))
    sec.append({"titulo": "1. Datos de entrada", "hoja": "1_Entradas", "bloques": ent_bl})
    sec.append({"titulo": "2. Modelo y corrida", "hoja": "2_Modelo", "bloques": _seccion_modelo(sn)})
    tabla_res = out.copy()
    tabla_res.insert(0, "Hogar (fila del CSV)", np.arange(1, len(out) + 1))
    tabla_res = tabla_res.rename(columns={**{f"p_clase_{k}": f"P(nivel {k})" for k in C.CLASES}, "nivel_predicho": "Nivel predicho", "p_severa_ic_inf": "P(severa) IC95 inf", "p_severa_ic_sup": "P(severa) IC95 sup", "alerta_severa_umbral": "Supera umbral (1 = sí)"})
    cols_res = ["Hogar (fila del CSV)", "Nivel predicho"] + [f"P(nivel {k})" for k in C.CLASES] + ["P(severa) IC95 inf", "P(severa) IC95 sup", "Supera umbral (1 = sí)"]
    tabla_res = tabla_res[cols_res]
    f1 = _bloques_figura("fig_lote_clases", _fig_lote_clases(sn), dir_fig, "Distribución de niveles FIES predichos en el lote frente a la prevalencia de referencia",
                         {"que_muestra": f"De {sn['n']:,} hogares, {int(sn['conteo_clases'][3]):,} se predicen en el nivel severo (argmax); la prevalencia de referencia de ese nivel es {_pct(sn['prevalencia'][3])}.",
                          "como_leerlo": "Barras azules: porcentaje del lote en cada nivel predicho; naranja: prevalencia ponderada de Desarrollo.", "limitacion": "Un lote no aleatorio no es comparable con la población: las diferencias reflejan la composición del lote."},
                         [_d("Distribución del lote y prevalencia de referencia", cls, {"tipo": "barra", "titulo": "Niveles FIES predichos vs prevalencia de referencia (%)", "cat": "Nivel FIES",
                                                                                          "series": [{"col": "% del lote", "nombre": "Lote (predicho)"}, {"col": "Prevalencia de referencia (%)", "nombre": "Prevalencia de referencia"}], "y_titulo": "%"})])
    f2 = _bloques_figura("fig_lote_psevera", _fig_lote_psevera(sn), dir_fig, "Distribución de P(IA severa) en el lote",
                         {"que_muestra": f"{sn['n_alertas']:,} hogares ({_pct(sn['n_alertas'] / sn['n'])}) tienen P(IA severa) igual o superior al umbral operativo ({m['umbral']:.3f}).", "como_leerlo": "La línea punteada marca el umbral; los hogares a su derecha generan alerta.", "limitacion": "Una alerta es una señal estadística, no un diagnóstico."},
                         [_d("Frecuencias del histograma de P(IA severa)", hist[["intervalo", "desde", "hasta", "hogares"]], {"tipo": "barra", "titulo": "Hogares por intervalo de P(IA severa)", "cat": "intervalo", "series": [{"col": "hogares", "nombre": "Hogares"}], "y_titulo": "Hogares"})])
    f3 = _bloques_figura("fig_lote_probabilidades", _fig_lote_prob(sn), dir_fig, "Distribución de las probabilidades asignadas por nivel FIES",
                         {"que_muestra": "Dispersión de la probabilidad que el ensamble asigna a cada nivel entre los hogares del lote.", "como_leerlo": "Cajas estrechas indican poca variación de esa probabilidad entre hogares.", "limitacion": "Describe al lote, no la calidad del modelo."},
                         [_d("Percentiles de las probabilidades por nivel", pcts)])
    sec.append({"titulo": "3. Resultados del lote", "hoja": "3_Resultados", "bloques": [
        _tabla("Resumen agregado del lote", pd.DataFrame([
            ("Hogares en el lote", f"{sn['n']:,}"), ("Hogares que superan el umbral operativo de IA severa", f"{sn['n_alertas']:,} ({_pct(sn['n_alertas'] / sn['n'])})"),
            ("P(IA severa) media / mediana", f"{_pct(sn['p_severa_media'])} / {_pct(sn['p_severa_mediana'])}"), ("Umbral operativo", f"{m['umbral']:.4f}"),
            ("Suma de probabilidades por hogar (mín.-máx.)", f"{float(out[[f'p_clase_{k}' for k in C.CLASES]].sum(axis=1).min()):.4f} a {float(out[[f'p_clase_{k}' for k in C.CLASES]].sum(axis=1).max()):.4f}")], columns=["Indicador", "Valor"])),
        _tabla("Distribución de niveles FIES predichos", cls.assign(**{c: cls[c].map(lambda v: f"{v:.1f}") for c in ["% del lote", "P media del lote (%)", "Prevalencia de referencia (%)"]})), f1, f2, f3,
        _tabla("Resultados por hogar" + ("" if formato == "excel" else " (primeras filas)"), tabla_res, {"que_muestra": "Probabilidad por nivel, nivel predicho, IC95 de P(severa) y alerta de cada hogar.", "como_leerlo": "Las filas siguen el orden del CSV.", "limitacion": "Resultados individuales no deben usarse para negar ayudas."},
               max_filas=len(tabla_res) if formato == "excel" else 30)]})
    sec.append({"titulo": "3b. Perfil de vulnerabilidad de los hogares del lote", "hoja": "3b_Perfil", "bloques": _seccion_perfil_lote(sn)})
    bl4: list[dict[str, Any]] = []
    if sn["shap_global"]:
        imp = pd.Series(sn["shap_global"]["importancia"]).sort_values(ascending=False)
        d_imp = pd.DataFrame({"Rango": np.arange(1, len(imp) + 1), "Variable": [E.etq(v) for v in imp.index], "Código de la columna del modelo": imp.index, "|SHAP| medio": imp.to_numpy()})
        bl4.append(_p(f"Por lote no se calcula una explicación local para cada hogar (use la predicción individual). Se muestra la importancia global guardada del modelo ({sn['shap_global']['explainer']}"
                      f"{', muestra de ' + str(int(sn['shap_global']['n_muestra'])) + ' hogares' if sn['shap_global'].get('n_muestra') else ''}; archivo {sn['shap_global']['archivo']}), que describe al modelo y no a este lote."))
        top = imp.head(15).iloc[::-1]
        with plt.rc_context(P.RC_REVISTA):
            fig, ax = plt.subplots(figsize=(P.ANCHO_DOBLE, max(2.2, 0.24 * len(top) + 0.9)))
            ax.barh([E.etq(v) for v in top.index], top.to_numpy(), color=P.OKABE_ITO[0], height=0.65, zorder=2)
            ax.set_xlabel("|SHAP| medio")
            ax.grid(axis="y", visible=False)
            ax.set_title("Importancia global del modelo (|SHAP| medio, 15 variables principales)", loc="left")
        bl4.append(_bloques_figura("fig_lote_shap_global", fig, dir_fig, "Importancia global de las variables del modelo",
                                   {"que_muestra": "Variables que más pesan en las predicciones del modelo, en promedio sobre una muestra de hogares de entrenamiento.", "como_leerlo": "Barras más largas = mayor influencia media.", "limitacion": "Es una explicación del modelo, no del lote ni causal."},
                                   [_d("|SHAP| medio por variable", d_imp, {"tipo": "barra_h", "titulo": "Importancia global SHAP: 15 variables principales", "cat": "Variable", "series": [{"col": "|SHAP| medio", "nombre": "|SHAP| medio"}], "max_filas": 15, "alto_cm": 11})]))
        bl4.append(_tabla("Ranking de variables por importancia global", d_imp, max_filas=25))
    else:
        bl4.append({"tipo": "vacio", "texto": "No hay importancia global guardada para este modelo; la inferencia por lote no calcula explicaciones locales."})
    sec.append({"titulo": "4. Explicabilidad", "hoja": "4_Explicabilidad", "bloques": bl4})
    sec.append({"titulo": "5. Sensibilidad agregada del lote", "hoja": "5_Sensibilidad", "bloques": _bloques_lote_sens(sn, dir_fig)})
    sec.append({"titulo": "6. Validez de las entradas del lote (extrapolación)", "hoja": "6_Validez", "bloques": _bloques_lote_validez(sn)})
    k_prev = int(np.argmax(sn["prevalencia"]))
    sec.append({"titulo": "7. Interpretación en lenguaje claro", "hoja": "7_Interpretacion", "bloques": [_p(t) for t in [
        f"En el lote de {sn['n']:,} hogares, el modelo asigna en promedio {_pct(sn['prob_media'][3])} de probabilidad al nivel severo. {sn['n_alertas']:,} hogares ({_pct(sn['n_alertas'] / sn['n'])}) superan el umbral operativo ({m['umbral']:.3f}) y quedarían señalados para revisión.",
        f"En Desarrollo, el nivel más frecuente es el {k_prev} ({NOMBRE_NIVEL[k_prev].lower()}, {_pct(sn['prevalencia'][k_prev])}) y el nivel severo representa {_pct(sn['prevalencia'][3])}. Compare estas cifras con la distribución del lote solo si el lote es representativo.",
        "Cada hogar tiene sus propias probabilidades y un intervalo que refleja el desacuerdo entre los modelos del ensamble; un intervalo ancho indica una estimación menos estable.", _advertencias(m)[0]]]})
    sec.append({"titulo": "8. Metodología", "hoja": "81_Metodologia", "bloques": _seccion_metodologia(sn)})
    sec.append({"titulo": "9. Limitaciones y advertencias", "hoja": "82_Limitaciones", "bloques": _seccion_limitaciones(sn)})
    sec.append({"titulo": "10. Trazabilidad", "hoja": "83_Trazabilidad", "bloques": _seccion_traza(sn)})
    sec.append({"titulo": "Anexo A. Diccionario de variables usadas", "hoja": "9_Diccionario", "bloques": _seccion_anexo(sn)})
    doc["secciones"] = sec
    doc["excel_extra"] = {"metodologia": _metodologia_excel(sn)}
    return doc


# ============================================================ API pública
def nombre_archivo(sn: dict[str, Any], formato: str) -> str:
    return f"reporte_{sn['id']}.{FORMATOS[formato][0]}"


def construir_doc(sn: dict[str, Any], dir_fig: Path, formato: str = "word") -> dict[str, Any]:
    return construir_doc_individual(sn, dir_fig) if sn["tipo"] == "individual" else construir_doc_lote(sn, dir_fig, formato)


def generar_archivos(sn: dict[str, Any], destino: Path | None = None) -> dict[str, bytes]:
    """Genera los tres formatos. Si ``destino`` es una carpeta, deja ahí los archivos y las figuras PNG de 600 DPI."""
    out: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory(prefix="fsai_inf_") as tmp:
        base = Path(destino) if destino else Path(tmp)
        base.mkdir(parents=True, exist_ok=True)
        dir_fig = base / "figuras" if destino else base
        dir_fig.mkdir(parents=True, exist_ok=True)
        exportadores = {"word": exportar_word, "excel": lambda d, r: exportar_excel(d, r, {"id_inferencia": sn["id"], "huella_sha256": sn["huella"], "modelo": sn["modelo"]["id"], "corrida": sn["modelo"]["corrida"], "usuario": sn["usuario"] or "sin sesión"}),
                        "pdf": exportar_pdf}
        docs: dict[str, dict[str, Any]] = {}
        for fmt, fn in exportadores.items():
            clave = "excel" if (fmt == "excel" and sn["tipo"] == "lote") else "base"
            if clave not in docs:
                docs[clave] = construir_doc(sn, dir_fig, fmt)
            ruta = base / nombre_archivo(sn, fmt)
            fn(docs[clave], ruta)
            out[fmt] = ruta.read_bytes()
    return out
