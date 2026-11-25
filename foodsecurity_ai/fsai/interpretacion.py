"""interpretacion.py - Textos interpretativos generados A PARTIR de los datos mostrados.

Cada función devuelve ``{"que_muestra", "como_leerlo", "limitacion"}``. Las cifras salen SIEMPRE de los objetos
recibidos (DataFrames, dicts, arreglos); las plantillas solo añaden frases condicionales (p. ej. "asimétrica a la
derecha" si la asimetría supera 1). Con datos vacíos devuelven un estado vacío informativo; nunca inventan cifras.
Una asociación o un patrón descrito aquí NO es una relación causal.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from fsai.etiquetas import etq as _etq  # presentación: nombre legible de la variable

Texto = dict[str, str]
ETIQUETA_NIVEL = {0: "seguridad", 1: "inseguridad leve", 2: "inseguridad moderada", 3: "inseguridad severa"}


def _pct(x: float, d: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f} %"


def _n(x: float, d: int = 3) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def vacio(motivo: str = "No hay datos para interpretar todavía.") -> Texto:
    return {"que_muestra": motivo, "como_leerlo": "Cargue o genere los datos requeridos y vuelva a esta sección.",
            "limitacion": "Sin datos no se calcula ninguna cifra."}


def _w_mean(x: pd.Series, w: pd.Series | None) -> float:
    x = pd.to_numeric(x, errors="coerce")
    m = x.notna()
    return float(np.average(x[m], weights=None if w is None else w[m])) if m.any() else float("nan")


# ------------------------------ Fase 1 ------------------------------
def i_objetivo(df: pd.DataFrame, col: str = "fies_nivel", peso: str = "factor_expansion") -> Texto:
    if df is None or df.empty or col not in df.columns:
        return vacio()
    sin = df[col].value_counts(normalize=True).sort_index()
    con = df.groupby(col)[peso].sum() / df[peso].sum() if peso in df.columns else sin
    top = int(sin.idxmax())
    ia = float(sin[sin.index >= 2].sum())
    ia_w = float(con[con.index >= 2].sum())
    n_sev = int((df[col] == 3).sum())
    lim = (f"La clase severa tiene solo {n_sev} hogares: las métricas por clase serán inestables." if n_sev < 100
           else f"La clase severa tiene {n_sev} hogares; aun así conviene reportar intervalos de confianza.")
    return {"que_muestra": (f"Distribución del nivel FIES en {len(df):,} hogares: el nivel {top} ({ETIQUETA_NIVEL.get(top, '')}) es el más "
                            f"frecuente ({_pct(float(sin[top]))} sin ponderar). La inseguridad moderada o severa (niveles 2-3) suma "
                            f"{_pct(ia)} sin ponderar y {_pct(ia_w)} ponderada con el factor de expansión."),
            "como_leerlo": ("Cada barra es la proporción de hogares del nivel; la versión ponderada aproxima a la población que "
                            "representa la muestra y la no ponderada describe la muestra tal cual."),
            "limitacion": lim}


def i_nulos(df: pd.DataFrame) -> Texto:
    if df is None or df.empty:
        return vacio()
    p = df.isna().mean().sort_values(ascending=False)
    con_nulos = p[p > 0]
    if con_nulos.empty:
        return {"que_muestra": "Ninguna variable tiene valores faltantes.", "como_leerlo": "Sin faltantes no se requiere imputación.",
                "limitacion": "La ausencia de nulos no garantiza que los valores sean correctos (p. ej. ceros que codifican 'no sabe')."}
    return {"que_muestra": (f"{len(con_nulos)} de {len(p)} variables tienen faltantes; la más afectada es "
                            f"«{_etq(con_nulos.index[0])}» con {_pct(float(con_nulos.iloc[0]))}."),
            "como_leerlo": "Un porcentaje alto indica que la variable aporta poco o que requiere imputación (mediana/moda dentro de cada pliegue de entrenamiento).",
            "limitacion": "Los faltantes pueden no ser aleatorios; imputar puede sesgar. La imputación del pipeline se ajusta solo con train para evitar fuga."}


def i_numerica(df: pd.DataFrame, var: str, peso: str = "factor_expansion") -> Texto:
    if df is None or var not in df.columns:
        return vacio()
    x = pd.to_numeric(df[var], errors="coerce").dropna()
    if x.empty:
        return vacio(f"La variable «{_etq(var)}» no tiene valores numéricos.")
    w = df.loc[x.index, peso] if peso in df.columns else None
    asim = float(x.skew())
    forma = ("asimétrica a la derecha (cola de valores altos): la mediana describe mejor el caso típico que la media" if asim > 1
             else "asimétrica a la izquierda" if asim < -1 else "aproximadamente simétrica")
    q1, q3 = x.quantile([0.25, 0.75])
    n_out = int(((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))).sum())
    return {"que_muestra": (f"«{_etq(var)}»: media {_n(float(x.mean()))} (ponderada {_n(_w_mean(x, w))}), mediana {_n(float(x.median()))}, "
                            f"desviación estándar {_n(float(x.std()))}, rango {_n(float(x.min()))} a {_n(float(x.max()))}. Distribución {forma}."),
            "como_leerlo": f"El histograma muestra dónde se concentran los hogares; {n_out} valores ({_pct(n_out / len(x))}) quedan fuera de 1,5 veces el rango intercuartílico.",
            "limitacion": "Los valores atípicos pueden ser reales (hogares muy pobres o ricos) o errores de captura; no se eliminan automáticamente."}


def i_categorica(df: pd.DataFrame, var: str) -> Texto:
    if df is None or var not in df.columns:
        return vacio()
    f = df[var].astype(str).value_counts(normalize=True)
    return {"que_muestra": f"«{_etq(var)}» tiene {len(f)} categorías; la más frecuente es «{f.index[0]}» ({_pct(float(f.iloc[0]))}).",
            "como_leerlo": "Las barras son proporciones de hogares por categoría.",
            "limitacion": "Categorías con pocos hogares generan estimaciones inestables; agrúpelas antes de modelar si es necesario."}


def i_bivariada(df: pd.DataFrame, var: str, objetivo: str = "fies_nivel") -> Texto:
    if df is None or var not in df.columns or objetivo not in df.columns:
        return vacio()
    x = pd.to_numeric(df[var], errors="coerce")
    if x.notna().sum() < 10:
        return vacio(f"«{_etq(var)}» no es numérica o tiene pocos datos.")
    medias = x.groupby(df[objetivo]).mean()
    rho = float(x.corr(df[objetivo], method="spearman"))
    sentido = "aumenta" if rho > 0.05 else "disminuye" if rho < -0.05 else "no cambia de forma clara"
    return {"que_muestra": (f"Media de «{_etq(var)}» por nivel FIES: " + "; ".join(f"nivel {int(k)} = {_n(float(v))}" for k, v in medias.items())
                            + f". Correlación de Spearman con el nivel: {_n(rho)}; la variable {sentido} al subir el nivel de inseguridad."),
            "como_leerlo": "Si las medias crecen o decrecen de forma ordenada con el nivel, la variable está asociada (no necesariamente causa) al nivel FIES.",
            "limitacion": "Es una asociación bivariada sin controlar por otras variables; una correlación baja no excluye relaciones no lineales o de interacción."}


def i_correlacion(corr: pd.DataFrame) -> Texto:
    if corr is None or corr.empty:
        return vacio()
    c = corr.where(~np.eye(len(corr), dtype=bool)).abs().unstack().dropna().sort_values(ascending=False)
    if c.empty:
        return vacio()
    (a, b), v = c.index[0], float(c.iloc[0])
    fuertes = int((c.groupby(level=[0, 1]).first() > 0.8).sum() // 2) if len(c) else 0
    return {"que_muestra": f"La pareja más correlacionada es «{_etq(a)}» y «{_etq(b)}» (|r| = {_n(v)}). Hay {fuertes} parejas con |r| > 0,8.",
            "como_leerlo": "Colores intensos indican correlación lineal fuerte (positiva o negativa) entre dos variables.",
            "limitacion": "Correlaciones altas implican redundancia: afectan la interpretación de coeficientes y de SHAP, no necesariamente la predicción."}


def i_balance(tabla: pd.DataFrame, alfa: float = 0.05) -> Texto:
    if tabla is None or tabla.empty:
        return vacio()
    m = len(tabla)
    sig = tabla[tabla["p_valor"] < alfa]
    esperado = alfa * m
    base = (f"De {m} pruebas de balance entre particiones, {len(sig)} tienen p < {alfa}. "
            + ("No hay evidencia de desbalance en las variables evaluadas." if sig.empty else
               "Variables con p bajo: " + ", ".join(sorted({_etq(v) for v in sig['variable']})) + "."))
    return {"que_muestra": base,
            "como_leerlo": f"Un p-valor alto (≥ {alfa}) es compatible con particiones comparables; con {m} pruebas se esperan ≈{esperado:.1f} p-valores bajos por azar.",
            "limitacion": "Sin corrección por pruebas múltiples; la prueba ponderada usa el tamaño efectivo de Kish. Un p alto no prueba equivalencia."}


def i_particion(ver: Mapping[str, Any]) -> Texto:
    if not ver:
        return vacio()
    t, p = ver["tamanos"], ver["proporciones"]
    cr = ver.get("cruces_hogar_uid_total")
    cg = ver.get("cruces_conglome_total")
    msg = (f"Tamaños: desarrollo {t['desarrollo']:,} ({_pct(p['desarrollo'])}), validación {t['validacion']:,} ({_pct(p['validacion'])}), "
           f"prueba {t['prueba']:,} ({_pct(p['prueba'])}). Hogares repetidos entre particiones: {cr}. Conglomerados compartidos: {cg}.")
    nota = ("Los conglomerados compartidos son esperables si se agrupó por hogar_uid (esquema V1); use agrupación por conglome para eliminarlos."
            if cg else "No se comparten conglomerados entre particiones.")
    return {"que_muestra": msg, "como_leerlo": "Cruces = 0 en hogar_uid garantiza que ningún hogar se usa a la vez para entrenar y evaluar. " + nota,
            "limitacion": "Hogares del mismo conglomerado son más parecidos entre sí; si cruzan particiones, el desempeño puede ser algo optimista."}


# ------------------------------ Fase 2 ------------------------------
def i_nested(agregado: Mapping[str, Mapping[str, Mapping[str, float]]], metrica: str = "f1_macro") -> Texto:
    a = (agregado or {}).get("ponderada", {}).get(metrica)
    if not a or a.get("n", 0) == 0:
        return vacio("Aún no hay resultados de la validación cruzada anidada.")
    cv = a["sd"] / a["media"] if a["media"] else float("nan")
    est = "estable entre pliegues" if cv < 0.05 else "moderadamente variable entre pliegues" if cv < 0.15 else "muy variable entre pliegues"
    return {"que_muestra": (f"{metrica} ponderado en los {a['n']} pliegues externos: media {_n(a['media'])}, DE {_n(a['sd'])}, "
                            f"IC t 95 % [{_n(a['ic_inf'])}; {_n(a['ic_sup'])}]. Coeficiente de variación {_pct(cv)} ({est})."),
            "como_leerlo": "Cada punto es una única evaluación en outer_val de un modelo cuyos hiperparámetros se eligieron solo con outer_train; la media estima el desempeño esperado en hogares nuevos.",
            "limitacion": "Con 5 observaciones el IC t es ancho y supone normalidad; los pliegues comparten datos de entrenamiento, así que no son independientes."}


def i_curva_optuna(valores: list[float]) -> Texto:
    v = [x for x in valores if x is not None and not math.isnan(x)]
    if not v:
        return vacio("Aún no hay ensayos completados.")
    mejor = np.maximum.accumulate(v)
    mejoras = int((np.diff(mejor) > 1e-12).sum())
    ult = int(np.nonzero(np.diff(mejor) > 1e-12)[0][-1] + 2) if mejoras else 1
    estado = ("la búsqueda sigue mejorando al final: conviene más ensayos" if ult >= len(v) and len(v) > 3
              else "la mejora se estancó antes del final")
    return {"que_muestra": f"En {len(v)} ensayos, el mejor F1-macro interno es {_n(float(mejor[-1]))}; hubo {mejoras} mejoras y la última ocurrió en el ensayo {ult}; {estado}.",
            "como_leerlo": "La línea es el mejor valor acumulado; los puntos son cada ensayo. Un plateau largo sugiere saturación del espacio explorado.",
            "limitacion": "Con pocos ensayos el optimizador apenas explora el espacio; el valor interno es optimista respecto del de outer_val."}


def i_estabilidad_hp(folds: list[Mapping[str, Any]]) -> Texto:
    if not folds:
        return vacio()
    claves = sorted({k for f in folds for k in f["mejores_params"]})
    distintos = {k: len({str(f["mejores_params"].get(k)) for f in folds}) for k in claves}
    iguales = [k for k, d in distintos.items() if d == 1]
    return {"que_muestra": (f"Entre {len(folds)} pliegues externos, {len(iguales)} de {len(claves)} hiperparámetros coincidieron en todos"
                            + (f" ({', '.join(iguales)})." if iguales else ".")),
            "como_leerlo": "Si los mejores hiperparámetros cambian mucho entre pliegues, el desempeño es poco sensible a ellos o la búsqueda fue corta.",
            "limitacion": "Con pocos ensayos las diferencias reflejan más el azar del muestreo que una verdadera sensibilidad."}


# ------------------------------ Fase 3 ------------------------------
def i_metricas(df: pd.DataFrame) -> Texto:
    if df is None or df.empty or "f1_macro" not in df.columns:
        return vacio()
    mejor = df.sort_values("f1_macro", ascending=False).iloc[0]
    return {"que_muestra": f"En la Prueba, el modelo con mayor F1-macro ponderado es {mejor.get('modelo', '—')} ({_n(float(mejor['f1_macro']))}).",
            "como_leerlo": "Cada celda es la métrica con su IC bootstrap estratificado (percentil 2,5-97,5); si los IC de dos modelos se solapan, la diferencia no es concluyente.",
            "limitacion": "Evaluación única sobre el 15 % de Prueba; el IC refleja variabilidad de remuestreo, no de re-particionar."}


def i_confusion(cm: np.ndarray, clases: list[int] | None = None) -> Texto:
    cm = np.asarray(cm)
    if cm.size == 0 or cm.sum() == 0:
        return vacio()
    clases = clases or list(range(cm.shape[0]))
    diag = float(np.trace(cm) / cm.sum())
    fuera = cm.astype(float).copy()
    np.fill_diagonal(fuera, 0)
    i, j = np.unravel_index(fuera.argmax(), fuera.shape)
    k = clases.index(3) if 3 in clases else cm.shape[0] - 1
    rec = cm[k, k] / cm[k].sum() if cm[k].sum() else float("nan")
    return {"que_muestra": (f"Exactitud global {_pct(diag)}. El error más frecuente es clasificar nivel {clases[i]} como nivel {clases[j]} ({int(fuera[i, j])} hogares). "
                            f"Para IA severa se recupera {_pct(float(rec))} de los {int(cm[k].sum())} hogares verdaderos."),
            "como_leerlo": "Filas = nivel real, columnas = nivel predicho. En la versión normalizada por fila, la diagonal es la sensibilidad (recall) de cada nivel.",
            "limitacion": "Con pocos hogares severos en la Prueba, el recall de severa tiene un IC amplio; no extrapole."}


def i_roc_pr(auc: Mapping[int, float], ap: Mapping[int, float], prev: Mapping[int, float]) -> Texto:
    if not auc:
        return vacio()
    mejor = max(auc, key=auc.get)
    peor = min(auc, key=auc.get)
    return {"que_muestra": (f"AUC uno-contra-resto: " + ", ".join(f"clase {k} = {_n(v)}" for k, v in sorted(auc.items()))
                            + f". La clase mejor separada es {mejor} y la peor {peor}. PR-AUC clase severa = {_n(ap.get(3, float('nan')))} "
                              f"frente a una prevalencia de {_pct(prev.get(3, float('nan')))} (línea base)."),
            "como_leerlo": "ROC: cuanto más cerca de la esquina superior izquierda, mejor; AUC 0,5 es azar. PR: la referencia es la prevalencia de la clase, no 0,5.",
            "limitacion": "El AUC no mide calibración ni el costo de falsos negativos; para clases raras la curva PR es más informativa."}


def i_roc_pr_modelos(auc: Mapping[str, float], ap: Mapping[str, float], prev_severa: float, cual: str = "roc") -> Texto:
    """ROC o PR de la clase IA severa (uno-contra-resto) con un valor por MODELO (las claves de ``auc``/``ap`` son los códigos M1-M5)."""
    d = auc if cual == "roc" else ap
    if not d:
        return vacio()
    mejor, peor = max(d, key=d.get), min(d, key=d.get)
    lista = ", ".join(f"{k} = {_n(v)}" for k, v in d.items())
    if cual == "roc":
        return {"que_muestra": f"AUC ROC de la clase IA severa (uno-contra-resto) por modelo: {lista}. El modelo con mayor AUC es {mejor} y el de menor, {peor}.",
                "como_leerlo": "Cuanto más cerca de la esquina superior izquierda, mejor discrimina; la diagonal punteada (AUC 0,5) es el azar.",
                "limitacion": "El AUC no mide calibración ni el costo de los falsos negativos; no depende de la prevalencia, por eso para una clase rara conviene mirar también la curva precisión-recall."}
    return {"que_muestra": f"Precisión promedio (AP) de la clase IA severa por modelo: {lista}, frente a una prevalencia de {_pct(prev_severa)} (línea base de un clasificador al azar). Mayor AP: {mejor}; menor: {peor}.",
            "como_leerlo": "Cuanto más alta la curva respecto de la prevalencia, mejor: muestra cuántos de los hogares señalados como severos lo son realmente a cada nivel de recall.",
            "limitacion": "La precisión depende de la prevalencia de la muestra evaluada; con una clase rara (IA severa) la precisión es baja aunque el AUC parezca aceptable."}


def i_calibracion(brier: float, ece: float, brier_cal: float | None = None, ece_cal: float | None = None) -> Texto:
    if brier is None or (isinstance(brier, float) and math.isnan(brier)):
        return vacio()
    extra = ""
    if brier_cal is not None and ece_cal is not None:
        mejora = "mejoró" if ece_cal < ece else "no mejoró"
        extra = f" Tras calibrar con la Validación: Brier {_n(brier_cal)} y ECE {_n(ece_cal)} (el ECE {mejora})."
    return {"que_muestra": f"Brier multiclase {_n(brier)} y ECE {_n(ece)} del modelo sin calibrar.{extra}",
            "como_leerlo": "Una curva sobre la diagonal indica probabilidades subestimadas; por debajo, sobreestimadas. ECE es la distancia media ponderada a la diagonal.",
            "limitacion": "Con pocas observaciones por bin, la curva es ruidosa; el calibrador se ajusta en Validación y se evalúa en Prueba."}


def i_pruebas(tabla: pd.DataFrame, alfa: float = 0.05) -> Texto:
    if tabla is None or tabla.empty:
        return vacio("No hay pruebas estadísticas (se requieren al menos 2 modelos evaluados).")
    col = "p_ajustado" if "p_ajustado" in tabla.columns else "p_valor"
    sig = tabla[tabla[col] < alfa]
    return {"que_muestra": f"De {len(tabla)} comparaciones, {len(sig)} resultan significativas con {col} < {alfa}" + (": " + ", ".join(f"{r.prueba} {r.comparacion}" for r in sig.itertuples()) + "." if len(sig) else "."),
            "como_leerlo": "p ajustado por Holm controla el error tipo I al comparar varios pares; no significativo no implica igualdad de modelos.",
            "limitacion": "McNemar y DeLong asumen muestras simples; Wilcoxon con 5 pliegues tiene p mínimo posible 0,0625 y no puede ser significativo."}


# ------------------------------ Fase 4 ------------------------------
def i_importancia(imp: pd.DataFrame, col_val: str = "importancia", col_var: str = "variable") -> Texto:
    if imp is None or imp.empty:
        return vacio()
    d = imp.sort_values(col_val, ascending=False)
    tot = float(d[col_val].clip(lower=0).sum())
    top3 = d.head(3)
    conc = float(top3[col_val].clip(lower=0).sum() / tot) if tot > 0 else float("nan")
    return {"que_muestra": f"Las variables con mayor importancia son {', '.join(f'«{v}»' for v in top3[col_var])}, que concentran {_pct(conc)} de la importancia total.",
            "como_leerlo": "Mayor |SHAP| medio (o mayor caída de F1 al permutar) significa que el modelo depende más de esa variable para predecir.",
            "limitacion": "Importancia predictiva no es causalidad. Con el escenario fies/hibrido, los ítems FIES dominan por circularidad (reconstrucción del objetivo)."}


def i_heatmap(matriz: np.ndarray, etiqueta_filas: str = "variables") -> Texto:
    m = np.asarray(matriz)
    if m.size == 0:
        return vacio()
    return {"que_muestra": f"Mapa de calor de |SHAP| con {m.shape[0]} {etiqueta_filas} y {m.shape[1]} hogares; el valor máximo es {_n(float(np.nanmax(m)), 4)}.",
            "como_leerlo": "Cada celda es cuánto movió esa variable la predicción de ese hogar (más intenso = más influencia). Franjas verticales: hogares que dependen de pocas variables; filas intensas: variables influyentes para muchos hogares.",
            "limitacion": "SHAP se calcula sobre una submuestra y sobre variables transformadas (dummies). Es el análogo tabular de un 'mapa de atención'; Grad-CAM++ no aplica a datos tabulares."}


def i_mascara(top_k: int, n_vars: int, estabilidad: float | None = None) -> Texto:
    est = "" if estabilidad is None else f" En promedio, {_pct(estabilidad)} de las variables del top-k de un hogar coinciden con el top-k global."
    return {"que_muestra": f"Máscara binaria que marca, para cada hogar, las {top_k} de {n_vars} variables con mayor |SHAP| en la clase predicha.{est}",
            "como_leerlo": "Un cuadrado oscuro significa que la variable sostiene la predicción de ese hogar. Se valida con la curva de eliminación: si quitar el top-k hunde el desempeño, la máscara es fiel.",
            "limitacion": "Una máscara fiel al modelo no es una explicación causal del fenómeno."}


def i_eliminacion(curva: pd.DataFrame) -> Texto:
    if curva is None or curva.empty:
        return vacio()
    c0, c1 = float(curva["f1_eliminacion"].iloc[0]), float(curva["f1_eliminacion"].iloc[-1])
    i0, i1 = float(curva["f1_insercion"].iloc[0]), float(curva["f1_insercion"].iloc[-1])
    return {"que_muestra": (f"Al eliminar progresivamente las variables más importantes (permutándolas), el F1-macro cae de {_n(c0)} a {_n(c1)}. "
                            f"Al insertarlas desde cero (todas permutadas) sube de {_n(i0)} a {_n(i1)}."),
            "como_leerlo": "Una caída rápida en eliminación y una subida rápida en inserción indican que el orden de importancia es fiel al modelo.",
            "limitacion": "Permutar variables correlacionadas subestima su efecto conjunto; el F1 se evalúa en una submuestra."}


# ------------------------------ Fase 6 y 7 ------------------------------
def i_prediccion(prob: Mapping[int, float], nivel: int, p_severa: float, umbral: float, percentil: float | None) -> Texto:
    if not prob:
        return vacio()
    perc = "" if percentil is None else f" Su probabilidad de IA severa supera a {_pct(percentil)} de los hogares de referencia."
    alerta = "supera" if p_severa >= umbral else "no supera"
    return {"que_muestra": (f"El ensamble asigna mayor probabilidad al nivel {nivel} ({ETIQUETA_NIVEL.get(nivel, '')}, {_pct(prob[nivel])}). "
                            f"P(IA severa) = {_pct(p_severa)}, que {alerta} el umbral operativo de Validación ({_n(umbral)}).{perc}"),
            "como_leerlo": "Las probabilidades son del modelo, no certezas. El intervalo refleja desacuerdo entre los modelos del ensamble.",
            "limitacion": "No use este resultado para negar ayudas. Es una asociación estadística aprendida de otros hogares, no una causa."}


def i_optuna_resultados(df_trials: pd.DataFrame, importancias: Mapping[str, float] | None = None) -> Texto:
    if df_trials is None or df_trials.empty:
        return vacio("Aún no hay ensayos para analizar.")
    ok = df_trials[df_trials["state"] == "COMPLETE"]
    pr = int((df_trials["state"] == "PRUNED").sum())
    imp = ""
    if importancias:
        k = max(importancias, key=importancias.get)
        imp = f" El hiperparámetro más influyente en el valor objetivo es «{k}» ({_pct(float(importancias[k]))} de la importancia)."
    mejor = float(ok["value"].max()) if len(ok) else float("nan")
    return {"que_muestra": f"{len(df_trials)} ensayos: {len(ok)} completos, {pr} podados; mejor valor {_n(mejor)}.{imp}",
            "como_leerlo": "Coordenadas paralelas: cada línea es un ensayo; busque dónde se concentran las líneas de mayor valor. Slice/contour muestran el efecto de uno/dos hiperparámetros.",
            "limitacion": "Las importancias de hiperparámetros (fANOVA/aproximación) son inestables con pocos ensayos."}


# ------------------------------ Procesamiento FIES, vulnerabilidad y territorial ------------------------------
def i_fies(raw: pd.Series, concordancia: float | None, umbrales: Mapping[str, float], n_sin_dato: int = 0) -> Texto:
    if raw is None or len(raw) == 0:
        return vacio()
    dist = raw.value_counts(normalize=True).sort_index()
    modal = int(dist.idxmax())
    conc = "" if concordancia is None else f" El nivel guardado coincide con el recalculado con estos umbrales en {_pct(concordancia)} de los hogares."
    return {"que_muestra": (f"Puntaje crudo FIES (0-8, suma de respuestas afirmativas): el valor más frecuente es {modal} ({_pct(float(dist[modal]))}); "
                            f"el {_pct(float(dist[dist.index == 0].sum()))} de los hogares tiene puntaje 0. Cortes inferiores aplicados: leve ≥ {umbrales['leve']}, "
                            f"moderada ≥ {umbrales['moderada']}, severa ≥ {umbrales['severa']}.{conc}" + (f" {n_sin_dato} hogares sin puntaje por ítems faltantes." if n_sin_dato else "")),
            "como_leerlo": "Cada barra es la proporción de hogares con ese puntaje; las líneas verticales marcan los cortes de los niveles. Mover los cortes redistribuye a los hogares entre niveles.",
            "limitacion": "Los cortes sobre el puntaje crudo y los parámetros Rasch para el Perú están marcados [POR VALIDAR]; no se afirma calibración peruana."}


def i_vulnerabilidad(tabla: pd.DataFrame, nivel_bajo: int = 0, nivel_alto: int = 3) -> Texto:
    if tabla is None or tabla.empty or nivel_bajo not in tabla.columns or nivel_alto not in tabla.columns:
        return vacio("No hay indicadores calculables para comparar niveles de inseguridad.")
    dif = (tabla[nivel_alto] - tabla[nivel_bajo]).dropna()
    if dif.empty:
        return vacio()
    mayor = dif.abs().sort_values(ascending=False).index[0]
    return {"que_muestra": (f"Entre los hogares de nivel {nivel_alto} y de nivel {nivel_bajo}, el indicador que más difiere es «{mayor}» "
                            f"({_n(float(tabla.loc[mayor, nivel_alto]))} frente a {_n(float(tabla.loc[mayor, nivel_bajo]))}, diferencia {float(dif[mayor]):+.3f})."),
            "como_leerlo": "Cada fila es un indicador (media o prevalencia ponderada con el factor de expansión) por nivel FIES; una diferencia grande y ordenada entre niveles señala asociación con la inseguridad.",
            "limitacion": "Son asociaciones descriptivas sin control de otras variables y no son causales. Los umbrales de las banderas de vulnerabilidad son ajustables y están marcados [POR VALIDAR]."}


def i_territorial(tabla: pd.DataFrame, unidad: str = "departamento", n_min: int = 30) -> Texto:
    if tabla is None or tabla.empty:
        return vacio()
    t = tabla.sort_values("prevalencia_ia", ascending=False)
    alto, bajo = t.iloc[0], t.iloc[-1]
    chicos = int((t["n_hogares"] < n_min).sum())
    aviso = f" {chicos} de {len(t)} unidades tienen menos de {n_min} hogares y su estimación es muy imprecisa." if chicos else ""
    return {"que_muestra": (f"Prevalencia ponderada de inseguridad moderada o severa (niveles 2-3) por {unidad}: máxima en «{alto[unidad]}» ({_pct(float(alto['prevalencia_ia']))}, "
                            f"IC aprox. [{_pct(float(alto['ic_inf']))}; {_pct(float(alto['ic_sup']))}], n={int(alto['n_hogares'])}) y mínima en «{bajo[unidad]}» ({_pct(float(bajo['prevalencia_ia']))}, n={int(bajo['n_hogares'])}).{aviso}"),
            "como_leerlo": "Barras o mapa más intensos indican mayor prevalencia; si los intervalos de dos unidades se solapan, la diferencia no es concluyente.",
            "limitacion": "El IC es aproximado (normal con tamaño efectivo de Kish) y no incorpora el diseño muestral completo (estratos/conglomerados). La muestra ENAHO no está diseñada para estimar todos los departamentos con la misma precisión."}


# ------------------------------ Informe de manuscrito (plantillas condicionales) ------------------------------
def _p(x: float | None) -> str:
    """p-valor legible: '<0,001' o tres decimales."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return "<0,001" if x < 0.001 else f"{x:.3f}".replace(".", ",")


def i_prevalencia(tabla: pd.DataFrame, unidad: str = "área", metodo: str = "") -> Texto:
    """Prevalencia ponderada de IA moderada/severa por dominio. Columnas: grupo, prevalencia, ic_inf, ic_sup, n."""
    if tabla is None or tabla.empty:
        return vacio(f"Sin datos disponibles para estimar la prevalencia por {unidad}.")
    t = tabla.dropna(subset=["prevalencia"])
    if t.empty:
        return vacio(f"Sin datos disponibles para estimar la prevalencia por {unidad}.")
    alto, bajo = t.loc[t["prevalencia"].idxmax()], t.loc[t["prevalencia"].idxmin()]
    solapa = bool(len(t) > 1 and alto["ic_inf"] <= bajo["ic_sup"])
    cmp_txt = (f" Los intervalos de «{alto['grupo']}» y «{bajo['grupo']}» se solapan: la diferencia observada no es concluyente."
               if solapa else f" Los intervalos de «{alto['grupo']}» y «{bajo['grupo']}» no se solapan, lo que sugiere una diferencia real entre ambos.")
    return {"que_muestra": (f"Prevalencia ponderada de inseguridad alimentaria moderada o severa (niveles FIES 2-3) por {unidad}: "
                            f"máxima en «{alto['grupo']}» ({_pct(float(alto['prevalencia']))}; IC95 {_pct(float(alto['ic_inf']))}-{_pct(float(alto['ic_sup']))}) "
                            f"y mínima en «{bajo['grupo']}» ({_pct(float(bajo['prevalencia']))}; IC95 {_pct(float(bajo['ic_inf']))}-{_pct(float(bajo['ic_sup']))})." + cmp_txt),
            "como_leerlo": "Cada barra es la prevalencia estimada con el factor de expansión; la línea es su IC95. Intervalos que no se solapan indican diferencias probablemente reales.",
            "limitacion": ("Dominios con pocos hogares dan intervalos amplios. " + (metodo or "IC aproximados que no incorporan la estratificación del diseño muestral."))}


def i_rendimiento(agregado: pd.DataFrame, metrica: str = "f1_macro", etiqueta: str = "F1-macro") -> Texto:
    """Resumen de desempeño en CV anidada. Columnas: modelo, metrica, media, ic_inf, ic_sup."""
    if agregado is None or agregado.empty:
        return vacio("Sin datos disponibles de desempeño.")
    d = agregado[agregado["metrica"] == metrica].dropna(subset=["media"]).sort_values("media", ascending=False)
    if d.empty:
        return vacio("Sin datos disponibles de desempeño.")
    mej = d.iloc[0]
    txt = f"Mejor {etiqueta} medio: {mej['modelo']} ({_n(float(mej['media']))}; IC95 {_n(float(mej['ic_inf']))}-{_n(float(mej['ic_sup']))})."
    if len(d) > 1:
        seg = d.iloc[1]
        txt += (f" El segundo es {seg['modelo']} ({_n(float(seg['media']))}); "
                + ("sus IC95 se solapan, por lo que no puede afirmarse que {0} supere a {1} solo con estos intervalos.".format(mej["modelo"], seg["modelo"])
                   if seg["ic_sup"] >= mej["ic_inf"] else "sus IC95 no se solapan."))
    return {"que_muestra": txt,
            "como_leerlo": "Cada punto es la media sobre los pliegues externos de la CV anidada y la barra su IC95 (t de Student). Valores más altos son mejores salvo Brier y ECE.",
            "limitacion": "Con 5 pliegues el IC es amplio; las medias son de validación cruzada dentro del 70 % de Desarrollo, no de la Prueba independiente."}


def i_escenarios(tabla: pd.DataFrame) -> Texto:
    """Compara escenarios. Columnas: escenario, modelo, media (F1-macro ponderado)."""
    if tabla is None or tabla.empty or tabla["escenario"].nunique() < 2:
        return vacio("Sin datos disponibles: se necesita más de un escenario entrenado (socio y fies/hibrido) para compararlos.")
    m = tabla.groupby("escenario")["media"].mean().sort_values(ascending=False)
    base = float(m.get("socio", float("nan")))
    otros = [e for e in m.index if e != "socio"]
    extra = ""
    if "socio" in m.index and otros:
        extra = " " + "; ".join(f"«{e}» supera a «socio» en {_n(float(m[e]) - base)} puntos de F1-macro" if m[e] > base else f"«{e}» no supera a «socio»" for e in otros) + "."
    return {"que_muestra": "F1-macro ponderado medio por escenario: " + "; ".join(f"{e} = {_n(float(v))}" for e, v in m.items()) + "." + extra,
            "como_leerlo": "«socio» usa solo variables socioeconómicas (predicción real). «fies» e «hibrido» incluyen los ítems con los que se define el objetivo.",
            "limitacion": "Los escenarios fies e hibrido reconstruyen el nivel FIES a partir de sus propios ítems (circularidad): su desempeño casi perfecto no mide capacidad predictiva."}


def lectura_pruebas(tabla: pd.DataFrame, n_pliegues: int | None = None, alfa: float = 0.05) -> list[str]:
    """Frases en lenguaje claro, condicionadas a los p-valores (ajustados por Holm cuando existen)."""
    if tabla is None or tabla.empty:
        return ["Sin datos disponibles: la comparación estadística requiere al menos dos modelos y, para DeLong y McNemar, haber evaluado la Prueba."]
    col = "p_ajustado" if "p_ajustado" in tabla.columns else "p_valor"
    frases: list[str] = []
    nombres = {"delong_severa": "DeLong (AUC de la clase IA severa)", "mcnemar": "McNemar ponderado (aciertos)",
               "wilcoxon_pliegues": "Wilcoxon pareado (F1 por pliegue)", "nemenyi": "Nemenyi (rangos de Friedman)"}
    fr = tabla[tabla["prueba"] == "friedman"]
    if not fr.empty:
        r = fr.iloc[0]
        frases.append((f"Friedman global: p = {_p(float(r['p_valor']))}. " + ("Hay evidencia de que al menos un modelo difiere de los demás." if r["p_valor"] < alfa
                       else "No hay evidencia suficiente de que los modelos difieran en rango medio.")))
    for prueba, nombre in nombres.items():
        t = tabla[tabla["prueba"] == prueba]
        if t.empty:
            continue
        sig = t[t[col] < alfa]
        if len(sig):
            lista = ", ".join(f"{r.comparacion} (p ajustado {_p(float(getattr(r, col)))})" for r in sig.itertuples())
            frases.append(f"{nombre}: diferencias significativas en {lista}.")
        else:
            minp = float(t[col].min())
            frases.append(f"{nombre}: ninguna de las {len(t)} comparaciones es significativa con {col} < {alfa} (menor p = {_p(minp)}); no hay evidencia suficiente de diferencia, lo que no equivale a probar igualdad.")
    if n_pliegues is not None and n_pliegues <= 5 and (tabla["prueba"] == "wilcoxon_pliegues").any():
        frases.append(f"Con {n_pliegues} pliegues, el p mínimo alcanzable de Wilcoxon bilateral es {_p(2 / 2 ** n_pliegues)}, de modo que esta prueba no puede resultar significativa a nivel {alfa}.")
    return frases


def i_equidad(tabla: pd.DataFrame, umbral_pp: float = 5.0) -> Texto:
    """Brechas TPR por grupo. Columnas: grupo_col, grupo, es_referencia, brecha_tpr, ic_inf, ic_sup, alerta, n."""
    if tabla is None or tabla.empty:
        return vacio("Sin datos disponibles para evaluar equidad (se requieren predicciones por hogar).")
    t = tabla[~tabla["es_referencia"]]
    al = t[t["alerta"]]
    if al.empty:
        txt = f"Ningún grupo supera {umbral_pp:.0f} pp de brecha de sensibilidad (TPR) frente a su grupo de referencia."
    else:
        txt = (f"{len(al)} de {len(t)} grupos superan {umbral_pp:.0f} pp de brecha de TPR: "
               + "; ".join(f"{r.grupo_col}={r.grupo} ({100 * r.brecha_tpr:+.1f} pp)" for r in al.head(6).itertuples()) + ("…" if len(al) > 6 else "") + ".")
        dudosas = al[(al["ic_inf"] <= 0) & (al["ic_sup"] >= 0)]
        if len(dudosas):
            txt += f" En {len(dudosas)} de ellos el IC95 incluye 0, de modo que la brecha podría deberse al azar muestral."
    return {"que_muestra": txt,
            "como_leerlo": "La brecha es TPR del grupo menos TPR del grupo de referencia (el más numeroso). El sombreado marca ±5 pp; puntos naranja son alertas; las líneas son IC95 bootstrap.",
            "limitacion": "Una brecha de desempeño no demuestra discriminación ni causalidad; grupos con pocos positivos tienen IC muy amplios. El bootstrap ignora estrato y conglomerado."}


def i_comparacion_escenarios_shap(imp_fies: float, imp_socio: float) -> Texto:
    """Participación de las variables FIES frente a las socioeconómicas en la importancia total (proporciones 0-1)."""
    return {"que_muestra": f"Las variables FIES concentran {_pct(imp_fies)} de la importancia |SHAP| total y las socioeconómicas {_pct(imp_socio)}.",
            "como_leerlo": "Si los ítems FIES dominan, el modelo está reconstruyendo el objetivo y no prediciéndolo.",
            "limitacion": "La importancia global promedia contribuciones positivas y negativas en valor absoluto; no indica el sentido del efecto."}


# ------------------------------ Inferencia: LIME, sensibilidad, contrafactual, otros modelos, validez ------------------------------
_NO_CAUSAL = "Es una asociación predictiva del modelo aprendida de otros hogares, no un efecto causal ni una recomendación de política."


def i_lime(tabla: pd.DataFrame, clase: int, r2: float, p_modelo: float, p_local: float) -> Texto:
    if tabla is None or tabla.empty:
        return vacio("No se pudo calcular LIME.")
    top = tabla.head(3)
    partes = ", ".join(f"«{r.regla}» ({r.peso:+.3f})" for r in top.itertuples())
    return {"que_muestra": (f"LIME aproxima, solo en torno a este hogar, cómo cambia la probabilidad del nivel {clase} ({ETIQUETA_NIVEL.get(clase, '')}). Las condiciones que más pesan son {partes}. "
                            f"La aproximación local reproduce P = {_pct(p_local)} frente a {_pct(p_modelo)} del modelo (R² local {_n(r2, 2)})."),
            "como_leerlo": "Un peso positivo (bermellón) indica que cumplir esa condición aumenta la probabilidad del nivel en la vecindad del hogar; uno negativo (azul), que la reduce.",
            "limitacion": ("LIME es una aproximación LOCAL con perturbaciones aleatorias (semilla fija): puede variar con otra semilla y un R² local bajo indica que el modelo lineal local explica poco. " + _NO_CAUSAL)}


def i_lime_shap(c: Mapping[str, Any]) -> Texto:
    if not c.get("disponible"):
        return vacio(c.get("motivo", "No hay comparación disponible."))
    k = c["k"]
    rho = c["spearman_abs"]
    rho_t = "no calculable" if rho is None or (isinstance(rho, float) and math.isnan(rho)) else f"{rho:.2f}"
    return {"que_muestra": (f"De las {k} variables principales de cada método, {c['coincidencia_top_k']} coinciden entre LIME y SHAP. Los signos coinciden en {c['signos_coinciden']} de {c['signos_total']} variables "
                            f"con efecto en ambos métodos ({c['signos_top_coinciden']} de {c['signos_top_total']} entre las variables principales de cualquiera de los dos). Correlación de rangos de |efecto| (Spearman): {rho_t}."),
            "como_leerlo": "Los dos métodos están en unidades distintas, por lo que se comparan signos y orden de importancia, no magnitudes (las barras están normalizadas por el mayor |efecto| de cada método).",
            "limitacion": ("Una coincidencia alta refuerza la lectura local; una baja indica que la explicación depende del método y debe tomarse con cautela. SHAP explica al modelo final con un fondo de 100 hogares y LIME "
                           "con perturbaciones propias; ninguno es causal.")}


def i_sensibilidad(curvas: pd.DataFrame, principales: list[str], p_base: float) -> Texto:
    if curvas is None or curvas.empty:
        return vacio("No hay curvas de sensibilidad.")
    rangos = curvas.groupby("variable")["p_mod_sev"].agg(lambda s: float(s.max() - s.min())).sort_values(ascending=False)
    v, r = rangos.index[0], rangos.iloc[0]
    return {"que_muestra": (f"Al variar una a una las {len(principales)} variables principales (el resto del hogar queda fijo), la probabilidad de inseguridad moderada o severa del hogar, hoy {_pct(p_base)}, "
                            f"se mueve más con «{_etq(v)}» (rango {100 * r:.1f} puntos porcentuales entre los valores explorados)."),
            "como_leerlo": "Cada curva muestra la probabilidad de cada nivel FIES para valores plausibles de la variable (percentiles 5 a 95 de Desarrollo; todas las categorías en las categóricas). El punto marca el valor actual del hogar.",
            "limitacion": ("Cambiar una sola variable ignora que muchas se mueven juntas (p. ej. ingreso, gasto y pobreza) y puede producir combinaciones poco realistas. " + _NO_CAUSAL)}


def i_tornado(tor: pd.DataFrame, p_base: float) -> Texto:
    if tor is None or tor.empty:
        return vacio("No hay tornado de sensibilidad.")
    a = tor.iloc[0]
    n_pos = int((tor["efecto (pp)"] > 0).sum())
    return {"que_muestra": (f"Mover cada variable entre su percentil 10 y su percentil 90 de Desarrollo cambia la P(moderada o severa) del hogar (hoy {_pct(p_base)}) entre {tor['efecto (pp)'].min():+.1f} y {tor['efecto (pp)'].max():+.1f} "
                            f"puntos porcentuales; la mayor variación es «{a['Variable']}» ({a['efecto (pp)']:+.1f} pp). {n_pos} de {len(tor)} variables aumentan la probabilidad al pasar del valor bajo al alto."),
            "como_leerlo": "Barra a la derecha: el valor alto de la variable aumenta la probabilidad respecto del valor bajo; a la izquierda, la reduce. Binarias: 0 frente a 1. Categóricas: de la categoría de menor a la de mayor probabilidad.",
            "limitacion": ("Es el efecto de mover UNA variable con el resto del hogar fijo, según el modelo; las categóricas con muchas categorías (departamento) muestran la máxima diferencia posible. " + _NO_CAUSAL)}


def i_contrafactual(cf: Mapping[str, Any], metricas_txt: str = "") -> Texto:
    if not cf.get("aplica", False):
        return vacio(cf.get("motivo", "El contrafactual no aplica a este modelo."))
    esc = cf["escenarios"]
    base = cf["p_mod_sev_base"]
    if cf.get("ya_cumple"):
        resumen = f"La P(moderada o severa) actual ({_pct(base)}) ya es menor que el objetivo ({_pct(cf['objetivo'])}); no hay nada que reducir según este criterio."
    elif not esc:
        resumen = (f"Ningún cambio plausible en las variables accionables reduce la P(moderada o severa) ({_pct(base)}) según el modelo: los valores actuales ya están en el extremo favorable explorado "
                   "o el modelo no responde a esas variables.")
    else:
        mejor = min(esc, key=lambda e: e["p_mod_sev"])
        resumen = (f"Objetivo: bajar P(moderada o severa) de {_pct(base)} a menos de {_pct(cf['objetivo'])}. Se encontraron {len(esc)} escenario(s); el de mayor reducción lleva la probabilidad a {_pct(mejor['p_mod_sev'])} "
                   f"({mejor['reduccion_pp']:+.1f} pp); el objetivo " + ("se alcanza." if cf["alcanzado"] else "NO se alcanza con cambios plausibles."))
    return {"que_muestra": resumen + (" " + metricas_txt if metricas_txt else ""),
            "como_leerlo": "Cada escenario cambia pocas variables accionables, dentro del rango plausible de Desarrollo, y recalcula la probabilidad con el MISMO ensamble. «Distancia» suma el cambio relativo al rango p10-p90 de cada variable (binarias: 1).",
            "limitacion": ("Esto es una asociación predictiva del modelo, NO un efecto causal ni una recomendación de política: que el modelo asigne menor probabilidad al hogar modificado no significa que hacer ese cambio reduzca la inseguridad alimentaria. "
                           "La lista de variables accionables es [POR VALIDAR] y las variables derivadas (gasto, pobreza) se mantienen fijas, por lo que el perfil modificado puede ser incoherente.")}


def i_otros_modelos(t: pd.DataFrame, seleccionado: str) -> Texto:
    if t is None or t.empty:
        return vacio("No hay otros modelos de la corrida para este escenario.")
    votos = t["Nivel predicho"].value_counts()
    return {"que_muestra": (f"Con el mismo hogar, {int(votos.iloc[0])} de {len(t)} modelos predicen el nivel {int(votos.index[0])} ({ETIQUETA_NIVEL.get(int(votos.index[0]), '')}); "
                            f"la P(moderada o severa) va de {_pct(float(t['P(mod.+sev.)'].min()))} a {_pct(float(t['P(mod.+sev.)'].max()))} según el modelo."),
            "como_leerlo": f"La fila marcada con ★ es el modelo seleccionado ({seleccionado}). Si los modelos discrepan en el nivel o en la probabilidad, la predicción de este hogar es menos estable.",
            "limitacion": "Los modelos comparten datos de entrenamiento y variables: que coincidan no demuestra que acierten. Cada modelo promedia su propio ensamble de 6 miembros."}


def i_validez(t: pd.DataFrame) -> Texto:
    if t is None or t.empty:
        return vacio("No hay revisión de validez de entradas.")
    fuera = t[t["Estado"].str.startswith("fuera")]
    ext = t[t["Estado"].str.startswith("extremo")]
    if len(fuera) or len(ext):
        txt = (f"{len(fuera)} variable(s) fuera del rango observado en Desarrollo y {len(ext)} en el extremo (fuera de P1-P99): "
               + ", ".join(f"«{v}»" for v in list(fuera["Variable"]) + list(ext["Variable"])) + ". ADVERTENCIA DE EXTRAPOLACIÓN: el modelo casi no vio hogares así y su predicción es menos confiable.")
    else:
        txt = "Todas las entradas están dentro del rango observado en Desarrollo (entre P1 y P99): no hay extrapolación evidente."
    return {"que_muestra": txt,
            "como_leerlo": "Se compara cada valor con el mínimo, el máximo y los percentiles 1 y 99 (ponderados) de Desarrollo; en las categóricas, con la frecuencia de la categoría.",
            "limitacion": "Estar dentro del rango de cada variable por separado no garantiza que la combinación de valores sea frecuente."}
