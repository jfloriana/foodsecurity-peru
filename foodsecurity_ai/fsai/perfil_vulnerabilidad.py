"""perfil_vulnerabilidad.py - Perfil DESCRIPTIVO de hogares vulnerables (prevalencia ponderada de inseguridad alimentaria por subgrupo).

Alcance y límites (se repiten en la app, los reportes y la inferencia):
  * Es una descripción, NO un índice: no hay puntaje, pesos ni calibración, y no está validado como índice de vulnerabilidad.
  * El recuento 0-3 es un conteo simple de características presentes; cada característica pesa lo mismo por construcción.
  * No hay desenlaces nutricionales directos (talla, peso, anemia, consumo de nutrientes): la vulnerabilidad nutricional se
    aborda solo como señal indirecta mediante la inseguridad alimentaria (FIES nivel 2-3).
  * Las comparaciones son asociativas; no establecen causalidad.

Características (mismas definiciones que usa la sección 3.1 del artículo, sobre el mismo conjunto analítico):
  menor5        n_menores5 > 0               (miembros con edad < 5 años, módulo 200 ENAHO, variable P208A)
  adulto_mayor  n_adultos_mayores > 0        (miembros con edad >= 65 años)
  pobreza       pobreza_monetaria == 1       (variable POBREZA de la Sumaria: categorías 1 y 2 = 1; categoría 3 = 0)

Medida: inseguridad alimentaria moderada o severa = ``fies_nivel`` >= 2, ponderada con ``factor_expansion``.
IC95: linealización de Taylor con los conglomerados (``conglome``) como unidades primarias de muestreo, SIN estratificación
(aproximado; igual que ``reportes_fases.prevalencia_ic``). Sin pruebas de hipótesis.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

COL_OBJ = "fies_nivel"
COL_PESO = "factor_expansion"
COL_CONGLO = "conglome"
UMBRAL_MENOR = 5          # años (estrictamente menor)
UMBRAL_ADULTO_MAYOR = 65  # años (mayor o igual)
N_MIN = 30
NIVELES_FIES = {0: "Seguridad", 1: "Inseguridad leve", 2: "Inseguridad moderada", 3: "Inseguridad severa"}

ALCANCE = ("Perfil descriptivo: no es un índice ponderado ni validado de vulnerabilidad (no hay puntaje ni pesos; el recuento es un conteo simple). "
           "No incluye desenlaces nutricionales directos: la vulnerabilidad nutricional se aborda solo mediante la inseguridad alimentaria (FIES) como señal indirecta. "
           "Las comparaciones son asociativas y no establecen causalidad.")
METODO_IC = ("IC95 por linealización de Taylor con los conglomerados como unidades primarias de muestreo, sin incorporar la estratificación del diseño "
             "(aproximado); estimaciones ponderadas con el factor de expansión; sin pruebas de hipótesis.")

# clave -> (etiqueta, etiqueta corta, columna origen, regla legible, estado de validación)
CARACTERISTICAS: dict[str, dict[str, str]] = {
    "menor5": {
        "etiqueta": "Hogar con al menos un menor de 5 años", "corta": "Menor de 5 años", "columna": "n_menores5",
        "regla": f"n_menores5 > 0 (miembros con edad < {UMBRAL_MENOR} años)",
        "estado": "Regla verificada en el código (fuentes_publicas.py) y coincide con el artículo; el umbral de 5 años es convencional [POR VALIDAR] frente al protocolo."},
    "adulto_mayor": {
        "etiqueta": "Hogar con al menos un adulto mayor (65 años o más)", "corta": "Adulto mayor (65+)", "columna": "n_adultos_mayores",
        "regla": f"n_adultos_mayores > 0 (miembros con edad >= {UMBRAL_ADULTO_MAYOR} años)",
        "estado": "Regla verificada en el código (fuentes_publicas.py) y coincide con el artículo; la adopción de 65 años como umbral de adulto mayor es convencional [POR VALIDAR] frente al protocolo."},
    "pobreza": {
        "etiqueta": "Hogar en pobreza monetaria", "corta": "Pobreza monetaria", "columna": "pobreza_monetaria",
        "regla": "pobreza_monetaria == 1 (variable POBREZA de la Sumaria: categorías 1 y 2 = pobre; categoría 3 = no pobre)",
        "estado": "Regla verificada en el código (fuentes_publicas.py) y coincide con el artículo; el significado de las categorías de POBREZA debe contrastarse con el diccionario oficial del INEI [POR VALIDAR]."},
}
CLAVES = list(CARACTERISTICAS)


# ------------------------------ definiciones ------------------------------
def definiciones() -> pd.DataFrame:
    """Tabla de definiciones (trazabilidad a variables reales del dataset)."""
    return pd.DataFrame([{"característica": v["etiqueta"], "variable (dataset)": v["columna"], "regla": v["regla"], "validación": v["estado"]}
                         for v in CARACTERISTICAS.values()])


def indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """Indicadores 0/1 (Int64; NA si falta el dato de origen) con las columnas ``menor5``, ``adulto_mayor`` y ``pobreza``."""
    out = pd.DataFrame(index=df.index)
    for k, v in CARACTERISTICAS.items():
        c = v["columna"]
        if c not in df.columns:
            out[k] = pd.array([pd.NA] * len(df), dtype="Int64")
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        out[k] = (s > 0).astype("Int64").where(s.notna())
    return out


# ------------------------------ estimación ------------------------------
def prevalencia_ic(d: pd.DataFrame) -> tuple[float, float, float, int]:
    """Prevalencia ponderada de IA moderada/severa (fies_nivel >= 2) con IC95 por linealización con conglomerados. (p, inf, sup, n)."""
    d = d[d[COL_OBJ].notna() & d[COL_PESO].notna()]
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    w = d[COL_PESO].to_numpy(float)
    y = (d[COL_OBJ].to_numpy(float) >= 2).astype(float)
    sw = w.sum()
    p = float((w * y).sum() / sw)
    z = w * (y - p) / sw
    if COL_CONGLO in d.columns and d[COL_CONGLO].nunique() > 1:
        zc = pd.Series(z).groupby(d[COL_CONGLO].to_numpy()).sum().to_numpy()
        m = len(zc)
        var = m / (m - 1) * float((zc ** 2).sum())
    else:
        var = n / (n - 1) * float((z ** 2).sum()) if n > 1 else float("nan")
    se = float(np.sqrt(var)) if var == var else float("nan")
    return p, max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se), n


def _base(df: pd.DataFrame) -> pd.DataFrame:
    """Hogares con FIES y factor de expansión, más los indicadores y el recuento (NA si falta alguna característica)."""
    d = df[df[COL_OBJ].notna() & df[COL_PESO].notna()].copy()
    ind = indicadores(d)
    for k in CLAVES:
        d[f"_{k}"] = ind[k]
    cols = [f"_{k}" for k in CLAVES]
    completos = d[cols].notna().all(axis=1)
    d["_recuento"] = d[cols].sum(axis=1, min_count=len(cols)).where(completos)
    return d


def _nota(n: int) -> str:
    return f"n < {N_MIN}: inestable" if n < N_MIN else ""


def tabla_subgrupos(df: pd.DataFrame) -> pd.DataFrame:
    """Prevalencia ponderada por característica, con y sin ella. Columnas: característica, grupo, n, % de hogares (ponderado), prevalencia, ic_inf, ic_sup, nota."""
    d = _base(df)
    wt = d[COL_PESO].sum()
    filas = []
    for k in CLAVES:
        for val, etq in ((1, "Con la característica"), (0, "Sin la característica")):
            s = d[d[f"_{k}"] == val]
            p, lo, hi, n = prevalencia_ic(s)
            filas.append({"clave": k, "característica": CARACTERISTICAS[k]["corta"], "grupo": etq, "n": n,
                          "pct_hogares": float(s[COL_PESO].sum() / wt) if wt else float("nan"),
                          "prevalencia": p, "ic_inf": lo, "ic_sup": hi, "nota": _nota(n)})
    return pd.DataFrame(filas)


def tabla_diferencias(sub: pd.DataFrame) -> pd.DataFrame:
    """Diferencia de prevalencia (con − sin) en puntos porcentuales y si los IC95 se solapan (lectura descriptiva, sin prueba)."""
    filas = []
    for k in CLAVES:
        con = sub[(sub["clave"] == k) & (sub["grupo"].str.startswith("Con"))].iloc[0]
        sin = sub[(sub["clave"] == k) & (sub["grupo"].str.startswith("Sin"))].iloc[0]
        solapa = bool(con["ic_inf"] <= sin["ic_sup"] and sin["ic_inf"] <= con["ic_sup"])
        filas.append({"característica": CARACTERISTICAS[k]["corta"], "prevalencia con": con["prevalencia"], "prevalencia sin": sin["prevalencia"],
                      "diferencia (pp)": 100 * (con["prevalencia"] - sin["prevalencia"]), "IC95 solapados": "sí" if solapa else "no"})
    return pd.DataFrame(filas)


def tabla_recuento(df: pd.DataFrame) -> pd.DataFrame:
    """Prevalencia por recuento simple 0/1/2/3 de características. Columnas: recuento, n, pct_hogares, prevalencia, ic_inf, ic_sup, nota."""
    d = _base(df)
    d = d[d["_recuento"].notna()]
    wt = d[COL_PESO].sum()
    filas = []
    for r in range(len(CLAVES) + 1):
        s = d[d["_recuento"] == r]
        p, lo, hi, n = prevalencia_ic(s)
        filas.append({"recuento": r, "n": n, "pct_hogares": float(s[COL_PESO].sum() / wt) if wt and n else 0.0,
                      "prevalencia": p, "ic_inf": lo, "ic_sup": hi, "nota": _nota(n) if n else "sin hogares"})
    return pd.DataFrame(filas)


def _dist_fies(s: pd.DataFrame) -> dict[str, float]:
    w = s[COL_PESO].sum()
    return {NIVELES_FIES[i]: (float(s.loc[s[COL_OBJ] == i, COL_PESO].sum() / w) if w else float("nan")) for i in NIVELES_FIES}


def tabla_fies_por_recuento(df: pd.DataFrame) -> pd.DataFrame:
    """Distribución ponderada (fila = 100 %) del nivel FIES dentro de cada recuento. Columnas: recuento, n, un nivel por columna."""
    d = _base(df)
    d = d[d["_recuento"].notna()]
    return pd.DataFrame([{"recuento": r, "n": int((d["_recuento"] == r).sum()), **_dist_fies(d[d["_recuento"] == r])} for r in range(len(CLAVES) + 1)])


def tabla_fies_por_caracteristica(df: pd.DataFrame) -> pd.DataFrame:
    """Distribución ponderada del nivel FIES con y sin cada característica. Columnas: característica, grupo, n, un nivel por columna."""
    d = _base(df)
    filas = []
    for k in CLAVES:
        for val, etq in ((1, "Con la característica"), (0, "Sin la característica")):
            s = d[d[f"_{k}"] == val]
            filas.append({"característica": CARACTERISTICAS[k]["corta"], "grupo": etq, "n": len(s), **_dist_fies(s)})
    return pd.DataFrame(filas)


def perfil(df: pd.DataFrame) -> dict[str, Any]:
    """Todas las tablas del perfil, más el total y los hogares excluidos por dato faltante en alguna característica."""
    d = _base(df)
    p, lo, hi, n = prevalencia_ic(d)
    sub = tabla_subgrupos(df)
    rec = tabla_recuento(df)
    return {"total": {"n": n, "prevalencia": p, "ic_inf": lo, "ic_sup": hi},
            "n_con_fies": n, "n_recuento": int(rec["n"].sum()), "n_excluidos_recuento": n - int(rec["n"].sum()),
            "definiciones": definiciones(), "subgrupos": sub, "diferencias": tabla_diferencias(sub), "recuento": rec,
            "fies_recuento": tabla_fies_por_recuento(df), "fies_caracteristica": tabla_fies_por_caracteristica(df)}


# ------------------------------ un hogar ------------------------------
def pertenencia_hogar(registro: Mapping[str, Any]) -> dict[str, Any]:
    """Pertenencia de UN hogar a los grupos del perfil. ``registro``: valores crudos (n_menores5, n_adultos_mayores, pobreza_monetaria).

    Devuelve {'claves': [...presentes], 'desconocidas': [...sin dato], 'recuento': int | None (None si falta algún dato)}.
    """
    d = pd.DataFrame([dict(registro)])
    ind = indicadores(d).iloc[0]
    presentes = [k for k in CLAVES if pd.notna(ind[k]) and int(ind[k]) == 1]
    desconocidas = [k for k in CLAVES if pd.isna(ind[k])]
    return {"claves": presentes, "desconocidas": desconocidas, "recuento": None if desconocidas else len(presentes)}


def contexto_hogar(registro: Mapping[str, Any], pf: Mapping[str, Any] | None) -> dict[str, Any]:
    """Pertenencia + prevalencia observada de cada grupo (contexto descriptivo del conjunto analítico; NO es una predicción del hogar)."""
    pert = pertenencia_hogar(registro)
    out: dict[str, Any] = {**pert, "grupos": [], "recuento_fila": None, "disponible": pf is not None}
    if pf is None:
        return out
    sub, rec = pf["subgrupos"], pf["recuento"]
    for k in pert["claves"]:
        f = sub[(sub["clave"] == k) & (sub["grupo"].str.startswith("Con"))]
        if len(f):
            f = f.iloc[0]
            out["grupos"].append({"clave": k, "etiqueta": CARACTERISTICAS[k]["etiqueta"], "n": int(f["n"]), "prevalencia": float(f["prevalencia"]),
                                  "ic_inf": float(f["ic_inf"]), "ic_sup": float(f["ic_sup"])})
    if pert["recuento"] is not None:
        f = rec[rec["recuento"] == pert["recuento"]]
        if len(f) and int(f.iloc[0]["n"]) > 0:
            f = f.iloc[0]
            out["recuento_fila"] = {"recuento": int(pert["recuento"]), "n": int(f["n"]), "prevalencia": float(f["prevalencia"]),
                                    "ic_inf": float(f["ic_inf"]), "ic_sup": float(f["ic_sup"])}
    return out


def texto_contexto_hogar(ctx: Mapping[str, Any]) -> list[str]:
    """Frases listas para la página y los reportes de inferencia (una por línea)."""
    pct = lambda v: f"{100 * v:.1f} %".replace(".", ",")  # noqa: E731
    if not ctx.get("disponible"):
        return ["Perfil de vulnerabilidad: no se pudo calcular el contexto del conjunto de datos activo; se muestra solo la pertenencia a los grupos."]
    if ctx["desconocidas"]:
        faltan = ", ".join(CARACTERISTICAS[k]["corta"] for k in ctx["desconocidas"])
        pref = f"Sin dato para: {faltan}; el recuento no es calculable."
    else:
        pref = ""
    if ctx["claves"]:
        grupos = "; ".join(f"{g['etiqueta']} (prevalencia observada {pct(g['prevalencia'])}, IC95 {pct(g['ic_inf'])}-{pct(g['ic_sup'])}, n = {g['n']})" for g in ctx["grupos"])
        lin = [f"Grupos del perfil a los que pertenece este hogar: {grupos}."]
    else:
        lin = ["Este hogar no pertenece a ninguno de los tres grupos del perfil (menor de 5 años, adulto mayor, pobreza monetaria)." if not ctx["desconocidas"]
               else "No se identificó pertenencia a los grupos con los datos disponibles."]
    if pref:
        lin.append(pref)
    rf = ctx.get("recuento_fila")
    if rf:
        lin.append(f"Recuento de características: {rf['recuento']} de {len(CLAVES)}; prevalencia observada en hogares con ese recuento {pct(rf['prevalencia'])} "
                   f"(IC95 {pct(rf['ic_inf'])}-{pct(rf['ic_sup'])}, n = {rf['n']}).")
    lin.append("La prevalencia del grupo es un contexto descriptivo del conjunto analítico (no es la probabilidad de este hogar ni una predicción); el perfil no es un índice validado.")
    return lin


# ------------------------------ interpretación (qué muestra / cómo leerlo / limitación) ------------------------------
def _pc(v: float) -> str:
    return f"{100 * v:.1f} %".replace(".", ",")


def i_subgrupos(sub: pd.DataFrame, dif: pd.DataFrame) -> dict[str, str]:
    if sub is None or sub.empty:
        return {"que_muestra": "No hay hogares con los datos necesarios para el perfil de subgrupos.", "como_leerlo": "", "limitacion": ""}
    top = dif.sort_values("diferencia (pp)", ascending=False).iloc[0]
    dpp = f"{top['diferencia (pp)']:+.1f}".replace(".", ",")
    return {"que_muestra": (f"La mayor diferencia de prevalencia entre hogares con y sin la característica se observa en «{top['característica']}» "
                            f"({_pc(top['prevalencia con'])} frente a {_pc(top['prevalencia sin'])}; {dpp} pp; IC95 {'solapados' if top['IC95 solapados'] == 'sí' else 'no solapados'})."),
            "como_leerlo": "Cada barra es la prevalencia ponderada de inseguridad alimentaria moderada o severa (FIES nivel 2-3) del subgrupo; la línea es su IC95. Intervalos que se solapan no permiten afirmar una diferencia.",
            "limitacion": ALCANCE + " " + METODO_IC}


def i_recuento(rec: pd.DataFrame) -> dict[str, str]:
    r = rec[rec["n"] > 0]
    if r.empty:
        return {"que_muestra": "No hay hogares con los datos necesarios para el recuento de características.", "como_leerlo": "", "limitacion": ""}
    a, b = r.iloc[0], r.iloc[-1]
    ordenado = bool(r["prevalencia"].is_monotonic_increasing)
    chicos = [int(x) for x in r.loc[r["n"] < N_MIN, "recuento"]]
    aviso = f" Los recuentos {', '.join(map(str, chicos))} tienen menos de {N_MIN} hogares y su estimación es inestable." if chicos else ""
    na, nb = f"{int(a['n']):,}".replace(",", " "), f"{int(b['n']):,}".replace(",", " ")
    return {"que_muestra": (f"La prevalencia pasa de {_pc(a['prevalencia'])} con recuento {int(a['recuento'])} (n = {na}) a {_pc(b['prevalencia'])} con recuento {int(b['recuento'])} (n = {nb}); "
                            + ("el aumento es continuo entre recuentos." if ordenado else "el aumento no es continuo entre recuentos.") + aviso),
            "como_leerlo": "El recuento es la cantidad de características presentes (menor de 5 años, adulto mayor, pobreza monetaria); todas pesan igual por construcción. El gradiente describe asociación, no un efecto acumulativo.",
            "limitacion": ALCANCE + " " + METODO_IC}


def i_fies(fr: pd.DataFrame) -> dict[str, str]:
    if fr is None or fr.empty:
        return {"que_muestra": "No hay hogares con los datos necesarios para el cruce con FIES.", "como_leerlo": "", "limitacion": ""}
    r = fr[fr["n"] > 0]
    sev = NIVELES_FIES[3]
    a, b = r.iloc[0], r.iloc[-1]
    return {"que_muestra": (f"La proporción ponderada de inseguridad severa es {_pc(a[sev])} con recuento {int(a['recuento'])} y {_pc(b[sev])} con recuento {int(b['recuento'])}."),
            "como_leerlo": "Cada fila suma 100 %: reparte los hogares con ese recuento entre los cuatro niveles FIES.",
            "limitacion": ALCANCE + " Proporciones puntuales ponderadas, sin intervalos."}


# ------------------------------ tablas para la inferencia (individual y lote) ------------------------------
def tabla_contexto_hogar(ctx: Mapping[str, Any]) -> pd.DataFrame:
    """Tabla del hogar evaluado: pertenencia a cada grupo y prevalencia observada del grupo (contexto, no predicción)."""
    pc = lambda v: f"{100 * v:.1f} %".replace(".", ",")  # noqa: E731
    por_clave = {g["clave"]: g for g in ctx.get("grupos", [])}
    filas = []
    for k in CLAVES:
        g = por_clave.get(k)
        estado = "Sin dato" if k in ctx.get("desconocidas", []) else ("Sí" if k in ctx.get("claves", []) else "No")
        filas.append({"Grupo del perfil": CARACTERISTICAS[k]["etiqueta"], "¿Pertenece el hogar?": estado,
                      "n del grupo (conjunto analítico)": f"{g['n']:,}".replace(",", " ") if g else "—",
                      "Prevalencia observada del grupo": pc(g["prevalencia"]) if g else "—",
                      "IC95": f"{pc(g['ic_inf'])} a {pc(g['ic_sup'])}" if g else "—"})
    rf = ctx.get("recuento_fila")
    filas.append({"Grupo del perfil": f"Recuento de características ({len(CLAVES)} posibles)", "¿Pertenece el hogar?": f"Recuento {rf['recuento']}" if rf else "No calculable",
                  "n del grupo (conjunto analítico)": f"{rf['n']:,}".replace(",", " ") if rf else "—",
                  "Prevalencia observada del grupo": pc(rf["prevalencia"]) if rf else "—", "IC95": f"{pc(rf['ic_inf'])} a {pc(rf['ic_sup'])}" if rf else "—"})
    return pd.DataFrame(filas)


def perfil_lote(dfl: pd.DataFrame, pf: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Distribución de los hogares de un lote por característica y por recuento 0-3, con la prevalencia observada de referencia.

    Devuelve None si el CSV no trae ninguna de las columnas requeridas (n_menores5, n_adultos_mayores, pobreza_monetaria).
    """
    cols = [CARACTERISTICAS[k]["columna"] for k in CLAVES]
    if not any(c in dfl.columns for c in cols):
        return None
    ind = indicadores(dfl)
    n = len(ind)
    pc = lambda v: f"{100 * v:.1f} %".replace(".", ",")  # noqa: E731
    por_car = pd.DataFrame([{"Característica": CARACTERISTICAS[k]["etiqueta"], "Hogares del lote con la característica": int((ind[k] == 1).sum()),
                             "% del lote": pc((ind[k] == 1).sum() / n) if n else "—", "Hogares sin dato": int(ind[k].isna().sum())} for k in CLAVES])
    completos = ind.notna().all(axis=1)
    rec = ind.sum(axis=1, min_count=len(CLAVES)).where(completos)
    ref = pf["recuento"].set_index("recuento") if pf is not None else None
    filas = []
    for r in range(len(CLAVES) + 1):
        cnt = int((rec == r).sum())
        fila = {"Recuento": r, "Hogares del lote": cnt, "% del lote": pc(cnt / n) if n else "—"}
        if ref is not None and r in ref.index and ref.loc[r, "n"] > 0:
            fila["Prevalencia observada en el conjunto de referencia"] = pc(ref.loc[r, "prevalencia"])
            fila["n de referencia"] = f"{int(ref.loc[r, 'n']):,}".replace(",", " ")
        else:
            fila["Prevalencia observada en el conjunto de referencia"] = "—"
            fila["n de referencia"] = "—"
        filas.append(fila)
    return {"por_caracteristica": por_car, "por_recuento": pd.DataFrame(filas), "n": n, "n_sin_recuento": int((~completos).sum())}


# ------------------------------ presentación (texto con coma decimal, para tablas de reportes) ------------------------------
def formatear_tabla(df: pd.DataFrame, pct: tuple[str, ...] = (), enteros: tuple[str, ...] = (), pp: tuple[str, ...] = ()) -> pd.DataFrame:
    """Copia de ``df`` con columnas de proporción como «30,0 %», conteos con espacio de miles y diferencias como «+3,7»; solo para mostrar."""
    out = df.copy()
    for c in pct:
        out[c] = [("" if pd.isna(v) else f"{100 * float(v):.1f} %".replace(".", ",")) for v in out[c]]
    for c in enteros:
        out[c] = [("" if pd.isna(v) else f"{int(v):,}".replace(",", " ")) for v in out[c]]
    for c in pp:
        out[c] = [("" if pd.isna(v) else f"{float(v):+.1f}".replace(".", ",")) for v in out[c]]
    return out
