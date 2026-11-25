"""verificar_informe.py - Verificación independiente de un informe generado (Excel / Word / PDF).

Comprueba que los conteos por grupo de la hoja «6_Equidad» del Excel coinciden con un recuento INDEPENDIENTE hecho directamente sobre
datos_publicos/procesado/enaho_analitico.parquet para el mismo conjunto de hogares (los de las predicciones del modelo evaluado en el informe),
para TODOS los atributos de equidad (sexo, área, región, edad, educación, pobreza). Los grupos de edad y educación se recalculan aquí con
np.select a partir de edad_jefe / nivel_educ_jefe (contrato: edad <30, 30-44, 45-59, 60+; educación 0-1, 2-3, 4-5, >=6), sin reutilizar código
del generador. También exige que el Word/Excel (y el PDF si pypdf está instalado) no contengan «DEMO», «sintético» ni «Sin datos disponibles».

Uso:  python _con_env.py ../protocolo/codigo_verificacion_app/verificar_informe.py <carpeta_del_informe> [corrida] [escenario]
      (o importar ``verificar_informe(carpeta, dir_run, escenario)``, que devuelve la lista de fallos).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROYECTO = Path(__file__).resolve().parents[2]
PARQUET = PROYECTO / "datos_publicos" / "procesado" / "enaho_analitico.parquet"
PROHIBIDO = re.compile(r"\bDEMO\b|sint[eé]tic|Sin datos disponibles|INV-?13", re.IGNORECASE)
# prefijo de la etiqueta de la hoja -> columna de grupo en el parquet
VARIABLES = {"sexo": "sexo_jefe", "área": "area", "area": "area", "región": "region_natural", "region": "region_natural",
             "grupo de edad": "grupo_edad_jefe", "grupo educativo": "grupo_educ_jefe", "pobreza": "pobreza"}


def _grupos_independientes(p: pd.DataFrame) -> pd.DataFrame:
    e, n = p["edad_jefe"], p["nivel_educ_jefe"]
    p = p.copy()
    for g in ("grupo_edad_jefe", "grupo_educ_jefe"):
        if g in p.columns:
            p["almacenado_" + g] = p[g]
    p["grupo_edad_jefe"] = np.select([e.isna(), e < 30, e < 45, e < 60], [None, "<30", "30-44", "45-59"], "60+")
    p["grupo_educ_jefe"] = np.select([n.isna(), n < 2, n < 4, n < 6], [None, "sin_primaria", "primaria", "secundaria"], "superior")
    p["pobreza"] = p["pobreza_monetaria"].map({1: "pobre", 0: "no_pobre", 1.0: "pobre", 0.0: "no_pobre"})
    return p


def _tabla_equidad_xlsx(xlsx: Path) -> tuple[pd.DataFrame, str]:
    import openpyxl
    ws = openpyxl.load_workbook(xlsx, data_only=True)["6_Equidad"]
    filas, cab, texto = [], None, ""
    for r in ws.iter_rows(values_only=True):
        v = [c for c in r if c is not None]
        if not v:
            continue
        if isinstance(v[0], str) and v[0].startswith("Modelo evaluado"):
            texto = v[0]
        if v[:3] == ["variable", "grupo", "n"]:
            cab = v
            continue
        if cab and len(v) >= 3 and isinstance(v[2], (int, float)) and isinstance(v[0], str):
            filas.append({"variable": v[0], "grupo": str(v[1]), "n": int(v[2])})
    return pd.DataFrame(filas), texto


def _columna(etiqueta: str) -> str | None:
    e = etiqueta.strip().lower()
    return next((c for k, c in VARIABLES.items() if e.startswith(k)), None)


def verificar_equidad(xlsx: Path, dir_run: Path, escenario: str) -> list[str]:
    fallos: list[str] = []
    t, texto = _tabla_equidad_xlsx(xlsx)
    if t.empty:
        return [f"{xlsx.name}: la hoja 6_Equidad no tiene tabla de grupos"]
    m = re.search(r"Modelo evaluado: (\w+).*?predicciones de (Prueba|Validaci.n)", texto)
    if not m:
        return [f"{xlsx.name}: no se pudo leer el modelo/fuente de la hoja 6_Equidad"]
    cod, fuente = m.group(1), m.group(2)
    f = dir_run / (f"pred_prueba_{escenario}_{cod}.npz" if fuente == "Prueba" else f"pred_val_{escenario}_{cod}.npz")
    z = np.load(f, allow_pickle=False)
    uid = pd.Index(z["uid"].astype(str))
    pq = pd.read_parquet(PARQUET)
    pq["hogar_uid"] = pq["hogar_uid"].astype(str)
    pq = _grupos_independientes(pq.drop_duplicates("hogar_uid").set_index("hogar_uid"))
    ok = uid.isin(pq.index)
    if int(ok.sum()) != len(uid):
        fallos.append(f"{int(len(uid) - ok.sum())} hogares de las predicciones no están en el parquet")
    d = pq.loc[uid[ok]]
    # el parquet tiene los grupos almacenados: deben coincidir con el recálculo independiente
    for g in ("grupo_edad_jefe", "grupo_educ_jefe"):
        a = "almacenado_" + g
        if a in pq.columns and (pq[g].fillna("x").astype(str) != pq[a].fillna("x").astype(str)).any():
            fallos.append(f"{g}: el recálculo independiente difiere del grupo almacenado en el parquet")
    t["col"] = t["variable"].map(_columna)
    if t["col"].isna().any():
        fallos.append(f"variables de la hoja sin mapear: {sorted(t[t['col'].isna()]['variable'].unique())}")
    columnas_vistas = set()
    for col, sub in t.dropna(subset=["col"]).groupby("col"):
        columnas_vistas.add(col)
        esperado = d[col].dropna().astype(str).value_counts()
        informe = sub["n"].tolist()
        if sorted(informe) != sorted(esperado.tolist()):
            fallos.append(f"{col}: conteos del informe {sorted(informe)} != recuento independiente {sorted(esperado.tolist())}")
        # cuando la hoja usa el código crudo del grupo (edad, educación, pobreza) se compara también por etiqueta
        crudo = {str(k): int(v) for k, v in esperado.items()}
        if set(sub["grupo"]) <= set(crudo):
            if {r.grupo: r.n for r in sub.itertuples()} != crudo:
                fallos.append(f"{col}: conteos por etiqueta {dict(zip(sub['grupo'], sub['n']))} != {crudo}")
        if int(sub["n"].sum()) != int(d[col].notna().sum()):
            fallos.append(f"{col}: la suma de n ({int(sub['n'].sum())}) != hogares con dato ({int(d[col].notna().sum())})")
    faltan = {"sexo_jefe", "area", "region_natural", "grupo_edad_jefe", "grupo_educ_jefe", "pobreza"} - columnas_vistas
    if faltan:
        fallos.append(f"atributos de equidad ausentes en el informe: {sorted(faltan)}")
    print(f"  equidad: modelo {cod} ({fuente}), {len(uid)} hogares; atributos verificados: {sorted(columnas_vistas)}")
    return fallos


def verificar_textos(carpeta: Path) -> list[str]:
    fallos: list[str] = []
    for x in sorted(carpeta.glob("*.xlsx")):
        import openpyxl
        wb = openpyxl.load_workbook(x, data_only=True)
        for ws in wb.worksheets:
            for r in ws.iter_rows(values_only=True):
                for c in r:
                    if isinstance(c, str) and PROHIBIDO.search(c):
                        fallos.append(f"{x.name}/{ws.title}: texto prohibido «{PROHIBIDO.search(c).group(0)}»")
    for w in sorted(carpeta.glob("*.docx")):
        import docx
        doc = docx.Document(str(w))
        partes = [p.text for p in doc.paragraphs]
        for tb in doc.tables:
            partes += [c.text for fila in tb.rows for c in fila.cells]
        for s in partes:
            if PROHIBIDO.search(s):
                fallos.append(f"{w.name}: texto prohibido «{PROHIBIDO.search(s).group(0)}»")
    for p in sorted(carpeta.glob("*.pdf")):
        try:
            import fitz   # PyMuPDF
            txt = " ".join(pg.get_text() for pg in fitz.open(str(p)))
        except ImportError:
            try:
                from pypdf import PdfReader
            except ImportError:
                print(f"  AVISO: ni PyMuPDF ni pypdf instalados; {p.name} no se pudo leer")
                continue
            txt = " ".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
        if PROHIBIDO.search(txt):
            fallos.append(f"{p.name}: texto prohibido «{PROHIBIDO.search(txt).group(0)}»")
    return fallos


COL_TABLA = 13   # las tablas de datos de las hojas de figuras empiezan en la columna M (fsai.excel_graficos.COL_TABLA)


def _tabla_hoja(ws, titulo: str, col: int = COL_TABLA) -> pd.DataFrame | None:
    """Lee la tabla cuyo título (celda de la columna ``col``) empieza por ``titulo``; el encabezado es la fila siguiente."""
    for r in range(1, ws.max_row + 1):
        v = ws.cell(r, col).value
        if isinstance(v, str) and v.startswith(titulo):
            cab, j = [], col
            while ws.cell(r + 1, j).value is not None:
                cab.append(str(ws.cell(r + 1, j).value))
                j += 1
            filas, k = [], r + 2
            while any(ws.cell(k, col + i).value is not None for i in range(len(cab))):   # curvas de distinta longitud: fila vacía = fin
                filas.append([ws.cell(k, col + i).value for i in range(len(cab))])
                k += 1
            return pd.DataFrame(filas, columns=cab)
    return None


def _cerca(a, b, tol: float = 1e-9) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=tol, atol=tol, equal_nan=True))


def verificar_graficos(xlsx: Path, dir_run: Path, escenario: str) -> list[str]:
    """Las tablas que respaldan los gráficos del Excel deben coincidir con los JSON/NPZ/CSV fuente de la corrida (recálculo independiente)."""
    import json
    import openpyxl
    from sklearn.metrics import confusion_matrix, roc_auc_score
    fallos: list[str] = []
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    imagenes = sum(len(w._images) for w in openpyxl.load_workbook(xlsx).worksheets)   # noqa: SLF001
    graficos = sum(len(w._charts) for w in openpyxl.load_workbook(xlsx).worksheets)   # noqa: SLF001
    for hoja in ("Índice", "Pronóstico", "Diccionario", "Metodología y límites"):
        if hoja not in wb.sheetnames:
            fallos.append(f"falta la hoja «{hoja}»")
    if fallos:
        return fallos
    # índice: todos los enlaces internos apuntan a hojas existentes
    wbl = openpyxl.load_workbook(xlsx)
    for fila in wbl["Índice"].iter_rows(min_row=5):
        c = fila[0]
        if c.value and (not c.hyperlink or c.value not in wbl.sheetnames or c.hyperlink.location is None and f"'{c.value}'" not in str(c.hyperlink.target)):
            fallos.append(f"Índice: la hoja «{c.value}» no tiene un enlace interno válido")
    # pronóstico
    ws = wb["Pronóstico"]
    m = re.search(r"modelo (\w+) · (Prueba|Validaci.n)", str(ws["A1"].value))
    if not m:
        return fallos + ["Pronóstico: no se pudo leer el modelo del título"]
    cod, fuente = m.group(1), m.group(2)
    z = np.load(dir_run / (f"pred_prueba_{escenario}_{cod}.npz" if fuente == "Prueba" else f"pred_val_{escenario}_{cod}.npz"), allow_pickle=False)
    uid, y, proba, w = z["uid"].astype(str), z["y"].astype(int), np.asarray(z["proba"], float), np.asarray(z["w"], float)
    cab = [c.value for c in ws[6]]
    filas = [[c for c in r] for r in ws.iter_rows(min_row=7, values_only=True) if r[0] is not None]
    pr = pd.DataFrame(filas, columns=cab)
    if len(pr) != len(uid):
        fallos.append(f"Pronóstico: {len(pr)} filas != {len(uid)} hogares de {fuente} en {cod}")
    else:
        pk = [f"proba_{k}" for k in range(proba.shape[1])]
        if not _cerca(pr[pk].to_numpy(float), proba):
            fallos.append("Pronóstico: las probabilidades por clase no coinciden con el NPZ de predicciones")
        if not np.allclose(pr[pk].to_numpy(float).sum(1), 1.0, atol=1e-6):
            fallos.append("Pronóstico: hay filas cuyas probabilidades no suman 1")
        if not (pr["clase_observada"].to_numpy(int) == y).all() or not (pr["clase_predicha"].to_numpy(int) == proba.argmax(1)).all():
            fallos.append("Pronóstico: clase observada/predicha no coincide con el NPZ")
        if not _cerca(pr["peso"].to_numpy(float), w):
            fallos.append("Pronóstico: los pesos muestrales no coinciden con el NPZ")
        if (pr["acierto"].eq("Sí") != (pr["clase_predicha"] == pr["clase_observada"])).any():
            fallos.append("Pronóstico: la columna ¿Acierto? es inconsistente")
        if pr["id_hogar_interno"].nunique() != len(pr) or any(str(i) in set(uid) for i in pr["id_hogar_interno"].head(50)):
            fallos.append("Pronóstico: el identificador no es un número de orden único")
        # variables explicativas numéricas contra el parquet (independiente)
        pq = pd.read_parquet(PARQUET).drop_duplicates("hogar_uid")
        pq["hogar_uid"] = pq["hogar_uid"].astype(str)
        pq = pq.set_index("hogar_uid").reindex(uid)
        for col in [c for c in cab if isinstance(c, str) and c.startswith("[")]:
            v = col.strip("[]")
            if v in pq.columns and pd.api.types.is_numeric_dtype(pq[v]) and pd.api.types.is_numeric_dtype(pd.to_numeric(pr[col], errors="coerce")):
                a, b = pd.to_numeric(pr[col], errors="coerce").to_numpy(float), pq[v].to_numpy(float)
                if not np.allclose(a, b, rtol=1e-4, atol=1e-6, equal_nan=True):
                    fallos.append(f"Pronóstico: la variable {v} no coincide con el parquet")
    # desempeño por modelo (JSON de la CV anidada)
    t = _tabla_hoja(wb["Fig_Desempeno"], "Datos exactos de la figura")
    if t is None:
        fallos.append("Fig_Desempeno: falta la tabla de datos")
    else:
        for r in t.itertuples():
            j = json.loads((dir_run / f"resultado_{escenario}_{r.modelo}.json").read_text(encoding="utf-8"))["agregado"]["ponderada"][r._3]
            if not _cerca([r.media, r.DE, r._6, r._7, r.n], [j["media"], j["sd"], j["ic_inf"], j["ic_sup"], j["n"]]):
                fallos.append(f"Fig_Desempeno: {r.modelo}/{r._3} no coincide con resultado_{escenario}_{r.modelo}.json")
    # matrices de confusión, ROC y calibración (recalculados desde los NPZ)
    wc, wr, wk = wb["Fig_Confusion"], wb["Fig_ROC"], wb["Fig_Calibracion"]
    for p in sorted(dir_run.glob(f"pred_prueba_{escenario}_M*.npz")) or sorted(dir_run.glob(f"pred_val_{escenario}_M*.npz")):
        c = p.stem.rsplit("_", 1)[1]
        zz = np.load(p, allow_pickle=False)
        yy, pp = zz["y"].astype(int), np.asarray(zz["proba"], float)
        cm = confusion_matrix(yy, pp.argmax(1), labels=[0, 1, 2, 3])
        tc = _tabla_hoja(wc, f"Modelo {c} · {fuente}: conteos")
        if tc is None or not _cerca(tc.iloc[:, 1:].to_numpy(float), cm):
            fallos.append(f"Fig_Confusion: la matriz de {c} no coincide con el recálculo desde el NPZ")
        tn = _tabla_hoja(wc, f"Modelo {c} · {fuente}: normalizada")
        if tn is None or not _cerca(tn.iloc[:, 1:].to_numpy(float), cm / cm.sum(1, keepdims=True)):
            fallos.append(f"Fig_Confusion: la matriz normalizada de {c} no coincide")
        tr = _tabla_hoja(wr, "Curvas ROC")
        if tr is None or f"{c} FPR" not in tr.columns:
            fallos.append(f"Fig_ROC: faltan los puntos de {c}")
        else:
            xs = pd.to_numeric(tr[f"{c} FPR"], errors="coerce").dropna().to_numpy(float)
            ys = pd.to_numeric(tr[f"{c} TPR"], errors="coerce").dropna().to_numpy(float)
            auc_hoja = float(np.sum(np.diff(xs) * (ys[1:] + ys[:-1]) / 2))
            auc_ind = float(roc_auc_score((yy == 3).astype(int), pp[:, 3]))
            if abs(auc_hoja - auc_ind) > 1e-9:
                fallos.append(f"Fig_ROC: AUC reconstruido de {c} ({auc_hoja:.6f}) != recalculado ({auc_ind:.6f})")
        tk = _tabla_hoja(wk, "Curvas de calibración")
        if tk is None or f"{c} n hogares" not in tk.columns:
            fallos.append(f"Fig_Calibracion: faltan los datos de {c}")
        else:
            n = pd.to_numeric(tk[f"{c} n hogares"], errors="coerce").fillna(0).to_numpy(float)
            fr = pd.to_numeric(tk[f"{c} frecuencia observada"], errors="coerce").fillna(0).to_numpy(float)
            if int(n.sum()) != len(yy) or abs(float((n * fr).sum()) - float((yy == 3).sum())) > 1e-6:
                fallos.append(f"Fig_Calibracion: n o eventos de {c} no cuadran con el NPZ")
    # SHAP guardado
    f = dir_run / "analisis" / f"shap_{escenario}_{cod}.json"
    ts = _tabla_hoja(wb["Fig_SHAP"], "|SHAP| medio")
    if ts is None:
        fallos.append("Fig_SHAP: falta la tabla")
    elif f.is_file():
        imp = json.loads(f.read_text(encoding="utf-8"))["importancia"]
        if len(ts) != len(imp) or not _cerca(sorted(ts["importancia"].astype(float)), sorted(float(v) for v in imp.values())):
            fallos.append("Fig_SHAP: la importancia del Excel no coincide con shap_*.json")
    # escenarios (F1-macro de cada resultado_*.json)
    te = _tabla_hoja(wb["Fig_Escenarios"], "F1-macro ponderado por escenario")
    if te is None:
        fallos.append("Fig_Escenarios: falta la tabla")
    else:
        for r in te.itertuples():
            j = json.loads((dir_run / f"resultado_{r.escenario}_{r.modelo}.json").read_text(encoding="utf-8"))["agregado"]["ponderada"]["f1_macro"]
            if not _cerca([r.media], [j["media"]]):
                fallos.append(f"Fig_Escenarios: {r.escenario}/{r.modelo} no coincide con su JSON")
    # distribución FIES contra el parquet y equidad contra la hoja 6_Equidad
    td = _tabla_hoja(wb["Fig_Distribucion_FIES"], "Distribución de los niveles FIES")
    if td is not None:
        q = pd.read_parquet(PARQUET).drop_duplicates("hogar_uid")["fies_nivel"].dropna().astype(int).value_counts().sort_index()
        if list(td["hogares"].astype(int)) != list(q.values):
            fallos.append(f"Fig_Distribucion_FIES: hogares por nivel {list(td['hogares'])} != parquet {list(q.values)}")
    te2, _ = _tabla_equidad_xlsx(xlsx)
    tq = _tabla_hoja(wb["Fig_Equidad"], "Brechas de TPR, FPR y error por grupo")
    if tq is None or sorted(tq["n"].astype(int)) != sorted(te2["n"].astype(int)):
        fallos.append("Fig_Equidad: los conteos n no coinciden con la hoja 6_Equidad")
    if imagenes < 17 or graficos < 10:
        fallos.append(f"el libro tiene {imagenes} imágenes y {graficos} gráficos nativos (se esperan >= 17 y >= 10)")
    print(f"  gráficos del Excel: {imagenes} imágenes, {graficos} gráficos nativos; pronóstico {len(uid)} hogares ({cod}, {fuente}); tablas verificadas contra NPZ/JSON/parquet")
    return fallos


# Valores esperados del perfil (articulo/CUERPO.py, sección 3.1): (con %, sin %, n con) por característica; prevalencia total 26,8 %.
ESPERADOS_PERFIL = {"menor5": (30.0, 26.3, 5332), "adulto_mayor": (27.1, 26.7, 10418), "pobreza": (39.8, 23.5, 5860)}
TOTAL_PERFIL = 26.8


def _perfil_independiente() -> tuple[dict, dict, int, float]:
    """Recuento independiente sobre el parquet (sin importar fsai): {característica: (prev_con, prev_sin, n_con, n_sin)}, {recuento: (n, prev)}, n total, prevalencia total."""
    d = pd.read_parquet(PARQUET)
    d = d[d["fies_nivel"].notna()]
    y = (d["fies_nivel"].to_numpy(float) >= 2).astype(float)
    w = d["factor_expansion"].to_numpy(float)
    car = {"menor5": d["n_menores5"].to_numpy(float) > 0, "adulto_mayor": d["n_adultos_mayores"].to_numpy(float) > 0,
           "pobreza": d["pobreza_monetaria"].astype(float).to_numpy() == 1}

    def prev(m):
        return float((w[m] * y[m]).sum() / w[m].sum())

    por_car = {k: (prev(m), prev(~m), int(m.sum()), int((~m).sum())) for k, m in car.items()}
    rec = sum(m.astype(int) for m in car.values())
    por_rec = {r: (int((rec == r).sum()), prev(rec == r)) for r in range(4)}
    return por_car, por_rec, len(d), prev(np.ones(len(d), bool))


def verificar_perfil(carpeta: Path) -> list[str]:
    """El perfil del informe (hoja 1b_Perfil y Fig_perfil_*) coincide con el recuento independiente del parquet y con los valores del artículo."""
    import openpyxl
    fallos: list[str] = []
    xl = sorted(Path(carpeta).glob("*.xlsx"))
    if not xl:
        return [f"{carpeta}: sin Excel"]
    wb = openpyxl.load_workbook(xl[-1], data_only=True)
    wbi = openpyxl.load_workbook(xl[-1])
    por_car, por_rec, n_tot, p_tot = _perfil_independiente()
    if round(100 * p_tot, 1) != TOTAL_PERFIL:
        fallos.append(f"prevalencia total independiente {round(100 * p_tot, 1)} != esperada {TOTAL_PERFIL}")
    for k, (pc, ps, nc, ns) in por_car.items():
        e = ESPERADOS_PERFIL[k]
        if (round(100 * pc, 1), round(100 * ps, 1), nc) != e:
            fallos.append(f"perfil independiente {k}: {round(100 * pc, 1)}/{round(100 * ps, 1)}/{nc} != esperado {e}")
    for h in ("1b_Perfil", "Fig_perfil_subgrupos", "Fig_perfil_recuento", "Fig_perfil_fies"):
        if h not in wb.sheetnames:
            fallos.append(f"falta la hoja «{h}»")
    if fallos:
        return fallos
    for h in ("Fig_perfil_subgrupos", "Fig_perfil_recuento", "Fig_perfil_fies"):
        if len(wbi[h]._images) < 1 or len(wbi[h]._charts) < 1:   # noqa: SLF001
            fallos.append(f"{h}: debe tener imagen y gráfico nativo ({len(wbi[h]._images)} imágenes, {len(wbi[h]._charts)} gráficos)")   # noqa: SLF001
    t = _tabla_hoja(wb["Fig_perfil_subgrupos"], "Prevalencia ponderada de IA moderada/severa con y sin")
    if t is None or len(t) != 3:
        fallos.append("Fig_perfil_subgrupos: no se encontró la tabla de datos (3 filas)")
    else:
        for (k, (pc, ps, nc, ns)), (_, f) in zip(por_car.items(), t.iterrows()):
            if not (_cerca(f["prevalencia con"], pc, 1e-9) and _cerca(f["prevalencia sin"], ps, 1e-9) and int(f["n con"]) == nc and int(f["n sin"]) == ns):
                fallos.append(f"Fig_perfil_subgrupos {k}: {f['prevalencia con']:.4f}/{f['prevalencia sin']:.4f}/{f['n con']} != recuento independiente {pc:.4f}/{ps:.4f}/{nc}")
    t = _tabla_hoja(wb["Fig_perfil_recuento"], "Prevalencia ponderada de IA moderada/severa por recuento")
    if t is None:
        fallos.append("Fig_perfil_recuento: no se encontró la tabla de datos")
    else:
        if int(t["n"].sum()) != n_tot:
            fallos.append(f"Fig_perfil_recuento: los n por recuento suman {int(t['n'].sum())} y el total es {n_tot}")
        for _, f in t.iterrows():
            n_i, p_i = por_rec[int(f["recuento"])]
            if int(f["n"]) != n_i or not _cerca(f["prevalencia"], p_i, 1e-9):
                fallos.append(f"Fig_perfil_recuento {f['recuento']}: {f['n']}/{f['prevalencia']:.4f} != independiente {n_i}/{p_i:.4f}")
    t = _tabla_hoja(wb["Fig_perfil_fies"], "Distribución ponderada del nivel FIES por recuento")
    if t is None or int(t["n"].sum()) != n_tot:
        fallos.append("Fig_perfil_fies: tabla ausente o n que no suma el total")
    else:
        sumas = t[[c for c in t.columns if c not in ("recuento", "n")]].sum(axis=1)
        if not _cerca(sumas.to_numpy(), np.ones(len(sumas)), 1e-9):
            fallos.append("Fig_perfil_fies: las proporciones por recuento no suman 1")
    from docx import Document
    for w in sorted(Path(carpeta).glob("*.docx")):
        txt = " ".join(p.text for p in Document(str(w)).paragraphs)
        for frase in ("Perfil descriptivo de hogares vulnerables", "no es un índice ponderado ni validado"):
            if frase not in txt:
                fallos.append(f"{w.name}: falta la frase «{frase}»")
    n_png = len(list(Path(carpeta).glob("fig_perfil_*.png")))
    if n_png != 3:
        fallos.append(f"se esperaban 3 PNG fig_perfil_* y hay {n_png}")
    print(f"  perfil: 3 características y recuento 0-3 (n total {n_tot}) coinciden con el recuento independiente del parquet y con los valores del artículo")
    return fallos


def verificar_informe(carpeta: Path, dir_run: Path, escenario: str = "socio") -> list[str]:
    carpeta = Path(carpeta)
    fallos: list[str] = []
    xl = sorted(carpeta.glob("*.xlsx"))
    if not xl:
        return [f"{carpeta}: sin Excel"]
    fallos += verificar_equidad(xl[-1], Path(dir_run), escenario)
    nombres = __import__("openpyxl").load_workbook(xl[-1], read_only=True).sheetnames
    if "Pronóstico" in nombres:   # informes con hojas de gráficos
        fallos += verificar_graficos(xl[-1], Path(dir_run), escenario)
    if "1b_Perfil" in nombres:
        fallos += verificar_perfil(carpeta)
    else:
        fallos.append("el informe no tiene la hoja 1b_Perfil (perfil descriptivo de hogares vulnerables)")
    fallos += verificar_textos(carpeta)
    return fallos


if __name__ == "__main__":
    carpeta = Path(sys.argv[1])
    corrida = sys.argv[2] if len(sys.argv) > 2 else "cv_anidada_real"
    esc = sys.argv[3] if len(sys.argv) > 3 else "socio"
    art = Path(os.environ["FSAI_ARTIFACTS_DIR"])
    dir_run = art / "fases" / corrida
    res = verificar_informe(carpeta, dir_run, esc)
    for r in res:
        print("[FALLA]", r)
    print("RESULTADO:", "OK" if not res else f"{len(res)} FALLAS")
    sys.exit(1 if res else 0)
