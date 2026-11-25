"""Verificación del reporte descargable de la página de Inferencia (Word, Excel y PDF) con st.testing.AppTest sobre datos REALES.

Uso (con PostgreSQL del despliegue iniciado):
    python ..\\..\\despliegue_local\\_con_env.py verificar_inferencia_reporte.py [carpeta_de_salida_del_ejemplo]

Comprueba, para la inferencia individual (ejecutada desde la página) y para un lote (mismo generador): que se generan los tres formatos, que abren
(python-docx, openpyxl, PyMuPDF), que contienen entradas, modelo y probabilidades que suman 1, figuras incrustadas a 600 DPI, ausencia de textos
DEMO/sintético, congelación del snapshot, coincidencia con lo mostrado en la página y registro en fsai.auditoria. Deja un ejemplo en la carpeta de salida.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "app"))

from streamlit.testing.v1 import AppTest  # noqa: E402

fallos: list[str] = []
SALIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ["FSAI_ARTIFACTS_DIR"]).parent / "reportes" / "ejemplo_inferencia"


def ok(cond: bool, mensaje: str) -> None:
    print(("[OK] " if cond else "[FALLA] ") + mensaje)
    if not cond:
        fallos.append(mensaje)


def abrir() -> AppTest:
    at = AppTest.from_file(str(RAIZ / "app/pages/13_Inferencia.py"), default_timeout=300)
    at.session_state["auth_usuario"] = "prueba_apptest"
    at.session_state["auth_rol"] = "admin"
    at.session_state["auth_ultimo"] = time.time()
    return at


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).lower()


PROHIBIDOS = ("demo", "sintétic", "sintetic", "synthetic", "inv" + "-13", "inv" + "13", "sin datos disponibles")


def texto_docx(ruta: Path) -> tuple[str, int]:
    from docx import Document
    d = Document(str(ruta))
    partes = [p.text for p in d.paragraphs]
    for t in d.tables:
        for f in t.rows:
            partes += [c.text for c in f.cells]
    return " ".join(partes), len(d.inline_shapes)


def texto_xlsx(ruta: Path) -> tuple[str, dict]:
    from openpyxl import load_workbook
    wb = load_workbook(str(ruta))
    partes, info = [], {"hojas": wb.sheetnames, "imagenes": 0, "graficos": 0}
    for ws in wb.worksheets:
        info["imagenes"] += len(ws._images)
        info["graficos"] += len(ws._charts)
        for fila in ws.iter_rows(values_only=True):
            partes += [str(v) for v in fila if v is not None]
    return " ".join(partes), info


def texto_pdf(ruta: Path) -> tuple[str, int, int]:
    import fitz
    d = fitz.open(str(ruta))
    return " ".join(p.get_text() for p in d), len(d), sum(len(p.get_images()) for p in d)


def revisar_archivos(sn: dict, carpeta: Path, etiqueta: str, valores_esperados: list[str], textos_esperados: list[str]) -> None:
    from fsai import reporte_inferencia as RI
    from fsai.reportes_fases import dpi_png
    arch = RI.generar_archivos(sn, carpeta)
    ok(all(len(b) > 20000 for b in arch.values()), f"{etiqueta}: los 3 formatos se generan y no están vacíos ({ {k: len(v) for k, v in arch.items()} })")
    rutas = {f: carpeta / RI.nombre_archivo(sn, f) for f in arch}
    td, n_img_docx = texto_docx(rutas["word"])
    tx, info_x = texto_xlsx(rutas["excel"])
    tp, n_pag, n_img_pdf = texto_pdf(rutas["pdf"])
    ok(n_img_docx >= 3 and n_img_pdf >= 3 and info_x["imagenes"] >= 3, f"{etiqueta}: figuras incrustadas (Word {n_img_docx}, PDF {n_img_pdf}, Excel {info_x['imagenes']})")
    ok(info_x["graficos"] >= 1 and "Índice" in info_x["hojas"] and "Diccionario" in info_x["hojas"],
       f"{etiqueta}: Excel con Índice, Diccionario y {info_x['graficos']} gráficos nativos ({len(info_x['hojas'])} hojas)")
    ok(n_pag >= 8, f"{etiqueta}: PDF con {n_pag} páginas")
    for nombre, t in (("Word", td), ("Excel", tx), ("PDF", tp)):
        mal = [p for p in PROHIBIDOS if p in t.lower()]
        ok(not mal, f"{etiqueta}: {nombre} sin textos DEMO/sintético {mal if mal else ''}")
        faltan = [v for v in textos_esperados if norm(v) not in norm(t)]
        ok(not faltan, f"{etiqueta}: {nombre} contiene modelo, resultado y advertencias {faltan[:3] if faltan else ''}")
    for nombre, t in (("Word", td), ("PDF", tp)):
        faltan = [v for v in valores_esperados if norm(v) not in norm(t)]
        ok(len(faltan) <= max(1, len(valores_esperados) // 20), f"{etiqueta}: {nombre} contiene los valores de entrada/probabilidades ({len(valores_esperados) - len(faltan)}/{len(valores_esperados)}) {faltan[:3] if faltan else ''}")
    figs = sorted((carpeta / "figuras").glob("*.png"))
    dpis = {dpi_png(f) for f in figs}
    ok(len(figs) >= 3 and dpis == {(600, 600)}, f"{etiqueta}: {len(figs)} figuras PNG exportadas a 600 DPI ({dpis})")


_DEV: list = []


def st_dev(corrida: str):
    """Muestra de Desarrollo de la corrida (la misma partición que usa la página)."""
    if not _DEV:
        cod = """
import streamlit as st
from fases_comun import selector_dataset
from fsai.splits import cargar_particion
from fsai import evaluacion as EV
df, info = selector_dataset()
d = EV.dir_corrida("%s")
part, _ = cargar_particion(df, d / "particion.json")
st.session_state["_dev"] = df.iloc[part["desarrollo"]].reset_index(drop=True)
""" % corrida
        a = AppTest.from_string(cod, default_timeout=300)
        a.session_state["auth_usuario"] = "prueba_apptest"
        a.run()
        _DEV.append(a.session_state["_dev"])
    return _DEV[0]


def revisar_analisis_individual(sn: dict, carpeta: Path) -> None:
    """LIME, sensibilidad, contrafactual, otros modelos y validez: se generan, son válidos y el contrafactual realmente reduce el riesgo según el modelo."""
    import numpy as np
    from fsai import analisis_local as AL
    from fsai import config as C
    from fsai import etiquetas as E
    from fsai import evaluacion as EV
    from fsai import inference as INF
    from fsai import reporte_inferencia as RI

    ok(not sn["extra_err"], f"Análisis adicionales sin errores ({sn['extra_err']}); tiempos (s) {sn['tiempos']}")
    L = sn["lime"]
    ok(L is not None and len(L["tabla"]) == L["n_variables"] and np.isfinite(L["tabla"]["peso"]).all() and L["semilla"] == C.SEED and L["n_muestras"] >= 1000,
       f"LIME: {len(L['tabla'])} variables con peso finito, semilla {L['semilla']}, {L['n_muestras']} perturbaciones, R2 local {L['r2_local']:.2f}, P local {L['prob_local']:.3f} vs modelo {L['prob_modelo']:.3f}")
    ok(0 <= L["prob_modelo"] <= 1 and set(L["tabla"]["variable"]) <= set(sn["modelo"]["features_num"] + sn["modelo"]["features_cat"]), "LIME: probabilidades válidas y variables del modelo")
    c = sn["lime_shap"]
    ok(c["disponible"] and c["signos_total"] > 0 and 0 <= c["coincidencia_top_k"] <= c["k"], f"LIME vs SHAP: top-{c['k']} coinciden {c['coincidencia_top_k']}; signos {c['signos_coinciden']}/{c['signos_total']}; Spearman {c['spearman_abs']:.2f}")
    dir_run = EV.dir_corrida(sn["modelo"]["corrida"])
    res = EV.resultados_corrida(sn["modelo"]["corrida"])[sn["modelo"]["id"]]
    ens = INF.cargar_ensamble(dir_run, sn["modelo"]["id"])
    dev = st_dev(sn["modelo"]["corrida"])
    L2 = AL.lime_local(ens["final"], dev, sn["valores"], res, L["clase"])
    ok(np.allclose(L2["tabla"].set_index("variable")["peso"].reindex(L["tabla"]["variable"]).to_numpy(), L["tabla"]["peso"].to_numpy()), "LIME: reproducible con la misma semilla")
    S = sn["sens"]
    cur = S["curvas"]
    pc = cur[[f"p{k}" for k in range(4)]]
    ok(((pc >= 0) & (pc <= 1)).all().all() and abs(pc.sum(axis=1) - 1).max() < 1e-6, f"Sensibilidad: {len(cur)} filas de curvas, probabilidades en [0,1] y suman 1")
    spec = {c_["variable"]: c_ for c_ in sn["spec"]}
    cat_ok = all(set(cur[cur.variable == v]["valor"].astype(str)) == set(dev[v].dropna().astype(str)) for v in S["principales"] if spec[v]["tipo"] == "categorica")
    num_ok = all(cur[cur.variable == v]["valor"].astype(float).between(dev[v].min(), dev[v].max()).all() for v in S["principales"] if spec[v]["tipo"] != "categorica")
    ok(cat_ok and num_ok, "Sensibilidad: categóricas con TODAS sus categorías y numéricas dentro del rango real de Desarrollo (cuantiles, no inventados)")
    ok(len(S["tornado"]) == len(spec) and S["tornado"]["|efecto| (pp)"].max() > 0, f"Tornado: {len(S['tornado'])} variables (todas), efecto máx. {S['tornado']['|efecto| (pp)'].max():.1f} pp")
    r0 = cur.iloc[len(cur) // 2]
    fila = dict(sn["valores"])
    fila[r0["variable"]] = r0["valor"]
    pp = INF.predecir_ensamble(ens["miembros"], INF.fila_desde_formulario(fila, res), n_boot=10)["media"][0]
    ok(abs(pp[2] + pp[3] - r0["p_mod_sev"]) < 1e-9, "Sensibilidad: una fila de la curva coincide con una predicción independiente del ensamble")
    cf = sn["contra"]
    ok(cf["aplica"], "Contrafactual aplica al escenario socio")
    ok(1 <= len(cf["escenarios"]) <= AL.CF_N_ESCENARIOS, f"Contrafactual: {len(cf['escenarios'])} escenario(s); objetivo {cf['objetivo']:.3f}; P base {cf['p_mod_sev_base']:.3f}")
    base_pm = float(INF.predecir_ensamble(ens["miembros"], INF.fila_desde_formulario(sn["valores"], res), n_boot=10)["media"][0][2:].sum())
    ok(abs(base_pm - cf["p_mod_sev_base"]) < 1e-9, "Contrafactual: P base = predicción independiente del ensamble")
    no_acc = ("edad_jefe", "sexo_jefe", "departamento", "area", "region_natural", "n_miembros", "nivel_educ_jefe", "gasto_pc", "pobreza_monetaria")
    for e in cf["escenarios"]:
        f2 = dict(sn["valores"])
        for ch in e["cambios"]:
            f2[ch["variable"]] = ch["valor_nuevo"]
        P2 = INF.predecir_ensamble(ens["miembros"], INF.fila_desde_formulario(f2, res), n_boot=10)["media"][0]
        acc_ok = all(E.es_accionable(ch["variable"]) and not ch["variable"].startswith("fies_") and ch["variable"] not in no_acc for ch in e["cambios"])
        igual = all(f2[k] == sn["valores"][k] for k in sn["valores"] if k not in {ch["variable"] for ch in e["cambios"]})
        rangos_ok = True
        for ch in e["cambios"]:
            lo, hi = cf["rangos_p10_p90"][ch["variable"]]
            rangos_ok &= min(lo, ch["valor_actual"]) - 1e-9 <= ch["valor_nuevo"] <= max(hi, ch["valor_actual"]) + 1e-9
        ok(abs(float(P2[2:].sum()) - e["p_mod_sev"]) < 1e-9 and e["p_mod_sev"] < cf["p_mod_sev_base"] and abs(P2.sum() - 1) < 1e-6,
           f"Contrafactual {e['n']}: P(mod+sev) {cf['p_mod_sev_base']:.3f} -> {float(P2[2:].sum()):.3f} (recalculada de forma independiente; REDUCE el riesgo según el propio modelo); alcanza objetivo: {e['alcanza_objetivo']}")
        ok(acc_ok and igual and rangos_ok and 1 <= len(e["cambios"]) <= AL.CF_MAX_CAMBIOS, f"Contrafactual {e['n']}: solo variables accionables {[c_['variable'] for c_ in e['cambios']]}, el resto del hogar intacto, valores dentro del rango plausible")
    ok(not any(v.startswith("fies_") for v in cf["accionables"]) and set(cf["accionables"]) <= set(E.ACCIONABLES), "Contrafactual: ninguna variable FIES ni no accionable en la lista de búsqueda")
    o = sn["otros"]
    ok(len(o) >= 4 and (abs(o[[f"P(nivel {k})" for k in range(4)]].sum(axis=1) - 1) < 1e-6).all() and (o["Seleccionado"] == "★").sum() == 1, f"Otros modelos: {len(o)} modelos de la corrida, probabilidades suman 1, 1 seleccionado")
    ok(sn["validez"] is not None and len(sn["validez"]) == len(spec), f"Validez de entradas: {sn['validez']['Estado'].value_counts().to_dict()}")
    v_fuera = dict(sn["valores"])
    v_num = [k for k, c_ in spec.items() if c_["tipo"] == "numerica"][0]
    v_fuera[v_num] = float(dev[v_num].max()) * 3 + 10
    vv = AL.validez_entradas(sn["spec"], v_fuera, dev)
    ok(vv[vv["Código"] == v_num]["Estado"].iloc[0].startswith("fuera"), "Extrapolación: un valor fuera del rango observado se marca «fuera del rango observado»")
    nuevas = {"fig_inf_lime", "fig_inf_lime_shap", "fig_inf_sensibilidad", "fig_inf_tornado", "fig_inf_contrafactual", "fig_inf_otros_modelos"}
    presentes = {f.stem for f in (carpeta / "figuras").glob("*.png")}
    ok(nuevas <= presentes, f"Figuras nuevas exportadas a PNG (faltan {sorted(nuevas - presentes)})")
    tx, info_x = texto_xlsx(carpeta / RI.nombre_archivo(sn, "excel"))
    hojas_n = {"4_Explicabilidad", "7_Sensibilidad", "8_Contrafactual", "9_Otros_modelos", "Fig_LIME", "Fig_LIME_vs_SHAP", "Fig_Sensibilidad", "Fig_Tornado", "Fig_Contrafactual", "Fig_Otros_modelos"}
    ok(hojas_n <= set(info_x["hojas"]), f"Excel: hojas nuevas presentes (faltan {sorted(hojas_n - set(info_x['hojas']))}); {info_x['graficos']} gráficos nativos")
    from openpyxl import load_workbook
    wb = load_workbook(str(carpeta / RI.nombre_archivo(sn, "excel")))
    n_nat = {h: len(wb[h]._charts) for h in ("Fig_LIME", "Fig_LIME_vs_SHAP", "Fig_Sensibilidad", "Fig_Tornado", "Fig_Contrafactual", "Fig_Otros_modelos") if h in wb.sheetnames}
    ok(all(v >= 1 for v in n_nat.values()) and n_nat.get("Fig_Sensibilidad", 0) >= 4, f"Excel: gráficos nativos en las hojas nuevas {n_nat}")
    td, _ = texto_docx(carpeta / RI.nombre_archivo(sn, "word"))
    tp, _, _ = texto_pdf(carpeta / RI.nombre_archivo(sn, "pdf"))
    for nombre, t in (("Word", td), ("PDF", tp), ("Excel", tx)):
        faltan = [x for x in ("Análisis de sensibilidad", "Análisis contrafactual", "Explicación local con LIME", "Comparación con los demás modelos", "Tornado") if norm(x) not in norm(t)]
        ok(not faltan, f"{nombre}: secciones nuevas presentes {faltan if faltan else ''}")
    ok(all(norm(x) in norm(td) and norm(x) in norm(tp) for x in ("no efecto causal", "capacidad predictiva es modesta")), "Word y PDF: declaran asociación predictiva (no causal) y capacidad predictiva modesta junto al contrafactual")


# ------------------------------------------------------------------ individual (desde la página)
at = abrir().run()
ok(not at.exception, "Inferencia: la página carga sin excepciones")
sel = [s for s in at.selectbox if s.label.startswith("Modelo")][0]
sel.select("socio_M3").run()
# hogar con las tres características del perfil (menor de 5 años, adulto mayor y pobreza) para comprobar la pertenencia y el recuento 3
[n for n in at.number_input if n.label.startswith("Menores de 5")][0].set_value(1.0)
[n for n in at.number_input if n.label.startswith("Adultos mayores")][0].set_value(1.0)
[s_ for s_ in at.selectbox if s_.label.startswith("Pobreza monetaria")][0].select(1.0)
t_pred = time.perf_counter()
[b for b in at.button if b.label == "Predecir"][0].click().run()
t_pred = time.perf_counter() - t_pred
ok(t_pred < 120, f"Tiempo de «Predecir» (ensamble, SHAP, LIME, sensibilidad, contrafactual, otros modelos): {t_pred:.1f} s")
ok(not at.exception, f"Inferencia tras predecir: sin excepciones {[e.value[:200] for e in at.exception]}")
sn = at.session_state["inf_ind"]
m = sn["modelo"]
mets = {x.label: x.value for x in at.metric}
ok(mets.get("Nivel predicho (0-3)") == str(sn["nivel"]) and mets.get("P(IA severa)") == f"{sn['p_severa']:.1%}", f"Página: nivel {mets.get('Nivel predicho (0-3)')} y P(severa) {mets.get('P(IA severa)')} = snapshot")
ok(abs(float(sn["media"].sum()) - 1) < 1e-6, f"Probabilidades suman 1 ({float(sn['media'].sum()):.8f})")
def botones_reporte(a):
    return [b.proto.label for b in a.get("download_button") if b.proto.label in ("Descargar Word", "Descargar Excel", "Descargar PDF")]


ok(len(botones_reporte(at)) == 0 and any(b.label.startswith("Preparar reporte") for b in at.button), "Página: el reporte NO se genera solo; hay un botón «Preparar reporte» (la pantalla de resultados no espera)")
t_rep = time.perf_counter()
[b for b in at.button if b.label.startswith("Preparar reporte")][0].click().run()
t_rep = time.perf_counter() - t_rep
ok(t_rep < 240, f"Tiempo de generación del reporte desde la página (Word + Excel + PDF, figuras a 600 DPI): {t_rep:.1f} s")
ok(not at.exception, f"Página tras preparar el reporte: sin excepciones {[e.value[:200] for e in at.exception]}")
botones = botones_reporte(at)
ok(len(botones) == 3, f"Página: botones de descarga {botones}")
clave = f"inf_archivos_{sn['id']}"
arch_pag = at.session_state[clave]
ok(not isinstance(arch_pag, Exception) and set(arch_pag) == {"word", "excel", "pdf"}, "Página: archivos del reporte en el estado de sesión")
# congelación: cambiar un control y volver a ejecutar no altera el resultado ni el reporte
num = [n for n in at.number_input if n.label.startswith("Ingreso")][0]
num.set_value(float(num.value) + 1.0).run()
sn2 = at.session_state["inf_ind"]
ok(sn2["id"] == sn["id"] and (sn2["media"] == sn["media"]).all() and len(botones_reporte(at)) == 3, "Snapshot congelado: tras modificar un widget el resultado y los botones no cambian")
nuevo = [b for b in at.button if b.label == "Predecir"][0].click().run()
ok(at.session_state["inf_ind"]["id"] != sn["id"] and sn["valores"] != at.session_state["inf_ind"]["valores"], "Nueva inferencia con el valor modificado genera otro snapshot e id")
# auditoría
from sqlalchemy import create_engine, text  # noqa: E402

eng = create_engine(os.environ["FSAI_DATABASE_URL"])
with eng.connect() as c:
    n_aud = c.execute(text("SELECT count(*) FROM fsai.auditoria WHERE accion = 'inferencia_individual' AND detalle->>'id_inferencia' = :i"), {"i": sn["id"]}).scalar()
ok(n_aud == 1, f"Auditoría: la inferencia {sn['id']} está registrada en fsai.auditoria ({n_aud})")
# los bytes servidos por la página son un reporte completo
esperados_val = [str(v) for v in sn["entradas"]["Valor"].tolist() if len(str(v)) > 1]
esperados_prob = [f"{p * 100:.2f} %" for p in sn["media"]]
esperados_txt = [m["id"], m["algoritmo"], sn["huella"][:16], f"{sn['p_severa'] * 100:.2f} %", "capacidad predictiva modesta", "[POR VALIDAR]", "no es un diagnóstico", "reconstrucción",
                  "LIME", "aproximación local", "sensibilidad", "contrafactual", "asociación predictiva", "no efecto causal", "variables accionables", "validez de las entradas", "semilla"]
revisar_archivos(sn, SALIDA / "individual", "Individual", esperados_val + esperados_prob, esperados_txt)
revisar_analisis_individual(sn, SALIDA / "individual")
# perfil de vulnerabilidad del hogar: pertenencia y prevalencia del grupo contra un recuento independiente del parquet
sys.path.insert(0, str(RAIZ))
from verificar_informe import _perfil_independiente  # noqa: E402
from fsai import reporte_inferencia as RI  # noqa: E402

_pc, _pr, _nt, _ = _perfil_independiente()
_v = sn["valores"]
_pert = {"menor5": float(_v.get("n_menores5", 0)) > 0, "adulto_mayor": float(_v.get("n_adultos_mayores", 0)) > 0, "pobreza": float(_v.get("pobreza_monetaria", 0)) == 1}
_ctx = sn["perfil_ctx"]
ok(set(_ctx["claves"]) == {k for k, b in _pert.items() if b} and _ctx["recuento"] == sum(_pert.values()),
   f"Perfil individual: pertenencia {_ctx['claves']} y recuento {_ctx['recuento']} = derivados de las entradas {_pert}")
ok(all(abs(g["prevalencia"] - _pc[g["clave"]][0]) < 0.003 and g["n"] > 0 for g in _ctx["grupos"]) and (_ctx["recuento_fila"] is None or abs(_ctx["recuento_fila"]["prevalencia"] - _pr[_ctx["recuento"]][1]) < 0.003),
   f"Perfil individual: la prevalencia observada del grupo coincide con el recuento independiente del parquet ({[(g['clave'], round(g['prevalencia'], 3)) for g in _ctx['grupos']]})")
ok(_ctx["recuento"] == 3 and len(_ctx["grupos"]) == 3 and _ctx["recuento_fila"] is not None, f"Perfil individual: el hogar de prueba pertenece a los 3 grupos (recuento {_ctx['recuento']})")
ok(any("Perfil de vulnerabilidad del hogar" in m_.value for m_ in at.markdown), "Página: muestra «Perfil de vulnerabilidad del hogar»")
_td, _ = texto_docx(SALIDA / "individual" / RI.nombre_archivo(sn, "word"))
_tx, _ = texto_xlsx(SALIDA / "individual" / RI.nombre_archivo(sn, "excel"))
_tp, _, _ = texto_pdf(SALIDA / "individual" / RI.nombre_archivo(sn, "pdf"))
for _n, _t in (("Word", _td), ("Excel", _tx), ("PDF", _tp)):
    ok(norm("Perfil de vulnerabilidad del hogar") in norm(_t) and norm("no es la probabilidad de este hogar") in norm(_t) and norm("no es un índice ponderado ni validado") in norm(_t),
       f"Perfil individual: {_n} con la sección, el contexto (no predicción) y la aclaración de índice no validado")
# el reporte coincide con la página: clase y probabilidades
from docx import Document  # noqa: E402

ruta_w = SALIDA / "individual" / f"reporte_{sn['id']}.docx"
tdoc, _ = texto_docx(ruta_w)
ok(f"{sn['nivel']} · " in tdoc and f"{sn['p_severa'] * 100:.2f} %" in tdoc, "Reporte Word: misma clase predicha y misma probabilidad que la página")
# los bytes que la página sirve para descargar equivalen a lo generado (mismo tamaño aproximado y abren)
import io  # noqa: E402

d_pag = Document(io.BytesIO(arch_pag["word"]))
ok(len(d_pag.paragraphs) > 50 and len(d_pag.inline_shapes) >= 4, f"Descarga de la página: Word abre con {len(d_pag.paragraphs)} párrafos y {len(d_pag.inline_shapes)} figuras")

# ------------------------------------------------------------------ lote (mismo generador)
cod = '''
import streamlit as st
from fases_comun import selector_dataset
from fsai.splits import cargar_particion
from fsai import evaluacion as EV
df, info = selector_dataset()
d = EV.dir_corrida("%s")
part, _ = cargar_particion(df, d / "particion.json")
st.session_state["_dev"] = df.iloc[part["desarrollo"]].reset_index(drop=True)
st.session_state["_val"] = df.iloc[part["validacion"]].reset_index(drop=True)
''' % m["corrida"]
a2 = AppTest.from_string(cod, default_timeout=300)
a2.session_state["auth_usuario"] = "prueba_apptest"
a2.run()
dev, val = a2.session_state["_dev"], a2.session_state["_val"]
from fsai import evaluacion as EV  # noqa: E402
from fsai import inference as INF  # noqa: E402
from fsai import reporte_inferencia as RI  # noqa: E402

dir_run = EV.dir_corrida(m["corrida"])
res = EV.resultados_corrida(m["corrida"])[m["id"]]
ens = INF.cargar_ensamble(dir_run, m["id"])
req = res["features_num"] + res["features_cat"]
dfl = val.iloc[:120][req].reset_index(drop=True)
csv_bytes = dfl.to_csv(index=False).encode("utf-8")
snl = RI.ejecutar_inferencia_lote(nombre_corrida=m["corrida"], ident=m["id"], dir_run=dir_run, res=res, ens=ens, dev=dev, dfl=dfl, bytes_csv=csv_bytes,
                                  nombre_archivo="hogares_validacion_120.csv", usuario="prueba_apptest", dataset={"nombre": "dataset activo", "n": len(dev) + len(val)})
sumas = snl["salida"][[f"p_clase_{k}" for k in range(4)]].sum(axis=1)
ok(len(snl["salida"]) == 120 and abs(sumas - 1).max() < 1e-6, "Lote: 120 hogares reales de Validación; probabilidades suman 1 por hogar")
esperados_txt_l = [m["id"], m["algoritmo"], snl["huella"][:16], f"{snl['n']:,}", f"{snl['n_alertas']:,}", "capacidad predictiva modesta", "[POR VALIDAR]", "no es un diagnóstico"]
esperados_txt_l += ["sensibilidad agregada", "validez de las entradas", "extrapolación"]
revisar_archivos(snl, SALIDA / "lote", "Lote", [f"{snl['conteo_clases'][k]}" for k in range(4)], esperados_txt_l)
ok(not snl["extra_err"] and snl["sens_lote"] is not None and snl["validez_lote"] is not None, f"Lote: sensibilidad agregada ({snl['sens_lote']['n_muestra']} hogares) y validez de entradas calculadas {snl['extra_err']}")
_pl = snl["perfil_lote"]
_ind = {"menor5": dfl["n_menores5"] > 0, "adulto_mayor": dfl["n_adultos_mayores"] > 0, "pobreza": dfl["pobreza_monetaria"] == 1}
_rec_l = sum(m_.astype(int) for m_ in _ind.values())
ok(_pl is not None and int(_pl["por_recuento"]["Hogares del lote"].sum()) + _pl["n_sin_recuento"] == len(dfl) and
   all(int((_rec_l == r).sum()) == int(_pl["por_recuento"].loc[_pl["por_recuento"]["Recuento"] == r, "Hogares del lote"].iloc[0]) for r in range(4)),
   f"Lote: distribución por recuento 0-3 = recuento independiente y suma {len(dfl)} hogares ({_pl['por_recuento']['Hogares del lote'].tolist() if _pl else None})")
_tdl, _ = texto_docx(SALIDA / "lote" / RI.nombre_archivo(snl, "word"))
_tpl, _, _ = texto_pdf(SALIDA / "lote" / RI.nombre_archivo(snl, "pdf"))
_txl, _ = texto_xlsx(SALIDA / "lote" / RI.nombre_archivo(snl, "excel"))
for _n, _t in (("Word", _tdl), ("PDF", _tpl), ("Excel", _txl)):
    ok(norm("Perfil de vulnerabilidad de los hogares del lote") in norm(_t) and norm("Hogares del lote por recuento") in norm(_t), f"Perfil del lote: {_n} con la distribución por recuento")
from openpyxl import load_workbook as _lw  # noqa: E402
_wbl = _lw(str(SALIDA / "lote" / RI.nombre_archivo(snl, "excel")))
ok({"5_Sensibilidad", "6_Validez", "Fig_Lote_Tornado"} <= set(_wbl.sheetnames) and len(_wbl["Fig_Lote_Tornado"]._charts) >= 1, f"Lote Excel: hojas de sensibilidad agregada y validez, con gráfico nativo")
print("\nEjemplo en:", SALIDA)
print("RESULTADO:", "TODO OK" if not fallos else f"{len(fallos)} FALLAS: {fallos}")
sys.exit(1 if fallos else 0)
