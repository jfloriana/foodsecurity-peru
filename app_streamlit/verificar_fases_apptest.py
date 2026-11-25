"""Verificación end-to-end de los módulos de análisis con st.testing.AppTest contra la base REAL del despliegue local.

Uso (con PostgreSQL del despliegue iniciado):  python verificar_fases_apptest.py
Carga despliegue_local/.env en el proceso SIN imprimir valores. Por defecto solo comprueba que TODAS las páginas cargan sin excepciones,
los permisos por rol, la ausencia de textos retirados de la interfaz y los módulos de datos. Las comprobaciones pesadas (lanzar una CV anidada M1
y un estudio Optuna como subprocesos, evaluar la Prueba, generar el informe) requieren FSAI_VERIF_PESADA=1 y dejan corridas con el prefijo «verif_»
en despliegue_local/artifacts; no las ejecute mientras otro proceso entrena sobre los mismos artefactos.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PROYECTO = RAIZ.parents[1]
for linea in (PROYECTO / "despliegue_local" / ".env").read_text(encoding="utf-8").splitlines():
    m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$", linea)
    if m:
        os.environ[m.group(1)] = m.group(2)
os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, str(RAIZ / "app"))

from streamlit.testing.v1 import AppTest  # noqa: E402

SUFIJO = time.strftime("%H%M%S")
FASES = Path(os.environ["FSAI_ARTIFACTS_DIR"]) / "fases"
OPTUNA = Path(os.environ["FSAI_ARTIFACTS_DIR"]) / "optuna"
fallos: list[str] = []


def abrir(ruta: str, rol: str = "admin") -> AppTest:
    at = AppTest.from_file(str(RAIZ / ruta), default_timeout=300)
    at.session_state["auth_usuario"] = "prueba_apptest"
    at.session_state["auth_rol"] = rol
    at.session_state["auth_ultimo"] = time.time()
    return at


def ok(cond: bool, mensaje: str) -> None:
    print(("[OK] " if cond else "[FALLA] ") + mensaje)
    if not cond:
        fallos.append(mensaje)


PROHIBIDOS = ("DEMO", "INV" + "-13", "SINTÉTIC", "SINTETIC", "SIN DATOS DISPONIBLES", "NO SON RESULTADOS", "PROMPTUARIO", "ADVERTENCIA ÉTICA", "ADAPTACIÓN TABULAR")


def textos(at: AppTest) -> str:
    partes: list[str] = []
    for lista in (at.markdown, at.caption, at.info, at.warning, at.error, at.success, at.title, at.subheader, at.header, at.text):
        # Los captions de procedencia («Datos: <archivo> · huella de partición ...») muestran el nombre real del archivo de la corrida: es dato, no leyenda.
        partes += [str(e.value) for e in lista if "huella de partición" not in str(e.value)]
    return " ".join(partes).upper()


def sin_excepciones(at: AppTest, nombre: str) -> bool:
    exc = [e.value for e in at.exception]
    ok(not exc, f"{nombre}: sin excepciones" + (f" -> {exc[0][:300]}" if exc else ""))
    malos = [t for t in PROHIBIDOS if t in textos(at)]
    ok(not malos, f"{nombre}: sin textos retirados de la interfaz {malos if malos else ''}")
    return not exc


def buscar(lista, etiqueta: str, ultimo: bool = False):
    for w in (reversed(list(lista)) if ultimo else lista):
        if (w.label or "").startswith(etiqueta):
            return w
    raise KeyError(f"No se encontró el widget «{etiqueta}»; hay: {[x.label for x in lista][:12]}")


def esperar(ruta_json: Path, estado: str, segundos: int = 900) -> dict:
    t0 = time.time()
    while time.time() - t0 < segundos:
        try:
            d = json.loads(ruta_json.read_text(encoding="utf-8"))
            if d.get("estado") in (estado, "error"):
                return d
        except (OSError, ValueError):
            pass
        time.sleep(3)
    return {}


# ------------------------------------------------------------------ 1) todas las páginas (existentes y nuevas) con rol admin
PAGINAS = ["app.py", "pages/01_Dashboard.py", "pages/02_Importacion_ENAHO.py", "pages/03_Procesamiento_FIES.py", "pages/04_EDA.py",
           "pages/05_Vulnerabilidad.py", "pages/06_Entrenamiento.py", "pages/07_Optuna.py", "pages/08_Comparacion_Estadistica.py",
           "pages/09_SHAP.py", "pages/10_Territorial.py", "pages/11_Equidad.py", "pages/12_Reportes.py", "pages/13_Inferencia.py"]
for p in PAGINAS:
    at = abrir("app/" + p).run()
    sin_excepciones(at, p)

# ------------------------------------------------------------------ 2) roles: el lector no importa, procesa, entrena, optimiza ni genera reportes
for p, esperado in (("pages/02_Importacion_ENAHO.py", False), ("pages/03_Procesamiento_FIES.py", False), ("pages/06_Entrenamiento.py", False),
                    ("pages/07_Optuna.py", False), ("pages/12_Reportes.py", False),
                    ("pages/04_EDA.py", True), ("pages/05_Vulnerabilidad.py", True), ("pages/08_Comparacion_Estadistica.py", True),
                    ("pages/09_SHAP.py", True), ("pages/10_Territorial.py", True), ("pages/13_Inferencia.py", True)):
    at = abrir("app/" + p, rol="lector").run()
    bloqueado = any("no tiene acceso" in e.value for e in at.error)
    ok(bloqueado == (not esperado) and not at.exception, f"{p}: rol lector {'bloqueado' if not esperado else 'puede consultar'}")

# ------------------------------------------------------------------ 2b) módulos nuevos: Procesamiento FIES, Vulnerabilidad, Territorial, EDA, Importación
at = abrir("app/pages/03_Procesamiento_FIES.py").run()
sin_excepciones(at, "Procesamiento FIES")
ok(len(at.tabs) == 3 and len(at.dataframe) >= 1, "Procesamiento FIES: puntaje crudo/niveles, Rasch y calidad (3 pestañas)")
at = abrir("app/pages/05_Vulnerabilidad.py").run()
sin_excepciones(at, "Vulnerabilidad")
ok(len(at.dataframe) >= 3, "Vulnerabilidad: tablas de banderas por nivel FIES, medias y por área")
# perfil descriptivo de hogares vulnerables: la página carga y sus cifras coinciden con el recuento independiente del parquet
ok(len(at.tabs) == 2 and "Perfil descriptivo de hogares vulnerables" in [t.label for t in at.tabs], "Vulnerabilidad: pestañas «Perfil descriptivo» e «Indicadores por nivel FIES»")
import numpy as _np  # noqa: E402
import pandas as _pd  # noqa: E402
_d = _pd.read_parquet(PROYECTO / "datos_publicos" / "procesado" / "enaho_analitico.parquet")
_d = _d[_d["fies_nivel"].notna()]
_y, _w = (_d["fies_nivel"].to_numpy(float) >= 2).astype(float), _d["factor_expansion"].to_numpy(float)
_tot = 100 * float((_y * _w).sum() / _w.sum())
_vals = {m.label: m.value for m in at.metric}
_m_tot = [v for k, v in _vals.items() if k.startswith("Prevalencia total")]
ok(bool(_m_tot) and _m_tot[0] == f"{_tot:.1f} %".replace(".", ","), f"Perfil (página): prevalencia total {_m_tot} = recuento independiente {_tot:.1f} %")
ok(_vals.get("Hogares con FIES") == f"{len(_d):,}", f"Perfil (página): hogares con FIES {_vals.get('Hogares con FIES')} = {len(_d):,}")
_n_rec = [df_ for df_ in at.dataframe if "Hogares con FIES" not in str(df_.value) and "recuento" in [str(c) for c in df_.value.columns]]
ok(len(_n_rec) >= 2 and int(_n_rec[0].value["n"].sum()) == len(_d), f"Perfil (página): el recuento 0-3 suma n total ({len(_d):,})")
_txt_pagina = textos(at)
ok("NO ES UN ÍNDICE PONDERADO NI VALIDADO" in _txt_pagina and ("INV" + "-13") not in _txt_pagina, "Perfil (página): declara que es descriptivo (no índice validado) y no menciona el código de estudio retirado")
at = abrir("app/pages/10_Territorial.py").run()
buscar(at.selectbox, "Unidad territorial").select("area").run()
sin_excepciones(at, "Territorial por área")
buscar(at.selectbox, "Unidad territorial").select("departamento").run()
sin_excepciones(at, "Territorial por departamento")
ok(len(at.dataframe) >= 1, "Territorial: prevalencia ponderada por departamento con IC")
at = abrir("app/pages/04_EDA.py").run()
buscar(at.multiselect, "Área").set_value(["U"]).run()
sin_excepciones(at, "EDA con filtro de área")
ok(len(at.tabs) == 2 and len(at.dataframe) >= 3, "EDA: visualización y descriptiva con tablas")
at = abrir("app/pages/02_Importacion_ENAHO.py").run()
sin_excepciones(at, "Importación ENAHO")
ok(len(at.tabs) == 3 and len(at.dataframe) >= 3, "Importación ENAHO: carga, metadata y validación de grupos (3 pestañas)")

# ------------------------------------------------------------------ 2c) informe real más reciente: equidad y textos (modo ligero, solo lectura)
sys.path.insert(0, str(RAIZ))
from verificar_informe import verificar_informe  # noqa: E402

_informes = sorted((FASES.parent.parent / "reportes").glob("cv_anidada_real_socio_v*"), key=lambda p: p.stat().st_mtime)
if _informes and (FASES / "cv_anidada_real").is_dir():
    _f = verificar_informe(_informes[-1], FASES / "cv_anidada_real", "socio")
    ok(not _f, f"Informe real {_informes[-1].name}: conteos de equidad = recuento independiente del parquet; sin DEMO/sintético/«Sin datos disponibles» {_f if _f else ''}")
else:
    print("[AVISO] No hay informe real (cv_anidada_real_socio_v*) para verificar sus conteos de equidad.")

if os.environ.get("FSAI_VERIF_PESADA") != "1":
    print("\nSe omiten las comprobaciones pesadas (entrenar, evaluar la Prueba, Optuna, informe). Use FSAI_VERIF_PESADA=1 para ejecutarlas.")
    print("RESULTADO:", "TODO OK" if not fallos else f"{len(fallos)} FALLAS: {fallos}")
    sys.exit(1 if fallos else 0)

# ------------------------------------------------------------------ 4) Entrenamiento: lanzar CV anidada real (M1, 2 ensayos)
nombre_f2 = f"verif_f2_{SUFIJO}"
at = abrir("app/pages/06_Entrenamiento.py").run()
buscar(at.multiselect, "Modelos", ultimo=True).set_value(["M1"])
buscar(at.selectbox, "Escenario", ultimo=True).select("socio")
buscar(at.number_input, "Ensayos Optuna por pliegue externo").set_value(2)
buscar(at.number_input, "Ensayos de la búsqueda final").set_value(2)
buscar(at.text_input, "Nombre de la corrida").set_value(nombre_f2)
buscar(at.button, "Lanzar CV anidada").click().run()
sin_excepciones(at, "Entrenamiento tras lanzar")
ok(any("lanzada" in s.value for s in at.success), "Entrenamiento: la corrida se lanzó como subproceso")
prog = esperar(FASES / nombre_f2 / "progreso.json", "completado")
ok(prog.get("estado") == "completado", f"Entrenamiento: CV anidada completada (estado={prog.get('estado')})")
at.run()
sin_excepciones(at, "Entrenamiento con resultados")
res = json.loads((FASES / nombre_f2 / "resultado_socio_M1.json").read_text(encoding="utf-8"))
ok(len(res["folds"]) == 5 and res["n_inner"] == 3 and res["agregado"]["ponderada"]["f1_macro"]["n"] == 5, "Entrenamiento: 5 pliegues externos x 3 internos y 5 observaciones agregadas")
ok(not list((FASES / nombre_f2).glob("prueba_*")), "Entrenamiento: la Prueba permanece intacta (sin candado ni predicciones)")
ver = json.loads((FASES / nombre_f2 / "config.json").read_text(encoding="utf-8"))["verificacion_particion"]
ok(ver["cruces_hogar_uid_total"] == 0 and abs(ver["proporciones"]["desarrollo"] - 0.70) < 0.01, "Entrenamiento: partición 70/15/15 sin cruces de hogar_uid")

# ------------------------------------------------------------------ 5) Comparación: evaluar la Prueba UNA vez
at = abrir("app/pages/08_Comparacion_Estadistica.py").run()
buscar(at.selectbox, "Corrida de Entrenamiento").select(nombre_f2).run()
buscar(at.checkbox, "Confirmo que el modelo final").check()
buscar(at.button, "Evaluar la Prueba").click().run()
sin_excepciones(at, "Comparación tras evaluar")
ok((FASES / nombre_f2 / "prueba_socio_M1.lock").exists() and (FASES / nombre_f2 / "pred_prueba_socio_M1.npz").exists(), "Comparación: Prueba evaluada, candado y predicciones guardados")
at.run()
sin_excepciones(at, "Comparación con resultados")
ok(any("EVALUADO" in str(df.value.to_dict()) for df in at.dataframe), "Comparación: el registro muestra el modelo como EVALUADO (bloqueado)")
ok(not [b for b in at.button if (b.label or "").startswith("Evaluar la Prueba")], "Comparación: el botón de evaluación desaparece (no se puede repetir)")
sys.path.insert(0, str(Path(os.environ["FSAI_RAIZ_PROYECTO"])))
from fsai import evaluacion as EV  # noqa: E402
from fsai import datasets as D  # noqa: E402

df_demo = D.cargar_hogares_ruta(Path(os.environ["FSAI_ARTIFACTS_DIR"]).parent / "datos" / "demo_sintetico.csv")
try:
    EV.evaluar_prueba_una_vez(df_demo, FASES / nombre_f2, "socio_M1", "prueba")
    ok(False, "Comparación: segunda evaluación debió ser bloqueada")
except EV.PruebaYaEvaluada:
    ok(True, "Comparación: segunda evaluación de la Prueba bloqueada (PruebaYaEvaluada)")

# ------------------------------------------------------------------ 6) SHAP: explicabilidad con botones
at = abrir("app/pages/09_SHAP.py").run()
buscar(at.selectbox, "Corrida").select(nombre_f2).run()
sin_excepciones(at, "SHAP (SHAP global)")
for etiqueta in ("Calcular curvas de eliminación", "Calcular explicación LIME", "Calcular importancia por permutación"):
    buscar(at.button, etiqueta).click().run()
    sin_excepciones(at, f"SHAP: {etiqueta}")
ok(len(at.tabs) >= 7, "SHAP: pestañas de explicaciones almacenadas y de modelos anidados (global, mapas de calor, máscaras, local/LIME, permutación/PDP/ALE)")

# ------------------------------------------------------------------ 7) Reportes: reporte a 600 DPI
at = abrir("app/pages/12_Reportes.py").run()
buscar(at.selectbox, "Corrida de entrenamiento").select(str(FASES / nombre_f2))
at.run()
buscar(at.button, "Generar informe").click().run()
sin_excepciones(at, "Reportes tras generar")
out = at.session_state["reporte_salida"]
ok(set(out["archivos"]) == {"PDF", "Word", "Excel"} and all(Path(r).stat().st_size > 1000 for r in out["archivos"].values()), "Reportes: PDF, Word y Excel generados")
sys.path.insert(0, str(Path(os.environ["FSAI_RAIZ_PROYECTO"])))
from fsai import reportes_fases as R  # noqa: E402

dpis = {R.dpi_png(Path(f)) for f in out["figuras"]}
ok(len(out["figuras"]) >= 3 and dpis == {(600, 600)}, f"Reportes: {len(out['figuras'])} figuras exportadas a 600 DPI ({dpis})")
# Los conteos por grupo de la hoja 6_Equidad deben coincidir con un recuento independiente del parquet (todos los atributos de equidad)
sys.path.insert(0, str(RAIZ))
from verificar_informe import verificar_equidad  # noqa: E402

_fallos_eq = verificar_equidad(Path(out["archivos"]["Excel"]), FASES / nombre_f2, "socio")
ok(not _fallos_eq, f"Reportes: conteos de equidad del Excel = recuento independiente del parquet {_fallos_eq if _fallos_eq else ''}")

# ------------------------------------------------------------------ 8) Inferencia: predicción individual y lote
at = abrir("app/pages/13_Inferencia.py").run()
buscar(at.selectbox, "Corrida").select(nombre_f2).run()
buscar(at.button, "Predecir").click().run()
sin_excepciones(at, "Inferencia tras predecir")
ok(len(at.metric) >= 4, "Inferencia: nivel, P(severa) con IC, umbral y percentil poblacional visibles")
_sn_i = at.session_state["inf_ind"]
ok(not _sn_i["extra_err"] and all(_sn_i[k] is not None for k in ("lime", "sens", "contra", "otros", "validez")),
   f"Inferencia: LIME, sensibilidad, contrafactual, otros modelos y validez de entradas se generan {_sn_i['extra_err']}")
_pc = _sn_i["sens"]["curvas"][[f"p{k}" for k in range(4)]]
ok(abs(_pc.sum(axis=1) - 1).max() < 1e-6 and ((_pc >= 0) & (_pc <= 1)).all().all() and abs(_sn_i["lime"]["tabla"]["peso"]).sum() > 0,
   "Inferencia: curvas de sensibilidad con probabilidades válidas y LIME con pesos")
if _sn_i["contra"]["aplica"]:
    from fsai import etiquetas as _E  # noqa: E402
    ok(all(_E.es_accionable(c["variable"]) for e in _sn_i["contra"]["escenarios"] for c in e["cambios"])
       and all(e["p_mod_sev"] < _sn_i["contra"]["p_mod_sev_base"] for e in _sn_i["contra"]["escenarios"]), "Inferencia: el contrafactual solo mueve variables accionables y reduce P(moderada o severa) según el modelo")
else:
    ok("circularidad" in _sn_i["contra"]["motivo"], "Inferencia: el contrafactual no aplica a escenarios con ítems FIES (circularidad declarada)")
ok(len(_sn_i["tornado"] if "tornado" in _sn_i else _sn_i["sens"]["tornado"]) > 0 and any(t.label == "LIME y SHAP" for t in at.tabs), "Inferencia: la página muestra las pestañas de análisis adicionales")
from fsai import inference as INF  # noqa: E402

ens = INF.cargar_ensamble(FASES / nombre_f2, "socio_M1")
lote = df_demo.iloc[:25][ens["resultado"]["features_num"] + ens["resultado"]["features_cat"]]
pl = INF.predecir_lote(ens, lote)
ok(len(pl) == 25 and abs(pl[[f"p_clase_{k}" for k in range(4)]].sum(axis=1) - 1).max() < 1e-6, "Inferencia: lote CSV de 25 hogares con probabilidades que suman 1")

# ------------------------------------------------------------------ 9) Optuna: Optuna UI (lanzar, monitorear y reanudar)
nombre_o = f"verif_opt_{SUFIJO}"
at = abrir("app/pages/07_Optuna.py").run()
buscar(at.selectbox, "Modelo").select("M1")
buscar(at.selectbox, "Escenario").select("socio")
at.run()
buscar(at.number_input, "n_trials").set_value(3)
buscar(at.text_input, "Nombre del estudio").set_value(nombre_o)
buscar(at.checkbox, "Entiendo que la búsqueda usa solo Desarrollo").check()
buscar(at.button, "Iniciar").click().run()
sin_excepciones(at, "Optuna tras lanzar")
prog = esperar(OPTUNA / nombre_o / "progreso.json", "completado")
ok(prog.get("estado") == "completado", f"Optuna: estudio completado (estado={prog.get('estado')})")
import optuna  # noqa: E402

url = "sqlite:///" + (OPTUNA / "estudios.db").as_posix()
n1 = len(optuna.load_study(study_name=nombre_o, storage=url).trials)
at = abrir("app/pages/07_Optuna.py").run()
sin_excepciones(at, "Optuna con estudio persistido")
buscar(at.selectbox, "Reanudar estudio existente").select(nombre_o)
at.run()
buscar(at.number_input, "n_trials").set_value(5)
buscar(at.checkbox, "Entiendo que la búsqueda usa solo Desarrollo").check()
buscar(at.button, "Iniciar").click().run()
# progreso.json aún dice «completado» de la primera ejecución: se espera a que el estudio alcance los 5 ensayos
n2 = n1
t0 = time.time()
while time.time() - t0 < 600 and n2 < 5:
    time.sleep(4)
    n2 = len(optuna.load_study(study_name=nombre_o, storage=url).trials)
esperar(OPTUNA / nombre_o / "progreso.json", "completado", 60)
ok(n1 == 3 and n2 == 5, f"Optuna: persistencia y reanudación del estudio ({n1} ensayos -> {n2})")

print("\nRESULTADO:", "TODO OK" if not fallos else f"{len(fallos)} FALLAS: {fallos}")
sys.exit(1 if fallos else 0)
