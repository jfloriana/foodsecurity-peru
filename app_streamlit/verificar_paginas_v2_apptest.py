"""Verificación con st.testing.AppTest de las páginas nuevas de V2 (14 Glosario, 15 Comparación de corridas, 16 Bitácora) y del
expander de configuración en 06 Entrenamiento. NO requiere PostgreSQL ni despliegue_local/.env: sin FSAI_DATABASE_URL las páginas
deben degradar con un aviso (no con excepciones).

Uso:  python app_streamlit/verificar_paginas_v2_apptest.py
Define FSAI_RAIZ_PROYECTO (carpeta foodsecurity_ai) y FSAI_RESULTADOS_AGREGADOS si no están definidas; quita FSAI_DATABASE_URL
del proceso para que la prueba sea determinista y no toque ninguna base real.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PROYECTO = RAIZ.parent
os.environ.setdefault("FSAI_RAIZ_PROYECTO", str(PROYECTO / "foodsecurity_ai"))
os.environ.setdefault("FSAI_RESULTADOS_AGREGADOS", str(PROYECTO / "resultados_agregados"))
os.environ.setdefault("FSAI_ARTIFACTS_DIR", str(PROYECTO / ".verif_artifacts"))
os.environ.pop("FSAI_DATABASE_URL", None)   # sin BD: se comprueba la degradación
os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, str(RAIZ / "app"))

from streamlit.testing.v1 import AppTest  # noqa: E402

fallos: list[str] = []


def ok(cond: bool, mensaje: str) -> None:
    print(("[OK] " if cond else "[FALLA] ") + mensaje)
    if not cond:
        fallos.append(mensaje)


def abrir(ruta: str, rol: str = "admin") -> AppTest:
    at = AppTest.from_file(str(RAIZ / "app" / ruta), default_timeout=120)
    at.session_state["auth_usuario"] = "prueba_apptest"
    at.session_state["auth_rol"] = rol
    at.session_state["auth_ultimo"] = time.time()
    return at


def sin_excepciones(at: AppTest, nombre: str) -> None:
    exc = [e.value for e in at.exception]
    ok(not exc, f"{nombre}: sin excepciones" + (f" -> {exc[0][:300]}" if exc else ""))


def texto(at: AppTest) -> str:
    partes: list[str] = []
    for lista in (at.markdown, at.caption, at.info, at.warning, at.error, at.success):
        partes += [str(e.value) for e in lista]
    return " ".join(partes)


# 14 Glosario: carga con cualquier rol, tiene términos, diccionario y el buscador filtra
for rol in ("admin", "analista", "lector"):
    at = abrir("pages/14_Glosario.py", rol).run()
    sin_excepciones(at, f"14_Glosario ({rol})")
at = abrir("pages/14_Glosario.py", "lector").run()
ok("calibración" in texto(at).lower() and "StratifiedGroupKFold" in texto(at), "14_Glosario: muestra los términos (calibración, StratifiedGroupKFold)")
ok(len(at.dataframe) >= 1 and len(at.dataframe[0].value) > 10, "14_Glosario: diccionario de variables con filas")
at.text_input[0].set_value("shap").run()
sin_excepciones(at, "14_Glosario con búsqueda")
ok("SHAP" in texto(at) and "StratifiedGroupKFold" not in texto(at), "14_Glosario: el buscador filtra los términos")
at.text_input[0].set_value("zzzz_inexistente").run()
ok(any("Ningún término" in i.value for i in at.info), "14_Glosario: búsqueda sin resultados muestra aviso")

# 15 Comparación de corridas: sin BD usa los JSON agregados reales (si existen) y muestra avisos de comparabilidad
hay_json = any((PROYECTO / "resultados_agregados").glob("*/resultado_*.json"))
for rol in ("admin", "lector"):
    at = abrir("pages/15_Comparacion_Corridas.py", rol).run()
    sin_excepciones(at, f"15_Comparacion_Corridas ({rol})")
if hay_json:
    ok(len(at.dataframe) >= 1, "15_Comparacion_Corridas: tabla con las corridas de resultados_agregados")
    ok(any("reconstrucción del objetivo" in w.value for w in at.warning), "15_Comparacion_Corridas: aviso de escenarios circulares (fies/hibrido)")
    ok(any("NO son comparables" in w.value for w in at.warning), "15_Comparacion_Corridas: aviso de escenarios mezclados")
else:
    print("[AVISO] No hay resultado_*.json en resultados_agregados: solo se comprobó el estado vacío.")
    ok("No hay corridas" in texto(at), "15_Comparacion_Corridas: estado vacío informativo")

# 16 Bitácora: solo admin; sin BD debe degradar con un aviso
at = abrir("pages/16_Bitacora.py", "admin").run()
sin_excepciones(at, "16_Bitacora (admin, sin BD)")
ok(any("no está disponible" in i.value or "No hay conexión" in i.value for i in list(at.info) + list(at.error)), "16_Bitacora: sin base de datos muestra aviso de indisponibilidad")
for rol in ("analista", "lector"):
    at = abrir("pages/16_Bitacora.py", rol).run()
    ok(any("no tiene acceso" in e.value for e in at.error) and not at.exception, f"16_Bitacora: rol {rol} bloqueado")

# 17 Fuentes de datos: todos los roles; tabla de procedencia con las fuentes del manifiesto y las limitaciones honestas
for rol in ("admin", "analista", "lector"):
    at = abrir("pages/17_Fuentes_Datos.py", rol).run()
    sin_excepciones(at, f"17_Fuentes_Datos ({rol})")
t17 = texto(at)
ok("endes_2024_m1629" in t17 and "ophi_mpi_peru" in t17 and "licencia no declarada" in t17.lower(), "17_Fuentes_Datos: lista las fuentes del manifiesto con su licencia")
ok("Desfase temporal" in t17 and "NASA POWER" in t17 and "FAOSTAT" in t17, "17_Fuentes_Datos: muestra desfase temporal, fuentes descartadas y fuentes V1")

# 01 Dashboard y 10 Territorial: sin BD ni dataset deben mostrar estados vacíos honestos y el mapa con indicadores territoriales (sin excepciones)
for pag in ("pages/01_Dashboard.py", "pages/10_Territorial.py"):
    at = abrir(pag, "lector").run()
    sin_excepciones(at, f"{pag} (sin BD)")
    ok(len(at.get("plotly_chart")) >= 1, f"{pag}: dibuja al menos un gráfico/mapa con datos territoriales reales")

# 06 Entrenamiento: la página debe seguir cargando (sin datasets se detiene con aviso, sin excepciones)
at = abrir("pages/06_Entrenamiento.py", "admin").run()
sin_excepciones(at, "06_Entrenamiento (sin BD)")

# 09 Interpretabilidad y 13 Inferencia: cargan sin excepciones (sin BD), y ningún callejón sin salida «No se almacenó muestra...» es alcanzable
for pag in ("pages/09_SHAP.py", "pages/13_Inferencia.py"):
    for rol in ("admin", "lector"):
        at = abrir(pag, rol).run()
        sin_excepciones(at, f"{pag} ({rol}, sin BD)")
        ok("No se almacenó muestra" not in texto(at), f"{pag} ({rol}): sin el mensaje sin salida «No se almacenó muestra»")
fuente09 = (RAIZ / "app" / "pages" / "09_SHAP.py").read_text(encoding="utf-8")
ok("No se almacenó muestra" not in fuente09 and "_beeswarm_faltante" in fuente09 and "Calcular el beeswarm" in fuente09,
   "09_SHAP: el beeswarm faltante se reemplazó por una explicación con botón «Calcular el beeswarm» (bajo demanda)")
ok(all(m not in fuente09 for m in ('st.info("Este modelo no tiene', 'st.info("No hay explicación local', 'st.info("No hay importancias')),
   "09_SHAP: los demás mensajes sin salida usan tema.vacio con siguiente paso")
# el catálogo de campos de la inferencia cubre todas las variables de los modelos reales
sys.path.insert(0, str(PROYECTO / "foodsecurity_ai"))
import json  # noqa: E402
from fsai import catalogo_campos as K  # noqa: E402
for f in sorted((PROYECTO / "resultados_agregados").glob("*/resultado_*.json"))[:15]:
    r = json.loads(f.read_text(encoding="utf-8"))
    ok(all(v in K.VARIABLES_CATALOGADAS for v in r["features_num"] + r["features_cat"]), f"catálogo de campos cubre {f.name}")

# Registro de módulos V2 y roles
import modulos  # noqa: E402
import auth  # noqa: E402

modulos.registrar_accesos()
claves = {m[0] for m in modulos.MODULOS}
ok({"glosario", "comparacion_corridas", "bitacora", "fuentes_datos"} <= claves, "modulos.py: Glosario, Comparación de corridas, Bitácora y Fuentes de datos registrados")
ok(auth.ACCESO_PAGINAS["fuentes_datos"] == ("admin", "analista", "lector"), "auth: fuentes_datos para admin/analista/lector")
ok(auth.ACCESO_PAGINAS["bitacora"] == ("admin",) and "lector" in auth.ACCESO_PAGINAS["glosario"], "auth: bitácora solo admin; glosario para todos los roles")
ok(all((RAIZ / "app" / m[4]).is_file() for m in modulos.MODULOS), "modulos.py: todos los archivos de página existen")

print("\nRESULTADO:", "TODO OK" if not fallos else f"{len(fallos)} FALLAS: {fallos}")
sys.exit(1 if fallos else 0)
