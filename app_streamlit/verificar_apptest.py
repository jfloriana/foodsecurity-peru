"""Verificación de carga de páginas con st.testing.AppTest (datos simulados SOLO para comprobar que
las páginas renderizan). Uso: python -I verificar_apptest.py"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "app"))
os.environ.pop("FSAI_DATABASE_URL", None)

from streamlit.testing.v1 import AppTest  # noqa: E402

import auth  # noqa: E402
import comun  # noqa: E402

registro = auth.crear_registro("prueba", "admin", "clave-de-prueba-123")
os.environ["FSAI_USUARIOS"] = json.dumps([registro])
assert auth.verificar_credenciales("prueba", "clave-de-prueba-123") == "admin"
assert auth.verificar_credenciales("prueba", "otra") is None
assert auth.verificar_credenciales("noexiste", "x") is None
print("auth: hash PBKDF2 verifica correctamente")

SESION = {"auth_usuario": "prueba", "auth_rol": "admin", "auth_ultimo": time.time()}


def abrir(ruta: str) -> AppTest:
    prueba = AppTest.from_file(str(RAIZ / ruta), default_timeout=60)
    for k, v in SESION.items():
        prueba.session_state[k] = v
    return prueba


PROHIBIDOS = ("DEMO", "NO SON RESULTADOS", "PROMPTUARIO", "ADVERTENCIA ÉTICA", "ADAPTACIÓN TABULAR")


def textos(prueba: AppTest) -> str:
    """Todo el texto visible (markdown, avisos, títulos, captions, etiquetas) en mayúsculas."""
    partes: list[str] = []
    for lista in (prueba.markdown, prueba.caption, prueba.info, prueba.warning, prueba.error, prueba.success,
                  prueba.title, prueba.subheader, prueba.header, prueba.text):
        partes += [str(e.value) for e in lista]
    return " ".join(partes).upper()


def sin_excepciones(prueba: AppTest, nombre: str) -> None:
    assert not prueba.exception, f"{nombre}: {[e.value for e in prueba.exception]}"
    visibles = textos(prueba)
    malos = [t for t in PROHIBIDOS if t in visibles]
    assert not malos, f"{nombre}: texto retirado de la interfaz sigue visible: {malos}"
    print(f"{nombre}: carga sin excepciones; errores={len(prueba.error)} info={len(prueba.info)}")


# 1) app.py sin sesión -> formulario; sin BD -> estados de error controlados
p = AppTest.from_file(str(RAIZ / "app/app.py"), default_timeout=60).run()
sin_excepciones(p, "app.py (sin sesión)")
assert any("Acceso" in s.value for s in p.subheader)

# 1b) flujo de acceso: login fallido -> mensaje sobrio + indicador; bloqueo tras MAX_INTENTOS; login correcto
def enviar(prueba: AppTest, usuario: str, clave: str) -> AppTest:
    prueba.text_input[0].set_value(usuario)
    prueba.text_input[1].set_value(clave)
    prueba.button[0].click()
    return prueba.run()


p = AppTest.from_file(str(RAIZ / "app/app.py"), default_timeout=60).run()
assert [b.label for b in p.button] == ["Ingresar"] and len(p.text_input) == 2
assert not p.sidebar.children or all(len(c.children) == 0 for c in p.sidebar.children.values()), "sidebar visible antes del acceso"
p = enviar(p, "noexiste", "x")
assert any("No fue posible verificar" in e.value for e in p.error), [e.value for e in p.error]
assert not any("noexiste" in e.value for e in p.error)  # no revela ni refleja el usuario
assert p.session_state["auth_fallos"] == 1 and "auth_usuario" not in p.session_state
p = enviar(p, "prueba", "clave-mala")  # usuario real con clave mala: mismo mensaje
assert any("No fue posible verificar" in e.value for e in p.error) and p.session_state["auth_fallos"] == 2
for _ in range(auth.MAX_INTENTOS - 2):
    p = enviar(p, "prueba", "clave-mala")
assert p.session_state["auth_bloqueo_hasta"] > time.time(), "no se activó el bloqueo"
p.run()
assert any("bloqueado" in e.value for e in p.error) and len(p.button) == 0, "el formulario debe ocultarse bloqueado"
print("login: fallo genérico, indicador de intentos y bloqueo verificados")
p = AppTest.from_file(str(RAIZ / "app/app.py"), default_timeout=60).run()
p = enviar(p, "prueba", "clave-de-prueba-123")
assert p.session_state["auth_usuario"] == "prueba" and p.session_state["auth_rol"] == "admin"
assert any("Inicio" in t.value for t in p.title), [t.value for t in p.title]
assert any("Ruta de análisis" in s.value for s in p.subheader), [s.value for s in p.subheader]
sin_excepciones(p, "Inicio con sesión")
assert any("Cerrar sesión" == b.label for b in p.sidebar.button) or any(b.label == "Cerrar sesión" for b in p.button)
print("login: acceso correcto muestra la página de inicio con cierre de sesión")

# 2) páginas con sesión pero SIN base de datos -> error controlado, sin excepción
for ruta in ("app/app.py", "app/pages/01_Dashboard.py", "app/pages/06_Entrenamiento.py",
             "app/pages/09_SHAP.py", "app/pages/11_Equidad.py"):
    sin_excepciones(abrir(ruta).run(), f"{ruta} (BD no disponible)")

# 3) páginas con rol sin permiso
p = AppTest.from_file(str(RAIZ / "app/pages/06_Entrenamiento.py"), default_timeout=60)
for k, v in {"auth_usuario": "x", "auth_rol": "lector", "auth_ultimo": time.time()}.items():
    p.session_state[k] = v
p.run()
assert any("no tiene acceso" in e.value for e in p.error)
print("06_Entrenamiento.py: rol lector bloqueado correctamente")

# 4) datos sintéticos mínimos (NO son resultados) para ejercitar las rutas de renderizado
rng = np.random.default_rng(0)
n = 400
sint_pred = pd.DataFrame({
    "hogar_uid": [f"{i:064x}" for i in range(n)],
    "y_real": rng.integers(0, 4, n), "y_pred": rng.integers(0, 4, n),
    "factor_expansion": rng.uniform(50, 300, n),
    "area": rng.choice(["U", "R"], n), "region_natural": rng.choice(["costa", "sierra", "selva"], n),
    "sexo_jefe": rng.choice(["H", "M"], n),
    "grupo_edad_jefe": rng.choice(["<30", "30-44", "45-59", "60+"], n),
    "grupo_educ_jefe": rng.choice(["sin_primaria", "primaria", "secundaria", "superior"], n),
    "tiene_menor5": rng.choice(["con_menor5", "sin_menor5"], n),
    "pobreza_monetaria": rng.choice(["pobre", "no_pobre"], n),
})
vars_ = [f"v{i}" for i in range(8)]
shap_arr = rng.normal(size=(300, 8))
global_json = {"version": 1, "n_muestra": 300,
               "importancia": {v: float(np.abs(shap_arr[:, i]).mean()) for i, v in enumerate(vars_)},
               "beeswarm": {"variables": vars_, "shap": shap_arr.tolist(),
                            "valor_normalizado": rng.uniform(size=(300, 8)).tolist()}}
local_json = {"version": 1, "clase_explicada": 3, "base_value": 0.1, "prediccion": 0.4,
              "contribuciones": {v: float(rng.normal()) for v in vars_}, "valores": {v: 1 for v in vars_}}
uid = f"{1:064x}"


def falso_consultar(sql, params=None, *, cache=True):
    s = " ".join(sql.split())
    if "SELECT DISTINCT anio" in s:
        return pd.DataFrame({"anio": [2025]})
    if "AS n_hogares, CAST(SUM(h.factor_expansion) AS double precision) AS hogares_expandidos" in s:
        return pd.DataFrame([{"n_hogares": 400, "hogares_expandidos": 5e4, "prev_ia": 0.3,
                              "prev_severa": 0.1, "raw_medio": 2.0}])
    if "GROUP BY h.departamento" in s:
        return pd.DataFrame({"departamento": ["A", "B"], "n_hogares": [10, 100], "prevalencia": [0.2, 0.4]})
    if "GROUP BY h.area" in s:
        return pd.DataFrame({"area": ["U", "R"], "n_hogares": [10, 100], "prevalencia": [0.2, 0.4]})
    if "GROUP BY r.fies_nivel" in s:
        return pd.DataFrame({"nivel": [0, 1, 2, 3], "n_hogares": [1, 2, 3, 4], "proporcion": [.4, .3, .2, .1]})
    if "FROM fsai.datasets ORDER BY" in s:
        return pd.DataFrame({"dataset_id": [1], "nombre": ["ENAHO_prueba"], "anio": [2025], "version": ["v"], "n_filas": [400]})
    if "FROM fsai.experimentos ORDER BY" in s:
        return pd.DataFrame({"experimento_id": [1], "nombre": ["e"], "escenario": ["socio"],
                             "esquema_validacion": ["V1"], "estado": ["ok"], "iniciado_en": [None]})
    if "FROM fsai.metricas" in s:
        return pd.DataFrame({"modelo": ["M1", "M2"], "nombre_metrica": ["f1_macro"] * 2,
                             "valor": [0.5, 0.6], "ic_inf": [0.4, 0.55], "ic_sup": [0.6, np.nan]})
    if "FROM fsai.modelos mo" in s:
        return pd.DataFrame({"modelo_id": [1], "codigo": ["M1"], "nombre": ["e"], "escenario": ["socio"],
                             "esquema_validacion": ["V1"]})
    if "tipo = 'shap_global'" in s:
        return pd.DataFrame({"contribuciones": [global_json]})
    if "SELECT hogar_uid FROM fsai.explicaciones" in s:
        return pd.DataFrame({"hogar_uid": [uid]})
    if "tipo = 'shap_local' AND hogar_uid = :hogar_uid" in s:
        return pd.DataFrame({"contribuciones": [local_json]})
    if "FROM fsai.hogares h LEFT JOIN" in s:
        return pd.DataFrame([{"departamento": "A", "area": "U", "region_natural": "costa", "fies_nivel": 2}])
    if "FROM fsai.predicciones p" in s:
        return sint_pred
    raise AssertionError(f"consulta no simulada: {s[:80]}")


import fases_comun  # noqa: E402
import ruta as ruta_mod  # noqa: E402

comun.consultar = falso_consultar
fases_comun.consultar = falso_consultar
ruta_mod.consultar = falso_consultar
fases_comun._cargar_bd = lambda dataset_id: pd.DataFrame({"hogar_uid": [f"{i:064x}" for i in range(n)], "fies_nivel": rng.integers(0, 4, n),
                                                           "factor_expansion": rng.uniform(50, 300, n)})
for ruta in ("app/pages/01_Dashboard.py", "app/pages/06_Entrenamiento.py",
             "app/pages/09_SHAP.py", "app/pages/11_Equidad.py"):
    prueba = abrir(ruta).run()
    sin_excepciones(prueba, f"{ruta} (datos sintéticos de prueba)")
    assert not prueba.error, f"{ruta}: errores visibles {[e.value for e in prueba.error]}"
print("verificación AppTest completada")
