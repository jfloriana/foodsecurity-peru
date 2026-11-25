"""Catálogo de campos de inferencia: cobertura de todas las variables del modelo, opciones válidas y fórmulas derivadas."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fsai import catalogo_campos as K
from fsai import etiquetas as E

AGREGADOS = Path(__file__).resolve().parents[2] / "resultados_agregados"
RESULTADOS = sorted(AGREGADOS.glob("*/resultado_*.json"))


@pytest.mark.skipif(not RESULTADOS, reason="no hay resultado_*.json en resultados_agregados")
def test_todas_las_variables_de_los_modelos_reales_estan_catalogadas():
    for f in RESULTADOS:
        r = json.loads(f.read_text(encoding="utf-8"))
        feats = r["features_num"] + r["features_cat"]
        faltan = [v for v in feats if v not in K.VARIABLES_CATALOGADAS]
        assert not faltan, f"{f.name}: sin entrada en el catálogo {faltan}"
        claves = K.entradas_para(feats)                       # no lanza KeyError y devuelve entradas del catálogo
        assert claves and all(k in K.ENTRADAS for k in claves)
        df = K.catalogo_df(feats)
        assert len(df) == len(feats) and (df["Qué es"].str.len() > 3).all() and (df["Cómo responder"].str.len() > 10).all()


def test_cada_entrada_tiene_etiqueta_ayuda_y_seccion():
    secciones = {s[0] for s in K.SECCIONES}
    for k, e in K.ENTRADAS.items():
        assert e["etiqueta"] and e["ayuda"] and e["seccion"] in secciones and e["tipo"] in {"si_no", "entero", "decimal", "opciones"}, k
        assert len(e["ayuda"]) <= 140, f"ayuda demasiado larga en {k}"        # una línea concisa
        if e["tipo"] in ("entero", "decimal"):
            assert e["min"] is not None and e["max"] is not None and e["min"] <= e["defecto"] <= e["max"], k
    for k, d in K.DERIVADOS.items():
        assert d["seccion"] in secciones and all(x in K.ENTRADAS for x in d["entradas"]), k
    assert set(K.ORDEN) <= set(K.ENTRADAS) and set(K.ENTRADAS) - set(K.ORDEN) == set()


def test_opciones_con_etiquetas_oficiales_y_valores_internos_validos():
    assert K.TIPO_VIVIENDA["1"] == "Casa independiente" and len(K.TIPO_VIVIENDA) == 8
    assert K.MATERIAL_PAREDES["3"] == "Adobe" and len(K.MATERIAL_PAREDES) == 9
    assert K.P301A[6] == "Secundaria completa" and K.NIVEL_EDUC_MODELO[5] == "Secundaria completa"      # modelo = P301A - 1
    assert K.NIVEL_EDUC_MODELO[10] == "Maestría o doctorado" and K.NIVEL_EDUC_MODELO[11] == "Básica especial" and len(K.NIVEL_EDUC_MODELO) == 12
    for k in ("tipo_vivienda", "material_vivienda", "nivel_educ_jefe", "pobreza_monetaria", "sexo_jefe", "area", "region_natural"):
        ops = K.opciones_de(k)
        assert ops and all(isinstance(t, str) and t for t in ops.values())
        assert K.ENTRADAS[k]["defecto"] in ops                  # el valor por defecto es un valor interno válido
    # solo se ofrecen los códigos que el modelo vio
    assert set(K.opciones_de("tipo_vivienda", ["1", "2", "3"])) == {"1", "2", "3"}
    assert K.opciones_de("departamento", ["la_libertad"]) == {"la_libertad": "La Libertad"}
    # etiquetas de presentación (gráficos) coherentes con el catálogo
    assert E.valor_etq("tipo_vivienda", "2") == "Departamento en edificio" and E.valor_etq("material_vivienda", "1") == "Ladrillo o bloque de cemento"
    # los valores del ejemplo existen en las opciones
    for k in ("tipo_vivienda", "material_vivienda", "nivel_educ_jefe", "pobreza_monetaria", "sexo_jefe", "area", "region_natural"):
        assert K.EJEMPLO[k] in K.opciones_de(k)


def test_formulas_derivadas_coinciden_con_el_entrenamiento():
    e = dict(K.EJEMPLO)
    v = K.valores_modelo(e, list(K.DERIVADOS))
    assert v["share_alimentario"] == pytest.approx(260 / 750)                    # gasto_alim_pc / gasto_pc (fuentes_publicas.py)
    assert v["hacinamiento"] == pytest.approx(4 / 3)                              # n_miembros / habitaciones
    assert v["dependencia_economica"] == pytest.approx(2 / 4)                     # n_ocupados / n_miembros
    assert v["asistencia_escolar"] == pytest.approx(1.0)
    assert v["fies_raw_score"] == 3.0                                             # suma de los 8 ítems
    e.update(escolares_6_17=0, escolares_asisten=0, gasto_pc=0.0)
    v = K.valores_modelo(e, ["asistencia_escolar", "share_alimentario"])
    assert np.isnan(v["asistencia_escolar"]) and np.isnan(v["share_alimentario"])     # «no aplica» = faltante, como en el entrenamiento
    e.update(escolares_6_17=4, escolares_asisten=3)
    assert K.valores_modelo(e, ["asistencia_escolar"])["asistencia_escolar"] == pytest.approx(0.75)


def test_validacion_con_mensajes_claros():
    assert K.validar_entradas(K.EJEMPLO) == []
    malo = dict(K.EJEMPLO, n_ocupados=9, gasto_alim_pc=900.0, escolares_asisten=5, n_menores5=3)
    msgs = " ".join(K.validar_entradas(malo))
    assert "trabajan" in msgs and "gasto en alimentos" in msgs and "asisten" in msgs and "menores de 5" in msgs
    assert any("entre" in m for m in K.validar_entradas(dict(K.EJEMPLO, edad_jefe=5)))


def test_prellenado_desde_hogar_real_y_plantilla():
    fila = pd.Series({"n_miembros": 5.0, "hacinamiento": 2.5, "asistencia_escolar": np.nan, "n_menores18": 2.0, "tipo_vivienda": "2", "nivel_educ_jefe": 6.0,
                      "pobreza_monetaria": 1, "agua_potable": 1.0, "ingreso_pc": 400.0, "departamento": "cusco"})
    b = K.entradas_desde_fila(fila)
    assert b["habitaciones"] == 2 and b["escolares_6_17"] == 0 and b["tipo_vivienda"] == "2" and b["nivel_educ_jefe"] == 6 and b["pobreza_monetaria"] == 1
    assert b["departamento"] == "cusco" and b["ingreso_pc"] == 400.0 and b["sexo_jefe"] == "H"                    # faltante -> valor por defecto
    feats = ["ingreso_pc", "hacinamiento", "fies_raw_score", "tipo_vivienda"]
    tabla = K.catalogo_df(feats, {"tipo_vivienda": ["1", "2"]}, {"ingreso_pc": (6.0, 36000.0)})
    assert list(tabla["Columna (encabezado del CSV)"]) == feats
    assert "1 = Casa independiente" in tabla.loc[3, "Valores permitidos"] and "6" in tabla.loc[0, "Valores permitidos"]
    assert "derivado" in set(tabla["Tipo"])
