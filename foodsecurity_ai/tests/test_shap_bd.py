"""Mejora #16: ``_shap_desde_bd`` filtra por modelo/escenario (fuente SQL simulada en memoria, sin PostgreSQL)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from fsai import reportes_fases as RF

# Filas simuladas de fsai.explicaciones (ya unidas con modelos y experimentos).
FILAS = [
    {"codigo": "M3", "escenario": "socio", "contribuciones": {"features": {"ingreso_pc": 0.4, "edad_jefe": 0.1}}},
    {"codigo": "M3", "escenario": "socio", "contribuciones": json.dumps({"features": {"ingreso_pc": 0.6, "edad_jefe": 0.3}})},
    {"codigo": "M2", "escenario": "socio", "contribuciones": {"features": {"ingreso_pc": 0.1, "edad_jefe": 0.9}}},
    {"codigo": "M3", "escenario": "fies", "contribuciones": {"features": {"fies_raw_score": 2.0}}},
]


def _ctx(filas=FILAS, falla=False):
    """Contexto mínimo: ``sql`` aplica en memoria los filtros que la consulta realmente declara."""
    llamadas = []

    def sql(consulta, **params):
        llamadas.append((consulta, params))
        if falla:
            raise RuntimeError("sin base de datos")
        sel = [f for f in filas
               if ("mo.codigo = :m" not in consulta or f["codigo"] == params["m"])
               and ("e.escenario = :e" not in consulta or f["escenario"] == params["e"])]
        return pd.DataFrame({"contribuciones": [f["contribuciones"] for f in sel]})

    return SimpleNamespace(dataset_id=7, sql=sql, llamadas=llamadas)


def _dict(df):
    return dict(zip(df["variable"], df["importancia"]))


def test_sin_filtros_mezcla_todo_comportamiento_historico():
    x = _ctx()
    imp = _dict(RF._shap_desde_bd(x))
    assert imp["ingreso_pc"] == pytest.approx((0.4 + 0.6 + 0.1) / 3)       # promedio de M3 y M2
    assert "fies_raw_score" in imp                                        # y del escenario fies
    consulta, params = x.llamadas[0]
    assert ":m" not in consulta and ":e" not in consulta and params == {"d": 7}


def test_filtro_por_modelo_y_escenario_no_mezcla():
    x = _ctx()
    imp = _dict(RF._shap_desde_bd(x, modelo="M3", escenario="socio"))
    assert imp == pytest.approx({"ingreso_pc": 0.5, "edad_jefe": 0.2})
    assert x.llamadas[0][1] == {"d": 7, "m": "M3", "e": "socio"}


def test_filtro_solo_por_modelo():
    imp = _dict(RF._shap_desde_bd(_ctx(), modelo="M2"))
    assert imp == pytest.approx({"ingreso_pc": 0.1, "edad_jefe": 0.9})


def test_sin_filas_para_el_modelo_devuelve_none():
    assert RF._shap_desde_bd(_ctx(), modelo="M5", escenario="socio") is None


def test_error_de_base_de_datos_devuelve_none():
    assert RF._shap_desde_bd(_ctx(falla=True), modelo="M3") is None


def test_valores_absolutos_se_promedian():
    filas = [{"codigo": "M3", "escenario": "socio", "contribuciones": {"features": {"a": -0.5}}},
             {"codigo": "M3", "escenario": "socio", "contribuciones": {"features": {"a": 0.5}}}]
    assert _dict(RF._shap_desde_bd(_ctx(filas), modelo="M3"))["a"] == pytest.approx(0.5)
