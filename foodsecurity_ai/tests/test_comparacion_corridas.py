"""Pruebas de fsai.comparacion_corridas: carga de JSON agregados reales (si existen), conversión desde BD y advertencias."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fsai import comparacion_corridas as CC


def _json_corrida(carpeta, escenario="socio", codigo="M1", n_trials=20, f1=0.35):
    carpeta.mkdir(parents=True, exist_ok=True)
    agregado = {"ponderada": {
        "f1_macro": {"media": f1, "ic_inf": f1 - 0.01, "ic_sup": f1 + 0.01, "n": 5},
        "roc_auc_ovr": {"media": 0.67, "ic_inf": 0.66, "ic_sup": 0.68, "n": 5},
        "brier": {"media": 0.69, "ic_inf": 0.68, "ic_sup": 0.70, "n": 5}}}
    (carpeta / f"resultado_{escenario}_{codigo}.json").write_text(
        json.dumps({"escenario": escenario, "codigo": codigo, "n_trials": n_trials, "n_trials_final": n_trials, "agregado": agregado}),
        encoding="utf-8")


def test_cargar_json_sintetico(tmp_path):
    _json_corrida(tmp_path / "corrida_a", n_trials=20)
    _json_corrida(tmp_path / "corrida_b", codigo="M2", n_trials=100, f1=0.4)
    t = CC.cargar_corridas_json(tmp_path)
    assert list(t.columns) == CC.COLUMNAS
    assert len(t) == 2
    fila = t[t["corrida"] == "corrida_a"].iloc[0]
    assert fila["f1_macro"] == pytest.approx(0.35) and fila["auc_ic_sup"] == pytest.approx(0.68) and fila["n_pliegues"] == 5


def test_json_ilegible_se_omite_y_se_informa(tmp_path):
    _json_corrida(tmp_path / "ok")
    (tmp_path / "mala").mkdir()
    (tmp_path / "mala" / "resultado_socio_M1.json").write_text("{no es json", encoding="utf-8")
    t = CC.cargar_corridas_json(tmp_path)
    assert len(t) == 1 and len(t.attrs["problemas"]) == 1


def test_advertencias_n_trials_escenarios_y_circularidad(tmp_path):
    _json_corrida(tmp_path / "a", escenario="socio", n_trials=20)
    _json_corrida(tmp_path / "b", escenario="hibrido", n_trials=100)
    avisos = " ".join(CC.advertencias_comparabilidad(CC.cargar_corridas_json(tmp_path)))
    assert "distinto número de ensayos" in avisos and "NO son comparables" in avisos and "reconstrucción del objetivo" in avisos


def test_sin_advertencias_si_es_comparable(tmp_path):
    _json_corrida(tmp_path / "a", n_trials=20)
    _json_corrida(tmp_path / "b", codigo="M2", n_trials=20)
    assert CC.advertencias_comparabilidad(CC.cargar_corridas_json(tmp_path)) == []


def test_tabla_desde_bd_largo_a_ancho():
    largo = pd.DataFrame([
        {"experimento_id": 1, "corrida": "exp1", "esquema": "V1", "escenario": "socio", "modelo": "M1", "n_trials": 10,
         "nombre_metrica": "f1_macro", "valor": 0.3, "ic_inf": 0.28, "ic_sup": 0.32},
        {"experimento_id": 1, "corrida": "exp1", "esquema": "V1", "escenario": "socio", "modelo": "M1", "n_trials": 10,
         "nombre_metrica": "roc_auc_ovr", "valor": 0.6, "ic_inf": 0.5, "ic_sup": 0.7},
    ])
    t = CC.tabla_desde_bd(largo)
    assert len(t) == 1 and t.iloc[0]["f1_macro"] == pytest.approx(0.3) and t.iloc[0]["auc"] == pytest.approx(0.6)
    assert np.isnan(t.iloc[0]["brier"]) and t.iloc[0]["origen"] == "bd"
    assert CC.tabla_desde_bd(pd.DataFrame()).empty


def test_cargar_corridas_cae_a_json_si_falla_la_bd(tmp_path):
    _json_corrida(tmp_path / "a")

    def falla(_sql):
        raise RuntimeError("sin BD")

    t, origen = CC.cargar_corridas(falla, tmp_path)
    assert origen == "json" and len(t) == 1


def test_json_reales_agregados():
    raiz = CC.ruta_resultados_agregados()
    if raiz is None or not list(raiz.glob("*/resultado_*.json")):
        pytest.skip("No hay resultados_agregados con resultado_*.json en este entorno")
    t = CC.cargar_corridas_json(raiz)
    assert not t.empty
    assert set(t["escenario"].dropna()) <= {"socio", "fies", "hibrido"}
    assert t["f1_macro"].between(0, 1).all()
    assert (t["f1_macro_ic_inf"] <= t["f1_macro_ic_sup"]).all()
    assert t["n_trials"].notna().all()
    avisos = " ".join(CC.advertencias_comparabilidad(t))
    if {"fies", "hibrido"} & set(t["escenario"]):
        assert "reconstrucción del objetivo" in avisos
