"""Brechas por subgrupo (fairness.py) con una tabla pequeña calculable a mano."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fsai import fairness as F


def _toy():
    # Grupo A (referencia, 10 hogares): 5 positivos reales (clase 1), el modelo detecta 4 -> TPR 0,8
    # Grupo B (6 hogares): 4 positivos reales, el modelo detecta 2 -> TPR 0,5 ; FPR: 1 de 2 negativos -> 0,5
    g = ["A"] * 10 + ["B"] * 6
    y = [1] * 5 + [0] * 5 + [1] * 4 + [0] * 2
    pred = [1, 1, 1, 1, 0] + [0, 0, 0, 0, 0] + [1, 1, 0, 0] + [1, 0]
    return pd.DataFrame({"grupo": g}), np.array(y), np.array(pred)


def test_tpr_fpr_y_brecha_a_mano():
    df, y, pred = _toy()
    t = F.brechas_por_grupo(df, y, pred, "grupo", clase_positiva=1, n_boot=200).set_index("grupo")
    assert t.loc["A", "tpr"] == pytest.approx(0.8)
    assert t.loc["B", "tpr"] == pytest.approx(0.5)
    assert t.loc["B", "fpr"] == pytest.approx(0.5)
    assert t.loc["A", "fpr"] == pytest.approx(0.0)
    assert bool(t.loc["A", "es_referencia"]) and not bool(t.loc["B", "es_referencia"])
    assert t.loc["B", "brecha_tpr"] == pytest.approx(-0.3)
    assert t.loc["A", "brecha_tpr"] == 0.0
    assert bool(t.loc["B", "alerta"]) and not bool(t.loc["A", "alerta"])


def test_ic_bootstrap_es_reproducible_y_acotado():
    df, y, pred = _toy()
    a = F.brechas_por_grupo(df, y, pred, "grupo", 1, n_boot=300, seed=1).set_index("grupo")
    b = F.brechas_por_grupo(df, y, pred, "grupo", 1, n_boot=300, seed=1).set_index("grupo")
    assert a.loc["B", "ic_inf"] == b.loc["B", "ic_inf"]
    assert a.loc["B", "ic_inf"] <= a.loc["B", "ic_sup"]


def test_pesos_cambian_el_tpr():
    df, y, pred = _toy()
    # Se duplica el peso de los positivos detectados de B -> TPR(B) = (2*2)/(2*2+2) = 2/3
    w = np.ones(len(df))
    w[10:12] = 2.0
    df = df.assign(factor_expansion=w)
    t = F.brechas_por_grupo(df, y, pred, "grupo", 1, n_boot=50).set_index("grupo")
    assert t.loc["B", "tpr"] == pytest.approx(4 / 6)


def test_clase_positiva_como_coleccion_y_referencia_explicita():
    df, y, pred = _toy()
    t = F.brechas_por_grupo(df, y, pred, "grupo", [1, 2], grupo_referencia="B", n_boot=50).set_index("grupo")
    assert bool(t.loc["B", "es_referencia"]) and t.loc["B", "brecha_tpr"] == 0.0


def test_errores_de_entrada():
    df, y, pred = _toy()
    with pytest.raises(KeyError):
        F.brechas_por_grupo(df, y, pred, "no_existe", 1)
    with pytest.raises(ValueError):
        F.brechas_por_grupo(df, y[:-1], pred, "grupo", 1)
    with pytest.raises(ValueError):
        F.brechas_por_grupo(df.assign(grupo="A"), y, pred, "grupo", 1)
    with pytest.raises(ValueError):
        F.brechas_por_grupo(df, y, pred, "grupo", 1, grupo_referencia="Z")
