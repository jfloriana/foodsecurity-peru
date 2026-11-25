"""Métricas ponderadas (metrics_weighted.py) con valores calculados a mano."""
from __future__ import annotations

import numpy as np
import pytest

from fsai import metrics_weighted as MW

# 4 hogares; el modelo asigna 0,7 a su clase predicha y 0,1 a cada una de las otras tres.
Y = np.array([0, 0, 1, 1])
PRED = np.array([0, 1, 1, 1])        # solo el hogar de índice 1 es error
W = np.array([3.0, 1.0, 1.0, 3.0])


def _proba(pred):
    p = np.full((len(pred), 4), 0.1)
    p[np.arange(len(pred)), pred] = 0.7
    return p


def test_accuracy_ponderada_vs_no_ponderada():
    p = _proba(PRED)
    assert MW.calcular_metricas(Y, p)["accuracy"] == pytest.approx(3 / 4)
    assert MW.calcular_metricas(Y, p, W)["accuracy"] == pytest.approx(7 / 8)   # (3+1+3)/8


def test_brier_a_mano():
    p = _proba(PRED)
    # fila correcta: 0,3^2 + 3*0,1^2 = 0,12 ; fila errada: 0,9^2 + 0,7^2 + 2*0,1^2 = 1,32
    assert MW.brier_multiclase(Y, p) == pytest.approx((3 * 0.12 + 1.32) / 4)
    assert MW.brier_multiclase(Y, p, W) == pytest.approx((7 * 0.12 + 1.32) / 8)


def test_ece_a_mano():
    p = _proba(PRED)       # confianza 0,7 en todos -> un solo bin
    assert MW.ece_top_label(Y, p) == pytest.approx(abs(0.75 - 0.7))
    assert MW.ece_top_label(Y, p, W) == pytest.approx(abs(0.875 - 0.7))


def test_recall_clase_ponderado():
    assert MW.recall_clase(Y, PRED, 0) == pytest.approx(0.5)
    assert MW.recall_clase(Y, PRED, 0, W) == pytest.approx(0.75)      # 3*1 / (3+1)
    assert np.isnan(MW.recall_clase(Y, PRED, 3))                     # clase ausente


def test_peso_unitario_equivale_a_sin_pesos():
    p = _proba(PRED)
    a = MW.calcular_metricas(Y, p)
    b = MW.calcular_metricas(Y, p, np.ones(len(Y)))
    for k in ("accuracy", "f1_macro", "brier", "ece", "mcc"):
        assert a[k] == pytest.approx(b[k])


def test_pesos_proporcionales_no_cambian_la_metrica():
    p = _proba(PRED)
    a = MW.calcular_metricas(Y, p, W)
    b = MW.calcular_metricas(Y, p, 100 * W)
    for k in ("accuracy", "f1_macro", "brier", "ece"):
        assert a[k] == pytest.approx(b[k])


def test_calcular_metricas_devuelve_las_11_claves():
    p = _proba(PRED)
    assert set(MW.calcular_metricas(Y, p, W)) == set(MW.NOMBRES_METRICAS)


def test_validaciones_de_entrada():
    p = _proba(PRED)
    with pytest.raises(ValueError):
        MW.calcular_metricas(Y, p[:, :3])
    with pytest.raises(ValueError):
        MW.calcular_metricas(Y, p, np.array([1, 1, 1, -1.0]))
    p_nan = p.copy()
    p_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        MW.calcular_metricas(Y, p_nan)


def test_umbral_clase_severa_respeta_especificidad():
    # severa (clase 3): puntajes bajos para negativos, altos para positivos
    y = np.array([0] * 8 + [3] * 4)
    s = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.8, 0.4, 0.6, 0.7, 0.9])
    p = np.zeros((12, 4))
    p[:, 3] = s
    p[:, 0] = 1 - s
    u = MW.umbral_clase_severa(y, p, especificidad_min=0.75)
    pred = s >= u
    esp = (~pred[y != 3]).mean()
    assert esp >= 0.75
    assert pred[y == 3].mean() >= 0.75        # con ese umbral se detecta al menos 3 de 4 severos
    with pytest.raises(ValueError):
        MW.umbral_clase_severa(np.zeros(5, int), np.full((5, 4), 0.25))


def test_predecir_con_umbral():
    p = np.array([[0.6, 0.2, 0.1, 0.1],
                  [0.1, 0.2, 0.3, 0.4],      # P(severa)=0,4 >= 0,35 -> 3
                  [0.1, 0.2, 0.5, 0.2]])      # P(severa)=0,2 < 0,35 -> mejor de las otras = 2
    assert MW.predecir_con_umbral(p, 0.35).tolist() == [0, 3, 2]
