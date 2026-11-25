"""Pruebas estadísticas entre modelos (statistical_tests.py) con datos sintéticos."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from fsai import statistical_tests as ST


def test_holm_valores_a_mano():
    # m=3: 0,01*3=0,03 ; 0,03*2=0,06 ; 0,04*1=0,04 -> máximo acumulado 0,06
    aj = ST.holm([0.01, 0.04, 0.03])
    assert aj.tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_holm_monotonia_y_cotas_con_datos_aleatorios():
    rng = np.random.default_rng(0)
    for _ in range(25):
        p = rng.uniform(0, 1, size=int(rng.integers(2, 12)))
        aj = ST.holm(p)
        assert (aj >= p - 1e-12).all() and (aj <= 1).all()          # nunca menor al original, tope 1
        orden = np.argsort(p)
        assert (np.diff(aj[orden]) >= -1e-12).all()                 # monótono según el orden de p


def _tabla_mcnemar():
    y = np.ones(40, int)
    p1, p2 = np.ones(40, int), np.ones(40, int)
    p2[:10] = 0            # b = 10 (1 acierta y 2 falla)
    p1[10:12] = 0          # c = 2  (2 acierta y 1 falla)
    return y, p1, p2


def test_mcnemar_a_mano_sin_pesos():
    r = ST.mcnemar_ponderado(*_tabla_mcnemar())
    assert (r["b"], r["c"]) == (10.0, 2.0)
    assert r["estadistico"] == pytest.approx((abs(10 - 2) - 1) ** 2 / 12)          # 49/12
    assert r["p_valor"] == pytest.approx(stats.chi2.sf(49 / 12, 1))


def test_mcnemar_ponderado_invariante_a_la_escala_de_pesos():
    y, p1, p2 = _tabla_mcnemar()
    base = ST.mcnemar_ponderado(y, p1, p2, pesos=np.full(40, 3.0))
    escala = ST.mcnemar_ponderado(y, p1, p2, pesos=np.full(40, 3000.0))
    sin = ST.mcnemar_ponderado(y, p1, p2)
    assert base["estadistico"] == pytest.approx(sin["estadistico"])   # Kish: pesos iguales == sin pesos
    assert escala["estadistico"] == pytest.approx(base["estadistico"])


def test_mcnemar_pesos_desiguales_a_mano():
    # b = un hogar con peso 4, c = un hogar con peso 1; sin reescalado Kish: (|4-1|-1)^2/5 = 0,8
    y = np.array([1, 1, 1])
    p1 = np.array([1, 0, 1])
    p2 = np.array([0, 1, 1])
    r = ST.mcnemar_ponderado(y, p1, p2, pesos=np.array([4.0, 1.0, 1.0]), n_efectivo_kish=False)
    assert (r["b"], r["c"]) == (4.0, 1.0)
    assert r["estadistico"] == pytest.approx(0.8)


def test_mcnemar_sin_discordantes():
    y = np.array([0, 1, 1, 0])
    r = ST.mcnemar_ponderado(y, y, y)
    assert r["p_valor"] == 1.0 and r["estadistico"] == 0.0


def test_wilcoxon_detecta_diferencia_sistematica():
    rng = np.random.default_rng(1)
    b = rng.uniform(0.4, 0.5, size=20)
    a = b + 0.05 + rng.normal(0, 0.005, size=20)
    r = ST.wilcoxon_pareado(a, b)
    assert r["p_valor"] < 0.001 and r["n"] == 20 and r["mediana_diferencia"] > 0


def test_wilcoxon_series_identicas_y_p_minimo_con_5_pares():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert ST.wilcoxon_pareado(x, x)["p_valor"] == 1.0
    r = ST.wilcoxon_pareado(x + 0.1, x)          # 5 pares, todos a favor: p exacto mínimo 2/32
    assert r["p_valor"] == pytest.approx(0.0625)
    assert r["p_minimo_posible"] == pytest.approx(0.0625)


def test_friedman_nemenyi_ordena_al_mejor_modelo():
    rng = np.random.default_rng(2)
    base = rng.uniform(0.3, 0.4, size=(10, 1))
    M = np.hstack([base + 0.20, base + 0.10, base + 0.05, base]) + rng.normal(0, 0.005, size=(10, 4))
    r = ST.friedman_nemenyi(M, ["A", "B", "C", "D"])
    assert r["p_valor"] < 0.001
    assert r["rangos_medios"]["A"] == pytest.approx(1.0) and r["rangos_medios"]["D"] == pytest.approx(4.0)
    assert r["nemenyi_p"].loc["A", "D"] < 0.05
    assert (r["nemenyi_p"].to_numpy() == r["nemenyi_p"].to_numpy().T).all()
    assert r["diferencia_critica"] > 0


def test_friedman_requiere_3_modelos_y_sin_nan():
    with pytest.raises(ValueError):
        ST.friedman_nemenyi(np.random.rand(5, 2))
    m = np.random.rand(5, 3)
    m[0, 0] = np.nan
    with pytest.raises(ValueError):
        ST.friedman_nemenyi(m)


def test_delong_modelo_bueno_vs_azar():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=400)
    bueno = y * 0.6 + rng.uniform(0, 0.6, size=400)
    azar = rng.uniform(0, 1, size=400)
    r = ST.delong_test(y, bueno, azar)
    assert r["auc1"] > 0.85 and 0.4 < r["auc2"] < 0.6
    assert r["p_valor"] < 0.001 and r["diferencia"] > 0
    with pytest.raises(ValueError):
        ST.delong_test(np.zeros(10, int), rng.random(10), rng.random(10))


def test_bootstrap_estratificado_ic_contiene_el_estimado_y_es_reproducible():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, size=200)
    p = np.clip(y * 0.5 + rng.uniform(0, 0.5, size=200), 0, 1)

    def exactitud(yy, pp):
        return float(((pp >= 0.5).astype(int) == yy).mean())

    r = ST.bootstrap_estratificado(y, p, exactitud, n_boot=200, seed=0)
    assert r["ic_inf"] <= r["estimado"] <= r["ic_sup"]
    assert r == ST.bootstrap_estratificado(y, p, exactitud, n_boot=200, seed=0)
