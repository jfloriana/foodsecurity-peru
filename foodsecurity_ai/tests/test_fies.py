"""Clasificación FIES (fies.py) sobre entradas pequeñas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fsai import fies as FI


def test_clasificar_con_umbrales_por_defecto():
    raw = pd.Series([0, 1, 3, 4, 6, 7, 8, np.nan])
    nivel = FI.clasificar_fies(raw, FI.UMBRALES_RAW)
    # leve>=1, moderada>=4, severa>=7
    assert nivel.iloc[:7].tolist() == [0, 1, 1, 2, 2, 3, 3]
    assert pd.isna(nivel.iloc[7])


def test_clasificar_umbrales_propios_inclusivos():
    nivel = FI.clasificar_fies(pd.Series([2.0, 3.0, 5.0, 5.5]), {"leve": 3, "moderada": 5, "severa": 5.5})
    assert nivel.tolist() == [0, 1, 2, 3]


def test_clasificar_rechaza_umbrales_invalidos():
    with pytest.raises(ValueError):
        FI.clasificar_fies(pd.Series([1.0]), {"leve": 4, "moderada": 4, "severa": 7})
    with pytest.raises(KeyError):
        FI.clasificar_fies(pd.Series([1.0]), {"leve": 1, "moderada": 4})


def _items(filas):
    return pd.DataFrame(filas, columns=FI.ITEMS)


def test_raw_score_suma_y_nan_si_falta_un_item():
    df = _items([[1, 0, 1, 0, 0, 0, 0, 0], [1] * 8, [0] * 8, [1, 1, 1, 1, 1, 1, 1, np.nan]])
    raw = FI.calcular_raw_score(df)
    assert raw.iloc[:3].tolist() == [2, 8, 0]
    assert np.isnan(raw.iloc[3])
    with pytest.raises(KeyError):
        FI.calcular_raw_score(df.drop(columns=["fies_q1"]))
    with pytest.raises(ValueError):
        FI.calcular_raw_score(df.replace(1, 2))


def test_agregar_columnas_fies_binaria_moderada_o_severa():
    df = _items([[0] * 8, [1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0], [1] * 8])
    out = FI.agregar_columnas_fies(df)
    assert out["fies_nivel"].tolist() == [0, 1, 2, 3]
    assert out["ia_binaria"].tolist() == [0, 0, 1, 1]
    assert out["fies_rasch_logit"].isna().all()       # sin parámetros inyectados no se inventa nada


def test_rasch_logit_monotono_en_el_puntaje():
    df = _items([[0] * 8, [1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0], [1] * 8])
    b = {c: d for c, d in zip(FI.ITEMS, np.linspace(-1.5, 1.5, 8))}
    logit = FI.calcular_rasch_logit(df, b)
    assert (np.diff(logit.to_numpy()) > 0).all()
    with pytest.raises(KeyError):
        FI.calcular_rasch_logit(df, {"fies_q1": 0.0})
