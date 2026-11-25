"""Beeswarm de SHAP: forma del bloque almacenable, suma de dummies por variable cruda, normalización y submuestra estratificada (datos sintéticos)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fsai import explainability as X

NOMBRES = ["num__ingreso_pc", "num__agua_potable", "cat__area_R", "cat__area_U"]


def _muestra(n: int = 40, semilla: int = 3):
    rng = np.random.default_rng(semilla)
    df = pd.DataFrame({"ingreso_pc": rng.lognormal(6, 1, n), "agua_potable": rng.integers(0, 2, n).astype(float),
                       "area": rng.choice(["U", "R"], n)}, index=np.arange(100, 100 + n))
    vals = rng.normal(size=(n, len(NOMBRES)))
    return df, vals


def test_forma_y_suma_de_dummies():
    df, vals = _muestra()
    b = X.beeswarm_datos(vals, df, NOMBRES, clase=3, split="prueba")
    assert set(b) >= {"variables", "shap", "valor_normalizado", "indice", "clase", "split", "n_filas"}
    shap = np.array(b["shap"], dtype=float)
    assert shap.shape == (40, 3) == np.array(b["valor_normalizado"], dtype=float).shape and len(b["variables"]) == 3
    j = b["variables"].index("area")
    assert shap[:, j] == pytest.approx(np.round(vals[:, 2] + vals[:, 3], 5))        # SHAP aditivo: dummies sumadas
    assert b["indice"][0] == "100" and b["clase"] == 3 and b["split"] == "prueba"
    medias = np.abs(shap).mean(axis=0)
    assert list(medias) == sorted(medias, reverse=True)                           # variables de mayor a menor |SHAP|


def test_normalizacion_y_categoricas_sin_color():
    df, vals = _muestra()
    b = X.beeswarm_datos(vals, df, NOMBRES)
    v = {c: np.array([np.nan if x is None else x for x in col], dtype=float) for c, col in zip(b["variables"], zip(*b["valor_normalizado"]))}
    assert np.isnan(v["area"]).all() and b["sin_color"] == ["area"]                # categórica: gris
    assert set(np.unique(v["agua_potable"])) == {0.0, 1.0}                         # binaria: 0/1
    assert 0 < v["ingreso_pc"].min() <= v["ingreso_pc"].max() == 1.0               # continua: rango percentil
    assert X.valor_normalizado(pd.Series([5.0, 5.0, 5.0])).tolist() == pytest.approx([np.nan] * 3, nan_ok=True)


def test_limites_serializable_y_errores():
    df, vals = _muestra(60)
    b = X.beeswarm_datos(vals, df, NOMBRES, top=2, max_filas=25)
    assert len(b["variables"]) == 2 and b["n_filas"] == 25 and len(b["shap"]) == 25 and len(b["indice"]) == 25
    json.dumps(b, allow_nan=False)                                                # JSONB válido: sin NaN
    assert X.beeswarm_datos(vals, df, NOMBRES, top=2, max_filas=25) == b          # reproducible (semilla fija)
    with pytest.raises(ValueError):
        X.beeswarm_datos(vals[:, :3], df, NOMBRES)


def test_submuestra_estratificada_reparte_por_nivel():
    y = np.array([0] * 500 + [1] * 300 + [2] * 150 + [3] * 12)
    pos = X.submuestra_estratificada(y, 100)
    assert len(pos) == 100 and len(set(pos)) == 100
    cuenta = np.bincount(y[pos], minlength=4)
    assert cuenta[3] == 12 and cuenta.sum() == 100 and cuenta[:3].max() - cuenta[:3].min() <= 1     # severa completa; el resto parejo
    assert (X.submuestra_estratificada(y, 100) == pos).all()                      # reproducible
    assert len(X.submuestra_estratificada(y, 5000)) == len(y)                     # n mayor que la población: todos


def test_shap_global_devuelve_beeswarm_retrocompatible():
    pytest.importorskip("shap")
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({"ingreso_pc": rng.normal(size=n), "agua_potable": rng.integers(0, 2, n).astype(float), "area": rng.choice(["U", "R"], n)})
    y = np.clip((df["ingreso_pc"] * -1 + rng.normal(size=n) > 0.3).astype(int) + (df["ingreso_pc"] < -1.2), 0, 3)
    y = np.where(rng.random(n) < 0.05, 3, y)
    pre = ColumnTransformer([("num", StandardScaler(), ["ingreso_pc", "agua_potable"]), ("cat", OneHotEncoder(), ["area"])])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=300))]).fit(df, y)
    if len(pipe.named_steps["clf"].classes_) < 4:
        pytest.skip("la muestra sintética no tiene las 4 clases")
    sg = X.shap_global(pipe, df, "M1", n_muestra=80, split="prueba")
    assert {"nombres", "n_muestra", "importancia_por_clase", "importancia_media", "beeswarm"} <= set(sg)
    b = sg["beeswarm"]
    assert b["n_filas"] == 80 and b["clase"] == 3 and b["split"] == "prueba" and "area" in b["variables"]
    json.dumps(b, allow_nan=False)
