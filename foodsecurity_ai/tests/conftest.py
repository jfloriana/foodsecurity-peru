"""conftest.py - Rutas y datos sintéticos compartidos por la suite de pruebas.

Solo datos sintéticos: ninguna prueba depende del parquet real ni de PostgreSQL.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

DEPARTAMENTOS = {"TUMBES": "costa", "LIMA": "costa", "CUSCO": "sierra", "PUNO": "sierra",
                 "LORETO": "selva", "UCAYALI": "selva"}


def hogares_sinteticos(n_hogares: int = 600, filas_por_hogar: int = 2, semilla: int = 7) -> pd.DataFrame:
    """Dataset sintético con ``hogar_uid`` repetido ``filas_por_hogar`` veces (para que la fuga sea detectable)."""
    rng = np.random.default_rng(semilla)
    uid = np.repeat([f"h{i:05d}" for i in range(n_hogares)], filas_por_hogar)
    n = len(uid)
    dpto = rng.choice(list(DEPARTAMENTOS), size=n_hogares)
    dpto_f = np.repeat(dpto, filas_por_hogar)
    area_h = rng.choice(["U", "R"], size=n_hogares, p=[0.6, 0.4])
    nivel_h = rng.choice([0, 1, 2, 3], size=n_hogares, p=[0.45, 0.3, 0.17, 0.08])
    return pd.DataFrame({
        "hogar_uid": uid,
        "conglome": np.repeat([f"c{i // 5:04d}" for i in range(n_hogares)], filas_por_hogar),
        "departamento": dpto_f,
        "region_natural": [DEPARTAMENTOS[d] for d in dpto_f],
        "area": np.repeat(area_h, filas_por_hogar),
        "fies_nivel": np.repeat(nivel_h, filas_por_hogar),
        "factor_expansion": rng.uniform(50, 400, size=n),
    })


@pytest.fixture(scope="session")
def df_hogares() -> pd.DataFrame:
    return hogares_sinteticos()
