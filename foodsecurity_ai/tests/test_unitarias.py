"""Pruebas unitarias rápidas de FoodSecurity AI (sin base de datos ni entrenamiento).

Ejecución (desde la carpeta foodsecurity_ai):  python -X utf8 -m pytest tests -q      (o:  python -X utf8 tests/test_unitarias.py)

Cubren: etiquetas (cobertura del parquet), métricas ponderadas y el perfil descriptivo de hogares vulnerables (valores a mano, linealización
frente a reportes_fases.prevalencia_ic y valores del artículo sobre el parquet, que se omiten si el parquet no está).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
PARQUET = RAIZ.parents[1] / "datos_publicos" / "procesado" / "enaho_analitico.parquet"

from fsai import etiquetas as E  # noqa: E402
from fsai import metrics_weighted as MW  # noqa: E402
from fsai import perfil_vulnerabilidad as PV  # noqa: E402


def _mini() -> pd.DataFrame:
    """6 hogares con valores conocidos: FIES (0,1,2,3), pesos y características."""
    return pd.DataFrame({
        "conglome": ["a", "a", "b", "b", "c", "c"], "factor_expansion": [1.0, 1.0, 2.0, 2.0, 4.0, 4.0],
        "fies_nivel": [0, 2, 3, 1, 2, 0], "n_menores5": [0, 1, 1, 0, 0, 2], "n_adultos_mayores": [0, 0, 1, 1, 0, 0],
        "pobreza_monetaria": pd.array([0, 1, 1, 0, 1, 0], dtype="Int64"),
    })


def test_etiquetas_cubren_el_parquet():
    if not PARQUET.is_file():
        return
    assert E.validar_cobertura(pd.read_parquet(PARQUET).columns) == []


def test_metricas_ponderadas_perfectas():
    y = np.array([0, 1, 2, 3])
    p = np.eye(4)
    m = MW.calcular_metricas(y, p, np.ones(4))
    assert abs(m["accuracy"] - 1.0) < 1e-12 and abs(m["brier"]) < 1e-12


def test_indicadores_y_recuento_a_mano():
    d = _mini()
    ind = PV.indicadores(d)
    assert ind["menor5"].tolist() == [0, 1, 1, 0, 0, 1]
    assert ind["adulto_mayor"].tolist() == [0, 0, 1, 1, 0, 0]
    assert ind["pobreza"].tolist() == [0, 1, 1, 0, 1, 0]
    assert PV._base(d)["_recuento"].tolist() == [0, 2, 3, 1, 1, 1]   # noqa: SLF001


def test_recuento_suma_total_y_valores():
    d = _mini()
    rec = PV.tabla_recuento(d)
    assert rec["n"].tolist() == [1, 3, 1, 1] and int(rec["n"].sum()) == 6
    # prevalencia ponderada del recuento 1 (hogares 4, 5 y 6; pesos 2, 4, 4; IA = fies>=2 -> 0, 1, 0): 4 / 10
    f = rec[rec["recuento"] == 1].iloc[0]
    assert abs(f["prevalencia"] - 0.4) < 1e-12


def test_prevalencia_por_subgrupo_a_mano():
    sub = PV.tabla_subgrupos(_mini())
    con = sub[(sub["clave"] == "pobreza") & sub["grupo"].str.startswith("Con")].iloc[0]
    # pobres: hogares 2, 3 y 5 (pesos 1, 2, 4; IA 1, 1, 1) -> 100 %; n = 3
    assert con["n"] == 3 and abs(con["prevalencia"] - 1.0) < 1e-12


def test_pertenencia_hogar_y_datos_faltantes():
    p = PV.pertenencia_hogar({"n_menores5": 1, "n_adultos_mayores": 0, "pobreza_monetaria": 1})
    assert p["claves"] == ["menor5", "pobreza"] and p["recuento"] == 2 and p["desconocidas"] == []
    q = PV.pertenencia_hogar({"n_menores5": 1})
    assert q["recuento"] is None and set(q["desconocidas"]) == {"adulto_mayor", "pobreza"}


def test_linealizacion_coincide_con_reportes_fases():
    from fsai import reportes_fases as R
    d = _mini()
    a, b = PV.prevalencia_ic(d), R.prevalencia_ic(d)
    assert all(abs(x - y) < 1e-12 for x, y in zip(a[:3], b[:3])) and a[3] == b[3]


def test_valores_del_articulo_en_el_parquet():
    if not PARQUET.is_file():
        return
    d = pd.read_parquet(PARQUET)
    sub = PV.tabla_subgrupos(d)
    esperado = {"menor5": (30.0, 26.3, 5332), "adulto_mayor": (27.1, 26.7, 10418), "pobreza": (39.8, 23.5, 5860)}
    for k, (c, s, n) in esperado.items():
        con = sub[(sub["clave"] == k) & sub["grupo"].str.startswith("Con")].iloc[0]
        sin = sub[(sub["clave"] == k) & sub["grupo"].str.startswith("Sin")].iloc[0]
        assert (round(100 * con["prevalencia"], 1), round(100 * sin["prevalencia"], 1), int(con["n"])) == (c, s, n)
    assert int(PV.tabla_recuento(d)["n"].sum()) == int(d["fies_nivel"].notna().sum())
    assert round(100 * PV.perfil(d)["total"]["prevalencia"], 1) == 26.8


if __name__ == "__main__":
    fallos = 0
    for nombre, fn in sorted(globals().items()):
        if nombre.startswith("test_") and callable(fn):
            try:
                fn()
                print("[OK]", nombre)
            except AssertionError as exc:
                fallos += 1
                print("[FALLA]", nombre, exc)
    sys.exit(1 if fallos else 0)
