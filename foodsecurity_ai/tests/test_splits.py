"""Pruebas de la partición 70/15/15 (splits.py) y de la validación geográfica V2 (validation.py)."""
from __future__ import annotations

import numpy as np
import pytest

from fsai import splits as S
from fsai import validation as V
from conftest import hogares_sinteticos
from fsai.config import NOMBRES_PARTICION


def test_particion_exhaustiva_y_sin_fuga_de_hogar(df_hogares):
    part = S.particionar_70_15_15(df_hogares, semilla=42)
    ver = S.verificar_particion(df_hogares, part)
    assert ver["exhaustiva"] is True
    assert ver["cruces_hogar_uid_total"] == 0          # ningún hogar cruza particiones
    assert sum(ver["tamanos"].values()) == len(df_hogares)
    # el dataset tiene 2 filas por hogar: si la agrupación fallara, habría cruces
    assert df_hogares["hogar_uid"].duplicated().any()


def test_proporciones_cercanas_a_70_15_15(df_hogares):
    part = S.particionar_70_15_15(df_hogares, semilla=42)
    prop = S.verificar_particion(df_hogares, part)["proporciones"]
    assert prop["desarrollo"] == pytest.approx(0.70, abs=0.03)
    assert prop["validacion"] == pytest.approx(0.15, abs=0.03)
    assert prop["prueba"] == pytest.approx(0.15, abs=0.03)


def test_particion_reproducible_con_la_misma_semilla(df_hogares):
    a = S.particionar_70_15_15(df_hogares, semilla=123)
    b = S.particionar_70_15_15(df_hogares, semilla=123)
    for k in NOMBRES_PARTICION:
        assert np.array_equal(a[k], b[k])
    assert S.huella_particion(df_hogares, a, 123, "hogar_uid") == S.huella_particion(df_hogares, b, 123, "hogar_uid")


def test_semilla_distinta_cambia_la_particion(df_hogares):
    a = S.particionar_70_15_15(df_hogares, semilla=1)
    b = S.particionar_70_15_15(df_hogares, semilla=2)
    assert not np.array_equal(a["prueba"], b["prueba"])


def test_verificar_particion_detecta_fuga_provocada(df_hogares):
    part = S.particionar_70_15_15(df_hogares, semilla=42)
    # se mueve una fila de Prueba a Desarrollo SIN mover a su gemelo del mismo hogar -> fuga
    i = int(part["prueba"][0])
    mala = {"desarrollo": np.sort(np.append(part["desarrollo"], i)),
            "validacion": part["validacion"], "prueba": part["prueba"][1:]}
    assert S.verificar_particion(df_hogares, mala)["cruces_hogar_uid_total"] > 0


def test_minimo_de_hogares_y_columna_inexistente(df_hogares):
    with pytest.raises(ValueError):
        S.particionar_70_15_15(df_hogares.head(100))
    with pytest.raises(KeyError):
        S.particionar_70_15_15(df_hogares, agrupar_por="no_existe")


def test_guardar_y_cargar_particion_round_trip(tmp_path):
    # cargar_particion reconstruye por hogar_uid: el dataset real tiene UNA fila por hogar
    df_hogares = hogares_sinteticos(n_hogares=400, filas_por_hogar=1)
    part = S.particionar_70_15_15(df_hogares, semilla=5)
    ruta = tmp_path / "p.json"
    huella = S.guardar_particion(df_hogares, part, ruta, 5, "hogar_uid")
    cargada, meta = S.cargar_particion(df_hogares, ruta)
    assert meta["huella"] == huella
    for k in NOMBRES_PARTICION:
        assert np.array_equal(part[k], cargada[k])


# ---------------- validación geográfica V2 (ya implementada en validation.py) ----------------
def test_lodo_deja_un_departamento_fuera_por_pliegue(df_hogares):
    parts = V.generar_particiones(df_hogares, "V2_lodo")
    assert len(parts) == df_hogares["departamento"].nunique()
    for tr, te in parts:
        assert df_hogares["departamento"].iloc[te].nunique() == 1
        assert df_hogares["departamento"].iloc[tr].nunique() == df_hogares["departamento"].nunique() - 1
        assert set(df_hogares["hogar_uid"].iloc[tr]).isdisjoint(df_hogares["hogar_uid"].iloc[te])


def test_macro_costa_sierra_selva(df_hogares):
    parts = V.generar_particiones(df_hogares, "V2", variante="macro")
    assert sorted(p.estrato for p in parts) == ["costa", "selva", "sierra"]
    for p in parts:
        assert set(df_hogares["region_natural"].iloc[p[1]]) == {p.estrato}
        assert p.estrato not in set(df_hogares["region_natural"].iloc[p[0]])


def test_v5_no_es_particion(df_hogares):
    with pytest.raises(ValueError):
        V.generar_particiones(df_hogares, "V5")


def test_v1_sin_solape_y_cobertura(df_hogares):
    parts = V.generar_particiones(df_hogares, "V1")
    assert len(parts) == 5 * 4
    V.verificar_sin_solape(parts, df_hogares["hogar_uid"].to_numpy())
