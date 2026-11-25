"""Etiquetas (etiquetas.py): cobertura del diccionario sobre las columnas analíticas del proyecto."""
from __future__ import annotations

from fsai import config as C
from fsai import etiquetas as E


def test_validar_cobertura_vacio_para_columnas_del_protocolo():
    columnas = (C.FEATURES_FIES + C.FEATURES_SOCIO + [C.COL_OBJETIVO, C.COL_BINARIA, C.COL_PESO, C.COL_GRUPO]
                + C.LLAVES_INEI)
    assert E.validar_cobertura(columnas) == []


def test_validar_cobertura_reporta_columnas_desconocidas():
    assert E.validar_cobertura(["ingreso_pc", "columna_inventada_xyz"]) == ["columna_inventada_xyz"]


def test_etq_devuelve_texto_no_vacio_para_conocidas():
    for col in ("ingreso_pc", "fies_q1", "departamento"):
        assert isinstance(E.etq(col), str) and E.etq(col)
