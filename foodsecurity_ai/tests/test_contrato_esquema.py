"""Contrato de esquema de datos (contrato_esquema.py) con datos sintéticos."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fsai import config as C
from fsai import contrato_esquema as CE
from fsai import datasets as D


def _df_valido(n: int = 30) -> pd.DataFrame:
    """Un dataset sintético que cumple el contrato por defecto."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "hogar_uid": [f"h{i}" for i in range(n)],
        "conglome": [f"c{i // 3}" for i in range(n)],
        C.COL_PESO: rng.uniform(10, 300, n),
        "departamento": rng.choice(["LIMA", "CUSCO"], n),
        "area": rng.choice(["U", "R"], n),
        "region_natural": rng.choice(["costa", "sierra", "selva"], n),
        "sexo_jefe": rng.choice(["H", "M"], n),
        "tipo_vivienda": rng.integers(1, 7, n),
        "material_vivienda": rng.integers(1, 9, n),
    })
    for i in range(1, 9):
        df[f"fies_q{i}"] = rng.integers(0, 2, n)
    df["fies_raw_score"] = df[[f"fies_q{i}" for i in range(1, 9)]].sum(axis=1)
    df[C.COL_OBJETIVO] = rng.integers(0, 4, n)
    df[C.COL_BINARIA] = (df[C.COL_OBJETIVO] >= 2).astype(int)
    for col in C.FEATURES_SOCIO_NUM:
        df[col] = rng.integers(0, 2, n) if col in ("pobreza_monetaria", "desempleo", "agua_potable", "saneamiento") else rng.uniform(0, 1, n)
    df["nivel_educ_jefe"] = rng.integers(0, 12, n)
    return df


def test_dataset_valido_no_tiene_errores():
    inf = CE.validar_esquema(_df_valido())
    assert inf.ok, inf.a_dataframe().to_string()
    assert inf.n_filas == 30 and inf.errores == []


def test_columna_obligatoria_faltante():
    inf = CE.validar_esquema(_df_valido().drop(columns=["factor_expansion", "fies_q3"]))
    assert not inf.ok
    faltantes = {h.columna for h in inf.errores if h.codigo == "columna_faltante"}
    assert faltantes == {"factor_expansion", "fies_q3"}


def test_columna_opcional_ausente_no_es_error():
    assert "anio" not in _df_valido().columns
    assert "columna_faltante" not in CE.validar_esquema(_df_valido()).codigos()


def test_fuera_de_rango_y_conteo():
    df = _df_valido()
    df.loc[0:2, "nivel_educ_jefe"] = 99
    h = [x for x in CE.validar_esquema(df).errores if x.codigo == "fuera_de_rango"]
    assert len(h) == 1 and h[0].columna == "nivel_educ_jefe" and h[0].n_afectados == 3


def test_peso_cero_o_negativo_es_error_por_limite_exclusivo():
    df = _df_valido()
    df.loc[0, C.COL_PESO] = 0.0
    df.loc[1, C.COL_PESO] = -5.0
    h = [x for x in CE.validar_esquema(df).errores if x.columna == C.COL_PESO]
    assert h and h[0].n_afectados == 2


def test_categoria_no_permitida():
    df = _df_valido()
    df.loc[0, "area"] = "X"
    df.loc[1, "region_natural"] = "costa"      # permitida
    h = [x for x in CE.validar_esquema(df).errores if x.codigo == "categoria_no_permitida"]
    assert [x.columna for x in h] == ["area"] and h[0].n_afectados == 1


def test_binaria_con_valor_invalido_y_dtype_texto():
    df = _df_valido()
    df.loc[0, "fies_q1"] = 2
    df["edad_jefe"] = df["edad_jefe"].astype(str)
    df.loc[0, "gasto_pc"] = np.nan
    df["gasto_pc"] = df["gasto_pc"].astype(object)
    df.loc[1, "gasto_pc"] = "abc"
    inf = CE.validar_esquema(df)
    assert any(h.columna == "fies_q1" and h.codigo == "no_binaria" for h in inf.errores)
    assert any(h.columna == "edad_jefe" and h.codigo == "dtype_convertible" for h in inf.advertencias)   # solo advertencia
    assert any(h.columna == "gasto_pc" and h.codigo == "dtype_invalido" for h in inf.errores)


def test_nulos_no_permitidos_y_exceso_de_nulos():
    df = _df_valido()
    df.loc[0, "hogar_uid"] = None
    df.loc[:20, "ingreso_pc"] = np.nan          # 21 de 30 = 70 % > 50 %
    inf = CE.validar_esquema(df)
    assert any(h.columna == "hogar_uid" and h.codigo == "nulos_no_permitidos" for h in inf.errores)
    assert any(h.columna == "ingreso_pc" and h.codigo == "muchos_nulos" for h in inf.advertencias)
    assert not any(h.columna == "ingreso_pc" for h in inf.errores)


def test_duplicados_en_hogar_uid():
    df = _df_valido()
    df.loc[1, "hogar_uid"] = df.loc[0, "hogar_uid"]
    assert any(h.codigo == "duplicados" and h.n_afectados == 1 for h in CE.validar_esquema(df).errores)


def test_no_lanza_excepciones_con_entradas_malas():
    assert CE.validar_esquema("no soy un dataframe").codigos() == ["no_es_dataframe"]
    assert "sin_filas" in CE.validar_esquema(_df_valido().iloc[0:0]).codigos()
    dup = pd.DataFrame([[1, 2]], columns=["a", "a"])
    assert CE.validar_esquema(dup).codigos() == ["columnas_duplicadas"]
    assert "mapa_invalido" in CE.validar_esquema(_df_valido(), mapa_columnas=["no", "es", "dict"]).codigos()


def test_modo_estricto_advierte_columnas_extra():
    df = _df_valido().assign(columna_rara=1)
    assert "columnas_extra" not in CE.validar_esquema(df).codigos()
    inf = CE.validar_esquema(df, estricto=True)
    assert "columnas_extra" in inf.codigos(CE.ADVERTENCIA) and inf.ok


def test_mapa_de_columnas_permite_validar_codigos_de_otro_anio():
    df = _df_valido().rename(columns={"factor_expansion": "FACTOR07", "area": "ESTRATO_U_R"})
    sin_mapa = CE.validar_esquema(df)
    assert not sin_mapa.ok and {"factor_expansion", "area"} <= {h.columna for h in sin_mapa.errores}
    con_mapa = CE.validar_esquema(df, mapa_columnas={"factor07": "factor_expansion", "ESTRATO_U_R": "area"})
    assert con_mapa.ok
    assert con_mapa.columnas_renombradas == {"FACTOR07": "factor_expansion", "ESTRATO_U_R": "area"}


def test_mapa_no_sobrescribe_una_columna_existente():
    df = _df_valido().assign(PESO_OTRO=1.0)
    df2, ren, h = CE.aplicar_mapa_columnas(df, {"PESO_OTRO": "factor_expansion"})
    assert ren == {} and "PESO_OTRO" in df2.columns
    assert h and h[0].codigo == "mapa_conflicto"


def test_mapa_se_toma_de_config_cuando_no_se_pasa(monkeypatch):
    monkeypatch.setattr(C, "MAPA_COLUMNAS", {"FACTOR07": "factor_expansion"})
    df = _df_valido().rename(columns={"factor_expansion": "FACTOR07"})
    assert CE.validar_esquema(df).ok


def test_mapa_en_config_se_carga_desde_variable_de_entorno(tmp_path, monkeypatch):
    ruta = tmp_path / "mapa.json"
    ruta.write_text(json.dumps({"FACTOR07": "factor_expansion"}), encoding="utf-8")
    monkeypatch.setenv("FSAI_MAPA_COLUMNAS", str(ruta))
    assert C._cargar_mapa_columnas() == {"FACTOR07": "factor_expansion"}
    ruta.write_text("[1, 2]", encoding="utf-8")                    # JSON inválido para el contrato: se ignora
    assert C._cargar_mapa_columnas() == {}
    monkeypatch.delenv("FSAI_MAPA_COLUMNAS")
    assert C._cargar_mapa_columnas() == {}


def test_leer_tabla_aplica_el_mapa(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MAPA_COLUMNAS", {"FACTOR07": "factor_expansion"})
    ruta = tmp_path / "x.csv"
    pd.DataFrame({"FACTOR07": [1.0, 2.0], "otra": [1, 2]}).to_csv(ruta, index=False)
    assert list(D.leer_tabla(ruta).columns) == ["factor_expansion", "otra"]


def test_contrato_por_defecto_deriva_de_config_y_rangos():
    c = CE.contrato_defecto()
    nombres = {x.nombre for x in c.columnas}
    assert set(C.FEATURES_SOCIO) <= nombres and set(C.FEATURES_FIES) <= nombres
    por = {x.nombre: x for x in c.columnas}
    assert (por["fies_nivel"].minimo, por["fies_nivel"].maximo) == D.RANGOS_CONTRATO["fies_nivel"]
    assert por["area"].categorias == ("U", "R")
    assert por["anio"].obligatoria is False


def test_contrato_round_trip_dict_y_validacion_de_reglas():
    c = CE.contrato_defecto()
    c2 = CE.ContratoEsquema.desde_dict(json.loads(json.dumps(c.a_dict())))
    assert c2 == c
    with pytest.raises(ValueError):
        CE.ColumnaContrato("x", tipo="inexistente")
    with pytest.raises(ValueError):
        CE.ColumnaContrato("x", minimo=5, maximo=1)
    with pytest.raises(ValueError):
        CE.ContratoEsquema("dup", (CE.ColumnaContrato("a"), CE.ColumnaContrato("a")))


def test_contrato_propio_minimo():
    contrato = CE.ContratoEsquema("mini", (CE.ColumnaContrato("x", "entero", minimo=0, maximo=3),
                                           CE.ColumnaContrato("y", "categorica", categorias=("a", "b"))))
    inf = CE.validar_esquema(pd.DataFrame({"x": [0, 1, 5, 2.5], "y": ["a", "b", "c", "a"]}), contrato)
    assert sorted(inf.codigos(CE.ERROR)) == ["categoria_no_permitida", "fuera_de_rango", "no_entero"]
    assert inf.a_dict()["ok"] is False and "error" in inf.resumen()
    assert list(inf.a_dataframe().columns) == ["severidad", "columna", "codigo", "mensaje", "n_afectados"]


def test_nombres_obligatorios_faltantes_solo_con_cabecera():
    faltan = CE.nombres_obligatorios_faltantes(["hogar_uid", "conglome"])
    assert "factor_expansion" in faltan and "hogar_uid" not in faltan
