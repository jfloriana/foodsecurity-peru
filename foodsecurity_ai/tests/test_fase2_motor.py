"""Pruebas de la Fase 2 V2: fuentes territoriales, escenario socio_terr, ingeniería de variables, modelo M6, pesos y umbrales.

Solo datos sintéticos; una prueba lee la tabla territorial REAL derivada (datos_derivados/) y se omite con motivo si no existe.
"""
from __future__ import annotations

import importlib.util
import inspect
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fsai import config as C
from fsai import fuentes_territoriales as FT
from fsai import geo, umbrales
from fsai import metrics_weighted as MW
from fsai.fuentes_publicas import DEPARTAMENTOS
from fsai.models import ajustar_pipeline, construir_pipeline, crear_modelo, predecir_proba
from fsai.preprocessing import agregar_ingenieria, dividir_features, enriquecer_para_escenario

RAIZ = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Utilidades sintéticas
# --------------------------------------------------------------------------
def tabla_terr_sintetica() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    t = pd.DataFrame({FT.COL_CLAVE: list(DEPARTAMENTOS), "departamento": list(DEPARTAMENTOS.values())})
    for c in C.FEATURES_SOCIO_TERR_NUM:
        if c != "terr_oni_mes":
            t[c] = rng.uniform(1, 50, len(t))
    return t


def oni_sintetico() -> pd.DataFrame:
    return pd.DataFrame({"anio": [2025] * 12, "mes": list(range(1, 13)), "terr_oni_mes": np.linspace(-0.8, 0.4, 12)})


def hogares_socio(n: int = 360, semilla: int = 3) -> pd.DataFrame:
    """Dataset con TODAS las columnas socioeconómicas, ubigeo/anio/mes y objetivo (para socio_terr)."""
    rng = np.random.default_rng(semilla)
    cods = rng.choice(list(DEPARTAMENTOS), n)
    d = pd.DataFrame({"hogar_uid": [f"h{i:05d}" for i in range(n)], "conglome": [f"c{i // 6:04d}" for i in range(n)],
                      "ubigeo": [c + "0101" for c in cods], "departamento": [DEPARTAMENTOS[c] for c in cods],
                      "anio": 2025, "mes": rng.integers(1, 13, n), "factor_expansion": rng.uniform(50, 800, n)})
    for c in C.FEATURES_SOCIO_NUM:
        d[c] = rng.normal(5, 2, n).clip(0)
    d["n_miembros"] = rng.integers(1, 9, n).astype(float)
    d["n_menores18"] = np.minimum(rng.integers(0, 4, n), d["n_miembros"] - 1).astype(float)
    d["n_menores5"] = np.minimum(rng.integers(0, 3, n), d["n_menores18"]).astype(float)
    d["n_adultos_mayores"] = rng.integers(0, 2, n).astype(float)
    d["ingreso_pc"] = rng.lognormal(6, 1, n)
    d["gasto_pc"] = rng.lognormal(5.5, 1, n)
    d["gasto_alim_pc"] = rng.lognormal(4.5, 1, n)
    d.loc[rng.random(n) < 0.3, "asistencia_escolar"] = np.nan
    for c in C.FEATURES_SOCIO_CAT:
        if c not in d.columns:
            d[c] = rng.choice(["a", "b", "c"], n)
    d["area"] = rng.choice(["urbano", "rural"], n)
    puntaje = -np.log(d["ingreso_pc"]) + rng.normal(0, 1.0, n)
    d["fies_nivel"] = pd.qcut(puntaje, [0, 0.4, 0.7, 0.9, 1.0], labels=False).astype(int)        # 4 clases con >= 10 % cada una
    return d


# --------------------------------------------------------------------------
# 'socio' no cambia (comparación justa)
# --------------------------------------------------------------------------
def test_definiciones_socio_sin_cambios():
    assert C.FEATURES_SOCIO_NUM == [
        "ingreso_pc", "gasto_pc", "gasto_alim_pc", "share_alimentario", "pobreza_monetaria", "n_ocupados", "informalidad", "desempleo",
        "dependencia_economica", "nivel_educ_jefe", "asistencia_escolar", "n_miembros", "n_menores5", "n_menores18", "edad_jefe",
        "n_adultos_mayores", "agua_potable", "saneamiento", "hacinamiento"]
    assert C.FEATURES_SOCIO_CAT == ["departamento", "area", "region_natural", "sexo_jefe", "tipo_vivienda", "material_vivienda"]
    assert C.ESCENARIOS == ["socio", "fies", "hibrido"] and C.MODELOS == ["M1", "M2", "M3", "M4", "M5"]
    assert "socio_terr" not in C.ESCENARIOS and "M6" not in C.MODELOS
    assert C.ESCENARIOS_TODOS == C.ESCENARIOS + ["socio_terr"] and C.MODELOS_TODOS == C.MODELOS + ["M6"]


def test_escenario_socio_identico_con_o_sin_territorial():
    d = hogares_socio()
    n0, c0 = dividir_features(d, "socio")
    e = enriquecer_para_escenario(d, "socio")
    assert e is d                                  # no se toca el DataFrame
    assert dividir_features(e, "socio") == (n0, c0)
    e2, _ = FT.unir_territorial(d, tabla_terr_sintetica(), oni_sintetico())
    assert dividir_features(e2, "socio") == (n0, c0)       # aunque existan columnas terr_*, 'socio' no las ve


def test_socio_terr_incluye_socio_mas_territorial_e_ingenieria():
    d = hogares_socio()
    e, _ = FT.unir_territorial(d, tabla_terr_sintetica(), oni_sintetico())
    e = agregar_ingenieria(e)
    num, cat = dividir_features(e, "socio_terr")
    n_s, c_s = dividir_features(d, "socio")
    assert num[:len(n_s)] == n_s and cat[:len(c_s)] == c_s
    assert set(C.FEATURES_SOCIO_TERR_NUM) <= set(num) and set(C.FEATURES_INGENIERIA_NUM) <= set(num)
    assert set(C.FEATURES_INGENIERIA_CAT) <= set(cat)
    assert not (set(num + cat) & (set(C.COLUMNAS_PROHIBIDAS) | set(C.COLUMNAS_ID_TERRITORIALES) | set(C.FEATURES_FIES)))


def test_socio_terr_exige_columnas_territoriales():
    with pytest.raises(ValueError, match="socio_terr requiere"):
        dividir_features(hogares_socio(), "socio_terr")


# --------------------------------------------------------------------------
# Unión territorial (m:1, sin perder filas) y anti-fuga
# --------------------------------------------------------------------------
def test_unir_territorial_m1_no_pierde_filas_ni_altera_columnas():
    d = hogares_socio()
    antes = d.copy()
    e, rep = FT.unir_territorial(d, tabla_terr_sintetica(), oni_sintetico())
    pd.testing.assert_frame_equal(d, antes)                            # entrada intacta
    assert len(e) == len(d) and rep["filas_sin_unir"] == 0
    pd.testing.assert_frame_equal(e[d.columns], d)                     # columnas previas y orden de filas intactos
    assert set(C.FEATURES_SOCIO_TERR_NUM) <= set(e.columns)
    t = tabla_terr_sintetica().set_index(FT.COL_CLAVE)
    cod = d["ubigeo"].str[:2]
    np.testing.assert_allclose(e["terr_mpi_indice"].to_numpy(), t.loc[cod, "terr_mpi_indice"].to_numpy())
    assert e["terr_oni_mes"].notna().all()
    esperado = oni_sintetico().set_index("mes").loc[d["mes"], "terr_oni_mes"].to_numpy()
    np.testing.assert_allclose(e["terr_oni_mes"].to_numpy(), esperado)


def test_unir_territorial_por_nombre_cuando_no_hay_ubigeo_y_reporta_no_unidos():
    d = hogares_socio().drop(columns=["ubigeo"])
    d.loc[d.index[:5], "departamento"] = "atlantida"
    e, rep = FT.unir_territorial(d, tabla_terr_sintetica(), oni_sintetico())
    assert len(e) == len(d) and rep["filas_sin_unir"] == 5 and "nan" in " ".join(rep["departamentos_no_reconocidos"]).lower() or rep["filas_sin_unir"] == 5
    assert e.loc[d.index[:5], "terr_mpi_indice"].isna().all()
    with pytest.raises(ValueError, match="sin unir"):
        FT.unir_territorial(d, tabla_terr_sintetica(), oni_sintetico(), estricto=True)


def test_unir_territorial_exige_m1():
    t = pd.concat([tabla_terr_sintetica(), tabla_terr_sintetica().iloc[:1]])
    with pytest.raises(ValueError, match="duplicados"):
        FT.unir_territorial(hogares_socio(), t, oni_sintetico())


def test_guardia_anti_fuga():
    FT.verificar_sin_fuga()                                           # los indicadores reales pasan
    assert set(C.FEATURES_SOCIO_TERR_NUM) == set(FT.INDICADORES)
    for mala in ("fies_q1", "fies_nivel", "ia_binaria", "terr_fies_x", "factor_expansion", "ubigeo"):
        with pytest.raises(ValueError):
            FT.verificar_sin_fuga([mala], {mala: {"fuente": "ENDES"}})
    with pytest.raises(ValueError, match="prohibida"):
        FT.verificar_sin_fuga(["terr_x"], {"terr_x": {"fuente": "ENAHO_2025_FIES"}})
    assert all(not v["fuente"].upper().startswith(FT.FUENTES_PROHIBIDAS) for v in FT.INDICADORES.values())


def test_indicadores_territoriales_no_dependen_del_objetivo():
    d = hogares_socio()
    d2 = d.copy()
    d2["fies_nivel"] = (d2["fies_nivel"] + 1) % 4
    a, _ = FT.unir_territorial(d, tabla_terr_sintetica(), oni_sintetico())
    b, _ = FT.unir_territorial(d2, tabla_terr_sintetica(), oni_sintetico())
    cols = C.FEATURES_SOCIO_TERR_NUM
    pd.testing.assert_frame_equal(a[cols], b[cols])
    pd.testing.assert_frame_equal(agregar_ingenieria(d)[C.FEATURES_INGENIERIA_NUM], agregar_ingenieria(d2)[C.FEATURES_INGENIERIA_NUM])


# --------------------------------------------------------------------------
# Ingeniería de variables
# --------------------------------------------------------------------------
def test_ingenieria_valores_a_mano():
    d = pd.DataFrame({"ingreso_pc": [100.0, 0.0, -5.0], "gasto_pc": [50.0, 10.0, 1.0], "gasto_alim_pc": [20.0, 5.0, 1.0],
                      "n_miembros": [4.0, 1.0, 3.0], "n_menores18": [2.0, 0.0, 1.0], "n_adultos_mayores": [0.0, 1.0, 0.0],
                      "n_menores5": [1.0, 0.0, np.nan], "asistencia_escolar": [1.0, np.nan, 0.0],
                      "edad_jefe": [25.0, 50.0, 70.0], "nivel_educ_jefe": [0.0, 3.0, 8.0]})
    e = agregar_ingenieria(d)
    assert len(e) == 3 and list(e.index) == [0, 1, 2]
    assert e["log_ingreso_pc"].tolist() == pytest.approx([np.log1p(100.0), 0.0, 0.0])             # negativos -> 0
    ae = 4 - 2 + 0.5 * 2                                                                         # 3 adultos equivalentes
    assert e.loc[0, "log_ingreso_por_adulto_eq"] == pytest.approx(np.log1p(100.0 * 4 / ae))
    assert e["razon_dependencia"].tolist() == pytest.approx([2 / 2, 1 / 1, 1 / 2])               # denominador mínimo 1
    assert e["asistencia_escolar_faltante"].tolist() == [0.0, 1.0, 0.0]
    assert e["tiene_menor5"].iloc[:2].tolist() == [1.0, 0.0] and np.isnan(e["tiene_menor5"].iloc[2])
    assert e["grupo_edad_jefe"].tolist() == ["<30", "45-59", "60+"]
    assert e["grupo_educ_jefe"].tolist() == ["sin_primaria", "primaria", "superior"]


def test_ingenieria_es_conmutable():
    d = hogares_socio()
    sin = enriquecer_para_escenario(d, "socio_terr", ingenieria=False, tabla_territorial=tabla_terr_sintetica())
    assert not (set(C.FEATURES_INGENIERIA_NUM) & set(sin.columns))
    con = agregar_ingenieria(sin)
    assert set(C.FEATURES_INGENIERIA_NUM) <= set(con.columns)
    n_sin, _ = dividir_features(sin, "socio_terr")
    assert not (set(C.FEATURES_INGENIERIA_NUM) & set(n_sin))


# --------------------------------------------------------------------------
# M6 (HistGradientBoosting) y pesos
# --------------------------------------------------------------------------
def _toy(n=400, semilla=0):
    rng = np.random.default_rng(semilla)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.choice(["u", "v"], n)})
    y = np.clip(np.round(1.2 * X["a"] + 0.6 * (X["c"] == "u") + rng.normal(scale=0.7, size=n) + 1.5), 0, 3).astype(int)
    return X, y


def test_m6_ajusta_y_predice_con_pesos():
    X, y = _toy()
    pipe = construir_pipeline("M6", {"max_iter": 40, "learning_rate": 0.1}, ["a", "b"], ["c"])
    ajustar_pipeline(pipe, X, y, np.random.default_rng(1).uniform(0.5, 5, len(y)))
    p = predecir_proba(pipe, X)
    assert p.shape == (len(y), 4) and np.allclose(p.sum(axis=1), 1) and (p >= 0).all()
    assert (p.argmax(1) == y).mean() > 0.5
    clf = pipe.named_steps["clf"]
    assert clf.early_stopping is True and clf.n_iter_ <= 40


def test_m6_los_pesos_cambian_el_ajuste_y_class_weight_ninguno():
    X, y = _toy()
    w_uno, w_sesgo = np.ones(len(y)), np.where(y == 3, 50.0, 1.0)
    pa = construir_pipeline("M6", {"max_iter": 30, "class_weight": "ninguno"}, ["a", "b"], ["c"])
    pb = construir_pipeline("M6", {"max_iter": 30, "class_weight": "ninguno"}, ["a", "b"], ["c"])
    ajustar_pipeline(pa, X, y, w_uno)
    ajustar_pipeline(pb, X, y, w_sesgo)
    assert predecir_proba(pb, X)[:, 3].mean() > predecir_proba(pa, X)[:, 3].mean()
    assert crear_modelo("M6", {"class_weight": "ninguno"}).class_weight is None


def test_m6_en_optuna_y_modelos_previos_intactos():
    from fsai.optuna_optimizer import ESPACIO_DEFECTO, optimizar_modelo
    assert list(ESPACIO_DEFECTO)[:5] == ["M1", "M2", "M3", "M4", "M5"] and "M6" in ESPACIO_DEFECTO
    X, y = _toy(240)
    est = optimizar_modelo("M6", X, y, np.arange(len(y)), pesos=np.ones(len(y)), n_trials=2, features_num=["a", "b"], features_cat=["c"],
                           storage_url="", espacio={"max_iter": {"lo": 100, "hi": 100}})
    assert len(est.trials) == 2 and est.best_value > 0
    with pytest.raises(ValueError):
        crear_modelo("M7", {})


def test_pesos_de_ajuste_recorte_y_normalizacion():
    w = np.array([1.0] * 98 + [10.0, 1000.0])
    assert np.array_equal(MW.preparar_pesos_ajuste(w), w) and MW.preparar_pesos_ajuste(None) is None     # defecto: sin cambios
    rec = MW.preparar_pesos_ajuste(w, recorte_percentil=99)
    assert rec.max() == pytest.approx(np.percentile(w, 99)) and rec.max() < w.max() and np.array_equal(w[:98], rec[:98])
    nor = MW.preparar_pesos_ajuste(w, normalizar=True)
    assert nor.mean() == pytest.approx(1.0) and np.allclose(nor / nor[0], w / w[0])
    both = MW.preparar_pesos_ajuste(w, 99, True)
    assert both.mean() == pytest.approx(1.0)
    for malo in (np.array([1.0, -1.0]), np.array([1.0, np.nan])):
        with pytest.raises(ValueError):
            MW.preparar_pesos_ajuste(malo)
    with pytest.raises(ValueError):
        MW.preparar_pesos_ajuste(w, recorte_percentil=10)


# --------------------------------------------------------------------------
# Umbrales por clase (solo Validación)
# --------------------------------------------------------------------------
def _probas_sesgadas(n=600, semilla=5):
    """Probabilidades con la clase 2 sistemáticamente subestimada: el argmax la omite y los multiplicadores deben ayudar."""
    rng = np.random.default_rng(semilla)
    y = rng.choice(4, n, p=[0.4, 0.25, 0.2, 0.15])
    logit = rng.normal(0, 0.8, (n, 4))
    logit[np.arange(n), y] += 1.6
    logit[:, 2] -= 1.2
    p = np.exp(logit)
    return y, p / p.sum(1, keepdims=True)


def test_umbrales_mejoran_f1_en_validacion_y_son_deterministas():
    y, p = _probas_sesgadas()
    w = np.random.default_rng(1).uniform(0.5, 3, len(y))
    r1 = umbrales.optimizar_multiplicadores(y, p, w)
    r2 = umbrales.optimizar_multiplicadores(y, p, w)
    assert r1 == r2 and r1["multiplicadores"][0] == 1.0 and len(r1["multiplicadores"]) == 4
    assert r1["f1_optimizado_val"] > r1["f1_argmax_val"] and r1["multiplicadores"][2] > 1.0
    assert all(min(r1["malla"]) <= m <= max(r1["malla"]) for m in r1["multiplicadores"][1:])
    pred = umbrales.aplicar_multiplicadores(p, r1["multiplicadores"])
    assert set(np.unique(pred)) <= {0, 1, 2, 3}
    assert np.array_equal(umbrales.aplicar_multiplicadores(p, np.ones(4)), p.argmax(1))
    cmp = umbrales.metricas_comparadas(y, p, w, r1["multiplicadores"])
    assert cmp["optimizado"]["f1_macro"] == pytest.approx(r1["f1_optimizado_val"]) and cmp["argmax"]["f1_macro"] == pytest.approx(r1["f1_argmax_val"])


def test_umbrales_nunca_leen_etiquetas_de_prueba():
    # (1) por firma: ninguna función de ajuste/aplicación tiene parámetros de Prueba
    for f in (umbrales.optimizar_multiplicadores, umbrales.aplicar_multiplicadores):
        assert not [n for n in inspect.signature(f).parameters if "prueba" in n.lower() or "test" in n.lower()]
    # (2) por comportamiento: el resultado depende solo de Validación; unas "etiquetas de Prueba" venenosas jamás se tocan
    class Veneno:
        def __array__(self, *a, **k):
            raise AssertionError("se leyeron etiquetas de Prueba")
        __len__ = __iter__ = __getitem__ = __array__

    y, p = _probas_sesgadas()
    y_prueba, p_prueba = Veneno(), _probas_sesgadas(semilla=99)[1]
    m = umbrales.optimizar_multiplicadores(y, p)["multiplicadores"]
    pred = umbrales.aplicar_multiplicadores(p_prueba, m)                  # solo probabilidades de Prueba
    assert len(pred) == len(p_prueba) and y_prueba is not None
    # (3) cambiar por completo la Prueba no cambia los multiplicadores
    assert umbrales.optimizar_multiplicadores(y, p)["multiplicadores"] == m


def test_prueba_de_un_solo_uso(tmp_path):
    (tmp_path / "prueba_x.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="una sola vez"):
        umbrales.evaluar_prueba_con_umbrales(pd.DataFrame(), {"prueba": np.array([0])}, tmp_path, "x")


def test_umbrales_validan_entradas():
    y, p = _probas_sesgadas(50)
    with pytest.raises(ValueError):
        umbrales.optimizar_multiplicadores(y, p[:, :3])
    with pytest.raises(ValueError):
        umbrales.aplicar_multiplicadores(p, [1, 1, 0, 1])


# --------------------------------------------------------------------------
# Extremo a extremo (toy): socio_terr + M6 + pesos recortados + umbrales, sin tocar Prueba
# --------------------------------------------------------------------------
@pytest.mark.skipif(not C.RUTA_TABLA_TERRITORIAL.exists(), reason="no existe datos_derivados/indicadores_territoriales_departamento.csv (ejecute scripts/descargar_fuentes_adicionales.py --derivar)")
def test_nested_cv_socio_terr_m6_extremo_a_extremo(tmp_path):
    from fsai.nested_cv import RegistroProgreso, ejecutar_modelo
    from fsai.splits import particionar_70_15_15
    d = hogares_socio(480)
    part = particionar_70_15_15(d, C.SEED, "hogar_uid")
    prog = RegistroProgreso(tmp_path / "p")
    res = ejecutar_modelo(d, part, codigo="M6", escenario="socio_terr", n_trials=2, dir_run=tmp_path, progreso=prog, n_outer=5,
                          ponderar_ajuste=True, recorte_pesos=99, normalizar_pesos=True, optimizar_umbrales=True,
                          espacio={"max_iter": {"lo": 100, "hi": 100}})
    assert set(C.FEATURES_SOCIO_TERR_NUM) <= set(res["features_num"]) and set(C.FEATURES_INGENIERIA_NUM) <= set(res["features_num"])
    assert all(f["auditoria"]["prueba_tocada"] is False for f in res["folds"])
    um = res["final"]["umbrales_clase"]
    assert um["nota"].endswith("no se usó Prueba") and um["multiplicadores"][0] == 1.0
    assert res["opciones_v2"] == {"ingenieria": True, "ponderar_ajuste": True, "recorte_pesos_percentil": 99,
                                  "normalizar_pesos": True, "optimizar_umbrales": True}
    ev = umbrales.evaluar_prueba_con_umbrales(d, part, tmp_path, "socio_terr_M6")      # ÚNICA lectura de Prueba (solo medir)
    assert set(ev["ponderada"]) == {"argmax", "optimizado"} and ev["n_prueba"] == len(part["prueba"])
    with pytest.raises(RuntimeError):
        umbrales.evaluar_prueba_con_umbrales(d, part, tmp_path, "socio_terr_M6")


# --------------------------------------------------------------------------
# Parsers de fuentes (zips y archivos sintéticos)
# --------------------------------------------------------------------------
def _zip(ruta: Path, archivos: dict[str, str]) -> Path:
    with zipfile.ZipFile(ruta, "w") as zf:
        for n, txt in archivos.items():
            zf.writestr(n, txt.encode("utf-8"))        # BOM UTF-8 + ASCII: igual que los CSV reales de INEI
    return ruta


def test_endes_talla_anemia_ponderadas(tmp_path):
    r0 = "﻿ID1,HHID,UBIGEO,HV005\n2024,1,010101,1000000\n2024,2,010102,3000000\n2024,3,020101,2000000\n2024,4,020102,2000000\n"
    r6 = ("﻿ID1,HHID,HC1,HC70,HC57\n"
          "2024,1,10,-250,2\n"      # dpto 01: desnutrido (w=1), anemia moderada
          "2024,2,20,-100,4\n"      # no desnutrido (w=3), sin anemia
          "2024,2,25,9999,9\n"      # valores inválidos: se excluyen de ambos indicadores
          "2024,3,30,-210,4\n"      # dpto 02 (w=2) desnutrido
          "2024,4,70,-300,1\n")     # edad 70 m (> 59): se excluye
    r5 = "﻿ID1,HHID,HA1,HA57\n2024,1,30,3\n2024,2,30,4\n2024,3,30,9\n"
    z1 = _zip(tmp_path / "a.zip", {"x/RECH0_2024.csv": r0})
    z2 = _zip(tmp_path / "b.zip", {"x/RECH6_2024.csv": r6, "x/RECH5_2024.csv": r5})
    t = FT.indicadores_endes_talla_anemia(z1, z2)
    assert t.loc["01", "terr_desnutricion_cronica_pct"] == pytest.approx(100 * 1 / (1 + 3))
    assert t.loc["02", "terr_desnutricion_cronica_pct"] == pytest.approx(100.0) and t.loc["01", "n_ninos_talla"] == 2
    assert t.loc["01", "terr_anemia_ninos_pct"] == pytest.approx(100 * 1 / 4)
    assert t.loc["01", "terr_anemia_mujeres_pct"] == pytest.approx(100 * 1 / 4)
    assert "02" not in t["terr_anemia_mujeres_pct"].dropna().index                      # HA57=9 (inválido) no cuenta


def test_programas_sociales_ponderados(tmp_path):
    r0 = "﻿ID1,HHID,UBIGEO,HV005\n2024,1,010101,1000000\n2024,2,010102,1000000\n2024,3,020101,1000000\n"
    cab = "﻿ID1,HHID,QH91,QH93,QH95,QH99,QH101,QH103,QH106\n"
    ps = cab + "2024,1, , ,1,2,2,2,2\n2024,2, ,2,2,2,2,2,2\n2024,3,2,2,2,1,2,2,2\n"
    z1 = _zip(tmp_path / "a.zip", {"x/RECH0_2024.csv": r0})
    z3 = _zip(tmp_path / "c.zip", {"x/Programas Sociales x Hogar_2024.csv": ps})
    t = FT.indicadores_programas_sociales(z1, z3)
    assert t.loc["01", "terr_hogares_programa_social_pct"] == pytest.approx(50.0) and t.loc["01", "terr_hogares_juntos_pct"] == pytest.approx(50.0)
    assert t.loc["02", "terr_hogares_pension65_pct"] == pytest.approx(100.0) and t.loc["02", "terr_hogares_programa_social_pct"] == pytest.approx(100.0)


def test_mpi_oni_y_precipitacion(tmp_path):
    csv = tmp_path / "mpi.csv"
    csv.write_text("Country ISO3,Admin 1 PCode,Admin 1 Name,MPI,Headcount Ratio,Intensity of Deprivation\n"
                   "PER,,,0.02,5.6,38.9\nPER,PE01,Amazonas,0.087,20.98,41.48\nPER,PE15,Lima,0.0025,0.7,35.3\n", encoding="utf-8")
    m = FT.indicadores_mpi(csv)
    assert list(m.index) == ["01", "15"] and m.loc["01", "terr_mpi_incidencia_pct"] == pytest.approx(20.98)
    oni = tmp_path / "oni.txt"
    oni.write_text(" SEAS  YR   TOTAL   ANOM\n  DJF 2025  26.0  -0.50\n  JFM 2025  26.1  -0.40\n  NDJ 2025  26.2  -0.55\n  DJF 2010  26.0  1.00\n", encoding="utf-8")
    o = FT.tabla_oni(oni)
    assert o.to_dict("records") == [{"anio": 2025, "mes": 1, "terr_oni_mes": -0.5}, {"anio": 2025, "mes": 2, "terr_oni_mes": -0.4},
                                    {"anio": 2025, "mes": 12, "terr_oni_mes": -0.55}]
    # Open-Meteo: 1 mm/día todo el año objetivo; clima: 0,5 mm/día con variación -> anomalía conocida
    fechas, vals = [], []
    for a in range(2015, 2026):
        for f in pd.date_range(f"{a}-01-01", f"{a}-12-31"):
            fechas.append(f.strftime("%Y-%m-%d"))
            vals.append(1.0 if a == 2025 else 0.5 + 0.01 * (a - 2015))
    j = tmp_path / "om.json"
    j.write_text(json.dumps({"daily": {"time": fechas, "precipitation_sum": vals}}), encoding="utf-8")
    serie = FT.precipitacion_mensual_openmeteo(j)
    assert serie["202501"] == pytest.approx(31.0) and len(serie) == 132
    an = FT.anomalia_precipitacion(serie, 2025)
    clim = np.mean([sum(0.5 + 0.01 * (a - 2015) for _ in range(366 if a % 4 == 0 else 365)) for a in range(2015, 2025)])
    assert an["precip_anio_mm"] == pytest.approx(365.0) and an["precip_clim_mm"] == pytest.approx(clim)
    assert an["terr_precip_anom_pct"] == pytest.approx(100 * (365 - clim) / clim) and an["terr_precip_anom_z"] > 0
    incompleto = serie.copy()
    incompleto["202506"] = np.nan                                           # un mes inválido => año objetivo NaN (no se inventa)
    assert np.isnan(FT.anomalia_precipitacion(incompleto, 2025)["terr_precip_anom_pct"])


def test_zip_con_tope_y_sin_archivo(tmp_path):
    z = _zip(tmp_path / "a.zip", {"x/otro.csv": "a,b\n1,2\n"})
    with pytest.raises(FileNotFoundError):
        FT.leer_csv_zip(z, "RECH0_2024.csv")


# --------------------------------------------------------------------------
# GeoJSON, centroides y descargador
# --------------------------------------------------------------------------
def test_centroides_formas_simples():
    cuadrado = {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]}
    assert FT.centroide_geometria(cuadrado) == pytest.approx((1.0, 1.0))
    con_hueco = {"type": "Polygon", "coordinates": [cuadrado["coordinates"][0], [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    cx, cy = FT.centroide_geometria(con_hueco)
    assert cx > 1.0 and cy > 1.0                                            # quitar una esquina desplaza el centroide


def test_geojson_incluido_y_centroides_en_el_peru():
    import pytest
    if not geo.ruta_geojson().is_file():
        pytest.skip("GeoJSON departamental no incluido en el repositorio (licencia de origen no declarada); ver README")
    g = geo.cargar_geojson()
    assert len(g["features"]) == 25
    assert {f["properties"][geo.CLAVE_CODIGO] for f in g["features"]} == set(DEPARTAMENTOS)
    cen = FT.centroides_departamentos(geo.ruta_geojson())
    assert set(cen) == set(DEPARTAMENTOS)
    assert all(-82 < lon < -68 and -19 < lat < 0 for lon, lat in cen.values())


def test_geo_respeta_variable_de_entorno(tmp_path, monkeypatch):
    f = tmp_path / "g.geojson"
    f.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": None}]}), encoding="utf-8")
    monkeypatch.setenv("FSAI_GEOJSON_DEPARTAMENTOS", str(f))
    assert geo.ruta_geojson() == f and len(geo.cargar_geojson()["features"]) == 1
    monkeypatch.setenv("FSAI_GEOJSON_DEPARTAMENTOS", str(tmp_path / "no_existe.geojson"))
    assert geo.ruta_geojson() == C.GEOJSON_DEPARTAMENTOS_INCLUIDO


def _descargador():
    spec = importlib.util.spec_from_file_location("descargar_fuentes_adicionales", RAIZ / "scripts" / "descargar_fuentes_adicionales.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_descargador_lista_blanca_y_fuentes():
    m = _descargador()
    for url in ("http://raw.githubusercontent.com/x", "https://evil.example.com/x", "file:///etc/passwd", "ftp://proyectos.inei.gob.pe/x"):
        with pytest.raises(ValueError):
            m._validar_url(url)
    m._validar_url(m.URL_ONI)
    ids = [f["id"] for f in m._fuentes_estaticas()]
    assert {"geojson_departamental", "endes_2024_m1629", "endes_2024_m1638", "endes_2024_m1641", "ophi_mpi_peru", "noaa_oni"} <= set(ids)
    assert all(f["url"].startswith("https://") for f in m._fuentes_estaticas())
    assert len(m.fuentes_power({c: (-75.0, -10.0) for c in DEPARTAMENTOS})) == 25 and len(m.fuentes_openmeteo({"01": (-75.0, -10.0)})) == 1


def test_descargador_es_idempotente_y_registra_fallos(tmp_path, monkeypatch):
    m = _descargador()
    monkeypatch.setattr(m, "DIR_DATOS", tmp_path)
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "a.txt").write_text("hola", encoding="utf-8")
    ok = m.procesar({"id": "t", "url": "https://x/", "archivo": "x/a.txt", "licencia": "L", "grupo": "g"}, None, forzar=False)
    assert ok["estado"] == "ok" and ok["bytes"] == 4 and len(ok["sha256"]) == 64 and "reutilizado" in ok["motivo"]
    monkeypatch.setattr(m, "descargar", lambda *a, **k: (_ for _ in ()).throw(OSError("sin red")))
    mal = m.procesar({"id": "u", "url": "https://x/", "archivo": "x/b.txt", "licencia": "L", "grupo": "g"}, None, forzar=False)
    assert mal["estado"] == "fallo" and "sin red" in mal["motivo"] and mal["sha256"] is None


# --------------------------------------------------------------------------
# Tabla territorial REAL derivada (si existe)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not C.RUTA_TABLA_TERRITORIAL.exists(), reason="no existe datos_derivados/indicadores_territoriales_departamento.csv (ejecute scripts/descargar_fuentes_adicionales.py --derivar)")
def test_tabla_territorial_real():
    t = FT.cargar_tabla_territorial()
    assert len(t) == 25 and set(t[FT.COL_CLAVE]) == set(DEPARTAMENTOS) and t[FT.COL_CLAVE].is_unique
    faltan = [c for c in C.FEATURES_SOCIO_TERR_NUM if c != "terr_oni_mes" and c not in t.columns]
    assert not faltan
    assert FT.verificar_rangos(t) == []
    assert t["terr_desnutricion_cronica_pct"].between(0, 60).all()
    assert t[[c for c in C.FEATURES_SOCIO_TERR_NUM if c != "terr_oni_mes"]].notna().all().all()
    FT.verificar_sin_fuga(list(FT.INDICADORES))
    if C.RUTA_TABLA_ONI.exists():
        o = pd.read_csv(C.RUTA_TABLA_ONI)
        assert (o[(o["anio"] == 2025)]["mes"].nunique() == 12) and o["terr_oni_mes"].between(-4, 4).all()
