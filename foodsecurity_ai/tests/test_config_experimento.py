"""Exportar/importar configuración de experimento (config_experimento.py)."""
from __future__ import annotations

import copy
import json

import pytest

from fsai import config as C
from fsai import config_experimento as CX

ESPACIO = {"M1": {"C": {"tipo": "float", "lo": 0.001, "hi": 100.0, "log": True},
                  "penalty": {"tipo": "cat", "opciones": ["l2"]}},
           "M2": {"n_estimators": {"tipo": "int", "lo": 100, "hi": 600, "step": 50}}}


def _exportar(ruta=None, **extra):
    base = dict(run_id="corrida_prueba", escenario="socio", modelos=["M1", "M2"], semillas=[42, 43], n_trials=100,
                esquema_particion="V1", hash_dataset="ab" * 32, espacios_hiperparametros=copy.deepcopy(ESPACIO),
                notas="prueba sintética")
    base.update(extra)
    return CX.exportar_config(ruta, **base)


def test_round_trip_por_archivo(tmp_path):
    ruta = tmp_path / "sub" / "config.json"
    cfg = _exportar(ruta)
    assert ruta.is_file()
    assert json.loads(ruta.read_text(encoding="utf-8")) == cfg          # un solo archivo JSON
    leido = CX.importar_config(ruta)
    assert leido.run_id == "corrida_prueba" and leido.escenario == "socio"
    assert leido.modelos == ["M1", "M2"] and leido.semillas == [42, 43] and leido.n_trials == 100
    assert leido.espacios_hiperparametros == ESPACIO and leido.hash_dataset == "ab" * 32
    assert leido.a_dict() == cfg                                        # re-exportable sin pérdidas


def test_el_archivo_lleva_formato_y_version():
    cfg = _exportar()
    assert cfg["formato"] == CX.FORMATO and cfg["version_esquema"] == CX.VERSION_ESQUEMA


def test_defectos_semilla_y_espacios_del_protocolo():
    cfg = CX.exportar_config(run_id="r", escenario="fies", modelos=["M3"], hash_dataset="x")
    assert cfg["semillas"] == [C.SEED] and cfg["n_trials"] == C.N_TRIALS
    assert "depth" in cfg["espacios_hiperparametros"]["M3"]


def test_clave_faltante_se_reporta():
    cfg = _exportar()
    del cfg["n_trials"], cfg["hash_dataset"]
    with pytest.raises(CX.ConfigInvalida) as exc:
        CX.importar_config(cfg)
    assert any("faltan claves obligatorias" in p and "n_trials" in p and "hash_dataset" in p for p in exc.value.problemas)


def test_clave_desconocida_se_rechaza():
    cfg = _exportar()
    cfg["optimizador_secreto"] = 1
    with pytest.raises(CX.ConfigInvalida) as exc:
        CX.importar_config(cfg)
    assert any("claves desconocidas" in p and "optimizador_secreto" in p for p in exc.value.problemas)


def test_version_futura_o_formato_ajeno_se_rechazan():
    futuro = _exportar() | {"version_esquema": CX.VERSION_ESQUEMA + 1}
    with pytest.raises(CX.ConfigInvalida, match="más nueva"):
        CX.importar_config(futuro)
    ajeno = _exportar() | {"formato": "otra.cosa"}
    with pytest.raises(CX.ConfigInvalida, match="formato"):
        CX.importar_config(ajeno)


@pytest.mark.parametrize("cambio, texto", [
    ({"escenario": "inexistente"}, "escenario"),
    ({"modelos": ["M1", "M9"]}, "modelos desconocidos"),
    ({"modelos": ["M1", "M1"]}, "repetidos"),
    ({"semillas": []}, "semillas"),
    ({"semillas": [1.5]}, "semillas"),
    ({"n_trials": 0}, "n_trials"),
    ({"n_trials": True}, "n_trials"),
    ({"esquema_particion": "V9"}, "esquema_particion"),
    ({"hash_dataset": " "}, "hash_dataset"),
    ({"run_id": ""}, "run_id"),
    ({"agrupar_por": "otra"}, "agrupar_por"),
])
def test_valores_invalidos_al_exportar(cambio, texto):
    with pytest.raises(CX.ConfigInvalida, match=texto):
        _exportar(**cambio)


def test_espacio_invalido_y_espacio_de_modelo_ajeno():
    malo = copy.deepcopy(ESPACIO)
    malo["M1"]["C"] = {"tipo": "float", "lo": 5, "hi": 1}
    with pytest.raises(CX.ConfigInvalida, match="lo > hi"):
        _exportar(espacios_hiperparametros=malo)
    ajeno = copy.deepcopy(ESPACIO) | {"M3": {}}
    with pytest.raises(CX.ConfigInvalida, match="no están en 'modelos'"):
        _exportar(espacios_hiperparametros=ajeno)
    sin_tipo = {"M1": {"C": {"lo": 1, "hi": 2}}}
    with pytest.raises(CX.ConfigInvalida, match="tipo"):
        _exportar(modelos=["M1"], espacios_hiperparametros=sin_tipo)


def test_reporta_todos_los_problemas_juntos():
    cfg = _exportar()
    cfg["escenario"] = "x"
    cfg["n_trials"] = -1
    cfg["extra"] = 0
    with pytest.raises(CX.ConfigInvalida) as exc:
        CX.importar_config(cfg)
    assert len(exc.value.problemas) >= 3


def test_no_escribe_archivo_si_es_invalido(tmp_path):
    ruta = tmp_path / "no.json"
    with pytest.raises(CX.ConfigInvalida):
        _exportar(ruta, n_trials=0)
    assert not ruta.exists()


def test_importar_archivo_inexistente_o_mal_formado(tmp_path):
    with pytest.raises(CX.ConfigInvalida, match="no se pudo leer"):
        CX.importar_config(tmp_path / "nada.json")
    mal = tmp_path / "mal.json"
    mal.write_text("{no es json", encoding="utf-8")
    with pytest.raises(CX.ConfigInvalida, match="JSON mal formado"):
        CX.importar_config(mal)
    assert CX.validar_config([1, 2]) == ["la configuración debe ser un objeto JSON"]


def test_hash_del_dataset(tmp_path):
    datos = tmp_path / "d.csv"
    datos.write_text("a,b\n1,2\n", encoding="utf-8")
    h = CX.hash_archivo(datos)
    assert len(h) == 64
    cfg = CX.importar_config(_exportar(hash_dataset=h))
    assert CX.verificar_hash_dataset(cfg, datos) is True
    datos.write_text("a,b\n1,3\n", encoding="utf-8")
    assert CX.verificar_hash_dataset(cfg, datos) is False
