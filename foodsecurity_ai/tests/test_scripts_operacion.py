"""Scripts de operación (respaldo/restauración/lanzador 100 ensayos): se prueba la construcción de comandos, sin ejecutar nada."""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

DIR_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts_operacion"
if str(DIR_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DIR_SCRIPTS))

import entrenar_100_trials as E  # noqa: E402
import respaldo_bd as R  # noqa: E402
import restaurar_bd as RS  # noqa: E402

CLAVE = "S3cr3t@/:?#%x"
ENV_PG = {"POSTGRES_DB": "fsai", "POSTGRES_USER": "fsai_admin", "POSTGRES_PASSWORD": CLAVE}


@pytest.fixture(autouse=True)
def _sin_subprocesos(monkeypatch):
    """Garantía: ninguna prueba de este archivo puede lanzar pg_dump/pg_restore ni entrenar."""
    def prohibido(*a, **k):
        raise AssertionError("las pruebas no deben ejecutar subprocesos")
    monkeypatch.setattr(subprocess, "run", prohibido)
    monkeypatch.setattr(subprocess, "Popen", prohibido)


def test_conexion_desde_variables_postgres():
    c = R.conexion_desde_entorno(ENV_PG)
    assert (c.host, c.puerto, c.bd, c.usuario, c.clave) == ("localhost", 5432, "fsai", "fsai_admin", CLAVE)
    c2 = R.conexion_desde_entorno({**ENV_PG, "POSTGRES_HOST": "db", "POSTGRES_PORT": "5433"})
    assert (c2.host, c2.puerto) == ("db", 5433)


def test_conexion_desde_url_con_clave_codificada():
    url = "postgresql+psycopg2://usr:p%40ss%2Fw@servidor:6543/mi_bd"
    c = R.conexion_desde_entorno({"FSAI_DATABASE_URL": url, **ENV_PG})      # la URL tiene prioridad
    assert (c.host, c.puerto, c.bd, c.usuario, c.clave) == ("servidor", 6543, "mi_bd", "usr", "p@ss/w")


@pytest.mark.parametrize("env", [{}, {"POSTGRES_DB": "x"}, {"FSAI_DATABASE_URL": "mysql://u:p@h/db"},
                                 {"FSAI_DATABASE_URL": "postgresql://u:p@h"}, {**ENV_PG, "POSTGRES_PORT": "abc"}])
def test_conexion_invalida_lanza_error_de_configuracion(env):
    with pytest.raises(R.ConfigConexionError):
        R.conexion_desde_entorno(env)


def test_la_clave_no_aparece_en_repr_ni_en_el_comando():
    c = R.conexion_desde_entorno(ENV_PG)
    assert CLAVE not in repr(c) and "***" in repr(c)
    cmd = R.construir_comando_pg_dump(c, "salida.dump")
    assert CLAVE not in " ".join(cmd)
    assert R.entorno_subproceso(c, {})["PGPASSWORD"] == CLAVE            # única vía de la clave
    assert "PGPASSWORD" not in R.entorno_subproceso(R.conexion_desde_entorno({**ENV_PG, "POSTGRES_PASSWORD": ""}), {})


def test_comando_pg_dump_completo_y_esquema_opcional():
    c = R.conexion_desde_entorno({**ENV_PG, "POSTGRES_HOST": "h", "POSTGRES_PORT": "5440"})
    assert R.construir_comando_pg_dump(c, "x/y.dump") == [
        "pg_dump", "--host", "h", "--port", "5440", "--username", "fsai_admin", "--dbname", "fsai",
        "--format=custom", "--no-password", "--file", "x/y.dump"]
    assert R.construir_comando_pg_dump(c, "y.dump", esquema="fsai")[-2:] == ["--schema", "fsai"]


def test_nombre_de_respaldo_con_marca_de_tiempo_y_saneado():
    ahora = datetime(2026, 11, 16, 9, 5, 7)
    assert R.nombre_respaldo("fsai", ahora) == "fsai_20261116_090507.dump"
    assert R.nombre_respaldo("../mi bd;rm", ahora) == "___mi_bd_rm_20261116_090507.dump"


def test_comando_pg_restore_con_y_sin_limpiar():
    c = R.conexion_desde_entorno(ENV_PG)
    base = RS.construir_comando_pg_restore(c, "r.dump")
    assert base[0] == "pg_restore" and base[-1] == "r.dump" and "--clean" not in base
    assert "--exit-on-error" in base and "--no-password" in base and CLAVE not in " ".join(base)
    limpio = RS.construir_comando_pg_restore(c, "r.dump", limpiar=True)
    assert "--clean" in limpio and "--if-exists" in limpio and limpio[-1] == "r.dump"


def test_dry_run_no_ejecuta_ni_crea_archivos(tmp_path, capsys):
    rc = R.main(["--dry-run", "--destino-dir", str(tmp_path / "salida")], env=ENV_PG)
    out = capsys.readouterr().out
    assert rc == 0 and "pg_dump" in out and CLAVE not in out
    assert not (tmp_path / "salida").exists()
    rc = RS.main([str(tmp_path / "a.dump"), "--dry-run", "--limpiar", "--bd-destino", "fsai_prueba"], env=ENV_PG)
    out = capsys.readouterr().out
    assert rc == 0 and "fsai_prueba" in out and "--clean" in out and CLAVE not in out


def test_restaurar_exige_confirmar_y_valida_huella(tmp_path, capsys):
    dump = tmp_path / "a.dump"
    dump.write_bytes(b"contenido")
    assert RS.main([str(dump)], env=ENV_PG) == 2                       # sin --confirmar
    assert "--confirmar" in capsys.readouterr().err
    assert RS.verificar_huella(dump) == "sin_huella"
    Path(str(dump) + ".sha256").write_text(f"{R.sha256_archivo(dump)}  a.dump\n", encoding="utf-8")
    assert RS.verificar_huella(dump) == "ok"
    dump.write_bytes(b"alterado")
    assert RS.verificar_huella(dump) == "no_coincide"
    assert RS.main([str(dump), "--confirmar"], env=ENV_PG) == 3        # se detiene ANTES de llamar a pg_restore


def test_sin_configuracion_devuelve_codigo_2(capsys):
    assert R.main(["--dry-run"], env={}) == 2
    assert "ERROR de configuración" in capsys.readouterr().err


# ---------------- lanzador de 100 ensayos ----------------
def test_comandos_del_lanzador_usan_trials_100(tmp_path):
    cmds = E.construir_comandos("datos.parquet", "real_100", ("M1", "M3"), ("socio",), python="py")
    assert cmds[0][-1] == "--solo-particion" and "--trials" not in cmds[0]
    assert len(cmds) == 3
    m3 = cmds[2]
    assert m3[m3.index("--modelos") + 1] == "M3"
    assert m3[m3.index("--trials") + 1] == "100" and m3[m3.index("--trials-final") + 1] == "100"
    assert m3[m3.index("--nombre") + 1] == "real_100"
    assert Path(m3[1]).name == "entrenar_real_fase2.py"


def test_el_script_lanzado_realmente_acepta_esos_argumentos():
    """Las opciones que construye el lanzador existen en el CLI real de entrenar_real_fase2.py."""
    fuente = E.SCRIPT_ENTRENAR.read_text(encoding="utf-8")
    for opcion in ("--ruta-datos", "--nombre", "--modelos", "--escenarios", "--trials", "--trials-final", "--solo-particion"):
        assert f'"{opcion}"' in fuente


def test_estimacion_de_tiempos_concuerda_con_d1():
    h20, h100 = E.estimar_horas(20), E.estimar_horas(100)
    assert h20["M5"] == pytest.approx(3.53, abs=0.05) and h20["M4"] == pytest.approx(2.87, abs=0.05)     # D1: ~3,5 h y ~3 h
    assert h100["M5"] == pytest.approx(17.67, abs=0.05) and h100["M4"] == pytest.approx(14.33, abs=0.05)  # D1: ~17 h y ~14 h
    assert all(h100[m] == pytest.approx(5 * h20[m]) for m in E.MODELOS)
    assert "17.7" in E.tabla_tiempos(100)


def test_lanzador_sin_ejecutar_solo_imprime_el_plan(capsys):
    assert E.main(["--ruta-datos", "d.parquet", "--nombre", "x"]) == 0       # subprocess.run prohibido: falla si ejecutara algo
    assert "Nada se ejecutó" in capsys.readouterr().out
