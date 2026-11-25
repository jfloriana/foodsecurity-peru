"""restaurar_bd.py - Restauración de un respaldo creado por ``respaldo_bd.py`` con ``pg_restore`` (mejora #6).

Uso:
    python scripts_operacion/restaurar_bd.py respaldos/fsai_20261116_101500.dump --dry-run
    python scripts_operacion/restaurar_bd.py respaldos/fsai_20261116_101500.dump --confirmar
    python scripts_operacion/restaurar_bd.py <archivo> --confirmar --limpiar     # borra y recrea los objetos existentes

La restauración es DESTRUCTIVA cuando se usa --limpiar (elimina objetos antes de recrearlos): por eso, sin
``--dry-run`` el script exige ``--confirmar``. Si existe ``<archivo>.sha256`` se verifica antes de restaurar.
La conexión sale de las mismas variables que ``respaldo_bd.py``; la contraseña viaja por PGPASSWORD y nunca se imprime.
Recomendación: pruebe primero la restauración en una base vacía distinta (``--bd-destino fsai_prueba``). Este
procedimiento NO se ejecutó contra una base real en la construcción de V2 (solo se prueba la construcción de comandos).
"""
from __future__ import annotations

import argparse
import dataclasses
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping

from respaldo_bd import (ConfigConexionError, Conexion, conexion_desde_entorno, entorno_subproceso,  # noqa: E402
                         sha256_archivo)


def construir_comando_pg_restore(conn: Conexion, archivo: Path | str, limpiar: bool = False,
                                 ejecutable: str = "pg_restore") -> list[str]:
    """Lista de argumentos de ``pg_restore``. ``--exit-on-error`` evita restauraciones parciales silenciosas."""
    cmd = [ejecutable, "--host", conn.host, "--port", str(conn.puerto), "--username", conn.usuario,
           "--dbname", conn.bd, "--no-password", "--no-owner", "--exit-on-error"]
    if limpiar:
        cmd += ["--clean", "--if-exists"]
    cmd.append(str(archivo))
    return cmd


def verificar_huella(archivo: Path) -> str:
    """Compara con ``<archivo>.sha256`` si existe. Devuelve 'ok', 'sin_huella' o 'no_coincide'."""
    lado = Path(str(archivo) + ".sha256")
    if not lado.is_file():
        return "sin_huella"
    esperada = lado.read_text(encoding="utf-8").split()[0].strip().lower()
    return "ok" if sha256_archivo(archivo) == esperada else "no_coincide"


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Restaura un respaldo de FoodSecurity AI con pg_restore.")
    ap.add_argument("archivo", type=Path, help="Archivo .dump creado por respaldo_bd.py")
    ap.add_argument("--bd-destino", default=None, help="Base de destino (por defecto la de las variables de entorno).")
    ap.add_argument("--limpiar", action="store_true", help="DESTRUCTIVO: elimina los objetos existentes antes de recrearlos.")
    ap.add_argument("--confirmar", action="store_true", help="Obligatorio para ejecutar la restauración real.")
    ap.add_argument("--dry-run", action="store_true", help="Muestra el comando y termina sin ejecutar pg_restore.")
    return ap


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    a = _parser().parse_args(argv)
    try:
        conn = conexion_desde_entorno(env)
    except ConfigConexionError as exc:
        print(f"ERROR de configuración: {exc}", file=sys.stderr)
        return 2
    if a.bd_destino:
        conn = dataclasses.replace(conn, bd=a.bd_destino)
    cmd = construir_comando_pg_restore(conn, a.archivo, a.limpiar)
    print("Comando:", " ".join(cmd))
    if a.dry_run:
        print("[dry-run] No se ejecutó nada.")
        return 0
    if not a.confirmar:
        print("ERROR: la restauración requiere --confirmar (o use --dry-run para ver el comando).", file=sys.stderr)
        return 2
    if not a.archivo.is_file():
        print(f"ERROR: no existe el archivo {a.archivo}.", file=sys.stderr)
        return 2
    estado = verificar_huella(a.archivo)
    if estado == "no_coincide":
        print("ERROR: el SHA-256 del archivo no coincide con su .sha256; el respaldo puede estar dañado.", file=sys.stderr)
        return 3
    if estado == "sin_huella":
        print("Aviso: no hay archivo .sha256; no se pudo verificar la integridad del respaldo.")
    if shutil.which(cmd[0]) is None:
        print(f"ERROR: no se encontró '{cmd[0]}' en el PATH. Instale las herramientas cliente de PostgreSQL.", file=sys.stderr)
        return 2
    res = subprocess.run(cmd, env=entorno_subproceso(conn, env), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"ERROR: pg_restore terminó con código {res.returncode}.\n{res.stderr.strip()}", file=sys.stderr)
        return res.returncode
    print(f"Restauración completada en la base '{conn.bd}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
