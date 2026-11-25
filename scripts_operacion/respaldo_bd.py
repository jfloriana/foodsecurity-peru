"""respaldo_bd.py - Copia de seguridad de la base PostgreSQL de FoodSecurity AI con ``pg_dump`` (mejora #6).

Uso (desde la raíz del proyecto):
    python scripts_operacion/respaldo_bd.py --dry-run          # muestra el comando, no ejecuta nada
    python scripts_operacion/respaldo_bd.py                    # crea respaldos/<bd>_<AAAAMMDD_HHMMSS>.dump (+ .sha256)
    python scripts_operacion/respaldo_bd.py --esquema fsai --destino-dir D:/respaldos

Conexión (variables de entorno, las mismas de app_streamlit/.env.example):
    FSAI_DATABASE_URL   postgresql+psycopg2://usuario:clave@host:5432/bd   (tiene prioridad)
    POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST (por defecto localhost), POSTGRES_PORT (5432)

Seguridad: la contraseña se entrega a ``pg_dump`` por la variable PGPASSWORD del proceso hijo; nunca va en la línea de
comandos ni se imprime. El archivo de respaldo contiene datos de hogares (seudonimizados): guárdelo con permisos
restringidos. Este script NO se ejecutó contra una base real en la construcción de V2: las pruebas solo verifican la
construcción de comandos (``foodsecurity_ai/tests/test_scripts_operacion.py``).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping
from urllib.parse import unquote, urlparse

RAIZ = Path(__file__).resolve().parent.parent
DESTINO_DEFECTO = RAIZ / "respaldos"


class ConfigConexionError(ValueError):
    """Faltan o son inválidas las variables de conexión."""


@dataclass(frozen=True)
class Conexion:
    host: str
    puerto: int
    bd: str
    usuario: str
    clave: str = ""

    def __repr__(self) -> str:      # la clave nunca se muestra, ni en logs ni en trazas
        return f"Conexion(host={self.host!r}, puerto={self.puerto}, bd={self.bd!r}, usuario={self.usuario!r}, clave=***)"


def conexion_desde_entorno(env: Mapping[str, str] | None = None) -> Conexion:
    """Lee la conexión de ``FSAI_DATABASE_URL`` o de ``POSTGRES_*``. Lanza ``ConfigConexionError`` si falta algo."""
    env = os.environ if env is None else env
    url = (env.get("FSAI_DATABASE_URL") or "").strip()
    if url:
        p = urlparse(url)
        if not p.scheme.startswith("postgres") or not p.hostname or not p.path.strip("/"):
            raise ConfigConexionError("FSAI_DATABASE_URL debe tener la forma postgresql+psycopg2://usuario:clave@host:puerto/bd")
        return Conexion(host=p.hostname, puerto=p.port or 5432, bd=unquote(p.path.lstrip("/")),
                        usuario=unquote(p.username or ""), clave=unquote(p.password or ""))
    faltan = [k for k in ("POSTGRES_DB", "POSTGRES_USER") if not env.get(k)]
    if faltan:
        raise ConfigConexionError(f"Defina FSAI_DATABASE_URL o las variables {faltan} (ver app_streamlit/.env.example).")
    try:
        puerto = int(env.get("POSTGRES_PORT") or 5432)
    except ValueError as exc:
        raise ConfigConexionError("POSTGRES_PORT debe ser un entero.") from exc
    return Conexion(host=env.get("POSTGRES_HOST") or "localhost", puerto=puerto, bd=env["POSTGRES_DB"],
                    usuario=env["POSTGRES_USER"], clave=env.get("POSTGRES_PASSWORD", ""))


def entorno_subproceso(conn: Conexion, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Entorno del proceso hijo con PGPASSWORD (única vía por la que viaja la contraseña)."""
    env = dict(os.environ if base is None else base)
    if conn.clave:
        env["PGPASSWORD"] = conn.clave
    return env


def nombre_respaldo(bd: str, ahora: datetime | None = None) -> str:
    """``<bd>_<AAAAMMDD_HHMMSS>.dump``; el nombre de la BD se sanea para ser un nombre de archivo seguro."""
    seguro = "".join(c if (c.isalnum() or c in "-_") else "_" for c in bd) or "bd"
    return f"{seguro}_{(ahora or datetime.now()).strftime('%Y%m%d_%H%M%S')}.dump"


def construir_comando_pg_dump(conn: Conexion, destino: Path | str, esquema: str | None = None,
                              ejecutable: str = "pg_dump") -> list[str]:
    """Lista de argumentos de ``pg_dump`` (formato custom, comprimido). Nunca incluye la contraseña."""
    cmd = [ejecutable, "--host", conn.host, "--port", str(conn.puerto), "--username", conn.usuario,
           "--dbname", conn.bd, "--format=custom", "--no-password", "--file", str(destino)]
    if esquema:
        cmd += ["--schema", esquema]
    return cmd


def sha256_archivo(ruta: Path | str, bloque: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while chunk := f.read(bloque):
            h.update(chunk)
    return h.hexdigest()


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Respaldo de la base PostgreSQL de FoodSecurity AI con pg_dump.")
    ap.add_argument("--destino-dir", type=Path, default=DESTINO_DEFECTO, help=f"Carpeta de salida (defecto: {DESTINO_DEFECTO})")
    ap.add_argument("--esquema", default=None, help="Respaldar solo este esquema (p. ej. fsai). Por defecto, toda la base.")
    ap.add_argument("--dry-run", action="store_true", help="Muestra el comando y termina sin ejecutar pg_dump.")
    return ap


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    a = _parser().parse_args(argv)
    try:
        conn = conexion_desde_entorno(env)
    except ConfigConexionError as exc:
        print(f"ERROR de configuración: {exc}", file=sys.stderr)
        return 2
    destino = a.destino_dir / nombre_respaldo(conn.bd)
    cmd = construir_comando_pg_dump(conn, destino, a.esquema)
    print("Comando:", " ".join(cmd))
    if a.dry_run:
        print("[dry-run] No se ejecutó nada ni se creó ningún archivo.")
        return 0
    if shutil.which(cmd[0]) is None:
        print(f"ERROR: no se encontró '{cmd[0]}' en el PATH. Instale las herramientas cliente de PostgreSQL.", file=sys.stderr)
        return 2
    a.destino_dir.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(cmd, env=entorno_subproceso(conn, env), capture_output=True, text=True)
    if res.returncode != 0:
        destino.unlink(missing_ok=True)
        print(f"ERROR: pg_dump terminó con código {res.returncode}.\n{res.stderr.strip()}", file=sys.stderr)
        return res.returncode
    huella = sha256_archivo(destino)
    Path(str(destino) + ".sha256").write_text(f"{huella}  {destino.name}\n", encoding="utf-8")
    print(f"Respaldo creado: {destino} ({destino.stat().st_size} bytes). SHA-256: {huella}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
