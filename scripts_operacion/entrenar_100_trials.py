"""entrenar_100_trials.py - Lanzador delgado del re-entrenamiento con 100 ensayos Optuna (protocolo) en vez de 20 (V1, D1).

NO entrena nada por sí mismo: arma y (solo con --ejecutar) lanza los comandos de ``foodsecurity_ai/scripts/entrenar_real_fase2.py``,
que SÍ acepta ``--trials`` y ``--trials-final``. Sin ``--ejecutar`` imprime el plan y la tabla de tiempos estimados.

Plan que construye (el mismo flujo de V1, ver referencia_v1/DESVIACIONES.md):
  1) python foodsecurity_ai/scripts/entrenar_real_fase2.py --ruta-datos <parquet> --nombre <corrida> --solo-particion
  2) por modelo: ... --modelos Mk --escenarios socio --trials 100 --trials-final 100    (secuencial; --paralelo = un proceso por modelo)
  3) después, a mano: python foodsecurity_ai/scripts/post_fase2_real.py --corrida <corrida> --ruta-datos <parquet> --dataset-id <id>

Ejemplo:
    python scripts_operacion/entrenar_100_trials.py --ruta-datos D:/datos/enaho_analitico.parquet --nombre real_100          # solo plan
    python scripts_operacion/entrenar_100_trials.py --ruta-datos D:/datos/enaho_analitico.parquet --nombre real_100 --ejecutar

AVISO DE COSTO: con 100 ensayos las estimaciones son de ~0,4 h (M1) a ~17,7 h (M5) por modelo en el equipo de V1
(32 hilos, sin GPU). Son extrapolaciones lineales de tiempos medidos por ensayo, NO una corrida medida con 100 ensayos.
Los escenarios fies/hibrido (circulares) quedan fuera por defecto; si se piden, usan los mismos 100 ensayos (costo adicional).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SCRIPT_ENTRENAR = RAIZ / "foodsecurity_ai" / "scripts" / "entrenar_real_fase2.py"
N_TRIALS = 100
MODELOS = ("M1", "M2", "M3", "M4", "M5")
BUSQUEDAS_POR_MODELO = 6      # 5 pliegues externos + la búsqueda final sobre todo Desarrollo (D1)

# Segundos por ensayo medidos en V1 en una corrida de humo (D1: M1 45 s, M2 239 s, M3 755 s, M4 1 548 s, M5 1 917 s por 18 ensayos),
# con hilos repartidos entre los 5 modelos (D7: M1 1, M2 3, M3 6, M4 10, M5 10).
SEGUNDOS_POR_ENSAYO = {"M1": 2.5, "M2": 13.0, "M3": 42.0, "M4": 86.0, "M5": 106.0}
HILOS_POR_MODELO = {"M1": 1, "M2": 3, "M3": 6, "M4": 10, "M5": 10}
# Referencias de D1 para contrastar: horas con 100 ensayos que declara el documento (aprox.): M4 ~14 h, M5 ~17 h; con 20: M4 ~3 h, M5 ~3,5 h.


def estimar_horas(n_trials: int, modelos: tuple[str, ...] = MODELOS) -> dict[str, float]:
    """Horas estimadas por modelo = ensayos x búsquedas (6) x segundos por ensayo / 3600."""
    return {m: n_trials * BUSQUEDAS_POR_MODELO * SEGUNDOS_POR_ENSAYO[m] / 3600.0 for m in modelos}


def tabla_tiempos(n_trials: int = N_TRIALS, modelos: tuple[str, ...] = MODELOS) -> str:
    """Tabla de texto con la estimación para 20 (V1) y ``n_trials`` ensayos."""
    h20, hn = estimar_horas(20, modelos), estimar_horas(n_trials, modelos)
    filas = ["modelo | s/ensayo | horas con 20 ensayos (V1) | horas con %d ensayos (estimado)" % n_trials,
             "-------+----------+--------------------------+-----------------------------------"]
    for m in modelos:
        filas.append(f"{m:<6} | {SEGUNDOS_POR_ENSAYO[m]:>8.1f} | {h20[m]:>24.1f} | {hn[m]:>33.1f}")
    filas.append(f"Total secuencial: {sum(h20.values()):.1f} h con 20 ensayos; {sum(hn.values()):.1f} h con {n_trials}. "
                 "En paralelo (un proceso por modelo) el tiempo de pared se acerca al del modelo más lento.")
    return "\n".join(filas)


def construir_comandos(ruta_datos: Path | str, nombre: str, modelos: tuple[str, ...] = MODELOS,
                       escenarios: tuple[str, ...] = ("socio",), n_trials: int = N_TRIALS,
                       python: str | None = None) -> list[list[str]]:
    """Comandos en orden: primero la partición compartida, luego un comando por modelo."""
    py = python or sys.executable
    base = [py, str(SCRIPT_ENTRENAR), "--ruta-datos", str(ruta_datos), "--nombre", nombre]
    cmds = [base + ["--solo-particion"]]
    for m in modelos:
        cmds.append(base + ["--modelos", m, "--escenarios", *escenarios,
                            "--trials", str(n_trials), "--trials-final", str(n_trials)])
    return cmds


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Lanzador del re-entrenamiento con 100 ensayos Optuna.")
    ap.add_argument("--ruta-datos", required=True, type=Path, help="Parquet/CSV analítico a nivel hogar.")
    ap.add_argument("--nombre", required=True, help="Nombre de la corrida (carpeta en artifacts/fases).")
    ap.add_argument("--modelos", nargs="+", default=list(MODELOS), choices=MODELOS)
    ap.add_argument("--escenarios", nargs="+", default=["socio"], choices=["socio", "fies", "hibrido"])
    ap.add_argument("--trials", type=int, default=N_TRIALS, help="Ensayos por modelo y búsqueda (defecto 100).")
    ap.add_argument("--paralelo", action="store_true", help="Un proceso por modelo a la vez, con FSAI_N_HILOS repartidos (D7).")
    ap.add_argument("--ejecutar", action="store_true", help="Lanza realmente los comandos. Sin esto solo imprime el plan.")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    modelos = tuple(a.modelos)
    cmds = construir_comandos(a.ruta_datos, a.nombre, modelos, tuple(a.escenarios), a.trials)
    print(tabla_tiempos(a.trials, modelos))
    print("\nPlan:")
    for c in cmds:
        print("  " + " ".join(c))
    if not a.ejecutar:
        print("\n[solo plan] Agregue --ejecutar para lanzarlo. Nada se ejecutó.")
        return 0
    if not SCRIPT_ENTRENAR.is_file():
        print(f"ERROR: no existe {SCRIPT_ENTRENAR}", file=sys.stderr)
        return 2
    cwd = str(RAIZ / "foodsecurity_ai")
    if subprocess.run(cmds[0], cwd=cwd).returncode != 0:
        print("ERROR: falló la creación de la partición.", file=sys.stderr)
        return 1
    modelo_cmds = list(zip(modelos, cmds[1:]))
    if a.paralelo:
        procs = [(m, subprocess.Popen(c, cwd=cwd, env={**os.environ, "FSAI_N_HILOS": str(HILOS_POR_MODELO[m])}))
                 for m, c in modelo_cmds]
        codigos = {m: p.wait() for m, p in procs}
    else:
        codigos = {m: subprocess.run(c, cwd=cwd).returncode for m, c in modelo_cmds}
    malos = {m: c for m, c in codigos.items() if c != 0}
    if malos:
        print(f"ERROR: modelos con fallo: {malos}", file=sys.stderr)
        return 1
    print("Entrenamiento terminado. Siguiente paso (manual): scripts/post_fase2_real.py --corrida", a.nombre)
    return 0


if __name__ == "__main__":
    sys.exit(main())
