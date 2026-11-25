"""Experimento de tamaño medio: socio vs socio_terr, M1 M2 M3 M6, 30 ensayos, misma semilla y partición.

Declarado como búsqueda PARCIAL (30 de 100 ensayos; M3 con profundidad <= 6; sin M4/M5). Un proceso por
(escenario, modelo), con FSAI_N_HILOS repartidos para no exceder los hilos del equipo.
Uso: python scripts_operacion/lanzar_experimento_medio.py --ruta-datos <parquet> [--trials 30] [--nombre cmp_terr30]
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time, json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SCRIPT = RAIZ / "foodsecurity_ai" / "scripts" / "entrenar_real_fase2.py"
HILOS = {"M1": 1, "M2": 3, "M3": 6, "M6": 3}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruta-datos", required=True)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--nombre", default="cmp_terr30")
    ap.add_argument("--escenarios", nargs="+", default=["socio", "socio_terr"])
    ap.add_argument("--modelos", nargs="+", default=["M1", "M2", "M3", "M6"])
    a = ap.parse_args()
    env0 = dict(os.environ, FSAI_ARTIFACTS_DIR=str(RAIZ / "artifacts_experimentos"),
                PYTHONPATH=str(RAIZ / "foodsecurity_ai"), PYTHONIOENCODING="utf-8")
    logs = RAIZ / "artifacts_experimentos" / "logs"; logs.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, str(SCRIPT), "--ruta-datos", a.ruta_datos, "--nombre", a.nombre]
    t0 = time.time()
    r = subprocess.run(base + ["--solo-particion"], env=env0, capture_output=True, text=True)
    (logs / "particion.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        print("FALLO la partición:", r.stderr[-800:]); return 1
    procs = []
    for esc in a.escenarios:
        for m in a.modelos:
            env = dict(env0, FSAI_N_HILOS=str(HILOS.get(m, 2)))
            f = open(logs / f"{esc}_{m}.log", "w", encoding="utf-8")
            cmd = base + ["--escenarios", esc, "--modelos", m, "--trials", str(a.trials)]
            procs.append((esc, m, subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT), time.time()))
    resumen = {}
    for esc, m, p, ti in procs:
        rc = p.wait(); resumen[f"{esc}_{m}"] = {"rc": rc, "segundos": round(time.time() - ti)}
        print(esc, m, "rc=", rc, "s=", round(time.time() - ti), flush=True)
    resumen["total_segundos"] = round(time.time() - t0)
    (logs / "resumen_tiempos.json").write_text(json.dumps(resumen, indent=1), encoding="utf-8")
    return 0 if all(v["rc"] == 0 for k, v in resumen.items() if isinstance(v, dict)) else 2

if __name__ == "__main__":
    raise SystemExit(main())
