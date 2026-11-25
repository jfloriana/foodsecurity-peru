"""entrenar_real_fase2.py - CV anidada 5x3 (Fase 2) sobre datos reales, un proceso por modelo (ejecución en paralelo).

Reutiliza fsai.nested_cv.ejecutar_modelo SIN cambios de lógica; solo añade: (a) partición 70/15/15 compartida y guardada con
huella (``--solo-particion`` la crea una vez; los procesos por modelo la cargan), (b) progreso/registro por modelo para que
procesos paralelos no se pisen, (c) recorte declarado del espacio de M1 (solo penalización L2; ver DESVIACIONES) y
(d) el mismo presupuesto de ensayos Optuna para los 5 modelos.

Uso (con despliegue_local\\_con_env.py):
  python scripts/entrenar_real_fase2.py --ruta-datos <parquet> --nombre <corrida> --solo-particion
  python scripts/entrenar_real_fase2.py --ruta-datos <parquet> --nombre <corrida> --modelos M3 --escenarios socio --trials 20

V2: ``--escenarios socio_terr`` (socio + indicadores territoriales + ingeniería de variables; requiere
``datos_derivados/indicadores_territoriales_departamento.csv``) y ``--modelos ... M6`` (HistGradientBoosting, extra opcional).
Este script NUNCA usa PostgreSQL ni Optuna persistente (estudios en memoria): ``--sin-db``/``--sin-optuna-db`` solo existen en train_pipeline.py.
Para comparar justo ``socio`` vs ``socio_terr`` use la MISMA ``--nombre`` (misma partición.json), ``--trials`` y semilla.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from argparse import Namespace
from pathlib import Path

RAIZ_PAQUETE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_PAQUETE))

from fsai import config as C  # noqa: E402
from fsai.nested_cv import RegistroProgreso, ejecutar_modelo  # noqa: E402
from fsai.splits import cargar_particion, guardar_particion, particionar_70_15_15, verificar_particion  # noqa: E402

logger = logging.getLogger("fsai.entrenar_real")

# Recorte declarado del espacio (se registra en config/resultados): saga+L1 no converge en tiempo razonable
# (~0,09 s/iteración; >300 iteraciones sin converger con tol=1e-3 sobre ~18 000 x 65), de modo que M1 se busca solo con L2.
ESPACIO_POR_MODELO = {"M1": {"penalty": {"opciones": ["l2"]}},
                      "M3": {"depth": {"hi": 6}}}   # V2: M3 con profundidad <= 6 (declarado en DESVIACIONES; acorta ~2x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ruta-datos", required=True, type=Path)
    ap.add_argument("--nombre", required=True, help="Carpeta de la corrida en <ARTIFACTS>/fases")
    ap.add_argument("--modelos", nargs="+", default=C.MODELOS, choices=C.MODELOS_TODOS)
    ap.add_argument("--escenarios", nargs="+", default=["socio"], choices=C.ESCENARIOS_TODOS)
    ap.add_argument("--trials", type=int, default=C.N_TRIALS)
    ap.add_argument("--trials-final", type=int, default=None)
    ap.add_argument("--solo-particion", action="store_true")
    ap.add_argument("--ponderar-ajuste", action="store_true", help="usa factor_expansion como sample_weight al AJUSTAR (defecto: no; solo pondera las métricas)")
    ap.add_argument("--recorte-pesos-p", type=float, default=None, metavar="P", help="recorta los pesos de ajuste en el percentil P (requiere --ponderar-ajuste)")
    ap.add_argument("--normalizar-pesos", action="store_true", help="normaliza los pesos de ajuste a media 1 (requiere --ponderar-ajuste)")
    ap.add_argument("--sin-ingenieria", action="store_true", help="en socio_terr, no agrega variables de ingeniería (solo territoriales)")
    ap.add_argument("--optimizar-umbrales", action="store_true", help="multiplicadores por clase para F1-macro ajustados SOLO en Validación")
    ap.add_argument("--evaluar-prueba", action="store_true",
                    help="al terminar, evalúa UNA vez en Prueba (argmax vs multiplicadores; requiere --optimizar-umbrales). No usar si Prueba debe seguir intacta")
    ap.add_argument("--agrupar-por", default="hogar_uid", choices=["hogar_uid", "conglome"])
    a = ap.parse_args(argv)
    if (a.recorte_pesos_p is not None or a.normalizar_pesos) and not a.ponderar_ajuste:
        ap.error("--recorte-pesos-p / --normalizar-pesos requieren --ponderar-ajuste")
    if a.evaluar_prueba and not a.optimizar_umbrales:
        ap.error("--evaluar-prueba requiere --optimizar-umbrales")
    import train_pipeline as TP
    dir_run = C.DIR_FASES / a.nombre
    dir_run.mkdir(parents=True, exist_ok=True)
    tag = "particion" if a.solo_particion else "_".join(a.modelos)
    C.configurar_logging(dir_run / f"pipeline_{tag}.log")
    df = TP.preparar_datos(Namespace(ruta_datos=a.ruta_datos, modulos=None, fies_crudo=False, params_rasch=None, umbrales_fies=None))
    ruta_part = dir_run / "particion.json"
    if a.solo_particion:
        part = particionar_70_15_15(df, C.SEED, a.agrupar_por)
        huella = guardar_particion(df, part, ruta_part, C.SEED, a.agrupar_por, {"ruta_datos": str(a.ruta_datos)})
        ver = verificar_particion(df, part)
        if ver["cruces_hogar_uid_total"] or not ver["exhaustiva"]:
            raise RuntimeError(f"Partición inválida: {ver}")
        cfg = {"nombre": a.nombre, "modo": "cv_anidada", "ruta_datos": str(a.ruta_datos), "hash_datos": TP._hash_ruta(a.ruta_datos),
               "huella_particion": huella, "agrupar_por": a.agrupar_por, "escenario": a.escenarios, "modelos": C.MODELOS,
               "trials": a.trials, "trials_final": a.trials_final or a.trials, "seed": C.SEED, "n_outer": C.N_SPLITS_EXTERNOS,
               "n_inner": C.INNER_FOLDS, "verificacion_particion": ver, "circularidad": TP.NOTA_CIRCULARIDAD,
               "espacio_recortado": ESPACIO_POR_MODELO, "creado_en": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (dir_run / "config.json").write_text(json.dumps(cfg, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        logger.info("Partición creada: %s | huella %s", ver["tamanos"], huella[:12])
        return 0
    part, carga = cargar_particion(df, ruta_part)
    for esc in a.escenarios:
        for cod in a.modelos:
            prog = RegistroProgreso(dir_run / "progreso" / f"{esc}_{cod}")
            prog.actualizar(estado="ejecutando", modelo=cod, escenario=esc, trials=[])
            t0 = time.time()
            try:
                ejecutar_modelo(df, part, codigo=cod, escenario=esc, n_trials=a.trials, dir_run=dir_run, progreso=prog,
                                n_trials_final=a.trials_final or a.trials, espacio=ESPACIO_POR_MODELO.get(cod),
                                ponderar_ajuste=a.ponderar_ajuste, ingenieria=not a.sin_ingenieria, recorte_pesos=a.recorte_pesos_p,
                                normalizar_pesos=a.normalizar_pesos, optimizar_umbrales=a.optimizar_umbrales)
                if a.evaluar_prueba:
                    from fsai.umbrales import evaluar_prueba_con_umbrales
                    evaluar_prueba_con_umbrales(df, part, dir_run, f"{esc}_{cod}")
                prog.actualizar(estado="completado")
            except Exception as exc:
                logger.exception("Fallo %s/%s", esc, cod)
                prog.actualizar(estado="error", mensaje=f"{type(exc).__name__}: {exc}")
                return 1
            logger.info("%s/%s completado en %.0fs", esc, cod, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
