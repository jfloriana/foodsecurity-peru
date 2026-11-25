"""post_fase2_real.py - Posproceso de una corrida de CV anidada (Fase 2) sobre datos REALES.

Para cada resultado_<escenario>_<modelo>.json de la corrida:
  1. Evalúa la Prueba (15 %) UNA sola vez (candado exclusivo de fsai.evaluacion; un segundo intento reutiliza lo guardado).
  2. Métricas ponderadas y no ponderadas con IC bootstrap estratificado, matriz de confusión, foco en severa, calibración.
  3. Pruebas estadísticas entre modelos (DeLong, McNemar, Wilcoxon, Friedman/Nemenyi) con Holm por familia.
  4. SHAP global (TreeSHAP en M2/M3/M5; LinearExplainer exacto en M1; KernelExplainer en submuestra en M4) y permutación.
  5. Equidad por grupos (brechas de TPR de la clase severa y de IA moderada/severa) con IC bootstrap.
  6. Persistencia en PostgreSQL: experimentos, modelos, predicciones (Prueba), metricas, explicaciones.

Uso (desde foodsecurity_ai/, con el entorno de despliegue_local\\.env cargado, p. ej. vía despliegue_local\\_con_env.py):
    python scripts/post_fase2_real.py --corrida <nombre> --ruta-datos <parquet> --dataset-id <id> [--n-boot 1000]
Todas las cifras salen de los archivos de la corrida; nada se inventa. La Prueba no se usa para elegir nada.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from argparse import Namespace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ_PAQUETE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_PAQUETE))

from fsai import config as C  # noqa: E402
from fsai import evaluacion as EV  # noqa: E402
from fsai.explainability import contribuciones_a_jsonb, importancia_permutacion, shap_global  # noqa: E402
from fsai.fairness import brechas_por_grupo  # noqa: E402
from fsai.preprocessing import preparar_matriz  # noqa: E402
from fsai.splits import cargar_particion  # noqa: E402

logger = logging.getLogger("fsai.post_fase2")
GRUPOS_EQUIDAD = ["sexo_jefe", "grupo_edad_jefe", "grupo_educ_jefe", "area", "region_natural", "pobreza_monetaria", "tiene_menor5"]


def _cargar_df(ruta: Path) -> pd.DataFrame:
    import train_pipeline as TP
    return TP.preparar_datos(Namespace(ruta_datos=ruta, modulos=None, fies_crudo=False, params_rasch=None, umbrales_fies=None))


def _seg(f):
    t0 = time.time()
    r = f()
    return r, round(time.time() - t0, 1)


def analizar(df: pd.DataFrame, dir_run: Path, n_boot: int, n_shap: int, usuario: str, escenarios: list[str] | None = None) -> dict:
    """Pasos 1-5. Escribe CSV/JSON en ``dir_run/analisis`` y devuelve las estructuras para persistir."""
    out_dir = dir_run / "analisis"
    out_dir.mkdir(exist_ok=True)
    res = {k: v for k, v in EV.resultados_corrida(dir_run.name).items() if not escenarios or v["escenario"] in escenarios}
    part, carga = cargar_particion(df, dir_run / "particion.json")
    te = df.iloc[part["prueba"]].reset_index(drop=True)
    info: dict = {"modelos": {}, "huella_particion": carga["huella"], "n_prueba": int(len(te))}
    for ident, r in res.items():
        t0 = time.time()
        try:
            reg = EV.evaluar_prueba_una_vez(df, dir_run, ident, usuario)
            nuevo = True
        except EV.PruebaYaEvaluada:
            reg = EV.registro_prueba(dir_run, ident)
            nuevo = False
        pr = EV.cargar_prediccion_prueba(dir_run, ident)
        y, proba, w = pr["y"], pr["proba"], pr["w"]
        met = EV.metricas_con_ic(y, proba, w, n_boot=n_boot)
        met.to_csv(out_dir / f"metricas_prueba_{ident}.csv", index=False)
        umb = r["final"]["validacion"]["umbral_severa"]
        sev = EV.foco_severa(y, proba, w, umb)
        cal = EV.calibracion_resumen(y, proba, w, r["final"].get("calibrador"))
        cm = {"w": EV.matriz_confusion(y, proba.argmax(1), w).tolist(), "n": EV.matriz_confusion(y, proba.argmax(1)).tolist()}
        curv = EV.curvas_roc_pr(y, proba)
        (out_dir / f"extra_{ident}.json").write_text(json.dumps({"foco_severa": sev, "calibracion": cal, "confusion": cm,
                                                                  "auc_por_clase": curv.get("auc", {}), "ap_por_clase": curv.get("ap", {}),
                                                                  "auc_macro": curv.get("auc_macro"), "registro_prueba": reg}, default=float), encoding="utf-8")
        info["modelos"][ident] = {"registro_prueba": reg, "prueba_nueva": nuevo, "metricas_prueba": met, "foco_severa": sev,
                                  "calibracion": cal, "confusion": cm, "auc_por_clase": curv.get("auc", {}),
                                  "ap_por_clase": curv.get("ap", {}), "auc_macro": curv.get("auc_macro"), "resultado": r}
        logger.info("%s: Prueba evaluada (%s) y métricas con IC en %.0fs", ident, "nueva" if nuevo else "reutilizada", time.time() - t0)
    escenarios = sorted({r["escenario"] for r in res.values()})
    pruebas = []
    for esc in escenarios:
        cods = sorted(r["codigo"] for r in res.values() if r["escenario"] == esc)
        if len(cods) >= 2:
            p = EV.pruebas_entre_modelos(dir_run, esc, cods)
            p.insert(0, "escenario", esc)
            pruebas.append(p)
    info["pruebas"] = pd.concat(pruebas, ignore_index=True) if pruebas else pd.DataFrame()
    info["pruebas"].to_csv(out_dir / "pruebas_estadisticas.csv", index=False)

    # Explicabilidad y equidad (sobre la Prueba ya evaluada; no se usa para elegir nada)
    equidad = []
    for ident, r in res.items():
        cod, esc = r["codigo"], r["escenario"]
        pipe = joblib.load(dir_run / r["final"]["modelo"])
        X = preparar_matriz(te, r["features_num"], r["features_cat"])
        d = info["modelos"][ident]
        try:
            n_m = min(n_shap, 100) if cod == "M4" else n_shap
            sg, seg = _seg(lambda: shap_global(pipe, X, cod, n_muestra=n_m, split="prueba"))
            d["shap"] = contribuciones_a_jsonb(sg["nombres"], sg["importancia_media"], {
                "n_muestra": sg["n_muestra"], "version": 1, "beeswarm": sg.get("beeswarm"), "explainer": "TreeSHAP" if cod in ("M2", "M3", "M5") else ("LinearSHAP" if cod == "M1" else "KernelSHAP"),
                "por_clase": {str(k): dict(zip(sg["nombres"], v)) for k, v in sg["importancia_por_clase"].items()}})
            (out_dir / f"shap_{ident}.json").write_text(json.dumps(d["shap"], ensure_ascii=False), encoding="utf-8")
            logger.info("%s: SHAP global en %.0fs (n=%d)", ident, seg, sg["n_muestra"])
        except Exception:
            logger.exception("SHAP falló para %s", ident)
            d["shap"] = None
        try:
            idx = np.random.default_rng(C.SEED).choice(len(X), size=min(4000, len(X)), replace=False)
            imp, seg = _seg(lambda: importancia_permutacion(pipe, X.iloc[idx], te[C.COL_OBJETIVO].to_numpy()[idx],
                                                            te[C.COL_PESO].to_numpy()[idx], n_repeticiones=5))
            imp.to_csv(out_dir / f"permutacion_{ident}.csv", index=False)
            d["permutacion"] = contribuciones_a_jsonb(imp["variable"].tolist(), imp["importancia"].to_numpy(),
                                                      {"desv_est": imp["desv_est"].tolist(), "n_muestra": int(len(idx)), "repeticiones": 5})
            logger.info("%s: permutación en %.0fs", ident, seg)
        except Exception:
            logger.exception("Permutación falló para %s", ident)
            d["permutacion"] = None
        pred = d["registro_prueba"] and EV.cargar_prediccion_prueba(dir_run, ident)["proba"].argmax(1)
        for col in GRUPOS_EQUIDAD:
            if col in te.columns and te[col].nunique() > 1:
                for nombre, clase in (("severa", 3), ("ia_binaria", [2, 3])):
                    try:
                        t = brechas_por_grupo(te, te[C.COL_OBJETIVO].to_numpy(), pred, col, clase, n_boot=min(n_boot, 500))
                        equidad.append(t.assign(escenario=esc, codigo=cod, objetivo=nombre, grupo_col=col))
                    except Exception:
                        logger.exception("Equidad falló %s/%s", ident, col)
    info["equidad"] = pd.concat(equidad, ignore_index=True) if equidad else pd.DataFrame()
    info["equidad"].to_csv(out_dir / "equidad.csv", index=False)
    return info


def _nom(escenario: str, nombre_corrida: str) -> str:
    return f"ENAHO_2025_REAL_{nombre_corrida}_{escenario}"


def persistir(df: pd.DataFrame, dir_run: Path, info: dict, dataset_id: int, desviaciones: dict) -> dict[str, int]:
    from sqlalchemy.orm import Session
    from fsai import database as D
    engine = D.get_engine()
    D.crear_esquema(engine)
    cfg_base = json.loads((dir_run / "config.json").read_text(encoding="utf-8"))
    ids: dict[str, int] = {}
    part, _ = cargar_particion(df, dir_run / "particion.json")
    te = df.iloc[part["prueba"]].reset_index(drop=True)
    escenarios = sorted({m["resultado"]["escenario"] for m in info["modelos"].values()})
    with Session(engine) as s, s.begin():
        exp_id: dict[str, int] = {}
        for esc in escenarios:
            pr = info["pruebas"]
            pr_esc = [] if pr.empty else json.loads(pr[pr["escenario"] == esc].to_json(orient="records"))
            cfg = {**cfg_base, "modo": "cv_anidada_70_15_15", "esquema_nota": "V1 = agrupado por hogar_uid (partición 70/15/15 + CV anidada 5x3)",
                   "pruebas_estadisticas": pr_esc, "desviaciones_protocolo": desviaciones,
                   "circularidad": cfg_base.get("circularidad"),
                   "nota_escenario": ("PREDICCION real (sin fies_q*/raw_score)" if esc == "socio" else "RECONSTRUCCION circular del objetivo (no es prediccion)")}
            exp_id[esc] = D.guardar_experimento(s, dataset_id, _nom(esc, dir_run.name), "V1", esc, C.SEED, cfg)
        for ident, m in info["modelos"].items():
            r = m["resultado"]
            mid = D.guardar_modelo(s, exp_id[r["escenario"]], r["codigo"], {**r["final"]["hiperparametros"], "_features_descartadas_100pct_nan": r.get("features_descartadas_100pct_nan", [])},
                                   r["n_trials_final"], r["final"]["mejor_valor_interno"], str(dir_run / r["final"]["modelo"]))
            ids[ident] = mid
            # Prueba (una sola evaluación)
            for _, f in m["metricas_prueba"].iterrows():
                D.guardar_metricas(s, mid, "V1", bool(f["ponderada"]), {f["metrica"]: f["valor"]}, {f["metrica"]: (f["ic_inf"], f["ic_sup"])}, None, "prueba")
            # CV anidada: un valor por pliegue externo + agregado (media, IC t 95 %)
            for k, fo in enumerate(r["folds"]):
                for clave, pond in (("ponderada", True), ("no_ponderada", False)):
                    D.guardar_metricas(s, mid, "V1", pond, fo["metricas"][clave], None, k, "cv_anidada_externo")
            for clave, pond in (("ponderada", True), ("no_ponderada", False)):
                ag = r["agregado"][clave]
                D.guardar_metricas(s, mid, "V1", pond, {n: v["media"] for n, v in ag.items()},
                                   {n: (v["ic_inf"], v["ic_sup"]) for n, v in ag.items()}, None, "cv_anidada_agregado")
            # Validación (umbral y calibración)
            va = r["final"]["validacion"]
            D.guardar_metricas(s, mid, "V1", True, va["metricas_ponderadas"], None, None, "validacion")
            D.guardar_metricas(s, mid, "V1", True, va["metricas_ponderadas_calibradas"], None, None, "validacion_calibrada")
            D.guardar_metricas(s, mid, "V1", True, {"umbral_severa": va["umbral_severa"], "recall_severa_umbral": va["recall_severa_umbral"],
                                                    "especificidad_severa_umbral": va["especificidad_severa_umbral"]}, None, None, "validacion_umbral")
            D.guardar_metricas(s, mid, "V1", True, {k: v for k, v in m["foco_severa"].items()}, None, None, "prueba_foco_severa")
            D.guardar_metricas(s, mid, "V1", True, {k: v for k, v in m["calibracion"].items() if v is not None}, None, None, "prueba_calibracion")
            # Predicciones de la Prueba
            pr = EV.cargar_prediccion_prueba(dir_run, ident)
            uid_prueba = te[C.COL_GRUPO].astype(str).to_numpy()
            assert (pr["uid"] == uid_prueba).all()
            D.guardar_predicciones(s, mid, uid_prueba, np.zeros(len(uid_prueba), int), pr["y"], pr["proba"].argmax(1), pr["proba"])
            if m.get("shap"):
                D.guardar_explicacion(s, mid, "shap_global", m["shap"])
            if m.get("permutacion"):
                D.guardar_explicacion(s, mid, "permutation", m["permutacion"])
        for esc in escenarios:
            D.finalizar_experimento(s, exp_id[esc])
        D.registrar_auditoria(s, "post_fase2_real", "persistencia_cv_anidada", "dataset", str(dataset_id),
                              {"corrida": dir_run.name, "huella_particion": info["huella_particion"], "n_prueba": info["n_prueba"]})
    return ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corrida", required=True, help="Nombre de la carpeta en <ARTIFACTS>/fases")
    ap.add_argument("--ruta-datos", required=True, type=Path)
    ap.add_argument("--dataset-id", required=True, type=int)
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    ap.add_argument("--n-shap", type=int, default=500)
    ap.add_argument("--escenarios", nargs="+", default=None, choices=C.ESCENARIOS, help="Solo estos escenarios (por defecto todos los resultados de la corrida)")
    ap.add_argument("--sin-db", action="store_true")
    ap.add_argument("--desviaciones-json", type=Path, default=None, help="JSON con las desviaciones declaradas (se guarda en config_json)")
    a = ap.parse_args(argv)
    dir_run = EV.dir_corrida(a.corrida)
    C.configurar_logging(dir_run / "post_fase2.log")
    df = _cargar_df(a.ruta_datos)
    info = analizar(df, dir_run, a.n_boot, a.n_shap, "post_fase2_real", a.escenarios)
    desv = json.loads(a.desviaciones_json.read_text(encoding="utf-8")) if a.desviaciones_json else {}
    if not a.sin_db:
        ids = persistir(df, dir_run, info, a.dataset_id, desv)
        (dir_run / "analisis" / "modelo_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        logger.info("Persistido en PostgreSQL: %s", ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
