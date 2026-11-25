"""train_pipeline.py - Pipeline CRISP-DM de extremo a extremo para FoodSecurity AI (FIES / ENAHO).

Un solo comando ejecuta: carga -> calidad -> FIES -> preprocesado -> particiones guardadas ->
Optuna por modelo y pliegue (anidado) -> entrenamiento final por pliegue -> métricas
ponderadas/no ponderadas -> pruebas estadísticas -> SHAP/permutación -> equidad ->
persistencia en PostgreSQL -> reportes (Excel/Word/PDF).

Reglas anti-fuga: la imputación, el escalado, el SMOTE (opcional), los hiperparámetros (Optuna)
y el umbral de la clase severa se ajustan EXCLUSIVAMENTE con las filas de train de cada pliegue.

Ejemplo:
    export FSAI_SALT="<sal-secreta>" FSAI_DATABASE_URL="postgresql+psycopg2://u:c@host:5432/db"
    python train_pipeline.py --ruta-datos data/enaho_2025_hogares.parquet --escenario todos \
        --modelos M1 M2 M3 M4 M5 --esquemas V1 V1c V2 V3 V5 --variante-v2 ambas --trials 100

Modos de las FASES 1-7 (adaptación tabular; ver CONTRATO.md "Módulos por fases"):
    --particion-70-15-15   genera y guarda la partición Desarrollo/Validación/Prueba (agrupada, estratificada) y termina
    --cv-anidada           además ejecuta la CV anidada 5x3 SOLO en Desarrollo + modelo final + umbral/calibración en Validación
    --solo-optuna          estudio Optuna persistente y reanudable sobre Desarrollo (Fase 7)
Estos modos NO alteran el flujo clásico (V1..V5) descrito abajo ni tocan la Prueba (15 %).

Esquemas (sección 8.1): V1 = GroupKFold(5) por hogar_uid repetido N_REPETICIONES veces (20 bloques);
V1c = igual, agrupando por conglome; V2 = geográfico (lodo y/o macro); V3 = estratificado urbano/rural;
V4 = temporal (requiere 'anio'); V5 = ponderación con factor_expansion (modo de métricas, no partición:
siempre se reporta ponderado y no ponderado; si V5 es el único esquema pedido se aplica sobre V1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

from fsai import config as C
from fsai.data_loader import (cargar_enaho, construir_hogar_uid, derivar_grupos_equidad, recodificar_fies_crudo,
                              validar_calidad)
from fsai.explainability import contribuciones_a_jsonb, importancia_permutacion, shap_global
from fsai.fairness import brechas_por_grupo
from fsai.fies import UMBRALES_RAW, agregar_columnas_fies
from fsai.metrics_weighted import (calcular_metricas, predecir_con_umbral, preparar_pesos_ajuste, recall_clase,
                                   umbral_clase_severa)
from fsai.models import ajustar_pipeline, construir_pipeline, predecir_proba
from fsai.optuna_optimizer import optimizar_modelo
from fsai.preprocessing import dividir_features, enriquecer_para_escenario, preparar_matriz
from fsai.report_generator import exportar_excel, exportar_pdf, exportar_word
from fsai.statistical_tests import (bootstrap_estratificado, delong_ovr, friedman_nemenyi, holm, mcnemar_ponderado,
                                    wilcoxon_pareado)
from fsai.validation import (columna_grupo_interna, esquema_base, generar_particiones, guardar_particiones,
                             hash_particiones, normalizar_esquemas)

logger = logging.getLogger("fsai.train")

NOTA_CIRCULARIDAD = (
    "ADVERTENCIA DE CIRCULARIDAD FIES: fies_nivel se deriva de fies_q1..fies_q8. Los escenarios 'fies' e "
    "'hibrido' usan esos ítems (o su suma) como predictores del propio objetivo: son tareas de "
    "RECONSTRUCCION/CLASIFICACION y su desempeño casi perfecto no mide capacidad predictiva. Solo el "
    "escenario 'socio' (sin variables FIES) constituye la PREDICCION real a partir de condiciones del hogar.")


# Desviación declarada de cómputo (--m1-solo-l2): saga+L1 tarda 1-1,5 min por ajuste con ~18 000 x 65 (hasta 1 089 iteraciones),
# lo que hace inviable la búsqueda de M1; se restringe la penalización a L2. Se registra en config_json.
ESPACIO_M1_SOLO_L2: dict[str, dict] = {"penalty": {"opciones": ["l2"]}}

# Sección 8.1.2: en V2 se excluye del conjunto de predictores la variable que define los bloques,
# porque una categoría ausente del train no puede aprenderse.
EXCLUIDAS_POR_ESQUEMA: dict[str, list[str]] = {"V2_lodo": ["departamento"],
                                               "V2_macro": ["departamento", "region_natural"]}


# --------------------------------------------------------------------------
# Argumentos
# --------------------------------------------------------------------------
def parsear_argumentos(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline CRISP-DM FoodSecurity AI (inseguridad alimentaria, ENAHO/FIES)")
    p.add_argument("--ruta-datos", required=True, type=Path, help="Archivo analítico o directorio de módulos ENAHO")
    p.add_argument("--modulos", nargs="*", default=None, help="Módulos a unir si --ruta-datos es un directorio")
    p.add_argument("--fies-crudo", action="store_true", help="fies_q* vienen con codificación INEI (1=Sí, 2=No)")
    p.add_argument("--escenario", default="todos", choices=C.ESCENARIOS_TODOS + ["todos"],
                   help="'todos' = socio, fies, hibrido (NO incluye socio_terr, que se pide explícitamente)")
    p.add_argument("--modelos", nargs="+", default=C.MODELOS, choices=C.MODELOS_TODOS,
                   help="M1-M5 del protocolo; M6 (HistGradientBoosting) es un extra opcional")
    p.add_argument("--esquemas", nargs="+", default=["V1"], choices=list(dict.fromkeys(C.ESQUEMAS_VALIDACION + C.ESQUEMAS_PARTICION)),
                   help="V1, V1c, V2 (o V2_lodo/V2_macro), V3, V4; V5 = métricas ponderadas (no es partición)")
    p.add_argument("--variante-v2", default="ambas", choices=C.VARIANTES_V2 + ["ambas"],
                   help="Variantes de V2: lodo (departamentos), macro (costa/sierra/selva) o ambas")
    p.add_argument("--trials", type=int, default=C.N_TRIALS, help="Ensayos Optuna por modelo y pliegue externo")
    p.add_argument("--n-boot", type=int, default=C.N_BOOT)
    p.add_argument("--smote", action="store_true", help="SMOTE dentro del pipeline (excluyente con pesos de ajuste)")
    p.add_argument("--ponderar-ajuste", action="store_true", help="Usa factor_expansion como sample_weight al ajustar")
    p.add_argument("--recorte-pesos-p", type=float, default=None, metavar="P",
                   help="V2: recorta los pesos de AJUSTE en el percentil P (p. ej. 99); requiere --ponderar-ajuste. Defecto: sin recorte")
    p.add_argument("--normalizar-pesos", action="store_true", help="V2: normaliza los pesos de AJUSTE a media 1 (requiere --ponderar-ajuste)")
    p.add_argument("--sin-ingenieria", action="store_true", help="V2: en socio_terr NO agrega las variables de ingeniería (solo territoriales)")
    p.add_argument("--optimizar-umbrales", action="store_true",
                   help="V2 (modo --cv-anidada): multiplicadores por clase para F1-macro, ajustados SOLO en Validación (fsai.umbrales)")
    p.add_argument("--umbral-severa", action="store_true", help="Umbral de clase severa fijado en train (esp. mín.)")
    p.add_argument("--umbrales-fies", type=json.loads, default=None, help='JSON {"leve":..,"moderada":..,"severa":..} [POR VALIDAR]')
    p.add_argument("--params-rasch", type=Path, default=None, help="JSON con dificultades de ítem fies_q1..q8 [POR VALIDAR]")
    p.add_argument("--m1-solo-l2", action="store_true", help="Restringe la penalización de M1 a L2 (desviación de cómputo declarada)")
    p.add_argument("--n-shap", type=int, default=500)
    p.add_argument("--nombre-dataset", default="ENAHO")
    p.add_argument("--fuente-dataset", default="ENAHO/INEI", help="Etiqueta de fuente en fsai.datasets")
    p.add_argument("--reusar-dataset", type=int, default=None,
                   help="dataset_id ya cargado en fsai.datasets: no vuelve a insertar hogares ni FIES")
    p.add_argument("--nombre-experimento", default=None, help="Prefijo del nombre del experimento (por defecto, el hash)")
    p.add_argument("--anio", type=int, default=None)
    p.add_argument("--sin-db", action="store_true", help="No persiste en PostgreSQL")
    p.add_argument("--sin-optuna-db", action="store_true", help="Optuna en memoria (no persiste estudios)")
    p.add_argument("--salida", type=Path, default=C.ARTIFACTS_DIR)
    # --- Fases 2 y 7 (partición 70/15/15, CV anidada, Optuna UI) ---
    p.add_argument("--particion-70-15-15", action="store_true", help="Partición Desarrollo 70 %% / Validación 15 %% / Prueba 15 %%")
    p.add_argument("--cv-anidada", action="store_true", help="CV anidada 5 externos x 3 internos SOLO en Desarrollo (implica la partición)")
    p.add_argument("--agrupar-por", default="hogar_uid", choices=["hogar_uid", "conglome"], help="Grupo que no cruza particiones")
    p.add_argument("--trials-final", type=int, default=None, help="Ensayos de la búsqueda final sobre todo Desarrollo (por defecto = --trials)")
    p.add_argument("--solo-optuna", action="store_true", help="Fase 7: estudio Optuna persistente sobre Desarrollo")
    p.add_argument("--estudio-nombre", default=None, help="Nombre del estudio Optuna (reanuda si existe)")
    p.add_argument("--almacenamiento", default="sqlite", choices=["sqlite", "memoria"], help="Persistencia del estudio (Fase 7)")
    p.add_argument("--espacio-json", type=Path, default=None, help="JSON {param: {lo, hi | opciones}} recortado al rango válido")
    p.add_argument("--timeout", type=float, default=None, help="Segundos máximos de la búsqueda Optuna (Fase 7)")
    p.add_argument("--semilla-optuna", type=int, default=C.SEED)
    p.add_argument("--pruner", default="mediana", choices=["mediana", "ninguno"])
    p.add_argument("--particion-json", type=Path, default=None, help="particion.json de una corrida de Fase 2 (Fase 7)")
    return p.parse_args(argv)


def _hash_ruta(ruta: Path) -> str:
    """SHA-256 del archivo (o de nombres+tamaños si es un directorio)."""
    h = hashlib.sha256()
    if ruta.is_file():
        with ruta.open("rb") as f:
            for bloque in iter(lambda: f.read(1 << 20), b""):
                h.update(bloque)
    else:
        for q in sorted(ruta.glob("*")):
            h.update(f"{q.name}:{q.stat().st_size}".encode())
    return h.hexdigest()


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
def preparar_datos(args: argparse.Namespace) -> pd.DataFrame:
    df = cargar_enaho(args.ruta_datos, args.modulos)
    if "hogar_uid" not in df.columns:
        df["hogar_uid"] = construir_hogar_uid(df, C.SALT)
    if args.fies_crudo:
        df = recodificar_fies_crudo(df)
    params_items = json.loads(args.params_rasch.read_text(encoding="utf-8")) if args.params_rasch else None
    df = agregar_columnas_fies(df, args.umbrales_fies or UMBRALES_RAW, params_items)
    antes = len(df)
    df = df[df[C.COL_OBJETIVO].notna()].copy()
    if len(df) < antes:
        logger.warning("Se excluyeron %d hogares sin FIES completo", antes - len(df))
    df[C.COL_OBJETIVO] = df[C.COL_OBJETIVO].astype(int)
    df[C.COL_BINARIA] = df[C.COL_BINARIA].astype(int)
    df = derivar_grupos_equidad(df)
    informe = validar_calidad(df)
    if not informe["ok"]:
        raise ValueError(f"Calidad de datos insuficiente: {informe['errores']}")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Núcleo de modelado
# --------------------------------------------------------------------------
def _umbral_en_train(codigo, params, X, y, w, g, num, cat, usar_smote) -> float:
    """Umbral de la clase severa con predicciones fuera de muestra de la CV interna (solo train)."""
    oof = np.zeros((len(y), len(C.CLASES)))
    for tr, va in GroupKFold(n_splits=C.INNER_FOLDS).split(X, y, g):
        pipe = construir_pipeline(codigo, params, num, cat, usar_smote)
        ajustar_pipeline(pipe, X.iloc[tr], y[tr], None if (w is None or usar_smote) else w[tr], g[tr])
        oof[va] = predecir_proba(pipe, X.iloc[va])
    return umbral_clase_severa(y, oof, w)


def ejecutar_cv(df, X, num, cat, escenario, codigo, esquema, particiones, args, run_id, dir_run) -> dict[str, Any]:
    """Entrena y predice cada bloque (rep, fold). ``oof``/``fold`` reúnen la predicción fuera de muestra de
    referencia (bloques agregables: repetición 0 y sin transferencia); ``bloques`` conserva TODOS los bloques
    (p. ej. las 20 repeticiones x pliegues de V1/V1c) para métricas por bloque y pruebas pareadas."""
    y = df[C.COL_OBJETIVO].to_numpy()
    w = df[C.COL_PESO].to_numpy()
    g = df[columna_grupo_interna(esquema)].to_numpy()      # CV interna agrupada: hogar_uid (conglome en V1c)
    w_fit = preparar_pesos_ajuste(w, getattr(args, "recorte_pesos_p", None), getattr(args, "normalizar_pesos", False))
    w_ajuste = w_fit if args.ponderar_ajuste else None
    oof = np.full((len(df), len(C.CLASES)), np.nan)
    pred_umbral = np.full(len(df), -1)
    fold_id = np.full(len(df), -1)
    params_folds: dict[int, dict] = {}
    mejores: list[float] = []
    bloques: list[dict[str, Any]] = []
    pipe_f0 = None
    for j, part in enumerate(particiones):
        tr, te = part
        t0 = time.time()
        Xtr, ytr, gtr = X.iloc[tr], y[tr], g[tr]
        wtr = None if w_ajuste is None else w_ajuste[tr]
        # Optuna usa SOLO train del bloque; el F1 de validación interna se pondera con factor_expansion
        estudio = optimizar_modelo(
            codigo, Xtr, ytr, gtr, pesos=w[tr], n_trials=args.trials, features_num=num, features_cat=cat,
            usar_smote=args.smote, ponderar_ajuste=args.ponderar_ajuste, pesos_ajuste=w_fit[tr],
            nombre_estudio=f"{run_id}_{escenario}_{esquema}_{codigo}_r{part.rep}f{part.fold}",
            storage_url="" if args.sin_optuna_db else (C.DATABASE_URL or None),   # "" = estudio en memoria
            espacio=ESPACIO_M1_SOLO_L2 if (codigo == "M1" and getattr(args, "m1_solo_l2", False)) else None)
        params_folds[j] = estudio.best_params if codigo != "M5" else _params_m5(estudio)
        mejores.append(float(estudio.best_value))
        # Entrenamiento final del bloque con los mejores hiperparámetros
        pipe = construir_pipeline(codigo, params_folds[j], num, cat, args.smote)
        ajustar_pipeline(pipe, Xtr, ytr, wtr, gtr)
        proba = predecir_proba(pipe, X.iloc[te])
        pu = None
        if args.umbral_severa:
            u = _umbral_en_train(codigo, params_folds[j], Xtr, ytr, w[tr], gtr, num, cat, args.smote)
            pu = predecir_con_umbral(proba, u)
        bloques.append({"bloque": j, "rep": part.rep, "fold": part.fold, "estrato": part.estrato,
                        "agregable": part.agregable, "transferencia": part.transferencia, "te": te, "proba": proba, "pred_umbral": pu})
        if part.agregable:
            oof[te] = proba
            fold_id[te] = part.fold
            if pu is not None:
                pred_umbral[te] = pu
        if esquema == "V1" and part.rep == 0:
            if part.fold == 0:
                pipe_f0 = (pipe, te)
            joblib.dump(pipe, dir_run / "modelos" / f"{escenario}_{codigo}_{esquema}_f{part.fold}.joblib")
        logger.info("%s %s %s rep %d fold %d: F1 interno %.4f (%.0fs)", escenario, codigo, esquema, part.rep,
                    part.fold, mejores[-1], time.time() - t0)
    return {"oof": oof, "fold": fold_id, "pred_umbral": pred_umbral, "params": params_folds,
            "mejor_valor": float(np.mean(mejores)), "pipe_f0": pipe_f0, "bloques": bloques}


def _params_m5(estudio) -> dict:
    """Reconstruye params de M5 (capas como lista) desde el mejor ensayo."""
    from fsai.optuna_optimizer import CONFIGS_CAPAS
    p = dict(estudio.best_params)
    p["capas"] = CONFIGS_CAPAS[p.pop("capas_cfg")]
    return p


def _metricas_bloque(y, w, proba, pred_umbral, ponderada) -> dict[str, float]:
    met = calcular_metricas(y, proba, w if ponderada else None)
    if pred_umbral is not None and np.any(pred_umbral >= 0):
        neg = y != C.CLASE_SEVERA
        met["recall_severa_umbral"] = recall_clase(y, pred_umbral, C.CLASE_SEVERA, w if ponderada else None)
        met["especificidad_severa_umbral"] = float(np.average(pred_umbral[neg] != C.CLASE_SEVERA,
                                                              weights=w[neg] if ponderada else None))
    return met


def metricas_del_modelo(df, res, escenario, codigo, esquema, n_boot) -> list[dict]:
    """Métricas por bloque (rep x fold), ponderadas (V5) y no ponderadas; en V3 además por estrato U/R; y
    agregadas sobre la predicción fuera de muestra de referencia con IC bootstrap estratificado.

    Cada fila lleva ``nivel``: bloque | bloque_estrato | agregado | agregado_estrato. ``fold`` es el índice de
    bloque (en V1/V1c: rep*N_SPLITS+pliegue) y ``estrato`` etiqueta el bloque (repR_foldK, departamento...)."""
    y, w = df[C.COL_OBJETIVO].to_numpy(), df[C.COL_PESO].to_numpy()
    filas: list[dict] = []
    base = {"escenario": escenario, "esquema": esquema, "codigo": codigo}
    por_area = esquema_base(esquema) == "V3" and "area" in df.columns
    area = df["area"].astype(str).to_numpy() if por_area else None

    def fila(extra, ponderada, nombre, valor, ic=(None, None)):
        return {**base, **extra, "ponderada": ponderada, "nombre": nombre, "valor": valor,
                "ic_inf": ic[0], "ic_sup": ic[1]}

    for b in res["bloques"]:
        te, pu = b["te"], b["pred_umbral"]
        ident = {"fold": b["bloque"], "rep": b["rep"], "pliegue": b["fold"], "estrato": b["estrato"],
                 "agregable": b["agregable"], "comparable": not b["transferencia"], "nivel": "bloque"}
        for ponderada in (True, False):
            met = _metricas_bloque(y[te], w[te], b["proba"], pu, ponderada)
            filas += [fila(ident, ponderada, n, v) for n, v in met.items()]
        if por_area:     # V3: métricas por estrato urbano/rural dentro del bloque
            for a in sorted(set(area[te])):
                ma = area[te] == a
                if ma.sum() < 2:
                    continue
                ident_a = {**ident, "estrato": f"{b['estrato']}|{a}", "nivel": "bloque_estrato"}
                for ponderada in (True, False):
                    met = _metricas_bloque(y[te][ma], w[te][ma], b["proba"][ma], None if pu is None else pu[ma], ponderada)
                    filas += [fila(ident_a, ponderada, n, v) for n, v in met.items()]
    # Agregado sobre la predicción fuera de muestra de referencia, con IC percentil (bootstrap por clase)
    m = res["fold"] >= 0
    agr = {"fold": None, "rep": None, "pliegue": None, "agregable": True, "comparable": True}
    f1 = lambda yy, pp, ww: precision_recall_fscore_support(  # noqa: E731
        yy, pp.argmax(1), labels=C.CLASES, average="macro", zero_division=0, sample_weight=ww)[2]
    rs = lambda yy, pp, ww: recall_clase(yy, pp.argmax(1), C.CLASE_SEVERA, ww)  # noqa: E731
    for ponderada in (True, False):
        met = calcular_metricas(y[m], res["oof"][m], w[m] if ponderada else None)
        ident = {**agr, "estrato": "agregado", "nivel": "agregado"}
        for nombre, fn in (("f1_macro", f1), ("recall_severa", rs)):
            ic = bootstrap_estratificado(y[m], res["oof"][m], fn, n_boot=n_boot, pesos=w[m] if ponderada else np.ones(m.sum()))
            filas.append(fila(ident, ponderada, nombre, met[nombre], (ic["ic_inf"], ic["ic_sup"])))
        filas += [fila(ident, ponderada, n, v) for n, v in met.items() if n not in ("f1_macro", "recall_severa")]
        if por_area:
            for a in sorted(set(area[m])):
                ma = m & (area == a)
                if ma.sum() < 2:
                    continue
                met_a = calcular_metricas(y[ma], res["oof"][ma], w[ma] if ponderada else None)
                filas += [fila({**agr, "estrato": f"area_{a}", "nivel": "agregado_estrato"}, ponderada, n, v)
                          for n, v in met_a.items()]
    return filas


def matriz_bloques(filas_metricas: list[dict], escenario, esquema, codigos: list[str], nombre="f1_macro") -> pd.DataFrame:
    """Matriz bloques x modelos de una métrica ponderada. En V1/V1c son rep x fold = 20 bloques (pares)."""
    mf = pd.DataFrame([f for f in filas_metricas if f["escenario"] == escenario and f["esquema"] == esquema
                       and f["ponderada"] and f["nombre"] == nombre and f["nivel"] == "bloque" and f["comparable"]])
    if mf.empty:
        return pd.DataFrame()
    return mf.pivot(index="fold", columns="codigo", values="valor")[codigos].dropna()


def pruebas_estadisticas(df, resultados_esc: dict[str, dict], filas_metricas: list[dict], escenario, esquema) -> list[dict]:
    """Friedman+Nemenyi, Wilcoxon, McNemar ponderado y DeLong (clase severa) entre modelos; Holm por familia.

    Friedman y Wilcoxon usan el F1-macro ponderado por bloque (rep x fold): 20 pares en V1/V1c."""
    codigos = sorted(resultados_esc)
    out: list[dict] = []
    if len(codigos) < 2:
        return out
    y, w = df[C.COL_OBJETIVO].to_numpy(), df[C.COL_PESO].to_numpy()
    mask = np.all([resultados_esc[c]["fold"] >= 0 for c in codigos], axis=0)
    mat = matriz_bloques(filas_metricas, escenario, esquema, codigos)
    hay_mat = not mat.empty and len(mat) >= 2
    if hay_mat and len(codigos) >= 3:
        fr = friedman_nemenyi(mat.to_numpy(), codigos)
        out.append({"escenario": escenario, "esquema": esquema, "prueba": "friedman", "comparacion": "global",
                    "estadistico": fr["chi2"], "p_valor": fr["p_valor"], "p_ajustado": fr["p_valor"],
                    "detalle": json.dumps({"rangos": {k: float(v) for k, v in fr["rangos_medios"].items()},
                                           "dif_critica": fr["diferencia_critica"], "n_bloques": fr["n_bloques"]})})
        for a, b in combinations(codigos, 2):
            out.append({"escenario": escenario, "esquema": esquema, "prueba": "nemenyi", "comparacion": f"{a}-{b}",
                        "estadistico": None, "p_valor": float(fr["nemenyi_p"].loc[a, b]),
                        "p_ajustado": float(fr["nemenyi_p"].loc[a, b]), "detalle": ""})
    familias: dict[str, list[dict]] = {"wilcoxon": [], "mcnemar": [], "delong_severa": []}
    for a, b in combinations(codigos, 2):
        if hay_mat:
            r = wilcoxon_pareado(mat[a].to_numpy(), mat[b].to_numpy())
            familias["wilcoxon"].append({"comparacion": f"{a}-{b}", "estadistico": r["estadistico"], "p_valor": r["p_valor"],
                                         "detalle": f"n={r['n']} pares (rep x fold); p_min_posible={r['p_minimo_posible']:.2e}"})
        pa, pb = (resultados_esc[c]["oof"][mask].argmax(1) for c in (a, b))
        r = mcnemar_ponderado(y[mask], pa, pb, w[mask])
        familias["mcnemar"].append({"comparacion": f"{a}-{b}", "estadistico": r["estadistico"], "p_valor": r["p_valor"],
                                    "detalle": f"b={r['b']:.1f}; c={r['c']:.1f}"})
        r = delong_ovr(y[mask], resultados_esc[a]["oof"][mask], resultados_esc[b]["oof"][mask], C.CLASE_SEVERA)
        familias["delong_severa"].append({"comparacion": f"{a}-{b}", "estadistico": r["z"], "p_valor": r["p_valor"],
                                          "detalle": f"AUC {a}={r['auc1']:.4f}; {b}={r['auc2']:.4f}"})
    for prueba, lista in familias.items():
        if lista:
            ajust = holm([f["p_valor"] for f in lista])
            out += [{"escenario": escenario, "esquema": esquema, "prueba": prueba, **f, "p_ajustado": float(pa)}
                    for f, pa in zip(lista, ajust)]
    return out


# --------------------------------------------------------------------------
# Persistencia
# --------------------------------------------------------------------------
def persistir(df, args, cfg, hash_archivo, almacen) -> None:
    from sqlalchemy.orm import Session
    from fsai import database as D
    engine = D.get_engine()
    D.crear_esquema(engine)
    with Session(engine) as s, s.begin():
        if args.reusar_dataset is not None:
            ds = args.reusar_dataset
        else:
            ds = D.guardar_dataset(s, args.nombre_dataset, args.fuente_dataset,
                                 args.anio or (int(df['anio'].max()) if 'anio' in df.columns else None), cfg["hash"], hash_archivo, len(df))
        if args.reusar_dataset is None and "area" in df.columns and "region_natural" in df.columns:
            D.guardar_hogares(s, df, ds)
            D.guardar_respuestas_fies(s, df)
            D.guardar_indicadores(s, df)
            D.guardar_condiciones(s, df)
        ids_exp: dict[tuple, int] = {}
        for (esc, esq) in sorted({(k[0], k[1]) for k in almacen["res"]}):
            # esq puede ser una clave con variante (V2_lodo, V2_macro): la BD guarda el código base (V2)
            ids_exp[(esc, esq)] = D.guardar_experimento(s, ds, f"{args.nombre_experimento or cfg['hash']}_{esc}_{esq}", esquema_base(esq), esc,
                                                        C.SEED, {**cfg, "esquema_ejecutado": esq})
        for (esc, esq, cod), res in almacen["res"].items():
            mid = D.guardar_modelo(s, ids_exp[(esc, esq)], cod, res["params"], args.trials, res["mejor_valor"],
                                   str(almacen["dir_run"] / "modelos"))
            if esq == "V1":
                m = res["fold"] >= 0
                D.guardar_predicciones(s, mid, df.loc[m, C.COL_GRUPO], res["fold"][m], df.loc[m, C.COL_OBJETIVO],
                                       res["oof"][m].argmax(1), res["oof"][m])
            for f in (x for x in almacen["metricas"] if (x["escenario"], x["esquema"], x["codigo"]) == (esc, esq, cod)):
                D.guardar_metricas(s, mid, esquema_base(esq), f["ponderada"], {f["nombre"]: f["valor"]},
                                   {f["nombre"]: (f["ic_inf"], f["ic_sup"])} if f["ic_inf"] is not None else None,
                                   None if f["fold"] is None else int(f["fold"]),
                                   None if f["estrato"] == "agregado" else str(f["estrato"]))
            for ex in almacen["explicaciones"]:
                if (ex["escenario"], ex["esquema"], ex["codigo"]) == (esc, esq, cod):
                    D.guardar_explicacion(s, mid, ex["tipo"], ex["contribuciones"])
        for esc, esq in ids_exp:
            D.finalizar_experimento(s, ids_exp[(esc, esq)])
        D.registrar_auditoria(s, "train_pipeline", "ejecucion_completa", "dataset", str(ds), {"hash_config": cfg["hash"]})
    logger.info("Resultados persistidos en PostgreSQL (dataset_id=%s)", ds)


# --------------------------------------------------------------------------
# Principal
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = parsear_argumentos(argv)
    if args.solo_optuna:                       # Fase 7: no altera el flujo clásico
        return _modo_optuna(args)
    if args.cv_anidada or args.particion_70_15_15:   # Fase 2
        return _modo_fase2(args)
    cfg = {"constantes": C.constantes_protocolo(), "escenario": args.escenario, "modelos": args.modelos,
           "esquemas": args.esquemas, "variante_v2": args.variante_v2, "trials": args.trials, "smote": args.smote, "ponderar_ajuste": args.ponderar_ajuste,
           "umbral_severa": args.umbral_severa, "m1_solo_l2": args.m1_solo_l2, "umbrales_fies": args.umbrales_fies or UMBRALES_RAW,
           "rasch_inyectado": args.params_rasch is not None, "hash_datos": _hash_ruta(args.ruta_datos)}
    cfg["hash"] = C.hash_configuracion(cfg)
    run_id = cfg["hash"]
    dir_run = args.salida / f"run_{run_id}"
    (dir_run / "modelos").mkdir(parents=True, exist_ok=True)
    C.configurar_logging(dir_run / "pipeline.log")
    (dir_run / "config.json").write_text(json.dumps(cfg, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    logger.info("Ejecución %s | %s", run_id, NOTA_CIRCULARIDAD)
    np.random.seed(C.SEED)

    try:
        df = preparar_datos(args)
    except Exception:
        logger.exception("Fallo en carga/calidad/FIES")
        return 2
    escenarios = C.ESCENARIOS if args.escenario == "todos" else [args.escenario]
    almacen: dict[str, Any] = {"res": {}, "metricas": [], "pruebas": [], "explicaciones": [], "equidad": [],
                               "importancia": [], "dir_run": dir_run}
    fallos: list[str] = []

    # Particiones: UNA vez por esquema, verificadas, guardadas y compartidas por M1-M5 y por los 3 escenarios
    esquemas_exec, pidio_v5 = normalizar_esquemas(args.esquemas, args.variante_v2)
    logger.info("Esquemas a ejecutar: %s | V5 (métricas ponderadas y no ponderadas) %s", esquemas_exec,
                "solicitado" if pidio_v5 else "siempre reportado")
    particiones: dict[str, Any] = {}
    for esq in esquemas_exec:
        try:
            particiones[esq] = generar_particiones(df, esq)
            guardar_particiones(particiones[esq], dir_run / f"particiones_{esq}.json",
                                {"esquema": esq, "seed": C.SEED, "n_repeticiones": C.N_REPETICIONES})
        except Exception:
            logger.exception("Esquema %s no disponible", esq)
            fallos.append(f"particiones_{esq}")
    esquemas_ok = list(particiones)

    for esc in escenarios:
        df_esc = enriquecer_para_escenario(df, esc, ingenieria=not getattr(args, "sin_ingenieria", False))   # solo socio_terr cambia
        num, cat = dividir_features(df_esc, esc)
        X = preparar_matriz(df_esc, num, cat)
        for esq in esquemas_ok:
            cat_e = [c for c in cat if c not in EXCLUIDAS_POR_ESQUEMA.get(esq, [])]
            res_esc: dict[str, dict] = {}
            for cod in args.modelos:
                try:
                    res = ejecutar_cv(df, X, num, cat_e, esc, cod, esq, particiones[esq], args, run_id, dir_run)
                    res_esc[cod] = res
                    almacen["res"][(esc, esq, cod)] = res
                    almacen["metricas"] += metricas_del_modelo(df, res, esc, cod, esq, args.n_boot)
                except Exception:
                    logger.exception("Fallo %s/%s/%s", esc, esq, cod)
                    fallos.append(f"{esc}_{esq}_{cod}")
            if res_esc:
                try:
                    almacen["pruebas"] += pruebas_estadisticas(df, res_esc, almacen["metricas"], esc, esq)
                    if esq == "V1":
                        _explicar_y_equidad(df, X, esc, esq, res_esc, almacen, args)
                except Exception:
                    logger.exception("Fallo en pruebas/explicabilidad/equidad (%s/%s)", esc, esq)
                    fallos.append(f"post_{esc}_{esq}")

    # Artefactos tabulares
    t_met = pd.DataFrame(almacen["metricas"])
    t_met.to_csv(dir_run / "metricas.csv", index=False)
    t_pru = pd.DataFrame(almacen["pruebas"])
    t_pru.to_csv(dir_run / "pruebas_estadisticas.csv", index=False)
    t_eq = pd.concat(almacen["equidad"], ignore_index=True) if almacen["equidad"] else pd.DataFrame()
    t_eq.to_csv(dir_run / "equidad.csv", index=False)
    (dir_run / "hiperparametros.json").write_text(json.dumps(
        {"|".join(k): v["params"] for k, v in almacen["res"].items()}, indent=2, default=str), encoding="utf-8")

    # Persistencia en PostgreSQL
    if not args.sin_db and almacen["res"]:
        try:
            persistir(df, args, cfg, cfg["hash_datos"], almacen)
        except Exception:
            logger.exception("Fallo al persistir en PostgreSQL")
            fallos.append("persistencia")

    # Reportes
    try:
        agg = (t_met[t_met["nivel"] == "bloque"].groupby(["escenario", "esquema", "codigo", "ponderada", "nombre"])["valor"]
               .agg(["mean", "std"]).reset_index()) if not t_met.empty else pd.DataFrame()
        def _blk(titulo, df):
            if df is None or not len(df):
                return {"tipo": "vacio", "texto": "Sin datos disponibles."}
            return {"tipo": "tabla", "titulo": titulo, "df": df, "pie": None, "max_filas": None}
        doc_rep = {
            "titulo": f"Informe FoodSecurity AI ({run_id})", "subtitulo": "FoodSecurity AI",
            "encabezado": f"FoodSecurity AI · {run_id}",
            "portada": [("Corrida", run_id), ("Nota metodológica", NOTA_CIRCULARIDAD)],
            "resumen": [NOTA_CIRCULARIDAD],
            "secciones": [
                {"titulo": "Advertencia metodológica", "bloques": [{"tipo": "p", "texto": NOTA_CIRCULARIDAD}]},
                {"titulo": "Resumen de métricas por modelo", "hoja": "resumen_pliegues",
                 "bloques": [_blk("Media y DE entre bloques rep x pliegue", agg)]},
                {"titulo": "Pruebas estadísticas", "hoja": "pruebas", "bloques": [_blk("Pruebas estadísticas", t_pru)]},
                {"titulo": "Equidad", "hoja": "equidad", "bloques": [_blk("Brechas de TPR", t_eq)]},
                {"titulo": "Métricas por pliegue", "hoja": "metricas_pliegue", "bloques": [_blk("Métricas por pliegue", t_met)]},
            ],
        }
        exportar_excel(doc_rep, dir_run / "reporte.xlsx", {"hash_config": run_id, "nota": NOTA_CIRCULARIDAD})
        exportar_word(doc_rep, dir_run / "informe.docx")
        exportar_pdf(doc_rep, dir_run / "informe.pdf")
    except Exception:
        logger.exception("Fallo al generar reportes")
        fallos.append("reportes")

    if fallos:
        logger.error("Finalizó con fallos: %s", fallos)
        return 1
    logger.info("Pipeline completado. Artefactos en %s", dir_run)
    return 0


# --------------------------------------------------------------------------
# Modos de las Fases 2 y 7 (partición 70/15/15, CV anidada, Optuna UI)
# --------------------------------------------------------------------------
PATRON_NOMBRE_FASE = re.compile(r"^[A-Za-z0-9_.\-]{3,60}$")


def _preparar_particion(args, dir_run: Path):
    """Datos + partición 70/15/15 (nueva o la de ``--particion-json``) verificada y guardada con huella."""
    from fsai.splits import cargar_particion, guardar_particion, particionar_70_15_15, verificar_particion
    df = preparar_datos(args)
    ruta = dir_run / "particion.json"
    if args.particion_json is not None:
        part, carga = cargar_particion(df, args.particion_json)
        huella = carga["huella"]
    else:
        part = particionar_70_15_15(df, C.SEED, args.agrupar_por)
        huella = guardar_particion(df, part, ruta, C.SEED, args.agrupar_por, {"ruta_datos": str(args.ruta_datos)})
    ver = verificar_particion(df, part)
    if ver["cruces_hogar_uid_total"] or not ver["exhaustiva"]:
        raise RuntimeError(f"Partición inválida: {ver}")
    logger.info("Partición 70/15/15: %s | huella %s | cruces conglome=%s", ver["tamanos"], huella[:12], ver.get("cruces_conglome_total"))
    return df, part, huella, ver


def _modo_fase2(args: argparse.Namespace) -> int:
    from fsai.nested_cv import ejecutar_fase2
    nombre = args.nombre_experimento or time.strftime("fase2_%Y%m%d_%H%M%S")
    if not PATRON_NOMBRE_FASE.fullmatch(nombre):
        raise ValueError("--nombre-experimento inválido (3-60 caracteres: letras, números, . _ -).")
    dir_run = C.DIR_FASES / nombre
    dir_run.mkdir(parents=True, exist_ok=True)
    C.configurar_logging(dir_run / "pipeline.log")
    df, part, huella, ver = _preparar_particion(args, dir_run)
    cfg = {"nombre": nombre, "modo": "cv_anidada" if args.cv_anidada else "solo_particion", "ruta_datos": str(args.ruta_datos),
           "hash_datos": _hash_ruta(args.ruta_datos), "huella_particion": huella, "agrupar_por": args.agrupar_por,
           "escenario": args.escenario, "modelos": args.modelos, "trials": args.trials, "trials_final": args.trials_final,
           "seed": C.SEED, "n_outer": C.N_SPLITS_EXTERNOS, "n_inner": C.INNER_FOLDS, "verificacion_particion": ver,
           "circularidad": NOTA_CIRCULARIDAD, "creado_en": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (dir_run / "config.json").write_text(json.dumps(cfg, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    if not args.cv_anidada:
        logger.info("Solo partición: no se ejecutó CV anidada. Carpeta %s", dir_run)
        return 0
    escenarios = C.ESCENARIOS if args.escenario == "todos" else [args.escenario]
    ejecutar_fase2(df, part, modelos=args.modelos, escenarios=escenarios, n_trials=args.trials, dir_run=dir_run,
                   n_trials_final=args.trials_final, ponderar_ajuste=args.ponderar_ajuste, usar_smote=args.smote,
                   ingenieria=not args.sin_ingenieria, recorte_pesos=args.recorte_pesos_p, normalizar_pesos=args.normalizar_pesos,
                   optimizar_umbrales=args.optimizar_umbrales)
    logger.info("CV anidada completada. Artefactos en %s", dir_run)
    return 0


def _modo_optuna(args: argparse.Namespace) -> int:
    """Fase 7: un estudio Optuna sobre Desarrollo, persistente (SQLite) y reanudable, con progreso en JSON."""
    from fsai.nested_cv import RegistroProgreso
    from fsai.optuna_optimizer import espacio_efectivo
    nombre = args.estudio_nombre
    if not nombre or not PATRON_NOMBRE_FASE.fullmatch(nombre):
        raise ValueError("--estudio-nombre requerido (3-60 caracteres: letras, números, . _ -).")
    if len(args.modelos) != 1 or args.escenario == "todos":
        raise ValueError("--solo-optuna requiere UN modelo y UN escenario.")
    codigo, esc = args.modelos[0], args.escenario
    C.DIR_OPTUNA.mkdir(parents=True, exist_ok=True)
    dir_est = C.DIR_OPTUNA / nombre
    dir_est.mkdir(parents=True, exist_ok=True)
    C.configurar_logging(dir_est / "pipeline.log")
    df, part, huella, _ = _preparar_particion(args, dir_est)
    espacio = json.loads(args.espacio_json.read_text(encoding="utf-8")) if args.espacio_json else None
    espacio_efectivo(codigo, espacio)                                  # valida y recorta rangos
    url = ("sqlite:///" + (C.DIR_OPTUNA / "estudios.db").as_posix()) if args.almacenamiento == "sqlite" else ""
    dev = enriquecer_para_escenario(df, esc, ingenieria=not args.sin_ingenieria).iloc[part["desarrollo"]].reset_index(drop=True)
    num, cat = dividir_features(dev, esc)
    X = preparar_matriz(dev, num, cat)
    prog = RegistroProgreso(dir_est)
    cfg = {"estudio": nombre, "modelo": codigo, "escenario": esc, "n_trials": args.trials, "timeout": args.timeout,
           "semilla": args.semilla_optuna, "sampler": "tpe", "pruner": args.pruner, "almacenamiento": args.almacenamiento,
           "espacio": espacio, "huella_particion": huella, "inicio": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (dir_est / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    prog.actualizar(estado="ejecutando", modelo=codigo, escenario=esc, fase="optuna", n_trials=args.trials,
                    mensaje="Búsqueda Optuna en Desarrollo (Prueba y Validación intactas)")
    try:
        est = optimizar_modelo(codigo, X, dev[C.COL_OBJETIVO].to_numpy(), dev[C.COL_GRUPO].to_numpy(),
                               pesos=dev[C.COL_PESO].to_numpy(), n_trials=args.trials, features_num=num, features_cat=cat,
                               usar_smote=args.smote, ponderar_ajuste=args.ponderar_ajuste, nombre_estudio=nombre,
                               storage_url=url, espacio=espacio, callbacks=[prog.callback_optuna(nombre)],
                               timeout=args.timeout, semilla=args.semilla_optuna, pruner=args.pruner)
    except Exception as exc:
        prog.actualizar(estado="error", mensaje=f"{type(exc).__name__}: {exc}")
        raise
    prog.actualizar(estado="completado", mensaje=f"Estudio completado: mejor valor {est.best_value:.4f}")
    logger.info("Estudio %s completado: mejor F1-macro interno %.4f", nombre, est.best_value)
    return 0


def _explicar_y_equidad(df, X, esc, esq, res_esc, almacen, args) -> None:
    """SHAP + permutación del mejor modelo (F1 ponderado agregado) y brechas de equidad con sus predicciones OOF."""
    y, w = df[C.COL_OBJETIVO].to_numpy(), df[C.COL_PESO].to_numpy()
    puntaje = {}
    for cod, r in res_esc.items():
        m = r["fold"] >= 0
        puntaje[cod] = calcular_metricas(y[m], r["oof"][m], w[m])["f1_macro"]
    mejor = max(puntaje, key=puntaje.get)
    logger.info("Mejor modelo en %s/%s por F1-macro ponderado OOF: %s", esc, esq, mejor)
    r = res_esc[mejor]
    pipe, te = r["pipe_f0"]                       # modelo del pliegue 0 explicado en SU conjunto de prueba
    ident = {"escenario": esc, "esquema": esq, "codigo": mejor}
    try:
        sg = shap_global(pipe, X.iloc[te], mejor, n_muestra=min(args.n_shap, 100) if mejor == "M4" else args.n_shap, split="prueba_pliegue_0")
        almacen["explicaciones"].append({**ident, "tipo": "shap_global", "contribuciones": contribuciones_a_jsonb(
            sg["nombres"], sg["importancia_media"], {"n_muestra": sg["n_muestra"], "version": 1, "beeswarm": sg.get("beeswarm"),
                                                    "por_clase": {str(k): dict(zip(sg["nombres"], v))
                                                                  for k, v in sg["importancia_por_clase"].items()}})})
    except Exception:
        logger.exception("SHAP falló para %s/%s", esc, mejor)
    imp = importancia_permutacion(pipe, X.iloc[te], y[te], w[te])
    almacen["importancia"].append(imp.assign(**ident))
    almacen["explicaciones"].append({**ident, "tipo": "permutation", "contribuciones": contribuciones_a_jsonb(
        imp["variable"].tolist(), imp["importancia"].to_numpy(), {"desv_est": imp["desv_est"].tolist()})})
    m = r["fold"] >= 0
    dfm = df[m].reset_index(drop=True)
    pred = r["oof"][m].argmax(1)
    for col in ["sexo_jefe", "grupo_edad_jefe", "grupo_educ_jefe", "area", "region_natural", "pobreza_monetaria"]:
        if col in dfm.columns and dfm[col].nunique() > 1:
            t = brechas_por_grupo(dfm, dfm[C.COL_OBJETIVO].to_numpy(), pred, col, C.CLASE_SEVERA, n_boot=args.n_boot)
            almacen["equidad"].append(t.assign(escenario=esc, esquema=esq, codigo=mejor))


if __name__ == "__main__":
    sys.exit(main())
