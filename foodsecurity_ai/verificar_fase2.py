"""verificar_fase2.py - Verificación instrumentada de la partición 70/15/15 y de la CV anidada (5 externos x 3 internos).

Uso (desde esta carpeta, con el .env de despliegue_local cargado en el entorno si hace falta FSAI_SALT):
    python verificar_fase2.py --ruta-datos ../../../despliegue_local/datos/demo_sintetico.csv --modelos M1 M2 --trials 3

Comprueba, con trazas reales (no solo por construcción):
  1. proporciones 70/15/15, partición exhaustiva y sin cruces de hogar_uid (y de conglome al agrupar por conglome);
  2. 5 pliegues externos x 3 internos; las búsquedas de Optuna SOLO ven filas de outer_train;
  3. cada outer_val se predice exactamente UNA vez por el modelo reentrenado del pliegue (y nunca durante la búsqueda);
  4. ninguna fila de Validación ni de Prueba entra en la CV anidada (la Prueba queda intacta);
  5. 5 observaciones agregadas por métrica (una por pliegue externo).
Usa datos DEMO sintéticos: la salida valida la mecánica, NO son resultados del estudio.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

import fsai.nested_cv as N
import fsai.optuna_optimizer as O
from fsai import config as C
from fsai.datasets import cargar_hogares_ruta
from fsai.splits import particionar_70_15_15, verificar_particion


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruta-datos", type=Path, required=True)
    ap.add_argument("--modelos", nargs="+", default=["M1", "M2"])
    ap.add_argument("--trials", type=int, default=3)
    a = ap.parse_args()
    df = cargar_hogares_ruta(a.ruta_datos)
    n = len(df)

    # 1) partición
    for ag in ("hogar_uid", "conglome"):
        part = particionar_70_15_15(df, C.SEED, ag)
        v = verificar_particion(df, part)
        prop = v["proporciones"]
        assert v["exhaustiva"] and v["cruces_hogar_uid_total"] == 0, v
        assert abs(prop["desarrollo"] - 0.70) < 0.01 and abs(prop["validacion"] - 0.15) < 0.01 and abs(prop["prueba"] - 0.15) < 0.01, prop
        if ag == "conglome":
            assert v["cruces_conglome_total"] == 0, v
        print(f"[OK] partición agrupada por {ag}: {prop} | cruces hogar_uid={v['cruces_hogar_uid_total']} conglome={v['cruces_conglome_total']}")
    part = particionar_70_15_15(df, C.SEED, "hogar_uid")
    dev, val, pru = (set(part[k]) for k in C.NOMBRES_PARTICION)
    assert not (dev & val) and not (dev & pru) and not (val & pru)

    # 2-4) instrumentación
    traza = {"optuna": [], "fit": [], "pred": []}
    orig_opt, orig_fit, orig_pred_n, orig_pred_o = O.optimizar_modelo, O.ajustar_pipeline, N.predecir_proba, O.predecir_proba

    def opt_espia(codigo, X, *args, **kw):
        traza["optuna"].append(set(X.index))
        traza["fit"].append([])
        return orig_opt(codigo, X, *args, **kw)

    def fit_espia(pipe, X, *args, **kw):
        if traza["fit"]:
            traza["fit"][-1].append(("fit", frozenset(X.index)))
        return orig_fit(pipe, X, *args, **kw)

    def pred_inner(pipe, X, *args, **kw):
        if traza["fit"]:
            traza["fit"][-1].append(("pred", frozenset(X.index)))
        return orig_pred_o(pipe, X, *args, **kw)

    llamadas_nested = []

    def pred_nested(pipe, X, *args, **kw):
        llamadas_nested.append(frozenset(X.index))
        return orig_pred_n(pipe, X, *args, **kw)

    N.optimizar_modelo, O.ajustar_pipeline, O.predecir_proba, N.predecir_proba = opt_espia, fit_espia, pred_inner, pred_nested
    # el pipeline usa la función importada en optuna_optimizer; ajustar_pipeline también se usa en nested_cv (reentrenamiento)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dir_run = Path(tmp)
            prog = N.RegistroProgreso(dir_run)
            df_dev_idx = part["desarrollo"]
            for cod in a.modelos:
                for k in traza:
                    traza[k].clear()
                llamadas_nested.clear()
                res = N.ejecutar_modelo(df, part, codigo=cod, escenario="socio", n_trials=a.trials, dir_run=dir_run, progreso=prog,
                                        n_trials_final=a.trials)
                assert len(res["folds"]) == 5 and res["n_outer"] == 5 and res["n_inner"] == 3
                # búsquedas: 5 externas + 1 final
                assert len(traza["optuna"]) == 6, len(traza["optuna"])
                outer = traza["optuna"][:5]
                union = set().union(*outer)
                assert len(union) == len(df_dev_idx) and all(len(s) < len(df_dev_idx) for s in outer), "Cada búsqueda externa debe ver solo outer_train"
                # outer_val de cada pliegue = Desarrollo \ outer_train
                for j, ot in enumerate(outer):
                    ov = set(range(len(df_dev_idx))) - ot
                    usados_inner = set().union(*[idx for _, idx in traza["fit"][j]])
                    assert usados_inner <= ot, f"pliegue {j}: la búsqueda interna tocó filas fuera de outer_train"
                    assert not (usados_inner & ov), f"pliegue {j}: la búsqueda interna vio outer_val"
                    n_fit_inner = sum(1 for t, _ in traza["fit"][j] if t == "fit")
                    assert n_fit_inner >= 3, n_fit_inner
                    # primer ensayo: 3 pliegues internos cuyas predicciones son disjuntas y cubren exactamente outer_train
                    preds_t1 = [idx for t, idx in traza["fit"][j] if t == "pred"][:3]
                    assert len(preds_t1) == 3 and not (preds_t1[0] & preds_t1[1]) and not (preds_t1[0] & preds_t1[2]) and not (preds_t1[1] & preds_t1[2])
                    assert set().union(*preds_t1) == ot, "los 3 pliegues internos deben particionar outer_train"
                # outer_val: una predicción por pliegue externo + una de Validación (modelo final); ninguna sobre Prueba
                n_val = len(part["validacion"])
                assert len(llamadas_nested) == 6, f"se esperaban 6 predicciones (5 outer_val + Validación), hubo {len(llamadas_nested)}"
                pred_outer, pred_validacion = llamadas_nested[:5], llamadas_nested[5]
                assert len(pred_validacion) == n_val
                for j, ov_idx in enumerate(pred_outer):          # cada outer_val == Desarrollo menos outer_train de su pliegue
                    assert set(ov_idx) == set(range(len(df_dev_idx))) - outer[j]
                assert all(f["auditoria"]["evaluaciones_outer_val"] == 1 and f["auditoria"]["prueba_tocada"] is False for f in res["folds"])
                agg = res["agregado"]["ponderada"]["f1_macro"]
                assert agg["n"] == 5
                print(f"[OK] {cod}: 5 externos x {res['n_inner']} internos; Optuna solo en outer_train; 1 evaluación por outer_val; "
                      f"Prueba sin tocar; F1 ponderado = {agg['media']:.3f} ± {agg['sd']:.3f} IC [{agg['ic_inf']:.3f}; {agg['ic_sup']:.3f}] (n={agg['n']}; DEMO)")
    finally:
        N.optimizar_modelo, O.ajustar_pipeline, O.predecir_proba, N.predecir_proba = orig_opt, orig_fit, orig_pred_o, orig_pred_n
    print("VERIFICACIÓN FASE 2: TODAS LAS COMPROBACIONES PASARON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
