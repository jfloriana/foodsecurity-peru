# Desviaciones declaradas del protocolo (entrenamiento REAL, ENAHO 2025)

Este documento lista, sin atenuar, lo que se hizo de forma distinta al protocolo (`protocolo/CONTRATO.md`, `N_TRIALS = 100`, 20 bloques en V1/V1c, etc.) y por que. Los mismos datos quedan en `fsai.experimentos.config_json["desviaciones_protocolo"]`.

## D1. Presupuesto de ensayos Optuna: 20 en lugar de 100 (escenario primario `socio`)
- Protocolo: 100 ensayos por modelo y pliegue externo.
- Ejecutado: **20 ensayos por modelo, en cada uno de los 5 pliegues externos y en la busqueda final sobre todo Desarrollo (6 busquedas por modelo = 120 ensayos por modelo)**. Igual para M1-M5 (misma semilla, misma particion, mismo sampler TPE y pruner por mediana).
- Motivo: tiempos medidos en una corrida de humo sobre datos reales con 3 ensayos y hilos repartidos entre los 5 modelos (M1 45 s, M2 239 s, M3 755 s, M4 1 548 s, M5 1 917 s por 18 ensayos, es decir 2,5 s, 13 s, 42 s, 86 s y 106 s por ensayo). Con 100 ensayos, M5 requeriria unas 17 h y M4 unas 14 h en este equipo (32 hilos, sin GPU); con 20 ensayos son unas 3,5 h y 3 h. 20 se eligio como el mayor valor uniforme que cabe en una jornada. Tiempos reales de la corrida primaria (CV anidada + busqueda final, escenario `socio`, procesos en paralelo): M1 235 s, M2 1 860 s, M3 7 589 s, M4 8 596 s, M5 10 280 s (2,9 h); fies/hibrido (5 ensayos): entre 30 s (M1) y 2 213 s (M4).
- Consecuencia: la busqueda de hiperparametros es mas pobre que la del protocolo (con 20 ensayos TPE explora poco los espacios de M4 y M5, de 8-11 dimensiones). Los modelos complejos (M4, M5) pueden estar subajustados en hiperparametros frente a M1/M2; las diferencias entre modelos deben leerse con esa cautela.

## D2. Escenarios `fies` e `hibrido`: 5 ensayos
- Son reconstruccion circular del objetivo; se ejecutaron con 5 ensayos por modelo y busqueda (igual para los 5 modelos) porque no miden prediccion y su desempeno casi perfecto no justifica el computo.

## D3. M1: solo penalizacion L2
- Protocolo: espacio de M1 con penalizacion L1 o L2 (solver `saga` para L1).
- Ejecutado: `penalty = l2` unicamente (`ESPACIO_POR_MODELO` en `scripts/entrenar_real_fase2.py`, `--m1-solo-l2` en `train_pipeline.py`).
- Motivo (medido): con ~18 000 x 65, L1 con `saga` tardo 1,9 s (C=0,01), 59 s (C=1, 822 iteraciones) y 86 s (C=100, 1 089 iteraciones) por ajuste, frente a 0,2-0,7 s con L2. Una busqueda de 20 ensayos x 3 pliegues internos x 6 busquedas era inviable. `class_weight` ('balanced'/'ninguno') y `C` se buscan igual.

## D4. Flujo clasico (V1/V1c/V2/V3): subconjunto y 3 ensayos
- Protocolo: V1 y V1c con 5x4 = 20 bloques, V2 (LODO y macro) y V3, todos con Optuna por bloque.
- Ejecutado: ver la seccion 4 de `RESULTADOS_ENTRENAMIENTO_REAL.md` (esquemas efectivamente ejecutados, solo escenario `socio`, **3 ensayos por bloque**, `n_boot = 200`, `n_shap = 200`). Lo que no aparece alli no se ejecuto (V1c y V2-LODO; V4 es imposible con un solo ano).
- Con 3 ensayos TPE es casi busqueda aleatoria: estos resultados son de sensibilidad del esquema de validacion, no una busqueda de hiperparametros comparable con la CV anidada primaria.

## D5. SHAP
- TreeSHAP en M2 (RandomForest), M5 (XGBoost) y, para M3, el `ShapValues` nativo de CatBoost (equivalente a TreeSHAP; el conversor de `shap.TreeExplainer` no terminaba con arboles simetricos de profundidad 10 y 4 clases).
- M1: `LinearExplainer` (exacto para el modelo logistico), no `KernelExplainer`; M4: `KernelExplainer` con submuestra de 100 filas y resumen k-means de 30 centros (nsamples=200).
- Muestra: submuestra aleatoria de la Prueba (seed 42), calculada DESPUES de la evaluacion unica y sin usarla para elegir nada.

## D6. Persistencia de la CV anidada en PostgreSQL
- El modo `--cv-anidada` del pipeline original solo escribe JSON/NPZ. Se anadio `scripts/post_fase2_real.py` (evaluacion unica de la Prueba, metricas con IC, pruebas, SHAP, equidad y escritura en `fsai.experimentos/modelos/predicciones/metricas/explicaciones`). El esquema de validacion registrado para estos experimentos es `V1` (agrupado por hogar_uid) y el nombre los distingue (`ENAHO_2025_REAL_<corrida>_<escenario>`); `config_json.modo = cv_anidada_70_15_15`.
- `fsai.predicciones` guarda las predicciones de la Prueba (fold = 0 como marcador, sin significado de pliegue).
- Las pruebas estadisticas entre modelos (DeLong, McNemar, Wilcoxon, Friedman/Nemenyi con Holm) quedan en `config_json.pruebas_estadisticas` y en `analisis/pruebas_estadisticas.csv` (no existe tabla para ellas en el DDL).

## D7. Hilos por modelo
- Se anadio `FSAI_N_HILOS` (fsai.config) para repartir los 32 hilos entre procesos paralelos (M1 1, M2 3, M3 6, M4 10, M5 10). No cambia el algoritmo; si altera los tiempos medidos (contienda por CPU) y, por tanto, las estimaciones de D1.

## D8. Predictor descartado
- `informalidad` es 100 % NaN: se elimina de los predictores con aviso en el registro (`fsai.preprocessing.DESCARTADAS_100_NAN`) y consta en `resultado_*.json` (`features_descartadas_100pct_nan`) y en `fsai.modelos.hiperparametros._features_descartadas_100pct_nan`.

## D9. Poblacion de entrenamiento
- Se entrena y se carga en la BD solo con los hogares con FIES completo (32 065 de 33 702). La exclusion no es aleatoria por area/region/departamento (seccion 1 del reporte).
