# Resultados del entrenamiento REAL (ENAHO 2025, FIES real) - FoodSecurity AI

Documento generado por `generar_reporte_real.py` a partir de artefactos y de PostgreSQL. **Solo valores observados.** Las asociaciones son predictivas, no causales. Los IC de este documento ignoran el diseno muestral (no hay estrato/conglomerado en la BD) y por tanto son aproximados.

## 0. Advertencias que condicionan toda lectura

- **Umbrales FIES [POR VALIDAR]**: leve >=1, moderada >=4, severa >=7 sobre el puntaje crudo (0-8). No es la clasificacion oficial del INEI/FAO (Rasch). Todas las clases y la prevalencia dependen de ese corte; no se hizo analisis de sensibilidad de umbrales.
- **Circularidad**: `fies_nivel` se deriva de `fies_q1..q8`. Los escenarios `fies` e `hibrido` son RECONSTRUCCION del objetivo (no prediccion). Solo `socio` (sin variables FIES) es prediccion a partir de condiciones del hogar.
- **`informalidad` y `fies_rasch_logit` son 100 % NaN** (no se inventaron): `informalidad` se excluye de los predictores con aviso y registro; el Rasch no se uso.
- **Un solo ano (2025)**: la validacion temporal V4 es imposible. El FIES refiere al mes previo a la entrevista (estacionalidad).
- **Diseno muestral ausente**: la BD no guarda conglomerado/estrato; los IC bootstrap ignoran el diseno. `asistencia_escolar` es NaN en el 61 % de los hogares (sin ninos en edad escolar) y se imputa por mediana dentro del pipeline.
- **Uso etico**: los modelos describen asociaciones en datos de 2025 y no deben usarse para asignar o denegar beneficios a hogares individuales ni para inferencias causales.

## 1. Dataset cargado en PostgreSQL

| dataset_id | nombre | fuente | anio | version | n_filas | hash_archivo |
|---|---|---|---|---|---|---|
| 3 | ENAHO_2025_REAL | INEI ENAHO 2025 anual, módulo 2082 FIES | 2025 | enaho2025-fies-v1; parquet 33702 filas, 32065 con FIES completo | 32065 | 96bf57e1fd67c283... |

| hogares | respuestas_fies | indicadores | condiciones | personas_anonimizadas |
|---|---|---|---|---|
| 32065 | 32065 | 32065 | 32065 | 0 |

`personas_anonimizadas` no se cargo (los microdatos de personas del modulo 200 no se procesaron a esa tabla; no se invento nada). El dataset demo sigue en la base y no se toco.

Prevalencia de inseguridad alimentaria moderada o severa (ia_binaria), BD vs parquet (hogares, umbrales [POR VALIDAR]):

| area | prevalencia_ia_ponderada | prevalencia_sin_ponderar | n | prevalencia_parquet_ponderada | prevalencia_parquet_sin_ponderar | coincide |
|---|---|---|---|---|---|---|
| R | 0.3051 | 0.2949 | 11291 | 0.3051 | 0.2949 | True |
| Total | 0.2684 | 0.2544 | 32065 | 0.2684 | 0.2544 | True |
| U | 0.2585 | 0.2323 | 20774 | 0.2585 | 0.2323 | True |

SHA-256 del parquet: `96bf57e1fd67c28380a16ad7435db7ffb5efd3ebae910a765939e2dd608c23b9`; `salt_hogar_uid_propia` = True.

### Exclusion de hogares sin FIES completo (n = 1637 de 33702; 4.9 % sin ponderar, 7.3 % ponderado)

Los hogares con `fies_nivel` NaN se excluyen del entrenamiento y de la BD (no se imputan). Patron de exclusion:

| variable | chi2 | gl | p | V de Cramer |
|---|---|---|---|---|
| area | 167.2 | 1 | 2.99e-38 | 0.07 |
| region_natural | 382.4 | 2 | 9.18e-84 | 0.107 |
| departamento | 1624.7 | 24 | 0.00e+00 | 0.22 |
| mes | 16.1 | 11 | 1.37e-01 | 0.022 |

% excluido por area: R 2.8 %, U 6.0 %; por region: costa 7.5 %, selva 2.6 %, sierra 3.0 %.

**Lectura**: la exclusion NO es aleatoria respecto del area, la region ni el departamento (p < 0,001; efecto pequeno a moderado), con mas exclusion en urbano y costa. No hay asociacion detectable con el mes. Las metricas describen la poblacion con FIES completo; no se corrigio por seleccion (sin reponderacion), por lo que podria haber sesgo de seleccion que no se cuantifico.

## 2. Diseno, presupuesto de ensayos y tiempos

Particion agrupada por `hogar_uid` y estratificada por fies_nivel x area: Desarrollo 22447, Validacion 4809, Prueba 4809 (huella `f1815abeac5f7f30...`; cruces de hogar_uid entre particiones = 0). CV anidada 5 externos x 3 internos solo en Desarrollo; umbral de severa (especificidad >= 0,70) y calibracion isotonica con Validacion; la Prueba se evalua una sola vez por modelo (candado exclusivo).

| Escenario | Modelo | ensayos por pliegue externo | ensayos busqueda final | ensayos totales | segundos totales | s/pliegue externo (media) | predictores descartados 100 % NaN |
|---|---|---|---|---|---|---|---|
| fies | M1 | 5 | 5 | 30 | 30.0 | 3.3 | - |
| fies | M2 | 5 | 5 | 30 | 235.9 | 38.1 | - |
| fies | M3 | 5 | 5 | 30 | 687.9 | 118.7 | - |
| fies | M4 | 5 | 5 | 30 | 1837.4 | 314.8 | - |
| fies | M5 | 5 | 5 | 30 | 1880.1 | 280.3 | - |
| hibrido | M1 | 5 | 5 | 30 | 63.8 | 8.8 | informalidad |
| hibrido | M2 | 5 | 5 | 30 | 364.3 | 54.9 | informalidad |
| hibrido | M3 | 5 | 5 | 30 | 1496.3 | 251.9 | informalidad |
| hibrido | M4 | 5 | 5 | 30 | 2213.2 | 369.1 | informalidad |
| hibrido | M5 | 5 | 5 | 30 | 2123.8 | 338.2 | informalidad |
| socio | M1 | 20 | 20 | 120 | 235.3 | 35.6 | informalidad |
| socio | M2 | 20 | 20 | 120 | 1859.9 | 297.5 | informalidad |
| socio | M3 | 20 | 20 | 120 | 7589.2 | 1293.4 | informalidad |
| socio | M4 | 20 | 20 | 120 | 8596.0 | 1421.0 | informalidad |
| socio | M5 | 20 | 20 | 120 | 10279.6 | 1698.5 | informalidad |

El presupuesto de ensayos es IGUAL para los cinco modelos dentro de cada escenario. Es una **desviacion declarada** respecto de los 100 ensayos del protocolo (ver `DESVIACIONES.md`).

## 3. Escenario primario `socio` (PREDICCION real, sin variables FIES)

### Prueba (15 %, evaluada una sola vez) - metricas PONDERADAS con factor_expansion, valor [IC95 % bootstrap estratificado por clase]

| Modelo | F1-macro | Exactitud balanceada | AUC-ROC (OvR) | AUC-PR macro | MCC | Sensibilidad severa | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| M1 Regresion logistica | 0.357 [0.341; 0.377] | 0.408 [0.387; 0.432] | 0.675 [0.659; 0.692] | 0.391 [0.378; 0.412] | 0.194 [0.172; 0.220] | 0.563 [0.492; 0.633] | 0.688 [0.678; 0.697] | 0.025 [0.016; 0.047] |
| M2 Random Forest | 0.375 [0.356; 0.395] | 0.374 [0.356; 0.393] | 0.677 [0.660; 0.693] | 0.391 [0.376; 0.409] | 0.204 [0.178; 0.230] | 0.146 [0.106; 0.194] | 0.642 [0.633; 0.652] | 0.045 [0.030; 0.065] |
| M3 CatBoost | 0.375 [0.355; 0.393] | 0.411 [0.387; 0.435] | 0.681 [0.665; 0.696] | 0.397 [0.381; 0.418] | 0.201 [0.176; 0.226] | 0.453 [0.378; 0.529] | 0.672 [0.662; 0.683] | 0.024 [0.018; 0.048] |
| M4 Stacking | 0.360 [0.343; 0.378] | 0.410 [0.386; 0.434] | 0.680 [0.666; 0.696] | 0.396 [0.382; 0.416] | 0.196 [0.171; 0.219] | 0.554 [0.480; 0.623] | 0.690 [0.680; 0.701] | 0.028 [0.020; 0.049] |
| M5 Autoencoder+XGBoost | 0.353 [0.332; 0.372] | 0.353 [0.337; 0.369] | 0.663 [0.646; 0.678] | 0.383 [0.368; 0.402] | 0.189 [0.162; 0.215] | 0.106 [0.068; 0.150] | 0.644 [0.630; 0.658] | 0.064 [0.049; 0.082] |

**Lectura automatica (solo describe las tablas)**: el F1-macro ponderado en la Prueba va de 0.353 a 0.375 y el AUC-ROC (OvR) de 0.663 a 0.681 entre los cinco modelos. El mayor F1-macro ponderado es M2 (0.375); los IC95 % de M1, M3, M4, M5 se solapan con el suyo, asi que esa diferencia no es concluyente. Con 4 clases y la prevalencia observada, un F1-macro cercano a 0,37 y un AUC cercano a 0,68 indican capacidad predictiva modesta a partir solo de condiciones socioeconomicas, y NO debe leerse como un instrumento de focalizacion individual.

### Prueba - metricas NO ponderadas

| Modelo | F1-macro | Exactitud balanceada | AUC-ROC (OvR) | AUC-PR macro | MCC | Sensibilidad severa | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| M1 Regresion logistica | 0.374 [0.361; 0.388] | 0.429 [0.410; 0.445] | 0.697 [0.686; 0.707] | 0.406 [0.394; 0.423] | 0.216 [0.198; 0.233] | 0.595 [0.538; 0.646] | 0.670 [0.664; 0.678] | 0.024 [0.019; 0.040] |
| M2 Random Forest | 0.403 [0.385; 0.417] | 0.399 [0.382; 0.413] | 0.698 [0.686; 0.709] | 0.409 [0.396; 0.424] | 0.229 [0.209; 0.249] | 0.207 [0.162; 0.252] | 0.626 [0.619; 0.633] | 0.049 [0.038; 0.062] |
| M3 CatBoost | 0.397 [0.383; 0.411] | 0.433 [0.415; 0.451] | 0.700 [0.689; 0.711] | 0.410 [0.397; 0.425] | 0.230 [0.211; 0.249] | 0.474 [0.420; 0.529] | 0.654 [0.646; 0.661] | 0.024 [0.017; 0.039] |
| M4 Stacking | 0.383 [0.371; 0.396] | 0.433 [0.415; 0.450] | 0.702 [0.691; 0.713] | 0.411 [0.399; 0.427] | 0.223 [0.206; 0.241] | 0.571 [0.513; 0.625] | 0.669 [0.661; 0.677] | 0.012 [0.010; 0.028] |
| M5 Autoencoder+XGBoost | 0.362 [0.345; 0.379] | 0.360 [0.346; 0.374] | 0.685 [0.674; 0.696] | 0.394 [0.383; 0.408] | 0.204 [0.184; 0.224] | 0.117 [0.081; 0.153] | 0.625 [0.615; 0.633] | 0.060 [0.048; 0.073] |

### Validacion cruzada anidada (solo Desarrollo, n=5 pliegues externos) y Validacion

| Modelo | F1-macro CV anidada (media pliegues ext., IC t95 %) | AUC-ROC CV anidada | F1-macro Validacion (pond.) | umbral severa (Validacion) | sens./espec. severa en Validacion con umbral |
|---|---|---|---|---|---|
| M1 | 0.353 [0.340; 0.367] | 0.668 [0.659; 0.676] | 0.347 | 0.2811 | 0.551 / 0.701 |
| M2 | 0.378 [0.362; 0.394] | 0.672 [0.666; 0.678] | 0.369 | 0.1523 | 0.610 / 0.700 |
| M3 | 0.375 [0.362; 0.388] | 0.671 [0.664; 0.678] | 0.357 | 0.2416 | 0.571 / 0.700 |
| M4 | 0.361 [0.343; 0.378] | 0.674 [0.666; 0.683] | 0.355 | 0.2804 | 0.605 / 0.701 |
| M5 | 0.341 [0.333; 0.350] | 0.669 [0.658; 0.679] | 0.335 | 0.0657 | 0.565 / 0.700 |

### Clase severa en la Prueba (ponderado) y calibracion

| Modelo | sens. severa (argmax) | espec. (argmax) | VPP (argmax) | sens. (umbral Val.) | espec. (umbral Val.) | VPP (umbral Val.) | Brier / ECE sin calibrar | Brier / ECE calibrado (isotonica, Val.) |
|---|---|---|---|---|---|---|---|---|
| M1 | 0.563 | 0.748 | 0.150 | 0.631 | 0.707 | 0.145 | 0.688 / 0.025 | 0.631 / 0.035 |
| M2 | 0.146 | 0.957 | 0.212 | 0.633 | 0.708 | 0.146 | 0.642 / 0.045 | 0.631 / 0.015 |
| M3 | 0.453 | 0.839 | 0.182 | 0.625 | 0.709 | 0.145 | 0.672 / 0.024 | 0.628 / 0.021 |
| M4 | 0.554 | 0.761 | 0.155 | 0.620 | 0.706 | 0.143 | 0.690 / 0.028 | 0.627 / 0.037 |
| M5 | 0.106 | 0.983 | 0.330 | 0.592 | 0.700 | 0.135 | 0.644 / 0.064 | 0.634 / 0.016 |

### Pruebas estadisticas entre modelos (Holm dentro de cada familia)

| prueba | comparacion | estadistico | p_valor | p_ajustado | detalle |
|---|---|---|---|---|---|
| delong_severa | M1-M2 | 1.568 | 0.1169 | 0.3297 | AUC M1=0.764; M2=0.749 |
| delong_severa | M1-M3 | 0.41 | 0.6818 | 0.6818 | AUC M1=0.764; M3=0.761 |
| delong_severa | M1-M4 | -1.919 | 0.055 | 0.275 | AUC M1=0.764; M4=0.774 |
| delong_severa | M1-M5 | 2.884 | 0.003921 | 0.02745 | AUC M1=0.764; M5=0.733 |
| delong_severa | M2-M3 | -1.878 | 0.06041 | 0.275 | AUC M2=0.749; M3=0.761 |
| delong_severa | M2-M4 | -3.359 | 0.0007811 | 0.00703 | AUC M2=0.749; M4=0.774 |
| delong_severa | M2-M5 | 1.599 | 0.1099 | 0.3297 | AUC M2=0.749; M5=0.733 |
| delong_severa | M3-M4 | -2.321 | 0.0203 | 0.1218 | AUC M3=0.761; M4=0.774 |
| delong_severa | M3-M5 | 3.059 | 0.002224 | 0.01779 | AUC M3=0.761; M5=0.733 |
| delong_severa | M4-M5 | 4.248 | 2.153e-05 | 0.0002153 | AUC M4=0.774; M5=0.733 |
| mcnemar | M1-M2 | 38.56 | 5.311e-10 | 3.718e-09 | b=193.7; c=337.8 |
| mcnemar | M1-M3 | 5.709 | 0.01688 | 0.05064 | b=160.2; c=206.9 |
| mcnemar | M1-M4 | 0.1388 | 0.7095 | 0.7691 | b=132.8; c=125.8 |
| mcnemar | M1-M5 | 41.28 | 1.321e-10 | 1.057e-09 | b=241.4; c=405.8 |
| mcnemar | M2-M3 | 24.19 | 8.739e-07 | 5.244e-06 | b=240.7; c=143.3 |
| mcnemar | M2-M4 | 42.74 | 6.242e-11 | 6.242e-10 | b=339.3; c=188.1 |
| mcnemar | M2-M5 | 0.7562 | 0.3845 | 0.7691 | b=235.8; c=256.1 |
| mcnemar | M3-M4 | 10.06 | 0.001516 | 0.006064 | b=165.3; c=111.5 |
| mcnemar | M3-M5 | 22.08 | 2.622e-06 | 1.311e-05 | b=249.4; c=367.1 |
| mcnemar | M4-M5 | 42.26 | 8.001e-11 | 7.201e-10 | b=258.0; c=429.4 |
| wilcoxon_pliegues | M1-M2 | 0 | 0.0625 | 0.625 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M1-M3 | 0 | 0.0625 | 0.625 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M1-M4 | 0 | 0.0625 | 0.625 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M1-M5 | 2 | 0.1875 | 0.625 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M2-M3 | 6 | 0.8125 | 0.8125 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M2-M4 | 0 | 0.0625 | 0.625 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M2-M5 | 0 | 0.0625 | 0.625 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M3-M4 | 0 | 0.0625 | 0.625 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M3-M5 | 0 | 0.0625 | 0.625 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M4-M5 | 1 | 0.125 | 0.625 | n=5 pliegues; p mínimo posible=0.0625 |
| friedman | global | 17.12 | 0.001832 | 0.001832 | rangos medios={'M1': 4.2, 'M2': 1.6, 'M3': 1.4, 'M4': 3.2, 'M5': 4.6} |
| nemenyi | M1-M2 |  | 0.07032 | 0.07032 | nan |
| nemenyi | M1-M3 |  | 0.04088 | 0.04088 | nan |
| nemenyi | M1-M4 |  | 0.8555 | 0.8555 | nan |
| nemenyi | M1-M5 |  | 0.9946 | 0.9946 | nan |
| nemenyi | M2-M3 |  | 0.9996 | 0.9996 | nan |
| nemenyi | M2-M4 |  | 0.4973 | 0.4973 | nan |
| nemenyi | M2-M5 |  | 0.02266 | 0.02266 | nan |
| nemenyi | M3-M4 |  | 0.3735 | 0.3735 | nan |
| nemenyi | M3-M5 |  | 0.01199 | 0.01199 | nan |
| nemenyi | M4-M5 |  | 0.6277 | 0.6277 | nan |

Wilcoxon sobre 5 pliegues externos tiene un p minimo posible de 0,0625 (bilateral exacto): no puede ser significativo a 0,05. DeLong y McNemar usan las predicciones de la Prueba (n pareado).

### Explicabilidad (SHAP global, media de |SHAP| sobre una submuestra de la Prueba; permutacion de F1-macro)

- **M1** (LinearSHAP, n=500): nivel_educ_jefe 0.109; ingreso_pc 0.106; gasto_alim_pc 0.093; n_ocupados 0.063; region_natural_sierra 0.055; n_miembros 0.054; hacinamiento 0.054; region_natural_costa 0.049
- **M2** (TreeSHAP, n=500): ingreso_pc 0.031; gasto_pc 0.020; nivel_educ_jefe 0.016; gasto_alim_pc 0.012; region_natural_sierra 0.010; share_alimentario 0.008; n_miembros 0.008; pobreza_monetaria 0.008
- **M3** (TreeSHAP, n=500): ingreso_pc 0.159; nivel_educ_jefe 0.098; region_natural_sierra 0.075; gasto_pc 0.075; gasto_alim_pc 0.064; n_miembros 0.063; hacinamiento 0.063; n_ocupados 0.052
- **M4** (KernelSHAP, n=100): ingreso_pc 0.044; nivel_educ_jefe 0.021; gasto_alim_pc 0.016; n_miembros 0.013; region_natural_costa 0.010; hacinamiento 0.009; n_ocupados 0.008; departamento_loreto 0.008
- **M5** (TreeSHAP, n=500): ingreso_pc 0.203; nivel_educ_jefe 0.092; gasto_pc 0.088; gasto_alim_pc 0.084; z_14 0.077; n_miembros 0.073; z_0 0.062; hacinamiento 0.060

- Permutacion **M1**: departamento 0.0269; ingreso_pc 0.0152; dependencia_economica 0.0112; region_natural 0.0101; desempleo 0.0065
- Permutacion **M2**: departamento 0.0200; region_natural 0.0145; ingreso_pc 0.0120; nivel_educ_jefe 0.0061; n_adultos_mayores 0.0041
- Permutacion **M3**: ingreso_pc 0.0381; departamento 0.0203; region_natural 0.0082; sexo_jefe 0.0065; nivel_educ_jefe 0.0044
- Permutacion **M4**: ingreso_pc 0.0324; departamento 0.0273; n_ocupados 0.0086; n_miembros 0.0085; region_natural 0.0085
- Permutacion **M5**: departamento 0.0399; ingreso_pc 0.0394; region_natural 0.0206; nivel_educ_jefe 0.0190; gasto_alim_pc 0.0189

### Equidad (Prueba): brechas de sensibilidad (TPR) vs. grupo de referencia; alerta si |brecha| > 5 pp

| codigo | objetivo | filas grupo | con alerta |
|---|---|---|---|
| M1 | ia_binaria | 19 | 6 |
| M1 | severa | 19 | 8 |
| M2 | ia_binaria | 19 | 10 |
| M2 | severa | 19 | 8 |
| M3 | ia_binaria | 19 | 9 |
| M3 | severa | 19 | 7 |
| M4 | ia_binaria | 19 | 7 |
| M4 | severa | 19 | 9 |
| M5 | ia_binaria | 19 | 8 |
| M5 | severa | 19 | 7 |

Grupos con alerta para M2 (objetivo IA moderada o severa; TPR = sensibilidad ponderada):

| grupo_col | grupo | n | tpr | brecha_tpr | ic_inf | ic_sup |
|---|---|---|---|---|---|---|
| sexo_jefe | M | 1821 | 0.511 | 0.16 | 0.08 | 0.243 |
| grupo_edad_jefe | 45-59 | 1619 | 0.391 | -0.062 | -0.157 | 0.034 |
| grupo_edad_jefe | <30 | 263 | 0.291 | -0.162 | -0.33 | 0.033 |
| grupo_educ_jefe | primaria | 1528 | 0.516 | 0.11 | 0.016 | 0.197 |
| grupo_educ_jefe | sin_primaria | 297 | 0.519 | 0.113 | -0.019 | 0.242 |
| grupo_educ_jefe | superior | 1170 | 0.226 | -0.18 | -0.292 | -0.063 |
| area | R | 1693 | 0.475 | 0.061 | -0.007 | 0.131 |
| region_natural | selva | 1071 | 0.531 | 0.094 | 0.008 | 0.188 |
| region_natural | sierra | 1732 | 0.371 | -0.066 | -0.141 | 0.019 |
| pobreza_monetaria | 1 | 892 | 0.712 | 0.398 | 0.322 | 0.473 |

Tabla completa (grupos, n, TPR, FPR, brecha e IC) en `artifacts\real\fases\cv_anidada_real\analisis\equidad.csv`. Una alerta indica diferencia de desempeno entre grupos, no su causa.

## Escenario `fies` (RECONSTRUCCION circular - no es prediccion)

> Estas cifras miden cuan bien se reconstruye el nivel FIES a partir de los propios items; no indican capacidad predictiva.

### Prueba (15 %, evaluada una sola vez) - metricas PONDERADAS con factor_expansion, valor [IC95 % bootstrap estratificado por clase]

| Modelo | F1-macro | Exactitud balanceada | AUC-ROC (OvR) | AUC-PR macro | MCC | Sensibilidad severa | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| M1 Regresion logistica | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.000 [0.000; 0.000] | 0.000 [0.000; 0.000] |
| M2 Random Forest | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.000 [0.000; 0.000] | 0.000 [0.000; 0.000] |
| M3 CatBoost | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.000 [0.000; 0.000] | 0.000 [0.000; 0.000] |
| M4 Stacking | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.000 [0.000; 0.000] | 0.000 [0.000; 0.000] |
| M5 Autoencoder+XGBoost | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.000 [0.000; 0.000] | 0.000 [0.000; 0.000] |

### Prueba - metricas NO ponderadas

| Modelo | F1-macro | Exactitud balanceada | AUC-ROC (OvR) | AUC-PR macro | MCC | Sensibilidad severa | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| M1 Regresion logistica | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.000 [0.000; 0.000] | 0.000 [0.000; 0.000] |
| M2 Random Forest | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.000 [0.000; 0.000] | 0.000 [0.000; 0.000] |
| M3 CatBoost | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.000 [0.000; 0.000] | 0.000 [0.000; 0.000] |
| M4 Stacking | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.000 [0.000; 0.000] | 0.000 [0.000; 0.000] |
| M5 Autoencoder+XGBoost | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.000 [0.000; 0.000] | 0.000 [0.000; 0.000] |

### Validacion cruzada anidada (solo Desarrollo, n=5 pliegues externos) y Validacion

| Modelo | F1-macro CV anidada (media pliegues ext., IC t95 %) | AUC-ROC CV anidada | F1-macro Validacion (pond.) | umbral severa (Validacion) | sens./espec. severa en Validacion con umbral |
|---|---|---|---|---|---|
| M1 | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 | 0.0000 | 1.000 / 0.769 |
| M2 | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 | 0.0017 | 1.000 / 0.999 |
| M3 | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 | 0.0001 | 1.000 / 0.707 |
| M4 | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 | 0.0001 | 1.000 / 0.898 |
| M5 | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 | 0.0000 | 1.000 / 0.705 |

### Clase severa en la Prueba (ponderado) y calibracion

| Modelo | sens. severa (argmax) | espec. (argmax) | VPP (argmax) | sens. (umbral Val.) | espec. (umbral Val.) | VPP (umbral Val.) | Brier / ECE sin calibrar | Brier / ECE calibrado (isotonica, Val.) |
|---|---|---|---|---|---|---|---|---|
| M1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.759 | 0.247 | 0.000 / 0.000 | 0.000 / 0.000 |
| M2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 | 0.980 | 0.000 / 0.000 | 0.000 / 0.000 |
| M3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.707 | 0.212 | 0.000 / 0.000 | 0.000 / 0.000 |
| M4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.883 | 0.403 | 0.000 / 0.000 | 0.000 / 0.000 |
| M5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.687 | 0.201 | 0.000 / 0.000 | 0.000 / 0.000 |

### Pruebas estadisticas entre modelos (Holm dentro de cada familia)

| prueba | comparacion | estadistico | p_valor | p_ajustado | detalle |
|---|---|---|---|---|---|
| delong_severa | M1-M2 | 0 | 1 | 1 | AUC M1=1.000; M2=1.000 |
| delong_severa | M1-M3 | 0 | 1 | 1 | AUC M1=1.000; M3=1.000 |
| delong_severa | M1-M4 | 0 | 1 | 1 | AUC M1=1.000; M4=1.000 |
| delong_severa | M1-M5 | 0 | 1 | 1 | AUC M1=1.000; M5=1.000 |
| delong_severa | M2-M3 | 0 | 1 | 1 | AUC M2=1.000; M3=1.000 |
| delong_severa | M2-M4 | 0 | 1 | 1 | AUC M2=1.000; M4=1.000 |
| delong_severa | M2-M5 | 0 | 1 | 1 | AUC M2=1.000; M5=1.000 |
| delong_severa | M3-M4 | 0 | 1 | 1 | AUC M3=1.000; M4=1.000 |
| delong_severa | M3-M5 | 0 | 1 | 1 | AUC M3=1.000; M5=1.000 |
| delong_severa | M4-M5 | 0 | 1 | 1 | AUC M4=1.000; M5=1.000 |
| mcnemar | M1-M2 | 0 | 1 | 1 | b=0.0; c=0.0 |
| mcnemar | M1-M3 | 0 | 1 | 1 | b=0.0; c=0.0 |
| mcnemar | M1-M4 | 0 | 1 | 1 | b=0.0; c=0.0 |
| mcnemar | M1-M5 | 0 | 1 | 1 | b=0.0; c=0.0 |
| mcnemar | M2-M3 | 0 | 1 | 1 | b=0.0; c=0.0 |
| mcnemar | M2-M4 | 0 | 1 | 1 | b=0.0; c=0.0 |
| mcnemar | M2-M5 | 0 | 1 | 1 | b=0.0; c=0.0 |
| mcnemar | M3-M4 | 0 | 1 | 1 | b=0.0; c=0.0 |
| mcnemar | M3-M5 | 0 | 1 | 1 | b=0.0; c=0.0 |
| mcnemar | M4-M5 | 0 | 1 | 1 | b=0.0; c=0.0 |
| wilcoxon_pliegues | M1-M2 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M1-M3 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M1-M4 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M1-M5 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M2-M3 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M2-M4 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M2-M5 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M3-M4 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M3-M5 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M4-M5 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| friedman | global |  |  |  | rangos medios={'M1': 3.0, 'M2': 3.0, 'M3': 3.0, 'M4': 3.0, 'M5': 3.0} |
| nemenyi | M1-M2 |  | 1 | 1 | nan |
| nemenyi | M1-M3 |  | 1 | 1 | nan |
| nemenyi | M1-M4 |  | 1 | 1 | nan |
| nemenyi | M1-M5 |  | 1 | 1 | nan |
| nemenyi | M2-M3 |  | 1 | 1 | nan |
| nemenyi | M2-M4 |  | 1 | 1 | nan |
| nemenyi | M2-M5 |  | 1 | 1 | nan |
| nemenyi | M3-M4 |  | 1 | 1 | nan |
| nemenyi | M3-M5 |  | 1 | 1 | nan |
| nemenyi | M4-M5 |  | 1 | 1 | nan |

Wilcoxon sobre 5 pliegues externos tiene un p minimo posible de 0,0625 (bilateral exacto): no puede ser significativo a 0,05. DeLong y McNemar usan las predicciones de la Prueba (n pareado).

## Escenario `hibrido` (RECONSTRUCCION circular - no es prediccion)

> Estas cifras miden cuan bien se reconstruye el nivel FIES a partir de los propios items; no indican capacidad predictiva.

### Prueba (15 %, evaluada una sola vez) - metricas PONDERADAS con factor_expansion, valor [IC95 % bootstrap estratificado por clase]

| Modelo | F1-macro | Exactitud balanceada | AUC-ROC (OvR) | AUC-PR macro | MCC | Sensibilidad severa | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| M1 Regresion logistica | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.000 [0.000; 0.000] | 0.001 [0.001; 0.001] |
| M2 Random Forest | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.008 [0.007; 0.008] | 0.045 [0.043; 0.046] |
| M3 CatBoost | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.000 [0.000; 0.000] | 0.000 [0.000; 0.001] |
| M4 Stacking | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.000 [0.000; 0.000] | 0.000 [0.000; 0.000] |
| M5 Autoencoder+XGBoost | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.000 [0.000; 0.000] | 0.000 [0.000; 0.000] |

### Prueba - metricas NO ponderadas

| Modelo | F1-macro | Exactitud balanceada | AUC-ROC (OvR) | AUC-PR macro | MCC | Sensibilidad severa | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| M1 Regresion logistica | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.000 [0.000; 0.000] | 0.001 [0.001; 0.001] |
| M2 Random Forest | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.007 [0.007; 0.007] | 0.043 [0.042; 0.044] |
| M3 CatBoost | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.000 [0.000; 0.000] | 0.001 [0.000; 0.001] |
| M4 Stacking | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.000 [0.000; 0.000] | 0.000 [0.000; 0.000] |
| M5 Autoencoder+XGBoost | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 0.000 [0.000; 0.000] | 0.000 [0.000; 0.000] |

### Validacion cruzada anidada (solo Desarrollo, n=5 pliegues externos) y Validacion

| Modelo | F1-macro CV anidada (media pliegues ext., IC t95 %) | AUC-ROC CV anidada | F1-macro Validacion (pond.) | umbral severa (Validacion) | sens./espec. severa en Validacion con umbral |
|---|---|---|---|---|---|
| M1 | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 | 0.0000 | 1.000 / 0.700 |
| M2 | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 | 0.0000 | 1.000 / 0.700 |
| M3 | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 | 0.0001 | 1.000 / 0.700 |
| M4 | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 | 0.0001 | 1.000 / 0.700 |
| M5 | 1.000 [1.000; 1.000] | 1.000 [1.000; 1.000] | 1.000 | 0.0000 | 1.000 / 0.700 |

### Clase severa en la Prueba (ponderado) y calibracion

| Modelo | sens. severa (argmax) | espec. (argmax) | VPP (argmax) | sens. (umbral Val.) | espec. (umbral Val.) | VPP (umbral Val.) | Brier / ECE sin calibrar | Brier / ECE calibrado (isotonica, Val.) |
|---|---|---|---|---|---|---|---|---|
| M1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.690 | 0.203 | 0.000 / 0.001 | 0.000 / 0.000 |
| M2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.686 | 0.201 | 0.008 / 0.045 | 0.000 / 0.000 |
| M3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.698 | 0.207 | 0.000 / 0.000 | 0.000 / 0.000 |
| M4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.675 | 0.195 | 0.000 / 0.000 | 0.000 / 0.000 |
| M5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.686 | 0.201 | 0.000 / 0.000 | 0.000 / 0.000 |

### Pruebas estadisticas entre modelos (Holm dentro de cada familia)

| prueba | comparacion | estadistico | p_valor | p_ajustado | detalle |
|---|---|---|---|---|---|
| delong_severa | M1-M2 | 0 | 1 | 1 | AUC M1=1.000; M2=1.000 |
| delong_severa | M1-M3 | 0 | 1 | 1 | AUC M1=1.000; M3=1.000 |
| delong_severa | M1-M4 | 0 | 1 | 1 | AUC M1=1.000; M4=1.000 |
| delong_severa | M1-M5 | 0 | 1 | 1 | AUC M1=1.000; M5=1.000 |
| delong_severa | M2-M3 | 0 | 1 | 1 | AUC M2=1.000; M3=1.000 |
| delong_severa | M2-M4 | 0 | 1 | 1 | AUC M2=1.000; M4=1.000 |
| delong_severa | M2-M5 | 0 | 1 | 1 | AUC M2=1.000; M5=1.000 |
| delong_severa | M3-M4 | 0 | 1 | 1 | AUC M3=1.000; M4=1.000 |
| delong_severa | M3-M5 | 0 | 1 | 1 | AUC M3=1.000; M5=1.000 |
| delong_severa | M4-M5 | 0 | 1 | 1 | AUC M4=1.000; M5=1.000 |
| mcnemar | M1-M2 | 0 | 1 | 1 | b=0.0; c=0.0 |
| mcnemar | M1-M3 | 0 | 1 | 1 | b=0.0; c=0.0 |
| mcnemar | M1-M4 | 0 | 1 | 1 | b=0.0; c=0.0 |
| mcnemar | M1-M5 | 0 | 1 | 1 | b=0.0; c=0.0 |
| mcnemar | M2-M3 | 0 | 1 | 1 | b=0.0; c=0.0 |
| mcnemar | M2-M4 | 0 | 1 | 1 | b=0.0; c=0.0 |
| mcnemar | M2-M5 | 0 | 1 | 1 | b=0.0; c=0.0 |
| mcnemar | M3-M4 | 0 | 1 | 1 | b=0.0; c=0.0 |
| mcnemar | M3-M5 | 0 | 1 | 1 | b=0.0; c=0.0 |
| mcnemar | M4-M5 | 0 | 1 | 1 | b=0.0; c=0.0 |
| wilcoxon_pliegues | M1-M2 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M1-M3 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M1-M4 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M1-M5 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M2-M3 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M2-M4 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M2-M5 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M3-M4 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M3-M5 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| wilcoxon_pliegues | M4-M5 | 0 | 1 | 1 | n=5 pliegues; p mínimo posible=0.0625 |
| friedman | global |  |  |  | rangos medios={'M1': 3.0, 'M2': 3.0, 'M3': 3.0, 'M4': 3.0, 'M5': 3.0} |
| nemenyi | M1-M2 |  | 1 | 1 | nan |
| nemenyi | M1-M3 |  | 1 | 1 | nan |
| nemenyi | M1-M4 |  | 1 | 1 | nan |
| nemenyi | M1-M5 |  | 1 | 1 | nan |
| nemenyi | M2-M3 |  | 1 | 1 | nan |
| nemenyi | M2-M4 |  | 1 | 1 | nan |
| nemenyi | M2-M5 |  | 1 | 1 | nan |
| nemenyi | M3-M4 |  | 1 | 1 | nan |
| nemenyi | M3-M5 |  | 1 | 1 | nan |
| nemenyi | M4-M5 |  | 1 | 1 | nan |

Wilcoxon sobre 5 pliegues externos tiene un p minimo posible de 0,0625 (bilateral exacto): no puede ser significativo a 0,05. DeLong y McNemar usan las predicciones de la Prueba (n pareado).

## 4. Flujo clasico (validacion V1/V3/V2 sobre todos los hogares, sin Prueba bloqueada)

| experimento_id | nombre | esquema_validacion | estado |
|---|---|---|---|
| 8 | ENAHO_2025_REAL_clasico_socio_V1 | V1 | completado |
| 9 | ENAHO_2025_REAL_clasico_socio_V2_macro | V2 | completado |
| 10 | ENAHO_2025_REAL_clasico_socio_V3 | V3 | completado |

Metricas agregadas fuera de muestra (prediccion OOF de referencia: cada hogar predicho por un modelo que no lo vio; IC95 % bootstrap estratificado por clase, 200 replicas). Optuna con **3 ensayos por bloque** (D4): son evidencia de sensibilidad al esquema de validacion, no una busqueda comparable con la CV anidada primaria.

| esquema | codigo | f1_macro (no pond.) | f1_macro (pond.) | recall_severa (no pond.) | recall_severa (pond.) | roc_auc_ovr (no pond.) | roc_auc_ovr (pond.) |
|---|---|---|---|---|---|---|---|
| V1 | M1 | 0.376 [0.372; 0.380] | 0.356 [0.350; 0.363] | 0.540 [0.519; 0.563] | 0.517 [0.484; 0.546] | 0.691 | 0.669 |
| V1 | M2 | 0.400 [0.395; 0.406] | 0.382 [0.376; 0.391] | 0.333 [0.318; 0.350] | 0.281 [0.258; 0.305] | 0.694 | 0.675 |
| V1 | M3 | 0.390 [0.384; 0.395] | 0.370 [0.364; 0.378] | 0.443 [0.423; 0.463] | 0.406 [0.377; 0.431] | 0.692 | 0.669 |
| V1 | M4 | 0.380 [0.375; 0.385] | 0.359 [0.353; 0.367] | 0.536 [0.515; 0.555] | 0.515 [0.485; 0.539] | 0.696 | 0.676 |
| V1 | M5 | 0.306 [0.302; 0.310] | 0.297 [0.292; 0.302] | 0.025 [0.020; 0.031] | 0.016 [0.012; 0.021] | 0.636 | 0.621 |
| V2_macro | M1 | 0.341 [0.335; 0.346] | 0.328 [0.321; 0.334] | 0.354 [0.331; 0.375] | 0.359 [0.331; 0.389] | 0.653 | 0.640 |
| V2_macro | M2 | 0.351 [0.345; 0.357] | 0.341 [0.333; 0.347] | 0.192 [0.174; 0.208] | 0.188 [0.166; 0.207] | 0.649 | 0.636 |
| V2_macro | M3 | 0.339 [0.334; 0.344] | 0.326 [0.319; 0.332] | 0.312 [0.293; 0.330] | 0.304 [0.279; 0.330] | 0.649 | 0.643 |
| V2_macro | M4 | 0.345 [0.340; 0.350] | 0.329 [0.323; 0.336] | 0.425 [0.404; 0.447] | 0.403 [0.371; 0.427] | 0.659 | 0.647 |
| V2_macro | M5 | 0.296 [0.291; 0.300] | 0.289 [0.284; 0.296] | 0.014 [0.009; 0.019] | 0.014 [0.008; 0.022] | 0.623 | 0.607 |
| V3 | M1 | 0.376 [0.371; 0.381] | 0.357 [0.350; 0.364] | 0.538 [0.519; 0.560] | 0.513 [0.480; 0.545] | 0.691 | 0.669 |
| V3 | M2 | 0.401 [0.395; 0.406] | 0.384 [0.377; 0.392] | 0.338 [0.318; 0.356] | 0.291 [0.262; 0.314] | 0.692 | 0.673 |
| V3 | M3 | 0.392 [0.385; 0.397] | 0.371 [0.364; 0.380] | 0.425 [0.404; 0.446] | 0.379 [0.348; 0.412] | 0.688 | 0.664 |
| V3 | M4 | 0.381 [0.376; 0.385] | 0.359 [0.353; 0.367] | 0.541 [0.522; 0.561] | 0.513 [0.483; 0.542] | 0.696 | 0.675 |
| V3 | M5 | 0.308 [0.303; 0.312] | 0.297 [0.290; 0.303] | 0.028 [0.021; 0.035] | 0.021 [0.014; 0.029] | 0.640 | 0.624 |

V1 = GroupKFold(5) por hogar_uid x 4 repeticiones (20 bloques; la prediccion OOF agregada usa la repeticion 0); V3 = particion estratificada urbano/rural; V2_macro = dejar fuera una region natural (costa/sierra/selva). V2 deja fuera una region ENTERA: mide transferencia entre regiones y se espera peor desempeno.

Observacion: con solo 3 ensayos M5 queda claramente por debajo de los demas (F1-macro ponderado 0,29-0,30 frente a 0,34 en la CV anidada con 20 ensayos) y casi no predice la clase severa con argmax; no se probo si la diferencia se debe al presupuesto de ensayos, por lo que no se atribuye a una causa.

## 5. Registro en PostgreSQL

| nombre | escenario | esquema | estado | modelos | metricas |
|---|---|---|---|---|---|
| ENAHO_2025_REAL_cv_anidada_real_socio | socio | V1 | completado | 5 | 945 |
| ENAHO_2025_REAL_cv_anidada_real_fies | fies | V1 | completado | 5 | 945 |
| ENAHO_2025_REAL_cv_anidada_real_hibrido | hibrido | V1 | completado | 5 | 945 |
| ENAHO_2025_REAL_clasico_socio_V1 | socio | V1 | completado | 5 | 2710 |
| ENAHO_2025_REAL_clasico_socio_V2_macro | socio | V2 | completado | 5 | 500 |
| ENAHO_2025_REAL_clasico_socio_V3 | socio | V3 | completado | 5 | 2800 |

## 5b. Verificaciones ejecutadas

- `py_compile` de todos los archivos Python editados o creados: OK.
- `protocolo/codigo_verificacion_app/verificar_apptest.py`: completo, sin excepciones.
- `protocolo/codigo_verificacion_app/verificar_fases_apptest.py` contra la BD con el dataset real presente (sin modo pesado): 56 comprobaciones [OK], 0 fallos, `RESULTADO: TODO OK`.
- El paso de reportes del flujo clasico (`exportar_excel` de `fsai/report_generator.py`) fallo con `KeyError: 'titulo'` al final de la corrida clasica, DESPUES de persistir en PostgreSQL y de escribir los CSV; no se edito ese archivo (esta siendo modificado por otro agente): la firma actual de `exportar_excel` no coincide con la llamada de `train_pipeline.py`. Los informes Excel/Word/PDF de esa corrida no se generaron.

## 6. Que NO se hizo o no es valido (limites)

- Presupuesto de ensayos 20 (no 100) y 5 en fies/hibrido: ver `DESVIACIONES.md` (D1-D2). Con 20 ensayos la busqueda de M4/M5 es pobre; las diferencias entre modelos no son concluyentes.
- V1c y V2-LODO del flujo clasico no se ejecutaron; V4 (temporal) es imposible con un solo ano; no hay analisis de sensibilidad de los umbrales FIES ni de la clasificacion Rasch oficial.
- `personas_anonimizadas` no se cargo. `informalidad` y `fies_rasch_logit` son NaN; no hay conglomerado/estrato en la BD, por lo que los IC no incorporan el diseno muestral.
- Los hogares sin FIES completo (n = 1637) se excluyeron y su exclusion no es aleatoria (seccion 1): posible sesgo de seleccion no corregido.
- Estudios Optuna en memoria (no persistidos): quedan los CSV de ensayos por modelo en `trials_<escenario>_<modelo>.csv` de la corrida.
- Equidad: las brechas se calculan sobre la Prueba (n pequeno en varios grupos, p. ej. jefes menores de 30 anos) y son descriptivas; no establecen causa ni discriminacion.
- Los resultados describen asociaciones predictivas en datos 2025; no son afirmaciones causales ni deben usarse para negar o asignar apoyos a hogares concretos.
