# Fuentes públicas adicionales (Fase 2 V2)

Verificadas el 2026-11-16: todas responden HTTP 200 **sin cuenta ni credenciales**. Se bajan con
`python -I foodsecurity_ai/scripts/descargar_fuentes_adicionales.py [--solo grupos] [--forzar] [--derivar]`
a `FSAI_DIR_DATOS_PUBLICOS` (por defecto `<repo>/datos_publicos`, ignorado por git). El script solo baja y hashea (SHA-256); registra
URL, fecha, estado HTTP, bytes, licencia, granularidad, años y estado `ok`/`fallo` (+ motivo) en `MANIFIESTO_ADICIONALES.json`
(copia versionada en `datos_derivados/`). Todo se trata como **no confiable**: los zips se leen sin extraerlos a disco y los
archivos se parsean solo como datos.

| Fuente | Qué es | Granularidad / años | Licencia | Uso |
|---|---|---|---|---|
| `geojson_departamental` (juaneladio/peru-geojson) | Límites de 25 departamentos (`NOMBDEP`, `FIRST_IDDP`) | departamento | **no declarada; verificar antes de redistribuir** | mapas; centroides |
| `geojson_geoboundaries` | Respaldo geoBoundaries gbOpen PER ADM1 (26 unidades) | ADM1 | Public Domain según su API (no reverificado) | respaldo (no incluido en la app) |
| `endes_2024_m1629/1638/1641` (INEI) | ENDES 2024: hogar (RECH0), niños (RECH6), mujeres (RECH5), programas sociales | microdatos con UBIGEO/HV024; 2024 | sin licencia explícita en las páginas consultadas; citar INEI | indicadores de salud/programas |
| `ophi_mpi_peru` (OPHI vía HDX) | MPI, incidencia e intensidad por departamento | departamento; ENDES 2023 | CC0 (según la ficha HDX) | pobreza multidimensional |
| `noaa_oni` (NOAA CPC) | Índice ONI (Niño 3.4) | nacional, trimestre móvil; 1950-2026 | dominio público (gobierno de EE. UU.) | ENSO por mes de encuesta |
| `openmeteo_<cod>` (Open-Meteo / ERA5) | Precipitación diaria en el centroide de cada departamento | punto; 2015-2025 | Open-Meteo CC BY 4.0 (uso no comercial) sobre ERA5 (Copernicus) | **anomalía de precipitación** |
| `nasa_power_<cod>` (NASA POWER, MERRA-2) | Precipitación mensual en el mismo punto | punto; 2015-2026 | datos abiertos NASA | **solo diagnóstico (descartada)** |

**Intentos fallidos o descartados (no reintentar ni afirmar):** API CKAN de datosabiertos.gob.pe, MIDIS REDinforma, UNICEF JME, IMF
(xls de commodities), CHIRPS, SENAMHI. Precios WFP: solo Lima y terminan en 2022. **NASA POWER** se descargó (25/25 puntos) pero su
serie de 2025 muestra un salto no verosímil (p. ej. Lima ~166 mm frente a 27-70 mm anuales en 2015-2024; Ayacucho 757 mm frente a
300-400 mm) y su anomalía no se correlaciona con la de ERA5 (Spearman ≈ -0,01), por lo que **no alimenta ningún rasgo**; ERA5 sí.
ERA5 sobreestima la lluvia absoluta en los Andes y la costa (celdas de ~25 km), de ahí que el rasgo sea una anomalía relativa a la
climatología del propio punto, no milímetros absolutos.

## Indicadores territoriales derivados (`datos_derivados/indicadores_territoriales_departamento.csv`)

Una fila por departamento (`cod_departamento` de 2 dígitos = `fuentes_publicas.DEPARTAMENTOS`; Lima = Lima Metropolitana + Provincias,
Callao aparte). Construidos por `fsai/fuentes_territoriales.py` (funciones puras, probadas con zips sintéticos).

| Columna | Fuente y año | Definición |
|---|---|---|
| `terr_desnutricion_cronica_pct` | ENDES 2024 RECH6 `HC70` + RECH0 `HV005` | % niños 0-59 m con talla/edad z < -2 (OMS), solo datos válidos (se excluyen los códigos 9996-9999), ponderado HV005/1e6 |
| `terr_anemia_ninos_pct` | ENDES 2024 RECH6 `HC57` | % niños con anemia (niveles 1-3 sobre 1-4), hemoglobina ajustada por altitud (RM 363-2022-MINSA) |
| `terr_anemia_mujeres_pct` | ENDES 2024 RECH5 `HA57` | % mujeres 15-49 con anemia (1-3 sobre 1-4) |
| `terr_hogares_programa_social_pct` | ENDES 2024 módulo 1641 (`QH91,93,95,99,101,103,106`) | % hogares con algún beneficiario de Beca 18, Trabaja Perú, Juntos, Pensión 65, Vaso de Leche, Comedor o Cuna Más |
| `terr_hogares_juntos_pct`, `_pension65_pct`, `_vaso_leche_pct` | idem (`QH95`, `QH99`, `QH101`) | % hogares beneficiarios de cada programa (denominador: respuestas 1/2) |
| `terr_mpi_indice`, `_incidencia_pct`, `_intensidad_pct` | OPHI MPI (ENDES 2023) | MPI, % de pobres multidimensionales (H) e intensidad (A) |
| `terr_precip_anom_pct`, `_z` | Open-Meteo/ERA5, **2025** vs 2015-2024 | precipitación de los 12 meses de 2025 vs media anual 2015-2024 (% y desviaciones estándar); exige 12 meses completos |
| `terr_oni_mes` | NOAA ONI, por (anio, mes) de la encuesta | ONI del trimestre móvil centrado en el mes (DJF = enero … NDJ = diciembre); `datos_derivados/oni_mensual.csv` |

Correcciones a la investigación previa: (1) la anemia **infantil** está en **RECH6** (`HC56/HC57`); RECH5 contiene `HA*` (mujeres
15-49) — se usan ambos y se nombran correctamente; (2) el mapa de variables de programas sociales **se verificó contra los
archivos `PS_*` del propio módulo** (n.º de hogares con respuesta 1 = hogares distintos de cada archivo): `QH99` = Pensión 65,
`QH101` = Vaso de Leche, `QH103` = Comedor, `QH106` = Cuna Más; `QH95` = Juntos se infiere de que `QH96*` (orden, tiempo, acuerdo
de compromiso) solo está llenado cuando `QH95 = 1` (5 271 filas); no existe un archivo separado que lo confirme.

## Advertencias que deben leerse con cualquier resultado

* **Desfase temporal.** ENDES 2024 y el MPI (ENDES 2023) no son de 2025 (año de la ENAHO). Son indicadores estructurales del
  departamento, no mediciones de 2025. Precipitación y ONI sí corresponden al año/mes de la ENAHO.
* **Variables constantes por departamento.** Hay solo 25 valores distintos; no pueden mejorar la discriminación *dentro* de un
  departamento y se solapan con la variable categórica `departamento` (que ya está en `socio`). Con validación geográfica por
  departamento (V2 LODO) los valores de un departamento retenido son extrapolación, no información conocida.
* **Anti-fuga.** Ningún indicador se deriva de ENAHO/FIES ni del objetivo; `verificar_sin_fuga` lo aserta (columnas y fuentes
  prohibidas) y hay pruebas que lo comprueban. Asociaciones, no causas (ética INV-13).
* **Calidad de datos.** Estimaciones por departamento con ~550-2 200 niños (`n_ninos_*` en el CSV): hay error de muestreo no
  reportado (no se calcularon IC con diseño muestral).

## Escenario `socio_terr` y motor

* `socio_terr` = `socio` (sin cambios) + `FEATURES_SOCIO_TERR_NUM` (13 columnas `terr_*`) + ingeniería de variables
  (`log_ingreso_pc`, `log_gasto_pc`, `log_gasto_alim_pc`, `log_ingreso_por_adulto_eq`, `razon_dependencia`,
  `asistencia_escolar_faltante`, `tiene_menor5`, `grupo_edad_jefe`, `grupo_educ_jefe`; todas solo de predictores; interruptor
  `--sin-ingenieria`). `grupo_*`/`tiene_menor5` son funciones deterministas de `edad_jefe`, `nivel_educ_jefe`, `n_menores5`
  (ya predictores): no son fuga, solo redundancia. `ESCENARIOS`/`MODELOS` y la opción `todos` **no cambian**.
* `M6` = `HistGradientBoostingClassifier` (parada temprana nativa, `class_weight`, `sample_weight`); extra opcional. LightGBM no
  está instalado y no se instaló. No se activa el soporte categórico nativo (se usa el mismo preprocesador one-hot que M1-M5).
  Limitación: la parada temprana estratifica por clase y falla si una clase tiene < 2 filas en el entrenamiento de un ajuste
  (Optuna absorbe los ensayos fallidos); `train_pipeline` no tiene explicador SHAP para M6 (se registra el fallo y sigue).
* Pesos de ajuste (`--ponderar-ajuste`, opcional, defecto **apagado**): `--recorte-pesos-p P` recorta en el percentil P y
  `--normalizar-pesos` reescala a media 1; solo afectan el ajuste, las métricas usan `factor_expansion` original.
* Umbrales por clase (`--optimizar-umbrales`): búsqueda coordenada de multiplicadores sobre las probabilidades calibradas de
  **Validación**; Prueba se evalúa una sola vez (`--evaluar-prueba`) y se reportan argmax y optimizado.
