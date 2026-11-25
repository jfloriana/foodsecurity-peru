# FoodSecurity AI V2 (INV-13)

Sistema de apoyo para estudiar la inseguridad alimentaria de los hogares peruanos con microdatos de la ENAHO y la escala FIES: carga y validación de datos, modelos de aprendizaje automático (M1-M5), validación cruzada anidada, interpretabilidad (SHAP, PDP/ALE), equidad por subgrupos e informes.

**Uso ético.** Los modelos describen asociaciones predictivas, no causas. No deben usarse para asignar ni negar ayudas a hogares individuales. Las métricas de equidad son una alerta, no una certificación.

V2 es la versión mejorada de `SISTEMA_ALIMENTARIO` (V1, carpeta `deposito_zenodo`). Los datos crudos de la ENAHO y la base `pgdata` **no** están en este repositorio.

## 1. Estructura

| Carpeta | Contenido |
|---|---|
| `foodsecurity_ai/fsai/` | Biblioteca: carga de datos, FIES, particiones, modelos, CV anidada, métricas ponderadas, pruebas estadísticas, equidad, informes. |
| `foodsecurity_ai/tests/` | Suite pytest (solo datos sintéticos; no requiere PostgreSQL ni el parquet real). |
| `foodsecurity_ai/scripts/` | Entrenamiento real por fases (`entrenar_real_fase2.py`), post-proceso, construcción del dataset ENAHO. |
| `foodsecurity_ai/train_pipeline.py` | Pipeline clásico (esquemas V1, V1c, V2, V3, V4, V5). |
| `app_streamlit/` | Aplicación Streamlit (`app/app.py` y `app/pages/`), `Dockerfile`, `docker-compose.yml`, `requirements*.txt`. |
| `sql/` | DDL del esquema `fsai` y consultas. `app_streamlit/sql/02_indices.sql` crea los índices. |
| `scripts_operacion/` | Respaldo/restauración de la BD y lanzador del entrenamiento con 100 ensayos. |
| `diccionario/` | `diccionario_variables.csv` (variables, unidades, ítems «por validar»). |
| `resultados_agregados/` | Resultados agregados de V1 (sin microdatos). |
| `referencia_v1/` | `DESVIACIONES.md`, `MEJORAS_PROPUESTAS.md`, `RESULTADOS_ENTRENAMIENTO_REAL.md` de V1. |

## 2. Instalación

Requisito: Python 3.12 (probado con 3.12.7 en Windows 11).

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r app_streamlit/requirements-lock.txt     # versiones exactas del entorno probado
# alternativa con rangos: pip install -r app_streamlit/requirements.txt
```

`requirements-lock.txt` reproduce las versiones del equipo donde se ejecutaron las pruebas; **no** se probó una instalación limpia con él ni en otro sistema operativo. `torch` se instala en su variante CPU desde el índice de PyTorch.

## 3. Ejecutar las pruebas

```bash
python -m pytest foodsecurity_ai/tests -q
```

Funciona desde la raíz del proyecto (el `conftest.py` de `tests/` agrega `foodsecurity_ai` al `sys.path`). Las pruebas de V1 que dependen del parquet real (`enaho_analitico.parquet`) terminan sin comprobar nada (cuentan como «passed», no como «skipped») si el archivo no existe. No hace falta base de datos.

## 4. Ejecutar la aplicación

La aplicación necesita PostgreSQL con el esquema `fsai` y usuarios definidos en `FSAI_USUARIOS` (ver `app_streamlit/.env.example`; el hash se genera con `python app_streamlit/app/auth.py`).

### 4.1 Sin Docker

1. Cree la base y aplique `sql/01_ddl_fsai.sql`, `sql/02_roles_grants.sql` y `app_streamlit/sql/02_indices.sql`.
2. Defina las variables (ejemplo en PowerShell):
   ```powershell
   $env:FSAI_DATABASE_URL  = "postgresql+psycopg2://usuario:clave@localhost:5432/fsai"
   $env:FSAI_SALT          = "<sal secreta larga>"           # seudónimo de hogar_uid; no se versiona
   $env:FSAI_USUARIOS      = '<lista JSON generada con auth.py>'
   $env:FSAI_RAIZ_PROYECTO = "C:\ruta\SISTEMA_ALIMENTARIO_V2\foodsecurity_ai"   # donde está el paquete fsai
   ```
3. `streamlit run app_streamlit/app/app.py`

### 4.2 Con docker-compose

```bash
cd app_streamlit
copy .env.example .env         # complete POSTGRES_PASSWORD y FSAI_USUARIOS (no versione .env)
docker compose up --build
```

La app queda en `http://127.0.0.1:8501` y PostgreSQL no se expone fuera de la red de Compose.

**Disposición V2 de la imagen.** El contexto de compilación es la raíz del proyecto (`context: ..` en `docker-compose.yml`; con `docker build`: `docker build -f app_streamlit/Dockerfile .`). El `Dockerfile` usa Python 3.12, instala `app_streamlit/requirements-lock.txt` (versiones exactas; torch CPU vía `--extra-index-url`) y copia `foodsecurity_ai/fsai` a `/app/fsai`, `foodsecurity_ai/train_pipeline.py` a `/app/train_pipeline.py`, `app_streamlit/app` a `/app/app` y `resultados_agregados` a `/app/resultados_agregados` (usado por «Comparación de corridas»). El DDL se monta desde `../sql/01_ddl_fsai.sql`. **Estado:** `docker compose config` valida el archivo, pero la imagen NO se construyó ni se ejecutó en esta sesión (no se probó la instalación del lock en Linux, la descarga de torch CPU ni el arranque con PostgreSQL). Al no haber `.dockerignore`, el contexto incluye toda la raíz (excepto lo que Docker excluya por defecto); conviene crear uno que omita `.git`, `referencia_v1`, `pgdata` y datos crudos.

## 5. Dónde van los datos

`foodsecurity_ai/fsai/config.py` define las rutas (valores por defecto entre paréntesis):

| Variable de entorno | Uso | Defecto |
|---|---|---|
| `FSAI_DIR_DATOS` | Carpeta con los archivos tabulares (`.csv`, `.dta`, `.sav`, `.parquet`) que la app permite cargar | `<artifacts>/../datos` |
| `FSAI_DATA_DIR` | Carpeta `data` de la biblioteca | `foodsecurity_ai/data` |
| `FSAI_ARTIFACTS_DIR` | Particiones, modelos, corridas (`fases/`), estudios Optuna, informes | `foodsecurity_ai/artifacts` |
| `FSAI_DATABASE_URL` | Conexión a PostgreSQL | vacío |
| `FSAI_SALT` | Sal del seudónimo `hogar_uid` | vacío (obligatoria para construir `hogar_uid`) |
| `FSAI_N_HILOS` | Hilos por modelo (-1 = todos) | `-1` |
| `FSAI_MAPA_COLUMNAS` | Ruta de un JSON `{"codigo_inei": "nombre_analitico"}` para otros años/archivos (nuevo en V2) | vacío |

Nota: la variable `FSAI_DATOS_DIR` **no existe** en el código; la que se usa es `FSAI_DIR_DATOS`. Los scripts de construcción de V1 (`scripts/construir_dataset_enaho.py`) y `verificar_*.py` esperan además una carpeta hermana `datos_publicos/procesado/enaho_analitico.parquet` (convención de V1); en V2 esa carpeta no se incluye y debe prepararse a mano.

## 6. Qué hay de nuevo en V2 respecto de V1

Hecho y probado con la suite (datos sintéticos):

- **Suite pytest ampliada** (de 8 a 121 pruebas): particiones 70/15/15 sin fuga de hogares y reproducibles, validación geográfica V2 (LODO y macrorregión), métricas ponderadas con valores calculados a mano, equidad por subgrupo, pruebas estadísticas (Holm, McNemar ponderado, Wilcoxon, Friedman/Nemenyi, DeLong, bootstrap), clasificación FIES, etiquetas, contrato de esquema, configuración de experimento y scripts de operación.
- **Mejora #16 (SHAP almacenado):** `reportes_fases._shap_desde_bd` acepta `modelo` y `escenario`; el informe consulta primero el mejor modelo y el escenario evaluado y solo cae al promedio mezclado (declarado en el texto) si no hay filas. Probado con una fuente SQL simulada; no contra PostgreSQL real.
- **Mejora #2 (contrato de esquema):** `fsai/contrato_esquema.py` (`validar_esquema` devuelve un informe estructurado sin excepciones por datos) y `MAPA_COLUMNAS` configurable (`FSAI_MAPA_COLUMNAS`), aplicado también en `datasets.leer_tabla`. Los rangos del contrato por defecto son los ya declarados en el proyecto; los códigos de otros años siguen **[POR VALIDAR]** contra el diccionario INEI.
- **Mejora #4 (configuración de experimento):** `fsai/config_experimento.py` exporta/importa un JSON versionado (corrida, escenario, modelos, espacios de hiperparámetros, semillas, `n_trials`, esquema de partición, hash del dataset) y rechaza claves faltantes o desconocidas. Todavía no hay botón de importación en la app.
- **Mejora #7 (entorno):** `app_streamlit/requirements-lock.txt` regenerado con las versiones instaladas.
- **Mejora #6 (respaldo):** `scripts_operacion/respaldo_bd.py` y `restaurar_bd.py` (`pg_dump`/`pg_restore`, contraseña solo por `PGPASSWORD`, `--dry-run`, huella SHA-256, `--confirmar` obligatorio para restaurar). Solo se probó la construcción de comandos: **no se ejecutó un respaldo ni una restauración reales.**
- **Reentrenamiento con 100 ensayos:** `scripts_operacion/entrenar_100_trials.py` arma el flujo de `entrenar_real_fase2.py` con `--trials 100 --trials-final 100` y muestra una tabla de tiempos estimados. **No se ejecutó.** Las horas son extrapolaciones lineales de los tiempos por ensayo medidos en V1 (D1), no una medición con 100 ensayos.

Páginas nuevas de V2 en `app_streamlit/app/pages/`: `14_Glosario.py` (términos con «cómo leerlo» y advertencias éticas, y diccionario de variables; todos los roles), `15_Comparacion_Corridas.py` (compara corridas desde `fsai.experimentos` o, sin BD, desde los JSON de `resultados_agregados/`; ruta configurable con `FSAI_RESULTADOS_AGREGADOS`; avisa de n_trials/escenarios distintos y de escenarios circulares; todos los roles) y `16_Bitacora.py` (consulta/CSV de `fsai.auditoria`; solo `admin`). En «Entrenamiento» (pestaña de CV anidada) el expander «Exportar / importar configuración» usa `fsai.config_experimento` y nunca lanza un entrenamiento. Los roles de las páginas nuevas se completan en `modulos.registrar_accesos()` (no se editó `auth.py`). Verificación sin PostgreSQL: `python app_streamlit/verificar_paginas_v2_apptest.py` (AppTest de las páginas 14-16 y 06 con la base ausente; no ejercita consultas reales a `fsai.auditoria` ni a `fsai.experimentos`).

Verificación de V2 geográfica: `validation.py` ya implementa `particiones_lodo` y `particiones_macroregion`, y `train_pipeline.py` las usa (`V2_lodo`, `V2_macro`); V2 es parte de V1 y no es una novedad.

## 6b. Fase 2: fuentes territoriales, escenario `socio_terr` y modelo M6

Detalle en `docs/FUENTES_ADICIONALES.md`. Resumen:

1. Descargar y derivar (necesita red; solo URL públicas, sin cuenta): `python -I foodsecurity_ai/scripts/descargar_fuentes_adicionales.py --derivar`
   (directorio `FSAI_DIR_DATOS_PUBLICOS`, por defecto `datos_publicos/`, ignorado por git; el resultado pequeño queda en `datos_derivados/` y se versiona).
2. Entrenar el escenario nuevo y la línea base con la MISMA partición, ensayos y semilla (comandos preparados, **no ejecutados** en esta sesión):
   - Partición única: `python foodsecurity_ai/scripts/entrenar_real_fase2.py --ruta-datos <parquet> --nombre cmp_terr --solo-particion`
   - Escenario nuevo: `python foodsecurity_ai/scripts/entrenar_real_fase2.py --ruta-datos <parquet> --nombre cmp_terr --escenarios socio_terr --modelos M1 M2 M3 M6 --trials 3`
   - Línea base: igual con `--escenarios socio`.
   - Opciones: `--ponderar-ajuste [--recorte-pesos-p 99] [--normalizar-pesos]`, `--sin-ingenieria`, `--optimizar-umbrales [--evaluar-prueba]`.
   - `train_pipeline.py --escenario socio_terr --modelos M1 M2 M3 M6 --sin-db --sin-optuna-db ...` admite los mismos controles (`--sin-db`/`--sin-optuna-db` solo existen ahí; `entrenar_real_fase2.py` nunca usa la BD).
3. `todos`, `ESCENARIOS` y `MODELOS` no incluyen `socio_terr`/`M6` (se piden explícitamente); la definición de `socio` no cambió (una prueba la fija).
4. Limitaciones: ENDES 2024/MPI 2023 vs ENAHO 2025 (desfase); indicadores constantes por departamento (25 valores); NASA POWER descartada por un salto en 2025 (se usa ERA5/Open-Meteo); licencia del GeoJSON no declarada; `fsai.config_experimento` (importar configuración) aún solo acepta los `ESCENARIOS`/`MODELOS` de V1.

## 7. Limitaciones conocidas (honestas)

Tomadas de `referencia_v1/DESVIACIONES.md` y `MEJORAS_PROPUESTAS.md`; siguen vigentes salvo que se indique lo contrario.

- **D1. 20 ensayos Optuna en lugar de 100** en la corrida real de V1 (6 búsquedas por modelo). La búsqueda de hiperparámetros de M4 y M5 es pobre y sus diferencias frente a M1/M2 deben leerse con cautela. V2 deja listo el lanzador de 100 ensayos, pero no se ha ejecutado.
- **Circularidad (D2).** `fies_nivel` se deriva de `fies_q1..q8`: los escenarios `fies` e `hibrido` reconstruyen el objetivo, no lo predicen (se corrieron con 5 ensayos). Solo `socio` es predicción.
- **D3.** M1 se buscó solo con penalización L2.
- **D4.** Los esquemas V1c, V2-LODO y V3 de la corrida real usaron 3 ensayos por bloque (casi búsqueda aleatoria); no son comparables con la CV anidada primaria. V4 (temporal) es imposible con un solo año.
- **Un solo año (2025)** y el FIES refiere al mes previo a la entrevista (estacionalidad).
- **Umbrales FIES [POR VALIDAR]:** leve >= 1, moderada >= 4, severa >= 7 sobre el puntaje crudo; no es la clasificación oficial (Rasch) y no se hizo análisis de sensibilidad. `fies_rasch_logit` está 100 % vacío.
- **`informalidad` está 100 % vacía** y se excluye de los predictores (D8). `asistencia_escolar` falta en ~61 % de los hogares.
- **Población de entrenamiento (D9):** solo hogares con FIES completo (32 065 de 33 702); la exclusión no es aleatoria.
- **Diseño muestral incompleto (brecha #3):** los IC de métricas y equidad ignoran estrato y conglomerado; no se guarda el estrato/UPM en la BD.
- **Rendimiento predictivo modesto** (F1-macro ponderado de la Prueba entre 0,353 y 0,375 en V1): no sirve como instrumento de focalización individual.
- **Códigos y textos (verificados contra el diccionario ENAHO 2025):** `P101`, `P102`, `P301A`, `POBREZA`, `P130_1` a `P130_8` y los valores monetarios deflactados están documentados en `fsai/catalogo_campos.py` (catálogo único del formulario de Inferencia). **Siguen [POR VALIDAR]:** equivalencia de `area` (estratos 1-5) con la definición urbana del INEI, definición de `OCU500` = 2 y 3 (desocupado), uso de `dependencia_economica` (ocupados ÷ miembros) como indicador de dependencia, y las variables accionables del contrafactual. Limitación FIES: el diccionario trae 1 Sí, 2 No, 3 No sabe, 4 No responde; el proyecto usa solo Sí/No y trata el resto como faltante.
- **Sin verificar:** accesibilidad (contraste, teclado, lectores de pantalla), carga con muchos usuarios, seguridad de `auth.py` más allá de roles y bloqueo, restauración de respaldos, contenido textual del PDF de informes, instalación limpia con el lock y el flujo de docker-compose en V2.
- **`cargar_particion`** reconstruye por `hogar_uid` y asume una fila por hogar (como el dataset real); con `hogar_uid` repetido no reproduce los índices.
