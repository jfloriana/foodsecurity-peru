# FoodSecurity AI - Auditoría y mejoras propuestas

Fecha: 2026-11-10. Alcance: código en `protocolo/codigo_verificacion/foodsecurity_ai/fsai`, `train_pipeline.py`, `protocolo/codigo_verificacion_app/app` y `despliegue_local`. Esfuerzo en días-persona de una persona que conoce el código (estimación gruesa, no medida). Riesgo = probabilidad de que el cambio rompa algo existente.

## 1. Estado actual verificado

Verificado leyendo el código y/o ejecutándolo en esta sesión:

- **Reporte PDP/ALE**: ahora se calcula siempre (8 variables principales del mejor modelo por F1-macro de la CV anidada, submuestra ponderada de 2 000 hogares de Validación, no de la Prueba). Informe real regenerado en `artifacts/reportes/cv_anidada_real_socio_v2/` con figuras, tablas y valores numéricos (revisado en Word y Excel; el PDF se generó pero no se pudo extraer su texto por falta de `pypdf`).
- **Etiquetas**: `fsai/etiquetas.py` cubre las 50 columnas del parquet (`validar_cobertura` devuelve vacío). Se aplican solo en presentación (gráficos, tablas del informe, Inferencia, EDA, SHAP, Equidad, textos interpretativos). Las columnas de entrenamiento no se renombran.
- **Pruebas existentes**: no hay suite pytest/unittest. La verificación es `verificar_apptest.py` y `verificar_fases_apptest.py` (AppTest de Streamlit). `verificar_fases_apptest.py` dio 56 comprobaciones OK y 0 fallos (modo ligero). `verificar_apptest.py` falla en `09_SHAP.py` cuando se ejecuta con `_con_env.py` porque usa el dataset demo contra corridas reales (desajuste de entorno de la prueba, no del cambio).
- **Reproducibilidad**: semilla global, hash de constantes, SHA-256 de datos, partición y artefactos en la sección 8 del informe; `requirements.txt` con rangos de versión (26 líneas, sin versiones exactas bloqueadas).
- **Control de acceso**: roles `admin/analista/lector` por módulo, bloqueo por intentos fallidos, auditoría en `fsai.auditoria` (`audit(...)`).
- **Candado de la Prueba**: archivos `.lock`/`prueba_*.json` por modelo; la Prueba se evalúa una sola vez.
- **Calibración**: isotónica sobre Validación y métricas Brier/ECE (`nested_cv.py`, `extra_*.json`).
- **Pesos**: métricas ponderadas y no ponderadas, IC bootstrap; prevalencia con linealización por conglomerado (`reportes_fases.prevalencia_ic`) sin estratificación.
- **Accesibilidad parcial**: paleta Okabe-Ito (segura para daltonismo) en figuras; logos SVG.
- **Caché**: `st.cache_data` en carga de datasets (ttl 600 s).

## 2. Brechas priorizadas

| # | Brecha (evidencia) | Prioridad | Esfuerzo | Riesgo |
|---|---|---|---|---|
| 1 | **Sin pruebas automatizadas unitarias** (no hay pytest; solo AppTest end-to-end). Añadir pruebas de `etiquetas`, `pdp_ale_principales`, particiones, métricas ponderadas y CI local. | P1 | 3-5 | Bajo |
| 2 | **Importación de otros datasets/años con validación de esquema**: la metadata (`datasets.py`) valida tipos/roles/rangos, pero la construcción ENAHO (`fuentes_publicas.py`) está atada a módulos y variables 2025 (`P130_x`, `MODULOS_ENAHO_2025`). Falta un contrato de esquema (columnas obligatorias, tipos, rangos) con informe de errores antes de cargar y un mapeo configurable (`MAPA_COLUMNAS` está vacío). | P1 | 5-8 | Medio |
| 3 | **Diseño muestral completo**: los IC de métricas y equidad ignoran estrato y conglomerado; la prevalencia solo incorpora conglomerados. Guardar estrato/UPM y usar réplicas por conglomerado. Conviene decidir si el estrato se persiste (hay columna `estrato` en la tabla de métricas, que es otra cosa: estrato de resultados). | P1 | 5-8 | Medio |
| 3b | Valores de categorías `tipo_vivienda`/`material_vivienda` solo como códigos INEI; falta el catálogo oficial de códigos y las preguntas FIES textuales [POR VALIDAR]. | P1 | 1-2 | Bajo |
| 4 | **Exportar/importar configuraciones de experimento** (corrida, hiperparámetros, escenario, semillas) como un único archivo reutilizable; hoy `config.json` por corrida existe, pero no hay función de importación en la UI. | P2 | 2-3 | Bajo |
| 5 | **Comparación de experimentos** entre corridas distintas (hoy se compara entre modelos de una corrida). | P2 | 3-4 | Bajo |
| 6 | **Respaldo y restauración**: no se encontró `pg_dump`, backup ni restauración en scripts ni `LEEME.md`. Añadir script de respaldo de BD y artefactos. | P2 | 1-2 | Bajo |
| 7 | **Entorno bloqueado**: usar `pip freeze`/lock con versiones exactas; hoy `requirements.txt` usa rangos. | P2 | 0.5 | Bajo |
| 8 | **Ayuda/glosario en la UI**: el diccionario existe (`etiquetas.DICCIONARIO`) y se usa como ayuda en Inferencia; falta una página de glosario (FIES, PDP, ALE, SHAP, ECE, Brier) y extender las ayudas a EDA y Equidad. | P2 | 2 | Bajo |
| 9 | **i18n**: textos en español incrustados en el código; no hay mecanismo de traducción (no se encontró gettext o diccionario de cadenas). Solo necesario si se quiere otro idioma. | P3 | 8-12 | Medio |
| 10 | **Accesibilidad**: no se auditó contraste, navegación por teclado ni lectores de pantalla; las figuras exportadas no incluyen texto alternativo en Word/PDF. | P2 | 3-4 | Bajo |
| 11 | **Rendimiento**: PDP/ALE de M4 (stacking) y SHAP son costosos; en el informe se mitiga con submuestra. Medir tiempos y cachear explicaciones por modelo; `n_jobs` por hilos ya es configurable (`FSAI_N_HILOS`). | P3 | 2-3 | Bajo |
| 12 | **Manejo de errores en informes**: una sección con error se convierte en «Sin datos disponibles (motivo)»; está bien, pero no hay registro estructurado de las secciones degradadas. | P3 | 1 | Bajo |
| 13 | **Permisos y auditoría**: existe auditoría de acciones; falta exportación/consulta de la bitácora y rotación de credenciales (solo se revisó `auth.py` parcialmente). | P3 | 2-3 | Bajo |
| 14 | **Documentación**: hay `LEEME.md`, `DESVIACIONES.md` e informes; falta guía de usuario por módulo y de arquitectura. | P3 | 3 | Bajo |
| 15 | **Calibración por subgrupo** y análisis de sensibilidad de los umbrales FIES (hoy cortes crudos [POR VALIDAR]). | P2 | 3-5 | Bajo |
| 16 | **Importancia SHAP «almacenada» mezclada**: `_shap_desde_bd` promedia todas las explicaciones guardadas del dataset (distintos modelos/escenarios) y el gráfico del informe puede no corresponder al mejor modelo; en esta sesión el orden resultante difirió del SHAP guardado de la corrida. Conviene filtrar por modelo/escenario. | P1 | 1 | Bajo |

## 3. Qué NO se pudo verificar

- Texto exacto de las preguntas FIES (P130_1..8), significado de los códigos de `P101`/`P102` y de `P104`, moneda/base de deflactación de `INGHOG2D`/`GASHOG2D`: marcados [POR VALIDAR] en el diccionario.
- Contenido textual del PDF generado (sin `pypdf`); sí se verificó Word y Excel.
- Contraste, lectores de pantalla y teclado (no se hicieron pruebas de accesibilidad).
- Comportamiento con más usuarios simultáneos, carga y tiempos con datasets mayores.
- Seguridad de `auth.py` más allá de roles, bloqueo y auditoría (no se hizo revisión de seguridad).
- Restauración de copias de seguridad (no existe el procedimiento).
- `verificar_apptest.py` con el entorno demo (no se re-ejecutó con artefactos demo).

## 4. Actualización 2026-11-11 (estado de las brechas P1)

Implementado en esta iteración (verificado con las pruebas indicadas):

- **Perfil descriptivo de hogares vulnerables** (`fsai/perfil_vulnerabilidad.py`; pestaña «Perfil descriptivo» de la página Vulnerabilidad; sección 1b y hojas `Fig_perfil_*` de los informes; pertenencia del hogar y distribución por recuento en Inferencia individual y por lote). Reproduce los valores de la sección 3.1 del artículo (30,0 % frente a 26,3 %, n = 5 332; 27,1 % frente a 26,7 %, n = 10 418; 39,8 % frente a 23,5 %, n = 5 860; total 26,8 %).
- **Brecha 1 (pruebas unitarias), parcial**: `foodsecurity_ai/tests/test_unitarias.py` (8 pruebas: etiquetas, métricas ponderadas, perfil a mano, linealización frente a `reportes_fases.prevalencia_ic`, valores del artículo). Siguen sin pruebas unitarias `pdp_ale_principales`, particiones y el resto de las métricas.
- **Brecha 16 (SHAP almacenada mezclada)**: ya estaba resuelta en el código (`_shap_modelo_guardado` se usa primero; el promedio de `fsai.explicaciones` solo es el último recurso y se declara como mezcla). Esta tabla estaba desactualizada.
- **Brecha 7 (entorno bloqueado), parcial**: `protocolo/codigo_verificacion_app/requirements-lock.txt` con las versiones exactas instaladas (no se probó una instalación limpia con ese archivo).
- **Brecha 8 (glosario/ayudas), parcial**: el glosario lateral incluye «Perfil descriptivo» y «Recuento»; mensajes de error más claros (`comun.explicar_error`; el aviso de esquema ausente ya no remite a una sección del protocolo).

Pendientes y motivo:

- **Brecha 2 (contrato de esquema/otros años)**: requiere un mapeo configurable de columnas y validar contra el diccionario INEI de cada año; riesgo medio y no hay datos de otros años para probarlo.
- **Brecha 3 (diseño muestral completo)**: exige guardar estrato y UPM (cambio de esquema de BD) y réplicas por conglomerado en métricas y equidad; fuera del alcance de «sin cambios de esquema arriesgados». El perfil usa linealización por conglomerado sin estratos y lo declara.
- **Brecha 3b (catálogos de códigos y preguntas FIES)**: necesita el diccionario oficial del INEI, que no se pudo verificar sin conexión a la fuente.
- **Brechas 4, 5, 6, 9-15**: sin cambios en esta iteración (P2/P3).
