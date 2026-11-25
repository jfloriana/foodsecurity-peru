-- =====================================================================
-- FoodSecurity AI | DDL del esquema fsai (PostgreSQL >= 13)
-- Orden de dependencia: datasets -> hogares -> tablas hijas -> experimentos
--                       -> modelos -> predicciones/metricas/explicaciones -> auditoria
-- Idempotente: se puede ejecutar varias veces sin error.
-- =====================================================================
BEGIN;

CREATE SCHEMA IF NOT EXISTS fsai;
COMMENT ON SCHEMA fsai IS 'FoodSecurity AI (predicción de inseguridad alimentaria y vulnerabilidad nutricional con IA explicable): datos anonimizados de ENAHO/FIES, experimentos de modelado y trazabilidad.';

-- ---------------------------------------------------------------------
-- 1. datasets
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fsai.datasets (
    dataset_id    serial       PRIMARY KEY,
    nombre        text         NOT NULL,
    fuente        text         NOT NULL,
    anio          smallint     NOT NULL CHECK (anio BETWEEN 2000 AND 2100),
    version       text         NOT NULL DEFAULT '1',
    hash_archivo  char(64)     CHECK (hash_archivo IS NULL OR hash_archivo ~ '^[0-9a-f]{64}$'),
    n_filas       integer      CHECK (n_filas IS NULL OR n_filas >= 0),
    creado_en     timestamptz  NOT NULL DEFAULT now(),
    CONSTRAINT uq_datasets_nombre_anio_version UNIQUE (nombre, anio, version)
);
COMMENT ON TABLE  fsai.datasets IS 'Catalogo de conjuntos de datos cargados (por ejemplo, ENAHO 2025 modulos integrados). Garantiza trazabilidad por hash.';
COMMENT ON COLUMN fsai.datasets.dataset_id   IS 'Identificador sustituto del dataset.';
COMMENT ON COLUMN fsai.datasets.nombre       IS 'Nombre descriptivo del dataset.';
COMMENT ON COLUMN fsai.datasets.fuente       IS 'Institucion y origen (por ejemplo, INEI-ENAHO).';
COMMENT ON COLUMN fsai.datasets.anio         IS 'Anio de referencia de la encuesta.';
COMMENT ON COLUMN fsai.datasets.version      IS 'Version del procesamiento o de la publicacion de microdatos.';
COMMENT ON COLUMN fsai.datasets.hash_archivo IS 'SHA-256 hexadecimal (64 caracteres) del archivo fuente, para reproducibilidad.';
COMMENT ON COLUMN fsai.datasets.n_filas      IS 'Numero de filas (hogares) cargadas tras validacion de calidad.';
COMMENT ON COLUMN fsai.datasets.creado_en    IS 'Marca de tiempo de registro.';

-- ---------------------------------------------------------------------
-- 2. hogares  (sin identificadores directos)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fsai.hogares (
    hogar_uid         char(64)         PRIMARY KEY CHECK (hogar_uid ~ '^[0-9a-f]{64}$'),
    dataset_id        integer          NOT NULL REFERENCES fsai.datasets(dataset_id) ON DELETE RESTRICT,
    conglome_hash     char(64)         CHECK (conglome_hash IS NULL OR conglome_hash ~ '^[0-9a-f]{64}$'),
    ubigeo            char(6)          CHECK (ubigeo IS NULL OR ubigeo ~ '^[0-9]{6}$'),
    departamento      text,
    provincia         text,
    distrito          text,
    area              char(1)          NOT NULL CHECK (area IN ('U','R')),
    region_natural    text             CHECK (region_natural IS NULL OR region_natural IN ('costa','sierra','selva')),
    factor_expansion  double precision NOT NULL CHECK (factor_expansion > 0),
    anio              smallint         NOT NULL CHECK (anio BETWEEN 2000 AND 2100)
);
COMMENT ON TABLE  fsai.hogares IS 'Hogar anonimizado. Clave hogar_uid = SHA-256(conglome|vivienda|hogar|salt). No contiene nombres, DNI, direcciones ni coordenadas.';
COMMENT ON COLUMN fsai.hogares.hogar_uid        IS 'Seudonimo del hogar: SHA-256 hexadecimal (64 caracteres) con sal secreta externa.';
COMMENT ON COLUMN fsai.hogares.dataset_id       IS 'Dataset de origen.';
COMMENT ON COLUMN fsai.hogares.conglome_hash    IS 'SHA-256 salado del conglomerado, para agrupar sin exponer el codigo original (validacion agrupada y analisis de fuga por conglomerado).';
COMMENT ON COLUMN fsai.hogares.ubigeo           IS 'Codigo de ubicacion geografica de 6 digitos (nivel distrito).';
COMMENT ON COLUMN fsai.hogares.departamento     IS 'Departamento.';
COMMENT ON COLUMN fsai.hogares.provincia        IS 'Provincia.';
COMMENT ON COLUMN fsai.hogares.distrito         IS 'Distrito.';
COMMENT ON COLUMN fsai.hogares.area             IS 'Area de residencia: U = urbana, R = rural.';
COMMENT ON COLUMN fsai.hogares.region_natural   IS 'Region natural: costa, sierra o selva.';
COMMENT ON COLUMN fsai.hogares.factor_expansion IS 'Factor de expansion del hogar (factor07 de ENAHO); debe ser positivo. Para estimaciones de personas se multiplica por n_miembros [POR VALIDAR].';
COMMENT ON COLUMN fsai.hogares.anio             IS 'Anio de la encuesta.';

-- ---------------------------------------------------------------------
-- 3. personas_anonimizadas
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fsai.personas_anonimizadas (
    persona_uid          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    hogar_uid            char(64) NOT NULL REFERENCES fsai.hogares(hogar_uid) ON DELETE CASCADE,
    parentesco           text,
    edad                 smallint CHECK (edad IS NULL OR edad BETWEEN 0 AND 120),
    sexo                 char(1)  CHECK (sexo IS NULL OR sexo IN ('H','M')),
    nivel_educ           smallint CHECK (nivel_educ IS NULL OR nivel_educ BETWEEN 0 AND 11),
    condicion_actividad  text,
    es_jefe              boolean  NOT NULL DEFAULT false
);
COMMENT ON TABLE  fsai.personas_anonimizadas IS 'Integrantes del hogar sin identificadores directos; solo atributos minimos necesarios para derivar variables del hogar.';
COMMENT ON COLUMN fsai.personas_anonimizadas.persona_uid         IS 'Identificador sustituto no derivable de datos personales.';
COMMENT ON COLUMN fsai.personas_anonimizadas.hogar_uid           IS 'Hogar al que pertenece la persona.';
COMMENT ON COLUMN fsai.personas_anonimizadas.parentesco          IS 'Relacion con el jefe del hogar (categoria INEI recodificada).';
COMMENT ON COLUMN fsai.personas_anonimizadas.edad                IS 'Edad en anios cumplidos (0-120).';
COMMENT ON COLUMN fsai.personas_anonimizadas.sexo                IS 'Sexo: H = hombre, M = mujer.';
COMMENT ON COLUMN fsai.personas_anonimizadas.nivel_educ          IS 'Nivel educativo INEI recodificado (0-11).';
COMMENT ON COLUMN fsai.personas_anonimizadas.condicion_actividad IS 'Condicion de actividad (ocupado, desocupado, inactivo).';
COMMENT ON COLUMN fsai.personas_anonimizadas.es_jefe             IS 'Verdadero si la persona es jefe(a) del hogar.';

-- ---------------------------------------------------------------------
-- 4. respuestas_fies
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fsai.respuestas_fies (
    respuesta_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    hogar_uid         char(64) NOT NULL UNIQUE REFERENCES fsai.hogares(hogar_uid) ON DELETE CASCADE,
    fies_q1           smallint CHECK (fies_q1 IN (0,1)),
    fies_q2           smallint CHECK (fies_q2 IN (0,1)),
    fies_q3           smallint CHECK (fies_q3 IN (0,1)),
    fies_q4           smallint CHECK (fies_q4 IN (0,1)),
    fies_q5           smallint CHECK (fies_q5 IN (0,1)),
    fies_q6           smallint CHECK (fies_q6 IN (0,1)),
    fies_q7           smallint CHECK (fies_q7 IN (0,1)),
    fies_q8           smallint CHECK (fies_q8 IN (0,1)),
    fies_raw_score    smallint CHECK (fies_raw_score BETWEEN 0 AND 8),
    fies_rasch_logit  double precision,
    fies_nivel        smallint CHECK (fies_nivel BETWEEN 0 AND 3),
    ia_binaria        smallint CHECK (ia_binaria IN (0,1)),
    -- Coherencia interna: raw = suma de items cuando los 8 estan presentes
    CONSTRAINT ck_fies_raw_suma CHECK (
        fies_q1 IS NULL OR fies_q2 IS NULL OR fies_q3 IS NULL OR fies_q4 IS NULL OR
        fies_q5 IS NULL OR fies_q6 IS NULL OR fies_q7 IS NULL OR fies_q8 IS NULL OR
        fies_raw_score IS NULL OR
        fies_raw_score = fies_q1 + fies_q2 + fies_q3 + fies_q4 + fies_q5 + fies_q6 + fies_q7 + fies_q8
    ),
    -- ia_binaria = 1 si fies_nivel >= 2
    CONSTRAINT ck_ia_binaria_nivel CHECK (
        fies_nivel IS NULL OR ia_binaria IS NULL OR
        ia_binaria = CASE WHEN fies_nivel >= 2 THEN 1 ELSE 0 END
    )
);
COMMENT ON TABLE  fsai.respuestas_fies IS 'Respuestas FIES (8 items) en codificacion analitica 0/1 y clasificacion derivada. El crudo INEI (codigos 1-7) se valida en la etapa de carga (ver fsai_stg.respuestas_fies_crudo).';
COMMENT ON COLUMN fsai.respuestas_fies.respuesta_id     IS 'Identificador sustituto.';
COMMENT ON COLUMN fsai.respuestas_fies.hogar_uid        IS 'Hogar (relacion 1 a 1).';
COMMENT ON COLUMN fsai.respuestas_fies.fies_q1          IS 'FIES item 1, analitico: 1 = si, 0 = no; NULL = sin dato / no sabe / no responde recodificado.';
COMMENT ON COLUMN fsai.respuestas_fies.fies_q2          IS 'FIES item 2 (0/1).';
COMMENT ON COLUMN fsai.respuestas_fies.fies_q3          IS 'FIES item 3 (0/1).';
COMMENT ON COLUMN fsai.respuestas_fies.fies_q4          IS 'FIES item 4 (0/1).';
COMMENT ON COLUMN fsai.respuestas_fies.fies_q5          IS 'FIES item 5 (0/1).';
COMMENT ON COLUMN fsai.respuestas_fies.fies_q6          IS 'FIES item 6 (0/1).';
COMMENT ON COLUMN fsai.respuestas_fies.fies_q7          IS 'FIES item 7 (0/1).';
COMMENT ON COLUMN fsai.respuestas_fies.fies_q8          IS 'FIES item 8 (0/1).';
COMMENT ON COLUMN fsai.respuestas_fies.fies_raw_score   IS 'Puntaje bruto FIES = suma de items afirmativos (0-8).';
COMMENT ON COLUMN fsai.respuestas_fies.fies_rasch_logit IS 'Medida Rasch del hogar en escala logit; parametros de items [POR VALIDAR].';
COMMENT ON COLUMN fsai.respuestas_fies.fies_nivel       IS 'Nivel FIES: 0 = seguridad, 1 = leve, 2 = moderada, 3 = severa; umbrales [POR VALIDAR].';
COMMENT ON COLUMN fsai.respuestas_fies.ia_binaria       IS 'Inseguridad alimentaria moderada o severa: 1 si fies_nivel >= 2, 0 en caso contrario.';

-- ---------------------------------------------------------------------
-- 5. indicadores_alimentarios
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fsai.indicadores_alimentarios (
    indicador_id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    hogar_uid           char(64) NOT NULL UNIQUE REFERENCES fsai.hogares(hogar_uid) ON DELETE CASCADE,
    gasto_alim_pc       double precision CHECK (gasto_alim_pc IS NULL OR gasto_alim_pc >= 0),
    share_alimentario   double precision CHECK (share_alimentario IS NULL OR share_alimentario >= 0),
    ingreso_pc          double precision,
    gasto_pc            double precision CHECK (gasto_pc IS NULL OR gasto_pc >= 0),
    pobreza_monetaria   smallint CHECK (pobreza_monetaria IN (0,1))
);
COMMENT ON TABLE  fsai.indicadores_alimentarios IS 'Indicadores monetarios y de gasto alimentario per capita del hogar.';
COMMENT ON COLUMN fsai.indicadores_alimentarios.hogar_uid         IS 'Hogar (relacion 1 a 1).';
COMMENT ON COLUMN fsai.indicadores_alimentarios.gasto_alim_pc     IS 'Gasto en alimentos per capita (moneda y deflactor segun diccionario ENAHO).';
COMMENT ON COLUMN fsai.indicadores_alimentarios.share_alimentario IS 'Proporcion gasto_alim_pc / gasto_pc (ley de Engel).';
COMMENT ON COLUMN fsai.indicadores_alimentarios.ingreso_pc        IS 'Ingreso per capita del hogar.';
COMMENT ON COLUMN fsai.indicadores_alimentarios.gasto_pc          IS 'Gasto total per capita del hogar.';
COMMENT ON COLUMN fsai.indicadores_alimentarios.pobreza_monetaria IS 'Indicador de pobreza monetaria oficial (1 = pobre).';

-- ---------------------------------------------------------------------
-- 6. condiciones_socioeconomicas
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fsai.condiciones_socioeconomicas (
    condicion_id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    hogar_uid              char(64) NOT NULL UNIQUE REFERENCES fsai.hogares(hogar_uid) ON DELETE CASCADE,
    n_miembros             smallint CHECK (n_miembros IS NULL OR n_miembros >= 1),
    n_menores5             smallint CHECK (n_menores5 IS NULL OR n_menores5 >= 0),
    n_menores18            smallint CHECK (n_menores18 IS NULL OR n_menores18 >= 0),
    n_adultos_mayores      smallint CHECK (n_adultos_mayores IS NULL OR n_adultos_mayores >= 0),
    edad_jefe              smallint CHECK (edad_jefe IS NULL OR edad_jefe BETWEEN 0 AND 120),
    sexo_jefe              char(1)  CHECK (sexo_jefe IS NULL OR sexo_jefe IN ('H','M')),
    nivel_educ_jefe        smallint CHECK (nivel_educ_jefe IS NULL OR nivel_educ_jefe BETWEEN 0 AND 11),
    asistencia_escolar     double precision CHECK (asistencia_escolar IS NULL OR asistencia_escolar BETWEEN 0 AND 1),
    n_ocupados             smallint CHECK (n_ocupados IS NULL OR n_ocupados >= 0),
    informalidad           double precision CHECK (informalidad IS NULL OR informalidad BETWEEN 0 AND 1),
    desempleo              smallint CHECK (desempleo IS NULL OR desempleo IN (0,1)),
    dependencia_economica  double precision CHECK (dependencia_economica IS NULL OR dependencia_economica >= 0),
    agua_potable           smallint CHECK (agua_potable IS NULL OR agua_potable IN (0,1)),
    saneamiento            smallint CHECK (saneamiento IS NULL OR saneamiento IN (0,1)),
    hacinamiento           double precision CHECK (hacinamiento IS NULL OR hacinamiento >= 0),
    tipo_vivienda          text,
    material_vivienda      text,
    CONSTRAINT ck_menores_coherentes CHECK (
        (n_menores5 IS NULL OR n_menores18 IS NULL OR n_menores5 <= n_menores18) AND
        (n_menores18 IS NULL OR n_miembros IS NULL OR n_menores18 <= n_miembros)
    )
);
COMMENT ON TABLE  fsai.condiciones_socioeconomicas IS 'Variables socioeconomicas, laborales, educativas, demograficas y habitacionales agregadas a nivel de hogar.';
COMMENT ON COLUMN fsai.condiciones_socioeconomicas.hogar_uid             IS 'Hogar (relacion 1 a 1).';
COMMENT ON COLUMN fsai.condiciones_socioeconomicas.n_miembros            IS 'Numero de miembros del hogar.';
COMMENT ON COLUMN fsai.condiciones_socioeconomicas.n_menores5            IS 'Numero de menores de 5 anios.';
COMMENT ON COLUMN fsai.condiciones_socioeconomicas.n_menores18           IS 'Numero de menores de 18 anios.';
COMMENT ON COLUMN fsai.condiciones_socioeconomicas.n_adultos_mayores     IS 'Numero de adultos mayores (umbral de edad segun diccionario [POR VALIDAR]).';
COMMENT ON COLUMN fsai.condiciones_socioeconomicas.edad_jefe             IS 'Edad del jefe(a) del hogar.';
COMMENT ON COLUMN fsai.condiciones_socioeconomicas.sexo_jefe             IS 'Sexo del jefe(a): H o M.';
COMMENT ON COLUMN fsai.condiciones_socioeconomicas.nivel_educ_jefe       IS 'Nivel educativo del jefe(a), escala INEI 0-11.';
COMMENT ON COLUMN fsai.condiciones_socioeconomicas.asistencia_escolar    IS 'Proporcion (0-1) de personas de 6 a 17 anios que asisten a la escuela.';
COMMENT ON COLUMN fsai.condiciones_socioeconomicas.n_ocupados            IS 'Numero de miembros ocupados.';
COMMENT ON COLUMN fsai.condiciones_socioeconomicas.informalidad          IS 'Proporcion (0-1) de ocupados con empleo informal.';
COMMENT ON COLUMN fsai.condiciones_socioeconomicas.desempleo             IS '1 si algun miembro esta desocupado, 0 en caso contrario.';
COMMENT ON COLUMN fsai.condiciones_socioeconomicas.dependencia_economica IS 'Razon n_ocupados / n_miembros (definicion del contrato).';
COMMENT ON COLUMN fsai.condiciones_socioeconomicas.agua_potable          IS '1 si el hogar accede a agua potable por red publica.';
COMMENT ON COLUMN fsai.condiciones_socioeconomicas.saneamiento           IS '1 si dispone de saneamiento adecuado.';
COMMENT ON COLUMN fsai.condiciones_socioeconomicas.hacinamiento          IS 'Personas por habitacion.';
COMMENT ON COLUMN fsai.condiciones_socioeconomicas.tipo_vivienda         IS 'Tipo de vivienda (categoria INEI recodificada).';
COMMENT ON COLUMN fsai.condiciones_socioeconomicas.material_vivienda     IS 'Material predominante de la vivienda (categoria INEI recodificada).';

-- ---------------------------------------------------------------------
-- 7. experimentos
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fsai.experimentos (
    experimento_id       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id           integer NOT NULL REFERENCES fsai.datasets(dataset_id) ON DELETE RESTRICT,
    nombre               text    NOT NULL,
    esquema_validacion   text    NOT NULL CHECK (esquema_validacion IN ('V1','V1c','V2','V3','V4','V5')),
    escenario            text    NOT NULL CHECK (escenario IN ('socio','fies','hibrido')),
    semilla              integer NOT NULL DEFAULT 42,
    config_json          jsonb   NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(config_json) = 'object'),
    iniciado_en          timestamptz NOT NULL DEFAULT now(),
    finalizado_en        timestamptz,
    estado               text    NOT NULL DEFAULT 'pendiente'
                         CHECK (estado IN ('pendiente','en_curso','completado','fallido','cancelado')),
    CONSTRAINT ck_experimentos_fechas CHECK (finalizado_en IS NULL OR finalizado_en >= iniciado_en)
);
COMMENT ON TABLE  fsai.experimentos IS 'Ejecucion de un esquema de validacion sobre un dataset y un escenario. El escenario hibrido es de reconstruccion/clasificacion (fuga de objetivo declarada); el escenario socio es el de prediccion real.';
COMMENT ON COLUMN fsai.experimentos.dataset_id         IS 'Dataset usado.';
COMMENT ON COLUMN fsai.experimentos.nombre             IS 'Nombre del experimento.';
COMMENT ON COLUMN fsai.experimentos.esquema_validacion IS 'Esquema de validacion V1, V1c, V2, V3, V4, V5 (seccion 8.1; V1 = GroupKFold(5) por hogar_uid repetido N_REPETICIONES veces; V1c = agrupado por conglome; V2 = geografico, variante lodo/macro en el nombre; V5 = metricas ponderadas, no particion).';
COMMENT ON COLUMN fsai.experimentos.escenario          IS 'Conjunto de predictores: socio (solo socioeconomicos), fies (solo FIES) o hibrido (ambos).';
COMMENT ON COLUMN fsai.experimentos.semilla            IS 'Semilla aleatoria (SEED = 42 por contrato).';
COMMENT ON COLUMN fsai.experimentos.config_json        IS 'Configuracion completa serializada (particiones, versiones de librerias, umbrales).';
COMMENT ON COLUMN fsai.experimentos.iniciado_en        IS 'Inicio de la ejecucion.';
COMMENT ON COLUMN fsai.experimentos.finalizado_en      IS 'Fin de la ejecucion.';
COMMENT ON COLUMN fsai.experimentos.estado             IS 'pendiente, en_curso, completado, fallido o cancelado.';

-- ---------------------------------------------------------------------
-- 8. modelos
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fsai.modelos (
    modelo_id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experimento_id         bigint NOT NULL REFERENCES fsai.experimentos(experimento_id) ON DELETE CASCADE,
    codigo                 text   NOT NULL CHECK (codigo IN ('M1','M2','M3','M4','M5')),
    hiperparametros        jsonb  NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(hiperparametros) = 'object'),
    n_trials               integer CHECK (n_trials IS NULL OR n_trials >= 0),
    mejor_valor_objetivo   double precision,
    ruta_artefacto         text,
    creado_en              timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_modelos_experimento_codigo UNIQUE (experimento_id, codigo)
);
COMMENT ON TABLE  fsai.modelos IS 'Modelo M1..M5 entrenado dentro de un experimento, con hiperparametros optimizados con Optuna.';
COMMENT ON COLUMN fsai.modelos.experimento_id       IS 'Experimento al que pertenece.';
COMMENT ON COLUMN fsai.modelos.codigo               IS 'M1 Logistica ordinal, M2 Random Forest, M3 CatBoost, M4 Stacking, M5 Autoencoder + XGBoost.';
COMMENT ON COLUMN fsai.modelos.hiperparametros      IS 'Mejores hiperparametros (JSONB).';
COMMENT ON COLUMN fsai.modelos.n_trials             IS 'Numero de ensayos Optuna ejecutados.';
COMMENT ON COLUMN fsai.modelos.mejor_valor_objetivo IS 'Mejor F1-macro ponderado en validacion interna (criterio Optuna).';
COMMENT ON COLUMN fsai.modelos.ruta_artefacto       IS 'Ruta relativa del artefacto serializado (no se guardan binarios en la base).';
COMMENT ON COLUMN fsai.modelos.creado_en            IS 'Marca de tiempo de registro.';

-- ---------------------------------------------------------------------
-- 9. predicciones
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fsai.predicciones (
    prediccion_id  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    modelo_id      bigint   NOT NULL REFERENCES fsai.modelos(modelo_id) ON DELETE CASCADE,
    hogar_uid      char(64) NOT NULL REFERENCES fsai.hogares(hogar_uid) ON DELETE CASCADE,
    fold           smallint CHECK (fold IS NULL OR fold >= 0),
    y_real         smallint CHECK (y_real BETWEEN 0 AND 3),
    y_pred         smallint NOT NULL CHECK (y_pred BETWEEN 0 AND 3),
    proba_json     jsonb    CHECK (proba_json IS NULL OR jsonb_typeof(proba_json) = 'object'),
    CONSTRAINT uq_predicciones_modelo_hogar_fold UNIQUE (modelo_id, hogar_uid, fold)
);
COMMENT ON TABLE  fsai.predicciones IS 'Predicciones fuera de muestra por hogar y modelo (una por fold de test).';
COMMENT ON COLUMN fsai.predicciones.modelo_id  IS 'Modelo que genero la prediccion.';
COMMENT ON COLUMN fsai.predicciones.hogar_uid  IS 'Hogar predicho.';
COMMENT ON COLUMN fsai.predicciones.fold       IS 'Indice del bloque de test: pliegue (0..N_SPLITS-1) de la repeticion 0, o particion LODO/macro/area.';
COMMENT ON COLUMN fsai.predicciones.y_real     IS 'Nivel FIES observado (0-3).';
COMMENT ON COLUMN fsai.predicciones.y_pred     IS 'Nivel FIES predicho (0-3).';
COMMENT ON COLUMN fsai.predicciones.proba_json IS 'Probabilidades por clase, formato {"0":p0,"1":p1,"2":p2,"3":p3}.';

-- ---------------------------------------------------------------------
-- 10. metricas
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fsai.metricas (
    metrica_id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    modelo_id           bigint NOT NULL REFERENCES fsai.modelos(modelo_id) ON DELETE CASCADE,
    esquema_validacion  text   NOT NULL CHECK (esquema_validacion IN ('V1','V1c','V2','V3','V4','V5')),
    ponderada           boolean NOT NULL DEFAULT false,
    nombre_metrica      text   NOT NULL,
    valor               double precision,
    ic_inf              double precision,
    ic_sup              double precision,
    fold                smallint CHECK (fold IS NULL OR fold >= 0),
    estrato             text,
    CONSTRAINT ck_metricas_ic CHECK (ic_inf IS NULL OR ic_sup IS NULL OR ic_inf <= ic_sup)
);
COMMENT ON TABLE  fsai.metricas IS 'Metricas de desempeno (globales, por fold o por estrato) con intervalos de confianza bootstrap. Solo se almacenan valores producidos por la ejecucion.';
COMMENT ON COLUMN fsai.metricas.modelo_id          IS 'Modelo evaluado.';
COMMENT ON COLUMN fsai.metricas.esquema_validacion IS 'Esquema de validacion (V1, V1c, V2, V3, V4, V5) bajo el cual se calculo.';
COMMENT ON COLUMN fsai.metricas.ponderada          IS 'Verdadero si se ponderó con factor_expansion.';
COMMENT ON COLUMN fsai.metricas.nombre_metrica     IS 'Nombre (accuracy, balanced_accuracy, f1_macro, roc_auc_ovr, pr_auc_macro, mcc, brier, recall_severa, ece, ...).';
COMMENT ON COLUMN fsai.metricas.valor              IS 'Valor puntual de la metrica.';
COMMENT ON COLUMN fsai.metricas.ic_inf             IS 'Limite inferior del IC 95 % (bootstrap estratificado).';
COMMENT ON COLUMN fsai.metricas.ic_sup             IS 'Limite superior del IC 95 %.';
COMMENT ON COLUMN fsai.metricas.fold               IS 'Indice de bloque de validacion (en V1/V1c: rep*N_SPLITS+pliegue, 0..19; el campo estrato lo etiqueta como repR_foldK); NULL para la metrica agregada.';
COMMENT ON COLUMN fsai.metricas.estrato            IS 'Estrato (por ejemplo area=R, departamento=Cusco); NULL para la metrica global.';

-- ---------------------------------------------------------------------
-- 11. explicaciones
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fsai.explicaciones (
    explicacion_id  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    modelo_id       bigint NOT NULL REFERENCES fsai.modelos(modelo_id) ON DELETE CASCADE,
    hogar_uid       char(64) REFERENCES fsai.hogares(hogar_uid) ON DELETE CASCADE,
    tipo            text   NOT NULL CHECK (tipo IN ('shap_global','shap_local','lime','permutation','pdp','ale')),
    contribuciones  jsonb  NOT NULL CHECK (jsonb_typeof(contribuciones) = 'object' AND contribuciones ? 'features'),
    creado_en       timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT ck_explicacion_local_requiere_hogar CHECK (tipo NOT IN ('shap_local','lime') OR hogar_uid IS NOT NULL)
);
COMMENT ON TABLE  fsai.explicaciones IS 'Explicaciones XAI. contribuciones tiene la forma {"clase":3,"base_value":0.0,"features":{"nombre_variable":valor,...}}; son asociaciones predictivas, no efectos causales.';
COMMENT ON COLUMN fsai.explicaciones.modelo_id      IS 'Modelo explicado.';
COMMENT ON COLUMN fsai.explicaciones.hogar_uid      IS 'Hogar explicado; NULL para explicaciones globales.';
COMMENT ON COLUMN fsai.explicaciones.tipo           IS 'shap_global, shap_local, lime, permutation, pdp o ale.';
COMMENT ON COLUMN fsai.explicaciones.contribuciones IS 'JSONB con llave obligatoria "features": {nombre: valor numerico}. En shap_global el valor es el promedio de |SHAP|; en shap_local es el SHAP firmado.';
COMMENT ON COLUMN fsai.explicaciones.creado_en      IS 'Marca de tiempo de registro.';

-- ---------------------------------------------------------------------
-- 12. auditoria
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fsai.auditoria (
    auditoria_id  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    usuario       text   NOT NULL DEFAULT session_user,
    accion        text   NOT NULL,
    entidad       text   NOT NULL,
    entidad_id    text,
    detalle       jsonb  CHECK (detalle IS NULL OR jsonb_typeof(detalle) IN ('object','array')),
    ocurrido_en   timestamptz NOT NULL DEFAULT now()
);
COMMENT ON TABLE  fsai.auditoria IS 'Bitacora de acciones (solo insercion). No registra datos personales ni contenido de hogares.';
COMMENT ON COLUMN fsai.auditoria.usuario     IS 'Rol de base de datos o usuario de aplicacion que ejecuto la accion.';
COMMENT ON COLUMN fsai.auditoria.accion      IS 'INSERT, UPDATE, DELETE, EXPORT, TRAIN u otra accion de aplicacion.';
COMMENT ON COLUMN fsai.auditoria.entidad     IS 'Tabla o entidad afectada.';
COMMENT ON COLUMN fsai.auditoria.entidad_id  IS 'Identificador de la entidad afectada (texto).';
COMMENT ON COLUMN fsai.auditoria.detalle     IS 'Detalle adicional en JSONB.';
COMMENT ON COLUMN fsai.auditoria.ocurrido_en IS 'Marca de tiempo del evento.';

-- ---------------------------------------------------------------------
-- Indices
-- ---------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS ix_hogares_dataset            ON fsai.hogares (dataset_id);
CREATE INDEX IF NOT EXISTS ix_hogares_anio_depto_area    ON fsai.hogares (anio, departamento, area);
CREATE INDEX IF NOT EXISTS ix_hogares_ubigeo             ON fsai.hogares (ubigeo);
CREATE INDEX IF NOT EXISTS ix_hogares_conglome_hash      ON fsai.hogares (conglome_hash);
CREATE INDEX IF NOT EXISTS ix_hogares_rurales            ON fsai.hogares (departamento) WHERE area = 'R';
CREATE INDEX IF NOT EXISTS ix_personas_hogar             ON fsai.personas_anonimizadas (hogar_uid);
CREATE INDEX IF NOT EXISTS ix_personas_jefes             ON fsai.personas_anonimizadas (hogar_uid) WHERE es_jefe;
CREATE INDEX IF NOT EXISTS ix_fies_nivel                 ON fsai.respuestas_fies (fies_nivel);
CREATE INDEX IF NOT EXISTS ix_fies_ia_positiva           ON fsai.respuestas_fies (hogar_uid) WHERE ia_binaria = 1;
CREATE INDEX IF NOT EXISTS ix_fies_severa                ON fsai.respuestas_fies (hogar_uid) WHERE fies_nivel = 3;
CREATE INDEX IF NOT EXISTS ix_indicadores_pobres         ON fsai.indicadores_alimentarios (hogar_uid) WHERE pobreza_monetaria = 1;
CREATE INDEX IF NOT EXISTS ix_experimentos_dataset       ON fsai.experimentos (dataset_id);
CREATE INDEX IF NOT EXISTS ix_experimentos_esq_esc       ON fsai.experimentos (esquema_validacion, escenario);
CREATE INDEX IF NOT EXISTS ix_experimentos_en_curso      ON fsai.experimentos (iniciado_en) WHERE estado = 'en_curso';
CREATE INDEX IF NOT EXISTS ix_experimentos_config_gin    ON fsai.experimentos USING gin (config_json jsonb_path_ops);
CREATE INDEX IF NOT EXISTS ix_modelos_experimento        ON fsai.modelos (experimento_id);
CREATE INDEX IF NOT EXISTS ix_modelos_hiperparam_gin     ON fsai.modelos USING gin (hiperparametros jsonb_path_ops);
CREATE INDEX IF NOT EXISTS ix_pred_modelo_fold           ON fsai.predicciones (modelo_id, fold);
CREATE INDEX IF NOT EXISTS ix_pred_hogar                 ON fsai.predicciones (hogar_uid);
CREATE INDEX IF NOT EXISTS ix_pred_errores               ON fsai.predicciones (modelo_id, hogar_uid) WHERE y_real IS DISTINCT FROM y_pred;
CREATE INDEX IF NOT EXISTS ix_pred_proba_gin             ON fsai.predicciones USING gin (proba_json jsonb_path_ops);
CREATE INDEX IF NOT EXISTS ix_metricas_modelo_nombre     ON fsai.metricas (modelo_id, esquema_validacion, nombre_metrica);
CREATE INDEX IF NOT EXISTS ix_metricas_globales          ON fsai.metricas (modelo_id, nombre_metrica) WHERE fold IS NULL AND estrato IS NULL;
CREATE INDEX IF NOT EXISTS ix_expl_modelo_tipo           ON fsai.explicaciones (modelo_id, tipo);
CREATE INDEX IF NOT EXISTS ix_expl_hogar                 ON fsai.explicaciones (hogar_uid) WHERE hogar_uid IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_expl_contrib_gin           ON fsai.explicaciones USING gin (contribuciones);
CREATE INDEX IF NOT EXISTS ix_auditoria_entidad          ON fsai.auditoria (entidad, entidad_id, ocurrido_en DESC);
CREATE INDEX IF NOT EXISTS ix_auditoria_detalle_gin      ON fsai.auditoria USING gin (detalle jsonb_path_ops);

-- ---------------------------------------------------------------------
-- Trigger de auditoria (opcional): registra cambios en tablas de gobernanza
-- (no se aplica a hogares/personas para no duplicar datos ni crecer en exceso)
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION fsai.fn_auditar() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = fsai, pg_temp AS $$
DECLARE
    v_pk   text := TG_ARGV[0];
    v_fila jsonb;
    v_usr  text := COALESCE(NULLIF(current_setting('fsai.usuario', true), ''), session_user);
BEGIN
    IF TG_OP = 'DELETE' THEN v_fila := to_jsonb(OLD); ELSE v_fila := to_jsonb(NEW); END IF;
    INSERT INTO fsai.auditoria (usuario, accion, entidad, entidad_id, detalle)
    VALUES (v_usr, TG_OP, TG_TABLE_NAME, v_fila ->> v_pk,
            jsonb_build_object('esquema', TG_TABLE_SCHEMA, 'nivel', TG_LEVEL));
    RETURN NULL;
END;
$$;
COMMENT ON FUNCTION fsai.fn_auditar() IS 'Trigger AFTER de auditoria: inserta usuario, accion y llave primaria afectada (sin copiar el contenido de la fila).';

DROP TRIGGER IF EXISTS trg_aud_datasets ON fsai.datasets;
CREATE TRIGGER trg_aud_datasets AFTER INSERT OR UPDATE OR DELETE ON fsai.datasets
    FOR EACH ROW EXECUTE FUNCTION fsai.fn_auditar('dataset_id');

DROP TRIGGER IF EXISTS trg_aud_experimentos ON fsai.experimentos;
CREATE TRIGGER trg_aud_experimentos AFTER INSERT OR UPDATE OR DELETE ON fsai.experimentos
    FOR EACH ROW EXECUTE FUNCTION fsai.fn_auditar('experimento_id');

DROP TRIGGER IF EXISTS trg_aud_modelos ON fsai.modelos;
CREATE TRIGGER trg_aud_modelos AFTER INSERT OR UPDATE OR DELETE ON fsai.modelos
    FOR EACH ROW EXECUTE FUNCTION fsai.fn_auditar('modelo_id');

-- La bitacora es de solo insercion: se bloquean UPDATE y DELETE
CREATE OR REPLACE FUNCTION fsai.fn_auditoria_inmutable() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'fsai.auditoria es de solo insercion (operacion % bloqueada)', TG_OP;
END;
$$;

DROP TRIGGER IF EXISTS trg_auditoria_inmutable ON fsai.auditoria;
CREATE TRIGGER trg_auditoria_inmutable BEFORE UPDATE OR DELETE ON fsai.auditoria
    FOR EACH ROW EXECUTE FUNCTION fsai.fn_auditoria_inmutable();

-- ---------------------------------------------------------------------
-- Staging opcional: validacion del crudo INEI (codigos 1-7) antes de recodificar
-- ---------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS fsai_stg;
CREATE TABLE IF NOT EXISTS fsai_stg.respuestas_fies_crudo (
    hogar_uid  char(64) PRIMARY KEY CHECK (hogar_uid ~ '^[0-9a-f]{64}$'),
    dataset_id integer  NOT NULL REFERENCES fsai.datasets(dataset_id) ON DELETE CASCADE,
    q1_raw smallint CHECK (q1_raw BETWEEN 1 AND 7),
    q2_raw smallint CHECK (q2_raw BETWEEN 1 AND 7),
    q3_raw smallint CHECK (q3_raw BETWEEN 1 AND 7),
    q4_raw smallint CHECK (q4_raw BETWEEN 1 AND 7),
    q5_raw smallint CHECK (q5_raw BETWEEN 1 AND 7),
    q6_raw smallint CHECK (q6_raw BETWEEN 1 AND 7),
    q7_raw smallint CHECK (q7_raw BETWEEN 1 AND 7),
    q8_raw smallint CHECK (q8_raw BETWEEN 1 AND 7),
    cargado_en timestamptz NOT NULL DEFAULT now()
);
COMMENT ON TABLE fsai_stg.respuestas_fies_crudo IS 'Staging opcional con la codificacion cruda INEI (1-7) de los 8 items FIES; la recodificacion a 0/1 (1 = si, 2 = no, resto = NULL [POR VALIDAR] segun diccionario) se hace en data_loader/fies.py y esta tabla se purga tras la validacion.';
COMMENT ON COLUMN fsai_stg.respuestas_fies_crudo.q1_raw IS 'Codigo crudo INEI del item 1 (1-7); items q2_raw..q8_raw analogos.';

COMMIT;
