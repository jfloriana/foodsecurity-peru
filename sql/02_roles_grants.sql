-- Roles de grupo (NOLOGIN) y privilegios minimos para el esquema fsai.
-- Los usuarios reales se crean aparte (con contrasena fuera del repositorio) y se agregan a un rol de grupo.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fsai_lector')   THEN CREATE ROLE fsai_lector   NOLOGIN; END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fsai_analista') THEN CREATE ROLE fsai_analista NOLOGIN; END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fsai_etl')      THEN CREATE ROLE fsai_etl      NOLOGIN; END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fsai_auditor')  THEN CREATE ROLE fsai_auditor  NOLOGIN; END IF;
END
$$;

REVOKE ALL ON SCHEMA fsai, fsai_stg FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA fsai, fsai_stg FROM PUBLIC;

GRANT USAGE ON SCHEMA fsai TO fsai_lector, fsai_analista, fsai_etl, fsai_auditor;
GRANT USAGE ON SCHEMA fsai_stg TO fsai_etl;

-- Lector: solo lectura de todo salvo la bitacora
GRANT SELECT ON fsai.datasets, fsai.hogares, fsai.personas_anonimizadas, fsai.respuestas_fies,
                fsai.indicadores_alimentarios, fsai.condiciones_socioeconomicas,
                fsai.experimentos, fsai.modelos, fsai.predicciones, fsai.metricas, fsai.explicaciones
      TO fsai_lector;

-- Analista: lector + escritura de resultados de modelado (nunca de datos de hogares)
GRANT fsai_lector TO fsai_analista;
GRANT INSERT, UPDATE ON fsai.experimentos, fsai.modelos TO fsai_analista;
GRANT INSERT ON fsai.predicciones, fsai.metricas, fsai.explicaciones TO fsai_analista;

-- ETL: carga de datos anonimizados y staging
GRANT SELECT, INSERT, UPDATE ON fsai.datasets, fsai.hogares, fsai.personas_anonimizadas, fsai.respuestas_fies,
                                fsai.indicadores_alimentarios, fsai.condiciones_socioeconomicas TO fsai_etl;
GRANT SELECT, INSERT, UPDATE, DELETE ON fsai_stg.respuestas_fies_crudo TO fsai_etl;

-- Auditor: lectura de la bitacora
GRANT SELECT ON fsai.auditoria TO fsai_auditor;

-- Secuencias necesarias para INSERT (identidades y serial)
GRANT USAGE ON ALL SEQUENCES IN SCHEMA fsai TO fsai_analista, fsai_etl;

-- Las funciones de auditoria las ejecutan los triggers (SECURITY DEFINER); no se otorga EXECUTE directo
REVOKE ALL ON FUNCTION fsai.fn_auditar() FROM PUBLIC;
