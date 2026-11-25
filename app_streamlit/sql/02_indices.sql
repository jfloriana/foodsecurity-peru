-- Índices de apoyo a las consultas de la aplicación (revisar contra el DDL de la sección 11
-- para no duplicar índices ya creados con otro nombre).
CREATE INDEX IF NOT EXISTS ix_hogares_anio_depto ON fsai.hogares (anio, departamento);
CREATE INDEX IF NOT EXISTS ix_hogares_area ON fsai.hogares (area);
CREATE INDEX IF NOT EXISTS ix_experimentos_inicio ON fsai.experimentos (iniciado_en DESC);
CREATE INDEX IF NOT EXISTS ix_experimentos_nombre ON fsai.experimentos (nombre);
CREATE INDEX IF NOT EXISTS ix_modelos_experimento ON fsai.modelos (experimento_id);
CREATE INDEX IF NOT EXISTS ix_predicciones_modelo ON fsai.predicciones (modelo_id, hogar_uid);
CREATE INDEX IF NOT EXISTS ix_metricas_modelo_nombre ON fsai.metricas (modelo_id, nombre_metrica);
CREATE INDEX IF NOT EXISTS ix_explicaciones_modelo_tipo ON fsai.explicaciones (modelo_id, tipo, hogar_uid);
CREATE INDEX IF NOT EXISTS ix_auditoria_ocurrido ON fsai.auditoria (ocurrido_en DESC);
