-- Consultas analiticas de ejemplo (fsai). Cada bloque es independiente.

-- Q1. Prevalencia ponderada de IA moderada/severa por departamento (hogares y personas)
WITH par AS (SELECT 2025::smallint AS anio)
SELECT h.departamento,
       count(*)                                             AS n_hogares_muestra,
       round(sum(h.factor_expansion)::numeric, 0)           AS hogares_expandidos,
       round((100 * sum(h.factor_expansion * r.ia_binaria) / sum(h.factor_expansion))::numeric, 2)
                                                            AS prev_ia_mod_sev_hogares_pct,
       round((100 * sum(h.factor_expansion * c.n_miembros * r.ia_binaria)
                  / NULLIF(sum(h.factor_expansion * c.n_miembros), 0))::numeric, 2)
                                                            AS prev_ia_mod_sev_personas_pct
FROM fsai.hogares h
JOIN par ON h.anio = par.anio
JOIN fsai.respuestas_fies r ON r.hogar_uid = h.hogar_uid
JOIN fsai.condiciones_socioeconomicas c ON c.hogar_uid = h.hogar_uid
WHERE r.ia_binaria IS NOT NULL
GROUP BY h.departamento
ORDER BY prev_ia_mod_sev_hogares_pct DESC;

-- Q2. Distribucion de predicciones por modelo (clase predicha, proporcion y confianza media)
SELECT mo.modelo_id, mo.codigo, ex.escenario, ex.esquema_validacion,
       p.y_pred,
       count(*)                                                        AS n,
       round((100.0 * count(*) / sum(count(*)) OVER (PARTITION BY mo.modelo_id))::numeric, 2) AS pct,
       round(avg(GREATEST((p.proba_json->>'0')::float8, (p.proba_json->>'1')::float8,
                          (p.proba_json->>'2')::float8, (p.proba_json->>'3')::float8))::numeric, 4)
                                                                       AS confianza_media
FROM fsai.predicciones p
JOIN fsai.modelos mo ON mo.modelo_id = p.modelo_id
JOIN fsai.experimentos ex ON ex.experimento_id = mo.experimento_id
GROUP BY mo.modelo_id, mo.codigo, ex.escenario, ex.esquema_validacion, p.y_pred
ORDER BY mo.modelo_id, p.y_pred;

-- Q3. Importancia SHAP agregada: top-20 variables por |SHAP| medio para un modelo
-- Estructura esperada: {"clase":3,"base_value":0.12,"features":{"ingreso_pc":0.31,"agua_potable":-0.08,...}}
WITH par AS (SELECT 1::bigint AS modelo_id)      -- reemplazar por el modelo de interes
SELECT f.key                                   AS variable,
       round(avg(abs(f.value::text::float8))::numeric, 5) AS shap_abs_medio,
       count(*)                                AS n_explicaciones
FROM fsai.explicaciones e
JOIN par ON e.modelo_id = par.modelo_id
CROSS JOIN LATERAL jsonb_each(e.contribuciones -> 'features') AS f(key, value)
WHERE e.tipo IN ('shap_global', 'shap_local')
  AND jsonb_typeof(f.value) = 'number'
GROUP BY f.key
ORDER BY shap_abs_medio DESC
LIMIT 20;

-- Q4. Metricas por modelo y esquema de validacion con IC (globales, ponderadas)
SELECT mo.codigo, ex.escenario, m.esquema_validacion, m.nombre_metrica,
       round(m.valor::numeric, 4)  AS valor,
       round(m.ic_inf::numeric, 4) AS ic_inf,
       round(m.ic_sup::numeric, 4) AS ic_sup
FROM fsai.metricas m
JOIN fsai.modelos mo ON mo.modelo_id = m.modelo_id
JOIN fsai.experimentos ex ON ex.experimento_id = mo.experimento_id
WHERE m.fold IS NULL AND m.estrato IS NULL AND m.ponderada
  AND m.nombre_metrica IN ('f1_macro', 'balanced_accuracy', 'recall_severa', 'roc_auc_ovr', 'brier')
ORDER BY m.nombre_metrica, m.esquema_validacion, ex.escenario, mo.codigo;

-- Q5. Brecha urbano-rural: prevalencia ponderada observada y sensibilidad (recall) de la clase severa predicha
WITH prev AS (
    SELECT h.area,
           sum(h.factor_expansion * r.ia_binaria) / sum(h.factor_expansion) AS prev_ia
    FROM fsai.hogares h
    JOIN fsai.respuestas_fies r USING (hogar_uid)
    WHERE r.ia_binaria IS NOT NULL
    GROUP BY h.area
), rec AS (
    SELECT p.modelo_id, h.area,
           COALESCE(sum(h.factor_expansion) FILTER (WHERE p.y_real = 3 AND p.y_pred = 3), 0)
           /NULLIF(sum(h.factor_expansion) FILTER (WHERE p.y_real = 3), 0) AS recall_severa
    FROM fsai.predicciones p
    JOIN fsai.hogares h USING (hogar_uid)
    GROUP BY p.modelo_id, h.area
)
SELECT pr.area,
       round((100 * pr.prev_ia)::numeric, 2) AS prev_ia_pct,
       round((100 * (pr.prev_ia - min(pr.prev_ia) OVER ()))::numeric, 2) AS brecha_vs_minimo_pp,
       rec.modelo_id,
       round(rec.recall_severa::numeric, 4) AS recall_severa_ponderado
FROM prev pr
LEFT JOIN rec ON rec.area = pr.area
ORDER BY rec.modelo_id, pr.area;

-- Q6. Control de anonimato: celdas distrito x area con menos de k hogares (k = 5 [POR VALIDAR]); suprimir o agregar a provincia antes de publicar
SELECT h.ubigeo, h.area, count(*) AS n_hogares
FROM fsai.hogares h
GROUP BY h.ubigeo, h.area
HAVING count(*) < 5
ORDER BY n_hogares, h.ubigeo;
