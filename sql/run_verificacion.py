"""Verifica 01-03 .sql: parser real de PostgreSQL (pglast) y, si es posible, ejecucion en un PostgreSQL embebido (pgserver).
Los datos de humo son SINTETICOS y solo sirven para comprobar que el SQL ejecuta; no son resultados."""
import sys, pathlib
base = pathlib.Path(__file__).parent
scratch = sys.argv[1]
sys.path[:0] = [scratch + "/pylib", scratch + "/pylib2"]
import pglast
for f in ["01_ddl_fsai.sql", "02_roles_grants.sql", "03_consultas.sql"]:
    stmts = pglast.parse_sql((base / f).read_text(encoding="utf-8"))
    print("PARSE OK", f, len(stmts), "sentencias")
import pgserver, tempfile
srv = pgserver.get_server(tempfile.mkdtemp(), cleanup_mode="stop")
print(srv.psql("select version();"))
def run(sql): return srv.psql(sql)
print(run("select encode(sha256(convert_to('a|b|c|salt','UTF8')),'hex');"))
for f in ["01_ddl_fsai.sql", "01_ddl_fsai.sql", "02_roles_grants.sql"]:
    out = run((base / f).read_text(encoding="utf-8")); print("EXEC", f, "->", out.strip().splitlines()[-1] if out.strip() else "ok")
seed = """
INSERT INTO fsai.datasets(nombre,fuente,anio) VALUES ('humo','sintetico',2025);
INSERT INTO fsai.hogares(hogar_uid,dataset_id,area,region_natural,factor_expansion,anio,departamento)
 SELECT encode(sha256(convert_to('c'||i||'|v|h|salt','UTF8')),'hex'),1,CASE WHEN i%2=0 THEN 'R' ELSE 'U' END,'sierra',100+i,2025,CASE WHEN i%3=0 THEN 'CUSCO' ELSE 'LIMA' END FROM generate_series(1,12) i;
INSERT INTO fsai.respuestas_fies(hogar_uid,fies_nivel,ia_binaria) SELECT hogar_uid,(row_number() over ())%4, CASE WHEN (row_number() over ())%4>=2 THEN 1 ELSE 0 END FROM fsai.hogares;
INSERT INTO fsai.condiciones_socioeconomicas(hogar_uid,n_miembros) SELECT hogar_uid,4 FROM fsai.hogares;
INSERT INTO fsai.experimentos(dataset_id,nombre,esquema_validacion,escenario) VALUES (1,'humo','V1','socio');
INSERT INTO fsai.modelos(experimento_id,codigo) VALUES (1,'M1');
INSERT INTO fsai.predicciones(modelo_id,hogar_uid,fold,y_real,y_pred,proba_json) SELECT 1,hogar_uid,0,(row_number() over ())%4,2,'{"0":0.1,"1":0.2,"2":0.5,"3":0.2}' FROM fsai.hogares;
INSERT INTO fsai.metricas(modelo_id,esquema_validacion,ponderada,nombre_metrica,valor,ic_inf,ic_sup) VALUES (1,'V1',true,'f1_macro',0.5,0.4,0.6);
INSERT INTO fsai.explicaciones(modelo_id,tipo,contribuciones) VALUES (1,'shap_global','{"clase":3,"features":{"ingreso_pc":0.3,"agua_potable":-0.1}}');
"""
print(run(seed))
for q in (base / "03_consultas.sql").read_text(encoding="utf-8").split("-- Q")[1:]:
    print("--- Q" + q.splitlines()[0][:50]); print(run(q.split("\n",1)[1]))
print(run("select entidad,accion,count(*) from fsai.auditoria group by 1,2 order by 1;"))
for bad in ["UPDATE fsai.auditoria SET accion='x';",
            "INSERT INTO fsai.respuestas_fies(hogar_uid,fies_q1) SELECT hogar_uid,2 FROM fsai.hogares LIMIT 1;",
            "INSERT INTO fsai.hogares(hogar_uid,dataset_id,area,factor_expansion,anio) VALUES (repeat('a',64),1,'X',1,2025);",
            "INSERT INTO fsai.respuestas_fies(hogar_uid,fies_nivel,ia_binaria) SELECT hogar_uid,1,1 FROM fsai.hogares ORDER BY 1 OFFSET 5 LIMIT 1;"]:
    try: print("DEBIO FALLAR:", run(bad))
    except Exception as e: print("RECHAZADO OK:", str(e).strip().splitlines()[0][:110])
