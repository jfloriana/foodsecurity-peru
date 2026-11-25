"""Catálogo de módulos de FoodSecurity AI y ruta de análisis de investigación.

Sin dependencias de Streamlit para poder importarse desde el enrutador (app.py), las páginas y las pruebas.
"""
from __future__ import annotations

# (clave de acceso, sección, título, descripción de una línea, archivo de página, icono Material)
MODULOS: list[tuple[str, str, str, str, str, str]] = [
    ("importacion_enaho", "Datos", "Carga de datos", "Cargue microdatos ENAHO, edite el diccionario y valide los grupos", "pages/02_Importacion_ENAHO.py", ":material/upload_file:"),
    ("procesamiento_fies", "Datos", "Procesamiento FIES", "Puntaje crudo, niveles de inseguridad, escala Rasch y calidad", "pages/03_Procesamiento_FIES.py", ":material/tune:"),
    ("eda", "Datos", "Exploración", "Distribuciones, asociaciones y calidad de los datos", "pages/04_EDA.py", ":material/query_stats:"),
    ("vulnerabilidad", "Datos", "Vulnerabilidad", "Perfil descriptivo de hogares vulnerables (menores de 5 años, adultos mayores, pobreza) e indicadores por nivel FIES", "pages/05_Vulnerabilidad.py", ":material/home_health:"),
    ("entrenamiento", "Modelado", "Entrenamiento", "Entrene los modelos M1-M5 con validación cruzada anidada", "pages/06_Entrenamiento.py", ":material/model_training:"),
    ("optuna", "Modelado", "Optimización", "Búsqueda automática de hiperparámetros con seguimiento", "pages/07_Optuna.py", ":material/tune:"),
    ("comparacion", "Evaluación e interpretación", "Comparación estadística", "Evalúe la Prueba una vez y compare modelos con pruebas estadísticas", "pages/08_Comparacion_Estadistica.py", ":material/compare_arrows:"),
    ("shap", "Evaluación e interpretación", "Interpretabilidad", "Qué variables explican las predicciones (SHAP, LIME, PDP/ALE)", "pages/09_SHAP.py", ":material/lightbulb:"),
    ("territorial", "Evaluación e interpretación", "Análisis territorial", "Prevalencia por departamento, área y región natural", "pages/10_Territorial.py", ":material/map:"),
    ("equidad", "Evaluación e interpretación", "Equidad y sesgos", "Brechas de clasificación entre grupos de hogares", "pages/11_Equidad.py", ":material/balance:"),
    ("dashboard", "Resultados y despliegue", "Panorama", "Resumen de la prevalencia de inseguridad alimentaria", "pages/01_Dashboard.py", ":material/dashboard:"),
    ("reportes", "Resultados y despliegue", "Reportes", "Informe científico en PDF, Word y Excel con figuras a 600 DPI", "pages/12_Reportes.py", ":material/description:"),
    ("inferencia", "Resultados y despliegue", "Inferencia", "Predicción para un hogar o un lote de hogares", "pages/13_Inferencia.py", ":material/psychology:"),
    ("glosario", "Evaluación e interpretación", "Glosario", "Qué significan las métricas y métodos, cómo leerlos y sus límites éticos; diccionario de variables", "pages/14_Glosario.py", ":material/menu_book:"),
    ("comparacion_corridas", "Evaluación e interpretación", "Comparación de corridas", "Compare experimentos entre corridas (F1-macro, AUC, Brier) con avisos de comparabilidad", "pages/15_Comparacion_Corridas.py", ":material/difference:"),
    ("fuentes_datos", "Datos", "Fuentes de datos", "Origen, licencia, año y estado de cada fuente pública usada; limitaciones y fuentes que fallaron", "pages/17_Fuentes_Datos.py", ":material/database:"),
    ("bitacora", "Resultados y despliegue", "Bitácora de auditoría", "Consulta y exportación de las acciones registradas en fsai.auditoria (solo administración)", "pages/16_Bitacora.py", ":material/history:"),
]
POR_CLAVE = {m[0]: m for m in MODULOS}

# Roles de los módulos añadidos en V2 (auth.ACCESO_PAGINAS es la tabla de referencia; aquí se completa sin editar auth.py).
ACCESO_V2: dict[str, tuple[str, ...]] = {
    "glosario": ("admin", "analista", "lector"),
    "comparacion_corridas": ("admin", "analista", "lector"),
    "bitacora": ("admin",),
    "fuentes_datos": ("admin", "analista", "lector"),
}


def registrar_accesos() -> None:
    """Agrega a ``auth.ACCESO_PAGINAS`` los módulos de ``ACCESO_V2`` que falten (idempotente; no sobrescribe)."""
    import auth  # import diferido: auth depende de Streamlit

    for clave, roles in ACCESO_V2.items():
        auth.ACCESO_PAGINAS.setdefault(clave, roles)

# Ruta de análisis (orden recomendado de trabajo). El «Panorama» es transversal y no forma parte de la ruta.
RUTA: list[str] = [
    "importacion_enaho", "procesamiento_fies", "eda", "vulnerabilidad", "entrenamiento", "optuna",
    "comparacion", "shap", "territorial", "equidad", "reportes", "inferencia",
]
# Pasos que no bloquean a los siguientes.
RUTA_OPCIONAL = {"optuna", "vulnerabilidad", "eda"}


def titulo(clave: str) -> str:
    return POR_CLAVE[clave][2]


def archivo(clave: str) -> str:
    return POR_CLAVE[clave][4]
