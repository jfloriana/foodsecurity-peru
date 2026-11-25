"""etiquetas.py - Diccionario central de variables (capa de PRESENTACIÓN).

Una sola fuente para: nombre legible en español (es-419), unidad, descripción corta y tipo de cada variable del dataset
analítico ENAHO 2025. Se usa en la interfaz, gráficos, tablas y en los informes (PDF/Word/Excel).

REGLA: estas etiquetas NUNCA se usan para renombrar columnas de entrenamiento o de inferencia; solo se aplican al
mostrar. El código original se conserva (tooltip, ayuda y anexo «Diccionario de variables»).

Las descripciones salen de lo verificable en el código de construcción del dataset (``fuentes_publicas.py``), de
``datasets.DESCRIPCIONES`` y de ``despliegue_local/INFORME_DATOS_PUBLICOS.md``. Lo que no se pudo verificar se marca
``[POR VALIDAR]``; no se inventan definiciones.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from fsai.catalogo_campos import FIES_LIMITACION, FIES_PREGUNTAS, MATERIAL_PAREDES, TIPO_VIVIENDA

POR_VALIDAR = "[POR VALIDAR]"


def _v(nombre: str, unidad: str, descripcion: str, tipo: str, validar: str = "") -> dict[str, str]:
    return {"nombre": nombre, "unidad": unidad, "descripcion": descripcion, "tipo": tipo, "por_validar": validar}


_FIES_ITEM = "Ítem {i} de la escala FIES (P130_{i} de la ENAHO, mes anterior, por falta de dinero u otras formas de conseguir alimentos): {q}. 1 = sí, 0 = no."
_FIES_VAL = FIES_LIMITACION

DICCIONARIO: dict[str, dict[str, str]] = {
    # --- identificación y diseño (no predictoras)
    "conglome": _v("Conglomerado muestral", "código", "Código INEI del conglomerado (llave del hogar).", "identificador"),
    "vivienda": _v("Número de vivienda", "código", "Número de vivienda dentro del conglomerado (llave INEI).", "identificador"),
    "hogar": _v("Número de hogar", "código", "Número de hogar dentro de la vivienda (llave INEI).", "identificador"),
    "hogar_uid": _v("Identificador anónimo del hogar", "código", "Seudónimo SHA-256 de conglome|vivienda|hogar|sal. No permite reconocer al hogar.", "identificador"),
    "factor_expansion": _v("Factor de expansión muestral", "hogares representados", "Peso de la encuesta (FACTOR07 de la ENAHO): número de hogares de la población que representa cada hogar de la muestra.", "peso"),
    "anio": _v("Año de la encuesta", "año", "Año de la ENAHO (en el dataset real solo 2025).", "numérica"),
    "mes": _v("Mes de la entrevista", "mes (1-12)", "Mes de la entrevista (variable MES). El FIES refiere al mes previo; hay estacionalidad.", "numérica"),
    "ubigeo": _v("Ubigeo", "código de 6 dígitos", "Código de ubicación geográfica (departamento, provincia, distrito).", "identificador"),
    "provincia": _v("Provincia (código)", "código de 4 dígitos", "Primeros 4 dígitos del ubigeo; las bases no traen el nombre.", "identificador"),
    "distrito": _v("Distrito (código)", "código de 6 dígitos", "Código de ubigeo del distrito; las bases no traen el nombre.", "identificador"),
    # --- territorio
    "departamento": _v("Departamento", "categoría (25)", "Departamento según los dos primeros dígitos del ubigeo (catálogo INEI de 25 departamentos; Callao = 07).", "categórica"),
    "area": _v("Área de residencia", "Urbano / Rural", "Derivada del estrato de la ENAHO: estratos 1-5 = urbano (U), 6-8 = rural (R). Definición propia del proyecto.", "categórica"),
    "region_natural": _v("Región natural", "costa / sierra / selva", "Derivada del dominio muestral (DOMINIO): aproximación, pues los dominios son áreas de muestreo.", "categórica"),
    # --- FIES y objetivo
    **{f"fies_q{i}": _v(f"Ítem FIES {i}", "sí = 1 / no = 0", _FIES_ITEM.format(i=i, q=FIES_PREGUNTAS[i][1]) + " " + _FIES_VAL, "binaria") for i in range(1, 9)},
    "fies_raw_score": _v("Puntaje crudo FIES", "puntos (0-8)", "Suma de los 8 ítems FIES; solo existe si los 8 ítems fueron observados.", "numérica"),
    "fies_rasch_logit": _v("Severidad FIES (Rasch)", "logit", "No disponible: requiere parámetros Rasch calibrados que no se fabricaron (100 % faltante).", "numérica", "parámetros Rasch"),
    "fies_nivel": _v("Nivel de inseguridad alimentaria (FIES)", "0 = seguro, 1 = leve, 2 = moderada, 3 = severa", "Objetivo multiclase. Cortes del puntaje crudo: leve >= 1, moderada >= 4, severa >= 7.", "objetivo", "cortes del puntaje crudo (no es la clasificación oficial Rasch del INEI/FAO)"),
    "ia_binaria": _v("Inseguridad alimentaria moderada o severa", "1 = sí / 0 = no", "1 si el nivel FIES es >= 2 (moderada o severa).", "objetivo", "cortes del puntaje crudo"),
    # --- económicas
    "ingreso_pc": _v("Ingreso per cápita mensual", "S/ de 2025 por persona al mes", "Ingreso del hogar (INGHOG2D de la Sumaria, anual y deflactado) dividido entre 12 y entre los miembros (MIEPERHO); soles reales a precios promedio de 2025 de Lima Metropolitana.", "numérica"),
    "gasto_pc": _v("Gasto per cápita mensual", "S/ de 2025 por persona al mes", "Gasto del hogar (GASHOG2D de la Sumaria, anual y deflactado) dividido entre 12 y entre los miembros (MIEPERHO); soles reales de 2025 (Lima Metropolitana).", "numérica"),
    "gasto_alim_pc": _v("Gasto en alimentos per cápita mensual", "S/ de 2025 por persona al mes", "Suma de los rubros del grupo 1 «Alimentos» de GASHOG2D (GRU11HD ... GRU14HD6), entre 12 y entre los miembros; soles reales de 2025.", "numérica"),
    "share_alimentario": _v("Proporción del gasto en alimentos", "proporción (0-1)", "gasto_alim_pc / gasto_pc (vacío si el gasto es 0).", "numérica"),
    "pobreza_monetaria": _v("Pobreza monetaria", "1 = pobre / 0 = no pobre", "Variable POBREZA de la Sumaria: 1 pobre extremo y 2 pobre no extremo = 1; 3 no pobre = 0 (códigos del diccionario del INEI).", "binaria"),
    # --- empleo
    "n_ocupados": _v("Miembros ocupados", "personas", "Miembros del hogar con OCU500 = 1 (ocupado).", "numérica"),
    "informalidad": _v("Informalidad laboral", "proporción de ocupados (0-1)", "No disponible: la base 500 publicada no trae la variable derivada; 100 % faltante y excluida de los predictores.", "numérica", "regla de informalidad"),
    "desempleo": _v("Algún miembro desocupado", "1 = sí / 0 = no", "1 si algún miembro tiene OCU500 = 2 o 3 (desocupado abierto u oculto).", "binaria"),
    "dependencia_economica": _v("Ocupados por miembro del hogar", "razón (0-1)", "n_ocupados / n_miembros (nombre original del proyecto: «dependencia económica»; valores altos = más ocupados por miembro).", "numérica"),
    # --- educación
    "nivel_educ_jefe": _v("Nivel educativo del jefe de hogar", "escala 0-11", "P301A - 1 (P301A: 1 sin nivel ... 11 maestría o doctorado, 12 básica especial): 0 = sin nivel, 5 = secundaria completa, 9 = superior universitaria completa, 10 = maestría o doctorado, 11 = básica especial. Los códigos fuera de 1-12 quedan como dato faltante.", "numérica"),
    "asistencia_escolar": _v("Asistencia escolar de 6 a 17 años", "proporción (0-1)", "Proporción de miembros de 6 a 17 años que asisten (P307). Falta en ~61 % de los hogares (sin niños en esa edad).", "numérica"),
    "grupo_educ_jefe": _v("Grupo educativo del jefe", "sin primaria / primaria / secundaria / superior", "Agrupación de nivel_educ_jefe: <2 sin primaria, 2-3 primaria, 4-5 secundaria, >=6 superior. Se usa en equidad.", "categórica"),
    # --- composición del hogar
    "n_miembros": _v("Miembros del hogar", "personas", "Residentes habituales (P204 = 1), excluidos parentesco 8 y 9; coincide con MIEPERHO en el 100 % de los hogares.", "numérica"),
    "n_menores5": _v("Menores de 5 años", "personas", "Miembros con edad < 5.", "numérica"),
    "n_menores18": _v("Menores de 18 años", "personas", "Miembros con edad < 18.", "numérica"),
    "n_adultos_mayores": _v("Adultos mayores (65 o más)", "personas", "Miembros con edad >= 65.", "numérica"),
    "edad_jefe": _v("Edad del jefe de hogar", "años", "Edad del jefe de hogar (parentesco = 1).", "numérica"),
    "sexo_jefe": _v("Sexo del jefe de hogar", "Hombre / Mujer", "Sexo del jefe de hogar (P207): H = hombre, M = mujer.", "categórica"),
    "grupo_edad_jefe": _v("Grupo de edad del jefe", "<30 / 30-44 / 45-59 / 60+ años", "Agrupación de edad_jefe, usada para equidad.", "categórica"),
    "tiene_menor5": _v("Hogar con menores de 5 años", "1 = sí / 0 = no", "1 si n_menores5 > 0. Se usa para equidad.", "binaria"),
    # --- vivienda
    "agua_potable": _v("Acceso a agua por red pública", "1 = sí / 0 = no", "1 si P110 = 1 o 2 (red pública). No usa la prueba de cloro. Definición propia del proyecto.", "binaria"),
    "saneamiento": _v("Acceso a saneamiento adecuado", "1 = sí / 0 = no", "1 si P111A = 1-4 (desagüe de red pública, letrina con tratamiento, pozo séptico o biodigestor). Definición propia del proyecto.", "binaria"),
    "hacinamiento": _v("Personas por habitación", "personas por habitación", "n_miembros / P104 (habitaciones de la vivienda; se descartan valores 0 o >= 99).", "numérica", "interpretación de P104 en el diccionario ENAHO"),
    "tipo_vivienda": _v("Tipo de vivienda", "categoría (código INEI P101)", "Código de la pregunta P101 de la ENAHO (1 casa independiente, 2 departamento en edificio, 3 vivienda en quinta, 4 casa de vecindad, 5 choza o cabaña, 6 vivienda improvisada, 7 local no destinado para habitación, 8 otro).", "categórica"),
    "material_vivienda": _v("Material de las paredes de la vivienda", "categoría (código INEI P102)", "Código de la pregunta P102 de la ENAHO: material predominante en las paredes exteriores (no incluye piso ni techo).", "categórica"),
}

# Etiquetas de valores categóricos verificadas en el código/datos
_VALORES: dict[str, dict[str, str]] = {
    "area": {"U": "Urbano", "R": "Rural"},
    "sexo_jefe": {"H": "Hombre", "M": "Mujer"},
    "region_natural": {"costa": "Costa", "sierra": "Sierra", "selva": "Selva"},
    "pobreza_monetaria": {"1": "Pobre", "0": "No pobre"},
    "tipo_vivienda": dict(TIPO_VIVIENDA),
    "material_vivienda": dict(MATERIAL_PAREDES),
}
_PREFIJOS = ("num__", "cat__", "fies__", "remainder__")
_ORDENADAS = sorted(DICCIONARIO, key=len, reverse=True)


def _split(nombre: str) -> tuple[str | None, str | None]:
    """Separa un nombre de columna (posiblemente transformado, p. ej. ``cat__area_U``) en (variable, categoría)."""
    n = str(nombre)
    for p in _PREFIJOS:
        if n.startswith(p):
            n = n[len(p):]
            break
    if n in DICCIONARIO:
        return n, None
    for v in _ORDENADAS:
        if n.startswith(v + "_"):
            return v, n[len(v) + 1:]
    return None, None


def valor_etq(variable: str, valor: Any) -> str:
    """Etiqueta legible de un valor de una variable categórica (departamentos con mayúscula inicial)."""
    s = str(valor)
    if variable in _VALORES and s in _VALORES[variable]:
        return _VALORES[variable][s]
    if variable == "departamento":
        return s.replace("_", " ").title().replace(" De ", " de ").replace(" Del ", " del ")
    return s


def _u(unidad: str) -> str:
    """Unidad apta para ir entre paréntesis: «puntos (0-8)» -> «puntos, 0-8»."""
    import re
    return re.sub(r"\s*\(([^)]*)\)", lambda m: ", " + m.group(1), unidad)


def etq(nombre: str, unidad: bool = False) -> str:
    """Etiqueta legible de una variable (acepta nombres transformados y dummies). Si no se conoce, devuelve el original."""
    v, cat = _split(nombre)
    if v is None:
        return str(nombre)
    base = DICCIONARIO[v]["nombre"]
    if cat is not None:
        base = f"{base}: {valor_etq(v, cat)}"
    elif unidad and DICCIONARIO[v]["unidad"]:
        base = f"{base} ({_u(DICCIONARIO[v]['unidad'])})"
    return base


def etq_codigo(nombre: str) -> str:
    """Etiqueta seguida del código original entre paréntesis (para tablas y anexos)."""
    v, cat = _split(nombre)
    return str(nombre) if v is None else f"{etq(nombre)} [{v}]"


def ayuda(variable: str) -> str:
    """Texto de ayuda contextual (tooltip): descripción, unidad, código original y avisos por validar."""
    d = DICCIONARIO.get(variable)
    if d is None:
        return f"Código original: {variable}"
    partes = [d["descripcion"]]
    if d["unidad"]:
        partes.append(f"Unidad: {d['unidad']}.")
    if d["por_validar"]:
        partes.append(f"{POR_VALIDAR} Falta verificar: {d['por_validar']}.")
    partes.append(f"Código original: {variable}.")
    return " ".join(partes)


def etiquetas(nombres) -> list[str]:
    return [etq(n) for n in nombres]


def renombrar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Copia de ``df`` con las COLUMNAS conocidas renombradas para mostrar (no usar para entrenar)."""
    return df.rename(columns={c: etq(c) for c in df.columns if _split(c)[0] is not None and _split(c)[1] is None})


def traducir_columna(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Copia de ``df`` donde los VALORES de ``col`` (nombres de variable) pasan a etiqueta legible. Para presentación."""
    if col not in df.columns:
        return df
    out = df.copy()
    out[col] = [etq(v) for v in out[col]]
    return out


def traducir_valores(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    out = df
    for c in columnas:
        out = traducir_columna(out, c)
    return out


def diccionario_df(variables: list[str] | None = None, incluir_codigo: bool = True) -> pd.DataFrame:
    """Tabla «Diccionario de variables» (anexo del informe). ``variables``: subconjunto/orden; por defecto todas."""
    claves = [k for k in (variables or DICCIONARIO) if k in DICCIONARIO]
    filas = []
    for k in claves:
        d = DICCIONARIO[k]
        fila = {"Variable": d["nombre"]}
        if incluir_codigo:
            fila["Código original"] = k
        fila.update({"Unidad / valores": d["unidad"], "Tipo": d["tipo"], "Descripción": d["descripcion"],
                     "Por validar": (POR_VALIDAR + " " + d["por_validar"]) if d["por_validar"] else ""})
        filas.append(fila)
    return pd.DataFrame(filas)


def validar_cobertura(columnas) -> list[str]:
    """Columnas del dataset sin entrada en el diccionario (debe ser vacío para el dataset real)."""
    return [c for c in columnas if c not in DICCIONARIO]


def presentar_tabla(df: pd.DataFrame, col: str = "variable") -> pd.DataFrame:
    """Copia de ``df`` para MOSTRAR: la columna ``col`` pasa a «Etiqueta [código]» y, si hay columna ``grupo``, sus valores a etiqueta legible."""
    if df is None or col not in df.columns:
        return df
    out = df.copy()
    orig = out[col].astype(str)
    if "grupo" in out.columns:
        out["grupo"] = [valor_etq(v, g) if v in DICCIONARIO else g for v, g in zip(orig, out["grupo"])]
    out[col] = [etq_codigo(v) for v in orig]
    return out


# --------------------------------------------------------------------------
# Variables ACCIONABLES para el análisis contrafactual (capa de presentación y de restricciones del análisis)
# --------------------------------------------------------------------------
# «Accionable» = variable que un hogar o un programa podría, en principio, modificar a mediano plazo. NO implica que modificarla
# cause un cambio en la inseguridad alimentaria ni que deba recomendarse: el modelo solo describe asociaciones predictivas.
# La lista completa es [POR VALIDAR] con profesionales del área (nutrición, política social): este conjunto es una propuesta técnica.
# ``direccion``: sentido plausible de mejora que se explora ("aumentar" / "reducir"); ``tipo``: continua | entero | binaria.
ACCIONABLES: dict[str, dict[str, str]] = {
    "ingreso_pc": {"direccion": "aumentar", "tipo": "continua", "por_validar": "sí",
                   "justificacion": "El ingreso puede aumentar por empleo, transferencias o actividad económica. Se mueve solo hasta el percentil 90 de Desarrollo; "
                                    "el gasto, la proporción del gasto en alimentos y la pobreza monetaria (derivados del ingreso y del gasto) se mantienen fijos."},
    "n_ocupados": {"direccion": "aumentar", "tipo": "entero", "por_validar": "sí",
                   "justificacion": "Más miembros ocupados (acceso a empleo). No supera el número de miembros del hogar ni el percentil 90 de Desarrollo."},
    "desempleo": {"direccion": "reducir", "tipo": "binaria", "por_validar": "sí",
                  "justificacion": "Que ningún miembro esté desocupado (empleo/intermediación laboral). Indicador 0/1."},
    "asistencia_escolar": {"direccion": "aumentar", "tipo": "continua", "por_validar": "sí",
                           "justificacion": "Mayor asistencia escolar de 6 a 17 años (programas de permanencia escolar). Solo si el hogar tiene dato (falta en ~61 % de hogares)."},
    "agua_potable": {"direccion": "aumentar", "tipo": "binaria", "por_validar": "sí",
                     "justificacion": "Acceso a agua por red pública (infraestructura de servicios). Indicador 0/1."},
    "saneamiento": {"direccion": "aumentar", "tipo": "binaria", "por_validar": "sí",
                    "justificacion": "Acceso a saneamiento adecuado (infraestructura de servicios). Indicador 0/1."},
    "hacinamiento": {"direccion": "reducir", "tipo": "continua", "por_validar": "sí",
                     "justificacion": "Menos personas por habitación (mejora de vivienda). Se mueve solo hasta el percentil 10 de Desarrollo."},
}
NO_ACCIONABLES: dict[str, str] = {
    "edad_jefe": "Característica personal del jefe de hogar, no modificable.", "sexo_jefe": "Característica personal del jefe de hogar, no modificable.",
    "departamento": "Ubicación geográfica: no es una intervención.", "area": "Ubicación geográfica: no es una intervención.",
    "region_natural": "Ubicación geográfica: no es una intervención.",
    "n_miembros": "Composición del hogar: no se plantea como intervención.", "n_menores5": "Composición del hogar: no se plantea como intervención.",
    "n_menores18": "Composición del hogar: no se plantea como intervención.", "n_adultos_mayores": "Composición del hogar: no se plantea como intervención.",
    "nivel_educ_jefe": "El nivel educativo del jefe no cambia en el horizonte corto del análisis.",
    "tipo_vivienda": "Característica estructural de la vivienda (P101); no se plantea como intervención.",
    "material_vivienda": "Material de las paredes (P102); no se plantea como intervención.",
    "gasto_pc": "Variable derivada de los recursos del hogar; cambiarla sola sería incoherente con el ingreso.",
    "gasto_alim_pc": "Variable derivada de los recursos del hogar (consecuencia, no palanca); cambiarla sola sería incoherente.",
    "share_alimentario": "Razón derivada (gasto en alimentos / gasto total).",
    "pobreza_monetaria": "Clasificación derivada del ingreso y del gasto frente a la línea de pobreza.",
    "dependencia_economica": "Razón derivada (ocupados / miembros).",
    "informalidad": "Sin datos en este dataset (100 % faltante).",
}
_RAIZ_FIES = ("fies_",)


def es_accionable(variable: str) -> bool:
    """True solo para las variables de ``ACCIONABLES``; ninguna variable derivada del FIES (``fies_*``) es accionable por construcción."""
    return variable in ACCIONABLES and not variable.startswith(_RAIZ_FIES)


def accionables_df() -> pd.DataFrame:
    """Tabla documentada de variables accionables y no accionables (con la justificación de cada decisión)."""
    filas = [{"Variable": etq(v), "Código": v, "¿Accionable?": "Sí [POR VALIDAR]", "Sentido explorado": d["direccion"], "Justificación": d["justificacion"]} for v, d in ACCIONABLES.items()]
    filas += [{"Variable": etq(v), "Código": v, "¿Accionable?": "No", "Sentido explorado": "—", "Justificación": m} for v, m in NO_ACCIONABLES.items()]
    filas.append({"Variable": "Ítems y puntaje FIES (fies_q1…fies_q8, fies_raw_score)", "Código": "fies_*", "¿Accionable?": "No (excluidos siempre)",
                  "Sentido explorado": "—", "Justificación": "El objetivo se deriva de ellos: cualquier cambio sería circular."})
    return pd.DataFrame(filas)
