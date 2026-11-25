"""catalogo_campos.py - Catálogo ÚNICO de los campos de entrada de la inferencia (formulario individual y plantilla del lote).

Para cada dato que la persona usuaria debe dar: etiqueta corta, una línea de ayuda (qué es, cómo responder, unidad), valores permitidos con
sus etiquetas oficiales (el valor del modelo es interno), tipo, límites y marca ``[POR VALIDAR]`` cuando algo no está confirmado.

Fuentes (ENAHO 2025, diccionarios oficiales del INEI incluidos en los datos públicos del proyecto):
* FIES: Diccionario módulo 2082 (P130_1 a P130_8): 1 Sí, 2 No, 3 No sabe, 4 No responde.
* P101 tipo de vivienda y P102 material de las paredes exteriores, P104 habitaciones, P110 agua, P111A servicio higiénico: Diccionario módulo 01.
* P301A nivel educativo: Diccionario módulo 03. POBREZA: Diccionario módulo 34 (Sumaria). ESTRATO y DOMINIO: Diccionario módulo 2082.
* La construcción de cada variable del modelo está en ``fuentes_publicas.py`` (es la referencia de las fórmulas derivadas).

Algunas variables del modelo son cocientes (hacinamiento, dependencia económica, proporción del gasto en alimentos, asistencia escolar y
puntaje FIES). Aquí se piden los datos simples (habitaciones, ocupados, gasto, escolares, ítems) y el cociente se CALCULA con la misma
fórmula del entrenamiento; así la persona no necesita conocer la definición interna. Este módulo no depende de Streamlit.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

POR_VALIDAR = "[POR VALIDAR]"

# ------------------------------------------------------------------------------------------------ códigos oficiales
TIPO_VIVIENDA = {"1": "Casa independiente", "2": "Departamento en edificio", "3": "Vivienda en quinta",
                 "4": "Vivienda en casa de vecindad (callejón, solar o corralón)", "5": "Choza o cabaña", "6": "Vivienda improvisada",
                 "7": "Local no destinado para habitación humana", "8": "Otro"}
MATERIAL_PAREDES = {"1": "Ladrillo o bloque de cemento", "2": "Piedra o sillar con cal o cemento", "3": "Adobe", "4": "Tapia",
                    "5": "Quincha (caña con barro)", "6": "Piedra con barro", "7": "Madera (pona, tornillo, etc.)",
                    "8": "Triplay, calamina o estera", "9": "Otro material"}
# P301A (códigos 1 a 12). El modelo usa P301A - 1, de modo que el valor del modelo es el código oficial menos uno.
P301A = {1: "Sin nivel", 2: "Educación inicial", 3: "Primaria incompleta", 4: "Primaria completa", 5: "Secundaria incompleta",
         6: "Secundaria completa", 7: "Superior no universitaria incompleta", 8: "Superior no universitaria completa",
         9: "Superior universitaria incompleta", 10: "Superior universitaria completa", 11: "Maestría o doctorado", 12: "Básica especial"}
NIVEL_EDUC_MODELO = {c - 1: t for c, t in P301A.items()}          # valor del modelo -> etiqueta oficial
SEXO = {"H": "Hombre", "M": "Mujer"}
AREA = {"U": "Urbano", "R": "Rural"}
REGION = {"costa": "Costa", "sierra": "Sierra", "selva": "Selva"}
POBREZA = {1: "Pobre (extremo o no extremo)", 0: "No pobre"}
FIES_PREGUNTAS = {  # (texto corto, texto de la pregunta oficial P130_i, ya sin el encabezado común)
    1: ("¿Se preocupó por no tener suficiente comida?", "estuvo preocupado de no tener suficiente comida para comer"),
    2: ("¿No pudo comer alimentos saludables y nutritivos?", "no pudo comer alimentos saludables y nutritivos"),
    3: ("¿No comió alimentos variados?", "no comió alimentos variados"),
    4: ("¿Dejó de desayunar, almorzar o cenar?", "tuvo que dejar de desayunar, almorzar o cenar"),
    5: ("¿Comió menos de lo normal?", "comió menor cantidad de lo normal"),
    6: ("¿El hogar se quedó sin alimentos?", "el hogar se quedó sin alimentos"),
    7: ("¿Tuvo hambre y no comió?", "tuvo hambre pero no comió"),
    8: ("¿Estuvo un día entero sin comer?", "estuvo sin comer durante un día entero"),
}
FIES_LIMITACION = ("El diccionario del INEI codifica cada ítem FIES como 1 Sí, 2 No, 3 No sabe y 4 No responde. El proyecto usa solo Sí (1) y No (0) y "
                   "trata «No sabe», «No responde» y los blancos como dato faltante; aquí solo se puede responder Sí o No.")

SECCIONES = [
    ("hogar", "Hogar y vivienda", "Quiénes viven en el hogar y cómo es la vivienda."),
    ("economia", "Economía del hogar", "Ingresos y gastos del hogar, siempre por persona y por mes."),
    ("jefe", "Jefe(a) del hogar, educación y empleo", "Datos de la persona jefa del hogar y de quienes trabajan o estudian."),
    ("territorio", "Territorio", "Dónde está el hogar."),
    ("fies", "Escala FIES (experiencia de hambre en el último mes)", "Ocho preguntas de Sí o No sobre el mes anterior."),
]


def _nan_si_none(x):
    return np.nan if x is None else x


# ------------------------------------------------------------------------------------------------ entradas (lo que se le pide a la persona)
# ``var``: variable del modelo que alimenta directamente (None = dato auxiliar que solo sirve para calcular una derivada).
# ``tipo``: si_no | entero | decimal | opciones. ``ayuda``: una línea visible; ``detalle``: texto del tooltip (código, fuente, avisos).
def _e(seccion, etiqueta, ayuda, tipo, detalle="", unidad="", minimo=None, maximo=None, paso=None, defecto=None, opciones=None, var="__key__", validar=""):
    return {"seccion": seccion, "etiqueta": etiqueta, "ayuda": ayuda, "tipo": tipo, "detalle": detalle, "unidad": unidad, "min": minimo,
            "max": maximo, "paso": paso, "defecto": defecto, "opciones": opciones, "var": var, "por_validar": validar}


ENTRADAS: dict[str, dict[str, Any]] = {
    # ---------------- Hogar y vivienda
    "n_miembros": _e("hogar", "Personas que viven en el hogar", "Residentes habituales, sin contar trabajadores del hogar ni pensionistas.", "entero",
                     "Residentes habituales (P204 = 1) sin parentesco 8 (trabajador del hogar) ni 9 (pensionista) (P203).", "personas", 1, 20, 1, 4),
    "n_menores5": _e("hogar", "Menores de 5 años", "Cuántas personas del hogar tienen menos de 5 años.", "entero", "Edad (P208A) menor de 5.", "personas", 0, 20, 1, 0),
    "n_menores18": _e("hogar", "Menores de 18 años", "Cuántas personas del hogar tienen menos de 18 años (incluya a los menores de 5).", "entero",
                      "Edad (P208A) menor de 18; incluye a los menores de 5.", "personas", 0, 20, 1, 1),
    "n_adultos_mayores": _e("hogar", "Adultos de 65 años o más", "Cuántas personas del hogar tienen 65 años o más.", "entero", "Edad (P208A) de 65 o más.", "personas", 0, 20, 1, 0),
    "habitaciones": _e("hogar", "Habitaciones de la vivienda", "Sin contar baño, cocina, pasadizos ni garaje.", "entero",
                       "Pregunta P104 de la ENAHO. Con ella y las personas del hogar se calcula el hacinamiento (personas por habitación).", "habitaciones", 1, 20, 1, 3,
                       var=None),
    "tipo_vivienda": _e("hogar", "Tipo de vivienda", "Cómo es la vivienda del hogar.", "opciones", "Pregunta P101 de la ENAHO; códigos y textos oficiales del INEI.",
                        opciones=TIPO_VIVIENDA, defecto="1"),
    "material_vivienda": _e("hogar", "Material de las paredes exteriores", "Material predominante en las paredes de la vivienda.", "opciones",
                            "Pregunta P102 de la ENAHO (solo paredes; no incluye piso ni techo). Variable del modelo: material_vivienda.", opciones=MATERIAL_PAREDES, defecto="1"),
    "agua_potable": _e("hogar", "Agua por red pública (dentro o fuera de la vivienda)", "Sí si el agua viene de la red pública, dentro de la vivienda o dentro del edificio.", "si_no",
                       "Sí = P110 en {1, 2}. Definición propia del proyecto: NO indica que el agua sea potable ni mide el cloro.", defecto=1),
    "saneamiento": _e("hogar", "Servicio higiénico adecuado", "Sí si el baño se conecta a red pública, letrina con tratamiento o pozo séptico/biodigestor.", "si_no",
                      "Sí = P111A en {1, 2, 3, 4} (red pública dentro o fuera de la vivienda, letrina con tratamiento, pozo séptico, tanque séptico o biodigestor). Definición propia del proyecto.",
                      defecto=1),
    # ---------------- Economía
    "ingreso_pc": _e("economia", "Ingreso mensual por persona (S/ de 2025)", "Ingreso total del hogar en un mes, dividido entre sus miembros.", "decimal",
                     "Ingreso anual del hogar (INGHOG2D, Sumaria) ÷ 12 ÷ miembros. Es un valor real: deflactado, a precios promedio de 2025 de Lima Metropolitana. Ejemplo: 3 600 al mes entre 4 personas = 900.",
                     "S/ por persona al mes", 0.0, 100000.0, 10.0, 850.0),
    "gasto_pc": _e("economia", "Gasto mensual por persona (S/ de 2025)", "Gasto total del hogar en un mes, dividido entre sus miembros.", "decimal",
                   "Gasto anual del hogar (GASHOG2D, Sumaria) ÷ 12 ÷ miembros, en soles reales de 2025 (precios de Lima Metropolitana).", "S/ por persona al mes", 1.0, 100000.0, 10.0, 700.0),
    "gasto_alim_pc": _e("economia", "Gasto en alimentos por persona (S/ de 2025)", "Parte del gasto mensual del hogar que se destina a alimentos, por persona.", "decimal",
                        "Grupo 1 «Alimentos» del gasto del hogar (Sumaria) ÷ 12 ÷ miembros, en soles reales de 2025. Debe ser menor o igual al gasto total.", "S/ por persona al mes", 0.0, 100000.0, 10.0, 240.0),
    "pobreza_monetaria": _e("economia", "Pobreza monetaria (clasificación del INEI)", "Cómo clasifica el INEI al hogar según su gasto frente a la línea de pobreza.", "opciones",
                            "Variable POBREZA de la Sumaria: 1 Pobre extremo y 2 Pobre no extremo cuentan como «Pobre»; 3 No pobre. La calcula el INEI; si no la tiene, el resultado es menos fiable.",
                            opciones=POBREZA, defecto=0),
    # ---------------- Jefe, educación y empleo
    "edad_jefe": _e("jefe", "Edad del jefe(a) del hogar", "Años cumplidos de la persona jefa del hogar.", "entero", "Edad (P208A) de quien tiene parentesco 1 (P203).", "años", 12, 110, 1, 50),
    "sexo_jefe": _e("jefe", "Sexo del jefe(a) del hogar", "Sexo declarado de la persona jefa del hogar.", "opciones", "P207 de la ENAHO (1 hombre, 2 mujer).", opciones=SEXO, defecto="H"),
    "nivel_educ_jefe": _e("jefe", "Último nivel educativo del jefe(a)", "Último nivel de estudios que aprobó la persona jefa del hogar.", "opciones",
                          "Pregunta P301A de la ENAHO. El modelo usa el código oficial menos 1; esa conversión es automática. «Básica especial» es el código 12 del INEI.",
                          opciones=NIVEL_EDUC_MODELO, defecto=5),
    "n_ocupados": _e("jefe", "Personas del hogar que trabajan", "Cuántas personas del hogar tienen un empleo o actividad económica.", "entero",
                     "Miembros con condición de actividad OCU500 = 1 (ocupado). Debe ser menor o igual a las personas del hogar.", "personas", 0, 20, 1, 2),
    "desempleo": _e("jefe", "¿Alguna persona del hogar está desocupada?", "Sí si alguien busca trabajo o quisiera trabajar y no tiene empleo.", "si_no",
                    "Sí = algún miembro con OCU500 = 2 o 3 (desocupado abierto u oculto). La definición exacta de cada código del INEI está " + POR_VALIDAR + ".",
                    defecto=0, validar="definición de OCU500 = 2 y 3"),
    "escolares_6_17": _e("jefe", "Personas del hogar de 6 a 17 años", "Edad escolar; si no hay ninguna, la asistencia escolar queda como «no aplica».", "entero",
                         "Se usa para calcular la asistencia escolar. Con 0, el dato queda faltante igual que en el entrenamiento (~61 % de los hogares no tiene hijos de esa edad).",
                         "personas", 0, 15, 1, 0, var=None),
    "escolares_asisten": _e("jefe", "De ellas, cuántas asisten a la escuela", "Asisten actualmente a un centro de enseñanza (P307).", "entero",
                            "Proporción de miembros de 6 a 17 años que asisten (P307 = 1 entre quienes respondieron 1 o 2).", "personas", 0, 15, 1, 0, var=None),
    # ---------------- Territorio
    "departamento": _e("territorio", "Departamento", "Departamento donde está el hogar (Callao aparece por separado).", "opciones",
                       "Primeros dos dígitos del ubigeo (catálogo INEI de 25 departamentos).", opciones=None, defecto="lima"),
    "area": _e("territorio", "Área de residencia", "Urbano si el centro poblado tiene 2 000 habitantes o más; rural en los demás casos.", "opciones",
               "Según el estrato muestral ENAHO: estratos 1 a 5 (centros poblados de 2 000 o más habitantes) = urbano; 6 a 8 (menos de 2 000 habitantes y áreas de empadronamiento rural) = rural. "
               "Equivalencia con la definición urbana oficial del INEI: " + POR_VALIDAR + ".", opciones=AREA, defecto="U", validar="equivalencia con la definición urbana del INEI"),
    "region_natural": _e("territorio", "Región natural", "Costa, sierra o selva, según el dominio de la encuesta.", "opciones",
                         "Derivada del dominio muestral (DOMINIO): costa norte, centro y sur y Lima Metropolitana = costa; sierra norte, centro y sur = sierra; selva = selva. "
                         "Es una aproximación: no siempre coincide con el departamento.", opciones=REGION, defecto="costa"),
}
for _i, (_corta, _oficial) in FIES_PREGUNTAS.items():
    ENTRADAS[f"fies_q{_i}"] = _e("fies", f"Ítem FIES {_i} · {_corta}", f"Mes anterior, por falta de dinero u otros medios: {_oficial}.", "si_no",
                                 f"Resumen de la pregunta P130_{_i} de la ENAHO (mes anterior, por falta de dinero u otras formas de conseguir alimentos): {_oficial}. "
                                 + FIES_LIMITACION, defecto=0)


# ------------------------------------------------------------------------------------------------ variables derivadas (se calculan, no se escriben)
def _div(a, b):
    return float(a) / float(b) if b not in (None, 0) and not (isinstance(b, float) and np.isnan(b)) else np.nan


def _asistencia(e: dict) -> float:
    n = e.get("escolares_6_17") or 0
    return np.nan if n <= 0 else float(e.get("escolares_asisten") or 0) / float(n)


DERIVADOS: dict[str, dict[str, Any]] = {
    "share_alimentario": {"seccion": "economia", "etiqueta": "Proporción del gasto en alimentos", "unidad": "0 a 1", "entradas": ["gasto_alim_pc", "gasto_pc"],
                          "formula": "gasto en alimentos ÷ gasto total", "fn": lambda e: _div(e["gasto_alim_pc"], e["gasto_pc"]) if e["gasto_pc"] > 0 else np.nan,
                          "ayuda": "Se calcula sola: gasto en alimentos dividido entre gasto total (0,35 = 35 % del gasto va a alimentos)."},
    "hacinamiento": {"seccion": "hogar", "etiqueta": "Personas por habitación (hacinamiento)", "unidad": "personas por habitación", "entradas": ["n_miembros", "habitaciones"],
                     "formula": "personas del hogar ÷ habitaciones", "fn": lambda e: _div(e["n_miembros"], e["habitaciones"]) if e["habitaciones"] > 0 else np.nan,
                     "ayuda": "Se calcula sola: personas del hogar entre habitaciones (4 personas en 2 habitaciones = 2,0)."},
    "dependencia_economica": {"seccion": "jefe", "etiqueta": "Personas que trabajan por cada miembro del hogar", "unidad": "0 a 1", "entradas": ["n_ocupados", "n_miembros"],
                              "formula": "personas que trabajan ÷ personas del hogar", "fn": lambda e: _div(e["n_ocupados"], e["n_miembros"]),
                              "ayuda": "Se calcula sola: 2 que trabajan en un hogar de 4 = 0,50. Más cerca de 1, más personas trabajan por miembro.",
                              "detalle": "El proyecto la llama «dependencia económica», pero es ocupados ÷ miembros (valores altos = más ocupados por miembro). "
                                         "Su uso como indicador de dependencia económica es " + POR_VALIDAR + "."},
    "asistencia_escolar": {"seccion": "jefe", "etiqueta": "Asistencia escolar de 6 a 17 años", "unidad": "0 a 1, o no aplica", "entradas": ["escolares_6_17", "escolares_asisten"],
                           "formula": "escolares que asisten ÷ personas de 6 a 17 años (vacío si no hay ninguna)", "fn": _asistencia,
                           "ayuda": "Se calcula sola: 2 de 2 escolares asisten = 1,00. Sin personas de 6 a 17 años queda «no aplica» (dato faltante)."},
    "fies_raw_score": {"seccion": "fies", "etiqueta": "Puntaje FIES (suma de los 8 ítems)", "unidad": "0 a 8 puntos", "entradas": [f"fies_q{i}" for i in range(1, 9)],
                       "formula": "suma de los ocho ítems FIES (Sí = 1)", "fn": lambda e: float(sum(int(e[f"fies_q{i}"]) for i in range(1, 9))),
                       "ayuda": "Se calcula solo: cuántas de las 8 preguntas respondió «Sí» (0 a 8)."},
}
# Orden de aparición (por sección). Las entradas auxiliares se muestran junto a lo que alimentan.
ORDEN = ["n_miembros", "n_menores5", "n_menores18", "n_adultos_mayores", "habitaciones", "tipo_vivienda", "material_vivienda", "agua_potable", "saneamiento",
         "ingreso_pc", "gasto_pc", "gasto_alim_pc", "pobreza_monetaria",
         "edad_jefe", "sexo_jefe", "nivel_educ_jefe", "n_ocupados", "desempleo", "escolares_6_17", "escolares_asisten",
         "departamento", "area", "region_natural"] + [f"fies_q{i}" for i in range(1, 9)]
VARIABLES_CATALOGADAS = [v for v, e in ENTRADAS.items() if e["var"] == "__key__"] + list(DERIVADOS)


def etiqueta_departamento(valor: str) -> str:
    from fsai import etiquetas as E
    return E.valor_etq("departamento", valor)


def opciones_de(clave: str, observadas: list | None = None) -> dict[Any, str]:
    """Opciones {valor del modelo: etiqueta oficial} de una entrada. ``observadas``: valores vistos en Desarrollo (se ofrecen solo esos, p. ej. el
    código 7 de P101 no aparece en los datos y el modelo no lo conoce)."""
    e = ENTRADAS[clave]
    base = {v: etiqueta_departamento(v) for v in (observadas or [])} if clave == "departamento" else dict(e["opciones"] or {})
    if observadas is not None and clave != "departamento":
        obs = {str(o) for o in observadas}
        base = {k: t for k, t in base.items() if str(k) in obs}
    return base


def entradas_para(features: list[str]) -> list[str]:
    """Entradas necesarias para alimentar ``features`` (orden de pantalla). Lanza KeyError si una variable no está catalogada."""
    claves: set[str] = set()
    for v in features:
        if v in DERIVADOS:
            claves.update(DERIVADOS[v]["entradas"])
        elif v in ENTRADAS and ENTRADAS[v]["var"] == "__key__":
            claves.add(v)
        else:
            raise KeyError(f"La variable «{v}» no está en el catálogo de campos de inferencia.")
    return [k for k in ORDEN if k in claves]


def valores_modelo(entradas: dict[str, Any], features: list[str]) -> dict[str, Any]:
    """Valores del modelo (en el formato del entrenamiento) a partir de lo que ingresó la persona; calcula las variables derivadas."""
    out: dict[str, Any] = {}
    for v in features:
        out[v] = DERIVADOS[v]["fn"](entradas) if v in DERIVADOS else entradas[v]
    return out


def validar_entradas(e: dict[str, Any]) -> list[str]:
    """Mensajes de error en español claro para combinaciones imposibles. Lista vacía = datos coherentes."""
    err = []
    def g(k, d=0):
        return e.get(k, d) if e.get(k) is not None else d
    n = g("n_miembros", None)
    if n is not None:
        if g("n_menores18") > n:
            err.append(f"Los menores de 18 años ({g('n_menores18')}) no pueden ser más que las personas del hogar ({n}).")
        if g("n_menores18") + g("n_adultos_mayores") > n:
            err.append("Menores de 18 años más adultos de 65 o más superan a las personas del hogar.")
        if g("n_ocupados") > n:
            err.append(f"Las personas que trabajan ({g('n_ocupados')}) no pueden ser más que las personas del hogar ({n}).")
        if g("escolares_6_17") > n:
            err.append("Las personas de 6 a 17 años no pueden ser más que las personas del hogar.")
    if g("n_menores5") > g("n_menores18"):
        err.append("Los menores de 5 años no pueden ser más que los menores de 18 (estos incluyen a los primeros).")
    if g("escolares_6_17") > g("n_menores18"):
        err.append("Las personas de 6 a 17 años no pueden ser más que los menores de 18 años.")
    if g("escolares_asisten") > g("escolares_6_17"):
        err.append("Los escolares que asisten no pueden ser más que las personas de 6 a 17 años.")
    if "gasto_alim_pc" in e and "gasto_pc" in e and g("gasto_alim_pc") > g("gasto_pc"):
        err.append("El gasto en alimentos no puede superar al gasto total del hogar.")
    for k, spec in ENTRADAS.items():
        if k in e and spec["min"] is not None and e[k] is not None and not (spec["min"] <= float(e[k]) <= spec["max"]):
            err.append(f"«{spec['etiqueta']}» debe estar entre {spec['min']:g} y {spec['max']:g} (valor ingresado: {float(e[k]):g}).")
    return err


# ------------------------------------------------------------------------------------------------ hogar de ejemplo (ilustrativo)
EJEMPLO_AVISO = "Ejemplo ilustrativo inventado para mostrar cómo se llena el formulario: no es un hogar real ni representa a ningún hogar."
EJEMPLO: dict[str, Any] = {
    "n_miembros": 4, "n_menores5": 1, "n_menores18": 2, "n_adultos_mayores": 0, "habitaciones": 3, "tipo_vivienda": "1", "material_vivienda": "1",
    "agua_potable": 1, "saneamiento": 1, "ingreso_pc": 900.0, "gasto_pc": 750.0, "gasto_alim_pc": 260.0, "pobreza_monetaria": 0,
    "edad_jefe": 42, "sexo_jefe": "H", "nivel_educ_jefe": 5, "n_ocupados": 2, "desempleo": 0, "escolares_6_17": 1, "escolares_asisten": 1,
    "departamento": "lima", "area": "U", "region_natural": "costa",
    "fies_q1": 1, "fies_q2": 1, "fies_q3": 1, "fies_q4": 0, "fies_q5": 0, "fies_q6": 0, "fies_q7": 0, "fies_q8": 0,
}


def valores_permitidos(variable: str, observadas: list | None = None, rango: tuple[float, float] | None = None) -> str:
    """Texto de «valores permitidos» de una columna del modelo (para la plantilla del lote)."""
    if variable in DERIVADOS:
        d = DERIVADOS[variable]
        return f"Número ({d['unidad']}); se calcula como {d['formula']}" + ("; vacío si no aplica" if variable == "asistencia_escolar" else "")
    e = ENTRADAS[variable]
    if e["tipo"] == "si_no":
        return "1 = Sí, 0 = No"
    if e["tipo"] == "opciones":
        return "; ".join(f"{k} = {t}" for k, t in opciones_de(variable, observadas).items())
    rg = f"; observado en los datos de Desarrollo: {rango[0]:.4g} a {rango[1]:.4g}" if rango else ""
    return f"Número ({e['unidad']}), de {e['min']:g} a {e['max']:g}{rg}"


def catalogo_df(features: list[str], observadas: dict[str, list] | None = None, rangos: dict[str, tuple[float, float]] | None = None):
    """Tabla del catálogo para las columnas del modelo (pestaña de lote): código, significado, ayuda, valores permitidos, tipo y avisos."""
    import pandas as pd
    filas = []
    for v in features:
        if v in DERIVADOS:
            d = DERIVADOS[v]
            filas.append({"Columna (encabezado del CSV)": v, "Qué es": d["etiqueta"], "Cómo responder": d["ayuda"], "Valores permitidos": valores_permitidos(v),
                          "Tipo": "derivado", "Aviso": POR_VALIDAR if POR_VALIDAR in d["ayuda"] else ""})
            continue
        e = ENTRADAS[v]
        filas.append({"Columna (encabezado del CSV)": v, "Qué es": e["etiqueta"], "Cómo responder": e["ayuda"],
                      "Valores permitidos": valores_permitidos(v, (observadas or {}).get(v), (rangos or {}).get(v)), "Tipo": e["tipo"],
                      "Aviso": (POR_VALIDAR + " " + e["por_validar"]) if e["por_validar"] else ""})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------------------------------------ prellenado desde un hogar real
def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v


def entradas_desde_fila(fila) -> dict[str, Any]:
    """Valores del formulario a partir de una fila del dataset analítico (hogar real de Validación), para PRELLENAR. Las entradas auxiliares se
    reconstruyen con aproximación: habitaciones = personas ÷ hacinamiento (redondeado); escolares de 6 a 17 años = menores de 18 (máx. 15) si el hogar
    tiene dato de asistencia escolar (0 si falta) y los que asisten = proporción × ese total. Toda entrada que falte toma el valor por defecto."""
    g = (lambda k: fila[k] if k in fila else np.nan)
    out: dict[str, Any] = {}
    for k, e in ENTRADAS.items():
        if e["var"] != "__key__":
            continue
        v = g(k)
        if e["tipo"] == "opciones" and e["opciones"] is not None and isinstance(next(iter(e["opciones"])), (int, np.integer)):
            n = _num(v)
            out[k] = int(round(n)) if not np.isnan(n) else e["defecto"]
        elif e["tipo"] == "opciones":
            out[k] = str(v) if v is not None and str(v) not in ("nan", "<NA>", "None") else e["defecto"]
        elif e["tipo"] == "si_no":
            n = _num(v)
            out[k] = int(n) if n in (0.0, 1.0) else e["defecto"]
        elif e["tipo"] == "entero":
            n = _num(v)
            out[k] = int(round(n)) if not np.isnan(n) else e["defecto"]
        else:
            n = _num(v)
            out[k] = float(n) if not np.isnan(n) else e["defecto"]
    hac, nm, asis = _num(g("hacinamiento")), _num(g("n_miembros")), _num(g("asistencia_escolar"))
    out["habitaciones"] = int(min(20, max(1, round(nm / hac)))) if hac and not np.isnan(hac) and not np.isnan(nm) and hac > 0 else ENTRADAS["habitaciones"]["defecto"]
    if np.isnan(asis):
        out["escolares_6_17"], out["escolares_asisten"] = 0, 0
    else:
        tot = int(min(15, max(1, _num(g("n_menores18")) if not np.isnan(_num(g("n_menores18"))) else 1)))
        out["escolares_6_17"], out["escolares_asisten"] = tot, int(round(asis * tot))
    return out
