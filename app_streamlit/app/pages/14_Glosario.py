"""Glosario: definiciones en lenguaje sencillo de las métricas y métodos del proyecto, con «cómo leerlo» y su límite ético, más el diccionario de variables.

Página de solo lectura: no consulta la base de datos. El diccionario de variables sale de ``fsai.etiquetas.DICCIONARIO``.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import modulos
from auth import barra_lateral, requerir_acceso
from comun import normalizar_texto
import tema
from tema import config_pagina, encabezado

modulos.registrar_accesos()
config_pagina("Glosario")
usuario, rol = requerir_acceso("glosario")
barra_lateral()

from fsai.etiquetas import DICCIONARIO  # noqa: E402

AVISO_ASOCIACION = "Es una asociación estadística, no una causa; no justifica por sí sola ninguna decisión sobre un hogar."
AVISO_AYUDAS = "No debe usarse para negar ni condicionar ayudas o programas sociales."

# (término, definición sencilla, cómo leerlo, advertencia ética o "")
TERMINOS: tuple[tuple[str, str, str, str], ...] = (
    ("FIES", "Escala de Experiencia de Inseguridad Alimentaria: ocho preguntas de sí/no sobre preocupaciones, recortes o hambre por falta de dinero. "
             "La suma de respuestas afirmativas (0-8) se agrupa en cuatro niveles (0-3).",
     "A más respuestas «sí», mayor inseguridad alimentaria. Los niveles 2 y 3 se agrupan como inseguridad moderada o severa.",
     "Describe la experiencia declarada por el hogar en un mes; no mide el estado nutricional de las personas."),
    ("Factor de expansión", "Peso de cada hogar de la encuesta: cuántos hogares de la población representa.",
     "Las cifras «ponderadas» usan este peso y se pueden leer como estimaciones para la población; las no ponderadas describen solo la muestra.", ""),
    ("Calibración", "Qué tan parecidas son las probabilidades que da el modelo a las frecuencias reales: si dice 30 % para muchos hogares, debería ocurrir en cerca del 30 %.",
     "Un modelo puede ordenar bien los hogares y aun así dar probabilidades poco fiables; revise Brier y ECE.", ""),
    ("F1-macro", "Promedio, sin dar más peso a ninguna clase, del equilibrio entre precisión (de los que el modelo marca, cuántos acierta) y exhaustividad (de los reales, cuántos encuentra).",
     "Va de 0 a 1; mayor es mejor. Compare siempre contra una referencia (por ejemplo, un modelo trivial) y mire su intervalo de confianza.",
     "Un F1 moderado significa que habrá errores en muchos hogares. " + AVISO_AYUDAS),
    ("AUC-ROC", "Probabilidad de que el modelo asigne mayor riesgo a un hogar con el problema que a uno sin él (aquí promediada entre clases, «uno contra el resto»).",
     "0,5 equivale a azar y 1 a separación perfecta. Mide el orden de los riesgos, no si las probabilidades están bien calibradas.", ""),
    ("PR-AUC", "Área bajo la curva precisión-exhaustividad; resume cuánto se sostiene la precisión al buscar más casos reales.",
     "Es más informativa que el AUC-ROC cuando una clase es poco frecuente. Su azar depende de la frecuencia de la clase, no es 0,5.", ""),
    ("MCC", "Coeficiente de correlación de Matthews: resume en un solo número los aciertos y errores de todas las clases.",
     "Va de -1 a 1; 0 es azar y 1 es perfecto. Es robusto cuando las clases están desbalanceadas.", ""),
    ("Brier", "Error cuadrático medio entre las probabilidades predichas y lo que ocurrió.",
     "Menor es mejor. En problemas de varias clases su escala depende del número de clases, por lo que solo es comparable entre modelos con el mismo objetivo y escenario.", ""),
    ("ECE", "Error de calibración esperado: diferencia promedio entre la probabilidad predicha y la frecuencia observada, agrupando las predicciones por tramos.",
     "Menor es mejor; cerca de 0 indica probabilidades fiables. Depende de cómo se agrupen los tramos.", ""),
    ("SHAP", "Reparte la predicción de un modelo entre las variables que usó: cuánto empujó cada una hacia más o menos riesgo.",
     "Barras largas significan que la variable pesó mucho en las predicciones del modelo; el signo indica la dirección.",
     "Describe cómo razona el modelo, no por qué ocurre el fenómeno. " + AVISO_ASOCIACION),
    ("PDP", "Gráfico de dependencia parcial: promedio de la predicción al variar una variable manteniendo el resto como está en los datos.",
     "Una curva creciente indica que, en promedio, el modelo predice más riesgo al subir esa variable.",
     "Puede mezclar combinaciones poco realistas si las variables están correlacionadas. " + AVISO_ASOCIACION),
    ("ALE", "Efectos locales acumulados: como el PDP, pero mira solo valores vecinos reales de cada variable.",
     "Se lee igual que el PDP y es más fiable cuando hay variables correlacionadas.", AVISO_ASOCIACION),
    ("StratifiedGroupKFold", "Forma de dividir los datos en pliegues que mantiene la proporción de cada clase y no separa a miembros del mismo grupo (por ejemplo, el mismo hogar o conglomerado).",
     "Evita que el modelo «vea» en el entrenamiento a los mismos hogares con los que luego se evalúa, lo que inflaría el desempeño.", ""),
    ("Validación geográfica (LODO / macrorregión)", "Se deja fuera un departamento completo (LODO, leave-one-department-out) o una macrorregión para evaluar al modelo en territorios que no vio al entrenar.",
     "Si el desempeño cae respecto a la validación habitual, el modelo generaliza peor a zonas nuevas.",
     "Un buen promedio nacional puede esconder territorios donde el modelo falla. " + AVISO_AYUDAS),
    ("CV anidada", "Validación cruzada con dos niveles: el bucle interno elige hiperparámetros y el externo evalúa, sin que la evaluación participe en la elección.",
     "Da una estimación menos optimista del desempeño que ajustar y evaluar sobre los mismos datos.", ""),
    ("Optuna / TPE", "Optuna es la librería que busca hiperparámetros probando configuraciones; TPE es el algoritmo que, con los ensayos previos, propone las siguientes más prometedoras.",
     "Más ensayos (n_trials) exploran más, pero con costo; solo se pueden comparar corridas que usaron un esfuerzo de búsqueda parecido.", ""),
    ("Equidad (brecha de sensibilidad)", "Compara, entre grupos de hogares (por ejemplo urbano y rural), la sensibilidad: de los hogares realmente en inseguridad severa, qué porcentaje detecta el modelo.",
     "Una brecha grande significa que el modelo deja sin detectar a más hogares de un grupo que de otro. Cuanto más cerca de 0, más parejo.",
     "Una brecha pequeña no prueba que el modelo sea justo ni autoriza su uso para asignar ayudas. " + AVISO_AYUDAS),
)

encabezado(
    "Glosario",
    "Qué significan las métricas y métodos del proyecto, cómo leerlos y qué no se puede concluir de ellos.",
    "Evaluación e interpretación",
    pasos=["Escriba una palabra en el buscador para filtrar los términos.",
           "Lea la definición, «Cómo leerlo» y, cuando exista, la advertencia ética.",
           "En «Diccionario de variables» consulte el significado de cada variable de los datos."],
    produce="Referencia de consulta; no ejecuta cálculos ni modifica datos.",
)

tema.llamada("etica", "Lectura responsable", "Los resultados de este proyecto describen asociaciones estadísticas, no causas, y no deben usarse para negar ayudas ni sancionar a hogares.")

consulta = st.text_input("Buscar", placeholder="p. ej. calibración, SHAP, F1", max_chars=60, help="Busca en el término, la definición y la explicación; ignora tildes y mayúsculas.")
clave = normalizar_texto(consulta)

tema.seccion("Métricas y métodos", "Definición sencilla, cómo leerla y su límite ético.")
visibles = [t for t in TERMINOS if not clave or clave in normalizar_texto(" ".join(t))]
if not visibles:
    st.info("Ningún término coincide con la búsqueda. Pruebe con otra palabra.")
for termino, definicion, como_leerlo, advertencia in visibles:
    tema.tarjeta_termino(termino, definicion, como_leerlo, advertencia)

tema.seccion("Diccionario de variables")
st.caption("Variables del conjunto de datos (fuente: fsai.etiquetas.DICCIONARIO). La columna «variable» es el nombre en los datos y «etiqueta» el nombre legible.")
filas = [{"variable": cod, "etiqueta": info.get("nombre", ""), "unidad": info.get("unidad", ""), "descripción": info.get("descripcion", "")}
         for cod, info in DICCIONARIO.items()]
tabla = pd.DataFrame(filas)
if clave:
    mascara = tabla.apply(lambda f: clave in normalizar_texto(" ".join(map(str, f.values))), axis=1)
    tabla = tabla[mascara]
st.dataframe(tabla, hide_index=True, use_container_width=True)
st.caption(f"{len(tabla)} de {len(filas)} variables.")
