<div align="center">

# foodsecurity-peru

**Modelos explicables de inseguridad alimentaria en hogares peruanos**
ENAHO 2025 · escala FIES · validación sin fuga · calibración · equidad · contexto territorial

[![Licencia: MIT](https://img.shields.io/badge/licencia-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Pruebas](https://img.shields.io/badge/pruebas-168%20OK-brightgreen.svg)
![Estado](https://img.shields.io/badge/estado-investigaci%C3%B3n-lightgrey.svg)

</div>

---

## Resumen

Este repositorio contiene el sistema que respalda el estudio *«Modelos explicables de inseguridad alimentaria en hogares peruanos: evaluación sin fuga de información e indicadores territoriales»*. Con 32 065 hogares de la Encuesta Nacional de Hogares (ENAHO) 2025, se comparan modelos de aprendizaje automático que clasifican el nivel de inseguridad alimentaria (escala FIES) a partir de variables socioeconómicas, y se evalúa si indicadores departamentales públicos (nutrición, pobreza multidimensional y clima) mejoran la predicción.

| Aspecto | Qué hace el sistema |
|---|---|
| **Validación sin fuga** | Partición 70/15/15 agrupada por hogar; validación cruzada anidada 5×3; la partición de prueba se evalúa una sola vez |
| **Optimización** | Búsqueda de hiperparámetros con Optuna (muestreo TPE y poda por mediana) restringida a los pliegues internos |
| **Modelos** | M1 regresión logística · M2 bosque aleatorio · M3 CatBoost · M4 apilamiento · M5 autoencoder + XGBoost · M6 HistGradientBoosting |
| **Calibración** | Regresión isotónica uno-contra-resto ajustada solo en validación |
| **Explicabilidad y equidad** | SHAP, importancia por permutación, PDP/ALE; brechas de sensibilidad por subgrupo con intervalos bootstrap |
| **Contexto territorial** | 13 indicadores departamentales (ENDES 2024, MPI, ERA5, ONI) con trazabilidad SHA-256 |
| **Aplicación** | Interfaz Streamlit con PostgreSQL, roles, auditoría, mapas, inferencia individual y por lote |

### Resultados principales

| Medida (partición de prueba, escenario socioeconómico) | Valor |
|---|---|
| AUC-ROC (cinco modelos) | 0,66 – 0,68 |
| F1 macro (cinco modelos) | 0,35 – 0,37 |
| Prevalencia ponderada de inseguridad moderada o severa | 26,8 % |
| Efecto de añadir indicadores territoriales sobre el F1 macro | −0,8 a +0,4 puntos porcentuales; los intervalos incluyen el cero |

La capacidad predictiva con variables socioeconómicas es **modesta**, ningún modelo resultó claramente superior y los indicadores territoriales **no mostraron una mejora demostrada**.

> **Uso responsable.** Los modelos describen asociaciones predictivas, no causas. No deben usarse para asignar ni negar ayudas a hogares individuales. Las métricas de equidad son una alerta, no una certificación.

## Estructura del repositorio

```text
foodsecurity-peru/
├── foodsecurity_ai/       Biblioteca fsai, pruebas y scripts de entrenamiento
│   ├── fsai/              Datos, FIES, particiones, modelos, validación anidada, métricas, equidad
│   ├── tests/             Suite pytest (datos sintéticos)
│   └── scripts/           Entrenamiento por fases y descarga de fuentes públicas
├── app_streamlit/         Aplicación web, Dockerfile y docker-compose
├── sql/                   Esquema PostgreSQL
├── scripts_operacion/     Respaldo, restauración y lanzadores de entrenamiento
├── datos_derivados/       Indicadores territoriales por departamento y manifiestos SHA-256
├── resultados_agregados/  Resultados agregados (sin microdatos)
├── diccionario/           Diccionario de variables
├── docs/                  Diseño de la aplicación y fuentes adicionales
└── referencia_v1/         Notas de la versión anterior (desviaciones y mejoras)
```

## Instalación y pruebas

Requisito: **Python 3.12**.

```bash
git clone https://github.com/jfloriana/foodsecurity-peru.git
cd foodsecurity-peru

python -m venv .venv
.venv\Scripts\activate                  # Linux/macOS: source .venv/bin/activate
pip install -r app_streamlit/requirements-lock.txt

python -m pytest foodsecurity_ai/tests -q
```

Las pruebas usan datos sintéticos: no necesitan PostgreSQL ni los microdatos reales. La guía completa (base de datos, variables de entorno, Docker y entrenamiento) está en [`LEEME.md`](LEEME.md).

## Datos

Este repositorio **no** redistribuye datos de terceros:

| Elemento | Dónde obtenerlo |
|---|---|
| Microdatos ENAHO 2025 y ENDES 2024 | [Portal de microdatos del INEI](https://proyectos.inei.gob.pe/microdatos/), con sus condiciones de uso |
| GeoJSON departamental | No se incluye porque la licencia de su origen no está declarada. Se descarga con `python foodsecurity_ai/scripts/descargar_fuentes_adicionales.py --solo geojson` y se indica con la variable `FSAI_GEOJSON_DEPARTAMENTOS`; sin él, los mapas de la aplicación no se muestran. Verifique su licencia antes de redistribuirlo |
| Base de datos, modelos entrenados y credenciales | No se versionan. Copie `app_streamlit/.env.example` a `.env` y genere usuarios con `python app_streamlit/app/auth.py` |

Sí se incluyen los indicadores territoriales derivados (`datos_derivados/`) y los resultados agregados (`resultados_agregados/`).

## Limitaciones

- Un solo año de encuesta (2025) y un solo país.
- Indicadores territoriales con solo 25 valores distintos (uno por departamento) y de 2023–2024, frente a la ENAHO 2025.
- Umbrales FIES 1-4-7 provisionales, sin la calibración oficial basada en el modelo de Rasch.
- 20 ensayos de Optuna en la corrida heredada y 30 en la comparación con indicadores territoriales; el protocolo prevé 100.
- La aplicación, el respaldo y el contenedor no se verificaron contra PostgreSQL ni Docker en una instalación limpia.

El detalle está en [`LEEME.md`](LEEME.md) y en [`referencia_v1/DESVIACIONES.md`](referencia_v1/DESVIACIONES.md).

## Cómo citar

Consulte [`CITATION.cff`](CITATION.cff); GitHub muestra la opción **«Cite this repository»** en la barra lateral. Cuando se publique el artículo, se añadirá su referencia aquí.

## Autores

| Autor | ORCID |
|---|---|
| **Florián Arévalo, Joel Anderson** (autor principal) | [0009-0002-8256-5779](https://orcid.org/0009-0002-8256-5779) |
| Gálvez Luna, Jason Anderson | [0009-0002-0198-2168](https://orcid.org/0009-0002-0198-2168) |

Escuela de Ingeniería de Sistemas, Universidad Nacional de Trujillo, Trujillo, Perú.

## Licencia

El código se distribuye bajo la licencia [MIT](LICENSE). Los datos de origen (INEI, OPHI, NOAA, Open-Meteo) conservan sus propias condiciones de uso.
