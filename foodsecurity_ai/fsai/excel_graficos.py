"""excel_graficos.py - Hojas adicionales del libro Excel del informe: Índice navegable, hojas de figuras (imagen + datos exactos +
gráfico nativo de Excel), Pronóstico, Diccionario ampliado y Metodología y límites.

Solo VUELCA datos ya calculados por ``fsai.reportes_fases`` (no calcula cifras). Contrato de entrada:

* Cada bloque ``figura`` del documento puede traer ``datos``: lista de ``{"titulo", "df", "chart"?, "color"?}`` con la tabla EXACTA que
  genera la figura. ``chart`` es un dict (ver ``_grafico``) que describe el gráfico nativo enlazado a esa tabla; ``color`` = "escala"
  aplica escala de color a las celdas numéricas (matrices de confusión).
* ``doc["excel_extra"]`` = ``{"pronostico": {...}, "metodologia": [(titulo, [texto, ...]), ...], "diccionario_extra": DataFrame}``.
"""
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl.chart import BarChart, Reference, ScatterChart, Series
from openpyxl.chart.data_source import NumDataSource, NumRef
from openpyxl.chart.error_bar import ErrorBars
from openpyxl.drawing.image import Image as XLImage
from openpyxl.formatting.rule import ColorScaleRule, DataBarRule, FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as L
from openpyxl.worksheet.datavalidation import DataValidation

AZUL = "1F4E79"
GRIS = "4F5B67"
PALETA_XL = ["0072B2", "E69F00", "009E73", "CC79A7", "D55E00", "56B4E9", "F0E442", "000000"]   # Okabe-Ito, la misma de las figuras PNG
COL_TABLA = 13            # las tablas de datos empiezan en la columna M; A-L quedan para la imagen y el gráfico
ALTO_FILA_PX = 20
_CAB_FILL = PatternFill("solid", fgColor=AZUL)
_CAB_FONT = Font(bold=True, color="FFFFFF")
_BORDE = Border(*(Side(style="thin", color="BFC7CF"),) * 4)
_ZEBRA = PatternFill("solid", fgColor="F2F6FA")

# hoja de destino de cada figura: nombre de archivo (sin .png) -> (hoja, descripción de la hoja)
GRUPOS_FIGURA: dict[str, tuple[str, str]] = {
    "fig_prevalencia_area": ("Fig_Prevalencia", "Prevalencia ponderada de inseguridad alimentaria moderada/severa por área, región natural y departamento (IC95), con las tablas que generan cada gráfico."),
    "fig_prevalencia_region": ("Fig_Prevalencia", ""), "fig_prevalencia_departamento": ("Fig_Prevalencia", ""),
    "fig_distribucion_fies": ("Fig_Distribucion_FIES", "Distribución de los niveles FIES (0 a 3), sin ponderar y ponderada; gráfico nativo de barras."),
    "fig_desempeno_modelos": ("Fig_Desempeno", "Desempeño de los modelos M1 a M5 en la validación cruzada anidada (media e IC95); gráfico nativo de barras con barras de error."),
    "fig_confusion_M1": ("Fig_Confusion", "Matrices de confusión (conteos y normalizadas por fila) de los modelos M1 a M5, con escala de color."),
    "fig_roc_severa": ("Fig_ROC", "Curvas ROC de la clase IA severa por modelo: puntos (FPR, TPR) exactos y gráfico nativo de dispersión."),
    "fig_pr_severa": ("Fig_PR", "Curvas precisión-recall de la clase IA severa por modelo: puntos exactos y gráfico nativo."),
    "fig_calibracion_severa": ("Fig_Calibracion", "Calibración de la probabilidad de IA severa por modelo: probabilidad media predicha frente a frecuencia observada por cuantil."),
    "fig_diferencia_critica": ("Fig_Dif_Critica", "Rangos medios de Friedman y diferencia crítica de Nemenyi para comparar los modelos."),
    "fig_shap_barras": ("Fig_SHAP", "Importancia global SHAP (|SHAP| medio) por variable, con nombres legibles; gráfico nativo de barras (15 principales)."),
    "fig_pdp_ale": ("Fig_PDP_ALE", "Valores numéricos de dependencia parcial (PDP) y efectos locales acumulados (ALE) que reproducen las figuras."),
    "fig_escenarios": ("Fig_Escenarios", "F1-macro ponderado por escenario y modelo (CV anidada, IC95); gráfico nativo de barras."),
    "fig_equidad": ("Fig_Equidad", "Brecha de sensibilidad (TPR) por grupo frente a su referencia, con IC95; gráfico nativo de barras."),
}


def grupo_de_figura(nombre: str) -> tuple[str, str]:
    if nombre in GRUPOS_FIGURA:
        return GRUPOS_FIGURA[nombre]
    if nombre.startswith("fig_confusion_"):
        return GRUPOS_FIGURA["fig_confusion_M1"]
    if nombre.startswith("fig_pdp_ale"):
        return GRUPOS_FIGURA["fig_pdp_ale"]
    return ("Fig_" + nombre.replace("fig_", "")[:24], "Figura del informe con su tabla de datos.")


# ------------------------------ utilidades de escritura ------------------------------
def _py(v: Any) -> Any:
    """Convierte tipos numpy/pandas a tipos nativos; NaN/None -> None."""
    if v is None:
        return None
    if isinstance(v, (np.bool_, bool)):
        return "Sí" if bool(v) else "No"
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v if isinstance(v, (int, str)) else str(v)


def _celda(ws, fila: int, col: int, valor: Any, negrita: bool = False, fmt: str | None = None):
    c = ws.cell(fila, col, _py(valor))
    if negrita:
        c.font = Font(bold=True)
    if fmt:
        c.number_format = fmt
    return c


def _formato(v: Any) -> str | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return "#,##0"
    if isinstance(v, float):
        return "0.0000"
    return None


def escribir_tabla(ws, fila: int, col: int, df: pd.DataFrame, titulo: str | None = None) -> tuple[int, int, dict[str, int]]:
    """Escribe ``df`` con encabezado. Devuelve (fila_encabezado, fila_final, {columna: índice de columna})."""
    if titulo:
        _celda(ws, fila, col, titulo, negrita=True)
        fila += 1
    cols: dict[str, int] = {}
    for j, nombre in enumerate(df.columns):
        c = ws.cell(fila, col + j, str(nombre))
        c.font, c.fill, c.border = _CAB_FONT, _CAB_FILL, _BORDE
        c.alignment = Alignment(wrap_text=True, vertical="center")
        cols[str(nombre)] = col + j
    cab = fila
    for r, reg in enumerate(df.itertuples(index=False), start=1):
        for j, v in enumerate(reg):
            c = _celda(ws, cab + r, col + j, v, fmt=_formato(_py(v)))
            c.border = _BORDE
            if r % 2:
                c.fill = _ZEBRA
    for j, nombre in enumerate(df.columns):
        letra = L(col + j)
        ancho = max([len(str(nombre))] + [len(str(_py(x))) for x in df.iloc[:60, j] if _py(x) is not None]) + 2
        actual = ws.column_dimensions[letra].width or 8.43
        ws.column_dimensions[letra].width = max(actual, min(max(ancho, 10), 48))
    return cab, cab + len(df), cols


def _enlace_indice(ws, celda: str = "A2") -> None:
    c = ws[celda]
    c.value = "← Volver al Índice"
    c.hyperlink = "#'Índice'!A1"
    c.font = Font(color="0563C1", underline="single")


def _ref_formula(ws, col: int, r1: int, r2: int) -> str:
    return f"'{ws.title}'!${L(col)}${r1}:${L(col)}${r2}"


# ------------------------------ gráficos nativos ------------------------------
def _grafico(ws, spec: dict[str, Any], cab: int, fin: int, cols: dict[str, int]):
    """Gráfico nativo enlazado a la tabla escrita. ``spec``: tipo ('barra'|'barra_h'|'dispersion'), titulo, x_titulo, y_titulo,
    cat (columna de categorías), series [{col, x?, nombre?, err_mas?, err_menos?, marcador?}], max_filas?"""
    n_max = spec.get("max_filas")
    ultima = fin if not n_max else min(fin, cab + int(n_max))
    if ultima <= cab:
        return None
    if spec["tipo"] in ("barra", "barra_h"):
        ch = BarChart()
        ch.type = "bar" if spec["tipo"] == "barra_h" else "col"
        ch.grouping = "clustered"
        cats = Reference(ws, min_col=cols[spec["cat"]], min_row=cab + 1, max_row=ultima)
        for s in spec["series"]:
            serie = Series(Reference(ws, min_col=cols[s["col"]], min_row=cab + 1, max_row=ultima), title=s.get("nombre", s["col"]))
            serie.graphicalProperties.solidFill = s.get("color") or PALETA_XL[len(ch.series) % len(PALETA_XL)]
            serie.graphicalProperties.line.solidFill = serie.graphicalProperties.solidFill
            if s.get("err_mas") and s.get("err_menos"):
                serie.errBars = ErrorBars(errDir="y", errBarType="both", errValType="cust",
                                          plus=NumDataSource(numRef=NumRef(f=_ref_formula(ws, cols[s["err_mas"]], cab + 1, ultima))),
                                          minus=NumDataSource(numRef=NumRef(f=_ref_formula(ws, cols[s["err_menos"]], cab + 1, ultima))))
            ch.series.append(serie)
        ch.set_categories(cats)
        if spec["tipo"] == "barra_h":
            ch.x_axis.scaling.orientation = "maxMin"
    else:
        ch = ScatterChart()
        ch.style = 13
        ch.scatterStyle = "lineMarker"
        for s in spec["series"]:
            xs = Reference(ws, min_col=cols[s["x"]], min_row=cab + 1, max_row=ultima)
            ys = Reference(ws, min_col=cols[s["col"]], min_row=cab + 1, max_row=ultima)
            serie = Series(ys, xs, title=s.get("nombre", s["col"]))
            serie.smooth = False
            serie.graphicalProperties.line.solidFill = s.get("color") or PALETA_XL[len(ch.series) % len(PALETA_XL)]
            serie.graphicalProperties.line.width = 19050
            if s.get("marcador"):
                serie.marker.symbol = "circle"
                serie.marker.size = 5
                serie.marker.graphicalProperties.solidFill = serie.graphicalProperties.line.solidFill
                serie.marker.graphicalProperties.line.solidFill = serie.graphicalProperties.line.solidFill
            else:
                serie.marker.symbol = "none"
            if s.get("punteada"):
                serie.graphicalProperties.line.dashStyle = "dash"
                serie.graphicalProperties.line.solidFill = "808080"
                serie.graphicalProperties.line.width = 12700
            ch.series.append(serie)
        ch.x_axis.scaling.min, ch.x_axis.scaling.max = spec.get("xlim", (0, 1))
        ch.y_axis.scaling.min, ch.y_axis.scaling.max = spec.get("ylim", (0, 1))
    if spec["tipo"] in ("barra", "barra_h"):
        # escala honesta: las barras parten de 0 (salvo que haya valores negativos) y las proporciones llegan a 1 como máximo
        vals = [ws.cell(r, cols[s["col"]]).value for s in spec["series"] for r in range(cab + 1, ultima + 1)]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if vals and min(vals) >= 0:
            ch.y_axis.scaling.min = 0
            if 0.5 < max(vals) <= 1.0:
                ch.y_axis.scaling.max = 1
        ch.x_axis.tickLblPos = "low"            # nombres de categoría fuera de las barras aun con valores negativos
        if spec["tipo"] == "barra_h":
            ch.y_axis.crosses = "max"           # con el eje de categorías invertido, el eje de valores queda abajo
        ch.y_axis.number_format = "General"
        if len(spec["series"]) == 1:
            ch.legend = None
    if spec["tipo"] == "dispersion":
        ch.x_axis.number_format = "0.0#"
        ch.y_axis.number_format = "0.0#"
    ch.title = spec.get("titulo")
    ch.x_axis.title = spec.get("x_titulo")
    ch.y_axis.title = spec.get("y_titulo")
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    if ch.legend is not None:
        ch.legend.position = "b"
    ch.width, ch.height = spec.get("ancho_cm", 17.5), spec.get("alto_cm", 9.0)
    return ch


# ------------------------------ hojas de figuras ------------------------------
def _imagen(ruta: Path, ancho_px: int, guardar: list):
    from PIL import Image
    with Image.open(ruta) as im:
        w, h = im.size
        reducida = im.convert("RGB") if im.mode not in ("RGB", "RGBA") else im.copy()
    if w > 1300:
        reducida = reducida.resize((1300, int(h * 1300 / w)))
    buf = io.BytesIO()
    reducida.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    guardar.append(buf)
    xl = XLImage(buf)
    xl.width, xl.height = ancho_px, int(ancho_px * h / w)
    return xl


ANCHO_AREA_CHARS = 12 * 8.43        # columnas A-L (área de impresión de las hojas de figuras)
ALTO_PAGINA_PX = 980                # alto útil aproximado de una página A4 vertical con el área A-L ajustada al ancho


def _envolver(ws, fila: int, ultima_col: int = 12, tam: float = 11.0) -> None:
    """Combina A..ultima_col de la fila, activa el ajuste de texto y fija la altura (si no, el texto se corta al imprimir)."""
    c = ws.cell(fila, 1)
    n = len(str(c.value or ""))
    por_linea = max(30, int(ANCHO_AREA_CHARS * 1.05 * 11.0 / tam))
    lineas = max(1, -(-n // por_linea))
    ws.merge_cells(start_row=fila, start_column=1, end_row=fila, end_column=ultima_col)
    c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[fila].height = (tam * 1.35) * lineas + 3


def _px_filas(ws, r1: int, r2: int) -> float:
    return sum((ws.row_dimensions[r].height or 15.0) * 4 / 3 for r in range(r1, r2 + 1))


def _pie_texto(pie: Any) -> list[str]:
    if not pie:
        return []
    if isinstance(pie, str):
        return [pie]
    return [f"{etq}. {pie[k]}" for k, etq in (("que_muestra", "Qué muestra"), ("como_leerlo", "Cómo leerlo"), ("limitacion", "Limitación")) if pie.get(k)]


def hojas_figuras(wb, doc: dict[str, Any], indice: list[tuple[str, str, str]], buffers: list) -> None:
    """Crea una hoja por grupo de figuras: imagen a la izquierda, tablas exactas a la derecha y gráfico nativo bajo la imagen."""
    hojas: dict[str, Any] = {}
    cursor: dict[str, int] = {}
    meta = getattr(wb, "_fsai_figs", None)
    if meta is None:
        meta = wb._fsai_figs = {}
    for sec in doc.get("secciones", []):
        for b in sec.get("bloques", []):
            if b["tipo"] != "figura" or not b.get("datos") or not b.get("ruta") or not Path(b["ruta"]).is_file():
                continue
            nombre_png = Path(b["ruta"]).stem
            hoja, desc = grupo_de_figura(nombre_png)
            if hoja not in hojas:
                ws = wb.create_sheet(hoja)
                hojas[hoja] = ws
                cursor[hoja] = 4
                ws["A1"] = f"Datos y gráficos · {hoja.replace('_', ' ')}"
                ws["A1"].font = Font(bold=True, size=14, color=AZUL)
                _enlace_indice(ws)
                ws["A3"] = desc
                ws["A3"].font = Font(italic=True, color=GRIS)
                _envolver(ws, 3)
                indice.append((hoja, "Figura + datos", desc))
            ws = hojas[hoja]
            f0 = cursor[hoja]
            meta.setdefault(hoja, {"cortes": [], "pag_ini": 1})
            if f0 > 4:
                meta[hoja]["cortes"].append(f0 - 1)     # salto de página antes de cada figura (salvo la primera)
                meta[hoja]["pag_ini"] = f0
            _celda(ws, f0, 1, f"Figura {b['num']}. {b['titulo']}", negrita=True)
            ws.cell(f0, 1).font = Font(bold=True, size=11, color=AZUL)
            fila_pie = f0 + 1
            for t in _pie_texto(b.get("pie")):
                c = ws.cell(fila_pie, 1, t)
                c.font = Font(italic=True, color=GRIS, size=9)
                _envolver(ws, fila_pie, tam=9.0)
                fila_pie += 1
            f_img = fila_pie + 1
            ancho = 740 if b.get("ancho") == "doble" else 540
            img = _imagen(Path(b["ruta"]), ancho, buffers)
            ws.add_image(img, f"A{f_img}")
            fin_img = f_img + math.ceil(img.height / ALTO_FILA_PX) + 1
            fila_t = f_img
            fin_chart = fin_img
            for d in b["datos"]:
                cab, fin, cols = escribir_tabla(ws, fila_t, COL_TABLA, d["df"], d.get("titulo"))
                if d.get("color") == "escala":
                    primero, ultimo = L(COL_TABLA + 1), L(COL_TABLA + len(d["df"].columns) - 1)
                    ws.conditional_formatting.add(f"{primero}{cab + 1}:{ultimo}{fin}", ColorScaleRule(start_type="min", start_color="FFFFFF", end_type="max", end_color="2E75B6"))
                if d.get("chart"):
                    ch = _grafico(ws, d["chart"], cab, fin, cols)
                    if ch is not None:
                        alto_px = ch.height * 37.8
                        if _px_filas(ws, meta[hoja]["pag_ini"], fin_chart - 1) + alto_px > ALTO_PAGINA_PX:
                            meta[hoja]["cortes"].append(fin_chart - 1)      # el gráfico no cabe bajo lo anterior: página nueva
                            meta[hoja]["pag_ini"] = fin_chart
                        ws.add_chart(ch, f"A{fin_chart}")
                        fin_chart += math.ceil(ch.height * 37.8 / ALTO_FILA_PX) + 2
                fila_t = fin + 3
            cursor[hoja] = max(fin_chart, fila_t, fin_img) + 3
            meta[hoja]["fin_figura"] = max(fin_chart, fin_img)
        # (las figuras sin datos no generan hoja)


# ------------------------------ Pronóstico ------------------------------
def hoja_pronostico(wb, p: dict[str, Any], indice: list[tuple[str, str, str]]) -> None:
    """``p``: {df, cab_legible: [str], cab_codigo: [str], titulo, nota, col_prob: [idx], col_clase: [idx], col_acierto: idx}."""
    ws = wb.create_sheet("Pronóstico")
    df: pd.DataFrame = p["df"]
    n_col = len(df.columns)
    ws["A1"] = p["titulo"]
    ws["A1"].font = Font(bold=True, size=14, color=AZUL)
    _enlace_indice(ws)
    ws["A3"] = p["nota"]
    ws["A3"].font = Font(italic=True, color=GRIS)
    ws["A3"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A3:L4")
    ws.row_dimensions[3].height = 42
    ws.row_dimensions[4].height = 42
    f_leg, f_cod = 5, 6
    for j in range(n_col):
        for fila, texto in ((f_leg, p["cab_legible"][j]), (f_cod, p["cab_codigo"][j])):
            c = ws.cell(fila, j + 1, texto)
            c.font = _CAB_FONT if fila == f_leg else Font(italic=True, color=AZUL)
            c.fill = _CAB_FILL if fila == f_leg else PatternFill("solid", fgColor="DCE6F1")
            c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
            c.border = _BORDE
    ws.row_dimensions[f_leg].height = 48
    for r, reg in enumerate(df.itertuples(index=False), start=f_cod + 1):
        for j, v in enumerate(reg):
            _celda(ws, r, j + 1, v, fmt=("0.0000" if j in p["col_prob"] or j == p["col_peso"] else _formato(_py(v))))
    ultima = f_cod + len(df)
    ws.freeze_panes = ws.cell(f_cod + 1, 3)
    ws.auto_filter.ref = f"A{f_cod}:{L(n_col)}{ultima}"
    for j in range(n_col):
        ancho = max(len(str(p["cab_legible"][j])) // 2 + 2, len(str(p["cab_codigo"][j])) + 2, 11)
        ws.column_dimensions[L(j + 1)].width = min(ancho, 26)
    ws.column_dimensions["A"].width = 14
    rng = lambda j: f"{L(j + 1)}{f_cod + 1}:{L(j + 1)}{ultima}"  # noqa: E731
    for j in p["col_prob"]:
        ws.conditional_formatting.add(rng(j), ColorScaleRule(start_type="num", start_value=0, start_color="FFFFFF", end_type="num", end_value=1, end_color="F4B183"))
    ws.conditional_formatting.add(rng(p["col_prob"][-1]), DataBarRule(start_type="num", start_value=0, end_type="num", end_value=1, color="C00000"))
    j1, j2, ja = p["col_clase"][0], p["col_clase"][1], p["col_acierto"]
    ws.conditional_formatting.add(f"{L(j1 + 1)}{f_cod + 1}:{L(ja + 1)}{ultima}",
                                  FormulaRule(formula=[f"${L(j1 + 1)}{f_cod + 1}<>${L(j2 + 1)}{f_cod + 1}"], fill=PatternFill("solid", bgColor="F8CBAD", fgColor="F8CBAD")))
    dv_p = DataValidation(type="decimal", operator="between", formula1="0", formula2="1", showErrorMessage=True,
                          errorTitle="Probabilidad", error="Las probabilidades deben estar entre 0 y 1.")
    dv_c = DataValidation(type="whole", operator="between", formula1="0", formula2="3", showErrorMessage=True,
                          errorTitle="Nivel FIES", error="El nivel FIES debe ser un entero de 0 a 3.")
    for j in p["col_prob"]:
        dv_p.add(rng(j))
    for j in p["col_clase"]:
        dv_c.add(rng(j))
    ws.add_data_validation(dv_p)
    ws.add_data_validation(dv_c)
    indice.append(("Pronóstico", "Datos del modelo", f"Predicciones por hogar ({len(df):,} filas): probabilidad por clase, clase predicha y observada, peso muestral y variables explicativas principales. Filtros, paneles congelados y formato condicional."))


# ------------------------------ Diccionario ampliado y Metodología ------------------------------
def completar_diccionario(wb, extra: pd.DataFrame | None) -> None:
    if extra is None or extra.empty or "Diccionario" not in wb.sheetnames:
        return
    ws = wb["Diccionario"]
    escribir_tabla(ws, ws.max_row + 3, 1, extra, "Campos de la hoja Pronóstico y de las hojas de gráficos")


def hoja_metodologia(wb, secciones: list[tuple[str, list[str]]], indice: list[tuple[str, str, str]]) -> None:
    ws = wb.create_sheet("Metodología y límites")
    ws["A1"] = "Metodología y límites de interpretación"
    ws["A1"].font = Font(bold=True, size=14, color=AZUL)
    _enlace_indice(ws)
    fila = 4
    for titulo, items in secciones:
        ws.cell(fila, 1, titulo).font = Font(bold=True, size=12, color=AZUL)
        fila += 1
        for it in items:
            c = ws.cell(fila, 1, "• " + it)
            c.alignment = Alignment(wrap_text=True, vertical="top")
            ws.row_dimensions[fila].height = max(18, 15 * (len(it) // 120 + 1))
            fila += 1
        fila += 1
    ws.column_dimensions["A"].width = 130
    indice.append(("Metodología y límites", "Referencia", "Cómo se obtuvieron las cifras y qué advertencias deben leerse antes de usarlas: desempeño modesto, intervalos aproximados, [POR VALIDAR] de la escala FIES/Rasch."))


# ------------------------------ Índice y ensamblado ------------------------------
DESCRIPCIONES_BASE = {
    "Portada": ("Informe", "Datos de identificación del informe: dataset, corrida, escenario, fecha, hash de configuración y semilla."),
    "Resumen": ("Informe", "Resumen ejecutivo generado desde los datos."),
    "1_Datos": ("Informe", "Tamaño muestral, prevalencias ponderadas con IC95, distribución FIES y diccionario resumido."),
    "2_Metodos": ("Informe", "Modelos, partición, esquemas de validación, hiperparámetros y versiones de software."),
    "3_Resultados": ("Informe", "Desempeño por modelo (CV anidada), recall de IA severa, discriminación y calibración, resultados por esquema."),
    "4_Comparacion": ("Informe", "Pruebas estadísticas entre modelos (Holm) y tamaños de efecto."),
    "5_Interpretabilidad": ("Informe", "Importancia SHAP, PDP/ALE y desempeño por escenario."),
    "6_Equidad": ("Informe", "Brechas de sensibilidad, FPR y error por grupo (sexo, área, región, edad, educación, pobreza)."),
    "7_Limitaciones": ("Informe", "Uso responsable y limitaciones técnicas del análisis."),
    "8_Reproducibilidad": ("Informe", "Hashes de datos y artefactos y comandos para reproducir."),
    "Diccionario": ("Referencia", "Variables del dataset: nombre legible, código original, unidad, tipo y descripción; incluye los campos de las hojas de datos."),
    "Figuras": ("Referencia", "Lista de figuras del informe y su archivo PNG de 600 DPI."),
    "Metadatos": ("Referencia", "Metadatos de generación del libro."),
}


def hoja_indice(wb, indice: list[tuple[str, str, str]]) -> None:
    ws = wb.create_sheet("Índice", 0)
    ws["A1"] = "Índice del libro"
    ws["A1"].font = Font(bold=True, size=16, color=AZUL)
    ws["A2"] = "Haga clic en el nombre de una hoja para ir a ella; cada hoja tiene un enlace «Volver al Índice»."
    ws["A2"].font = Font(italic=True, color=GRIS)
    for j, h in enumerate(["Hoja", "Tipo", "Contenido"], start=1):
        c = ws.cell(4, j, h)
        c.font, c.fill, c.border = _CAB_FONT, _CAB_FILL, _BORDE
    orden = {n: i for i, (n, _, _) in enumerate(indice)}
    existentes = [n for n in wb.sheetnames if n != "Índice"]
    desc = {n: (t, d) for n, t, d in indice}
    for n in existentes:
        if n not in desc and n in DESCRIPCIONES_BASE:
            desc[n] = DESCRIPCIONES_BASE[n]
    fila = 5
    for n in wb.sheetnames:
        if n == "Índice":
            continue
        t, d = desc.get(n, ("Informe", "Hoja del informe."))
        c = ws.cell(fila, 1, n)
        c.hyperlink = f"#'{n}'!A1"
        c.font = Font(color="0563C1", underline="single")
        ws.cell(fila, 2, t)
        e = ws.cell(fila, 3, d)
        e.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[fila].height = max(18, 15 * (len(d) // 100 + 1))
        for j in (1, 2, 3):
            ws.cell(fila, j).border = _BORDE
        fila += 1
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 110
    ws.freeze_panes = "A5"
    del orden


def configurar_impresion(wb) -> None:
    """Define área, orientación, ajuste a una página de ancho, encabezados repetidos y pie en TODAS las hojas, para que el PDF
    de Excel/LibreOffice no genere cientos de páginas. Hojas de figuras: se imprime solo la imagen y los gráficos nativos
    (columnas A-L, un salto de página por figura); las tablas de datos exactos (columna M en adelante) quedan en el libro."""
    from openpyxl.worksheet.pagebreak import Break
    from openpyxl.worksheet.properties import PageSetupProperties
    meta = getattr(wb, "_fsai_figs", {}) or {}
    for ws in wb.worksheets:
        ws.page_setup.paperSize = ws.PAPERSIZE_A4
        ws.page_setup.orientation = "portrait" if ws.title in meta else "landscape"
        ws.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        ws.page_margins.left = ws.page_margins.right = 0.4
        ws.page_margins.top, ws.page_margins.bottom = 0.5, 0.55
        ws.print_options.horizontalCentered = False
        ws.oddFooter.left.text = "FoodSecurity AI · &A"
        ws.oddFooter.right.text = "Página &P de &N"
        ws.oddFooter.left.size = ws.oddFooter.right.size = 8
        if ws.title in meta and meta[ws.title].get("fin_figura"):
            m = meta[ws.title]
            ws.print_area = f"A1:L{m['fin_figura']}"
            for r in m["cortes"]:
                ws.row_breaks.append(Break(id=r))
        elif ws.title == "Pronóstico":
            ws.print_title_rows = "5:6"
            ws.page_setup.paperSize = ws.PAPERSIZE_A3        # 21 columnas: A3 horizontal mantiene legible el texto al ajustar a 1 página de ancho
            ws.print_area = f"A1:{L(ws.max_column)}{ws.max_row}"
        elif ws.title == "Índice":
            ws.print_title_rows = "4:4"


def completar_libro(wb, doc: dict[str, Any]) -> None:
    """Añade al libro de ``exportar_excel`` las hojas de gráficos, Pronóstico, Metodología e Índice y los enlaces de retorno."""
    extra = doc.get("excel_extra") or {}
    indice: list[tuple[str, str, str]] = []
    buffers: list = []
    wb._fsai_buffers = buffers   # mantiene vivas las imágenes hasta guardar
    if "9_Diccionario" in wb.sheetnames:
        wb["9_Diccionario"].title = "Diccionario"
    completar_diccionario(wb, extra.get("diccionario_extra"))
    if extra.get("metodologia"):
        hoja_metodologia(wb, extra["metodologia"], indice)
    if extra.get("pronostico"):
        hoja_pronostico(wb, extra["pronostico"], indice)
    hojas_figuras(wb, doc, indice, buffers)
    # orden final: Índice, Portada, Resumen, secciones, Diccionario, Metodología, Pronóstico, hojas de figuras, Figuras, Metadatos
    base = ["Portada", "Resumen"] + [n for n in wb.sheetnames if n[:1].isdigit()]
    fin = ["Diccionario", "Metodología y límites", "Pronóstico"] + [n for n in wb.sheetnames if n.startswith("Fig_")] + ["Figuras", "Metadatos"]
    orden = [n for n in base + fin if n in wb.sheetnames]
    wb._sheets = [wb[n] for n in orden] + [s for s in wb._sheets if s.title not in orden]
    hoja_indice(wb, indice)
    celdas_retorno = {"Portada": "A3", "Figuras": "F1", "Metadatos": "D1"}
    for ws in wb.worksheets:
        if ws.title == "Índice" or ws["A2"].value == "← Volver al Índice":
            continue
        _enlace_indice(ws, celdas_retorno.get(ws.title, "A2"))
    configurar_impresion(wb)
