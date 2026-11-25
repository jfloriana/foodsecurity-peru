"""report_generator.py - Exportación de informes de nivel de manuscrito a Excel (openpyxl), Word (python-docx) y PDF (reportlab).

Los generadores solo VUELCAN resultados ya calculados: no calculan ni imputan cifras.

Modelo de documento (``doc``)::

    {"titulo": str, "subtitulo": str, "encabezado": str,
     "portada": [(clave, valor), ...],          # tabla de la portada (dataset, fecha, versión, hash, semilla...)
     "resumen": [str, ...],                     # resumen ejecutivo (párrafos generados desde datos)
     "secciones": [{"titulo": "1. ...", "hoja": "1_Datos", "bloques": [bloque, ...]}, ...]}

Bloques: ``{"tipo": "p", "texto"}``, ``{"tipo": "h2", "texto"}``, ``{"tipo": "lista", "items"}``,
``{"tipo": "codigo", "texto"}``, ``{"tipo": "vacio", "texto"}`` (sección sin datos),
``{"tipo": "tabla", "titulo", "df", "pie", "max_filas"}`` y ``{"tipo": "figura", "titulo", "ruta", "pie", "ancho"}``
(``ancho`` = "simple" 8,5 cm o "doble" columna completa). ``pie`` es un texto o el dict de ``fsai.interpretacion``
(``que_muestra``, ``como_leerlo``, ``limitacion``). Tablas y figuras se numeran automáticamente.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

MAX_FILAS_TABLA = 40   # filas por tabla en Word/PDF (el Excel conserva todo)
ETIQUETAS_PIE = (("que_muestra", "Qué muestra"), ("como_leerlo", "Cómo leerlo"), ("limitacion", "Limitación"))
AZUL = "1F4E79"
_PDF_SEGURO = {"≥": ">=", "≤": "<=", "≈": "~", "→": "->", "χ": "chi", "−": "-", "’": "'", "“": '"', "”": '"', "σ": "sigma",
               "Δ": "Delta", "±": "+/-", "×": "x", "‑": "-", "–": "-", "—": "-", "…": "...", "²": "2", "·": "·", "κ": "kappa", "ρ": "rho"}


# ------------------------------ utilidades ------------------------------
def _pdf_seguro(t: str) -> str:
    """Sustituye símbolos fuera de WinAnsi (fuentes estándar de reportlab) por equivalentes ASCII."""
    for a, b in _PDF_SEGURO.items():
        t = t.replace(a, b)
    return t


def _formatear(valor: Any) -> str:
    if valor is None:
        return ""
    if isinstance(valor, bool):
        return "Sí" if valor else "No"
    if isinstance(valor, float):
        if pd.isna(valor):
            return ""
        return f"{valor:,.0f}" if abs(valor) >= 10000 else f"{valor:.4f}"
    if isinstance(valor, int):
        return f"{valor:,}" if abs(valor) >= 1000 else str(valor)
    try:
        if pd.isna(valor):
            return ""
    except (TypeError, ValueError):
        pass
    return str(valor)


def _pie_partes(pie: Any) -> list[tuple[str, str]]:
    if not pie:
        return []
    if isinstance(pie, str):
        return [("", pie)]
    return [(etq, str(pie[k])) for k, etq in ETIQUETAS_PIE if pie.get(k)]


def numerar(doc: dict[str, Any]) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Asigna ``num`` a cada tabla y figura; devuelve (lista de tablas, lista de figuras) como (n, título)."""
    nt = nf = 0
    tablas, figuras = [], []
    for sec in doc.get("secciones", []):
        for b in sec.get("bloques", []):
            if b["tipo"] == "tabla":
                nt += 1
                b["num"] = nt
                tablas.append((nt, b["titulo"]))
            elif b["tipo"] == "figura":
                nf += 1
                b["num"] = nf
                figuras.append((nf, b["titulo"]))
    return tablas, figuras


def _recorte(b: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    df: pd.DataFrame = b["df"]
    tope = int(b.get("max_filas") or MAX_FILAS_TABLA)
    return df.head(tope), len(df)


def _figura_existe(b: dict[str, Any]) -> bool:
    return bool(b.get("ruta")) and Path(b["ruta"]).is_file()


# ------------------------------ Excel ------------------------------
def exportar_excel(doc: dict[str, Any], ruta: str | Path, metadatos: dict[str, Any] | None = None) -> Path:
    """Libro .xlsx: Portada, Resumen, una hoja por sección (tablas apiladas con título y pie), Figuras y Metadatos."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    numerar(doc)
    cab_fill = PatternFill("solid", fgColor=AZUL)
    cab_font = Font(bold=True, color="FFFFFF")
    borde = Border(*(Side(style="thin", color="BFC7CF"),) * 4)
    zebra = PatternFill("solid", fgColor="F2F6FA")
    envolver = Alignment(wrap_text=True, vertical="top")

    wb = Workbook()
    ws = wb.active
    ws.title = "Portada"
    ws["A1"] = doc["titulo"]
    ws["A1"].font = Font(bold=True, size=16, color=AZUL)
    ws["A2"] = doc.get("subtitulo", "")
    for i, (k, v) in enumerate(doc.get("portada", []), start=4):
        ws.cell(i, 1, str(k)).font = Font(bold=True)
        ws.cell(i, 2, _formatear(v) if not isinstance(v, str) else v).alignment = envolver
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 90

    wr = wb.create_sheet("Resumen")
    wr["A1"] = "Resumen ejecutivo"
    wr["A1"].font = Font(bold=True, size=14, color=AZUL)
    for i, par in enumerate(doc.get("resumen", []) or ["Sin datos disponibles para el resumen ejecutivo."], start=3):
        c = wr.cell(i, 1, par)
        c.alignment = envolver
        wr.row_dimensions[i].height = max(30, 15 * (len(par) // 110 + 1))
    wr.column_dimensions["A"].width = 120

    nombres_usados = {"Portada", "Resumen"}
    for sec in doc.get("secciones", []):
        base = re.sub(r"[\[\]\*\?/\\:]", "_", sec.get("hoja") or sec["titulo"])[:28]
        nombre, k = base, 2
        while nombre in nombres_usados:
            nombre, k = f"{base[:26]}_{k}", k + 1
        nombres_usados.add(nombre)
        w = wb.create_sheet(nombre)
        fila = 1
        w.cell(fila, 1, sec["titulo"]).font = Font(bold=True, size=14, color=AZUL)
        fila += 2
        anchos: dict[int, int] = {}
        textos: list[tuple[int, int]] = []        # (fila, nº de caracteres) de los textos libres de la columna A, para ajustarlos al ancho
        ncols = 0
        for b in sec.get("bloques", []):
            t = b["tipo"]
            if t in ("p", "h2", "vacio", "codigo"):
                c = w.cell(fila, 1, b["texto"])
                c.font = Font(bold=(t == "h2"), italic=(t == "vacio"))
                textos.append((fila, len(b["texto"])))
                fila += 1
            elif t == "lista":
                for it in b["items"]:
                    w.cell(fila, 1, f"• {it}")
                    textos.append((fila, len(it) + 2))
                    fila += 1
            elif t == "figura":
                w.cell(fila, 1, f"Figura {b['num']}. {b['titulo']}").font = Font(bold=True)
                fila += 1
                for etq, txt in _pie_partes(b.get("pie")):
                    w.cell(fila, 1, f"{etq}. {txt}" if etq else txt).font = Font(italic=True, color="4F5B67")
                    textos.append((fila, len(txt) + 14))
                    fila += 1
                nota = "(La figura y su tabla de datos están en la hoja de figuras de este libro; el PNG a 600 DPI se entrega aparte.)" if b.get("datos") else "(La figura PNG a 600 DPI se entrega como archivo aparte.)"
                w.cell(fila, 1, nota).font = Font(italic=True, color="4F5B67")
                fila += 1
            elif t == "tabla":
                df: pd.DataFrame = b["df"]
                ncols = max(ncols, len(df.columns))
                w.cell(fila, 1, f"Tabla {b['num']}. {b['titulo']}").font = Font(bold=True)
                textos.append((fila, len(b["titulo"]) + 12))
                fila += 1
                for j, col in enumerate(df.columns, start=1):
                    c = w.cell(fila, j, str(col))
                    c.font, c.fill, c.border = cab_font, cab_fill, borde
                    c.alignment = Alignment(wrap_text=True, vertical="center")
                    anchos[j] = max(anchos.get(j, 8), min(len(str(col)) // 2 + 4, 24))
                fila += 1
                for r, reg in enumerate(df.itertuples(index=False)):
                    for j, v in enumerate(reg, start=1):
                        if v is None or (isinstance(v, float) and pd.isna(v)):
                            c = w.cell(fila, j, None)
                        elif isinstance(v, (bool,)):
                            c = w.cell(fila, j, "Sí" if v else "No")
                        elif isinstance(v, (int, float)) and not isinstance(v, bool):
                            c = w.cell(fila, j, v)
                            c.number_format = "#,##0" if isinstance(v, int) or float(v).is_integer() and abs(v) >= 1000 else "0.0000"
                        else:
                            c = w.cell(fila, j, str(v))
                            c.alignment = envolver              # texto largo: se ajusta dentro de la celda (el ancho de columna se acota)
                            anchos[j] = max(anchos.get(j, 8), min(len(str(v)) + 2, 36))
                        c.border = borde
                        if r % 2:
                            c.fill = zebra
                    fila += 1
                for etq, txt in _pie_partes(b.get("pie")):
                    w.cell(fila, 1, f"{etq}. {txt}" if etq else txt).font = Font(italic=True, color="4F5B67")
                    textos.append((fila, len(txt) + 14))
                    fila += 1
                fila += 1
        w.column_dimensions["A"].width = max(anchos.get(1, 14), 14)
        for j, a in anchos.items():
            w.column_dimensions[get_column_letter(j)].width = max(w.column_dimensions[get_column_letter(j)].width or 0, a, 10)
        # textos libres: combinar A..última columna de tabla, ajustar y fijar la altura de fila (si no, se cortan al imprimir)
        if ncols < 2:
            w.column_dimensions["A"].width = max(w.column_dimensions["A"].width or 0, 120)
            ncols = 1
        ancho_total = sum(w.column_dimensions[get_column_letter(j)].width or 8.43 for j in range(1, ncols + 1))
        for f, n in textos:
            if ncols > 1:
                w.merge_cells(start_row=f, start_column=1, end_row=f, end_column=ncols)
            w.cell(f, 1).alignment = Alignment(wrap_text=True, vertical="top")
            w.row_dimensions[f].height = 15.5 * max(1, -(-int(n * 1.12) // max(int(ancho_total), 40)))

    wf = wb.create_sheet("Figuras")
    for j, h in enumerate(["Figura", "Título", "Archivo PNG (600 DPI)", "Ancho"], start=1):
        c = wf.cell(1, j, h)
        c.font, c.fill = cab_font, cab_fill
    i = 2
    for sec in doc.get("secciones", []):
        for b in sec.get("bloques", []):
            if b["tipo"] == "figura":
                wf.append([f"Figura {b['num']}", b["titulo"], Path(b["ruta"]).name if b.get("ruta") else "", b.get("ancho", "simple")])
                i += 1
    for col, a in zip("ABCD", (12, 80, 40, 10)):
        wf.column_dimensions[col].width = a

    wm = wb.create_sheet("Metadatos")
    meta = {"generado_en": datetime.now().isoformat(timespec="seconds"), **(metadatos or {})}
    meta["nota"] = "Las celdas contienen valores calculados por el pipeline y leídos de PostgreSQL/artefactos; el libro no contiene fórmulas."
    for j, h in enumerate(["clave", "valor"], start=1):
        c = wm.cell(1, j, h)
        c.font, c.fill = cab_font, cab_fill
    for k, v in meta.items():
        wm.append([str(k), str(v)])
    wm.column_dimensions["A"].width = 28
    wm.column_dimensions["B"].width = 110
    if doc.get("excel_extra") or any(b.get("datos") for s in doc.get("secciones", []) for b in s.get("bloques", [])):
        # hojas de gráficos (imagen + datos exactos + gráfico nativo), Pronóstico, Metodología e Índice navegable
        from fsai import excel_graficos
        excel_graficos.completar_libro(wb, doc)
    wb.save(ruta)
    return ruta


# ------------------------------ Word ------------------------------
def _sombrear(celda, hex_color: str) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    tc_pr = celda._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def _campo_pagina(parrafo) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    run = parrafo.add_run()
    for tipo, texto in (("begin", None), (None, "PAGE"), ("end", None)):
        if tipo:
            e = OxmlElement("w:fldChar")
            e.set(qn("w:fldCharType"), tipo)
        else:
            e = OxmlElement("w:instrText")
            e.set(qn("xml:space"), "preserve")
            e.text = texto
        run._r.append(e)


def exportar_word(doc_in: dict[str, Any], ruta: str | Path) -> Path:
    """Informe .docx: portada, contenido, resumen, secciones con estilos de título, tablas con encabezado y pies de figura."""
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tablas, figuras = numerar(doc_in)
    try:
        d = Document()
        sec0 = d.sections[0]
        sec0.page_width, sec0.page_height = Cm(21.0), Cm(29.7)
        for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(sec0, m, Cm(2.2))
        normal = d.styles["Normal"]
        normal.font.name, normal.font.size = "Calibri", Pt(10.5)
        normal.paragraph_format.space_after = Pt(6)
        for nombre, tam in (("Title", 26), ("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11)):
            st = d.styles[nombre]
            st.font.name, st.font.size = "Calibri", Pt(tam)
            st.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        cap = d.styles["Caption"]
        cap.font.size, cap.font.bold, cap.font.color.rgb = Pt(9.5), True, RGBColor(0x1B, 0x27, 0x33)
        cap.paragraph_format.space_after = Pt(3)
        cap.paragraph_format.keep_with_next = True

        sec0.header.paragraphs[0].text = doc_in.get("encabezado", doc_in["titulo"])
        sec0.header.paragraphs[0].runs[0].font.size = Pt(8)
        pf = sec0.footer.paragraphs[0]
        pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.add_run("Página ").font.size = Pt(8)
        _campo_pagina(pf)

        # portada
        for _ in range(4):
            d.add_paragraph()
        d.add_paragraph(doc_in["titulo"], style="Title")
        if doc_in.get("subtitulo"):
            p = d.add_paragraph(doc_in["subtitulo"])
            p.runs[0].font.size, p.runs[0].font.color.rgb = Pt(13), RGBColor(0x4F, 0x5B, 0x67)
        d.add_paragraph()
        if doc_in.get("portada"):
            t = d.add_table(rows=0, cols=2)
            t.style = "Table Grid"
            for k, v in doc_in["portada"]:
                c = t.add_row().cells
                c[0].text, c[1].text = str(k), _formatear(v) if not isinstance(v, str) else v
                _sombrear(c[0], "EAEFF4")
                c[0].paragraphs[0].runs[0].bold = True
                for celda in c:
                    for par in celda.paragraphs:
                        for r in par.runs:
                            r.font.size = Pt(9)
            for fila in t.rows:
                fila.cells[0].width, fila.cells[1].width = Cm(5), Cm(11.6)
        d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

        # contenido (estático: secciones, tablas y figuras)
        d.add_heading("Contenido", level=1)
        for s in doc_in.get("secciones", []):
            d.add_paragraph(s["titulo"])
        for etq, lista, pref in (("Índice de tablas", tablas, "Tabla"), ("Índice de figuras", figuras, "Figura")):
            if lista:
                d.add_heading(etq, level=2)
                for n, tit in lista:
                    p = d.add_paragraph(f"{pref} {n}. {tit}")
                    p.paragraph_format.space_after = Pt(1)
                    p.runs[0].font.size = Pt(9)
        d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

        d.add_heading("Resumen ejecutivo", level=1)
        for par in doc_in.get("resumen") or ["Sin datos disponibles para el resumen ejecutivo."]:
            d.add_paragraph(par)

        def pie(partes: list[tuple[str, str]]) -> None:
            for etq, txt in partes:
                p = d.add_paragraph()
                p.paragraph_format.space_after = Pt(2)
                if etq:
                    r = p.add_run(f"{etq}. ")
                    r.bold = True
                    r.font.size = Pt(9)
                r2 = p.add_run(txt)
                r2.font.size = Pt(9)
                r2.italic = True
            d.paragraphs[-1].paragraph_format.space_after = Pt(10) if partes else Pt(6)

        for s in doc_in.get("secciones", []):
            d.add_heading(s["titulo"], level=1)
            for b in s.get("bloques", []):
                t = b["tipo"]
                if t == "p":
                    d.add_paragraph(b["texto"])
                elif t == "h2":
                    d.add_heading(b["texto"], level=2)
                elif t == "vacio":
                    p = d.add_paragraph()
                    p.add_run(b["texto"]).italic = True
                elif t == "lista":
                    for it in b["items"]:
                        d.add_paragraph(it, style="List Bullet")
                elif t == "codigo":
                    p = d.add_paragraph()
                    r = p.add_run(b["texto"])
                    r.font.name, r.font.size = "Consolas", Pt(8.5)
                elif t == "figura":
                    if not _figura_existe(b):
                        continue
                    p = d.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p.paragraph_format.keep_with_next = True
                    p.add_run().add_picture(str(b["ruta"]), width=Cm(8.5) if b.get("ancho", "simple") == "simple" else Cm(16.6))
                    d.add_paragraph(f"Figura {b['num']}. {b['titulo']}", style="Caption")
                    pie(_pie_partes(b.get("pie")))
                elif t == "tabla":
                    df, total = _recorte(b)
                    d.add_paragraph(f"Tabla {b['num']}. {b['titulo']}", style="Caption")
                    tbl = d.add_table(rows=1, cols=len(df.columns))
                    tbl.style = "Table Grid"
                    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                    tam = Pt(8) if len(df.columns) <= 8 else Pt(6.5)
                    for j, c in enumerate(df.columns):
                        celda = tbl.rows[0].cells[j]
                        celda.text = str(c)
                        _sombrear(celda, AZUL)
                        for r in celda.paragraphs[0].runs:
                            r.bold, r.font.size, r.font.color.rgb = True, tam, RGBColor(0xFF, 0xFF, 0xFF)
                    tr_pr = tbl.rows[0]._tr.get_or_add_trPr()
                    h = OxmlElement("w:tblHeader")
                    h.set(qn("w:val"), "true")
                    tr_pr.append(h)
                    for i, reg in enumerate(df.itertuples(index=False)):
                        celdas = tbl.add_row().cells
                        for j, v in enumerate(reg):
                            celdas[j].text = _formatear(v)
                            par = celdas[j].paragraphs[0]
                            par.paragraph_format.space_after = Pt(0)
                            if isinstance(v, (int, float)) and not isinstance(v, bool):
                                par.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                            for r in par.runs:
                                r.font.size = tam
                            if i % 2:
                                _sombrear(celdas[j], "F2F6FA")
                    if total > len(df):
                        n = d.add_paragraph(f"(Se muestran {len(df)} de {total} filas; la tabla completa está en el libro Excel.)")
                        n.runs[0].italic, n.runs[0].font.size = True, Pt(8.5)
                    pie(_pie_partes(b.get("pie")))
        d.core_properties.title = doc_in["titulo"]
        d.core_properties.author = "FoodSecurity AI"
        d.core_properties.subject = "Predicción de inseguridad alimentaria y vulnerabilidad nutricional de los hogares mediante inteligencia artificial explicable"
        d.save(ruta)
    except Exception:
        logger.exception("Fallo al escribir Word %s", ruta)
        raise
    return ruta


# ------------------------------ PDF ------------------------------
def exportar_pdf(doc_in: dict[str, Any], ruta: str | Path) -> Path:
    """Informe PDF paginado (A4 vertical): portada, índice con páginas, secciones, tablas, figuras y pies."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph,
                                    Preformatted, Spacer, Table, TableStyle)
    from reportlab.platypus.tableofcontents import TableOfContents

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tablas, figuras = numerar(doc_in)
    base = getSampleStyleSheet()
    azul = colors.HexColor("#" + AZUL)
    est = {
        "T": ParagraphStyle("T", parent=base["Title"], fontSize=24, leading=28, textColor=azul, alignment=0),
        "S": ParagraphStyle("S", parent=base["Normal"], fontSize=12, leading=16, textColor=colors.HexColor("#4F5B67")),
        "H1": ParagraphStyle("H1", parent=base["Heading1"], fontSize=15, leading=19, textColor=azul, spaceBefore=10, spaceAfter=6),
        "HC": ParagraphStyle("HC", parent=base["Heading1"], fontSize=15, leading=19, textColor=azul, spaceBefore=10, spaceAfter=6),
        "H2": ParagraphStyle("H2", parent=base["Heading2"], fontSize=11.5, leading=14, textColor=azul, spaceBefore=8, spaceAfter=4),
        "P": ParagraphStyle("P", parent=base["BodyText"], fontSize=9.5, leading=13, spaceAfter=5),
        "CAP": ParagraphStyle("CAP", parent=base["BodyText"], fontSize=8.8, leading=11, fontName="Helvetica-Bold", spaceAfter=3, keepWithNext=1),
        "PIE": ParagraphStyle("PIE", parent=base["BodyText"], fontSize=8, leading=10.5, fontName="Helvetica-Oblique", textColor=colors.HexColor("#333333"), spaceAfter=2),
        "CEL": ParagraphStyle("CEL", parent=base["BodyText"], fontSize=7, leading=8.5),
        "CELH": ParagraphStyle("CELH", parent=base["BodyText"], fontSize=7, leading=8.5, fontName="Helvetica-Bold", textColor=colors.white),
        "VAC": ParagraphStyle("VAC", parent=base["BodyText"], fontSize=9.5, fontName="Helvetica-Oblique", textColor=colors.HexColor("#4F5B67")),
        "IDX": ParagraphStyle("IDX", parent=base["BodyText"], fontSize=8.5, leading=11),
    }
    esc = lambda t: _pdf_seguro(str(t)).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")  # noqa: E731
    encabezado = _pdf_seguro(doc_in.get("encabezado", doc_in["titulo"]))

    class _Doc(BaseDocTemplate):
        def afterFlowable(self, f):  # alimenta el índice con los títulos H1/H2
            if isinstance(f, Paragraph) and f.style.name in ("H1", "H2"):
                self.notify("TOCEntry", (0 if f.style.name == "H1" else 1, f.getPlainText(), self.page))

    def _pagina(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#4F5B67"))
        canvas.drawString(2 * cm, A4[1] - 1.2 * cm, encabezado[:110])
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Página {doc.page}")
        canvas.setStrokeColor(colors.HexColor("#D3DAE1"))
        canvas.line(2 * cm, A4[1] - 1.35 * cm, A4[0] - 2 * cm, A4[1] - 1.35 * cm)
        canvas.restoreState()

    def tabla_pdf(b):
        df, total = _recorte(b)
        ncol = len(df.columns)
        tamfuente = 7 if ncol <= 9 else 6
        ce = ParagraphStyle("c", parent=est["CEL"], fontSize=tamfuente, leading=tamfuente + 1.5)
        ch = ParagraphStyle("h", parent=est["CELH"], fontSize=tamfuente, leading=tamfuente + 1.5)
        datos = [[Paragraph(esc(c), ch) for c in df.columns]]
        for reg in df.itertuples(index=False):
            datos.append([Paragraph(esc(_formatear(v)), ce) for v in reg])
        largo = [max([max(len(w) for w in (str(c).split() or ["x"]))] + [len(_formatear(v)) for v in df[c].head(40)]) + 3 for c in df.columns]
        largo = [min(max(x, 7), 38) for x in largo]
        ancho_total = 17 * cm
        escala = ancho_total / sum(largo)
        tb = Table(datos, colWidths=[x * escala for x in largo], repeatRows=1)
        tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), azul), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BFC7CF")),
                                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6FA")]),
                                ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        return tb, total, len(df)

    def pie_pdf(partes):
        return [Paragraph((f"<b>{esc(etq)}.</b> " if etq else "") + esc(txt), est["PIE"]) for etq, txt in partes]

    try:
        doc = _Doc(str(ruta), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=1.8 * cm,
                   title=_pdf_seguro(doc_in["titulo"]), author="FoodSecurity AI")
        doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")], onPage=_pagina)])
        flujo: list[Any] = [Spacer(1, 5 * cm), Paragraph(esc(doc_in["titulo"]), est["T"])]
        if doc_in.get("subtitulo"):
            flujo.append(Paragraph(esc(doc_in["subtitulo"]), est["S"]))
        flujo.append(Spacer(1, 1 * cm))
        if doc_in.get("portada"):
            filas = [[Paragraph(f"<b>{esc(k)}</b>", est["CEL"]), Paragraph(esc(v if isinstance(v, str) else _formatear(v)), est["CEL"])] for k, v in doc_in["portada"]]
            tp = Table(filas, colWidths=[5 * cm, 12 * cm])
            tp.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BFC7CF")), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAEFF4")),
                                    ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            flujo.append(tp)
        flujo += [PageBreak(), Paragraph("Contenido", est["HC"])]
        toc = TableOfContents()
        toc.levelStyles = [ParagraphStyle("t0", fontSize=10, leading=14, leftIndent=0, fontName="Helvetica-Bold"),
                           ParagraphStyle("t1", fontSize=9, leading=12, leftIndent=14)]
        flujo.append(toc)
        for etq, lista, pref in (("Índice de tablas", tablas, "Tabla"), ("Índice de figuras", figuras, "Figura")):
            if lista:
                flujo.append(Paragraph(etq, ParagraphStyle("HI", parent=est["HC"], fontSize=11.5, leading=14, spaceBefore=8, spaceAfter=4)))
                flujo += [Paragraph(esc(f"{pref} {n}. {tit}"), est["IDX"]) for n, tit in lista]
        flujo += [PageBreak(), Paragraph("Resumen ejecutivo", est["H1"])]
        flujo += [Paragraph(esc(p), est["P"]) for p in (doc_in.get("resumen") or ["Sin datos disponibles para el resumen ejecutivo."])]
        for s in doc_in.get("secciones", []):
            flujo.append(Paragraph(esc(s["titulo"]), est["H1"]))
            for b in s.get("bloques", []):
                t = b["tipo"]
                if t == "p":
                    flujo.append(Paragraph(esc(b["texto"]), est["P"]))
                elif t == "h2":
                    flujo.append(Paragraph(esc(b["texto"]), est["H2"]))
                elif t == "vacio":
                    flujo.append(Paragraph(esc(b["texto"]), est["VAC"]))
                elif t == "lista":
                    flujo += [Paragraph("• " + esc(it), est["P"]) for it in b["items"]]
                elif t == "codigo":
                    flujo.append(Preformatted(_pdf_seguro(b["texto"]), ParagraphStyle("cod", fontName="Courier", fontSize=7.5, leading=9.5,
                                                                                   backColor=colors.HexColor("#F2F6FA"), borderPadding=3)))
                    flujo.append(Spacer(1, 4))
                elif t == "figura":
                    if not _figura_existe(b):
                        continue
                    ancho = 8.5 * cm if b.get("ancho", "simple") == "simple" else 17 * cm
                    img = Image(str(b["ruta"]), width=ancho, height=ancho * 0.9, kind="proportional")
                    img.hAlign = "CENTER"
                    flujo.append(KeepTogether([img, Paragraph(esc(f"Figura {b['num']}. {b['titulo']}"), ParagraphStyle("capf", parent=est["CAP"], keepWithNext=0, alignment=TA_CENTER)), *pie_pdf(_pie_partes(b.get("pie"))), Spacer(1, 6)]))
                elif t == "tabla":
                    tb, total, mostradas = tabla_pdf(b)
                    cab = Paragraph(esc(f"Tabla {b['num']}. {b['titulo']}"), est["CAP"])
                    extra = [Paragraph(esc(f"(Se muestran {mostradas} de {total} filas; la tabla completa está en el libro Excel.)"), est["PIE"])] if total > mostradas else []
                    flujo += [cab, tb, *extra, *pie_pdf(_pie_partes(b.get("pie"))), Spacer(1, 8)]
        doc.multiBuild(flujo)
    except Exception:
        logger.exception("Fallo al escribir PDF %s", ruta)
        raise
    return ruta
