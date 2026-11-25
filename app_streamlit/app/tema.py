"""Sistema de diseño de FoodSecurity AI V2 · dirección «Atlas clínico».

Dirección: periodismo de datos editorial (papel cálido, titulares con serifa, cifras tabulares, filetes finos) cruzado con la
precisión clínica de un tablero de salud pública (verde-azulado de tinta, semántica sobria, nada decorativo que no codifique datos).

Tipografía: pila de sistema afinada (display: Iowan Old Style / Palatino Linotype / Georgia; texto: Segoe UI Variable / system-ui;
mono: Cascadia Mono / Consolas). No se incrusta ninguna fuente: no fue posible verificar sin conexión la licencia (OFL) ni
la disponibilidad de un woff2 redistribuible, así que se prefirió la pila del sistema (cero red, cero licencias).

Solo tema claro: los tokens se inyectan como variables CSS (una única paleta, ``TOKENS``) y ``.streamlit/config.toml`` fija
``base = "light"``. El contraste de los pares críticos se verifica con ``tabla_contraste()``.

Seguridad: todo el HTML que se inyecta con ``unsafe_allow_html`` es constante o pasa por ``html.escape``; nunca se incluye
contenido escrito por el usuario ni proveniente de la base de datos sin escapar.
"""
from __future__ import annotations

import base64
import functools
import html
import re
from pathlib import Path
from typing import Iterable, Sequence

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import modulos

VERSION_APP = "2.0"
NOMBRE_APP = "FoodSecurity AI"
LEMA_APP = "Predicción de inseguridad alimentaria y vulnerabilidad nutricional de los hogares mediante inteligencia artificial explicable"
DIR_ASSETS = Path(__file__).resolve().parent / "assets"
SECCIONES = ("Datos", "Modelado", "Evaluación e interpretación", "Resultados y despliegue")

# ---------------------------------------------------------------- tokens de color
_BASE_CLARO = {
    "fondo": "#F2EFE7", "fondo-2": "#E8E4D8", "superficie": "#FCFBF8", "superficie-2": "#F5F2EA", "lateral": "#EAE6DA",
    "linea": "#DAD4C5", "linea-fuerte": "#B9B2A0", "borde-ctl": "#7C7664",
    "tinta": "#0F1B24", "texto": "#1D2A33", "suave": "#55616A",
    "acento": "#0B5C5E", "acento-osc": "#074344", "acento-texto": "#0B5C5E", "acento-suave": "#D6E8E3", "sobre-acento": "#FFFFFF",
    "enlace": "#0B5C5E", "foco": "#A64B0A", "hover": "rgba(11,92,94,.07)",
    "adv": "#9A5A00", "adv-texto": "#7A4600", "adv-f": "#FBEFD5",
    "peligro": "#B3261E", "peligro-texto": "#9B1C16", "pel-f": "#FBE4E1",
    "ok": "#1E7A4F", "ok-texto": "#17603E", "ok-f": "#DDF0E5",
    "info": "#1F5F99", "info-texto": "#1B5287",
    "n0": "#3C8A7B", "n0-t": "#14564A", "n0-f": "#DCEEE9",
    "n1": "#B2801A", "n1-t": "#6B4A00", "n1-f": "#FAEFD2",
    "n2": "#C2601F", "n2-t": "#7A3607", "n2-f": "#FADFCB",
    "n3": "#A12B1E", "n3-t": "#7F1D13", "n3-f": "#F6D9D4",
    "heroe-fondo": "#0C2F33", "heroe-texto": "#F3F1EA", "heroe-suave": "#B7CCC8", "heroe-ojo": "#8FD9C6", "heroe-brillo": "rgba(80,190,170,.38)",
    "sombra-a": "rgba(15,27,36,.04)", "sombra-b": "rgba(15,27,36,.07)", "sombra-c": "rgba(15,27,36,.20)",
}
TOKENS = _BASE_CLARO  # única paleta (tema claro)
# Compatibilidad con código anterior que importaba COLORES (paleta clara).
COLORES = {"primario": _BASE_CLARO["acento"], "primario_osc": _BASE_CLARO["acento-osc"], "acento": _BASE_CLARO["acento"],
           "texto": _BASE_CLARO["texto"], "texto_suave": _BASE_CLARO["suave"], "borde": _BASE_CLARO["linea"], "fondo": _BASE_CLARO["fondo"],
           "superficie": _BASE_CLARO["superficie"], "lateral": _BASE_CLARO["lateral"], "error": _BASE_CLARO["peligro"],
           "advertencia": _BASE_CLARO["adv"], "info": _BASE_CLARO["info"], "exito": _BASE_CLARO["ok"]}
# Categórica de Okabe-Ito (segura para daltonismo).
PALETA_CATEGORICA = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#999999"]
# Rampas secuenciales (luminancia monótona → legibles en escala de grises y con daltonismo) y divergente marrón-verdeazulado.
SEQ = ["#F4F1E6", "#D3E5DA", "#9CCBBE", "#58A39F", "#1F6F78", "#0B3C4D"]
DIV = ["#8C510A", "#D8B365", "#F5F0E1", "#5AB4AC", "#01665E"]
NIVELES_FIES = {
    0: ("N0", "Seguridad alimentaria"), 1: ("N1", "Inseguridad leve"), 2: ("N2", "Inseguridad moderada"), 3: ("N3", "Inseguridad severa"),
}
FUENTE_TEXTO = '"Segoe UI Variable Text", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif'
FUENTE_TITULOS = '"Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif'


def _e(texto: object) -> str:
    return html.escape(str(texto), quote=True)


def num_es(valor: object, decimales: int = 0) -> str:
    """Número en formato español: coma decimal y espacio fino sin separación (U+202F) como separador de miles desde 10 000 (norma RAE/SI).

    Ejemplos: 1234 → «1234», 12345.6 → «12 345,6», 0.912 con 3 decimales → «0,912». NaN/None → «—».
    """
    try:
        x = float(valor)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(valor)
    if x != x:
        return "—"
    txt = f"{abs(x):.{decimales}f}"
    ent, _, dec = txt.partition(".")
    if len(ent) >= 5:
        ent = f"{int(ent):,}".replace(",", "\u202f")
    signo = "-" if x < 0 and float(txt) != 0 else ""
    return signo + ent + ("," + dec if dec else "")


_RE_NUM = re.compile(r"^-?\d+(\.\d+)?$")


def _texto_num_es(txt: str) -> str:
    """Convierte «12345.678» (cadena numérica con punto decimal) a formato español; otra cosa se devuelve igual."""
    t = txt.strip()
    if not _RE_NUM.match(t):
        return txt
    dec = len(t.partition(".")[2])
    return num_es(float(t), dec)


def logo_data_uri(nombre: str) -> str:
    """SVG local de assets/ como data URI (st.logo no admite rutas SVG en esta versión de Streamlit)."""
    datos = base64.b64encode((DIR_ASSETS / nombre).read_bytes()).decode("ascii")
    return "data:image/svg+xml;base64," + datos


def logo_marca_data_uri(icono_solo: bool = False) -> str:
    """Logotipo (marca + nombre «FoodSecurity AI») o solo el icono, como data URI para st.logo."""
    marca = (f'<rect width="48" height="48" rx="11" fill="{TOKENS["acento"]}"/>'
             '<rect x="9" y="29" width="6" height="10" rx="1.5" fill="#FFFFFF" opacity=".55"/><rect x="18" y="23" width="6" height="16" rx="1.5" fill="#FFFFFF" opacity=".75"/>'
             '<rect x="27" y="16" width="6" height="23" rx="1.5" fill="#FFFFFF"/><rect x="36" y="9" width="3.5" height="30" rx="1.5" fill="#8FD9C6"/>')
    if icono_solo:
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48">{marca}</svg>'
    else:
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="236" height="48" viewBox="0 0 236 48">{marca}'
               f'<text x="58" y="31" font-family="Georgia, serif" font-size="19" font-weight="700" fill="{TOKENS["tinta"]}" textLength="170" lengthAdjust="spacingAndGlyphs">FoodSecurity AI</text></svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


# ---------------------------------------------------------------- iconos SVG en línea (24x24, trazo)
_ICONOS = {
    "home": "M3 11l9-8 9 8M5 10v10h5v-6h4v6h5V10",
    "map": "M9 4L3 6v14l6-2 6 2 6-2V4l-6 2-6-2zM9 4v14M15 6v14",
    "chart": "M4 20V10M10 20V4M16 20v-7M22 20H2",
    "database": "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6",
    "shield": "M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z",
    "alert": "M12 3l10 18H2zM12 10v5M12 18v.5",
    "check": "M5 12.5l4.5 4.5L19 7.5",
    "info": "M12 3a9 9 0 100 18 9 9 0 000-18zM12 11v6M12 7.5v.5",
    "arrow": "M5 12h14M13 6l6 6-6 6",
    "upload": "M12 16V4M7 9l5-5 5 5M4 20h16",
    "book": "M4 5a2 2 0 012-2h13v16H6a2 2 0 00-2 2zM4 19V5M8 7h7",
    "scale": "M12 3v18M5 21h14M5 7h14M5 7l-3 7a3 3 0 006 0zM19 7l-3 7a3 3 0 006 0z",
    "search": "M11 4a7 7 0 100 14 7 7 0 000-14zM20 20l-4-4",
    "layers": "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5",
    "flask": "M9 3h6M10 3v6l-5.5 9.5A1.5 1.5 0 005.8 21h12.4a1.5 1.5 0 001.3-2.5L14 9V3",
    "up": "M3 17l6-6 4 4 8-8M15 7h6v6",
    "down": "M3 7l6 6 4-4 8 8M15 17h6v-6",
    "flat": "M4 12h16M15 7l5 5-5 5",
    "file": "M6 3h8l5 5v13H6zM14 3v5h5M9 13h6M9 17h6",
    "model": "M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12v9M12 12L4 7.5",
    "history": "M12 3a9 9 0 100 18 9 9 0 000-18zM12 7v5l3 2",
    "link": "M10 14a4 4 0 005.7 0l3-3a4 4 0 00-5.7-5.7l-1 1M14 10a4 4 0 00-5.7 0l-3 3A4 4 0 0011 18.7l1-1",
}


def icono(nombre: str, tam: int = 20, etiqueta: str = "") -> str:
    """SVG en línea de trazo (currentColor). Decorativo (aria-hidden) salvo que se dé ``etiqueta``."""
    ruta = _ICONOS.get(nombre, _ICONOS["info"])
    a11y = f'role="img" aria-label="{_e(etiqueta)}"' if etiqueta else 'aria-hidden="true" focusable="false"'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{tam}" height="{tam}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" {a11y}><path d="{ruta}"/></svg>')


def _svg_marca(tam: int = 44, fondo: str | None = None) -> str:
    """Marca propia: cuatro barras ascendentes (niveles FIES) sobre un cuadrado verde-azulado."""
    f = fondo or _BASE_CLARO["acento"]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{tam}" height="{tam}" viewBox="0 0 48 48" role="img" aria-label="Logotipo de {NOMBRE_APP}">'
        f'<rect width="48" height="48" rx="11" fill="{f}"/>'
        '<rect x="9" y="29" width="6" height="10" rx="1.5" fill="#FFFFFF" opacity=".55"/>'
        '<rect x="18" y="23" width="6" height="16" rx="1.5" fill="#FFFFFF" opacity=".75"/>'
        '<rect x="27" y="16" width="6" height="23" rx="1.5" fill="#FFFFFF"/>'
        '<rect x="36" y="9" width="3.5" height="30" rx="1.5" fill="#8FD9C6"/></svg>'
    )


# ---------------------------------------------------------------- CSS
@functools.lru_cache(maxsize=1)
def _css_estatico() -> str:
    return (DIR_ASSETS / "estilos.css").read_text(encoding="utf-8")


def _css_variables() -> str:
    cuerpo = ";".join(f"--fs-{k}:{v}" for k, v in TOKENS.items())
    return f":root{{{cuerpo};color-scheme:light;}}"


_CSS_OCULTAR_LATERAL = """
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"],
[data-testid="stSidebarNav"]{display:none!important;}
"""


def config_pagina(titulo: str) -> None:
    """Configura la pestaña de la página si se ejecuta suelta; bajo st.navigation el enrutador ya lo hizo."""
    try:
        st.set_page_config(page_title=f"{titulo} | {NOMBRE_APP}", layout="wide")
    except st.errors.StreamlitAPIException:
        pass


def aplicar_estilos(ocultar_lateral: bool = False) -> None:
    """Inyecta los tokens (tema claro) + hoja de estilos local (constantes; sin contenido de usuario)."""
    css = _css_variables() + _css_estatico() + (_CSS_OCULTAR_LATERAL if ocultar_lateral else "")
    st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)
    st.session_state["_fsai_estilos_aplicados"] = True


# ---------------------------------------------------------------- marca de acceso y pie
def marca_acceso() -> None:
    """Panel de identidad de la pantalla de acceso (presentación; los hechos provienen de constantes del proyecto)."""
    st.markdown(
        '<section class="fs-acceso fs-rise" aria-label="Presentación de la plataforma"><div class="izq">'
        f'<div class="marca">{_svg_marca(40, "#0B5C5E")}<span>{_e(NOMBRE_APP)}</span></div>'
        '<div><h1>El hambre oculta tiene <em>mapa</em>. Aprender a leerlo es el primer paso.</h1>'
        f'<p>{_e(LEMA_APP)}.</p></div>'
        '<div class="hechos"><div><b>25</b><span>departamentos</span></div><div><b>12</b><span>indicadores territoriales</span></div>'
        '<div><b>5</b><span>modelos explicables</span></div></div></div></section>',
        unsafe_allow_html=True,
    )


def titulo_acceso(titulo: str, ayuda: str = "") -> None:
    st.markdown(f'<h2 class="fs-acceso-t">{_e(titulo)}</h2>' + (f'<p class="fs-sub" style="margin-bottom:.75rem">{_e(ayuda)}</p>' if ayuda else ""),
                unsafe_allow_html=True)


def aviso_uso_acceso() -> None:
    st.markdown('<p class="fs-aviso">Uso exclusivamente investigativo. Las predicciones describen asociaciones, no causas, y no deben usarse para negar ayudas.</p>',
                unsafe_allow_html=True)


def pie_pagina() -> None:
    st.markdown(
        f'<p class="fs-pie"><b>{_e(NOMBRE_APP)}</b> · versión {_e(VERSION_APP)}<br>'
        "Los datos se procesan localmente (PostgreSQL en este equipo) y no se envían a servicios externos.</p>",
        unsafe_allow_html=True,
    )


def indicador_intentos(fallos: int, maximo: int) -> None:
    """Indicador de intentos fallidos de la sesión (puntos + texto), sin revelar si el usuario existe."""
    fallos = max(0, min(int(fallos), int(maximo)))
    puntos = "".join('<span class="p u"></span>' if i < fallos else '<span class="p"></span>' for i in range(int(maximo)))
    st.markdown(
        f'<p class="fs-intentos" role="status">{puntos} Intentos fallidos: {fallos} de {int(maximo)}. '
        "Al alcanzar el límite, el acceso se bloquea temporalmente.</p>",
        unsafe_allow_html=True,
    )


def usuario_lateral(usuario: str, rol: str) -> None:
    ini = "".join(p[0] for p in str(usuario).replace(".", " ").replace("_", " ").split()[:2]).upper() or "?"
    st.markdown(
        f'<div class="fs-usuario"><span class="av" aria-hidden="true">{_e(ini)}</span><div><div class="r">Sesión · {_e(rol)}</div>'
        f'<div class="u">{_e(usuario)}</div></div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- cabecera de página + riel de ruta
def _clave_modulo(titulo: str) -> str | None:
    base = titulo.split(" (")[0].strip().lower()
    for clave, _s, t, *_ in modulos.MODULOS:
        if base == t.lower() or base.startswith(t.lower()):
            return clave
    return None


def riel_ruta(seccion: str, clave: str | None = None) -> tuple[str, str]:
    """Riel de progreso de la ruta: cuatro tramos (secciones) con el actual resaltado y «Paso n de 12» si el módulo está en la ruta."""
    try:
        idx = SECCIONES.index(seccion)
    except ValueError:
        return "", ""
    items = []
    for i, s in enumerate(SECCIONES):
        cls = "actual" if i == idx else ("hecho" if i < idx else "")
        cur = ' aria-current="step"' if i == idx else ""
        items.append(f'<li class="{cls}"{cur}><span class="n">0{i + 1}</span>{_e(s)}</li>')
    paso = ""
    if clave and clave in modulos.RUTA:
        paso = f'<span class="fs-paso">Paso {modulos.RUTA.index(clave) + 1} de {len(modulos.RUTA)}</span>'
    return f'<ol class="fs-riel" aria-label="Progreso de la ruta de análisis">{"".join(items)}</ol>', paso


def encabezado(titulo: str, subtitulo: str = "", seccion: str = "", pasos: list[str] | None = None, produce: str = "",
               clave: str | None = None, insight: str = "", compacto: bool = False) -> None:
    """Encabezado de página consistente: migas + riel de la ruta, titular, bajada y panel «Cómo usar este módulo».

    ``insight``: frase-hallazgo opcional (titular editorial) que se muestra bajo el título.
    """
    clave = clave or _clave_modulo(titulo)
    corto = titulo.split(" (")[0]
    riel, paso = riel_ruta(seccion, clave) if seccion else ("", "")
    migas = (f'<nav class="fs-migas" aria-label="Ubicación"><span>{_e(NOMBRE_APP)}</span><span class="sep" aria-hidden="true">/</span>'
             + (f'<span>{_e(seccion)}</span><span class="sep" aria-hidden="true">/</span>' if seccion else "")
             + f'<b aria-current="page">{_e(corto)}</b>{paso}</nav>')
    st.markdown(migas + ("" if compacto else riel) + ('<span class="fs-pagina-compacta" hidden></span>' if compacto else ""), unsafe_allow_html=True)
    st.title(titulo)
    if subtitulo:
        st.markdown(f'<p class="fs-sub">{_e(subtitulo)}</p>', unsafe_allow_html=True)
    if insight:
        st.markdown(f'<p class="fs-ojo">{_e(insight)}</p>', unsafe_allow_html=True)
    if pasos:
        guia_modulo(pasos, produce)


def guia_modulo(pasos: list[str], produce: str = "") -> None:
    """Panel plegable «Cómo usar este módulo» (pasos numerados y qué produce)."""
    with st.expander("Cómo usar este módulo", expanded=False, icon=":material/help:"):
        st.markdown(chr(10).join(f"{i}. {p}" for i, p in enumerate(pasos, start=1)))
        if produce:
            st.markdown(f"**Qué produce.** {produce}")


def seccion(titulo: str, descripcion: str = "") -> None:
    """Cabecera de sección (sustituye a st.subheader): titular serifado con filete y bajada opcional."""
    st.markdown(f'<div class="fs-sec"><h2>{_e(titulo)}</h2>' + (f"<p>{_e(descripcion)}</p>" if descripcion else "") + "</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- componentes de datos
def sparkline(valores: Sequence[float], etiqueta: str = "Tendencia", ancho: int = 160, alto: int = 34) -> str:
    """Minigráfico SVG (línea + área + punto final) con ``<title>`` accesible; sin información solo por color."""
    v = [float(x) for x in valores if x is not None]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    rng = (hi - lo) or 1.0
    pts = [(i * ancho / (len(v) - 1), alto - 4 - (x - lo) / rng * (alto - 8)) for i, x in enumerate(v)]
    linea = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"0,{alto} {linea} {ancho},{alto}"
    x2, y2 = pts[-1]
    return (f'<svg viewBox="0 0 {ancho} {alto}" preserveAspectRatio="none" role="img" aria-label="{_e(etiqueta)}: de {v[0]:.1f} a {v[-1]:.1f}, '
            f'mínimo {lo:.1f}, máximo {hi:.1f}"><title>{_e(etiqueta)}</title>'
            f'<polygon points="{area}" fill="var(--fs-acento)" opacity=".12"/>'
            f'<polyline points="{linea}" fill="none" stroke="var(--fs-acento)" stroke-width="1.8" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>'
            f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="3" fill="var(--fs-acento)"/></svg>')


def kpi_tarjetas(items: Iterable[dict]) -> None:
    """Tarjetas KPI: dict(etiqueta, valor, unidad='', delta='', tendencia='sube|baja|igual', nota='', spark=[...], spark_etiqueta='').

    La tendencia se comunica con icono + texto (no solo color).
    """
    tarjetas = []
    for it in items:
        d = ""
        if it.get("delta"):
            t = it.get("tendencia", "igual")
            ic = {"sube": "up", "baja": "down"}.get(t, "flat")
            d = f'<p class="d {_e(t)}">{icono(ic, 14)}<span>{_e(it["delta"])}</span></p>'
        sp = f'<div class="sp">{sparkline(it["spark"], it.get("spark_etiqueta", it["etiqueta"]))}</div>' if it.get("spark") else ""
        unidad = f"<small>{_e(it['unidad'])}</small>" if it.get("unidad") else ""
        nota = f'<p class="n">{_e(it["nota"])}</p>' if it.get("nota") else ""
        tarjetas.append(f'<article class="fs-kpi fs-rise"><p class="l">{_e(it["etiqueta"])}</p><p class="v">{_e(it["valor"])}{unidad}</p>{d}{nota}{sp}</article>')
    n = len(tarjetas)
    cols = n if n <= 4 else (3 if n in (5, 6, 9) else 4)  # columnas en ancho completo; el CSS reduce 4→2→1 según el ancho del contenedor
    st.markdown(f'<div class="fs-cq"><section class="fs-kpis" style="--fs-n:{max(cols, 1)}" aria-label="Indicadores clave">{"".join(tarjetas)}</section></div>',
                unsafe_allow_html=True)


def tarjetas_metricas(items: list[tuple[str, str, str | None]]) -> None:
    """Compatibilidad: fila de KPIs (etiqueta, valor, ayuda) con el estilo nuevo."""
    kpi_tarjetas([{"etiqueta": a, "valor": v, "nota": h or ""} for a, v, h in items])


def heroe(titular_html: str, bajada: str = "", ojo: str = "", bandas: Sequence[tuple[str, str]] = ()) -> None:
    """Héroe editorial. ``titular_html`` puede contener <em> (énfasis); el resto debe llegar ya escapado por quien llama."""
    b = "".join(f'<div class="b"><span class="v">{_e(v)}</span><span class="l">{_e(l)}</span></div>' for v, l in bandas)
    st.markdown(
        '<section class="fs-heroe fs-rise" aria-label="Resumen ejecutivo">' + (f'<p class="fs-ojo">{_e(ojo)}</p>' if ojo else "")
        + f'<p class="h" role="heading" aria-level="2">{titular_html}</p>' + (f"<p>{_e(bajada)}</p>" if bajada else "")
        + (f'<div class="fs-bandas">{b}</div>' if b else "") + "</section>",
        unsafe_allow_html=True,
    )


def insignia_fies(nivel: int) -> str:
    """Insignia de nivel FIES: forma + código + nombre (no depende solo del color). Devuelve HTML."""
    cod, nombre = NIVELES_FIES.get(int(nivel), ("N?", "Sin nivel"))
    formas = {0: '<circle cx="6" cy="6" r="4.2" fill="none" stroke="currentColor" stroke-width="1.8"/>',
              1: '<circle cx="6" cy="6" r="4.2" fill="currentColor" opacity=".55"/><circle cx="6" cy="6" r="4.2" fill="none" stroke="currentColor" stroke-width="1.8"/>',
              2: '<path d="M6 1.2l5 9H1z" fill="currentColor"/>',
              3: '<rect x="1.5" y="1.5" width="9" height="9" rx="1.5" fill="currentColor"/>'}
    return (f'<span class="fs-ins n{int(nivel) if int(nivel) in NIVELES_FIES else 0}"><svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">'
            f'{formas.get(int(nivel), formas[0])}</svg><span>{cod} · {_e(nombre)}</span></span>')


def insignia(texto: str, tipo: str = "neutro") -> str:
    """Insignia de estado: tipo ∈ ok|aviso|error|neutro; incluye icono además del color."""
    ic = {"ok": "check", "aviso": "alert", "error": "alert", "neutro": "info"}.get(tipo, "info")
    return f'<span class="fs-ins {_e(tipo)}">{icono(ic, 12)}<span>{_e(texto)}</span></span>'


def leyenda_fies() -> None:
    st.markdown('<div class="fs-leyenda" aria-label="Leyenda de niveles FIES">' + "".join(insignia_fies(n) for n in NIVELES_FIES) + "</div>",
                unsafe_allow_html=True)


def tira_estadisticos(items: Sequence[tuple[str, str]]) -> None:
    """Tira de estadísticos (valor, etiqueta) en una sola tarjeta."""
    c = "".join(f'<div class="c"><span class="v">{_e(v)}</span><span class="l">{_e(l)}</span></div>' for v, l in items)
    n = len(items)
    st.markdown(f'<div class="fs-cq"><div class="fs-tira" style="--fs-n:{min(max(n, 1), 4)}" role="group" aria-label="Estadísticos">{c}</div></div>', unsafe_allow_html=True)


def llamada(tipo: str, titulo: str, texto: str) -> None:
    """Llamada (ética/aviso/error/ok/info) con icono y título; ``texto`` se escapa."""
    ic = {"etica": "scale", "aviso": "alert", "error": "alert", "ok": "check", "info": "info"}.get(tipo, "info")
    rol = 'role="alert"' if tipo == "error" else 'role="note"'
    st.markdown(f'<aside class="fs-llamada {_e(tipo)}" {rol}>{icono(ic, 20)}<div><span class="t">{_e(titulo)}</span><p>{_e(texto)}</p></div></aside>',
                unsafe_allow_html=True)


def vacio(mensaje: str, siguiente: str = "", archivo: str = "", titulo: str = "Aún no hay datos para mostrar") -> None:
    """Estado vacío honesto: qué falta, qué hacer a continuación y enlace directo al módulo (si se indica)."""
    st.markdown(
        f'<div class="fs-vacio" role="status"><span class="ic">{icono("database", 22)}</span><div><p class="t">{_e(titulo)}</p>'
        f"<p>{_e(mensaje)}</p>" + (f'<span class="sig">Siguiente paso: {_e(siguiente)}</span>' if siguiente else "") + "</div></div>",
        unsafe_allow_html=True,
    )
    if archivo:
        try:
            st.page_link(archivo, label=siguiente or "Ir al módulo", icon=":material/arrow_forward:")
        except Exception:  # noqa: BLE001 - la página puede no estar registrada si se ejecuta suelta
            pass


def esqueleto(filas: int = 3) -> None:
    """Marcador de carga (barras pulsantes; sin animación si se pide reducir movimiento)."""
    barras = "".join(f'<i style="width:{w}%"></i>' for w in (92, 78, 64, 85, 55)[: max(1, filas)])
    st.markdown(f'<div class="fs-sk" role="status" aria-label="Cargando contenido">{barras}</div>', unsafe_allow_html=True)


def tabla_html(filas: Sequence[dict], columnas: Sequence[tuple[str, str]], leyenda: str = "", numericas: Sequence[str] = (),
               enlaces: Sequence[str] = (), mono: Sequence[str] = (), apilar: bool = False) -> None:
    """Tabla con estilo común. ``columnas``: (clave, encabezado). Todo valor se escapa; las columnas en ``enlaces`` se convierten en
    vínculos solo si empiezan por https:// (rel=noopener). Las columnas ``numericas`` se muestran en formato español (coma decimal).
    ``apilar``: en contenedores estrechos (teléfono) cada fila se convierte en una tarjeta con «encabezado: valor»; si es False, la
    tabla se desplaza horizontalmente dentro de su propio contenedor."""
    etiquetas = {k: h for k, h in columnas}

    def celda(k: str, v: object) -> str:
        if k in numericas:
            txt = num_es(v, 0 if isinstance(v, int) else 3) if isinstance(v, (int, float)) and not isinstance(v, bool) else _texto_num_es(str(v))
        else:
            txt = str(v)
        cls = " ".join(c for c, g in (("num", numericas), ("mono", mono)) if k in g)
        if k in enlaces and txt.startswith("https://"):
            from urllib.parse import urlparse

            corto = urlparse(txt).netloc or txt[:40]
            inner = f'<a href="{_e(txt)}" title="{_e(txt)}" target="_blank" rel="noopener noreferrer">{_e(corto)}</a>'
        else:
            inner = _e(txt)
        return f'<td class="{cls}" data-l="{_e(etiquetas[k])}">{inner}</td>' if cls else f'<td data-l="{_e(etiquetas[k])}">{inner}</td>'

    th = "".join(f'<th scope="col"{" class=num" if k in numericas else ""}>{_e(h)}</th>' for k, h in columnas)
    cuerpo = "".join("<tr>" + "".join(celda(k, f.get(k, "")) for k, _ in columnas) + "</tr>" for f in filas)
    cap = f"<caption>{_e(leyenda)}</caption>" if leyenda else ""
    ap = " apilada" if apilar else ""
    st.markdown(f'<div class="fs-cq"><div class="fs-tabla-w" tabindex="0" role="region" aria-label="{_e(leyenda or "Tabla")}"><table class="fs-tabla{ap}">{cap}'
                f"<thead><tr>{th}</tr></thead><tbody>{cuerpo}</tbody></table></div></div>", unsafe_allow_html=True)


def tarjeta_termino(termino: str, definicion: str, como_leerlo: str, advertencia: str = "") -> None:
    """Ficha de glosario: término (serifa), definición, «cómo leerlo» y, si existe, advertencia ética con icono."""
    adv = f'<p class="adv">{icono("alert", 16)}<span>{_e(advertencia)}</span></p>' if advertencia else ""
    st.markdown(f'<article class="fs-term"><h3>{_e(termino)}</h3><p>{_e(definicion)}</p><p class="leer"><b>Cómo leerlo.</b> {_e(como_leerlo)}</p>{adv}</article>',
                unsafe_allow_html=True)


def pildoras(etiqueta: str, opciones: Sequence, clave: str, formato=None, valor=None):
    """Selector tipo píldoras (radio horizontal estilizado). Devuelve la opción elegida."""
    with st.container(key=f"fs-pildoras-{clave}"):
        kw = {"format_func": formato} if formato else {}
        if valor is not None:
            kw["index"] = list(opciones).index(valor)
        return st.radio(etiqueta, list(opciones), horizontal=True, key=f"pil_{clave}", **kw)


def ruta_analisis(estados: dict[str, tuple[str, str]], rol: str, acceso: dict[str, tuple[str, ...]]) -> None:
    """La ruta de análisis como tubería visual: 4 fases con pasos conectados y estado (icono/texto, no solo color)."""
    nombres = {"hecho": "Completado", "listo": "Listo", "pendiente": "Pendiente"}
    cols = []
    n = 0
    for sec in SECCIONES:
        pasos = [c for c in modulos.RUTA if modulos.POR_CLAVE[c][1] == sec]
        if not pasos:
            continue
        hechos = sum(1 for c in pasos if estados.get(c, ("pendiente", ""))[0] == "hecho")
        li = []
        for c in pasos:
            n_i = modulos.RUTA.index(c) + 1
            est, det = estados.get(c, ("pendiente", ""))
            marca = icono("check", 15) if est == "hecho" else str(n_i)
            opc = " · opcional" if c in modulos.RUTA_OPCIONAL else ""
            acc = "" if rol in acceso.get(c, ()) else " · sin acceso con su rol"
            li.append(f'<li class="{_e(est)}"><span class="pt">{marca}</span><div><p class="tt">{_e(modulos.titulo(c))}</p>'
                      f'<p class="dd"><span class="est">{nombres.get(est, est)}{_e(opc)}{_e(acc)}</span> · {_e(det)}</p></div></li>')
        cols.append(f'<section class="fs-fase fs-rise" aria-label="Fase {_e(sec)}"><h3>{_e(sec)}<span class="cuenta">{hechos}/{len(pasos)}</span></h3><ol>{"".join(li)}</ol></section>')
        n += len(pasos)
    st.markdown(f'<div class="fs-cq"><div class="fs-ruta">{"".join(cols)}</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- Plotly
def plantilla_plotly() -> go.layout.Template:
    t = TOKENS
    eje = dict(
        showgrid=True, gridcolor=t["linea"], gridwidth=1, zeroline=False, showline=True, linecolor=t["linea-fuerte"],
        ticks="outside", tickcolor=t["linea-fuerte"], tickfont=dict(size=12, color=t["texto"]),
        title=dict(font=dict(size=12.5, color=t["suave"]), standoff=10), automargin=True,
    )
    return go.layout.Template(
        layout=go.Layout(
            font=dict(family=FUENTE_TEXTO, size=13, color=t["texto"]),
            title=dict(font=dict(family=FUENTE_TITULOS, size=17, color=t["tinta"]), x=0.0, xanchor="left"),
            colorway=PALETA_CATEGORICA,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=eje, yaxis=eje,
            legend=dict(font=dict(size=12, color=t["texto"]), bgcolor="rgba(0,0,0,0)", borderwidth=0, title=dict(font=dict(size=12, color=t["suave"]))),
            margin=dict(l=60, r=24, t=48, b=56),
            hoverlabel=dict(font=dict(family=FUENTE_TEXTO, size=12, color=t["texto"]), bgcolor=t["superficie"], bordercolor=t["linea-fuerte"]),
            colorscale=dict(sequential=[[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)],
                            diverging=[[i / (len(DIV) - 1), c] for i, c in enumerate(DIV)]),
            geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=False, showland=False, showcountries=False, showlakes=False),
            bargap=0.28,
        )
    )


def registrar_plantilla_plotly() -> None:
    """Registra la plantilla «fsai» (única, tema claro) y fija un estilo uniforme para st.plotly_chart.

    ``st.plotly_chart`` aplica por defecto el tema de Streamlit, que sobrescribe fuentes y colores; se envuelve (una sola vez)
    para usar ``theme=None`` con la plantilla propia.
    """
    import streamlit.elements.plotly_chart  # noqa: F401  (su import fija la plantilla "streamlit"; debe ir antes)
    from streamlit.delta_generator import DeltaGenerator

    pio.templates["fsai"] = plantilla_plotly()
    pio.templates.default = "fsai"

    original = DeltaGenerator.plotly_chart
    if getattr(original, "_fsai_envuelto", False):
        return

    @functools.wraps(original)
    def plotly_chart(self, figure_or_data, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("theme", None)
        if isinstance(figure_or_data, go.Figure):
            figure_or_data.update_layout(template=pio.templates["fsai"])
            figure_or_data.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        config = dict(kwargs.get("config") or {})
        config.setdefault("displaylogo", False)
        config.setdefault("toImageButtonOptions", {"format": "png", "scale": 2})
        kwargs["config"] = config
        kwargs.setdefault("use_container_width", True)
        return original(self, figure_or_data, *args, **kwargs)

    plotly_chart._fsai_envuelto = True  # type: ignore[attr-defined]
    DeltaGenerator.plotly_chart = plotly_chart  # type: ignore[method-assign]
    st.plotly_chart = st._main.plotly_chart  # type: ignore[attr-defined]


def mapa_coropletico(df, columna: str, titulo_barra: str, geojson: dict, codigo: str = "cod_departamento", divergente: bool = False,
                     formato: str = ".1f", sufijo: str = "", marcar_top: int = 5, alto: int = 520, barra: bool = True):
    """Mapa coroplético departamental (Plotly). Une por ``FIRST_IDDP``; marca con contorno grueso los ``marcar_top`` valores más altos
    para que la lectura no dependa solo del color. ``df`` debe tener ``codigo``, ``departamento`` y ``columna``.
    """
    t = TOKENS
    escala = DIV if divergente else SEQ
    d = df.dropna(subset=[columna]).copy()
    d["_txt"] = d[columna].map(lambda x: format(x, formato) + sufijo)
    d["_nom"] = d["departamento"].astype(str).str.replace("_", " ").str.title()
    fig = go.Figure(go.Choropleth(
        geojson=geojson, locations=d[codigo], featureidkey="properties.FIRST_IDDP", z=d[columna],
        colorscale=[[i / (len(escala) - 1), c] for i, c in enumerate(escala)],
        marker=dict(line=dict(color=t["superficie"], width=0.8)),
        zmid=0 if divergente else None,
        customdata=d[["_nom", "_txt"]].to_numpy(),
        hovertemplate="<b>%{customdata[0]}</b><br>" + _e(titulo_barra) + ": %{customdata[1]}<extra></extra>",
        showscale=barra,
        colorbar=dict(title=dict(text=titulo_barra, font=dict(size=11, color=t["suave"])), thickness=12, len=0.7, outlinewidth=0,
                      tickfont=dict(size=11, color=t["suave"])),
    ))
    if marcar_top and len(d) > marcar_top:
        top = d.nlargest(marcar_top, columna)
        fig.add_trace(go.Choropleth(
            geojson=geojson, locations=top[codigo], featureidkey="properties.FIRST_IDDP", z=[1] * len(top), showscale=False,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], marker=dict(line=dict(color=t["tinta"], width=2.6)), hoverinfo="skip",
        ))
    fig.update_geos(projection_type="mercator", lonaxis_range=[-82.0, -68.0], lataxis_range=[-18.8, 0.6], visible=False, bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=alto, margin=dict(l=0, r=0, t=8, b=0), showlegend=False, template=pio.templates["fsai"])
    return fig


def color_nivel(nivel: int) -> str:
    """Color de marca del nivel FIES 0-3."""
    return TOKENS[f"n{int(nivel)}"]


def barras_ranking(df, columna: str, titulo_eje: str, formato: str = ".1f", sufijo: str = "", alto: int | None = None, resaltar: int = 5):
    """Ranking horizontal ordenado, con valores escritos al final de cada barra (no depende del color)."""
    t = TOKENS
    d = df.dropna(subset=[columna]).sort_values(columna)
    n = len(d)
    colores = [t["acento"] if i >= n - resaltar else t["linea-fuerte"] for i in range(n)]
    fig = go.Figure(go.Bar(
        x=d[columna], y=d["departamento"].astype(str).str.replace("_", " ").str.title(), orientation="h", marker_color=colores,
        text=[format(v, formato) + sufijo for v in d[columna]], textposition="outside", cliponaxis=False, textfont=dict(size=11, color=t["texto"]),
        hovertemplate="<b>%{y}</b><br>" + _e(titulo_eje) + ": %{text}<extra></extra>",
    ))
    fig.update_layout(height=alto or max(360, 22 * n + 60), margin=dict(l=0, r=48, t=8, b=36), xaxis_title=titulo_eje, yaxis=dict(showgrid=False, showline=False, ticks="", automargin=True),
                      template=pio.templates["fsai"], bargap=0.35)
    fig.update_xaxes(showgrid=True)
    return fig


# ---------------------------------------------------------------- Glosario lateral
GLOSARIO: tuple[tuple[str, str], ...] = (
    ("FIES", "Escala de Experiencia de Inseguridad Alimentaria: 8 preguntas de sí/no; el puntaje crudo (0-8) define el nivel 0-3."),
    ("IA", "Inseguridad alimentaria. «IA moderada/severa» agrupa los niveles FIES 2 y 3."),
    ("ENAHO", "Encuesta Nacional de Hogares del INEI (Perú), fuente de los microdatos."),
    ("Rasch", "Modelo de medición que ubica a cada hogar en una escala continua de severidad (si hay parámetros disponibles)."),
    ("CV anidada", "Validación cruzada con un bucle externo (evalúa) y uno interno (elige hiperparámetros) para estimar el desempeño sin sesgo optimista."),
    ("Optuna", "Librería de búsqueda automática de hiperparámetros."),
    ("IC95", "Intervalo de confianza del 95 %: rango plausible del valor verdadero."),
    ("AUC / ROC", "Área bajo la curva ROC: probabilidad de ordenar bien un caso positivo y uno negativo (0,5 = azar; 1 = perfecto)."),
    ("PR / AP", "Curva precisión-recall y su área (precisión promedio); útil con clases poco frecuentes."),
    ("TPR / FPR", "Tasa de verdaderos positivos (sensibilidad) y de falsos positivos."),
    ("MCC", "Coeficiente de correlación de Matthews: resumen robusto de la matriz de confusión."),
    ("Brier / ECE", "Errores de calibración: qué tan cerca están las probabilidades predichas de las frecuencias observadas (menor es mejor)."),
    ("Perfil descriptivo", "Prevalencia de IA moderada/severa en hogares con menores de 5 años, adultos mayores o pobreza monetaria; no es un índice ponderado ni validado."),
    ("Recuento (0-3)", "Cantidad de esas tres características presentes en un hogar; es un conteo simple, no un puntaje."),
    ("SHAP", "Método que reparte la predicción entre variables; describe asociaciones del modelo, no causas."),
    ("LIME", "Explicación local aproximada de una predicción individual."),
    ("PDP / ALE", "Curvas que muestran cómo cambia la predicción al variar una variable (dependencia parcial / efectos locales acumulados)."),
    ("LODO", "Leave-One-Department-Out: validar dejando fuera un departamento completo."),
    ("DeLong, McNemar, Friedman, Nemenyi, Wilcoxon", "Pruebas estadísticas para comparar modelos; los p-valores se ajustan con Holm."),
    ("[POR VALIDAR]", "Parámetro o corte del protocolo aún no validado para el Perú."),
)


def glosario_lateral() -> None:
    """Expander con el glosario de siglas en la barra lateral."""
    with st.sidebar.expander("Glosario de siglas", icon=":material/menu_book:"):
        for sigla, texto in GLOSARIO:
            st.markdown(f"**{sigla}**: {texto}")


# ---------------------------------------------------------------- verificación de contraste (WCAG 2.x)
def _lum(hexcol: str) -> float:
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contraste(a: str, b: str) -> float:
    la, lb = _lum(a), _lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# (primer plano, fondo, mínimo exigido, descripción)
PARES_CONTRASTE: tuple[tuple[str, str, float, str], ...] = (
    ("tinta", "fondo", 4.5, "Titulares sobre fondo"),
    ("texto", "fondo", 4.5, "Texto base sobre fondo"),
    ("texto", "superficie", 4.5, "Texto base sobre tarjeta"),
    ("suave", "fondo", 4.5, "Texto secundario sobre fondo"),
    ("suave", "superficie", 4.5, "Texto secundario sobre tarjeta"),
    ("suave", "lateral", 4.5, "Texto secundario en barra lateral"),
    ("acento-texto", "superficie", 4.5, "Enlaces / acento como texto sobre tarjeta"),
    ("acento-texto", "fondo", 4.5, "Enlaces / acento como texto sobre fondo"),
    ("sobre-acento", "acento", 4.5, "Texto del botón primario"),
    ("heroe-texto", "heroe-fondo", 4.5, "Texto del héroe"),
    ("heroe-suave", "heroe-fondo", 4.5, "Bajada del héroe"),
    ("heroe-ojo", "heroe-fondo", 4.5, "Sobretítulo del héroe"),
    ("peligro-texto", "superficie", 4.5, "Texto de error"),
    ("adv-texto", "superficie", 4.5, "Texto de advertencia"),
    ("ok-texto", "superficie", 4.5, "Texto de éxito"),
    ("info-texto", "superficie", 4.5, "Texto informativo"),
    ("n0-t", "n0-f", 4.5, "Insignia FIES N0"),
    ("n1-t", "n1-f", 4.5, "Insignia FIES N1"),
    ("n2-t", "n2-f", 4.5, "Insignia FIES N2"),
    ("n3-t", "n3-f", 4.5, "Insignia FIES N3"),
    ("borde-ctl", "superficie", 3.0, "Borde de controles (no textual)"),
    ("foco", "fondo", 3.0, "Anillo de foco sobre fondo (no textual)"),
    ("foco", "superficie", 3.0, "Anillo de foco sobre tarjeta (no textual)"),
    ("n0", "superficie", 3.0, "Marca FIES N0 (no textual)"),
    ("n1", "superficie", 3.0, "Marca FIES N1 (no textual)"),
    ("n2", "superficie", 3.0, "Marca FIES N2 (no textual)"),
    ("n3", "superficie", 3.0, "Marca FIES N3 (no textual)"),
)


def tabla_contraste() -> list[dict]:
    """Calcula el contraste de cada par crítico de la paleta clara."""
    filas = []
    t = TOKENS
    for fg, bg, minimo, desc in PARES_CONTRASTE:
        r = contraste(t[fg], t[bg])
        filas.append({"par": f"{fg} / {bg}", "uso": desc, "fg": t[fg], "bg": t[bg], "ratio": round(r, 2), "minimo": minimo, "cumple": r >= minimo})
    return filas


registrar_plantilla_plotly()


if __name__ == "__main__":  # python app/tema.py → imprime la tabla de contraste
    fallos = 0
    print(f"{'par':32} {'ratio':>6} {'mín':>4}  estado")
    for f in tabla_contraste():
        fallos += not f["cumple"]
        print(f"{f['par']:32} {f['ratio']:6.2f} {f['minimo']:4.1f}  {'OK' if f['cumple'] else 'FALLA'}  ({f['uso']})")
    print(f"\nPares evaluados: {len(tabla_contraste())} · fallos: {fallos}")
