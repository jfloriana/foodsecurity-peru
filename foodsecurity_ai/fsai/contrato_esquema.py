"""contrato_esquema.py - Contrato de esquema de datos (mejora #2 de MEJORAS_PROPUESTAS.md).

Define QUÉ debe traer un archivo tabular para entrar al pipeline (columnas obligatorias, tipos, rangos,
categorías permitidas, nulos) y lo verifica ANTES de cargarlo, devolviendo un informe estructurado.

Reglas de diseño:
* ``validar_esquema`` NUNCA lanza excepciones por problemas de los datos: cada problema es un ``Hallazgo``
  (severidad ``error`` o ``advertencia``) dentro de ``InformeEsquema``. Solo los errores de programación
  (por ejemplo un contrato mal construido) lanzan ``ValueError`` al CREAR el contrato.
* El mapeo de columnas (código INEI de otro año -> nombre analítico) es configurable: se pasa por argumento o, por
  defecto, se toma de ``config.MAPA_COLUMNAS`` (que se llena con el JSON de la variable de entorno
  ``FSAI_MAPA_COLUMNAS``). Los códigos cambian entre años y deben validarse contra el diccionario INEI [POR VALIDAR].
* El contrato por defecto se deriva de ``config`` (listas de variables), de ``datasets.RANGOS_CONTRATO`` y de
  ``diccionario/diccionario_variables.csv``; no inventa rangos: solo usa los ya declarados en el proyecto.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from fsai import config as C

logger = logging.getLogger(__name__)

VERSION_CONTRATO = 1
TIPOS_VALIDOS = ("entero", "numerico", "binaria", "texto", "categorica", "cualquiera")
ERROR = "error"
ADVERTENCIA = "advertencia"


# --------------------------------------------------------------------------
# Estructuras
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnaContrato:
    """Regla de una columna.

    ``tipo``: entero | numerico | binaria (0/1) | texto | categorica | cualquiera.
    ``minimo``/``maximo``: rango inclusivo (``minimo_exclusivo`` hace estricto el límite inferior).
    ``categorias``: valores permitidos (comparados como texto). ``max_pct_nulos``: sobre este % se advierte.
    ``unica``: los valores no nulos no pueden repetirse.
    """

    nombre: str
    tipo: str = "cualquiera"
    obligatoria: bool = True
    minimo: float | None = None
    maximo: float | None = None
    minimo_exclusivo: bool = False
    categorias: tuple[str, ...] | None = None
    admite_nulos: bool = True
    max_pct_nulos: float | None = None
    unica: bool = False
    descripcion: str = ""

    def __post_init__(self) -> None:
        if self.tipo not in TIPOS_VALIDOS:
            raise ValueError(f"Tipo '{self.tipo}' inválido para '{self.nombre}'; use {TIPOS_VALIDOS}.")
        if self.minimo is not None and self.maximo is not None and self.minimo > self.maximo:
            raise ValueError(f"'{self.nombre}': minimo > maximo.")
        if self.categorias is not None:
            object.__setattr__(self, "categorias", tuple(str(c) for c in self.categorias))


@dataclass(frozen=True)
class ContratoEsquema:
    """Conjunto de reglas por columna y mapeo opcional de nombres de entrada."""

    nombre: str
    columnas: tuple[ColumnaContrato, ...]
    mapa_columnas: dict[str, str] = field(default_factory=dict)
    version: int = VERSION_CONTRATO

    def __post_init__(self) -> None:
        nombres = [c.nombre for c in self.columnas]
        repetidos = sorted({n for n in nombres if nombres.count(n) > 1})
        if repetidos:
            raise ValueError(f"Columnas repetidas en el contrato: {repetidos}")

    def obligatorias(self) -> list[str]:
        return [c.nombre for c in self.columnas if c.obligatoria]

    def a_dict(self) -> dict[str, Any]:
        return {"nombre": self.nombre, "version": self.version, "mapa_columnas": dict(self.mapa_columnas),
                "columnas": [asdict(c) for c in self.columnas]}

    @classmethod
    def desde_dict(cls, d: dict[str, Any]) -> "ContratoEsquema":
        cols = []
        for c in d["columnas"]:
            c = dict(c)
            if c.get("categorias") is not None:
                c["categorias"] = tuple(c["categorias"])
            cols.append(ColumnaContrato(**c))
        return cls(nombre=d["nombre"], columnas=tuple(cols), mapa_columnas=dict(d.get("mapa_columnas", {})),
                   version=int(d.get("version", VERSION_CONTRATO)))


@dataclass(frozen=True)
class Hallazgo:
    """Un problema detectado: severidad, columna (o None si es global), código estable, mensaje y nº de filas afectadas."""

    severidad: str
    columna: str | None
    codigo: str
    mensaje: str
    n_afectados: int = 0


@dataclass
class InformeEsquema:
    """Resultado de ``validar_esquema``: lista estructurada de hallazgos (nunca excepciones por datos)."""

    contrato: str
    n_filas: int = 0
    n_columnas: int = 0
    hallazgos: list[Hallazgo] = field(default_factory=list)
    columnas_renombradas: dict[str, str] = field(default_factory=dict)

    @property
    def errores(self) -> list[Hallazgo]:
        return [h for h in self.hallazgos if h.severidad == ERROR]

    @property
    def advertencias(self) -> list[Hallazgo]:
        return [h for h in self.hallazgos if h.severidad == ADVERTENCIA]

    @property
    def ok(self) -> bool:
        """True si no hay errores (las advertencias no bloquean)."""
        return not self.errores

    def codigos(self, severidad: str | None = None) -> list[str]:
        return [h.codigo for h in self.hallazgos if severidad is None or h.severidad == severidad]

    def a_dataframe(self) -> pd.DataFrame:
        cols = ["severidad", "columna", "codigo", "mensaje", "n_afectados"]
        return pd.DataFrame([asdict(h) for h in self.hallazgos], columns=cols)

    def a_dict(self) -> dict[str, Any]:
        return {"contrato": self.contrato, "ok": self.ok, "n_filas": self.n_filas, "n_columnas": self.n_columnas,
                "columnas_renombradas": dict(self.columnas_renombradas), "hallazgos": [asdict(h) for h in self.hallazgos]}

    def resumen(self) -> str:
        return (f"Contrato '{self.contrato}': {len(self.errores)} error(es), {len(self.advertencias)} advertencia(s) "
                f"en {self.n_filas} filas x {self.n_columnas} columnas.")


# --------------------------------------------------------------------------
# Mapeo de columnas
# --------------------------------------------------------------------------
def aplicar_mapa_columnas(df: pd.DataFrame, mapa: dict[str, str] | None = None) -> tuple[pd.DataFrame, dict[str, str], list[Hallazgo]]:
    """Renombra columnas de entrada (insensible a mayúsculas) según ``mapa`` {nombre_entrada: nombre_analítico}.

    No sobrescribe una columna que ya existe con el nombre destino (lo informa como advertencia). Con ``mapa=None``
    usa ``config.MAPA_COLUMNAS`` (vacío por defecto: no cambia nada). Devuelve (df, renombradas, hallazgos).
    """
    mapa = C.MAPA_COLUMNAS if mapa is None else mapa
    hallazgos: list[Hallazgo] = []
    if not mapa:
        return df, {}, hallazgos
    por_minuscula = {str(c).lower(): c for c in df.columns}
    renombres: dict[str, str] = {}
    for origen, destino in mapa.items():
        real = por_minuscula.get(str(origen).lower())
        if real is None or real == destino:
            continue
        if destino in df.columns or destino in renombres.values():
            hallazgos.append(Hallazgo(ADVERTENCIA, destino, "mapa_conflicto",
                                      f"No se renombró '{real}' a '{destino}': ya existe una columna con ese nombre."))
            continue
        renombres[real] = destino
    return (df.rename(columns=renombres) if renombres else df), renombres, hallazgos


# --------------------------------------------------------------------------
# Contrato por defecto (dataset analítico ENAHO)
# --------------------------------------------------------------------------
_CATEGORIAS_DEFECTO: dict[str, tuple[str, ...]] = {
    "area": ("U", "R"),
    "region_natural": ("costa", "sierra", "selva"),
    "sexo_jefe": ("H", "M"),
}
_OPCIONALES = ("anio", "mes", "fies_rasch_logit", "grupo_edad_jefe", "grupo_educ_jefe", "tiene_menor5")
_PCT_NULOS_ADVERTIR = 50.0   # sobre este % de faltantes se advierte (el parquet real tiene variables con mucho faltante declarado)


def contrato_defecto(mapa_columnas: dict[str, str] | None = None) -> ContratoEsquema:
    """Contrato del dataset analítico ENAHO derivado de ``config``, ``datasets.RANGOS_CONTRATO`` y el diccionario.

    Obligatorias: llaves y peso, territoriales, objetivo, ítems FIES y variables socioeconómicas del protocolo.
    Los rangos son los ya declarados en ``datasets.RANGOS_CONTRATO`` (no se agregan rangos nuevos).
    """
    from fsai.datasets import RANGOS_CONTRATO   # import diferido: datasets importa este módulo

    def rango(col: str) -> dict[str, Any]:
        lo, hi = RANGOS_CONTRATO.get(col, (None, None))
        return {"minimo": lo, "maximo": hi}

    cols: list[ColumnaContrato] = [
        ColumnaContrato("hogar_uid", "texto", admite_nulos=False, unica=True, descripcion="Seudónimo SHA-256 del hogar"),
        ColumnaContrato("conglome", "texto", admite_nulos=False, descripcion="Conglomerado muestral (llave INEI)"),
        ColumnaContrato(C.COL_PESO, "numerico", admite_nulos=False, minimo=0, minimo_exclusivo=True,
                        descripcion="Factor de expansión muestral (debe ser > 0)"),
        ColumnaContrato("departamento", "texto", descripcion="Departamento (25 categorías INEI)"),
        ColumnaContrato("area", "categorica", categorias=_CATEGORIAS_DEFECTO["area"], admite_nulos=False),
        ColumnaContrato("region_natural", "categorica", categorias=_CATEGORIAS_DEFECTO["region_natural"]),
        ColumnaContrato(C.COL_OBJETIVO, "entero", admite_nulos=False, **rango(C.COL_OBJETIVO)),
        ColumnaContrato(C.COL_BINARIA, "binaria", admite_nulos=False, **rango(C.COL_BINARIA)),
    ]
    for item in C.FEATURES_FIES[:8]:
        cols.append(ColumnaContrato(item, "binaria", max_pct_nulos=_PCT_NULOS_ADVERTIR, **rango(item)))
    cols.append(ColumnaContrato("fies_raw_score", "numerico", max_pct_nulos=_PCT_NULOS_ADVERTIR, **rango("fies_raw_score")))
    for col in C.FEATURES_SOCIO_NUM:
        tipo = "binaria" if col in ("pobreza_monetaria", "desempleo", "agua_potable", "saneamiento") else "numerico"
        cols.append(ColumnaContrato(col, tipo, max_pct_nulos=_PCT_NULOS_ADVERTIR, **rango(col)))
    for col in C.FEATURES_SOCIO_CAT:
        if col in ("departamento", "area", "region_natural"):
            continue   # ya declaradas arriba
        cats = _CATEGORIAS_DEFECTO.get(col)
        cols.append(ColumnaContrato(col, "categorica" if cats else "cualquiera", categorias=cats,
                                    max_pct_nulos=_PCT_NULOS_ADVERTIR))
    for col in _OPCIONALES:
        cols.append(ColumnaContrato(col, "cualquiera", obligatoria=False, **rango(col)))
    return ContratoEsquema("enaho_analitico", tuple(cols),
                           mapa_columnas=dict(C.MAPA_COLUMNAS if mapa_columnas is None else mapa_columnas))


# --------------------------------------------------------------------------
# Validación
# --------------------------------------------------------------------------
def _convertible_a_numerico(s: pd.Series) -> bool:
    no_nulos = s.dropna()
    if no_nulos.empty:
        return True
    return bool(pd.to_numeric(no_nulos, errors="coerce").notna().all())


def _validar_columna(s: pd.Series, regla: ColumnaContrato, n: int) -> list[Hallazgo]:
    h: list[Hallazgo] = []
    nom = regla.nombre
    n_nulos = int(s.isna().sum())
    if n_nulos and not regla.admite_nulos:
        h.append(Hallazgo(ERROR, nom, "nulos_no_permitidos", f"'{nom}' no admite nulos y tiene {n_nulos}.", n_nulos))
    if regla.max_pct_nulos is not None and n and 100.0 * n_nulos / n > regla.max_pct_nulos:
        h.append(Hallazgo(ADVERTENCIA, nom, "muchos_nulos",
                          f"'{nom}' tiene {100.0 * n_nulos / n:.1f} % de nulos (umbral {regla.max_pct_nulos:.0f} %).", n_nulos))
    numerica = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    x = s
    if regla.tipo in ("entero", "numerico", "binaria") or regla.minimo is not None or regla.maximo is not None:
        if not (numerica or pd.api.types.is_bool_dtype(s)):
            if _convertible_a_numerico(s):
                h.append(Hallazgo(ADVERTENCIA, nom, "dtype_convertible",
                                  f"'{nom}' viene como {s.dtype} pero es convertible a número; convierta antes de cargar."))
                x = pd.to_numeric(s, errors="coerce")
            else:
                malos = int(pd.to_numeric(s.dropna(), errors="coerce").isna().sum())
                h.append(Hallazgo(ERROR, nom, "dtype_invalido", f"'{nom}' debe ser numérica ({regla.tipo}); hay {malos} valores no numéricos.", malos))
                return h
        else:
            x = s.astype(float) if pd.api.types.is_bool_dtype(s) else s
    if regla.tipo == "entero":
        v = pd.to_numeric(x, errors="coerce").dropna()
        no_enteros = int((v != np.floor(v)).sum())
        if no_enteros:
            h.append(Hallazgo(ERROR, nom, "no_entero", f"'{nom}' debe ser entera y tiene {no_enteros} valores con decimales.", no_enteros))
    if regla.tipo == "binaria":
        v = pd.to_numeric(x, errors="coerce").dropna()
        fuera = int((~v.isin([0, 1])).sum())
        if fuera:
            h.append(Hallazgo(ERROR, nom, "no_binaria", f"'{nom}' debe contener solo 0/1; hay {fuera} valores distintos.", fuera))
    if regla.tipo == "texto" and numerica:
        h.append(Hallazgo(ADVERTENCIA, nom, "dtype_inesperado", f"'{nom}' debería ser texto/código y viene numérica ({s.dtype})."))
    if regla.minimo is not None or regla.maximo is not None:
        v = pd.to_numeric(x, errors="coerce").dropna()
        bajo = (v <= regla.minimo) if (regla.minimo is not None and regla.minimo_exclusivo) else (v < regla.minimo) if regla.minimo is not None else pd.Series(False, index=v.index)
        alto = (v > regla.maximo) if regla.maximo is not None else pd.Series(False, index=v.index)
        fuera = int((bajo | alto).sum())
        if fuera:
            lim_i = "(" if regla.minimo_exclusivo else "["
            h.append(Hallazgo(ERROR, nom, "fuera_de_rango",
                              f"'{nom}': {fuera} valores fuera de {lim_i}{regla.minimo}, {regla.maximo}].", fuera))
    if regla.categorias is not None:
        permitidas = set(regla.categorias)
        valores = s.dropna().astype(str)
        invalidos = valores[~valores.isin(permitidas)]
        if len(invalidos):
            ejemplos = sorted(invalidos.unique())[:5]
            h.append(Hallazgo(ERROR, nom, "categoria_no_permitida",
                              f"'{nom}': {len(invalidos)} valores fuera de {sorted(permitidas)} (ej.: {ejemplos}).", int(len(invalidos))))
    if regla.unica:
        repetidos = int(s.dropna().duplicated().sum())
        if repetidos:
            h.append(Hallazgo(ERROR, nom, "duplicados", f"'{nom}' debe ser única y tiene {repetidos} repeticiones.", repetidos))
    return h


def validar_esquema(df: Any, contrato: ContratoEsquema | None = None, mapa_columnas: dict[str, str] | None = None,
                    estricto: bool = False) -> InformeEsquema:
    """Valida ``df`` contra ``contrato`` (por defecto, el del dataset analítico ENAHO) y devuelve un ``InformeEsquema``.

    ``mapa_columnas``: {nombre_entrada: nombre_analítico}; si es None se usa el del contrato y, si este está vacío,
    ``config.MAPA_COLUMNAS``. ``estricto=True`` advierte además por columnas no contempladas en el contrato.
    Los problemas de datos NO lanzan excepciones: quedan como hallazgos.
    """
    contrato = contrato or contrato_defecto()
    informe = InformeEsquema(contrato=contrato.nombre)
    if not isinstance(df, pd.DataFrame):
        informe.hallazgos.append(Hallazgo(ERROR, None, "no_es_dataframe", f"Se esperaba un DataFrame y llegó {type(df).__name__}."))
        return informe
    informe.n_filas, informe.n_columnas = int(len(df)), int(df.shape[1])
    if len(df) == 0:
        informe.hallazgos.append(Hallazgo(ERROR, None, "sin_filas", "El dataset no tiene filas."))
    if df.columns.duplicated().any():
        dup = sorted({str(c) for c in df.columns[df.columns.duplicated()]})
        informe.hallazgos.append(Hallazgo(ERROR, None, "columnas_duplicadas", f"Nombres de columna repetidos: {dup}."))
        return informe
    mapa = mapa_columnas if mapa_columnas is not None else (contrato.mapa_columnas or None)
    try:
        df, renombradas, hallazgos_mapa = aplicar_mapa_columnas(df, mapa)
    except (AttributeError, TypeError) as exc:      # mapa mal formado
        informe.hallazgos.append(Hallazgo(ERROR, None, "mapa_invalido", f"El mapa de columnas no es un diccionario válido: {exc}"))
        return informe
    informe.columnas_renombradas = renombradas
    informe.hallazgos += hallazgos_mapa
    presentes = set(map(str, df.columns))
    for regla in contrato.columnas:
        if regla.nombre not in presentes:
            if regla.obligatoria:
                informe.hallazgos.append(Hallazgo(ERROR, regla.nombre, "columna_faltante", f"Falta la columna obligatoria '{regla.nombre}'."))
            continue
        informe.hallazgos += _validar_columna(df[regla.nombre], regla, len(df))
    if estricto:
        conocidas = {c.nombre for c in contrato.columnas}
        extras = sorted(presentes - conocidas)
        if extras:
            informe.hallazgos.append(Hallazgo(ADVERTENCIA, None, "columnas_extra", f"Columnas fuera del contrato: {extras}.", len(extras)))
    return informe


def nombres_obligatorios_faltantes(columnas: Iterable[str], contrato: ContratoEsquema | None = None) -> list[str]:
    """Atajo: obligatorias del contrato que no están en ``columnas`` (útil para validar solo la cabecera de un archivo)."""
    contrato = contrato or contrato_defecto()
    presentes = set(map(str, columnas))
    return [n for n in contrato.obligatorias() if n not in presentes]
