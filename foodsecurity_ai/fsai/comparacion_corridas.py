"""comparacion_corridas.py - Comparar experimentos entre corridas (mejora #5 de MEJORAS_PROPUESTAS.md).

Lógica pura (sin Streamlit): carga las métricas agregadas de cada corrida desde los JSON de ``resultados_agregados/<corrida>/``
(``resultado_<escenario>_<modelo>.json``, clave ``agregado.ponderada.<métrica>`` con ``media``/``ic_inf``/``ic_sup``) o desde
las tablas ``fsai.experimentos``/``fsai.modelos``/``fsai.metricas`` (resultado de ``SQL_EXPERIMENTOS_BD`` ya en un DataFrame),
y las deja en una tabla única. ``advertencias_comparabilidad`` indica cuándo la comparación NO es válida (distinto n_trials,
escenarios o esquemas) y cuándo el escenario es circular (``fies``/``hibrido``). No inventa valores: lo que falta queda NaN.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ESCENARIOS_CIRCULARES = ("fies", "hibrido")
METRICAS = {"f1_macro": "f1_macro", "auc": "roc_auc_ovr", "brier": "brier"}   # alias de columna -> nombre de la métrica
COLUMNAS = ["corrida", "esquema", "escenario", "modelo", "n_trials", "n_trials_final", "n_pliegues",
            "f1_macro", "f1_macro_ic_inf", "f1_macro_ic_sup", "auc", "auc_ic_inf", "auc_ic_sup",
            "brier", "brier_ic_inf", "brier_ic_sup", "origen"]
ESQUEMA_JSON = "CV anidada"

SQL_EXPERIMENTOS_BD = """
SELECT e.experimento_id, e.nombre AS corrida, e.esquema_validacion AS esquema, e.escenario, m.codigo AS modelo,
       m.n_trials, x.nombre_metrica, x.valor, x.ic_inf, x.ic_sup
FROM fsai.experimentos e
JOIN fsai.modelos m   ON m.experimento_id = e.experimento_id
JOIN fsai.metricas x  ON x.modelo_id = m.modelo_id
 AND x.esquema_validacion = e.esquema_validacion
WHERE x.ponderada AND x.fold IS NULL AND x.estrato IS NULL
  AND x.nombre_metrica IN ('f1_macro', 'roc_auc_ovr', 'brier')
ORDER BY e.experimento_id, m.codigo
"""


def _tabla_vacia() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNAS)


def ruta_resultados_agregados(base: str | Path | None = None) -> Path | None:
    """Carpeta ``resultados_agregados``: argumento, variable ``FSAI_RESULTADOS_AGREGADOS`` o búsqueda hacia arriba desde este archivo."""
    candidatos: list[Path] = []
    if base:
        candidatos.append(Path(base))
    if os.environ.get("FSAI_RESULTADOS_AGREGADOS"):
        candidatos.append(Path(os.environ["FSAI_RESULTADOS_AGREGADOS"]))
    candidatos += [p / "resultados_agregados" for p in Path(__file__).resolve().parents]
    for c in candidatos:
        if c.is_dir():
            return c
    return None


def _num(valor: Any) -> float:
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _fila_desde_json(corrida: str, datos: dict[str, Any]) -> dict[str, Any]:
    pond = (datos.get("agregado") or {}).get("ponderada") or {}
    fila: dict[str, Any] = {
        "corrida": corrida, "esquema": ESQUEMA_JSON, "escenario": datos.get("escenario"), "modelo": datos.get("codigo"),
        "n_trials": datos.get("n_trials"), "n_trials_final": datos.get("n_trials_final"), "n_pliegues": None, "origen": "json",
    }
    for alias, nombre in METRICAS.items():
        m = pond.get(nombre) or {}
        fila[alias] = _num(m.get("media"))
        fila[f"{alias}_ic_inf"] = _num(m.get("ic_inf"))
        fila[f"{alias}_ic_sup"] = _num(m.get("ic_sup"))
        if fila["n_pliegues"] is None and m.get("n") is not None:
            fila["n_pliegues"] = m.get("n")
    return fila


def cargar_corridas_json(raiz: str | Path | None = None) -> pd.DataFrame:
    """Lee todos los ``resultado_*.json`` de cada subcarpeta de ``raiz``. Los archivos ilegibles se omiten y se listan en ``df.attrs['problemas']``."""
    carpeta = ruta_resultados_agregados(raiz)
    filas: list[dict[str, Any]] = []
    problemas: list[str] = []
    if carpeta is None:
        tabla = _tabla_vacia()
        tabla.attrs["problemas"] = ["No se encontró la carpeta resultados_agregados."]
        return tabla
    for sub in sorted(p for p in carpeta.iterdir() if p.is_dir()):
        for archivo in sorted(sub.glob("resultado_*.json")):
            try:
                datos = json.loads(archivo.read_text(encoding="utf-8"))
                if not isinstance(datos, dict) or "agregado" not in datos:
                    raise ValueError("sin clave 'agregado'")
                filas.append(_fila_desde_json(sub.name, datos))
            except (OSError, ValueError) as exc:
                problemas.append(f"{sub.name}/{archivo.name}: {exc}")
    tabla = pd.DataFrame(filas, columns=COLUMNAS) if filas else _tabla_vacia()
    tabla.attrs["problemas"] = problemas
    return tabla


def tabla_desde_bd(largo: pd.DataFrame) -> pd.DataFrame:
    """Convierte el resultado (formato largo) de ``SQL_EXPERIMENTOS_BD`` a la tabla común (una fila por corrida/escenario/modelo)."""
    if largo is None or largo.empty:
        return _tabla_vacia()
    filas: dict[tuple, dict[str, Any]] = {}
    inverso = {v: k for k, v in METRICAS.items()}
    for r in largo.to_dict("records"):
        clave = (r.get("experimento_id"), r.get("corrida"), r.get("esquema"), r.get("escenario"), r.get("modelo"))
        fila = filas.setdefault(clave, {
            "corrida": r.get("corrida"), "esquema": r.get("esquema"), "escenario": r.get("escenario"), "modelo": r.get("modelo"),
            "n_trials": None if pd.isna(r.get("n_trials")) else r.get("n_trials"), "n_trials_final": None, "n_pliegues": None, "origen": "bd",
            **{c: float("nan") for c in COLUMNAS if c.startswith(tuple(METRICAS))},
        })
        alias = inverso.get(r.get("nombre_metrica"))
        if alias:
            fila[alias] = _num(r.get("valor"))
            fila[f"{alias}_ic_inf"] = _num(r.get("ic_inf"))
            fila[f"{alias}_ic_sup"] = _num(r.get("ic_sup"))
    return pd.DataFrame(list(filas.values()), columns=COLUMNAS)


def cargar_corridas(consulta_bd=None, raiz_json: str | Path | None = None) -> tuple[pd.DataFrame, str]:
    """Devuelve ``(tabla, origen)``. Con ``consulta_bd`` (callable ``sql -> DataFrame``) intenta la BD; si falla o no hay filas usa los JSON."""
    if consulta_bd is not None:
        try:
            tabla = tabla_desde_bd(consulta_bd(SQL_EXPERIMENTOS_BD))
            if not tabla.empty:
                return tabla, "bd"
        except Exception:  # noqa: BLE001 - sin BD o sin esquema: se cae a los JSON
            pass
    return cargar_corridas_json(raiz_json), "json"


def _distintos(serie: pd.Series) -> list[Any]:
    return sorted({v for v in serie.dropna().tolist()}, key=str)


def advertencias_comparabilidad(tabla: pd.DataFrame) -> list[str]:
    """Avisos en español sobre la validez de comparar las filas dadas (tabla ya filtrada por el usuario). Lista vacía si no hay nada que advertir."""
    avisos: list[str] = []
    if tabla is None or tabla.empty:
        return avisos
    esc = _distintos(tabla["escenario"])
    circ = [e for e in esc if e in ESCENARIOS_CIRCULARES]
    if circ:
        avisos.append("Los escenarios " + ", ".join(f"«{e}»" for e in circ) + " son circulares: usan los ítems con que se define el objetivo, "
                      "así que miden reconstrucción del objetivo, no predicción. Su desempeño casi perfecto no debe leerse como capacidad predictiva.")
    if len(esc) > 1:
        avisos.append("Se mezclan escenarios distintos (" + ", ".join(map(str, esc)) + "): usan predictores diferentes y NO son comparables entre sí.")
    nt = _distintos(tabla["n_trials"])
    if len(nt) > 1:
        avisos.append("Las corridas usaron distinto número de ensayos Optuna (n_trials = " + ", ".join(str(int(v)) for v in nt)
                      + "): una diferencia de desempeño puede deberse al esfuerzo de búsqueda y no al modelo.")
    if len(_distintos(tabla["esquema"])) > 1:
        avisos.append("Se mezclan esquemas de validación distintos (" + ", ".join(map(str, _distintos(tabla["esquema"])))
                      + "): sus métricas provienen de particiones diferentes.")
    if tabla["n_trials"].isna().any():
        avisos.append("Alguna fila no informa n_trials: no se puede verificar que las corridas tengan el mismo esfuerzo de optimización.")
    return avisos


def filtrar(tabla: pd.DataFrame, corridas: Iterable[str] | None = None, escenarios: Iterable[str] | None = None,
            modelos: Iterable[str] | None = None) -> pd.DataFrame:
    """Subtabla por corridas/escenarios/modelos (``None`` = sin filtro)."""
    out = tabla
    for col, vals in (("corrida", corridas), ("escenario", escenarios), ("modelo", modelos)):
        if vals is not None:
            out = out[out[col].isin(list(vals))]
    return out.reset_index(drop=True)
