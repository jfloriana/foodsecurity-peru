"""Estado de la ruta de análisis calculado desde PostgreSQL y los artefactos de entrenamiento.

Cada paso devuelve (estado, detalle): «hecho» (hay evidencia en la BD o en los artefactos), «listo» (se puede
ejecutar ahora) o «pendiente» (falta un paso previo). Nunca modifica datos.
"""
from __future__ import annotations

import sqlite3
from typing import Any

import modulos
from comun import consultar
from fases_comun import C

from fsai import reportes_fases as R  # noqa: E402

HECHO, LISTO, PENDIENTE = "hecho", "listo", "pendiente"


def _n(sql: str, **params: Any) -> int:
    try:
        df = consultar(sql, params, cache=False)
        return int(df.iloc[0, 0]) if not df.empty and df.iloc[0, 0] is not None else 0
    except Exception:  # noqa: BLE001 - sin esquema o sin BD: se trata como cero
        return 0


def _paginas_visitadas(usuario: str | None) -> set[str]:
    """Módulos que el usuario actual ya abrió (auditoría de accesos)."""
    if not usuario:
        return set()
    try:
        df = consultar("SELECT DISTINCT entidad_id FROM fsai.auditoria WHERE accion = 'acceso_pagina' AND usuario = :u", {"u": str(usuario)[:64]}, cache=False)
        return set(df["entidad_id"].dropna().astype(str))
    except Exception:  # noqa: BLE001
        return set()


def _estudios_optuna() -> int:
    db = C.DIR_OPTUNA / "estudios.db"
    if not db.is_file():
        return 0
    try:
        with sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True) as con:
            return int(con.execute("SELECT COUNT(*) FROM studies").fetchone()[0])
    except sqlite3.Error:
        return 0


def hechos_del_sistema(dataset_id: int | None, usuario: str | None = None) -> dict[str, Any]:
    """Cuenta lo que hay en BD y artefactos para el dataset activo."""
    d = dataset_id if dataset_id is not None else -1
    corridas = R.corridas_disponibles()
    carpeta_rep = C.ARTIFACTS_DIR / "reportes"
    return {
        "hogares": _n("SELECT COUNT(*) FROM fsai.hogares WHERE dataset_id = :d", d=d),
        "fies": _n("SELECT COUNT(*) FROM fsai.hogares h JOIN fsai.respuestas_fies r ON r.hogar_uid = h.hogar_uid "
                   "WHERE h.dataset_id = :d AND r.fies_nivel IS NOT NULL", d=d),
        "modelos_bd": _n("SELECT COUNT(*) FROM fsai.modelos m JOIN fsai.experimentos e ON e.experimento_id = m.experimento_id WHERE e.dataset_id = :d", d=d),
        "explicaciones": _n("SELECT COUNT(*) FROM fsai.explicaciones x JOIN fsai.modelos m ON m.modelo_id = x.modelo_id "
                            "JOIN fsai.experimentos e ON e.experimento_id = m.experimento_id WHERE e.dataset_id = :d", d=d),
        "corridas": len(corridas),
        "pruebas": sum(1 for c in corridas if any(c.glob("prueba_*.json"))),
        "modelos_finales": sum(1 for c in corridas if any(c.glob("modelos/final_*.joblib"))),
        "predicciones": sum(1 for c in corridas if any(c.glob("pred_val_*.npz")) or any(c.glob("pred_prueba_*.npz"))),
        "estudios_optuna": _estudios_optuna(),
        "reportes": (sum(1 for _ in carpeta_rep.rglob("informe_*.*")) + sum(1 for _ in carpeta_rep.rglob("reporte_*.*"))) if carpeta_rep.is_dir() else 0,
        "visitadas": _paginas_visitadas(usuario),
    }


def estado_ruta(dataset_id: int | None, usuario: str | None = None) -> dict[str, tuple[str, str]]:
    """{clave de módulo: (estado, detalle)} para los pasos de la ruta de análisis."""
    h = hechos_del_sistema(dataset_id, usuario)
    v = h["visitadas"]
    hay_datos = h["hogares"] > 0
    hay_fies = h["fies"] > 0
    hay_modelos = h["corridas"] > 0 or h["modelos_bd"] > 0

    def visita(clave: str, requisito: bool, texto_hecho: str, texto_listo: str, texto_falta: str) -> tuple[str, str]:
        if not requisito:
            return PENDIENTE, texto_falta
        return (HECHO, texto_hecho) if clave in v else (LISTO, texto_listo)

    return {
        "importacion_enaho": (HECHO, f"{h['hogares']:,} hogares cargados") if hay_datos else (LISTO, "Cargue un archivo de ENAHO o un dataset"),
        "procesamiento_fies": ((HECHO, f"{h['fies']:,} hogares con nivel FIES") if hay_fies else
                               ((LISTO, "Calcule el puntaje y los niveles FIES") if hay_datos else (PENDIENTE, "Requiere datos cargados"))),
        "eda": visita("eda", hay_fies, "Exploración realizada", "Explore distribuciones y calidad", "Requiere niveles FIES"),
        "vulnerabilidad": visita("vulnerabilidad", hay_fies, "Indicadores revisados", "Revise indicadores por nivel FIES", "Requiere niveles FIES"),
        "entrenamiento": ((HECHO, f"{h['corridas']} corrida(s) guardada(s)" + (f" · {h['modelos_bd']} modelo(s) en BD" if h["modelos_bd"] else "")) if hay_modelos else
                          ((LISTO, "Entrene M1-M5 con CV anidada") if hay_fies else (PENDIENTE, "Requiere niveles FIES"))),
        "optuna": ((HECHO, f"{h['estudios_optuna']} estudio(s)") if h["estudios_optuna"] else
                   ((LISTO, "Opcional: explore hiperparámetros") if hay_fies else (PENDIENTE, "Requiere niveles FIES"))),
        "comparacion": ((HECHO, f"Prueba evaluada en {h['pruebas']} corrida(s)") if h["pruebas"] else
                        ((LISTO, "Evalúe la Prueba una sola vez") if h["corridas"] else (PENDIENTE, "Requiere modelos entrenados"))),
        "shap": ((HECHO, f"{h['explicaciones']} explicación(es) guardadas") if h["explicaciones"] or "shap" in v else
                 ((LISTO, "Explique las predicciones") if hay_modelos else (PENDIENTE, "Requiere modelos entrenados"))),
        "territorial": visita("territorial", hay_fies, "Mapa/tabla territorial revisados", "Compare territorios", "Requiere niveles FIES"),
        "equidad": ((HECHO, "Brechas revisadas") if "equidad" in v and (hay_modelos or h["predicciones"]) else
                    ((LISTO, "Revise brechas por grupo") if (h["predicciones"] or h["modelos_bd"]) else (PENDIENTE, "Requiere predicciones de un modelo"))),
        "reportes": ((HECHO, f"{h['reportes']} archivo(s) generados") if h["reportes"] else
                     ((LISTO, "Genere el informe científico") if hay_fies else (PENDIENTE, "Requiere niveles FIES"))),
        "inferencia": ((HECHO, "Predicción utilizada") if "inferencia" in v and h["modelos_finales"] else
                       ((LISTO, "Prediga para un hogar o un lote") if h["modelos_finales"] or h["modelos_bd"] else (PENDIENTE, "Requiere un modelo final"))),
    }


def siguiente_paso(estados: dict[str, tuple[str, str]], rol_acceso: dict[str, tuple[str, ...]], rol: str) -> str | None:
    """Primer paso «listo» de la ruta, no opcional y permitido al rol (o el primer «listo» si solo quedan opcionales)."""
    permitidos = [k for k in modulos.RUTA if rol in rol_acceso.get(k, ())]
    for k in permitidos:
        if estados[k][0] == LISTO and k not in modulos.RUTA_OPCIONAL:
            return k
    for k in permitidos:
        if estados[k][0] == LISTO:
            return k
    return None
