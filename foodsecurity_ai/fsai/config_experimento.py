"""config_experimento.py - Exportar/importar la configuración de un experimento (mejora #4 de MEJORAS_PROPUESTAS.md).

Un experimento queda descrito en UN solo archivo JSON versionado, reutilizable para reproducir o repetir una corrida:
id de la corrida, escenario, modelos, espacios de hiperparámetros, semillas, n_trials, esquema de partición y hash
del dataset. El archivo lleva ``formato`` y ``version_esquema``; la importación valida claves faltantes, claves
desconocidas, tipos y valores permitidos, y agrupa TODOS los problemas en una sola excepción ``ConfigInvalida``.

No ejecuta nada: solo serializa/valida. El hash del dataset permite avisar si el archivo de datos cambió
(``verificar_hash_dataset``); no se inventa ningún valor por defecto de resultados.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fsai import config as C

logger = logging.getLogger(__name__)

FORMATO = "fsai.config_experimento"
VERSION_ESQUEMA = 1          # subir al cambiar la forma del JSON; ``importar_config`` rechaza versiones futuras
CLAVES_OBLIGATORIAS = ("run_id", "escenario", "modelos", "espacios_hiperparametros", "semillas", "n_trials",
                       "esquema_particion", "hash_dataset")
CLAVES_OPCIONALES = ("notas", "agrupar_por", "creado_en")
CLAVES_CABECERA = ("formato", "version_esquema")
TIPOS_PARAMETRO = ("float", "int", "cat")


class ConfigInvalida(ValueError):
    """La configuración no cumple el esquema; ``problemas`` lista todos los motivos."""

    def __init__(self, problemas: list[str]):
        self.problemas = list(problemas)
        super().__init__("Configuración de experimento inválida: " + "; ".join(self.problemas))


@dataclass
class ConfigExperimento:
    run_id: str
    escenario: str
    modelos: list[str]
    espacios_hiperparametros: dict[str, dict[str, dict[str, Any]]]
    semillas: list[int]
    n_trials: int
    esquema_particion: str
    hash_dataset: str
    notas: str = ""
    agrupar_por: str = C.COL_GRUPO
    creado_en: str = ""
    extras: dict[str, Any] = field(default_factory=dict, repr=False)

    def a_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("extras", None)
        return {"formato": FORMATO, "version_esquema": VERSION_ESQUEMA, **d}


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def hash_archivo(ruta: str | Path, bloque: int = 1 << 20) -> str:
    """SHA-256 (hex) del contenido de un archivo de datos, para fijar qué dataset usó el experimento."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while chunk := f.read(bloque):
            h.update(chunk)
    return h.hexdigest()


def espacios_por_defecto(modelos: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """Espacios de búsqueda del protocolo (``optuna_optimizer.ESPACIO_DEFECTO``) para los modelos pedidos (copia)."""
    import copy

    from fsai.optuna_optimizer import ESPACIO_DEFECTO   # import diferido: optuna es pesado
    faltan = [m for m in modelos if m not in ESPACIO_DEFECTO]
    if faltan:
        raise KeyError(f"Sin espacio por defecto para: {faltan}")
    return {m: copy.deepcopy(ESPACIO_DEFECTO[m]) for m in modelos}


# --------------------------------------------------------------------------
# Validación
# --------------------------------------------------------------------------
def _es_entero(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _problemas_espacio(modelo: str, espacio: Any) -> list[str]:
    if not isinstance(espacio, dict):
        return [f"espacios_hiperparametros['{modelo}'] debe ser un objeto {{parametro: especificación}}"]
    out: list[str] = []
    for par, spec in espacio.items():
        ref = f"espacios_hiperparametros['{modelo}']['{par}']"
        if not isinstance(spec, dict) or spec.get("tipo") not in TIPOS_PARAMETRO:
            out.append(f"{ref} requiere 'tipo' en {TIPOS_PARAMETRO}")
        elif spec["tipo"] == "cat":
            if not isinstance(spec.get("opciones"), list) or not spec["opciones"]:
                out.append(f"{ref} (cat) requiere 'opciones' no vacía")
        else:
            lo, hi = spec.get("lo"), spec.get("hi")
            numeros = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (lo, hi))
            if not numeros:
                out.append(f"{ref} requiere 'lo' y 'hi' numéricos")
            elif lo > hi:
                out.append(f"{ref} tiene lo > hi")
    return out


def validar_config(d: Any) -> list[str]:
    """Devuelve la lista de problemas del diccionario (vacía si es válido). Nunca lanza por contenido inválido."""
    if not isinstance(d, dict):
        return ["la configuración debe ser un objeto JSON"]
    p: list[str] = []
    if d.get("formato") != FORMATO:
        p.append(f"'formato' debe ser '{FORMATO}' (recibido: {d.get('formato')!r})")
    v = d.get("version_esquema")
    if not _es_entero(v):
        p.append("falta 'version_esquema' entero")
    elif v > VERSION_ESQUEMA:
        p.append(f"version_esquema {v} es más nueva que la soportada ({VERSION_ESQUEMA}); actualice FoodSecurity AI")
    elif v < 1:
        p.append("version_esquema debe ser >= 1")
    permitidas = set(CLAVES_CABECERA) | set(CLAVES_OBLIGATORIAS) | set(CLAVES_OPCIONALES)
    faltan = [k for k in CLAVES_OBLIGATORIAS if k not in d]
    desconocidas = sorted(set(d) - permitidas)
    if faltan:
        p.append(f"faltan claves obligatorias: {faltan}")
    if desconocidas:
        p.append(f"claves desconocidas: {desconocidas}")
    if "run_id" in d and not (isinstance(d["run_id"], str) and d["run_id"].strip()):
        p.append("'run_id' debe ser texto no vacío")
    if "escenario" in d and d["escenario"] not in C.ESCENARIOS:
        p.append(f"'escenario' debe estar en {C.ESCENARIOS}")
    modelos = d.get("modelos")
    if "modelos" in d:
        if not (isinstance(modelos, list) and modelos and all(isinstance(m, str) for m in modelos)):
            p.append("'modelos' debe ser una lista no vacía de códigos")
        else:
            malos = [m for m in modelos if m not in C.MODELOS]
            if malos:
                p.append(f"modelos desconocidos {malos}; válidos: {C.MODELOS}")
            if len(set(modelos)) != len(modelos):
                p.append("'modelos' tiene códigos repetidos")
    esp = d.get("espacios_hiperparametros")
    if "espacios_hiperparametros" in d:
        if not isinstance(esp, dict):
            p.append("'espacios_hiperparametros' debe ser un objeto {modelo: espacio}")
        else:
            if isinstance(modelos, list):
                extra = sorted(set(esp) - set(modelos))
                if extra:
                    p.append(f"espacios para modelos que no están en 'modelos': {extra}")
            for m, e in esp.items():
                p += _problemas_espacio(m, e)
    if "semillas" in d:
        s = d["semillas"]
        if not (isinstance(s, list) and s and all(_es_entero(x) for x in s)):
            p.append("'semillas' debe ser una lista no vacía de enteros")
    if "n_trials" in d and not (_es_entero(d["n_trials"]) and d["n_trials"] >= 1):
        p.append("'n_trials' debe ser un entero >= 1")
    if "esquema_particion" in d and d["esquema_particion"] not in (set(C.ESQUEMAS_PARTICION) | set(C.ESQUEMAS_VALIDACION)):
        p.append(f"'esquema_particion' debe estar en {sorted(set(C.ESQUEMAS_PARTICION) | set(C.ESQUEMAS_VALIDACION))}")
    if "hash_dataset" in d and not (isinstance(d["hash_dataset"], str) and d["hash_dataset"].strip()):
        p.append("'hash_dataset' debe ser texto no vacío (use hash_archivo)")
    if "agrupar_por" in d and d["agrupar_por"] not in (C.COL_GRUPO, "conglome"):
        p.append(f"'agrupar_por' debe ser '{C.COL_GRUPO}' o 'conglome'")
    for k in ("notas", "creado_en"):
        if k in d and not isinstance(d[k], str):
            p.append(f"'{k}' debe ser texto")
    return p


# --------------------------------------------------------------------------
# Exportar / importar
# --------------------------------------------------------------------------
def exportar_config(ruta: str | Path | None = None, *, run_id: str, escenario: str, modelos: list[str],
                    semillas: list[int] | None = None, n_trials: int = C.N_TRIALS,
                    esquema_particion: str = "V1", hash_dataset: str,
                    espacios_hiperparametros: dict[str, dict[str, dict[str, Any]]] | None = None,
                    notas: str = "", agrupar_por: str = C.COL_GRUPO, creado_en: str = "") -> dict[str, Any]:
    """Construye (y valida) la configuración; si se da ``ruta`` la escribe como JSON UTF-8. Devuelve el diccionario.

    ``semillas`` por defecto ``[SEED]``; ``espacios_hiperparametros`` por defecto los del protocolo para ``modelos``.
    Lanza ``ConfigInvalida`` si los datos entregados no cumplen el esquema (no se escribe nada en ese caso).
    """
    cfg = {
        "formato": FORMATO, "version_esquema": VERSION_ESQUEMA, "run_id": run_id, "escenario": escenario,
        "modelos": list(modelos),
        "espacios_hiperparametros": espacios_por_defecto(list(modelos)) if espacios_hiperparametros is None
        else espacios_hiperparametros,
        "semillas": list(semillas) if semillas is not None else [C.SEED], "n_trials": n_trials,
        "esquema_particion": esquema_particion, "hash_dataset": hash_dataset, "notas": notas,
        "agrupar_por": agrupar_por, "creado_en": creado_en,
    }
    # Todo debe ser JSON puro: se normaliza (tuplas -> listas) y se valida el resultado de ese viaje.
    cfg = json.loads(json.dumps(cfg, ensure_ascii=False))
    problemas = validar_config(cfg)
    if problemas:
        raise ConfigInvalida(problemas)
    if ruta is not None:
        destino = Path(ruta)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=False), encoding="utf-8")
    return cfg


def importar_config(origen: str | Path | dict[str, Any]) -> ConfigExperimento:
    """Lee un JSON (ruta) o un diccionario, valida y devuelve ``ConfigExperimento``. Lanza ``ConfigInvalida``."""
    if isinstance(origen, dict):
        d = origen
    else:
        try:
            d = json.loads(Path(origen).read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigInvalida([f"no se pudo leer el archivo: {exc}"]) from exc
        except ValueError as exc:
            raise ConfigInvalida([f"JSON mal formado: {exc}"]) from exc
    problemas = validar_config(d)
    if problemas:
        raise ConfigInvalida(problemas)
    return ConfigExperimento(
        run_id=d["run_id"], escenario=d["escenario"], modelos=list(d["modelos"]),
        espacios_hiperparametros=d["espacios_hiperparametros"], semillas=list(d["semillas"]),
        n_trials=int(d["n_trials"]), esquema_particion=d["esquema_particion"], hash_dataset=d["hash_dataset"],
        notas=d.get("notas", ""), agrupar_por=d.get("agrupar_por", C.COL_GRUPO), creado_en=d.get("creado_en", ""))


def verificar_hash_dataset(cfg: ConfigExperimento, ruta_datos: str | Path) -> bool:
    """True si el SHA-256 del archivo de datos coincide con el de la configuración (si no, se registra una advertencia)."""
    coincide = hash_archivo(ruta_datos) == cfg.hash_dataset
    if not coincide:
        logger.warning("El dataset %s no coincide con hash_dataset de la corrida '%s'.", ruta_datos, cfg.run_id)
    return coincide
