"""Descarga las fuentes PUBLICAS ADICIONALES (territoriales/climaticas) y escribe MANIFIESTO_ADICIONALES.json.

Fuentes (todas HTTP 200 sin cuenta ni credenciales al verificarse el 2026-11-16):
  geojson_departamental   GeoJSON de departamentos del Peru (juaneladio/peru-geojson; licencia NO declarada)
  geojson_geoboundaries   Respaldo geoBoundaries PER ADM1 (Public Domain segun su API)
  endes_2024_m1629        ENDES 2024 (INEI) modulo 1629: RECH0/RECH1/RECH4/RECHM (hogar)
  endes_2024_m1638        ENDES 2024 (INEI) modulo 1638: RECH5 (hemoglobina) y RECH6 (talla/peso de ninos)
  endes_2024_m1641        ENDES 2024 (INEI) modulo 1641: programas sociales por hogar
  ophi_mpi_peru           OPHI MPI por departamento (via HDX, CC0; ENDES 2023)
  noaa_oni                Indice ONI (NOAA CPC), mensual/trimestral movil, nacional
  nasa_power_<cod>        Precipitacion mensual NASA POWER en el centroide de cada departamento (se descarga y se registra, pero
                          NO alimenta el rasgo: su serie 2025 presenta un salto no verosimil; ver docs/FUENTES_ADICIONALES.md)
  openmeteo_<cod>         Precipitacion diaria ERA5 via Open-Meteo archive en el mismo centroide (fuente del rasgo de anomalia)

Directorio de salida: variable de entorno FSAI_DIR_DATOS_PUBLICOS (por defecto <repo>/datos_publicos, ignorado por git).
Todo lo descargado se trata como NO confiable: aqui solo se guarda (nunca se ejecuta ni se extrae).
Solo HTTPS y solo dominios de una lista blanca (tambien en cada redireccion). No se imprimen secretos (no se usan).

Uso (desde la raiz del repo):
  python -I foodsecurity_ai/scripts/descargar_fuentes_adicionales.py            # todas
  python -I foodsecurity_ai/scripts/descargar_fuentes_adicionales.py --solo endes noaa_oni
  python -I foodsecurity_ai/scripts/descargar_fuentes_adicionales.py --forzar   # vuelve a bajar aunque exista
Selectores de --solo: geojson, endes, ophi, noaa_oni, power, openmeteo (o un id exacto de fuente).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RAIZ_PAQUETE = Path(__file__).resolve().parents[1]
RAIZ_REPO = RAIZ_PAQUETE.parent
sys.path.insert(0, str(RAIZ_PAQUETE))
from fsai.fuentes_publicas import sha256_archivo  # noqa: E402

DIR_DATOS = Path(os.getenv("FSAI_DIR_DATOS_PUBLICOS", str(RAIZ_REPO / "datos_publicos")))
DOMINIOS_PERMITIDOS = {
    "raw.githubusercontent.com", "github.com", "media.githubusercontent.com", "objects.githubusercontent.com",
    "proyectos.inei.gob.pe", "data.humdata.org", "centre.humdata.org", "www.cpc.ncep.noaa.gov",
    "power.larc.nasa.gov", "archive-api.open-meteo.com",
    "s3.us-east-1.amazonaws.com",   # HDX redirige aqui las descargas (solo como destino de redireccion)
}
URL_GEO_PRINCIPAL = "https://raw.githubusercontent.com/juaneladio/peru-geojson/master/peru_departamental_simple.geojson"
URL_GEO_RESPALDO = ("https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/PER/ADM1/"
                    "geoBoundaries-PER-ADM1_simplified.geojson")
URL_ENDES = "https://proyectos.inei.gob.pe/iinei/srienaho/descarga/CSV/968-Modulo{n}.zip"
URL_OPHI = ("https://data.humdata.org/dataset/fb69e3e4-720a-4190-b8bc-6cad04a42c46/resource/"
            "c5db7e75-94a4-4d01-9dab-5cd3303d3fb0/download/per_mpi.csv")
URL_ONI = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
URL_POWER = ("https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=PRECTOTCORR&community=AG"
             "&longitude={lon:.4f}&latitude={lat:.4f}&start={ini}&end={fin}&format=JSON")
URL_OPENMETEO = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat:.4f}&longitude={lon:.4f}"
                 "&start_date={ini}-01-01&end_date={fin}-12-31&daily=precipitation_sum&timezone=GMT")
OPENMETEO_ANIO_INI, OPENMETEO_ANIO_FIN = 2015, 2025   # ventana climatologia 2015-2024 + anio ENAHO 2025
POWER_ANIO_INI, POWER_ANIO_FIN = 2015, datetime.now(timezone.utc).year   # el filtrado a ventana completa se hace al derivar

LIC_ENDES = "INEI: datos abiertos de microdatos; sin licencia explicita en las paginas consultadas; citar la fuente"
LIC_GEO_JUAN = "licencia no declarada; verificar antes de redistribuir"


def _fuentes_estaticas() -> list[dict]:
    s = [
        dict(id="geojson_departamental", grupo="geojson", url=URL_GEO_PRINCIPAL, archivo="geo/peru_departamental_simple.geojson",
             licencia=LIC_GEO_JUAN, granularidad="departamento (25 poligonos)", anios="n/a (limites)"),
        dict(id="geojson_geoboundaries", grupo="geojson", url=URL_GEO_RESPALDO, archivo="geo/geoBoundaries-PER-ADM1_simplified.geojson",
             licencia="geoBoundaries gbOpen: Public Domain segun su API (verificar en la ficha antes de redistribuir)",
             granularidad="ADM1 (26 unidades: Lima Provincia/Callao separados)", anios="n/a (limites)"),
    ]
    for n, desc in (("1629", "RECH0/RECH1/RECH4/RECHM"), ("1638", "RECH5/RECH6"), ("1641", "programas sociales por hogar")):
        s.append(dict(id=f"endes_2024_m{n}", grupo="endes", url=URL_ENDES.format(n=n), archivo=f"endes/2024/968-Modulo{n}.zip",
                      licencia=LIC_ENDES, granularidad="microdatos hogar/persona con departamento (UBIGEO/HV024)",
                      anios="2024", nota=desc))
    s.append(dict(id="ophi_mpi_peru", grupo="ophi", url=URL_OPHI, archivo="ophi/per_mpi.csv",
                  licencia="CC0 (segun la ficha HDX)", granularidad="departamento (Admin 1 PCode PE01..PE25)", anios="ENDES 2023"))
    s.append(dict(id="noaa_oni", grupo="noaa_oni", url=URL_ONI, archivo="noaa/oni.ascii.txt",
                  licencia="NOAA/CPC: dominio publico (obra del gobierno de EE. UU.)", granularidad="nacional (indice global), mensual", anios="1950-actual"))
    return s


def fuentes_power(centroides: dict[str, tuple[float, float]]) -> list[dict]:
    return [dict(id=f"nasa_power_{cod}", grupo="power", url=URL_POWER.format(lon=lon, lat=lat, ini=POWER_ANIO_INI, fin=POWER_ANIO_FIN),
                 archivo=f"power/{cod}.json", licencia="NASA POWER: datos abiertos (citar NASA POWER, community AG)",
                 granularidad=f"punto (centroide del departamento {cod}) mensual", anios=f"{POWER_ANIO_INI}-{POWER_ANIO_FIN}",
                 nota=f"lon={lon:.4f} lat={lat:.4f}")
            for cod, (lon, lat) in sorted(centroides.items())]


def fuentes_openmeteo(centroides: dict[str, tuple[float, float]]) -> list[dict]:
    return [dict(id=f"openmeteo_{cod}", grupo="openmeteo",
                 url=URL_OPENMETEO.format(lon=lon, lat=lat, ini=OPENMETEO_ANIO_INI, fin=OPENMETEO_ANIO_FIN),
                 archivo=f"openmeteo/{cod}.json",
                 licencia="Open-Meteo (CC BY 4.0, uso no comercial) sobre ERA5 de Copernicus/ECMWF; citar ambas fuentes",
                 granularidad=f"punto (centroide del departamento {cod}) diario -> mensual", anios=f"{OPENMETEO_ANIO_INI}-{OPENMETEO_ANIO_FIN}",
                 nota=f"lon={lon:.4f} lat={lat:.4f}")
            for cod, (lon, lat) in sorted(centroides.items())]


class _Redireccion(urllib.request.HTTPRedirectHandler):
    """Valida cada salto: solo https y solo dominios de la lista blanca."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        u = urllib.parse.urlparse(newurl)
        if u.scheme != "https" or u.hostname not in DOMINIOS_PERMITIDOS:
            raise urllib.error.URLError(f"redireccion bloqueada hacia {u.scheme}://{u.hostname}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _validar_url(url: str) -> None:
    u = urllib.parse.urlparse(url)
    if u.scheme != "https" or u.hostname not in DOMINIOS_PERMITIDOS:
        raise ValueError(f"Dominio/esquema no permitido: {u.scheme}://{u.hostname}")


def descargar(url: str, destino: Path, reintentos: int = 3, timeout: int = 120) -> tuple[int, int]:
    """Descarga en streaming a un .part y lo renombra al terminar. Devuelve (http_status, bytes)."""
    _validar_url(url)
    destino.parent.mkdir(parents=True, exist_ok=True)
    abrir = urllib.request.build_opener(_Redireccion).open
    ultimo: Exception | None = None
    for intento in range(1, reintentos + 1):
        parcial = destino.with_suffix(destino.suffix + ".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fsai-descarga-publica/2.0"})
            with abrir(req, timeout=timeout) as r, parcial.open("wb") as fh:
                estado = r.status
                total = 0
                for bloque in iter(lambda: r.read(1 << 20), b""):
                    fh.write(bloque)
                    total += len(bloque)
            if total == 0:
                raise OSError("respuesta vacia")
            parcial.replace(destino)
            return estado, total
        except Exception as exc:  # noqa: BLE001 (se registra el motivo y se reintenta)
            ultimo = exc
            parcial.unlink(missing_ok=True)
            if intento < reintentos:
                time.sleep(2 * intento)
    raise ultimo  # type: ignore[misc]


def _cargar_manifiesto(ruta: Path) -> dict[str, dict]:
    if not ruta.exists():
        return {}
    try:
        return {e["id"]: e for e in json.loads(ruta.read_text(encoding="utf-8")).get("fuentes", [])}
    except (OSError, ValueError, KeyError):
        return {}


def procesar(fuente: dict, previo: dict | None, forzar: bool) -> dict:
    destino = DIR_DATOS / fuente["archivo"]
    ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = {k: fuente.get(k) for k in ("id", "url", "licencia", "granularidad", "anios", "nota")}
    base["archivo"] = fuente["archivo"]
    try:
        if destino.exists() and destino.stat().st_size > 0 and not forzar:
            estado_http = (previo or {}).get("http_status")
            return {**base, "fecha": (previo or {}).get("fecha", ahora), "http_status": estado_http, "bytes": destino.stat().st_size,
                    "sha256": sha256_archivo(destino), "estado": "ok", "motivo": "reutilizado (archivo ya presente)"}
        estado_http, n = descargar(fuente["url"], destino)
        return {**base, "fecha": ahora, "http_status": estado_http, "bytes": n, "sha256": sha256_archivo(destino),
                "estado": "ok", "motivo": "descargado"}
    except urllib.error.HTTPError as exc:
        return {**base, "fecha": ahora, "http_status": exc.code, "bytes": 0, "sha256": None, "estado": "fallo", "motivo": f"HTTP {exc.code}: {exc.reason}"}
    except Exception as exc:  # noqa: BLE001
        return {**base, "fecha": ahora, "http_status": None, "bytes": 0, "sha256": None, "estado": "fallo", "motivo": f"{type(exc).__name__}: {exc}"}


def _selecciona(f: dict, solo: list[str] | None) -> bool:
    return not solo or f["grupo"] in solo or f["id"] in solo


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo", nargs="+", default=None, help="grupos o ids: geojson endes ophi noaa_oni power <id>")
    ap.add_argument("--forzar", action="store_true", help="volver a descargar aunque el archivo exista")
    ap.add_argument("--derivar", action="store_true",
                    help="tras descargar, construye datos_derivados/indicadores_territoriales_departamento.csv (+ oni_mensual.csv y manifiesto)")
    a = ap.parse_args(argv)
    DIR_DATOS.mkdir(parents=True, exist_ok=True)
    ruta_man = DIR_DATOS / "MANIFIESTO_ADICIONALES.json"
    previos = _cargar_manifiesto(ruta_man)
    resultados: dict[str, dict] = dict(previos)

    for f in (x for x in _fuentes_estaticas() if _selecciona(x, a.solo)):
        resultados[f["id"]] = procesar(f, previos.get(f["id"]), a.forzar)
        print(f"{f['id']}: {resultados[f['id']]['estado']} ({resultados[f['id']]['motivo']})")

    pide_clima = {"power": ("nasa_power_", fuentes_power), "openmeteo": ("openmeteo_", fuentes_openmeteo)}
    for grupo, (prefijo, fabrica) in pide_clima.items():
        if not (_selecciona({"grupo": grupo, "id": grupo}, a.solo) or any(str(x).startswith(prefijo) for x in (a.solo or []))):
            continue
        from fsai.fuentes_territoriales import centroides_departamentos
        ruta_geo = DIR_DATOS / "geo" / "peru_departamental_simple.geojson"
        if not ruta_geo.exists():
            resultados[prefijo + "*"] = dict(id=prefijo + "*", url="", archivo="", estado="fallo", http_status=None, bytes=0, sha256=None,
                                             fecha=datetime.now(timezone.utc).isoformat(timespec="seconds"), licencia="", granularidad="punto", anios=None,
                                             motivo="falta el GeoJSON departamental (necesario para calcular centroides); descargue 'geojson' primero")
            print(f"{grupo}: fallo (sin GeoJSON para los centroides)")
            continue
        resultados.pop(prefijo + "*", None)
        for f in (x for x in fabrica(centroides_departamentos(ruta_geo)) if _selecciona(x, a.solo)):
            resultados[f["id"]] = procesar(f, previos.get(f["id"]), a.forzar)
            if resultados[f["id"]]["estado"] == "fallo":
                print(f"{f['id']}: FALLO {resultados[f['id']]['motivo']}")
            time.sleep(0.5)   # cortesia con las APIs
        ok = sum(1 for k, v in resultados.items() if k.startswith(prefijo) and v["estado"] == "ok")
        print(f"{grupo}: {ok} puntos ok")

    manifiesto = dict(generado=datetime.now(timezone.utc).isoformat(timespec="seconds"), directorio=str(DIR_DATOS),
                      nota=("Archivos descargados de fuentes publicas; tratados como no confiables (solo lectura con parseo de datos). "
                            "Un estado 'fallo' significa que la fuente NO se pudo descargar en esa ejecucion."),
                      fuentes=sorted(resultados.values(), key=lambda e: e["id"]))
    ruta_man.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    fallos = [k for k, v in resultados.items() if v["estado"] != "ok"]
    print(f"{len(resultados)} fuentes en {ruta_man} | fallos: {fallos or 'ninguno'}")
    if a.derivar:
        from fsai import fuentes_territoriales as FT
        tabla, oni, informe = FT.construir_tabla_territorial(DIR_DATOS)
        rutas = FT.guardar_tabla_territorial(tabla, oni, informe)
        print(json.dumps(informe, indent=1, ensure_ascii=False, default=float))
        for aviso in FT.verificar_rangos(tabla):
            print("AVISO:", aviso)
        import shutil
        destino_man = rutas["tabla"].parent / ruta_man.name       # copia versionable del manifiesto (URL, fecha, sha256, estado)
        shutil.copyfile(ruta_man, destino_man)
        rutas["manifiesto_descargas"] = destino_man
        print("derivado:", {k: str(v) for k, v in rutas.items()})
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
