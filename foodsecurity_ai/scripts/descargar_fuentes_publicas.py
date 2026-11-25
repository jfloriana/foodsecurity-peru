"""Descarga fuentes PUBLICAS (sin cuenta ni credenciales) y escribe datos_publicos/MANIFIESTO.json.

Solo HTTPS a una lista blanca de dominios oficiales. Los archivos ya presentes se reutilizan
(se recalcula su SHA-256). Todo lo descargado se trata como no confiable (solo se guarda).
Uso (desde foodsecurity_ai/):  python -I scripts/descargar_fuentes_publicas.py
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RAIZ_PAQUETE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_PAQUETE))
from fsai.fuentes_publicas import MODULOS_ENAHO_2025, URL_BASE_INEI, sha256_archivo  # noqa: E402

DATOS = RAIZ_PAQUETE.parents[2] / "datos_publicos"
DOMINIOS_PERMITIDOS = {"proyectos.inei.gob.pe", "api.worldbank.org", "bulks-faostat.fao.org"}
WDI = {"SN.ITK.DEFC.ZS": "Prevalencia de subalimentacion (% poblacion)",
       "SN.ITK.MSFI.ZS": "Prevalencia de inseguridad alimentaria moderada o grave, FIES (% poblacion)",
       "SN.ITK.SVFI.ZS": "Prevalencia de inseguridad alimentaria grave, FIES (% poblacion)",
       "NY.GDP.PCAP.CD": "PIB per capita (US$ corrientes)",
       "NY.GDP.PCAP.PP.KD": "PIB per capita, PPA (US$ constantes 2021)",
       "SI.POV.NAHC": "Pobreza segun linea nacional (% poblacion)",
       "SP.RUR.TOTL.ZS": "Poblacion rural (% del total)"}
FAOSTAT = "https://bulks-faostat.fao.org/production/Food_Security_Data_E_All_Data_(Normalized).zip"


def tareas() -> list[dict]:
    t = []
    for mod, (_, desc) in MODULOS_ENAHO_2025.items():
        t.append(dict(fuente="INEI_ENAHO", url=f"{URL_BASE_INEI}/1031-Modulo{mod}.zip",
                      destino=DATOS / "enaho" / "2025" / "zip" / f"1031-Modulo{mod}.zip",
                      descripcion=f"ENAHO 2025 anual, modulo {mod}: {desc} (CSV)",
                      licencia="No consta licencia/terminos explicitos en las paginas consultadas; datos abiertos del INEI, citar la fuente"))
    for ind, desc in WDI.items():
        t.append(dict(fuente="BancoMundial_WDI", destino=DATOS / "wdi" / "2026" / f"PER_{ind}.json",
                      url=f"https://api.worldbank.org/v2/country/PER/indicator/{ind}?format=json&per_page=200&date=1990:2026",
                      descripcion=f"WDI Peru: {desc}", licencia="Banco Mundial: CC BY 4.0 (terminos de datos abiertos; verificar en la fuente)"))
    t.append(dict(fuente="FAOSTAT", url=FAOSTAT, destino=DATOS / "faostat" / "2026" / Path(urllib.parse.unquote(FAOSTAT)).name,
                  descripcion="FAOSTAT Food Security Data (incluye FIES SDG 2.1.2 agregado por pais/trienio)",
                  licencia="FAOSTAT: no verificada en esta ejecucion; revisar terminos de uso de FAO antes de redistribuir"))
    return t


def main() -> int:
    manifiesto = []
    for t in tareas():
        host = urllib.parse.urlparse(t["url"]).hostname
        if host not in DOMINIOS_PERMITIDOS or not t["url"].startswith("https://"):
            raise SystemExit(f"Dominio no permitido: {host}")
        dest: Path = t["destino"]
        estado = "reutilizado"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(t["url"], headers={"User-Agent": "fsai-descarga-publica/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310 (lista blanca arriba)
                dest.write_bytes(r.read())
            estado = "descargado"
        manifiesto.append(dict(
            fuente=t["fuente"], url=t["url"], archivo=str(dest.relative_to(DATOS)).replace("\\", "/"),
            descripcion=t["descripcion"], fecha_registro=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            tamano_bytes=dest.stat().st_size, sha256=sha256_archivo(dest), licencia_terminos=t["licencia"], estado=estado))
    inaccesibles = [dict(fuente="FAO_Microdata (microdata.fao.org, PER_2024_FIES)",
                         url="https://microdata.fao.org/index.php/catalog/2797",
                         estado="NO_DESCARGADO",
                         motivo="Exige inicio de sesion y firma de declaracion de confidencialidad; solo uso de investigacion/estadistica y no comercial. No se uso (fuera del alcance autorizado).")]
    (DATOS / "MANIFIESTO.json").write_text(json.dumps(dict(
        generado=datetime.now(timezone.utc).isoformat(timespec="seconds"), archivos=manifiesto,
        inaccesibles=inaccesibles), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(manifiesto)} archivos en MANIFIESTO.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
