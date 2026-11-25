"""Construye datos_publicos/procesado/enaho_analitico.parquet desde ENAHO 2025 (INEI, publico).

Uso (desde foodsecurity_ai/):  python -I scripts/construir_dataset_enaho.py [--umbrales leve moderada severa]
El salt de hogar_uid se toma de la variable de entorno FSAI_HOGAR_SALT.
Los umbrales de fies_nivel son [POR VALIDAR]; no representan calibracion peruana.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ_PAQUETE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_PAQUETE))
from fsai.fuentes_publicas import (MODULOS_ENAHO_2025, UMBRALES_FIES_DEFECTO,  # noqa: E402
                                   construir_dataset_enaho, extraer_zip_seguro)

DATOS = RAIZ_PAQUETE.parents[2] / "datos_publicos"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=DATOS / "enaho" / "2025")
    ap.add_argument("--salida", type=Path, default=DATOS / "procesado" / "enaho_analitico.parquet")
    ap.add_argument("--umbrales", type=float, nargs=3, metavar=("LEVE", "MODERADA", "SEVERA"),
                    default=[UMBRALES_FIES_DEFECTO[k] for k in ("leve", "moderada", "severa")])
    a = ap.parse_args()
    ext = a.base / "extraido"
    for mod in MODULOS_ENAHO_2025:
        d = ext / f"1031-Modulo{mod}"
        if not (d / MODULOS_ENAHO_2025[mod][0]).exists():
            extraer_zip_seguro(a.base / "zip" / f"1031-Modulo{mod}.zip", d)
    umb = dict(zip(("leve", "moderada", "severa"), a.umbrales))
    df, inf = construir_dataset_enaho(ext, umbrales=umb)
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.salida, index=False)
    a.salida.with_suffix(".calidad.json").write_text(json.dumps(inf, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: inf[k] for k in ("n_filas", "fies_nivel_valido", "fies_nivel_conteo")}, ensure_ascii=False))
    print(f"-> {a.salida} ({df.shape[0]} filas x {df.shape[1]} columnas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
