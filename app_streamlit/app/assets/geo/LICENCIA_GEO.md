# GeoJSON departamental del Perú

Archivo: `peru_departamental.geojson` (copia sin modificar de la fuente).

## Verificado (2026-11-16)
- Origen: https://raw.githubusercontent.com/juaneladio/peru-geojson/master/peru_departamental_simple.geojson (HTTP 200, sin login).
- 25 features (24 departamentos + Callao; Lima en una sola unidad). Propiedades: `NOMBDEP` (mayúsculas, sin tildes), `FIRST_IDDP` ("01".."25"), `COUNT`, `HECTARES`.
- Geometría simplificada (Polygon/MultiPolygon, CRS84). SHA-256 registrado en `datos_publicos/MANIFIESTO_ADICIONALES.json` al descargar (no versionado) y reproducible con `sha256sum`.

## NO verificado
- **Licencia: no declarada en el repositorio de origen; verificar antes de redistribuir.** Este archivo se incluye solo como activo de visualización del prototipo.
- Exactitud cartográfica: es una versión simplificada, apta para mapas coropléticos, no para mediciones de área/frontera.
- No se confirmó el organismo productor original (probablemente derivado de límites oficiales del INEI/IGN).

## Respaldo (no incluido)
- geoBoundaries gbOpen PER ADM1 (Public Domain según su API): https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/PER/ADM1/geoBoundaries-PER-ADM1_simplified.geojson
  (26 unidades: Lima Provincia y Callao separados; el nombre del departamento viene en `shapeName` y no coincide con `NOMBDEP`).
  Si la licencia del archivo principal impide redistribuirlo, sustituirlo por este y ajustar la clave de nombre.
