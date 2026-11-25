"""Bitácora de auditoría (solo administración): consulta fsai.auditoria con filtros de fecha, usuario y acción y descarga CSV.

Solo lectura. Nunca selecciona ni muestra claves, hashes ni sales: se omiten las claves sensibles del campo «detalle» y los valores
largos se recortan. Sin base de datos muestra un aviso y se detiene.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import streamlit as st

import modulos
from auth import barra_lateral, requerir_acceso
from comun import ErrorConsulta, consultar, leer_json, mostrar_error_bd
from tema import config_pagina, encabezado

modulos.registrar_accesos()
config_pagina("Bitácora de auditoría")
usuario, rol = requerir_acceso("bitacora")
barra_lateral()

encabezado(
    "Bitácora de auditoría",
    "Consulte y exporte las acciones registradas en la plataforma (accesos, entrenamientos, descargas).",
    "Resultados y despliegue",
    pasos=["Elija el rango de fechas y, si quiere, un usuario o una acción.",
           "Revise la tabla (más recientes primero; máximo según el límite elegido).",
           "Descargue el CSV si necesita conservar la evidencia."],
    produce="Tabla de eventos de auditoría; la bitácora es de solo inserción y esta página no la modifica.",
)

CLAVES_SENSIBLES = ("clave", "password", "passwd", "hash", "salt", "secret", "token", "credencial")
SQL_LISTAS = "SELECT DISTINCT usuario, accion FROM fsai.auditoria ORDER BY 1, 2"
SQL_EVENTOS = """
SELECT auditoria_id, ocurrido_en, usuario, accion, entidad, entidad_id, detalle
FROM fsai.auditoria
WHERE ocurrido_en >= :desde AND ocurrido_en < :hasta
  AND (CAST(:usuario AS text) IS NULL OR usuario = :usuario)
  AND (CAST(:accion AS text) IS NULL OR accion = :accion)
ORDER BY ocurrido_en DESC
LIMIT :limite
"""


def depurar_detalle(valor: object) -> str:
    """Detalle JSON como texto corto, sin claves sensibles (a cualquier profundidad)."""
    def limpiar(x):
        if isinstance(x, dict):
            return {k: ("[omitido]" if any(s in str(k).lower() for s in CLAVES_SENSIBLES) else limpiar(v)) for k, v in x.items()}
        if isinstance(x, list):
            return [limpiar(v) for v in x[:20]]
        return x
    base = valor if isinstance(valor, (dict, list)) else leer_json(valor)
    texto = json.dumps(limpiar(base), ensure_ascii=False, default=str)
    return texto if len(texto) <= 300 else texto[:297] + "..."


try:
    listas = consultar(SQL_LISTAS, cache=False)
except ErrorConsulta as exc:
    mostrar_error_bd(exc)
    st.info("No hay conexión con la base de datos, así que la bitácora no está disponible. **Siguiente paso:** inicie PostgreSQL y recargue la página.")
    st.stop()
except Exception:  # noqa: BLE001 - p. ej. esquema fsai no creado
    st.info("La tabla fsai.auditoria no está disponible. **Siguiente paso:** cree el esquema con sql/01_ddl_fsai.sql y recargue la página.")
    st.stop()

c1, c2, c3 = st.columns(3)
hoy = date.today()
rango = c1.date_input("Rango de fechas", value=(hoy - timedelta(days=7), hoy), max_value=hoy, help="Incluye ambos días.")
usuarios = ["(todos)", *sorted(listas["usuario"].dropna().unique())]
acciones = ["(todas)", *sorted(listas["accion"].dropna().unique())]
f_usuario = c2.selectbox("Usuario", usuarios)
f_accion = c3.selectbox("Acción", acciones)
limite = st.select_slider("Máximo de filas", options=[100, 500, 1000, 5000], value=500)

if not (isinstance(rango, (tuple, list)) and len(rango) == 2):
    st.info("Elija la fecha inicial y la final del rango.")
    st.stop()
desde, hasta = rango[0], rango[1] + timedelta(days=1)

try:
    eventos = consultar(SQL_EVENTOS, {"desde": desde.isoformat(), "hasta": hasta.isoformat(),
                                      "usuario": None if f_usuario == "(todos)" else f_usuario,
                                      "accion": None if f_accion == "(todas)" else f_accion, "limite": int(limite)}, cache=False)
except ErrorConsulta as exc:
    mostrar_error_bd(exc)
    st.stop()
except Exception:  # noqa: BLE001
    st.error("No se pudo consultar la bitácora. Verifique que PostgreSQL esté activo y que el esquema fsai exista.")
    st.stop()

if eventos.empty:
    st.info("No hay eventos con esos filtros. Amplíe el rango de fechas.")
    st.stop()

eventos["detalle"] = eventos["detalle"].map(depurar_detalle)
eventos["ocurrido_en"] = pd.to_datetime(eventos["ocurrido_en"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S UTC")
st.caption(f"{len(eventos)} eventos" + (f" (se alcanzó el máximo de {limite}; acote los filtros para ver el resto)" if len(eventos) >= limite else "") + ".")
st.dataframe(eventos, hide_index=True, use_container_width=True)
st.download_button("Descargar CSV", eventos.to_csv(index=False).encode("utf-8-sig"), file_name=f"bitacora_{desde}_{rango[1]}.csv", mime="text/csv")
