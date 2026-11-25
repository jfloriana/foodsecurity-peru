"""Autenticación básica y control de acceso por rol para FoodSecurity AI.

Los usuarios se leen de variables de entorno (no hay credenciales embebidas):

* ``FSAI_USUARIOS``: lista JSON de registros
  ``{"usuario": <texto>, "rol": <rol>, "salt": <hex>, "iteraciones": <int>, "hash": <hex>}``.
* ``FSAI_USUARIOS_ARCHIVO``: ruta a un archivo con el mismo JSON (p. ej. un Docker secret).

El hash es PBKDF2-HMAC-SHA256 (``hashlib.pbkdf2_hmac``). Para generar un registro:
``python app/auth.py`` (solicita usuario, rol y clave sin eco y escribe el JSON por stdout).
"""
from __future__ import annotations

import getpass
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from pathlib import Path

import streamlit as st

import tema
from comun import registrar_auditoria

ROLES = ("admin", "analista", "lector")
ITERACIONES_DEFECTO = 600_000
ITERACIONES_MIN = 100_000
ITERACIONES_MAX = 5_000_000
MAX_INTENTOS = 5
BLOQUEO_SEGUNDOS = 300
_SAL_FICTICIA = bytes(16)

# Roles con acceso a cada módulo (clave = identificador del módulo, ver sección 12.1).
ACCESO_PAGINAS: dict[str, tuple[str, ...]] = {
    "dashboard": ROLES,
    "importacion_enaho": ("admin",),
    "procesamiento_fies": ("admin", "analista"),
    "eda": ROLES,
    "vulnerabilidad": ROLES,
    "entrenamiento": ("admin", "analista"),
    "optuna": ("admin", "analista"),
    "comparacion": ROLES,
    "shap": ROLES,
    "territorial": ROLES,
    "equidad": ROLES,
    "reportes": ("admin", "analista"),
    # Predicción individual y por lote.
    "inferencia": ROLES,
}
# Acciones de escritura dentro de páginas de consulta (subir archivos, guardar metadata, consumir la Prueba).
ROLES_ESCRITURA = ("admin", "analista")


def _minutos_sesion() -> float:
    try:
        return float(os.environ.get("FSAI_SESION_MIN", "60"))
    except ValueError:
        return 60.0


def hash_clave(clave: str, salt: bytes, iteraciones: int = ITERACIONES_DEFECTO) -> str:
    return hashlib.pbkdf2_hmac("sha256", clave.encode("utf-8"), salt, iteraciones).hex()


def crear_registro(usuario: str, rol: str, clave: str) -> dict:
    """Genera el registro JSON de un usuario (para pegar en FSAI_USUARIOS)."""
    if rol not in ROLES:
        raise ValueError(f"Rol inválido: {rol!r}. Opciones: {ROLES}")
    salt = secrets.token_bytes(16)
    return {
        "usuario": usuario,
        "rol": rol,
        "salt": salt.hex(),
        "iteraciones": ITERACIONES_DEFECTO,
        "hash": hash_clave(clave, salt, ITERACIONES_DEFECTO),
    }


def _cargar_usuarios() -> dict[str, dict]:
    bruto = os.environ.get("FSAI_USUARIOS", "").strip()
    ruta = os.environ.get("FSAI_USUARIOS_ARCHIVO", "").strip()
    if not bruto and ruta:
        try:
            bruto = Path(ruta).read_text(encoding="utf-8")
        except OSError:
            return {}
    if not bruto:
        return {}
    try:
        registros = json.loads(bruto)
    except json.JSONDecodeError:
        return {}
    if not isinstance(registros, list):
        return {}
    usuarios: dict[str, dict] = {}
    for reg in registros:
        try:
            rol = str(reg["rol"])
            iteraciones = int(reg["iteraciones"])
            if rol not in ROLES or not ITERACIONES_MIN <= iteraciones <= ITERACIONES_MAX:
                continue
            usuarios[str(reg["usuario"])] = {
                "rol": rol,
                "salt": bytes.fromhex(reg["salt"]),
                "iteraciones": iteraciones,
                "hash": bytes.fromhex(reg["hash"]),
            }
        except (KeyError, ValueError, TypeError):
            continue
    return usuarios


def verificar_credenciales(usuario: str, clave: str) -> str | None:
    """Devuelve el rol si las credenciales son válidas; None en otro caso."""
    registro = _cargar_usuarios().get(usuario)
    if registro is None:
        # Cálculo ficticio para igualar el tiempo de respuesta y no revelar usuarios.
        hashlib.pbkdf2_hmac("sha256", clave.encode("utf-8"), _SAL_FICTICIA, ITERACIONES_DEFECTO)
        return None
    calculado = hashlib.pbkdf2_hmac(
        "sha256", clave.encode("utf-8"), registro["salt"], registro["iteraciones"]
    )
    return registro["rol"] if hmac.compare_digest(calculado, registro["hash"]) else None


def sesion_activa() -> bool:
    """True si hay sesión válida; renueva la marca de actividad (expiración deslizante)."""
    if not st.session_state.get("auth_usuario"):
        return False
    ahora = time.time()
    if ahora - st.session_state.get("auth_ultimo", 0.0) > _minutos_sesion() * 60:
        cerrar_sesion(motivo="expirada")
        return False
    st.session_state["auth_ultimo"] = ahora
    return True


def iniciar_sesion(usuario: str, rol: str) -> None:
    st.session_state["auth_usuario"] = usuario
    st.session_state["auth_rol"] = rol
    st.session_state["auth_ultimo"] = time.time()
    st.session_state["auth_fallos"] = 0
    st.session_state["auth_paginas_auditadas"] = set()


def cerrar_sesion(motivo: str = "cierre_manual") -> None:
    usuario = st.session_state.get("auth_usuario")
    if usuario:
        registrar_auditoria("logout", "sesion", None, {"motivo": motivo}, usuario=usuario)
    for clave in list(st.session_state.keys()):
        del st.session_state[clave]


def formulario_login() -> None:
    """Dibuja la pantalla de acceso (identidad, formulario, aviso de uso y pie) y procesa el intento."""
    izq, centro = st.columns([1.25, 1], gap="large")
    with izq:
        tema.marca_acceso()
    with centro:
        if not _cargar_usuarios():
            st.error(
                "No hay usuarios configurados. Defina FSAI_USUARIOS (o FSAI_USUARIOS_ARCHIVO) "
                "con registros generados mediante `python app/auth.py`."
            )
            tema.pie_pagina()
            return
        restante = st.session_state.get("auth_bloqueo_hasta", 0.0) - time.time()
        if restante > 0:
            minutos, segundos = divmod(int(restante) + 1, 60)
            st.subheader("Acceso a la plataforma")
            st.error(
                "Acceso bloqueado temporalmente por intentos fallidos. "
                f"Podrá intentarlo de nuevo en aproximadamente {minutos} min {segundos:02d} s."
            )
            tema.aviso_uso_acceso()
            tema.pie_pagina()
            return
        with st.form("form_login", clear_on_submit=False):
            st.subheader("Acceso a la plataforma")
            st.caption("Ingrese con las credenciales asignadas por el administrador del estudio.")
            usuario = st.text_input("Usuario", max_chars=64, autocomplete="username")
            clave = st.text_input("Clave", type="password", max_chars=128, autocomplete="current-password")
            enviar = st.form_submit_button("Ingresar", type="primary", use_container_width=True)
        fallos_previos = st.session_state.get("auth_fallos", 0)
        if not enviar:
            if fallos_previos:
                tema.indicador_intentos(fallos_previos, MAX_INTENTOS)
            tema.aviso_uso_acceso()
            tema.pie_pagina()
            return
        rol = verificar_credenciales(usuario.strip(), clave)
        if rol is None:
            fallos = fallos_previos + 1
            st.session_state["auth_fallos"] = fallos
            registrar_auditoria(
                "login_fallido", "sesion", None, {"usuario_intentado": usuario.strip()[:64]},
                usuario="anonimo",
            )
            if fallos >= MAX_INTENTOS:
                st.session_state["auth_bloqueo_hasta"] = time.time() + BLOQUEO_SEGUNDOS
                st.session_state["auth_fallos"] = 0
                st.error(
                    "No fue posible verificar las credenciales. Se alcanzó el límite de intentos: el acceso "
                    f"queda bloqueado durante {BLOQUEO_SEGUNDOS // 60} minutos."
                )
            else:
                st.error("No fue posible verificar las credenciales. Revise el usuario y la clave e intente de nuevo.")
                tema.indicador_intentos(fallos, MAX_INTENTOS)
            tema.aviso_uso_acceso()
            tema.pie_pagina()
            return
        iniciar_sesion(usuario.strip(), rol)
        registrar_auditoria("login_exitoso", "sesion", None, {"rol": rol})
        st.rerun()


def barra_lateral() -> None:
    """Muestra marca, usuario/rol y cierre de sesión en la barra lateral.

    Si el enrutador (app.py) ya la dibujó en esta ejecución, no se repite en las páginas.
    """
    if st.session_state.get("_fsai_barra_en_router"):
        return
    with st.sidebar:
        tema.usuario_lateral(st.session_state.get("auth_usuario", "—"), st.session_state.get("auth_rol", "—"))
        if st.button("Cerrar sesión", key="btn_cerrar_sesion", use_container_width=True):
            cerrar_sesion()
            st.rerun()


def requerir_acceso(pagina: str) -> tuple[str, str]:
    """Detiene la página si no hay sesión o el rol no tiene permiso; audita el acceso."""
    if not sesion_activa():
        st.warning("Sesión no iniciada o expirada. Ingrese desde la página principal.")
        try:
            st.page_link("app.py", label="Ir al inicio de sesión")
        except Exception:  # noqa: BLE001 - con st.navigation el enrutador es app.py y puede no ser página
            st.markdown("Recargue la página principal para iniciar sesión.")
        st.stop()
    usuario = st.session_state["auth_usuario"]
    rol = st.session_state["auth_rol"]
    if rol not in ACCESO_PAGINAS.get(pagina, ()):
        registrar_auditoria("acceso_denegado", "pagina", pagina)
        st.error(f"El rol «{rol}» no tiene acceso a este módulo.")
        st.stop()
    vistas = st.session_state.setdefault("auth_paginas_auditadas", set())
    if pagina not in vistas:
        vistas.add(pagina)
        registrar_auditoria("acceso_pagina", "pagina", pagina)
    return usuario, rol


if __name__ == "__main__":
    nombre = input("Usuario: ").strip()
    rol_in = input(f"Rol {ROLES}: ").strip()
    clave_in = getpass.getpass("Clave: ")
    sys.stdout.write(json.dumps(crear_registro(nombre, rol_in, clave_in), indent=2) + "\n")
