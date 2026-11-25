"""database.py - Modelo ORM (SQLAlchemy 2.0) del esquema PostgreSQL `fsai` y funciones guardar_*.

Las 12 tablas replican exactamente el contrato (nombres, PK, CHECK). Se usa el tipo JSONB
de PostgreSQL. Las tablas internas de Optuna (estudios/ensayos) las crea Optuna aparte.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sqlalchemy import (CHAR, BigInteger, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Integer, SmallInteger,
                        String, Text, UniqueConstraint, create_engine, func, insert, text)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from fsai.config import DATABASE_URL, SALT
from fsai.data_loader import hash_sha256

logger = logging.getLogger(__name__)
ESQUEMA = "fsai"


class Base(DeclarativeBase):
    """Base declarativa; todas las tablas viven en el esquema 'fsai'."""


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = {"schema": ESQUEMA}
    dataset_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    nombre: Mapped[str] = mapped_column(String(120), nullable=False)
    fuente: Mapped[str | None] = mapped_column(String(200))
    anio: Mapped[int | None] = mapped_column(SmallInteger)
    version: Mapped[str | None] = mapped_column(String(40))
    hash_archivo: Mapped[str | None] = mapped_column(String(64))
    n_filas: Mapped[int | None] = mapped_column(Integer)
    creado_en: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Hogar(Base):
    __tablename__ = "hogares"
    __table_args__ = (
        CheckConstraint("area IN ('U','R')", name="ck_hogares_area"),
        CheckConstraint("region_natural IN ('costa','sierra','selva')", name="ck_hogares_region"),
        CheckConstraint("factor_expansion > 0", name="ck_hogares_factor"),
        {"schema": ESQUEMA})
    hogar_uid: Mapped[str] = mapped_column(CHAR(64), primary_key=True)   # hex SHA-256
    dataset_id: Mapped[int] = mapped_column(ForeignKey(f"{ESQUEMA}.datasets.dataset_id"), nullable=False)
    conglome_hash: Mapped[str | None] = mapped_column(String(64))
    ubigeo: Mapped[str | None] = mapped_column(CHAR(6))
    departamento: Mapped[str | None] = mapped_column(String(60))
    provincia: Mapped[str | None] = mapped_column(String(60))
    distrito: Mapped[str | None] = mapped_column(String(80))
    area: Mapped[str] = mapped_column(String(1), nullable=False)
    region_natural: Mapped[str | None] = mapped_column(String(10))
    factor_expansion: Mapped[float] = mapped_column(Float, nullable=False)
    anio: Mapped[int | None] = mapped_column(SmallInteger)


class PersonaAnonimizada(Base):
    __tablename__ = "personas_anonimizadas"
    __table_args__ = (CheckConstraint("sexo IN ('H','M')", name="ck_personas_sexo"), {"schema": ESQUEMA})
    persona_uid: Mapped[str] = mapped_column(String(64), primary_key=True)
    hogar_uid: Mapped[str] = mapped_column(CHAR(64), ForeignKey(f"{ESQUEMA}.hogares.hogar_uid"), nullable=False)
    parentesco: Mapped[int | None] = mapped_column(SmallInteger)
    edad: Mapped[int | None] = mapped_column(SmallInteger)
    sexo: Mapped[str | None] = mapped_column(String(1))
    nivel_educ: Mapped[int | None] = mapped_column(SmallInteger)
    condicion_actividad: Mapped[str | None] = mapped_column(String(40))
    es_jefe: Mapped[bool | None] = mapped_column(Boolean)


class RespuestaFies(Base):
    __tablename__ = "respuestas_fies"
    __table_args__ = (
        UniqueConstraint("hogar_uid", name="uq_respuestas_fies_hogar"),
        *[CheckConstraint(f"fies_q{i} IN (0,1)", name=f"ck_fies_q{i}") for i in range(1, 9)],
        CheckConstraint("fies_raw_score BETWEEN 0 AND 8", name="ck_fies_raw"),
        CheckConstraint("fies_nivel BETWEEN 0 AND 3", name="ck_fies_nivel"),
        {"schema": ESQUEMA})
    respuesta_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    hogar_uid: Mapped[str] = mapped_column(CHAR(64), ForeignKey(f"{ESQUEMA}.hogares.hogar_uid"), nullable=False)
    fies_q1: Mapped[int | None] = mapped_column(SmallInteger)
    fies_q2: Mapped[int | None] = mapped_column(SmallInteger)
    fies_q3: Mapped[int | None] = mapped_column(SmallInteger)
    fies_q4: Mapped[int | None] = mapped_column(SmallInteger)
    fies_q5: Mapped[int | None] = mapped_column(SmallInteger)
    fies_q6: Mapped[int | None] = mapped_column(SmallInteger)
    fies_q7: Mapped[int | None] = mapped_column(SmallInteger)
    fies_q8: Mapped[int | None] = mapped_column(SmallInteger)
    fies_raw_score: Mapped[int | None] = mapped_column(SmallInteger)
    fies_rasch_logit: Mapped[float | None] = mapped_column(Float)
    fies_nivel: Mapped[int | None] = mapped_column(SmallInteger)
    ia_binaria: Mapped[int | None] = mapped_column(SmallInteger)


class IndicadorAlimentario(Base):
    __tablename__ = "indicadores_alimentarios"
    __table_args__ = {"schema": ESQUEMA}
    indicador_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    hogar_uid: Mapped[str] = mapped_column(CHAR(64), ForeignKey(f"{ESQUEMA}.hogares.hogar_uid"), nullable=False)
    gasto_alim_pc: Mapped[float | None] = mapped_column(Float)
    share_alimentario: Mapped[float | None] = mapped_column(Float)
    ingreso_pc: Mapped[float | None] = mapped_column(Float)
    gasto_pc: Mapped[float | None] = mapped_column(Float)
    pobreza_monetaria: Mapped[int | None] = mapped_column(SmallInteger)


class CondicionSocioeconomica(Base):
    __tablename__ = "condiciones_socioeconomicas"
    __table_args__ = (CheckConstraint("nivel_educ_jefe BETWEEN 0 AND 11", name="ck_cond_educ"),
                      {"schema": ESQUEMA})
    condicion_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    hogar_uid: Mapped[str] = mapped_column(CHAR(64), ForeignKey(f"{ESQUEMA}.hogares.hogar_uid"), nullable=False)
    n_miembros: Mapped[int | None] = mapped_column(SmallInteger)
    n_menores5: Mapped[int | None] = mapped_column(SmallInteger)
    n_menores18: Mapped[int | None] = mapped_column(SmallInteger)
    n_adultos_mayores: Mapped[int | None] = mapped_column(SmallInteger)
    edad_jefe: Mapped[int | None] = mapped_column(SmallInteger)
    sexo_jefe: Mapped[str | None] = mapped_column(String(1))
    nivel_educ_jefe: Mapped[int | None] = mapped_column(SmallInteger)
    asistencia_escolar: Mapped[float | None] = mapped_column(Float)
    n_ocupados: Mapped[int | None] = mapped_column(SmallInteger)
    informalidad: Mapped[float | None] = mapped_column(Float)
    desempleo: Mapped[int | None] = mapped_column(SmallInteger)
    dependencia_economica: Mapped[float | None] = mapped_column(Float)
    agua_potable: Mapped[int | None] = mapped_column(SmallInteger)
    saneamiento: Mapped[int | None] = mapped_column(SmallInteger)
    hacinamiento: Mapped[float | None] = mapped_column(Float)
    tipo_vivienda: Mapped[str | None] = mapped_column(String(60))
    material_vivienda: Mapped[str | None] = mapped_column(String(60))


class Experimento(Base):
    __tablename__ = "experimentos"
    __table_args__ = (
        CheckConstraint("esquema_validacion IN ('V1','V1c','V2','V3','V4','V5')", name="ck_exp_esquema"),
        CheckConstraint("escenario IN ('socio','fies','hibrido')", name="ck_exp_escenario"),
        {"schema": ESQUEMA})
    experimento_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey(f"{ESQUEMA}.datasets.dataset_id"), nullable=False)
    nombre: Mapped[str] = mapped_column(String(160), nullable=False)
    esquema_validacion: Mapped[str] = mapped_column(String(3), nullable=False)
    escenario: Mapped[str] = mapped_column(String(10), nullable=False)
    semilla: Mapped[int] = mapped_column(Integer, nullable=False)
    config_json: Mapped[dict | None] = mapped_column(JSONB)
    iniciado_en: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finalizado_en: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    estado: Mapped[str] = mapped_column(String(20), default="en_curso")


class Modelo(Base):
    __tablename__ = "modelos"
    __table_args__ = (CheckConstraint("codigo IN ('M1','M2','M3','M4','M5')", name="ck_modelos_codigo"),
                      {"schema": ESQUEMA})
    modelo_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experimento_id: Mapped[int] = mapped_column(ForeignKey(f"{ESQUEMA}.experimentos.experimento_id"), nullable=False)
    codigo: Mapped[str] = mapped_column(String(2), nullable=False)
    hiperparametros: Mapped[dict | None] = mapped_column(JSONB)
    n_trials: Mapped[int | None] = mapped_column(Integer)
    mejor_valor_objetivo: Mapped[float | None] = mapped_column(Float)
    ruta_artefacto: Mapped[str | None] = mapped_column(Text)
    creado_en: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Prediccion(Base):
    __tablename__ = "predicciones"
    __table_args__ = {"schema": ESQUEMA}
    prediccion_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    modelo_id: Mapped[int] = mapped_column(ForeignKey(f"{ESQUEMA}.modelos.modelo_id"), nullable=False)
    hogar_uid: Mapped[str] = mapped_column(CHAR(64), ForeignKey(f"{ESQUEMA}.hogares.hogar_uid"), nullable=False)
    fold: Mapped[int | None] = mapped_column(SmallInteger)
    y_real: Mapped[int | None] = mapped_column(SmallInteger)
    y_pred: Mapped[int | None] = mapped_column(SmallInteger)
    proba_json: Mapped[dict | list | None] = mapped_column(JSONB)


class Metrica(Base):
    __tablename__ = "metricas"
    __table_args__ = {"schema": ESQUEMA}
    metrica_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    modelo_id: Mapped[int] = mapped_column(ForeignKey(f"{ESQUEMA}.modelos.modelo_id"), nullable=False)
    esquema_validacion: Mapped[str | None] = mapped_column(String(3))
    ponderada: Mapped[bool] = mapped_column(Boolean, nullable=False)
    nombre_metrica: Mapped[str] = mapped_column(String(60), nullable=False)
    valor: Mapped[float | None] = mapped_column(Float)
    ic_inf: Mapped[float | None] = mapped_column(Float)
    ic_sup: Mapped[float | None] = mapped_column(Float)
    fold: Mapped[int | None] = mapped_column(SmallInteger)
    estrato: Mapped[str | None] = mapped_column(String(60))


class Explicacion(Base):
    __tablename__ = "explicaciones"
    __table_args__ = (
        CheckConstraint("tipo IN ('shap_global','shap_local','lime','permutation','pdp','ale')", name="ck_expl_tipo"),
        {"schema": ESQUEMA})
    explicacion_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    modelo_id: Mapped[int] = mapped_column(ForeignKey(f"{ESQUEMA}.modelos.modelo_id"), nullable=False)
    hogar_uid: Mapped[str | None] = mapped_column(CHAR(64), ForeignKey(f"{ESQUEMA}.hogares.hogar_uid"))
    tipo: Mapped[str] = mapped_column(String(20), nullable=False)
    contribuciones: Mapped[dict | list | None] = mapped_column(JSONB)
    creado_en: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Auditoria(Base):
    __tablename__ = "auditoria"
    __table_args__ = {"schema": ESQUEMA}
    auditoria_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    usuario: Mapped[str | None] = mapped_column(String(80))
    accion: Mapped[str] = mapped_column(String(80), nullable=False)
    entidad: Mapped[str | None] = mapped_column(String(60))
    entidad_id: Mapped[str | None] = mapped_column(String(64))
    detalle: Mapped[dict | None] = mapped_column(JSONB)
    ocurrido_en: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# --------------------------------------------------------------------------
# Conexión y utilidades
# --------------------------------------------------------------------------
def get_engine(url: str | None = None) -> Engine:
    """Motor SQLAlchemy 2.0 (psycopg2). La URL viene de FSAI_DATABASE_URL; nunca se escribe en código."""
    url = url or DATABASE_URL
    if not url:
        raise RuntimeError("Defina FSAI_DATABASE_URL (p. ej. postgresql+psycopg2://usuario:clave@host:5432/db).")
    return create_engine(url, pool_pre_ping=True, future=True)


def crear_esquema(engine: Engine) -> None:
    """Crea el esquema 'fsai' y las 12 tablas si no existen (idempotente)."""
    with engine.begin() as con:
        con.execute(text(f"CREATE SCHEMA IF NOT EXISTS {ESQUEMA}"))
    Base.metadata.create_all(engine)


def _py(v: Any) -> Any:
    """NumPy/pandas -> tipos nativos (NaN/NA -> None) para que psycopg2 los acepte."""
    if v is None or v is pd.NA:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _filas(df: pd.DataFrame, columnas: Iterable[str]) -> list[dict[str, Any]]:
    cols = [c for c in columnas if c in df.columns]
    return [{c: _py(v) for c, v in zip(cols, fila)} for fila in df[cols].itertuples(index=False, name=None)]


def _insertar_lotes(session: Session, modelo: type[Base], filas: list[dict[str, Any]], lote: int = 5000) -> int:
    for i in range(0, len(filas), lote):
        session.execute(insert(modelo), filas[i:i + lote])
    return len(filas)


# --------------------------------------------------------------------------
# guardar_*
# --------------------------------------------------------------------------
def guardar_dataset(session: Session, nombre: str, fuente: str, anio: int | None, version: str,
                    hash_archivo: str, n_filas: int) -> int:
    obj = Dataset(nombre=nombre, fuente=fuente, anio=anio, version=version, hash_archivo=hash_archivo, n_filas=n_filas)
    session.add(obj)
    session.flush()
    return obj.dataset_id


def guardar_hogares(session: Session, df: pd.DataFrame, dataset_id: int) -> int:
    """Inserta hogares. Se guarda solo el HASH del conglomerado (minimización de datos)."""
    d = df.copy()
    d["dataset_id"] = dataset_id
    d["conglome_hash"] = d["conglome"].astype(str).map(lambda c: hash_sha256(f"{c}|{SALT}"))
    return _insertar_lotes(session, Hogar, _filas(d, [
        "hogar_uid", "dataset_id", "conglome_hash", "ubigeo", "departamento", "provincia", "distrito",
        "area", "region_natural", "factor_expansion", "anio"]))


def guardar_respuestas_fies(session: Session, df: pd.DataFrame) -> int:
    return _insertar_lotes(session, RespuestaFies, _filas(df, [
        "hogar_uid", *[f"fies_q{i}" for i in range(1, 9)], "fies_raw_score", "fies_rasch_logit",
        "fies_nivel", "ia_binaria"]))


def guardar_indicadores(session: Session, df: pd.DataFrame) -> int:
    return _insertar_lotes(session, IndicadorAlimentario, _filas(df, [
        "hogar_uid", "gasto_alim_pc", "share_alimentario", "ingreso_pc", "gasto_pc", "pobreza_monetaria"]))


def guardar_condiciones(session: Session, df: pd.DataFrame) -> int:
    return _insertar_lotes(session, CondicionSocioeconomica, _filas(df, [
        "hogar_uid", "n_miembros", "n_menores5", "n_menores18", "n_adultos_mayores", "edad_jefe", "sexo_jefe",
        "nivel_educ_jefe", "asistencia_escolar", "n_ocupados", "informalidad", "desempleo",
        "dependencia_economica", "agua_potable", "saneamiento", "hacinamiento", "tipo_vivienda",
        "material_vivienda"]))


def guardar_experimento(session: Session, dataset_id: int, nombre: str, esquema: str, escenario: str,
                        semilla: int, config: dict) -> int:
    obj = Experimento(dataset_id=dataset_id, nombre=nombre, esquema_validacion=esquema, escenario=escenario,
                      semilla=semilla, config_json=config)
    session.add(obj)
    session.flush()
    return obj.experimento_id


def finalizar_experimento(session: Session, experimento_id: int, estado: str = "completado") -> None:
    obj = session.get(Experimento, experimento_id)
    if obj is None:
        raise KeyError(f"Experimento {experimento_id} inexistente")
    obj.estado = estado
    obj.finalizado_en = func.now()


def guardar_modelo(session: Session, experimento_id: int, codigo: str, hiperparametros: dict,
                   n_trials: int | None, mejor_valor: float | None, ruta_artefacto: str | None) -> int:
    obj = Modelo(experimento_id=experimento_id, codigo=codigo, hiperparametros=hiperparametros,
                 n_trials=n_trials, mejor_valor_objetivo=mejor_valor, ruta_artefacto=ruta_artefacto)
    session.add(obj)
    session.flush()
    return obj.modelo_id


def guardar_predicciones(session: Session, modelo_id: int, hogar_uid, folds, y_real, y_pred, proba) -> int:
    filas = [{"modelo_id": modelo_id, "hogar_uid": h, "fold": int(f), "y_real": int(a), "y_pred": int(b),
              "proba_json": {str(k): round(float(x), 6) for k, x in enumerate(p)}}   # {"0":p0,..,"3":p3} (CHECK del DDL)
             for h, f, a, b, p in zip(hogar_uid, folds, y_real, y_pred, np.asarray(proba))]
    return _insertar_lotes(session, Prediccion, filas)


def guardar_metricas(session: Session, modelo_id: int, esquema: str, ponderada: bool, metricas: dict[str, float],
                     ics: dict[str, tuple[float, float]] | None = None, fold: int | None = None,
                     estrato: str | None = None) -> int:
    ics = ics or {}
    filas = [{"modelo_id": modelo_id, "esquema_validacion": esquema, "ponderada": ponderada, "nombre_metrica": k,
              "valor": _py(v), "ic_inf": _py(ics.get(k, (None, None))[0]), "ic_sup": _py(ics.get(k, (None, None))[1]),
              "fold": fold, "estrato": estrato} for k, v in metricas.items()]
    return _insertar_lotes(session, Metrica, filas)


def guardar_explicacion(session: Session, modelo_id: int, tipo: str, contribuciones: dict | list,
                        hogar_uid: str | None = None) -> int:
    obj = Explicacion(modelo_id=modelo_id, hogar_uid=hogar_uid, tipo=tipo, contribuciones=contribuciones)
    session.add(obj)
    session.flush()
    return obj.explicacion_id


def registrar_auditoria(session: Session, usuario: str, accion: str, entidad: str | None = None,
                        entidad_id: str | None = None, detalle: dict | None = None) -> int:
    obj = Auditoria(usuario=usuario, accion=accion, entidad=entidad, entidad_id=entidad_id, detalle=detalle)
    session.add(obj)
    session.flush()
    return obj.auditoria_id
