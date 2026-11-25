"""models.py - Fábrica de los cinco modelos M1-M5 y utilidades de ajuste sin fuga.

M1 Regresión logística multinomial regularizada | M2 Random Forest | M3 CatBoost
M4 Stacking (CatBoost + RF + LR) con CV agrupada | M5 Autoencoder (PyTorch) + XGBoost.
M6 (EXTRA opcional, Fase 2 V2) HistGradientBoosting de scikit-learn con parada temprana nativa y pesos de clase/muestra.

Las librerías pesadas (catboost, xgboost, torch, imblearn) se importan de forma
perezosa para que M1/M2 funcionen aunque no estén instaladas.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from fsai.config import INNER_FOLDS, N_HILOS, SEED
from fsai.preprocessing import construir_preprocesador

logger = logging.getLogger(__name__)


def _sklearn_ge(mayor: int, menor: int) -> bool:
    """True si la versión instalada de scikit-learn es >= mayor.menor."""
    partes = sklearn.__version__.split(".")
    return (int(partes[0]), int(partes[1])) >= (mayor, menor)


def alinear_proba(proba: np.ndarray, clases_modelo: np.ndarray, n_clases: int = 4) -> np.ndarray:
    """Devuelve probabilidades con n_clases columnas (ceros para clases ausentes en train)."""
    clases_modelo = np.asarray(clases_modelo).astype(int)
    if proba.shape[1] == n_clases and np.array_equal(clases_modelo, np.arange(n_clases)):
        return proba
    out = np.zeros((proba.shape[0], n_clases), dtype=float)
    out[:, clases_modelo] = proba
    return out


# --------------------------------------------------------------------------
# M1 - Regresión logística multinomial
# --------------------------------------------------------------------------
def _crear_regresion_logistica(C: float = 1.0, penalizacion: str = "l2", class_weight: str | None = "balanced",
                               max_iter: int = 3000) -> LogisticRegression:
    """Logística multinomial regularizada. Compatible con scikit-learn < 1.8 y >= 1.8."""
    if penalizacion == "l2":
        return LogisticRegression(C=C, solver="lbfgs", class_weight=class_weight,
                                  max_iter=max_iter, random_state=SEED)
    if penalizacion == "l1":
        if _sklearn_ge(1, 8):   # 'penalty' quedó obsoleto: l1_ratio=1 equivale a L1
            return LogisticRegression(C=C, l1_ratio=1.0, solver="saga", class_weight=class_weight,
                                      max_iter=max_iter, random_state=SEED)
        return LogisticRegression(C=C, penalty="l1", solver="saga", class_weight=class_weight,
                                  max_iter=max_iter, random_state=SEED)
    raise ValueError(f"penalizacion '{penalizacion}' inválida (use 'l1' o 'l2')")


# --------------------------------------------------------------------------
# M2 / M3
# --------------------------------------------------------------------------
def _crear_random_forest(n_estimators: int = 300, max_depth: int | None = 12,
                         min_samples_leaf: int = 5) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf, class_weight="balanced_subsample",
                                  n_jobs=N_HILOS, random_state=SEED)


def _crear_catboost(depth: int = 6, learning_rate: float = 0.05, iterations: int = 500):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("M3/M4 requieren 'catboost' (pip install catboost).") from exc
    return CatBoostClassifier(loss_function="MultiClass", depth=depth, learning_rate=learning_rate,
                              iterations=iterations, auto_class_weights="Balanced",
                              random_seed=SEED, verbose=0, allow_writing_files=False, thread_count=N_HILOS)


# --------------------------------------------------------------------------
# M6 - HistGradientBoosting (extra opcional)
# --------------------------------------------------------------------------
def _crear_hist_gradient_boosting(learning_rate: float = 0.08, max_iter: int = 300, max_depth: int | None = 6,
                                  max_leaf_nodes: int = 31, min_samples_leaf: int = 40, l2_regularization: float = 1.0,
                                  class_weight: str | None = "balanced", n_iter_no_change: int = 15,
                                  validation_fraction: float = 0.1) -> HistGradientBoostingClassifier:
    """HGB multiclase con parada temprana NATIVA (``early_stopping=True``: reserva ``validation_fraction`` del train de cada ajuste).

    Admite ``sample_weight`` en ``fit`` y ``class_weight`` ('balanced' | None; se combinan de forma multiplicativa). Recibe la
    misma matriz preprocesada (one-hot) que M1-M5, de modo que la comparación es justa; NO se activa el soporte categórico
    nativo. LightGBM no se usa (no está instalado en el entorno de referencia y no se instala nada).
    """
    return HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=learning_rate, max_iter=max_iter, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization, class_weight=class_weight,
        early_stopping=True, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change,
        random_state=SEED)


# --------------------------------------------------------------------------
# M4 - Stacking con validación cruzada agrupada
# --------------------------------------------------------------------------
class StackingAgrupado(BaseEstimator, ClassifierMixin):
    """StackingClassifier(stack_method='predict_proba') cuya CV interna es AGRUPADA.

    Las probabilidades fuera de muestra que alimentan al metaaprendiz se generan con
    StratifiedGroupKFold sobre ``grupos`` (hogar_uid), de modo que ningún grupo aparece
    a la vez en el ajuste y en la predicción de un mismo nivel-0. ``grupos`` es
    obligatorio en ``fit``: se rechaza una CV no agrupada silenciosa.
    """

    def __init__(self, cb_depth: int = 6, cb_learning_rate: float = 0.05, cb_iterations: int = 300,
                 rf_n_estimators: int = 300, rf_max_depth: int | None = 12, rf_min_samples_leaf: int = 5,
                 lr_C: float = 1.0, C_meta: float = 1.0, n_splits: int = INNER_FOLDS,
                 random_state: int = SEED):
        self.cb_depth = cb_depth
        self.cb_learning_rate = cb_learning_rate
        self.cb_iterations = cb_iterations
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.lr_C = lr_C
        self.C_meta = C_meta
        self.n_splits = n_splits
        self.random_state = random_state

    def _estimadores_base(self) -> list[tuple[str, Any]]:
        return [
            ("catboost", _crear_catboost(self.cb_depth, self.cb_learning_rate, self.cb_iterations)),
            ("rf", _crear_random_forest(self.rf_n_estimators, self.rf_max_depth, self.rf_min_samples_leaf)),
            ("lr", _crear_regresion_logistica(self.lr_C, "l2")),
        ]

    def fit(self, X, y, sample_weight=None, grupos=None):
        if grupos is None:
            raise ValueError("StackingAgrupado.fit requiere 'grupos' (hogar_uid) para la CV agrupada.")
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        grupos = np.asarray(grupos)
        if len(grupos) != len(y):
            raise ValueError("'grupos' debe tener la misma longitud que y.")
        cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        particiones = list(cv.split(X, y, grupos))
        meta = LogisticRegression(C=self.C_meta, class_weight="balanced", max_iter=3000)
        self.modelo_ = StackingClassifier(estimators=self._estimadores_base(), final_estimator=meta,
                                          stack_method="predict_proba", cv=particiones,
                                          passthrough=False, n_jobs=1)
        if sample_weight is None:
            self.modelo_.fit(X, y)
        else:
            self.modelo_.fit(X, y, sample_weight=np.asarray(sample_weight, dtype=float))
        self.classes_ = self.modelo_.classes_
        return self

    def predict_proba(self, X):
        return self.modelo_.predict_proba(np.asarray(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# --------------------------------------------------------------------------
# M5 - Autoencoder (PyTorch) + XGBoost
# --------------------------------------------------------------------------
_AE_CLASE = None


def _clase_autoencoder():
    """Clase ``AutoencoderTorch`` a nivel de módulo (creada de forma perezosa para no exigir torch al importar).

    Debe ser accesible como ``fsai.models.AutoencoderTorch`` para que ``joblib``/``pickle`` puedan guardar y cargar
    los pipelines de M5 (una clase local dentro de una función no es serializable).
    """
    global _AE_CLASE
    if _AE_CLASE is None:
        import torch.nn as nn

        class AutoencoderTorch(nn.Module):
            def __init__(self, d: int, capas: tuple[int, ...], dim_latente: int, dropout: float) -> None:
                super().__init__()
                enc: list[nn.Module] = []
                previo = d
                for h in capas:
                    enc += [nn.Linear(previo, h), nn.ReLU(), nn.Dropout(dropout)]
                    previo = h
                enc.append(nn.Linear(previo, dim_latente))          # código latente (lineal)
                dec: list[nn.Module] = []
                previo = dim_latente
                for h in reversed(capas):
                    dec += [nn.Linear(previo, h), nn.ReLU(), nn.Dropout(dropout)]
                    previo = h
                dec.append(nn.Linear(previo, d))                    # reconstrucción (lineal)
                self.encoder = nn.Sequential(*enc)
                self.decoder = nn.Sequential(*dec)

            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z), z

        AutoencoderTorch.__module__ = __name__
        AutoencoderTorch.__qualname__ = "AutoencoderTorch"
        _AE_CLASE = AutoencoderTorch
    return _AE_CLASE


def __getattr__(nombre: str):
    """Permite a pickle resolver ``fsai.models.AutoencoderTorch`` al cargar un pipeline de M5."""
    if nombre == "AutoencoderTorch":
        return _clase_autoencoder()
    raise AttributeError(f"module {__name__!r} has no attribute {nombre!r}")


def _construir_autoencoder(d: int, capas: tuple[int, ...], dim_latente: int, dropout: float):
    """Autoencoder simétrico d -> capas -> dim_latente -> capas invertidas -> d."""
    return _clase_autoencoder()(d, tuple(capas), dim_latente, dropout)


class AutoencoderXGB(BaseEstimator, ClassifierMixin):
    """Autoencoder no supervisado + XGBoost multi:softprob (compatible con scikit-learn).

    ``modo_entrada``: 'x_z' alimenta XGBoost con [x_escalado || z]; 'z' solo con el latente.
    Se asume que ``X`` ya fue imputado y escalado por el preprocesador ajustado en train.
    El autoencoder se entrena EXCLUSIVAMENTE con las filas de ``fit`` (train); una fracción
    interna de ese train se reserva para el criterio de parada temprana.
    Con d -> 128 -> 64 -> 16, dropout 0,2 y decodificador simétrico por defecto.
    """

    def __init__(self, dim_latente: int = 16, capas: tuple[int, ...] = (128, 64), dropout: float = 0.2,
                 lr_ae: float = 1e-3, epocas_ae: int = 60, batch_size: int = 256, paciencia: int = 8,
                 fraccion_val_ae: float = 0.1, modo_entrada: str = "x_z", n_estimators: int = 300,
                 max_depth: int = 6, learning_rate: float = 0.1, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, reg_lambda: float = 1.0, n_clases: int = 4,
                 random_state: int = SEED, dispositivo: str = "cpu"):
        self.dim_latente = dim_latente
        self.capas = capas
        self.dropout = dropout
        self.lr_ae = lr_ae
        self.epocas_ae = epocas_ae
        self.batch_size = batch_size
        self.paciencia = paciencia
        self.fraccion_val_ae = fraccion_val_ae
        self.modo_entrada = modo_entrada
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.n_clases = n_clases
        self.random_state = random_state
        self.dispositivo = dispositivo

    # -- componentes -------------------------------------------------------
    def _crear_xgb(self):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("M5 requiere 'xgboost' (pip install xgboost).") from exc
        return XGBClassifier(objective="multi:softprob", num_class=len(self.clases_presentes_),
                             n_estimators=self.n_estimators, max_depth=self.max_depth,
                             learning_rate=self.learning_rate, subsample=self.subsample,
                             colsample_bytree=self.colsample_bytree, reg_lambda=self.reg_lambda,
                             tree_method="hist", eval_metric="mlogloss",
                             random_state=self.random_state, n_jobs=N_HILOS)

    def _entrenar_autoencoder(self, X: np.ndarray, w: np.ndarray) -> None:
        import torch
        if N_HILOS > 0:
            torch.set_num_threads(N_HILOS)
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        dev = torch.device(self.dispositivo)
        self.ae_ = _construir_autoencoder(X.shape[1], tuple(self.capas), self.dim_latente, self.dropout).to(dev)
        idx = rng.permutation(len(X))
        n_val = max(int(len(X) * self.fraccion_val_ae), 1)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        Xt = torch.tensor(X, dtype=torch.float32, device=dev)
        Wt = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
        opt = torch.optim.Adam(self.ae_.parameters(), lr=self.lr_ae)
        mejor, mejor_estado, sin_mejora = np.inf, None, 0
        self.historial_ae_: list[tuple[float, float]] = []
        for _ in range(self.epocas_ae):
            self.ae_.train()
            perm = rng.permutation(tr_idx)
            perdida_ep = 0.0
            for i in range(0, len(perm), self.batch_size):
                b = torch.as_tensor(perm[i:i + self.batch_size], device=dev)
                rec, _z = self.ae_(Xt[b])
                perdida = (((rec - Xt[b]) ** 2).mean(dim=1) * Wt[b]).mean()
                opt.zero_grad()
                perdida.backward()
                opt.step()
                perdida_ep += float(perdida) * len(b)
            self.ae_.eval()
            with torch.no_grad():
                bv = torch.as_tensor(val_idx, device=dev)
                rec_v, _ = self.ae_(Xt[bv])
                perdida_v = float((((rec_v - Xt[bv]) ** 2).mean(dim=1) * Wt[bv]).mean())
            self.historial_ae_.append((perdida_ep / max(len(tr_idx), 1), perdida_v))
            if perdida_v < mejor - 1e-6:
                mejor, sin_mejora = perdida_v, 0
                mejor_estado = {k: v.detach().clone() for k, v in self.ae_.state_dict().items()}
            else:
                sin_mejora += 1
                if sin_mejora >= self.paciencia:
                    break
        if mejor_estado is not None:
            self.ae_.load_state_dict(mejor_estado)
        self.ae_.eval()

    # -- API pública -------------------------------------------------------
    def extraer_latente(self, X) -> np.ndarray:
        """Código latente z (n x dim_latente) con el autoencoder ya entrenado en train."""
        import torch
        if not hasattr(self, "ae_"):
            raise RuntimeError("Llame a fit() antes de extraer_latente().")
        dev = next(self.ae_.parameters()).device
        with torch.no_grad():
            z = self.ae_.encoder(torch.tensor(np.asarray(X, dtype=np.float32), device=dev))
        return z.cpu().numpy()

    def transformar_entrada(self, X) -> np.ndarray:
        """Matriz que recibe XGBoost: [x || z] o z, según ``modo_entrada``."""
        X = np.asarray(X, dtype=np.float32)
        z = self.extraer_latente(X)
        if self.modo_entrada == "x_z":
            return np.hstack([X, z])
        if self.modo_entrada == "z":
            return z
        raise ValueError("modo_entrada debe ser 'x_z' o 'z'")

    def nombres_entrada(self, nombres_x: list[str]) -> list[str]:
        nombres_z = [f"z_{i}" for i in range(self.dim_latente)]
        return list(nombres_x) + nombres_z if self.modo_entrada == "x_z" else nombres_z

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)
        if np.isnan(X).any():
            raise ValueError("AutoencoderXGB requiere X sin NaN (use el preprocesador).")
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.arange(self.n_clases)
        self.clases_presentes_ = np.unique(y)                       # clases vistas en train
        mapa = {c: i for i, c in enumerate(self.clases_presentes_)}
        y_cod = np.array([mapa[c] for c in y])
        self._entrenar_autoencoder(X, w)
        self.xgb_ = self._crear_xgb()
        self.xgb_.fit(self.transformar_entrada(X), y_cod, sample_weight=w)
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = self.xgb_.predict_proba(self.transformar_entrada(X))
        return alinear_proba(p, self.clases_presentes_, self.n_clases)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# --------------------------------------------------------------------------
# Fábrica
# --------------------------------------------------------------------------
def _prefijo(params: dict, prefijo: str) -> dict:
    return {k[len(prefijo):]: v for k, v in params.items() if k.startswith(prefijo)}


def crear_modelo(codigo: str, params: dict, n_clases: int = 4):
    """Instancia el clasificador 'M1'..'M6' (sin preprocesado; M6 = extra opcional) con los hiperparámetros dados."""
    p = dict(params or {})
    if codigo == "M1":
        cw = p.get("class_weight", "balanced")
        return _crear_regresion_logistica(p.get("C", 1.0), p.get("penalty", "l2"),
                                          None if cw in (None, "ninguno") else cw)
    if codigo == "M2":
        return _crear_random_forest(p.get("n_estimators", 300), p.get("max_depth", 12), p.get("min_samples_leaf", 5))
    if codigo == "M3":
        return _crear_catboost(p.get("depth", 6), p.get("learning_rate", 0.05), p.get("iterations", 500))
    if codigo == "M4":
        return StackingAgrupado(**{k: v for k, v in p.items()
                                   if k in StackingAgrupado().get_params()})
    if codigo == "M5":
        if "capas" in p:
            p["capas"] = tuple(p["capas"])
        return AutoencoderXGB(n_clases=n_clases, **{k: v for k, v in p.items()
                                                    if k in AutoencoderXGB().get_params() and k != "n_clases"})
    if codigo == "M6":
        cw = p.get("class_weight", "balanced")
        return _crear_hist_gradient_boosting(p.get("learning_rate", 0.08), p.get("max_iter", 300), p.get("max_depth", 6),
                                             p.get("max_leaf_nodes", 31), p.get("min_samples_leaf", 40),
                                             p.get("l2_regularization", 1.0), None if cw in (None, "ninguno") else cw)
    raise ValueError(f"Código de modelo desconocido: {codigo} (use M1..M6; M6 es extra opcional)")


def construir_pipeline(codigo: str, params: dict, features_num: list[str], features_cat: list[str],
                       usar_smote: bool = False, n_clases: int = 4):
    """Pipeline preprocesador -> [SMOTE] -> clasificador. Todo se ajusta solo con train.

    SMOTE (si se activa) vive DENTRO del pipeline de imblearn: solo remuestrea la
    partición de entrenamiento de cada llamada a fit y nunca se aplica en predict.
    SMOTE no admite sample_weight: es excluyente con la ponderación muestral.
    """
    pre = construir_preprocesador(features_num, features_cat)
    clf = crear_modelo(codigo, params, n_clases)
    if not usar_smote:
        return Pipeline([("pre", pre), ("clf", clf)])
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
    except ImportError as exc:
        raise ImportError("usar_smote=True requiere 'imbalanced-learn'.") from exc
    return ImbPipeline([("pre", pre), ("smote", SMOTE(k_neighbors=3, random_state=SEED)), ("clf", clf)])


def ajustar_pipeline(pipe, X: pd.DataFrame, y, pesos=None, grupos=None):
    """Ajusta el pipeline pasando pesos/grupos solo al clasificador final."""
    fit_params: dict[str, Any] = {}
    if pesos is not None:
        if "smote" in pipe.named_steps:
            raise ValueError("SMOTE y sample_weight son excluyentes: desactive uno de los dos.")
        fit_params["clf__sample_weight"] = np.asarray(pesos, dtype=float)
    if isinstance(pipe.named_steps["clf"], StackingAgrupado):
        fit_params["clf__grupos"] = np.asarray(grupos)
    pipe.fit(X, np.asarray(y).astype(int), **fit_params)
    return pipe


def predecir_proba(pipe, X: pd.DataFrame, n_clases: int = 4) -> np.ndarray:
    """Probabilidades con n_clases columnas, alineadas a CLASES = [0,1,2,3]."""
    return alinear_proba(pipe.predict_proba(X), pipe.named_steps["clf"].classes_, n_clases)


def clonar_pipeline(pipe):
    """Copia sin ajustar (misma configuración)."""
    return clone(pipe)
