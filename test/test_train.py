import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from train_model import SEED, load_data  # noqa: E402

RUTA_MODELO = REPO_ROOT / "models" / "trained_model.pkl"
RUTA_METRICAS = REPO_ROOT / "metrics" / "train_metrics.json"

METRICAS_ESPERADAS = {
    "test_accuracy",
    "test_precision_macro",
    "test_recall_macro",
    "test_f1_macro",
    "test_recall_negativo",
    "test_roc_auc_ovr",
}
# Objetivo de calidad definido por el negocio
UMBRAL_ACCURACY = 0.80
# Mejor modelo según f1-score en notebooks/MD01_Experimentacion.ipynb
MODELO_ESPERADO = "LogisticRegression"
HIPERPARAMETROS_ESPERADOS = {"C": 1, "class_weight": "balanced"}
COLUMNAS_PROHIBIDAS = ["calificacion", "id_resena", "id_cliente", "fecha_resena"]
CLASES = {"Negativo", "Neutral", "Positivo"}
TOLERANCIA = 1e-6


@pytest.fixture(scope="module")
def metricas():
    """Carga las métricas de entrenamiento desde el archivo JSON.

    Args:
        None

    Returns:
        dict: Diccionario con las métricas del modelo entrenado.
    """
    with open(RUTA_METRICAS, encoding="utf-8") as archivo:
        return json.load(archivo)


@pytest.fixture(scope="module")
def modelo():
    """Carga el modelo entrenado desde el archivo pickle.

    Args:
        None

    Returns:
        sklearn.pipeline.Pipeline: Pipeline con el modelo entrenado.
    """
    return joblib.load(RUTA_MODELO)


@pytest.fixture(scope="module")
def datos():
    """Carga y divide los datos en entrenamiento, validación y prueba.

    Args:
        None

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, features_X, labels_y).
    """
    return load_data()


@pytest.fixture(scope="module")
def predicciones(modelo, datos):
    """Genera predicciones y probabilidades sobre el conjunto de prueba.

    Args:
        modelo (sklearn.pipeline.Pipeline): Pipeline con el modelo entrenado.
        datos (tuple): Tupla con los datos divididos.

    Returns:
        tuple: (y_pred, y_proba) con las predicciones y probabilidades de prueba.
    """
    X_test = datos[2]
    return modelo.predict(X_test), modelo.predict_proba(X_test)


def test_metricas_generadas(metricas):
    """Valida que se hayan generado todas las métricas esperadas con valores válidos.

    Args:
        metricas (dict): Diccionario con las métricas del modelo entrenado.

    Returns:
        None
    """
    esperadas = METRICAS_ESPERADAS
    obtenidas = set(metricas)
    faltantes = esperadas - obtenidas
    sobrantes = obtenidas - esperadas
    assert not faltantes and not sobrantes, (
        f"Metricas incorrectas: faltantes={faltantes}, sobrantes={sobrantes}"
    )
    for nombre, valor in metricas.items():
        assert isinstance(valor, (int, float)), (
            f"La metrica '{nombre}' no es numerica: {valor!r}"
        )
        assert math.isfinite(valor), (
            f"La metrica '{nombre}' no es finita: {valor!r}"
        )
        assert 0 <= valor <= 1, (
            f"La metrica '{nombre}' esta fuera de [0, 1]: {valor!r}"
        )


def test_metricas_coinciden(metricas, modelo, datos, predicciones):
    """Recalcula las metricas y compara con las almacenadas.

    Args:
        metricas (dict): Diccionario con las metricas del modelo entrenado.
        modelo (sklearn.pipeline.Pipeline): Pipeline con el modelo entrenado.
        datos (tuple): Tupla con los datos divididos.
        predicciones (tuple): Tupla con (y_pred, y_proba) del conjunto de prueba.

    Returns:
        None
    """
    y_test = datos[5]
    y_pred, y_proba = predicciones
    recalculadas = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "test_recall_macro": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "test_f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "test_recall_negativo": recall_score(
            y_test,
            y_pred,
            labels=["Negativo"],
            average="macro",
            zero_division=0,
        ),
        "test_roc_auc_ovr": roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro", labels=modelo.classes_
        ),
    }
    for nombre, valor_recalculado in recalculadas.items():
        valor_guardado = metricas[nombre]
        assert abs(valor_guardado - valor_recalculado) <= TOLERANCIA, (
            f"Metrica '{nombre}': guardada={valor_guardado}, recalculada={valor_recalculado}"
        )


def test_modelo_seleccionado(modelo):
    """Valida que el modelo seleccionado sea el esperado con los hiperparametros correctos.

    Args:
        modelo (sklearn.pipeline.Pipeline): Pipeline con el modelo entrenado.

    Returns:
        None
    """
    estimador = modelo.named_steps["model"]
    assert type(estimador).__name__ == MODELO_ESPERADO, (
        f"Modelo inesperado: {type(estimador).__name__}, esperado={MODELO_ESPERADO}"
    )
    for clave, valor_esperado in HIPERPARAMETROS_ESPERADOS.items():
        valor_obtenido = estimador.get_params()[clave]
        assert valor_obtenido == valor_esperado, (
            f"Hiperparametro '{clave}': obtenido={valor_obtenido}, esperado={valor_esperado}"
        )


def test_objetivo_calidad_negocio(metricas):
    """Valida que la accuracy cumpla el umbral de calidad del negocio.

    Args:
        metricas (dict): Diccionario con las metricas del modelo entrenado.

    Returns:
        None
    """
    accuracy = metricas["test_accuracy"]
    assert accuracy >= UMBRAL_ACCURACY, (
        f"Accuracy={accuracy} menor al umbral={UMBRAL_ACCURACY}"
    )


def test_mejor_que_linea_base(datos, predicciones):
    """Compara el modelo contra una línea base de clase más frecuente.

    Args:
        datos (tuple): Tupla con los datos divididos.
        predicciones (tuple): Tupla con (y_pred, y_proba) del conjunto de prueba.

    Returns:
        None
    """
    X_train, y_train = datos[0], datos[3]
    X_test, y_test = datos[2], datos[5]
    y_pred, _ = predicciones
    linea_base = DummyClassifier(strategy="most_frequent", random_state=SEED).fit(
        X_train, y_train
    )
    y_base = linea_base.predict(X_test)
    accuracy_modelo = accuracy_score(y_test, y_pred)
    accuracy_base = accuracy_score(y_test, y_base)
    assert accuracy_modelo > accuracy_base, (
        f"Accuracy modelo={accuracy_modelo} no supera a la linea base={accuracy_base}"
    )
    f1_modelo = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_base = f1_score(y_test, y_base, average="macro", zero_division=0)
    assert f1_modelo > f1_base, (
        f"F1 macro modelo={f1_modelo} no supera a la linea base={f1_base}"
    )


def test_sin_fuga_de_datos(modelo):
    """Valida que el modelo no use columnas prohibidas (fuga de informacion).

    Args:
        modelo (sklearn.pipeline.Pipeline): Pipeline con el modelo entrenado.

    Returns:
        None
    """
    columnas = set(modelo.feature_names_in_)
    prohibidas_encontradas = columnas & set(COLUMNAS_PROHIBIDAS)
    assert not prohibidas_encontradas, (
        f"Columnas prohibidas encontradas: {prohibidas_encontradas}"
    )


def test_modelo_usable(modelo, datos):
    """Valida que el modelo sea usable para predicciones sobre nuevos datos.

    Args:
        modelo (sklearn.pipeline.Pipeline): Pipeline con el modelo entrenado.
        datos (tuple): Tupla con los datos divididos.

    Returns:
        None
    """
    muestra = datos[2].head(20)
    predicciones_muestra = modelo.predict(muestra)
    assert set(predicciones_muestra) <= CLASES, (
        f"Clases predichas inesperadas: {set(predicciones_muestra) - CLASES}"
    )
    proba = modelo.predict_proba(muestra)
    assert proba.shape == (20, len(CLASES)), (
        f"Forma de probabilidades inesperada: {proba.shape}, esperada={(20, len(CLASES))}"
    )
    assert np.allclose(proba.sum(axis=1), 1.0), (
        "Las probabilidades por fila no suman 1"
    )
