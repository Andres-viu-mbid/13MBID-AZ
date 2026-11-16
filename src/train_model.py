# Se importan las librerías necesarias y se suprimen las advertencias
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_data(data_path: str = "data/processed/datos_integrados.csv"):
    """
    Carga los datos desde un archivo CSV y devuelve un DataFrame de pandas.

    Args:
        data_path (str): Ruta al archivo CSV que contiene los datos procesados.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, features_X, labels_y: Tupla que contiene los conjuntos de entrenamiento, validación y prueba, así como las características y etiquetas originales.
    """
    # Lectura de los datos
    df = pd.read_csv(data_path)

    target = "sentimiento"
    features_X = df.drop(columns=[target])
    labels_y = df[target]

    # Primero se separa el test final (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        features_X, labels_y, test_size=0.10, random_state=42, stratify=labels_y
    )

    # Luego se separan train y validation (22% del 90% es aprox. el 20% del total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.22, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, features_X, labels_y


def create_preprocessor(features_X: pd.DataFrame):
    """
    Crea un preprocesador para los datos de entrada.

    Args:
        features_X (pd.DataFrame): DataFrame que contiene las características de entrada.

    Returns:
        ColumnTransformer: Preprocesador que transforma las características de entrada.
    """
    # Identificación de columnas numéricas y categóricas
    numeric_features = features_X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_features = features_X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    # Definición de transformadores para cada tipo de característica
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    text_col = "texto_resena"
    text_transformer = TfidfVectorizer(max_features=300)

    # Creación del preprocesador utilizando ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("text", text_transformer, text_col),
        ]
    )

    return preprocessor


def train_model(
    data_path: str = "data/processed/datos_integrados.csv",
    model_output_path: str = "models/trained_model.pkl",
    preprocessor_output_path: str = "models/preprocessor.pkl",
    metrics_output_path: str = "metrics/train_metrics.json",
    figures_output_path: str = "docs/figures",
):
    """
    Entrena un modelo de clasificación utilizando los datos procesados y guarda el modelo, el preprocesador y las métricas.

    Args:
        data_path (str): Ruta al archivo CSV que contiene los datos procesados.
        model_output_path (str): Ruta donde se guardará el modelo entrenado.
        preprocessor_output_path (str): Ruta donde se guardará el preprocesador.
        metrics_output_path (str): Ruta donde se guardarán las métricas de entrenamiento.
        figures_output_path (str): Ruta donde se guardarán las figuras generadas durante el entrenamiento.
    """
    # Configuración de las variables de entorno para MLFlow
    # Recordar que debe existir un archivo .env en la raíz del proyecto con las variables de entorno necesarias
    # Las credenciales de MLFlow se pueden obtener desde la interfaz web de DagsHub en la sección de "Settings" y Tokens
    load_dotenv(find_dotenv())
    mlflow.set_experiment(
        "Proyecto 13MBID - V2 - Entrenamiento de modelos de clasificación de sentimientos"
    )

    # Carga de datos
    X_train, X_val, X_test, y_train, y_val, y_test, features_X, labels_y = load_data(
        data_path
    )

    # Creación del preprocesador
    preprocessor = create_preprocessor(features_X)

    # Definición del modelo a entrenar
    model = LogisticRegression(max_iter=1000)

    # Creación del pipeline que incluye el preprocesador y el modelo
    pipeline = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    # Entrenamiento del modelo
    pipeline.fit(X_train, y_train)

    # Evaluación del modelo en el conjunto de test
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)

    # Cálculo de métricas de evaluación
    metrics = {
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_precision_macro": precision_score(
            y_test, y_test_pred, average="macro", zero_division=0
        ),
        "test_recall_macro": recall_score(
            y_test, y_test_pred, average="macro", zero_division=0
        ),
        "test_f1_macro": f1_score(
            y_test, y_test_pred, average="macro", zero_division=0
        ),
        "test_recall_negative": recall_score(
            y_test, y_test_pred, pos_label="negativo", average="macro", zero_division=0
        ),
        "test_roc_aux_ovr": roc_auc_score(
            y_test, y_test_proba, multi_class="ovr", labels=pipeline.classes_
        ),
    }

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # Guardadado de la matriz de confusión
    Path(figures_output_path).mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_test_pred, labels=pipeline.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Logistic Regression - Test Set")
    cm_fig_path = f"{figures_output_path}/confusion_matrix_test.png"
    plt.savefig(cm_fig_path)
    plt.close()

    # Registro general en MLFlow
    signature = infer_signature(X_train, pipeline.predict(X_train))

    with mlflow.start_run(run_name="Pipeline (prod) - Logistic Regression") as run:
        mlflow.log_params(model.get_params())
        mlflow.log_params(
            {
                "train_sample_size": len(X_train),
                "validation_sample_size": len(X_val),
                "test_sample_size": len(X_test),
                "tfidf_max_features": 300,
            }
        )

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_fig_path, artifact_path="figures")
        mlflow.sklearn.log_model(
            pipeline, name="model", signature=signature, input_example=X_train.head(3)
        )

        run_id = mlflow.active_run().info.run_id
        print(f"Modelo registrado en MLFlow a raíz del run ID: {run_id}")

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(preprocessor_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)

    # Guardado del modelo entrenado y del preprocesador
    joblib.dump(pipeline, model_output_path)
    joblib.dump(pipeline.named_steps["prep"], preprocessor_output_path)

    # Guardado de métricas en un archivo JSON
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return pipeline, pipeline.named_steps["prep"], metrics


if __name__ == "__main__":
    train_model()