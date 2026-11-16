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
    RocCurveDisplay,
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


REPO_ROOT = Path(__file__).resolve().parents[1]

# Modelo e hiperparámetros seleccionados en notebooks/MD01_Experimentacion.ipynb
SEED = 42
TARGET = "sentimiento"
# calificacion se excluye por fuga de información hacia el sentimiento; ids y fecha no generalizan
COLUMNAS_EXCLUIDAS = ["calificacion", "id_resena", "id_cliente", "fecha_resena"]
TEXT_COL = "texto_resena"
TFIDF_MAX_FEATURES = 300
HIPERPARAMETROS_MODELO = {
    "C": 1,
    "class_weight": "balanced",
    "max_iter": 2000,
    "random_state": SEED,
}


def _resolver(ruta, base):
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else base / ruta


def load_data(data_path: str = "data/processed/datos_integrados.csv"):
    """
    Carga los datos desde un archivo CSV y prepara las matrices de características y etiquetas.

    Args:
        data_path (str): Ruta al archivo CSV que contiene los datos procesados.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, features_X, labels_y) con los conjuntos de entrenamiento, validación y prueba, así como las características y etiquetas originales.
    """
    # Lectura de los datos
    df = pd.read_csv(_resolver(data_path, REPO_ROOT), sep=";")

    features_X = df.drop(columns=[TARGET] + COLUMNAS_EXCLUIDAS)
    labels_y = df[TARGET]

    # Primero se separa el test final (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        features_X, labels_y, test_size=0.10, random_state=SEED, stratify=labels_y
    )

    # Luego se separan train y validation (22% del 90% es aprox. el 20% del total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.22, random_state=SEED, stratify=y_temp
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
    categorical_features = [
        col
        for col in features_X.select_dtypes(include=["object", "string"]).columns
        if col != TEXT_COL
    ]

    # Definición de transformadores para cada tipo de característica
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    text_transformer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)

    # Creación del preprocesador utilizando ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("text", text_transformer, TEXT_COL),
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
    Entrena un modelo de clasificación utilizando los datos procesados y guarda el modelo, el preprocesador, las figuras y las métricas.

    Args:
        data_path (str): Ruta al archivo CSV que contiene los datos procesados.
        model_output_path (str): Ruta donde se guardará el modelo entrenado.
        preprocessor_output_path (str): Ruta donde se guardará el preprocesador.
        metrics_output_path (str): Ruta donde se guardarán las métricas de entrenamiento.
        figures_output_path (str): Ruta donde se guardarán las figuras generadas durante el entrenamiento.

    Returns:
        tuple: (pipeline, preprocesador, metrics) con el pipeline entrenado, el preprocesador ajustado y el diccionario de métricas.
    """
    # Configuración de las variables de entorno para MLFlow
    # Recordar que debe existir un archivo .env en la raíz del proyecto con las variables de entorno necesarias
    # Las credenciales de MLFlow se pueden obtener desde la interfaz web de DagsHub en la sección de "Settings" y Tokens
    load_dotenv(find_dotenv())
    mlflow.set_experiment(
        "Proyecto 13MBID - V2 - Entrenamiento de modelos de clasificación de sentimientos"
    )

    # Resolución de rutas relativas respecto a la raíz del repositorio
    data_path = _resolver(data_path, REPO_ROOT)
    model_output_path = _resolver(model_output_path, REPO_ROOT)
    preprocessor_output_path = _resolver(preprocessor_output_path, REPO_ROOT)
    metrics_output_path = _resolver(metrics_output_path, REPO_ROOT)
    figures_output_path = _resolver(figures_output_path, REPO_ROOT)

    # Carga de datos
    X_train, X_val, X_test, y_train, y_val, y_test, features_X, labels_y = load_data(
        data_path
    )

    # Creación del preprocesador
    preprocessor = create_preprocessor(features_X)

    # Definición del modelo a entrenar
    model = LogisticRegression(**HIPERPARAMETROS_MODELO)

    # Creación del pipeline que incluye el preprocesador y el modelo
    pipeline = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    # Entrenamiento del modelo
    pipeline.fit(X_train, y_train)

    # Evaluación del modelo en el conjunto de test
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)

    # Cálculo de métricas de evaluación
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "test_precision_macro": float(
            precision_score(y_test, y_test_pred, average="macro", zero_division=0)
        ),
        "test_recall_macro": float(
            recall_score(y_test, y_test_pred, average="macro", zero_division=0)
        ),
        "test_f1_macro": float(
            f1_score(y_test, y_test_pred, average="macro", zero_division=0)
        ),
        # Recall de la clase Negativo (clientes insatisfechos)
        "test_recall_negativo": float(
            recall_score(
                y_test,
                y_test_pred,
                labels=["Negativo"],
                average="macro",
                zero_division=0,
            )
        ),
        "test_roc_auc_ovr": float(
            roc_auc_score(
                y_test,
                y_test_proba,
                multi_class="ovr",
                average="macro",
                labels=pipeline.classes_,
            )
        ),
    }

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # Guardado de la matriz de confusión
    Path(figures_output_path).mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_test_pred, labels=pipeline.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title("Matriz de confusión - Regresión logística (test)")
    cm_fig_path = Path(figures_output_path) / "confusion_matrix_test.png"
    fig.savefig(cm_fig_path, bbox_inches="tight")
    plt.close(fig)

    # Curva ROC one-vs-rest de cada clase en el conjunto de test
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    for i, clase in enumerate(pipeline.classes_):
        RocCurveDisplay.from_predictions(
            (y_test == clase).astype(int),
            y_test_proba[:, i],
            name=f"{clase} vs resto",
            ax=ax_roc,
        )
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar (AUC = 0.5)")
    ax_roc.set_title("Curva ROC one-vs-rest - Regresión logística (test)")
    ax_roc.legend(loc="lower right")
    roc_fig_path = Path(figures_output_path) / "roc_curve_test.png"
    fig_roc.savefig(roc_fig_path, bbox_inches="tight")
    plt.close(fig_roc)

    # Registro general en MLFlow
    signature = infer_signature(X_train, pipeline.predict(X_train))

    with mlflow.start_run(run_name="Pipeline (prod) - Logistic Regression") as run:
        mlflow.log_param("modelo", "LogisticRegression")
        mlflow.log_param("seed", SEED)
        mlflow.log_params(HIPERPARAMETROS_MODELO)
        mlflow.log_params(
            {
                "train_sample_size": len(X_train),
                "validation_sample_size": len(X_val),
                "test_sample_size": len(X_test),
                "tfidf_max_features": TFIDF_MAX_FEATURES,
            }
        )

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(cm_fig_path), artifact_path="figures")
        mlflow.log_artifact(str(roc_fig_path), artifact_path="figures")
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
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return pipeline, pipeline.named_steps["prep"], metrics


if __name__ == "__main__":
    train_model()