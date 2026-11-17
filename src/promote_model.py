import json
from pathlib import Path

import mlflow
from dotenv import find_dotenv, load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]

# Objetivo de calidad del negocio (el mismo que verifica test/test_train.py)
UMBRAL_ACCURACY = 0.80
ALIAS_PRODUCCION = "produccion"


def _resolver(ruta, base):
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else base / ruta


def promote_model(
    metrics_path: str = "metrics/train_metrics.json",
    registro_path: str = "metrics/modelo_registrado.json",
    output_path: str = "metrics/promocion.json",
) -> dict:
    """Promueve la versión registrada al alias de producción si cumple el objetivo de negocio.

    Args:
        metrics_path (str): Ruta al archivo JSON con las métricas de entrenamiento.
        registro_path (str): Ruta al archivo JSON con los datos del modelo registrado.
        output_path (str): Ruta donde se guardará el resultado de la promoción.

    Returns:
        dict: Resultado con el nombre del modelo, versión, test_accuracy, umbral_accuracy,
            promovido (booleano) y, si se promovió, el alias asignado; si no, el motivo.
    """
    load_dotenv(find_dotenv())
    metricas = json.loads(_resolver(metrics_path, REPO_ROOT).read_text(encoding="utf-8"))
    registro = json.loads(_resolver(registro_path, REPO_ROOT).read_text(encoding="utf-8"))

    accuracy = metricas["test_accuracy"]
    resultado = {
        "nombre": registro["nombre"],
        "version": registro["version"],
        "test_accuracy": accuracy,
        "umbral_accuracy": UMBRAL_ACCURACY,
    }

    if accuracy >= UMBRAL_ACCURACY:
        mlflow.MlflowClient().set_registered_model_alias(
            registro["nombre"], ALIAS_PRODUCCION, str(registro["version"])
        )
        resultado.update({"promovido": True, "alias": ALIAS_PRODUCCION})
        print(f"Versión {registro['version']} promovida al alias '{ALIAS_PRODUCCION}'")
    else:
        resultado.update({
            "promovido": False,
            "motivo": f"test_accuracy {accuracy:.4f} menor al umbral {UMBRAL_ACCURACY}",
        })
        print(f"Versión {registro['version']} NO promovida: {resultado['motivo']}")

    salida = _resolver(output_path, REPO_ROOT)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    return resultado


if __name__ == "__main__":
    promote_model()
