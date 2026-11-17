import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.data_prep import create_attributes

# ---------------------------------------------------------------------------
# Umbrales de alerta
# ---------------------------------------------------------------------------
UMBRAL_PSI = 0.2               # PSI > 0.2: cambio relevante en la distribución
UMBRAL_DIF_CLASES = 0.10       # variación de más de 10 puntos en la proporción de una clase
UMBRAL_ACCURACY = 0.80         # objetivo de negocio
UMBRAL_JS_TEXTO = 0.06         # divergencia Jensen-Shannon de las palabras (ruido de muestreo ≈ 0.04)
UMBRAL_DIF_OOV = 0.05          # aumento de más de 5 puntos en palabras fuera del vocabulario del modelo

VARIABLES_NUMERICAS = ["tiempo_respuesta_horas", "importe_compra"]


def calcular_psi(referencia: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index: mide cuánto cambió la distribución de una variable
    numérica respecto de la de referencia. Valores habituales de interpretación:
    < 0.1 sin cambios, 0.1 - 0.2 cambio moderado, > 0.2 cambio significativo."""
    # Los cortes se definen con los deciles de los datos de referencia
    cortes = np.unique(np.quantile(referencia, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    pct_ref = np.histogram(referencia, cortes)[0] / len(referencia)
    pct_act = np.histogram(actual, cortes)[0] / len(actual)
    # Se evita la división por cero en intervalos vacíos
    pct_ref = np.clip(pct_ref, 1e-6, None)
    pct_act = np.clip(pct_act, 1e-6, None)
    return float(np.sum((pct_act - pct_ref) * np.log(pct_act / pct_ref)))


def distribucion_palabras(textos: pd.Series, analizador) -> Counter:
    """Cuenta la frecuencia de cada palabra en un conjunto de textos, tokenizando con el
    mismo analizador del TF-IDF del modelo para que coincida con el entrenamiento."""
    return Counter(token for texto in textos for token in analizador(texto))


def divergencia_js(dist_ref: Counter, dist_act: Counter) -> float:
    """Divergencia de Jensen-Shannon (base 2) entre dos distribuciones de palabras.
    Vale 0 si son idénticas y 1 si no comparten ninguna palabra."""
    terminos = sorted(set(dist_ref) | set(dist_act))
    p = np.array([dist_ref.get(t, 0) for t in terminos], dtype=float)
    q = np.array([dist_act.get(t, 0) for t in terminos], dtype=float)
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2))


def proporcion_oov(distribucion: Counter, vocabulario: set) -> float:
    """Proporción de palabras (ocurrencias) que no forman parte del vocabulario del modelo."""
    total = sum(distribucion.values())
    fuera = sum(n for token, n in distribucion.items() if token not in vocabulario)
    return fuera / total if total else 0.0


def monitor(
    lote_path: str = "data/raw/resenas_nuevas.csv",
    referencia_path: str = "data/processed/datos_integrados.csv",
    model_path: str = "models/trained_model.pkl",
    output_path: str = "metrics/monitoring.json",
) -> dict:
    # ---------------------------------------------------------------------------
    # Lectura del modelo, los datos de referencia (entrenamiento) y el lote nuevo
    # ---------------------------------------------------------------------------
    pipeline = joblib.load(model_path)
    referencia = pd.read_csv(referencia_path, sep=";")
    lote = pd.read_csv(lote_path, sep=";")

    # ---------------------------------------------------------------------------
    # Predicción sobre el lote, con los mismos atributos derivados que en el entrenamiento
    # ---------------------------------------------------------------------------
    # El lote no identifica clientes: cada reseña se trata como cliente nuevo (historial vacío),
    # igual que en la API cuando no se indica id_cliente
    historial_vacio = pd.DataFrame(columns=["resenas_previas", "reclamos_previos"])
    X_lote = create_attributes(lote.drop(columns=["sentimiento"]), historial=historial_vacio)
    X_lote = X_lote[pipeline.feature_names_in_]
    predicciones = pd.Series(pipeline.predict(X_lote))

    metricas = {"cantidad_resenas": len(lote)}
    alertas = []

    # ---------------------------------------------------------------------------
    # 1. Drift en los datos de entrada (variables numéricas)
    # ---------------------------------------------------------------------------
    for col in VARIABLES_NUMERICAS:
        psi = calcular_psi(referencia[col], lote[col])
        metricas[f"psi_{col}"] = round(psi, 4)
        if psi > UMBRAL_PSI:
            alertas.append(f"Drift en '{col}': PSI = {psi:.3f} (umbral {UMBRAL_PSI})")

    # ---------------------------------------------------------------------------
    # 2. Distribución de las clases predichas frente a la del entrenamiento
    # ---------------------------------------------------------------------------
    dist_ref = referencia["sentimiento"].value_counts(normalize=True)
    dist_pred = predicciones.value_counts(normalize=True)
    for clase in dist_ref.index:
        dif = dist_pred.get(clase, 0) - dist_ref[clase]
        metricas[f"proporcion_predicha_{clase.lower()}"] = round(float(dist_pred.get(clase, 0)), 4)
        if abs(dif) > UMBRAL_DIF_CLASES:
            alertas.append(f"Cambio en la proporción de '{clase}': {dist_ref[clase]:.1%} -> {dist_pred.get(clase, 0):.1%}")

    # ---------------------------------------------------------------------------
    # 3. Desempeño del modelo (el lote incluye el sentimiento real)
    # ---------------------------------------------------------------------------
    y_real = lote["sentimiento"]
    metricas["accuracy"] = round(accuracy_score(y_real, predicciones), 4)
    metricas["f1_macro"] = round(f1_score(y_real, predicciones, average="macro"), 4)
    metricas["recall_negativo"] = round(recall_score(y_real, predicciones, labels=["Negativo"], average="macro"), 4)
    if metricas["accuracy"] < UMBRAL_ACCURACY:
        alertas.append(f"Accuracy por debajo del objetivo: {metricas['accuracy']:.3f} < {UMBRAL_ACCURACY}")

    # ---------------------------------------------------------------------------
    # 4. Drift en el texto de las reseñas (señal principal del modelo)
    # ---------------------------------------------------------------------------
    tfidf = pipeline.named_steps["prep"].named_transformers_["text"]
    analizador = tfidf.build_analyzer()
    vocabulario = set(tfidf.vocabulary_)

    dist_ref = distribucion_palabras(referencia["texto_resena"], analizador)
    dist_lote = distribucion_palabras(lote["texto_resena"], analizador)

    js_texto = divergencia_js(dist_ref, dist_lote)
    metricas["js_texto"] = round(js_texto, 4)
    if js_texto > UMBRAL_JS_TEXTO:
        alertas.append(f"Drift en el texto de las reseñas: divergencia JS = {js_texto:.3f} (umbral {UMBRAL_JS_TEXTO})")

    oov_ref = proporcion_oov(dist_ref, vocabulario)
    oov_lote = proporcion_oov(dist_lote, vocabulario)
    metricas["proporcion_oov_referencia"] = round(oov_ref, 4)
    metricas["proporcion_oov_lote"] = round(oov_lote, 4)
    if oov_lote - oov_ref > UMBRAL_DIF_OOV:
        alertas.append(f"Aumento de palabras fuera del vocabulario del modelo: {oov_ref:.1%} -> {oov_lote:.1%}")

    psi_longitud = calcular_psi(referencia["longitud_resena"], X_lote["longitud_resena"])
    metricas["psi_longitud_resena"] = round(psi_longitud, 4)
    if psi_longitud > UMBRAL_PSI:
        alertas.append(f"Drift en la longitud de las reseñas: PSI = {psi_longitud:.3f} (umbral {UMBRAL_PSI})")

    # ---------------------------------------------------------------------------
    # Reporte
    # ---------------------------------------------------------------------------
    reporte = {"lote": str(lote_path), "metricas": metricas, "alertas": alertas}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    print(f"Lote analizado: {lote_path} ({len(lote)} reseñas)")
    for k, v in metricas.items():
        print(f"  {k}: {v}")
    if alertas:
        print("ALERTAS:")
        for alerta in alertas:
            print(f"  - {alerta}")
        print("Se recomienda revisar los datos y evaluar el reentrenamiento del modelo.")
    else:
        print("Sin alertas: el modelo se comporta según lo esperado.")

    return reporte


if __name__ == "__main__":
    monitor(*sys.argv[1:2])