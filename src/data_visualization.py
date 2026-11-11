from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]

ORDEN_SENTIMIENTO = ["Negativo", "Neutral", "Positivo"]

COLORES_SENTIMIENTO = {
    "Negativo": "#c44e52",
    "Neutral": "#8c8c8c",
    "Positivo": "#55a868",
}


def _resolver(ruta, base):
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else base / ruta


def _guardar(fig, ruta):
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta


def _leyenda_sentimiento(fig):
    handles = [Patch(facecolor=COLORES_SENTIMIENTO[c], label=c) for c in ORDEN_SENTIMIENTO]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)


def _barras_apiladas(ax, tabla, titulo, etiqueta_x):
    tabla.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=[COLORES_SENTIMIENTO[c] for c in tabla.columns],
        legend=False,
    )
    for contenedor in ax.containers:
        etiquetas = [f"{valor:.1f}%" if valor >= 4 else "" for valor in contenedor.datavalues]
        ax.bar_label(contenedor, labels=etiquetas, label_type="center", fontsize=8, color="white")
    ax.set_title(titulo)
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel("")
    ax.set_xlim(0, 100)


def visualize_data(
    datos_cliente="data/raw/datos_clientes.csv",
    datos_resenas="data/raw/datos_resenas.csv",
    output="docs/figures/",
):
    """Genera las figuras exploratorias y devuelve las rutas de los PNG guardados."""
    df_clientes = pd.read_csv(_resolver(datos_cliente, REPO_ROOT), sep=";")
    df_resenas = pd.read_csv(_resolver(datos_resenas, REPO_ROOT), sep=";")

    directorio = _resolver(output, REPO_ROOT)
    directorio.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="muted")

    rutas = []

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        x="sentimiento",
        data=df_resenas,
        order=ORDEN_SENTIMIENTO,
        hue="sentimiento",
        palette=COLORES_SENTIMIENTO,
        legend=False,
        ax=ax,
    )
    ax.set_title("Distribución de la variable objetivo (sentimiento)")
    ax.set_xlabel("Sentimiento expresado en la reseña")
    ax.set_ylabel("Cantidad de reseñas")
    rutas.append(_guardar(fig, directorio / "distribucion_sentimiento.png"))

    df_longitud = df_resenas.assign(longitud_texto=df_resenas["texto_resena"].str.len())

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        x="sentimiento",
        y="longitud_texto",
        data=df_longitud,
        order=ORDEN_SENTIMIENTO,
        hue="sentimiento",
        palette=COLORES_SENTIMIENTO,
        legend=False,
        ax=ax,
    )
    ax.set_title("Longitud del texto de la reseña según sentimiento")
    ax.set_xlabel("Sentimiento")
    ax.set_ylabel("Longitud del texto (caracteres)")
    rutas.append(_guardar(fig, directorio / "longitud_resenas.png"))

    numericas_clientes = df_clientes.select_dtypes(include="number").drop(
        columns=["id_cliente"], errors="ignore"
    )
    numericas_resenas = df_resenas.select_dtypes(include="number").drop(
        columns=["id_cliente", "id_resena"], errors="ignore"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, datos, titulo in (
        (axes[0], numericas_clientes, "Clientes"),
        (axes[1], numericas_resenas, "Reseñas"),
    ):
        sns.heatmap(datos.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title(f"Matriz de correlaciones - {titulo}")
    fig.tight_layout()
    rutas.append(_guardar(fig, directorio / "correlaciones.png"))

    return rutas


def visualize_target_relationships(
    datos_resenas="data/raw/datos_resenas.csv",
    output="docs/figures/",
):
    """Grafica la relación entre el servicio postventa, el tiempo de respuesta y el sentimiento."""
    df_resenas = pd.read_csv(_resolver(datos_resenas, REPO_ROOT), sep=";")

    directorio = _resolver(output, REPO_ROOT)
    directorio.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    rutas = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, columna, titulo in (
        (axes[0], "problema_resuelto", "Problema resuelto"),
        (axes[1], "reclamo_formal", "Reclamo formal"),
    ):
        tabla = pd.crosstab(
            df_resenas[columna], df_resenas["sentimiento"], normalize="index"
        ).mul(100)[ORDEN_SENTIMIENTO]
        _barras_apiladas(ax, tabla, titulo, "Porcentaje de reseñas")
    fig.suptitle("Servicio postventa y sentimiento")
    _leyenda_sentimiento(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    rutas.append(_guardar(fig, directorio / "servicio_postventa.png"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(
        x="sentimiento",
        y="tiempo_respuesta_horas",
        data=df_resenas,
        order=ORDEN_SENTIMIENTO,
        hue="sentimiento",
        palette=COLORES_SENTIMIENTO,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Tiempo de respuesta por sentimiento")
    axes[0].set_xlabel("Sentimiento")
    axes[0].set_ylabel("Tiempo de respuesta (horas)")

    tramos = pd.cut(
        df_resenas["tiempo_respuesta_horas"],
        bins=[0, 6, 12, 24, 48, float("inf")],
        labels=["≤6 h", "6-12 h", "12-24 h", "24-48 h", ">48 h"],
        include_lowest=True,
    )
    tabla = pd.crosstab(tramos, df_resenas["sentimiento"], normalize="index").mul(100)[
        ORDEN_SENTIMIENTO
    ]
    _barras_apiladas(axes[1], tabla, "Sentimiento por tramo de respuesta", "Porcentaje de reseñas")

    fig.suptitle("Tiempo de respuesta y sentimiento")
    _leyenda_sentimiento(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    rutas.append(_guardar(fig, directorio / "tiempo_respuesta.png"))

    return rutas


if __name__ == "__main__":
    for ruta in visualize_data() + visualize_target_relationships():
        print(ruta)
