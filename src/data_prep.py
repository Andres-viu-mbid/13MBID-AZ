import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolver(ruta, base):
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else base / ruta


def process_data(
    datos_clientes: str = "data/raw/datos_clientes.csv",
    datos_resenas: str = "data/raw/datos_resenas.csv",
    output_dir: str = "data/processed",
) -> list[Path]:
    """ Función para procesar los datos de clientes y reseñas

    Args:
        datos_clientes (str): Ruta al archivo CSV con los datos de clientes
        datos_resenas (str): Ruta al archivo CSV con los datos de reseñas
        output_dir (str): Ruta al directorio donde se guardarán los archivos procesados

    Returns:
        list[Path]: Rutas de los archivos procesados que se guardaron
    """
    # Lectura de los datos
    df_clientes = pd.read_csv(_resolver(datos_clientes, REPO_ROOT), sep=";")
    df_resenas = pd.read_csv(_resolver(datos_resenas, REPO_ROOT), sep=";")

    ####################################################################
    # Selección de los datos (filtro de columnas)
    ####################################################################
    col_eliminar_clientes = ['canal_registro']
    df_clientes.drop(col_eliminar_clientes, inplace=True, axis=1)

    ####################################################################
    # Limpieza de los datos (filtro de filas)
    ####################################################################
    df_clientes_filtrado = df_clientes[(df_clientes['edad'] < 90) | (df_clientes['edad'].isna())].copy()
    df_resenas_filtrado = df_resenas.drop_duplicates().copy()

    # Integridad referencial: se descartan las reseñas de clientes que fueron filtrados
    df_resenas_filtrado = df_resenas_filtrado[
        df_resenas_filtrado['id_cliente'].isin(df_clientes_filtrado['id_cliente'])
    ].copy()

    ####################################################################
    # Tratamiento de valores nulos
    ####################################################################
    df_clientes_filtrado['edad'] = df_clientes_filtrado.groupby("ciudad")["edad"]\
                        .transform(lambda x: x.fillna(x.median()))
    df_resenas_filtrado['tiempo_respuesta_horas'] = df_resenas_filtrado.groupby("categoria_producto")["tiempo_respuesta_horas"]\
                        .transform(lambda x: x.fillna(x.median()))

    ####################################################################
    # Guardado de los datos procesados
    ####################################################################
    directorio = _resolver(output_dir, REPO_ROOT)
    directorio.mkdir(parents=True, exist_ok=True)

    ruta_clientes = directorio / "datos_clientes.csv"
    ruta_resenas = directorio / "datos_resenas.csv"
    df_clientes_filtrado.to_csv(ruta_clientes, sep=";", index=False)
    df_resenas_filtrado.to_csv(ruta_resenas, sep=";", index=False)

    return [ruta_clientes, ruta_resenas]


if __name__ == "__main__":
    for ruta in process_data():
        print(ruta)
