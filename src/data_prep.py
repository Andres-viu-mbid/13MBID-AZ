import pandas as pd
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolver(ruta, base):
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else base / ruta


def _relativa(ruta):
    ruta = Path(ruta)
    return str(ruta.relative_to(REPO_ROOT)) if ruta.is_relative_to(REPO_ROOT) else str(ruta)


def _generar_reporte(pasos, resumen, ruta):
    """ Genera un reporte HTML con el registro de las transformaciones aplicadas

    Args:
        pasos (list[dict]): Pasos aplicados (paso, descripcion, filas_antes, filas_despues, detalle)
        resumen (dict): Datos generales de la ejecución (clave -> valor)
        ruta (Path): Ruta del archivo HTML a generar

    Returns:
        Path: Ruta del reporte generado
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    df_resumen = pd.DataFrame(list(resumen.items()), columns=["Atributo", "Valor"])
    df_pasos = pd.DataFrame(pasos)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Reporte de transformaciones</title>
<style>
body {{ font-family: sans-serif; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #999; padding: 6px 8px; }}
</style>
</head>
<body>
<h1>Reporte de transformaciones</h1>
<h2>Resumen</h2>
{df_resumen.to_html(index=False)}
<h2>Pasos aplicados</h2>
{df_pasos.to_html(index=False)}
</body>
</html>
"""

    ruta.write_text(html, encoding="utf-8")
    return ruta


def process_data(
    datos_clientes: str = "data/raw/datos_clientes.csv",
    datos_resenas: str = "data/raw/datos_resenas.csv",
    output_dir: str = "data/processed",
    reporte: str = "docs/reporte_transformaciones.html",
) -> list[Path]:
    """ Función para procesar los datos de clientes y reseñas

    Args:
        datos_clientes (str): Ruta al archivo CSV con los datos de clientes
        datos_resenas (str): Ruta al archivo CSV con los datos de reseñas
        output_dir (str): Ruta al directorio donde se guardarán los archivos procesados
        reporte (str): Ruta al archivo HTML con el reporte de transformaciones

    Returns:
        list[Path]: [ruta_integrados, ruta_reporte] con las rutas del dataset integrado y el reporte
    """
    pasos = []

    ####################################################################
    # Lectura de los datos
    ####################################################################
    df_clientes = pd.read_csv(_resolver(datos_clientes, REPO_ROOT), sep=";")
    pasos.append({
        "paso": 1,
        "descripcion": "Lectura de datos de clientes",
        "filas_antes": len(df_clientes),
        "filas_despues": len(df_clientes),
        "detalle": f"{df_clientes.shape[1]} columnas",
    })

    df_resenas = pd.read_csv(_resolver(datos_resenas, REPO_ROOT), sep=";")
    pasos.append({
        "paso": 2,
        "descripcion": "Lectura de datos de reseñas",
        "filas_antes": len(df_resenas),
        "filas_despues": len(df_resenas),
        "detalle": f"{df_resenas.shape[1]} columnas",
    })

    ####################################################################
    # Selección de los datos (filtro de columnas)
    ####################################################################
    col_eliminar_clientes = ['canal_registro']
    filas_antes = len(df_clientes)
    df_clientes.drop(col_eliminar_clientes, inplace=True, axis=1)
    pasos.append({
        "paso": 3,
        "descripcion": "Eliminación de columna canal_registro",
        "filas_antes": filas_antes,
        "filas_despues": len(df_clientes),
        "detalle": "Columna eliminada: canal_registro",
    })

    ####################################################################
    # Limpieza de los datos (filtro de filas)
    ####################################################################
    filas_antes = len(df_clientes)
    clientes_descartados = int((df_clientes["edad"] >= 90).sum())
    df_clientes_filtrado = df_clientes[(df_clientes["edad"] < 90) | (df_clientes["edad"].isna())].copy()
    pasos.append({
        "paso": 4,
        "descripcion": "Filtro de clientes por edad menor a 90",
        "filas_antes": filas_antes,
        "filas_despues": len(df_clientes_filtrado),
        "detalle": f"{clientes_descartados} clientes descartados por edad >= 90",
    })

    filas_antes = len(df_resenas)
    df_resenas_dedup = df_resenas.drop_duplicates()
    duplicados_eliminados = filas_antes - len(df_resenas_dedup)
    df_resenas_filtrado = df_resenas_dedup.copy()
    pasos.append({
        "paso": 5,
        "descripcion": "Eliminación de reseñas duplicadas",
        "filas_antes": filas_antes,
        "filas_despues": len(df_resenas_filtrado),
        "detalle": f"{duplicados_eliminados} filas duplicadas eliminadas",
    })

    ####################################################################
    # Tratamiento de valores nulos
    ####################################################################
    nulos_edad = df_clientes_filtrado["edad"].isna().sum()
    df_clientes_filtrado["edad"] = df_clientes_filtrado.groupby("ciudad")["edad"]\
                        .transform(lambda x: x.fillna(x.median()))
    pasos.append({
        "paso": 6,
        "descripcion": "Imputación de edad por ciudad",
        "filas_antes": len(df_clientes_filtrado),
        "filas_despues": len(df_clientes_filtrado),
        "detalle": f"{nulos_edad} valores nulos imputados con la mediana de edad por ciudad",
    })

    ####################################################################
    # Integración de los datos
    ####################################################################
    filas_antes = len(df_resenas_filtrado)
    datos_integrados = df_resenas_filtrado.merge(
        df_clientes_filtrado, on="id_cliente", how="inner", validate="many_to_one"
    )
    resenas_descartadas = filas_antes - len(datos_integrados)
    pasos.append({
        "paso": 7,
        "descripcion": "Integración de reseñas y clientes",
        "filas_antes": filas_antes,
        "filas_despues": len(datos_integrados),
        "detalle": f"{resenas_descartadas} reseñas descartadas por no tener cliente tras la limpieza",
    })

    nulos_tiempo = datos_integrados["tiempo_respuesta_horas"].isna().sum()
    datos_integrados["tiempo_respuesta_horas"] = datos_integrados.groupby("categoria_producto")["tiempo_respuesta_horas"]\
                        .transform(lambda x: x.fillna(x.median()))
    pasos.append({
        "paso": 8,
        "descripcion": "Imputación de tiempo de respuesta por categoría",
        "filas_antes": len(datos_integrados),
        "filas_despues": len(datos_integrados),
        "detalle": f"{nulos_tiempo} valores nulos imputados con la mediana de tiempo_respuesta_horas por categoria_producto",
    })

    ####################################################################
    # Construcción de atributos derivados
    ####################################################################
    # Longitud en caracteres del texto de la reseña
    datos_integrados["longitud_resena"] = datos_integrados["texto_resena"].str.len()
    # Cantidad de palabras del texto de la reseña
    datos_integrados["cantidad_palabras_resena"] = datos_integrados["texto_resena"].str.split().str.len()
    # Tiempo de respuesta en días, redondeado a dos decimales
    datos_integrados["tiempo_respuesta_dias"] = (datos_integrados["tiempo_respuesta_horas"] / 24).round(2)
    # Número de reseñas (compras) registradas por cada cliente
    datos_integrados["cantidad_resenas_cliente"] = datos_integrados.groupby("id_cliente")["id_resena"].transform("size")
    # Fracción de reseñas de cada cliente con reclamo formal igual a "Sí"
    datos_integrados["proporcion_reclamos_cliente"] = (
        datos_integrados.groupby("id_cliente")["reclamo_formal"]
        .transform(lambda s: (s == "Sí").mean())
        .round(4)
    )
    pasos.append({
        "paso": 9,
        "descripcion": "Construcción de atributos derivados",
        "filas_antes": len(datos_integrados),
        "filas_despues": len(datos_integrados),
        "detalle": "Columnas creadas: longitud_resena, cantidad_palabras_resena, tiempo_respuesta_dias, cantidad_resenas_cliente, proporcion_reclamos_cliente",
    })

    ####################################################################
    # Guardado de los datos procesados
    ####################################################################
    directorio = _resolver(output_dir, REPO_ROOT)
    directorio.mkdir(parents=True, exist_ok=True)

    ruta_integrados = directorio / "datos_integrados.csv"
    datos_integrados.to_csv(ruta_integrados, sep=";", index=False)
    pasos.append({
        "paso": 10,
        "descripcion": "Guardado de datos integrados",
        "filas_antes": len(datos_integrados),
        "filas_despues": len(datos_integrados),
        "detalle": f"{_relativa(ruta_integrados)} ({datos_integrados.shape[1]} columnas)",
    })

    resumen = {
        "Fecha de ejecución": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Datos de clientes": _relativa(_resolver(datos_clientes, REPO_ROOT)),
        "Datos de reseñas": _relativa(_resolver(datos_resenas, REPO_ROOT)),
        "Dataset generado": _relativa(ruta_integrados),
        "Filas finales": len(datos_integrados),
        "Columnas finales": datos_integrados.shape[1],
        "Lista de columnas": ", ".join(datos_integrados.columns),
    }
    ruta_reporte = _generar_reporte(pasos, resumen, _resolver(reporte, REPO_ROOT))
    return [ruta_integrados, ruta_reporte]


if __name__ == "__main__":
    for ruta in process_data():
        print(ruta)
