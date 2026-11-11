import pandas as pd
import pytest
from pandera.pandas import DataFrameSchema, Column, Check


GENEROS = ["Femenino", "Masculino", "Otro"]
SENTIMIENTOS = ["Positivo", "Negativo", "Neutral"]
CATEGORIAS_PRODUCTO = ["Alimentos", "Belleza", "Calzado", "Deportes", "Electrónica", "Hogar", "Indumentaria", "Libros"]
CANALES_COMPRA = ["Online", "Offline"]
PLATAFORMAS = ["Amazon", "Falabella", "Flipkart", "JioMart", "Meesho", "MercadoLibre", "Nykaa", "Zepto"]
SI_NO = ["Sí", "No"]
UMBRAL_INCONSISTENCIA = 0.05

@pytest.fixture
def datos_clientes():
    """ Función para cargar los datos de clientes a un dataframe

    Args:
        None

    Returns:
        pd.DataFrame: Dataframe con los datos de clientes
    """
    df = pd.read_csv("data/raw/datos_clientes.csv", sep=";")
    return df


@pytest.fixture
def datos_resenas():
    """ Función para cargar los datos de reseñas a un dataframe

    Args:
        None

    Returns:
        pd.DataFrame: Dataframe con los datos de reseñas
    """
    df = pd.read_csv("data/raw/datos_resenas.csv", sep=";")
    return df


def test_schema_datos_clientes(datos_clientes):
    """ Función para validar el esquema de los datos de clientes

    Args:
        datos_clientes (pd.DataFrame): Dataframe con los datos de clientes

    Returns:
        None
    """
    schema = DataFrameSchema(
        {
            "id_cliente": Column(int, nullable=False),
            "edad": Column(float, nullable=True, checks=Check.in_range(18, 90)),
            "genero": Column(str, nullable=False, checks=Check.isin(GENEROS)),
            "ciudad": Column(str, nullable=False),
            "antiguedad_cliente_meses": Column(int, nullable=True, checks=Check.ge(0)),
            "canal_registro": Column(str, nullable=False),
        }
    )
    # Lazy = True permite que se validen todas las columnas y se muestren todos los
    #       errores encontrados en lugar de detenerse en el primer error.
    schema.validate(datos_clientes, lazy=True)


def test_schema_datos_resenas(datos_resenas):
    """ Función para validar el esquema de los datos de reseñas

    Args:
        datos_resenas (pd.DataFrame): Dataframe con los datos de reseñas

    Returns:
        None
    """
    schema = DataFrameSchema(
        {
            "id_resena": Column(int, nullable=False),
            "id_cliente": Column(int, nullable=False),
            "categoria_producto": Column(str, nullable=False, checks=Check.isin(CATEGORIAS_PRODUCTO)),
            "canal_compra": Column(str, nullable=False, checks=Check.isin(CANALES_COMPRA)),
            "plataforma": Column(str, nullable=False, checks=Check.isin(PLATAFORMAS)),
            "calificacion": Column(int, nullable=False, checks=Check.in_range(1, 5)),
            "texto_resena": Column(str, nullable=False, checks=Check.str_length(min_value=1)),
            "tiempo_respuesta_horas": Column(float, nullable=True, checks=Check.ge(0)),
            "problema_resuelto": Column(str, nullable=False, checks=Check.isin(SI_NO)),
            "reclamo_formal": Column(str, nullable=False, checks=Check.isin(SI_NO)),
            "importe_compra": Column(float, nullable=False, checks=Check.gt(0)),
            "fecha_resena": Column(
                str,
                nullable=True,
                checks=Check(
                    lambda s: pd.to_datetime(s, errors="coerce").notna(),
                    error="fecha_resena no es una fecha válida",
                ),
            ),
            "sentimiento": Column(str, nullable=False, checks=Check.isin(SENTIMIENTOS)),
        },
        strict=True,
    )
    schema.validate(datos_resenas, lazy=True)


def test_basicos_clientes(datos_clientes):
    """ Función para validar los datos básicos de los clientes

    Args:
        datos_clientes (pd.DataFrame): Dataframe con los datos de clientes

    Returns:
        None
    """
    # Validar que no existan valores nulos en las columnas obligatorias
    assert not datos_clientes.empty, "El dataframe de clientes está vacío"
    assert datos_clientes.shape[1] == 6, "El dataframe de clientes no tiene 6 columnas"
    assert datos_clientes["id_cliente"].notnull().all(), "La columna id_cliente tiene valores nulos"
    # EXTRA: pueden completar otras validaciones básicas

def test_basicos_resenas(datos_resenas):
    """ Función para validar los datos básicos de las reseñas

    Args:
        datos_resenas (pd.DataFrame): Dataframe con los datos de reseñas

    Returns:
        None
    """
    # Validar que no existan valores nulos en las columnas obligatorias
    assert not datos_resenas.empty, "El dataframe de reseñas está vacío"
    assert datos_resenas.shape[1] == 13, "El dataframe de reseñas no tiene 13 columnas"
    assert datos_resenas["id_resena"].notnull().all(), "La columna id_resena tiene valores nulos"
    assert datos_resenas["id_cliente"].notnull().all(), "La columna id_cliente tiene valores nulos"
    # EXTRA: pueden completar otras validaciones básicas


def test_unicidad_claves(datos_clientes, datos_resenas):
    """ Función para validar que las claves primarias no se repitan

    Args:
        datos_clientes (pd.DataFrame): Dataframe con los datos de clientes
        datos_resenas (pd.DataFrame): Dataframe con los datos de reseñas

    Returns:
        None
    """
    # El id_cliente no debe repetirse en datos_clientes
    schema_clientes = DataFrameSchema({"id_cliente": Column(int, nullable=False, unique=True)})
    schema_clientes.validate(datos_clientes, lazy=True)
    # El id_resena no debe repetirse en datos_resenas
    schema_resenas = DataFrameSchema({"id_resena": Column(int, nullable=False, unique=True)})
    schema_resenas.validate(datos_resenas, lazy=True)


def test_integridad_referencial(datos_clientes, datos_resenas):
    """ Función para validar la integridad referencial entre los datos de clientes y reseñas

    Args:
        datos_clientes (pd.DataFrame): Dataframe con los datos de clientes
        datos_resenas (pd.DataFrame): Dataframe con los datos de reseñas

    Returns:
        None
    """
    # Validar que todos los id_cliente en datos_resenas existan en datos_clientes
    df_ids = datos_resenas[['id_cliente']].merge(
        datos_clientes[['id_cliente']],
        on='id_cliente',
        how='left',
        indicator=True # genera la columna _merge con el resultado de la unión
    )
    integridad_referencial = DataFrameSchema(
        {
            "_merge": Column(str, nullable=False, checks=Check.isin(["both"]))
        }
    )
    integridad_referencial.validate(df_ids, lazy=True)


def test_consistencia_calificacion_sentimiento(datos_resenas):
    """ Función para validar la consistencia entre calificación y sentimiento

    Una calificación baja no debería ir con un sentimiento positivo, y una
    calificación alta no debería ir con uno negativo.

    Args:
        datos_resenas (pd.DataFrame): Dataframe con los datos de reseñas

    Returns:
        None
    """
    # Reseñas con calificación baja y su proporción de sentimientos positivos
    calificacion_baja = datos_resenas[datos_resenas["calificacion"] < 3]
    proporcion_positivos = (calificacion_baja["sentimiento"] == "Positivo").mean()
    assert proporcion_positivos <= UMBRAL_INCONSISTENCIA, (
        f"El {proporcion_positivos:.2%} de las reseñas con calificación menor a 3 tiene sentimiento Positivo "
        f"(máximo permitido {UMBRAL_INCONSISTENCIA:.0%})"
    )
    # Reseñas con calificación alta y su proporción de sentimientos negativos
    calificacion_alta = datos_resenas[datos_resenas["calificacion"] > 3]
    proporcion_negativos = (calificacion_alta["sentimiento"] == "Negativo").mean()
    assert proporcion_negativos <= UMBRAL_INCONSISTENCIA, (
        f"El {proporcion_negativos:.2%} de las reseñas con calificación mayor a 3 tiene sentimiento Negativo "
        f"(máximo permitido {UMBRAL_INCONSISTENCIA:.0%})"
    )
