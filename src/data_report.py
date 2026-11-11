from pathlib import Path

import pandas as pd
import sweetviz as sv

REPO_ROOT = Path(__file__).resolve().parents[1]

COLUMNAS_EXCLUIDAS = ["id_cliente", "id_resena", "texto_resena"]


def _resolver(ruta, base):
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else base / ruta


def generate_report(
    datos_cliente="data/raw/datos_clientes.csv",
    datos_resenas="data/raw/datos_resenas.csv",
    output="docs/reporte_sweetviz.html",
    target="sentimiento",
):
    """Genera el reporte general y una comparación por clase contra el resto; devuelve las rutas."""
    df_clientes = pd.read_csv(_resolver(datos_cliente, REPO_ROOT), sep=";")
    df_resenas = pd.read_csv(_resolver(datos_resenas, REPO_ROOT), sep=";")

    df = df_resenas.merge(df_clientes, on="id_cliente", how="inner")
    df["fecha_resena"] = pd.to_datetime(df["fecha_resena"])
    df["longitud_resena"] = df["texto_resena"].str.len()
    df["cantidad_palabras_resena"] = df["texto_resena"].str.split().str.len()

    destino = _resolver(output, REPO_ROOT)
    destino.parent.mkdir(parents=True, exist_ok=True)

    configuracion = sv.FeatureConfig(skip=COLUMNAS_EXCLUIDAS)

    general = sv.analyze(df, feat_cfg=configuracion, pairwise_analysis="off")
    general.show_html(filepath=str(destino), open_browser=False, layout="widescreen")

    rutas = [destino]
    for clase in sorted(df[target].unique()):
        comparacion = sv.compare_intra(
            df,
            df[target] == clase,
            [clase, "Resto"],
            feat_cfg=configuracion,
            pairwise_analysis="off",
        )
        ruta = destino.with_name(f"{destino.stem}_{clase.lower()}{destino.suffix}")
        comparacion.show_html(filepath=str(ruta), open_browser=False, layout="widescreen")
        rutas.append(ruta)

    return rutas


if __name__ == "__main__":
    for ruta in generate_report():
        print(f"{ruta} ({ruta.stat().st_size / 1024 / 1024:.1f} MB)")
