import pandas as pd
import numpy as np
import os
from utils import load_data

def generate_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa por cliente y clase para obtener transacciones y valores totales."""
    agg = (
        df
        .groupby(["Id_cliente", "clase"])["valor"]
        .agg(num_tx="count", total_valor="sum")
        .reset_index()
    )
    return agg


def pivot_features(agg: pd.DataFrame) -> pd.DataFrame:
    """Convierte la tabla a formato ancho, con columnas por clase de transacción."""
    pivot = (
        agg
        .pivot(index="Id_cliente", columns="clase", values=["num_tx", "total_valor"])
        .fillna(0)
    )
    pivot.columns = ["_".join(col).lower() for col in pivot.columns]
    pivot.index.name = "Id_cliente"
    return pivot


def add_ratio_avance_compra(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columna de ratio entre avance y compra."""
    df["ratio_avance_compra"] = df.get("total_valor_avance", 0) / (df.get("total_valor_compra", 1))
    return df


def add_recencia(df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega la recencia: días desde la última transacción de cada cliente."""
    last_tx = original_df.groupby("Id_cliente")["fecha_efectiva"].max()
    overall_latest = original_df["fecha_efectiva"].max()
    recencia = (overall_latest - last_tx).dt.days.rename("dias_desde_ultima_tx")
    return df.join(recencia, how="left")


def add_total_transacciones(df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega total de transacciones por cliente."""
    total_tx = original_df.groupby("Id_cliente")["valor"].count().rename("num_transacciones")
    df["num_transacciones"] = total_tx
    return df


def save_output(df: pd.DataFrame, path: str) -> None:
    """Guarda el DataFrame final en un archivo CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)



def main():
    input_path = "data/silver/master_table.csv"
    output_path = "data/gold/model_data.csv"

    df = load_data(input_path)
    agg = generate_aggregates(df)
    pivot = pivot_features(agg)
    pivot = add_ratio_avance_compra(pivot)
    pivot = add_recencia(pivot, df)
    pivot = add_total_transacciones(pivot, df)



    save_output(pivot, output_path)


    save_output(pivot, output_path)
    print("Data para modelar lista. Archivo guardado en:", output_path)


if __name__ == "__main__":
    main()
