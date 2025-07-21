#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Rutas y parámetros
data_dir = "data/gold"
INPUT_CSV = os.path.join(data_dir, "model_data.csv")
model_dir = "results/models"
MODEL_PATH = os.path.join(model_dir, "kmeans_model.pkl")
data_output_dir = "results/data"
CLUSTERED_PATH = os.path.join(data_output_dir, "clustered_data.csv")
N_CLUSTERS = 3
SEED = 1998
N_INIT = 10

# Columnas de features para clustering
FEATURE_COLS = [
    "num_transacciones",
    "total_valor_compra",
    "total_valor_avance",
    "ratio_avance_compra",
    "dias_desde_ultima_tx"
]

def load_data(path: str) -> pd.DataFrame:
    """Carga el CSV completo y devuelve un DataFrame."""
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Reemplaza infinitos, imputa valores faltantes y estandariza las features."""
    df_feat = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(df_feat)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, scaler


def train_kmeans(X: np.ndarray) -> KMeans:
    """Entrena el modelo K-Means y devuelve el objeto entrenado."""
    model = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=N_INIT)
    model.fit(X)
    return model


def evaluate(model: KMeans, X: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Calcula etiquetas y métricas de clustering."""
    labels = model.labels_
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    return labels, sil, ch, db


def print_results(
    labels: np.ndarray,
    sil: float,
    ch: float,
    db: float,
    scaler: StandardScaler,
    model: KMeans
) -> None:
    """Imprime distribución de clusters, centroides y métricas."""
    counts = np.bincount(labels)
    print("Clientes por cluster:")
    for i, c in enumerate(counts):
        print(f"  Cluster {i}: {c}")

    centroids = scaler.inverse_transform(model.cluster_centers_)
    centroids_df = pd.DataFrame(
        centroids,
        columns=FEATURE_COLS,
        index=[f"cluster_{i}" for i in range(N_CLUSTERS)]
    )
    print("\nCentroides de cada cluster:")
    print(centroids_df)

    print(f"\nSilhouette Score       : {sil:.3f}")
    print(f"Calinski-Harabasz Index: {ch:.1f}")
    print(f"Davies-Bouldin Index   : {db:.3f}")


def save_model(model: KMeans, path: str) -> None:
    """Guarda el modelo entrenado en disco."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"\nModelo guardado en: {path}")


def save_clustered_data(df: pd.DataFrame, labels: np.ndarray, path: str) -> None:
    """Guarda los datos completos con la etiqueta de cluster."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_copy = df.copy()
    df_copy['cluster'] = labels
    df_copy.to_csv(path, index=False)
    print(f"Datos con cluster guardados en: {path}")


def main():
    # 1) Carga completa de datos
    df = load_data(INPUT_CSV)

    # 2) Preprocesamiento (reemplazo de inf, imputación, escalado)
    X_scaled, scaler = preprocess(df)

    # 3) Entrenamiento de K-Means
    model = train_kmeans(X_scaled)

    # 4) Evaluación
    labels, sil, ch, db = evaluate(model, X_scaled)

    # 5) Mostrar resultados
    print_results(labels, sil, ch, db, scaler, model)

    # 6) Guardar modelo entrenado
    save_model(model, MODEL_PATH)

    # 7) Guardar DataFrame con clusters
    save_clustered_data(df, labels, CLUSTERED_PATH)


if __name__ == "__main__":
    main()
