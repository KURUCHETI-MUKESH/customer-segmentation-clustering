"""
train.py
--------
Trains three unsupervised clustering models on the preprocessed
customer data:
    1. K-Means (primary model)
    2. DBSCAN
    3. Agglomerative (Hierarchical) Clustering

Also contains helpers to determine the optimal number of clusters
for K-Means via the Elbow Method and Silhouette Score.

Run directly:
    python src/train.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from preprocessing import run_preprocessing_pipeline, scale_features, FEATURE_COLUMNS
from utils import save_fig, CLUSTERED_DATA_PATH, get_logger

logger = get_logger(__name__)


def compute_elbow_and_silhouette(scaled_df: pd.DataFrame,
                                  k_range=range(2, 11)) -> pd.DataFrame:
    """Compute inertia (elbow method) and silhouette score for a range of k.

    Parameters
    ----------
    scaled_df : pd.DataFrame
        Standardized feature matrix.
    k_range : range
        Range of cluster counts to evaluate.

    Returns
    -------
    pd.DataFrame
        Columns: ['k', 'inertia', 'silhouette'].
    """
    results = []
    for k in k_range:
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = model.fit_predict(scaled_df)
        sil = silhouette_score(scaled_df, labels)
        results.append({"k": k, "inertia": model.inertia_, "silhouette": sil})
        logger.info("k=%d | inertia=%.2f | silhouette=%.4f", k, model.inertia_, sil)
    return pd.DataFrame(results)


def plot_elbow_and_silhouette(metrics_df: pd.DataFrame) -> None:
    """Plot and save the Elbow Method and Silhouette Score curves."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(metrics_df["k"], metrics_df["inertia"], marker="o", color="#4C72B0")
    axes[0].set_title("Elbow Method for Optimal k")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(metrics_df["k"], metrics_df["silhouette"], marker="o", color="#DD8452")
    axes[1].set_title("Silhouette Score by k")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "elbow_and_silhouette.png")
    logger.info("Saved elbow_and_silhouette.png")


def train_kmeans(scaled_df: pd.DataFrame, n_clusters: int) -> KMeans:
    """Fit a K-Means model with the chosen number of clusters."""
    model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42)
    model.fit(scaled_df)
    logger.info("K-Means fitted with k=%d | inertia=%.2f", n_clusters, model.inertia_)
    return model


def train_dbscan(scaled_df: pd.DataFrame, eps: float = 0.6, min_samples: int = 5) -> DBSCAN:
    """Fit a DBSCAN model on the scaled features."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(scaled_df)
    n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
    n_noise = list(model.labels_).count(-1)
    logger.info("DBSCAN fitted | clusters=%d | noise points=%d", n_clusters, n_noise)
    return model


def train_hierarchical(scaled_df: pd.DataFrame, n_clusters: int) -> AgglomerativeClustering:
    """Fit an Agglomerative (Hierarchical) Clustering model."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    model.fit(scaled_df)
    logger.info("Hierarchical clustering fitted with n_clusters=%d", n_clusters)
    return model


def run_training_pipeline(optimal_k: int = 5) -> pd.DataFrame:
    """Run preprocessing, then fit K-Means, DBSCAN, and Hierarchical models.

    Parameters
    ----------
    optimal_k : int
        Number of clusters chosen from the elbow/silhouette analysis.

    Returns
    -------
    pd.DataFrame
        Original cleaned data annotated with cluster labels from all
        three algorithms.
    """
    df = run_preprocessing_pipeline()
    scaled_df, scaler = scale_features(df, FEATURE_COLUMNS)

    metrics_df = compute_elbow_and_silhouette(scaled_df)
    plot_elbow_and_silhouette(metrics_df)

    kmeans_model = train_kmeans(scaled_df, n_clusters=optimal_k)
    dbscan_model = train_dbscan(scaled_df)
    hier_model = train_hierarchical(scaled_df, n_clusters=optimal_k)

    df["KMeans_Cluster"] = kmeans_model.labels_
    df["DBSCAN_Cluster"] = dbscan_model.labels_
    df["Hierarchical_Cluster"] = hier_model.labels_

    df.to_csv(CLUSTERED_DATA_PATH, index=False)
    logger.info("Saved clustered dataset to %s", CLUSTERED_DATA_PATH)

    return df


if __name__ == "__main__":
    result_df = run_training_pipeline(optimal_k=5)
    print(result_df.head())
