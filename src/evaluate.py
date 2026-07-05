"""
evaluate.py
-----------
Evaluates and compares the three clustering algorithms (K-Means,
DBSCAN, Hierarchical) using internal validation metrics, and produces
visualizations plus a plain-language business-insight summary for the
final chosen segmentation.

Run directly:
    python src/evaluate.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score

from preprocessing import FEATURE_COLUMNS, scale_features
from utils import save_fig, get_logger

logger = get_logger(__name__)

sns.set_style("whitegrid")


def evaluate_clustering(scaled_df: pd.DataFrame, labels, name: str) -> dict:
    """Compute Silhouette Score and Davies-Bouldin Index for a clustering result.

    Noise points (label == -1, produced by DBSCAN) are excluded from the
    metric computation since they do not belong to any cluster.

    Parameters
    ----------
    scaled_df : pd.DataFrame
        Standardized feature matrix.
    labels : array-like
        Cluster labels produced by the model.
    name : str
        Name of the algorithm, used for logging.

    Returns
    -------
    dict
        {'algorithm', 'n_clusters', 'silhouette', 'davies_bouldin'}
    """
    labels = pd.Series(labels, index=scaled_df.index)
    mask = labels != -1
    n_clusters = labels[mask].nunique()

    if n_clusters < 2:
        logger.warning("%s produced fewer than 2 valid clusters; metrics skipped.", name)
        return {"algorithm": name, "n_clusters": n_clusters,
                "silhouette": None, "davies_bouldin": None}

    sil = silhouette_score(scaled_df[mask], labels[mask])
    dbi = davies_bouldin_score(scaled_df[mask], labels[mask])
    logger.info("%s | clusters=%d | silhouette=%.4f | davies_bouldin=%.4f",
                name, n_clusters, sil, dbi)
    return {"algorithm": name, "n_clusters": n_clusters,
            "silhouette": sil, "davies_bouldin": dbi}


def compare_algorithms(df: pd.DataFrame) -> pd.DataFrame:
    """Compare K-Means, DBSCAN, and Hierarchical clustering results.

    Parameters
    ----------
    df : pd.DataFrame
        Clustered dataset containing 'KMeans_Cluster', 'DBSCAN_Cluster',
        and 'Hierarchical_Cluster' columns.

    Returns
    -------
    pd.DataFrame
        Comparison table with metrics for each algorithm.
    """
    scaled_df, _ = scale_features(df, FEATURE_COLUMNS)
    rows = [
        evaluate_clustering(scaled_df, df["KMeans_Cluster"], "K-Means"),
        evaluate_clustering(scaled_df, df["DBSCAN_Cluster"], "DBSCAN"),
        evaluate_clustering(scaled_df, df["Hierarchical_Cluster"], "Hierarchical"),
    ]
    comparison_df = pd.DataFrame(rows)
    logger.info("\n%s", comparison_df.to_string(index=False))
    return comparison_df


def plot_cluster_scatter(df: pd.DataFrame, cluster_col: str, filename: str) -> None:
    """Scatter plot of Annual Income vs Spending Score colored by cluster."""
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = sns.color_palette("tab10", df[cluster_col].nunique())
    sns.scatterplot(
        data=df, x="Annual Income (k$)", y="Spending Score (1-100)",
        hue=cluster_col, palette=palette, s=70, ax=ax
    )
    ax.set_title(f"Customer Segments — {cluster_col.replace('_', ' ')}")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_fig(fig, filename)
    logger.info("Saved %s", filename)


def summarize_clusters(df: pd.DataFrame, cluster_col: str = "KMeans_Cluster") -> pd.DataFrame:
    """Produce per-cluster summary statistics used to derive business insights.

    Parameters
    ----------
    df : pd.DataFrame
    cluster_col : str
        Which cluster label column to summarize by.

    Returns
    -------
    pd.DataFrame
        Mean Age, Income, Spending Score, and count per cluster.
    """
    summary = df.groupby(cluster_col).agg(
        Count=("CustomerID", "count"),
        Avg_Age=("Age", "mean"),
        Avg_Income=("Annual Income (k$)", "mean"),
        Avg_Spending=("Spending Score (1-100)", "mean"),
    ).round(1).reset_index()
    return summary


def generate_business_insights(summary: pd.DataFrame) -> list:
    """Translate cluster statistics into plain-language business insights.

    Parameters
    ----------
    summary : pd.DataFrame
        Output of `summarize_clusters`.

    Returns
    -------
    list of str
        One insight sentence per cluster.
    """
    insights = []
    for _, row in summary.iterrows():
        cluster_id = int(row.iloc[0])
        income, spend, age = row["Avg_Income"], row["Avg_Spending"], row["Avg_Age"]

        if income >= 60 and spend >= 55:
            label = "Premium / High-Value Customers"
            action = "Target with loyalty programs and premium product launches."
        elif income >= 60 and spend < 45:
            label = "High Income, Cautious Spenders"
            action = "Use personalized offers and trust-building campaigns to increase spend."
        elif income < 45 and spend >= 55:
            label = "Budget-Conscious Big Spenders"
            action = "Offer installment plans and value bundles; high engagement potential."
        elif income < 45 and spend < 45:
            label = "Low Engagement Customers"
            action = "Re-engagement campaigns, discounts, and win-back offers recommended."
        else:
            label = "Standard / Average Customers"
            action = "Maintain steady engagement through seasonal promotions."

        insights.append(
            f"Cluster {cluster_id} — {label} (n={int(row['Count'])}, "
            f"avg age={age}, avg income=${income}k, avg spending score={spend}): {action}"
        )
    return insights


if __name__ == "__main__":
    from utils import CLUSTERED_DATA_PATH

    clustered_df = pd.read_csv(CLUSTERED_DATA_PATH)

    comparison = compare_algorithms(clustered_df)
    print("\nAlgorithm comparison:\n", comparison)

    plot_cluster_scatter(clustered_df, "KMeans_Cluster", "kmeans_clusters.png")
    plot_cluster_scatter(clustered_df, "DBSCAN_Cluster", "dbscan_clusters.png")
    plot_cluster_scatter(clustered_df, "Hierarchical_Cluster", "hierarchical_clusters.png")

    cluster_summary = summarize_clusters(clustered_df, "KMeans_Cluster")
    print("\nCluster summary:\n", cluster_summary)

    for line in generate_business_insights(cluster_summary):
        print("-", line)
