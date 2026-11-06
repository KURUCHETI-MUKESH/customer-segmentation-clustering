
"""
Improved Customer Segmentation Model Evaluation

New features:
1. Silhouette Score
2. Davies-Bouldin Index
3. Calinski-Harabasz Score
4. Algorithm comparison CSV
5. Customer segment summary
6. Business insights report
7. Cluster visualization
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from preprocessing import FEATURE_COLUMNS, scale_features

from utils import (
    CLUSTERED_DATA_PATH,
    REPORTS_DIR,
    get_logger,
    save_fig,
    ensure_dir
)

logger = get_logger(__name__)

sns.set_style("whitegrid")


def evaluate_clustering(
    scaled_df,
    labels,
    name
):
    """Evaluate a clustering algorithm."""

    labels = pd.Series(
        labels,
        index=scaled_df.index
    )

    # Remove DBSCAN noise points
    mask = labels != -1

    X = scaled_df.loc[mask]
    valid_labels = labels.loc[mask]

    n_clusters = valid_labels.nunique()

    # Check whether metrics can be calculated
    if (
        n_clusters < 2
        or n_clusters >= len(valid_labels)
    ):

        logger.warning(
            "%s produced invalid cluster structure.",
            name
        )

        return {
            "Algorithm": name,
            "Clusters": n_clusters,
            "Silhouette Score": None,
            "Davies-Bouldin Index": None,
            "Calinski-Harabasz Score": None
        }

    # Calculate metrics
    silhouette = silhouette_score(
        X,
        valid_labels
    )

    davies = davies_bouldin_score(
        X,
        valid_labels
    )

    calinski = calinski_harabasz_score(
        X,
        valid_labels
    )

    logger.info(
        "%s evaluated successfully.",
        name
    )

    return {
        "Algorithm": name,
        "Clusters": n_clusters,
        "Silhouette Score": round(silhouette, 4),
        "Davies-Bouldin Index": round(davies, 4),
        "Calinski-Harabasz Score": round(calinski, 4)
    }


def compare_algorithms(df):
    """Compare all three clustering algorithms."""

    scaled_df, _ = scale_features(
        df,
        FEATURE_COLUMNS
    )

    algorithms = {
        "K-Means": "KMeans_Cluster",
        "DBSCAN": "DBSCAN_Cluster",
        "Hierarchical": "Hierarchical_Cluster"
    }

    results = []

    for name, column in algorithms.items():

        if column not in df.columns:
            logger.warning(
                "Missing cluster column: %s",
                column
            )
            continue

        result = evaluate_clustering(
            scaled_df,
            df[column],
            name
        )

        results.append(result)

    comparison_df = pd.DataFrame(results)

    ensure_dir(REPORTS_DIR)

    report_path = os.path.join(
        REPORTS_DIR,
        "algorithm_comparison.csv"
    )

    comparison_df.to_csv(
        report_path,
        index=False
    )

    print("\nMODEL EVALUATION RESULTS")
    print("========================")
    print(comparison_df.to_string(index=False))

    print("\nSaved:", report_path)

    return comparison_df


def plot_cluster_scatter(
    df,
    cluster_col,
    filename
):
    """Visualize customer clusters."""

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue=cluster_col,
        palette="tab10",
        s=70,
        ax=ax
    )

    ax.set_title(
        f"Customer Segmentation - {cluster_col}"
    )

    ax.legend(
        title="Customer Cluster",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    fig.tight_layout()

    save_fig(fig, filename)

    print("Chart saved:", filename)


def summarize_clusters(
    df,
    cluster_col="KMeans_Cluster"
):
    """Generate customer segment statistics."""

    summary = df.groupby(cluster_col).agg(
        Customer_Count=("CustomerID", "count"),
        Average_Age=("Age", "mean"),
        Average_Income=("Annual Income (k$)", "mean"),
        Average_Spending=(
            "Spending Score (1-100)",
            "mean"
        )
    ).round(2).reset_index()

    ensure_dir(REPORTS_DIR)

    summary_path = os.path.join(
        REPORTS_DIR,
        "customer_segment_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\nCUSTOMER SEGMENT SUMMARY")
    print("========================")
    print(summary.to_string(index=False))

    print("\nSaved:", summary_path)

    return summary


def generate_business_insights(summary):
    """Generate business recommendations."""

    insights = []

    for _, row in summary.iterrows():

        income = row["Average_Income"]
        spending = row["Average_Spending"]

        if income >= 60 and spending >= 55:

            segment = "Premium Customers"

            action = (
                "Offer loyalty rewards and premium products."
            )

        elif income >= 60 and spending < 45:

            segment = "High Income, Low Spending"

            action = (
                "Use personalized promotions to encourage spending."
            )

        elif income < 45 and spending >= 55:

            segment = "Budget Conscious Active Customers"

            action = (
                "Offer discounts and value-for-money bundles."
            )

        elif income < 45 and spending < 45:

            segment = "Low Engagement Customers"

            action = (
                "Use targeted campaigns to improve engagement."
            )

        else:

            segment = "Average Customers"

            action = (
                "Maintain engagement through seasonal offers."
            )

        insights.append({
            "Cluster": int(row.iloc[0]),
            "Customer Segment": segment,
            "Recommendation": action
        })

    insights_df = pd.DataFrame(insights)

    insights_path = os.path.join(
        REPORTS_DIR,
        "business_insights.csv"
    )

    insights_df.to_csv(
        insights_path,
        index=False
    )

    print("\nBUSINESS INSIGHTS")
    print("=================")
    print(insights_df.to_string(index=False))

    print("\nSaved:", insights_path)

    return insights_df


def run_evaluation():

    print("\nStarting model evaluation...")

    # Load clustered dataset
    df = pd.read_csv(CLUSTERED_DATA_PATH)

    # Compare algorithms
    comparison = compare_algorithms(df)

    # Generate K-Means summary
    summary = summarize_clusters(
        df,
        "KMeans_Cluster"
    )

    # Generate business insights
    insights = generate_business_insights(
        summary
    )

    # Generate visualizations
    plot_cluster_scatter(
        df,
        "KMeans_Cluster",
        "improved_kmeans_clusters.png"
    )

    plot_cluster_scatter(
        df,
        "DBSCAN_Cluster",
        "improved_dbscan_clusters.png"
    )

    plot_cluster_scatter(
        df,
        "Hierarchical_Cluster",
        "improved_hierarchical_clusters.png"
    )

    print("\nMODEL EVALUATION COMPLETED!")

    return comparison, summary, insights


if __name__ == "__main__":

    run_evaluation()
