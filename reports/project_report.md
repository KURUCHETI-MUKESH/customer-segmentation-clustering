# Project Report: Customer Segmentation using Unsupervised Machine Learning

**Project Title:** Customer Segmentation using K-Means, DBSCAN, and Hierarchical Clustering
**Category:** Unsupervised Machine Learning
**Author:** Shubh
**Type:** College Minor Project

---

## 1. Abstract

Businesses often have rich customer data but no easy way to translate it into strategy. This
project applies three unsupervised clustering algorithms — **K-Means**, **DBSCAN**, and
**Agglomerative (Hierarchical) Clustering** — to a mall customer dataset containing age,
annual income, and spending score, in order to discover natural customer segments. The
optimal number of clusters is determined objectively using the **Elbow Method** and
**Silhouette Score**. The three algorithms are then compared using internal validation
metrics, and the best-performing model (K-Means) is used to generate concrete,
plain-language business recommendations for each discovered segment.

---

## 2. Introduction

### 2.1 Problem Statement

A mall/retail business wants to understand its customer base well enough to run targeted
marketing campaigns instead of one-size-fits-all promotions. However, the business has no
pre-labeled "customer type" — it only has raw demographic and behavioural data. This is a
classic **unsupervised learning** problem: we must discover the structure in the data
ourselves.

### 2.2 Objectives

1. Clean and explore a real-world-style customer dataset.
2. Determine the statistically optimal number of customer segments.
3. Train and rigorously compare three different clustering algorithms.
4. Select the best model using quantitative evaluation metrics.
5. Convert the resulting clusters into actionable business insights.

### 2.3 Why Unsupervised Learning?

Supervised learning requires labeled outcomes (e.g., "this customer churned" / "did not
churn"). Here, there is no ground-truth label for "customer segment" — segments are, by
definition, something we want the data to reveal. Clustering algorithms group customers by
similarity in feature space without needing any labels, which is exactly the right tool for
this business problem.

---

## 3. Dataset Description

The project uses a dataset structured identically to the widely-used **Mall Customer
Segmentation Dataset** available on Kaggle
([vjchoudhary7/customer-segmentation-tutorial-in-python](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)).

| Column | Type | Description |
|---|---|---|
| CustomerID | Integer | Unique identifier per customer |
| Gender | Categorical | Male / Female |
| Age | Integer | Customer's age in years |
| Annual Income (k$) | Integer | Annual income in thousands of USD |
| Spending Score (1-100) | Integer | Score assigned by the mall based on customer behaviour and spending |

**Size:** 200 customer records (203 before duplicate removal, to simulate real-world data
quality issues).

The bundled `data/mall_customers.csv` is a synthetically generated dataset engineered to
match the real dataset's statistical structure (five natural income/spending archetypes,
similar age and income ranges) plus a handful of injected missing values and duplicate rows,
so that the preprocessing pipeline in this project has genuine problems to solve. Swapping in
the original Kaggle CSV (same column names) reproduces the same pipeline on real data.

---

## 4. Methodology

### 4.1 Data Preprocessing

1. **Missing values:** 5 missing values were injected into `Annual Income (k$)`. These were
   imputed using the **column median**, which is robust to outliers compared to the mean.
2. **Duplicates:** 3 duplicate rows were injected and removed using `drop_duplicates`
   (ignoring the `CustomerID` column, since IDs are always unique even for duplicate records).
3. **Encoding:** `Gender` was label-encoded (`Female=0`, `Male=1`) for completeness, though it
   was ultimately excluded from the clustering feature set (see §4.2).
4. **Scaling:** `Age`, `Annual Income (k$)`, and `Spending Score (1-100)` were standardized
   using `StandardScaler` (zero mean, unit variance). This is essential because all three
   clustering algorithms used here are **distance-based**, and Income (range ~15–100) would
   otherwise dominate the distance calculation over Spending Score or Age.

### 4.2 Feature Selection

Clustering was performed on `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`.
Gender was explored during EDA but excluded from the clustering features themselves, since
the correlation analysis showed no meaningful relationship between gender and spending
behaviour — including it would have added noise without improving segment quality.

### 4.3 Exploratory Data Analysis

- **Distribution plots** showed Age, Income, and Spending Score are reasonably spread with
  no extreme skew after cleaning.
- **Correlation heatmap** showed near-zero linear correlation between Annual Income and
  Spending Score — confirming that income alone cannot predict spending behaviour, which is
  precisely why a 2D/3D clustering approach adds value over simple income-based segmentation.
- **Pairplots** visually previewed the natural groupings later confirmed by clustering.

### 4.4 Determining the Optimal Number of Clusters

Two complementary methods were used across `k = 2` to `10`:

- **Elbow Method:** plots K-Means inertia (within-cluster sum of squared distances) against
  k. Inertia always decreases as k increases, so we look for the point of diminishing
  returns — the "elbow."
- **Silhouette Score:** measures how similar a point is to its own cluster versus other
  clusters (range: -1 to 1). This gives an objective, single-number way to compare different
  values of k.

Both methods pointed to **k = 5**: the elbow visibly bends around k=4–5, and the silhouette
score peaks at k=5 before declining for higher k.

### 4.5 Model Training

Three clustering algorithms were trained on the same scaled feature matrix:

1. **K-Means** (`n_clusters=5`, `init="k-means++"`, `n_init=10`, `random_state=42`) —
   partitions data into k spherical clusters by minimizing within-cluster variance.
2. **DBSCAN** (`eps=0.6`, `min_samples=5`) — a density-based algorithm that discovers the
   number of clusters automatically and can flag outliers as noise (label `-1`).
3. **Agglomerative (Hierarchical) Clustering** (`n_clusters=5`, `linkage="ward"`) — builds a
   tree of nested clusters bottom-up, merging the pair that increases within-cluster variance
   the least at each step. Ward linkage was chosen because it optimizes the same objective as
   K-Means, making the two directly comparable.

### 4.6 Evaluation Metrics

- **Silhouette Score** (higher is better): how well-separated and internally cohesive
  clusters are.
- **Davies-Bouldin Index** (lower is better): ratio of within-cluster scatter to
  between-cluster separation.

For DBSCAN, noise points were excluded from metric computation, since they are not assigned
to any cluster.

---

## 5. Results

| Algorithm | Clusters Found | Silhouette Score ↑ | Davies-Bouldin Index ↓ |
|---|---|---|---|
| K-Means | 5 | **0.456** | 0.827 |
| DBSCAN | 3 (+ noise points) | 0.432 | **0.767** |
| Hierarchical | 5 | 0.440 | 0.863 |

**Observations:**
- K-Means produced the highest silhouette score, indicating the most internally cohesive,
  well-separated clusters among the three algorithms on this dataset.
- DBSCAN achieved a slightly better Davies-Bouldin index but collapsed into only 3 dense
  clusters (plus a small number of noise points) — useful for outlier detection, but less
  useful for a business that wants every customer assigned to an actionable segment.
- Hierarchical Clustering produced results very close to K-Means, as expected given the
  shared Ward-linkage / variance-minimizing objective — and its dendrogram is a valuable
  visualization tool for presenting the "family tree" of customer similarity to
  non-technical stakeholders.

**Model selected: K-Means (k=5)** — best quantitative performance, full coverage of all
customers, and directly interpretable centroids.

### 5.1 Cluster Profiles (K-Means, k=5)

| Cluster | Count | Avg. Age | Avg. Income ($k) | Avg. Spending Score | Segment Name |
|---|---|---|---|---|---|
| 0 | ~48 | ~38.5 | ~51.1 | ~50.0 | Standard / Average Customers |
| 1 | ~35 | ~32.6 | ~75.7 | ~69.3 | Premium / High-Value Customers |
| 2 | ~37 | ~48.4 | ~80.6 | ~22.0 | High Income, Cautious Spenders |
| 3 | ~40 | ~26.2 | ~28.3 | ~79.4 | Budget-Conscious Big Spenders |
| 4 | ~40 | ~59.2 | ~26.8 | ~21.2 | Low Engagement Customers |

*(Values shown are from the reference run on the bundled synthetic dataset; running the
pipeline will reproduce these deterministically due to the fixed random seed, and will differ
slightly if the original Kaggle CSV is substituted.)*

---

## 6. Business Insights & Recommendations

1. **Premium / High-Value Customers** (moderate-high income, high spending) — the mall's
   core revenue drivers today. *Recommendation:* loyalty tiers, early access to new
   collections, premium-brand cross-promotions.
2. **High Income, Cautious Spenders** (high income, low spending) — represent the single
   largest untapped revenue opportunity. *Recommendation:* trust-building, highly
   personalized offers (not generic discounts) to convert latent purchasing power into
   actual spend.
3. **Budget-Conscious Big Spenders** (lower income, high spending, younger skew) — highly
   engaged despite budget constraints. *Recommendation:* installment/EMI options, bundled
   value packs, referral incentives — this segment is receptive to volume-based promotions.
4. **Low Engagement Customers** (low income, low spending, older skew) — least active
   segment. *Recommendation:* simple, clear discount campaigns and win-back offers; avoid
   over-investing premium marketing budget here.
5. **Standard / Average Customers** — the stable baseline. *Recommendation:* maintain
   satisfaction through routine seasonal promotions without heavy targeted spend.

This segmentation allows a marketing team to allocate budget where it will generate the
highest return, rather than treating all 200+ customers identically.

---

## 7. Conclusion

This project delivered a complete, reproducible unsupervised machine learning pipeline for
customer segmentation — from raw, imperfect data through cleaning, exploratory analysis,
objective model-selection (Elbow + Silhouette), multi-algorithm comparison, and finally
translation into business-usable insights. K-Means with k=5 was selected as the final model
based on its superior silhouette score and full customer coverage, while DBSCAN and
Hierarchical Clustering served as valuable comparison points — DBSCAN for outlier detection,
Hierarchical for its interpretable dendrogram.

## 8. Future Work

- Incorporate additional behavioural features (purchase frequency, recency, basket size) if
  available, moving toward full **RFM (Recency, Frequency, Monetary) segmentation**.
- Use **PCA** to visualize clusters when more than 3 features are used.
- Deploy the trained model as a lightweight web dashboard for live customer scoring.
- Monitor **segment migration** over time to measure whether marketing actions are
  successfully moving customers toward higher-value segments.
- Experiment with **Gaussian Mixture Models** for probabilistic (soft) cluster membership.

---

## 9. References

1. Mall Customer Segmentation Dataset — Kaggle:
   https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
2. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of
   Machine Learning Research, 12, 2825–2830.
3. Rousseeuw, P. J. (1987). *Silhouettes: A graphical aid to the interpretation and
   validation of cluster analysis.* Journal of Computational and Applied Mathematics.
4. Davies, D. L., & Bouldin, D. W. (1979). *A Cluster Separation Measure.* IEEE
   Transactions on Pattern Analysis and Machine Intelligence.
