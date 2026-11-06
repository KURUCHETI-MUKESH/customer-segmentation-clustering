
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "mall_customers_clustered.csv"
EVALUATION_PATH = BASE_DIR / "reports" / "algorithm_comparison.csv"
SUMMARY_PATH = BASE_DIR / "reports" / "customer_segment_summary.csv"

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Analytics",
    page_icon="📊",
    layout="wide"
)

# Dashboard heading
st.title("📊 Customer Segmentation Analytics")
st.caption("Interactive Customer Behavior Analysis Dashboard")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_evaluation():
    if EVALUATION_PATH.exists():
        return pd.read_csv(EVALUATION_PATH)
    return pd.DataFrame()

@st.cache_data
def load_summary():
    if SUMMARY_PATH.exists():
        return pd.read_csv(SUMMARY_PATH)
    return pd.DataFrame()

try:
    df = load_data()
except FileNotFoundError:
    st.error("Customer dataset not found. Run model training first.")
    st.stop()

# Algorithm selection
algorithm_options = {
    "K-Means": "KMeans_Cluster",
    "DBSCAN": "DBSCAN_Cluster",
    "Hierarchical": "Hierarchical_Cluster"
}

st.sidebar.header("Dashboard Settings")

selected_algorithm = st.sidebar.selectbox(
    "Select Clustering Algorithm",
    list(algorithm_options.keys())
)

cluster_column = algorithm_options[selected_algorithm]

if cluster_column not in df.columns:
    st.error(f"Missing column: {cluster_column}")
    st.stop()

# Prepare data
df[cluster_column] = pd.to_numeric(
    df[cluster_column],
    errors="coerce"
)

df = df.dropna(subset=[cluster_column]).copy()

# Sidebar filters
if "Gender" in df.columns:
    gender_options = ["All"] + sorted(
        df["Gender"].dropna().astype(str).unique().tolist()
    )

    selected_gender = st.sidebar.selectbox(
        "Filter by Gender",
        gender_options
    )

    if selected_gender != "All":
        df = df[
            df["Gender"].astype(str) == selected_gender
        ]


# Age Filter
st.sidebar.subheader("Customer Filters")

min_age = int(df["Age"].min())
max_age = int(df["Age"].max())

age_range = st.sidebar.slider(
    "Select Age Range",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)

# Annual Income Filter
min_income = int(df["Annual Income (k$)"].min())
max_income = int(df["Annual Income (k$)"].max())

income_range = st.sidebar.slider(
    "Select Annual Income (k$)",
    min_value=min_income,
    max_value=max_income,
    value=(min_income, max_income)
)

# Spending Score Filter
min_spending = int(df["Spending Score (1-100)"].min())
max_spending = int(df["Spending Score (1-100)"].max())

spending_range = st.sidebar.slider(
    "Select Spending Score",
    min_value=min_spending,
    max_value=max_spending,
    value=(min_spending, max_spending)
)

# Apply all filters
df = df[
    (df["Age"].between(age_range[0], age_range[1])) &
    (df["Annual Income (k$)"].between(income_range[0], income_range[1])) &
    (df["Spending Score (1-100)"].between(spending_range[0], spending_range[1]))
].copy()


# Customer ID Search
st.sidebar.divider()
st.sidebar.subheader("Search Customer")

customer_ids = sorted(
    df["CustomerID"].dropna().unique().tolist()
)

selected_customer_id = st.sidebar.selectbox(
    "Select Customer ID",
    ["None"] + customer_ids
)

if selected_customer_id != "None":
    customer_details = df[
        df["CustomerID"] == selected_customer_id
    ]

    st.subheader("Customer Details")

    if not customer_details.empty:
        customer = customer_details.iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric("Customer ID", int(customer["CustomerID"]))
        col2.metric("Age", int(customer["Age"]))
        col3.metric("Gender", str(customer["Gender"]))

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Annual Income",
            f"${customer['Annual Income (k$)']:.2f}k"
        )
        col5.metric(
            "Spending Score",
            int(customer["Spending Score (1-100)"])
        )
        col6.metric(
            "Cluster",
            int(customer[cluster_column])
        )

    st.divider()




# KPI metrics
total_customers = len(df)
total_clusters = df[cluster_column].nunique()
average_income = df["Annual Income (k$)"].mean()
average_spending = df["Spending Score (1-100)"].mean()

st.subheader("Customer Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", total_customers)
col2.metric("Customer Groups", total_clusters)
col3.metric("Average Income", f"${average_income:.2f}k")
col4.metric("Average Spending", f"{average_spending:.2f}")

st.divider()


# Customer Distribution
st.subheader(f"{selected_algorithm} - Customer Distribution")

import plotly.express as px

cluster_counts = (
    df[cluster_column]
    .value_counts()
    .sort_index()
    .reset_index()
)

cluster_counts.columns = ["Cluster", "Customer Count"]
cluster_counts["Cluster"] = cluster_counts["Cluster"].astype(str)

colors = [
    "#3B82F6",
    "#14B8A6",
    "#F59E0B",
    "#EC4899",
    "#8B5CF6",
    "#EF4444"
]

fig = px.bar(
    cluster_counts,
    x="Cluster",
    y="Customer Count",
    color="Cluster",
    color_discrete_sequence=colors,
    text_auto=True,
    title="Number of Customers in Each Cluster",
    template="plotly_dark"
)

fig.update_traces(textposition="outside")

fig.update_layout(
    height=420,
    showlegend=False,
    xaxis_title="Customer Cluster",
    yaxis_title="Number of Customers"
)

st.plotly_chart(fig, use_container_width=True)



# Customer Cluster Visualization
st.subheader("Customer Cluster Visualization")

if not df.empty:

    import plotly.express as px

    plot_df = df.copy()
    plot_df[cluster_column] = plot_df[cluster_column].astype(str)

    fig = px.scatter(
        plot_df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color=cluster_column,
        hover_data=["CustomerID", "Age", "Gender"],
        title=f"{selected_algorithm} Customer Clusters",
        labels={
            "Annual Income (k$)": "Annual Income ($k)",
            "Spending Score (1-100)": "Spending Score",
            cluster_column: "Customer Cluster"
        },
        template="plotly_dark"
    )

    fig.update_traces(
        marker=dict(size=11, opacity=0.8)
    )

    fig.update_layout(
        height=550,
        legend_title_text="Customer Cluster"
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No customers available for visualization.")


st.divider()

# Cluster summary
st.subheader("Customer Segment Details")

summary = df.groupby(cluster_column).agg(
    Customer_Count=("CustomerID", "count"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending=("Spending Score (1-100)", "mean")
).round(2).reset_index()

st.dataframe(
    summary,
    use_container_width=True
)


# Customer Profile Analysis
st.divider()
st.subheader("Customer Profile Analysis")

if not df.empty:

    import plotly.express as px

    profile = df.groupby(cluster_column).agg(
        Average_Age=("Age", "mean"),
        Average_Income=("Annual Income (k$)", "mean"),
        Average_Spending=("Spending Score (1-100)", "mean")
    ).round(2).reset_index()

    profile[cluster_column] = profile[cluster_column].astype(str)

    colors = [
        "#3B82F6",
        "#14B8A6",
        "#F59E0B",
        "#EC4899",
        "#8B5CF6",
        "#EF4444"
    ]

    def create_profile_chart(column, title, y_label):
        fig = px.bar(
            profile,
            x=cluster_column,
            y=column,
            color=cluster_column,
            color_discrete_sequence=colors,
            text_auto=".2f",
            title=title,
            labels={
                cluster_column: "Customer Cluster",
                column: y_label
            },
            template="plotly_dark"
        )

        fig.update_layout(
            height=380,
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=30),
            title_font_size=18
        )

        fig.update_traces(
            textposition="outside",
            marker_line_width=0
        )

        return fig

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            create_profile_chart(
                "Average_Age",
                "Average Customer Age",
                "Average Age"
            ),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            create_profile_chart(
                "Average_Income",
                "Average Annual Income",
                "Income (k$)"
            ),
            use_container_width=True
        )

    st.plotly_chart(
        create_profile_chart(
            "Average_Spending",
            "Average Spending Score",
            "Spending Score"
        ),
        use_container_width=True
    )

    st.subheader("Customer Profile Summary")

    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("No customer data available for profile analysis.")



# Customer Segment Insights
st.divider()
st.subheader("Customer Segment Insights")

if not df.empty:

    # Calculate overall averages for comparison
    income_median = df["Annual Income (k$)"].median()
    spending_median = df["Spending Score (1-100)"].median()

    insights = []

    for cluster_id, group in df.groupby(cluster_column):

        avg_income = group["Annual Income (k$)"].mean()
        avg_spending = group["Spending Score (1-100)"].mean()
        customer_count = len(group)

        # Handle DBSCAN noise points
        if selected_algorithm == "DBSCAN" and cluster_id == -1:
            segment = "Noise / Outliers"
            recommendation = "Investigate unusual customer behavior."

        elif avg_income >= income_median and avg_spending >= spending_median:
            segment = "Premium Customers"
            recommendation = "Offer loyalty rewards and premium products."

        elif avg_income >= income_median and avg_spending < spending_median:
            segment = "High Income, Low Spending"
            recommendation = "Use personalized promotions to encourage spending."

        elif avg_income < income_median and avg_spending >= spending_median:
            segment = "Budget-Conscious Active Customers"
            recommendation = "Offer discounts and value-for-money bundles."

        else:
            segment = "Low Engagement Customers"
            recommendation = "Use targeted campaigns to improve engagement."

        insights.append({
            "Cluster": int(cluster_id),
            "Customer Count": customer_count,
            "Customer Segment": segment,
            "Average Income": round(avg_income, 2),
            "Average Spending": round(avg_spending, 2),
            "Recommendation": recommendation
        })

    insights_df = pd.DataFrame(insights)

    st.dataframe(
        insights_df,
        use_container_width=True,
        hide_index=True
    )


# Customer Segment Distribution Chart
st.subheader("Customer Segment Distribution")

if not insights_df.empty:

    # Prepare chart data
    chart_data = insights_df[
        ["Customer Segment", "Customer Count"]
    ].copy()

    # Cluster colors
    colors = [
        "#3B82F6",
        "#14B8A6",
        "#F59E0B",
        "#EC4899",
        "#8B5CF6",
        "#EF4444"
    ]

    # Create colorful bar chart
    fig = px.bar(
        chart_data,
        x="Customer Segment",
        y="Customer Count",
        color="Customer Segment",
        color_discrete_sequence=colors,
        text_auto=True,
        title="Customer Count by Segment",
        template="plotly_dark"
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        height=450,
        showlegend=False,
        xaxis_title="Customer Segment",
        yaxis_title="Number of Customers",
        xaxis_tickangle=-20
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Calculate customer percentage
    insights_df["Customer Percentage"] = (
        insights_df["Customer Count"] /
        insights_df["Customer Count"].sum() * 100
    ).round(2)

    # Display percentage table
    st.subheader("Customer Segment Percentage")

    st.dataframe(
        insights_df[
            ["Customer Segment", "Customer Percentage"]
        ],
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("No customers available for segment insights.")



# Model evaluation
st.subheader("Model Evaluation Comparison")

evaluation = load_evaluation()

if not evaluation.empty:
    st.dataframe(
        evaluation,
        use_container_width=True
    )
else:
    st.info("Run model evaluation to generate comparison results.")

# Download customer results
st.subheader("Download Results")

csv_data = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Customer Segmentation CSV",
    data=csv_data,
    file_name="customer_segmentation_results.csv",
    mime="text/csv"
)

# Download Business Insights Report
st.subheader("Download Business Report")

if not df.empty and not insights_df.empty:

    report_text = f"""
CUSTOMER SEGMENTATION BUSINESS REPORT
=====================================

Clustering Algorithm: {selected_algorithm}

Total Customers Analyzed: {len(df)}
Number of Customer Segments: {len(insights_df)}

CUSTOMER SEGMENT ANALYSIS
-------------------------

"""

    for _, row in insights_df.iterrows():
        report_text += f"""
Cluster: {row['Cluster']}
Customer Segment: {row['Customer Segment']}
Customer Count: {row['Customer Count']}
Average Income: ${row['Average Income']}k
Average Spending: {row['Average Spending']}
Recommendation: {row['Recommendation']}

-----------------------------------------
"""

    st.download_button(
        label="Download Business Report",
        data=report_text,
        file_name="customer_segmentation_business_report.txt",
        mime="text/plain"
    )

else:
    st.info("No data available to generate the business report.")


st.success("Dashboard loaded successfully!")
