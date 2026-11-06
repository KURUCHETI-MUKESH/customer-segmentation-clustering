
"""
Improved Customer Data Preprocessing
New features:
1. Data quality analysis
2. Missing value handling
3. Duplicate detection and removal
4. Categorical encoding
5. Feature scaling
6. Data quality report export
"""

import os
import pandas as pd

from typing import Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils import (
    RAW_DATA_PATH,
    CLEAN_DATA_PATH,
    REPORTS_DIR,
    get_logger,
    ensure_dir
)

logger = get_logger(__name__)

FEATURE_COLUMNS = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]


def load_data(path=RAW_DATA_PATH):
    """Load original customer dataset."""

    logger.info("Loading dataset: %s", path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    logger.info("Dataset loaded: %s", df.shape)

    return df


def generate_quality_report(df):
    """Generate a data quality report."""

    report = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(dtype) for dtype in df.dtypes
        ],
        "Missing Values": [
            int(value) for value in df.isnull().sum()
        ],
        "Missing Percentage": [
            round(value / len(df) * 100, 2)
            for value in df.isnull().sum()
        ],
        "Unique Values": [
            int(value) for value in df.nunique()
        ]
    })

    ensure_dir(REPORTS_DIR)

    report_path = os.path.join(
        REPORTS_DIR,
        "data_quality_report.csv"
    )

    report.to_csv(report_path, index=False)

    logger.info(
        "Data quality report saved: %s",
        report_path
    )

    print("\nDATA QUALITY REPORT")
    print(report.to_string(index=False))

    return report


def handle_missing_values(df):
    """Handle missing numeric and categorical values."""

    df = df.copy()

    for col in df.columns:

        missing_count = int(df[col].isnull().sum())

        if missing_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):

            median_value = df[col].median()

            if pd.notna(median_value):
                df[col] = df[col].fillna(median_value)
            else:
                raise ValueError(
                    f"Column {col} contains no valid numeric values."
                )

            logger.info(
                "Filled %d missing values in %s using median",
                missing_count,
                col
            )

        else:

            mode_values = df[col].mode()

            if not mode_values.empty:
                df[col] = df[col].fillna(mode_values.iloc[0])
            else:
                df[col] = df[col].fillna("Unknown")

            logger.info(
                "Filled %d missing values in %s using mode",
                missing_count,
                col
            )

    return df


def remove_duplicates(df):
    """Remove duplicate customer records."""

    before = len(df)

    check_columns = [
        col for col in df.columns
        if col != "CustomerID"
    ]

    df = df.drop_duplicates(
        subset=check_columns
    ).reset_index(drop=True)

    removed = before - len(df)

    logger.info("Duplicate records removed: %d", removed)

    return df


def validate_customer_data(df):
    """Check important customer data fields."""

    required_columns = [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    df = df.copy()

    # Convert required fields to numeric
    for col in required_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Identify invalid values
    invalid_rules = {
        "Age": (df["Age"] < 1) | (df["Age"] > 100),
        "Annual Income (k$)": (
            df["Annual Income (k$)"] <= 0
        ),
        "Spending Score (1-100)": (
            (df["Spending Score (1-100)"] < 1) |
            (df["Spending Score (1-100)"] > 100)
        )
    }

    for col, invalid_mask in invalid_rules.items():

        count = int(invalid_mask.sum())

        if count:
            logger.warning(
                "%d invalid values found in %s",
                count,
                col
            )

            df.loc[invalid_mask, col] = float("nan")

    return df


def encode_categoricals(df):
    """Encode Gender column."""

    df = df.copy()

    if "Gender" in df.columns:

        encoder = LabelEncoder()

        df["Gender_Encoded"] = encoder.fit_transform(
            df["Gender"].astype(str)
        )

        logger.info("Gender encoding completed.")

    return df


def scale_features(
    df,
    feature_columns=None
) -> Tuple[pd.DataFrame, StandardScaler]:

    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    scaler = StandardScaler()

    scaled_array = scaler.fit_transform(
        df[feature_columns]
    )

    scaled_df = pd.DataFrame(
        scaled_array,
        columns=feature_columns,
        index=df.index
    )

    logger.info("Feature scaling completed.")

    return scaled_df, scaler


def run_preprocessing_pipeline(
    raw_path=RAW_DATA_PATH,
    save_path=CLEAN_DATA_PATH
):

    # Step 1: Load data
    df = load_data(raw_path)

    original_rows = len(df)

    # Step 2: Generate original data quality report
    generate_quality_report(df)

    # Step 3: Validate data
    df = validate_customer_data(df)

    # Step 4: Handle missing values
    df = handle_missing_values(df)

    # Step 5: Remove duplicate records
    df = remove_duplicates(df)

    # Step 6: Encode categorical data
    df = encode_categoricals(df)

    # Step 7: Save cleaned data
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df.to_csv(save_path, index=False)

    logger.info(
        "Cleaned dataset saved: %s",
        save_path
    )

    print("\nPREPROCESSING SUMMARY")
    print("---------------------")
    print("Original rows:", original_rows)
    print("Final rows:", len(df))
    print("Rows removed:", original_rows - len(df))
    print("Missing values remaining:", df.isnull().sum().sum())

    return df


if __name__ == "__main__":

    cleaned = run_preprocessing_pipeline()

    print("\nCLEANED DATA PREVIEW")
    print(cleaned.head())
