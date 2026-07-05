"""
preprocessing.py
-----------------
Data-cleaning and feature-preparation pipeline for the Customer
Segmentation project.

Steps performed:
    1. Load the raw CSV file.
    2. Handle missing values (median imputation for numeric columns).
    3. Remove duplicate records.
    4. Encode categorical variables (Gender -> 0/1).
    5. Scale numeric features with StandardScaler for clustering.

The module can be run directly:
    python src/preprocessing.py
which will read data/mall_customers.csv, clean it, and write the
cleaned file to data/mall_customers_clean.csv.
"""

from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils import RAW_DATA_PATH, CLEAN_DATA_PATH, get_logger

logger = get_logger(__name__)

FEATURE_COLUMNS = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded data with shape %s", df.shape)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing numeric values using the column median.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values imputed.
    """
    missing_before = df.isna().sum().sum()
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            logger.info("Filled %d missing value(s) in '%s' with median=%.2f",
                        df[col].isna().sum(), col, median_value)
    logger.info("Total missing values handled: %d", missing_before)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows (ignoring CustomerID) and reset the index.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    before = len(df)
    check_cols = [c for c in df.columns if c != "CustomerID"]
    df = df.drop_duplicates(subset=check_cols).reset_index(drop=True)
    removed = before - len(df)
    logger.info("Removed %d duplicate row(s)", removed)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the categorical 'Gender' column as 0/1.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with an added 'Gender_Encoded' column.
    """
    if "Gender" in df.columns:
        encoder = LabelEncoder()
        df["Gender_Encoded"] = encoder.fit_transform(df["Gender"])
        logger.info("Encoded Gender categories: %s",
                     dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))
    return df


def scale_features(df: pd.DataFrame,
                    feature_columns=None) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize numeric feature columns (mean=0, std=1).

    Parameters
    ----------
    df : pd.DataFrame
    feature_columns : list, optional
        Columns to scale. Defaults to FEATURE_COLUMNS.

    Returns
    -------
    Tuple[pd.DataFrame, StandardScaler]
        A DataFrame of scaled features and the fitted scaler
        (needed later to inverse-transform cluster centers).
    """
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[feature_columns])
    scaled_df = pd.DataFrame(scaled_array, columns=feature_columns, index=df.index)
    logger.info("Scaled features: %s", feature_columns)
    return scaled_df, scaler


def run_preprocessing_pipeline(raw_path: str = RAW_DATA_PATH,
                                save_path: str = CLEAN_DATA_PATH) -> pd.DataFrame:
    """Execute the full preprocessing pipeline end to end.

    Parameters
    ----------
    raw_path : str
        Path to the raw CSV file.
    save_path : str
        Path where the cleaned CSV file should be written.

    Returns
    -------
    pd.DataFrame
        The cleaned (but not scaled) DataFrame.
    """
    df = load_data(raw_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = encode_categoricals(df)
    df.to_csv(save_path, index=False)
    logger.info("Saved cleaned data to %s (shape=%s)", save_path, df.shape)
    return df


if __name__ == "__main__":
    cleaned = run_preprocessing_pipeline()
    print(cleaned.head())
