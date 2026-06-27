import pandas as pd
import numpy as np


def load_data(filepath):
    """
    Load the raw rideshare CSV dataset into a DataFrame.
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def drop_missing_target(df, target_col='price'):
    """
    Drop rows where the target price variable is missing.
    """
    before = len(df)
    df = df.dropna(subset=[target_col])
    after = len(df)
    print(f"Dropped {before - after} rows with null target values.")
    return df


def remove_duplicates(df):
    """
    Remove fully duplicate rows from the dataset.
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before - after} duplicate rows.")
    return df


def cap_price_outliers(df, target_col='price'):
    """
    Cap price outliers using the IQR method boundaries.
    """
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_count = ((df[target_col] < lower) | (df[target_col] > upper)).sum()
    df[target_col] = df[target_col].clip(lower=lower, upper=upper)

    print(f"Capped {outlier_count} outliers in '{target_col}' to range [${lower:.2f}, ${upper:.2f}].")
    return df


def run_preprocessing(filepath):
    """
    Run the full data cleaning and preprocessing pipeline.
    """
    df = load_data(filepath)
    df = drop_missing_target(df)
    df = remove_duplicates(df)
    df = cap_price_outliers(df)

    print(f"Preprocessing complete. Final data shape: {df.shape}")
    return df


if __name__ == '__main__':
    import os

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'rideshare_kaggle.csv')
    if os.path.exists(data_path):
        run_preprocessing(data_path)
        
