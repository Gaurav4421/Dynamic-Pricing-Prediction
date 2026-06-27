# feature_engineering.py
# time feature extraction, column selection, and label encoding

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os


FEATURE_COLUMNS = [
    'cab_type',         # Uber or Lyft
    'name',             # Specific ride type (UberX, Lyft XL, etc.)
    'distance',         # Trip distance in miles
    'surge_multiplier', # Demand-based surge pricing multiplier
    'source',           # Pickup location
    'destination',      # Drop-off location
    'hour',             # Hour of day (extracted from timestamp)
    'day_of_week',      # Day of week (extracted from timestamp)
    'is_rush_hour',     # 1 if weekday rush hour, else 0
    'is_weekend',       # 1 if Saturday or Sunday, else 0
]

CATEGORICAL_COLUMNS = ['cab_type', 'name', 'source', 'destination']


def extract_time_features(df):
    """Create time-based features from the Unix timestamp."""

    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

    df['hour']        = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek  # 0 = Monday, 6 = Sunday

    # Weekday mornings (7-9 AM) and evenings (5-7 PM)
    df['is_rush_hour'] = (
        (df['day_of_week'] < 5) &
        ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19)))
    ).astype(int)

    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    print("Time features created.")
    return df


def select_features(df, target_col='price'):
    """Keep only the relevant feature columns and drop rows with nulls."""

    cols_to_keep = FEATURE_COLUMNS + [target_col]

    # Only keep columns that exist in the dataframe
    cols_available = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_available].copy()

    before = len(df)
    df = df.dropna()
    after = len(df)

    print(f"Selected {len(cols_available)} columns. Dropped {before - after:,} rows with nulls.")
    return df


def encode_categoricals(df, save_encoders_path=None):
    """Label encode categorical columns. Saves encoders if path is given."""

    label_encoders = {}

    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            print(f"Warning: '{col}' not found, skipping.")
            continue

        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

        print(f"Encoded '{col}' — {len(le.classes_)} classes")

    if save_encoders_path:
        joblib.dump(label_encoders, save_encoders_path)
        print(f"Encoders saved to {save_encoders_path}")

    return df, label_encoders


def run_feature_engineering(df, save_encoders_path=None):
    """Run the complete feature engineering pipeline."""

    df = extract_time_features(df)
    df = select_features(df)
    df, label_encoders = encode_categoricals(df, save_encoders_path)

    print(f"Feature engineering done. Shape: {df.shape}")
    return df, label_encoders


if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preprocessing import run_preprocessing

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'rideshare_kaggle.csv')

    if os.path.exists(data_path):
        df_clean = run_preprocessing(data_path)
        df_engineered, encoders = run_feature_engineering(df_clean)

        print("\nSample of engineered data:")
        print(df_engineered.head())
        print("\nFinal columns:", list(df_engineered.columns))
    else:
        print(f"Data file not found at: {data_path}")
        print("Place rideshare_kaggle.csv in the data/ folder and try again.")
