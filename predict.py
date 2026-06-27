"""
predict.py
----------
Inference layer for the Dynamic Pricing Prediction project.

Mirrors the exact preprocessing and feature engineering steps used during
training. Load once, call predict_price() for each new prediction request.

Depends on the three artifacts saved by train.py:
  model/best_model.pkl
  model/label_encoders.pkl
  model/feature_names.pkl
"""

import os
import numpy as np
import pandas as pd
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR        = os.path.join(os.path.dirname(__file__))
MODEL_DIR       = os.path.join(ROOT_DIR, 'model')
MODEL_PATH      = os.path.join(MODEL_DIR, 'best_model.pkl')
ENCODERS_PATH   = os.path.join(MODEL_DIR, 'label_encoders.pkl')
FEAT_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names.pkl')

# ── Valid values (mirrors what LabelEncoder saw at training time) ───────────────
VALID_CAB_TYPES = ['Uber', 'Lyft']

VALID_RIDE_NAMES = {
    'Uber': ['UberX', 'UberXL', 'UberBlack', 'UberPool', 'WAV'],
    'Lyft': ['Lyft', 'Lyft XL', 'Lyft Black', 'Lyft Black XL', 'Shared'],
}

VALID_LOCATIONS = [
    'Back Bay', 'Beacon Hill', 'Boston University', 'Fenway',
    'Financial District', 'Haymarket Square', 'North End',
    'North Station', 'Northeastern University', 'South Station',
    'Theatre District', 'West End',
]

# ── Lazy-loaded singletons ─────────────────────────────────────────────────────
_model          = None
_label_encoders = None
_feature_names  = None


def _load_artifacts():
    """Load model artifacts once and cache them globally."""
    global _model, _label_encoders, _feature_names

    if _model is not None:
        return  # already loaded

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Run train.py first to generate model artifacts."
        )

    _model          = joblib.load(MODEL_PATH)
    _label_encoders = joblib.load(ENCODERS_PATH)
    _feature_names  = joblib.load(FEAT_NAMES_PATH)


def _derive_time_features(hour: int, day_of_week: int) -> dict:
    """
    Reproduce the exact logic from feature_engineering.extract_time_features().

    Args:
        hour        : 0–23 (hour of day)
        day_of_week : 0 = Monday … 6 = Sunday

    Returns:
        dict with is_rush_hour and is_weekend flags
    """
    is_weekday   = day_of_week < 5
    morning_rush = 7 <= hour <= 9
    evening_rush = 17 <= hour <= 19
    is_rush_hour = int(is_weekday and (morning_rush or evening_rush))
    is_weekend   = int(day_of_week >= 5)

    return {
        'is_rush_hour': is_rush_hour,
        'is_weekend':   is_weekend,
    }


def _encode_categoricals(row: dict) -> dict:
    """
    Apply saved LabelEncoders to categorical fields.
    Reproduces encode_categoricals() from feature_engineering.py.

    Raises ValueError if an unseen label is encountered.
    """
    categorical_cols = ['cab_type', 'name', 'source', 'destination']

    for col in categorical_cols:
        le    = _label_encoders[col]
        value = str(row[col])

        if value not in le.classes_:
            raise ValueError(
                f"Unknown value '{value}' for column '{col}'. "
                f"Valid options: {list(le.classes_)}"
            )

        row[col] = int(le.transform([value])[0])

    return row


def predict_price(
    cab_type: str,
    ride_name: str,
    distance: float,
    surge_multiplier: float,
    source: str,
    destination: str,
    hour: int,
    day_of_week: int,
) -> float:
    """
    Predict the ride price for a given set of inputs.

    The pipeline exactly mirrors training:
      1. Derive time features (is_rush_hour, is_weekend)
      2. Label-encode categoricals using saved encoders
      3. Assemble a one-row DataFrame in the trained feature order
      4. Call model.predict()

    Args:
        cab_type         : 'Uber' or 'Lyft'
        ride_name        : e.g. 'UberX', 'Lyft XL'
        distance         : trip distance in miles (> 0)
        surge_multiplier : demand multiplier (>= 1.0)
        source           : pickup location name
        destination      : drop-off location name
        hour             : hour of day (0–23)
        day_of_week      : 0 (Monday) … 6 (Sunday)

    Returns:
        Predicted price in USD (float, rounded to 2 dp)

    Raises:
        FileNotFoundError : if model artifacts are missing
        ValueError        : if any input value is invalid
    """
    # Input validation
    if distance <= 0:
        raise ValueError("Distance must be greater than 0 miles.")
    if surge_multiplier < 1.0:
        raise ValueError("Surge multiplier must be at least 1.0.")
    if not (0 <= hour <= 23):
        raise ValueError("Hour must be between 0 and 23.")
    if not (0 <= day_of_week <= 6):
        raise ValueError("Day of week must be between 0 (Monday) and 6 (Sunday).")

    _load_artifacts()

    # Build raw feature dict
    time_feats = _derive_time_features(hour, day_of_week)

    raw = {
        'cab_type':         cab_type,
        'name':             ride_name,
        'distance':         distance,
        'surge_multiplier': surge_multiplier,
        'source':           source,
        'destination':      destination,
        'hour':             hour,
        'day_of_week':      day_of_week,
        'is_rush_hour':     time_feats['is_rush_hour'],
        'is_weekend':       time_feats['is_weekend'],
    }

    # Encode categoricals
    encoded = _encode_categoricals(raw)

    # Assemble DataFrame in the exact trained column order
    row_df = pd.DataFrame([encoded])[_feature_names]

    # Predict and clip to a sensible positive range
    price = float(_model.predict(row_df)[0])
    price = max(0.0, price)

    return round(price, 2)


def get_time_context(hour: int, day_of_week: int) -> dict:
    """
    Return a human-readable summary of the time context for display in the UI.

    Args:
        hour        : 0–23
        day_of_week : 0–6

    Returns:
        dict with is_rush_hour, is_weekend, label strings
    """
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    feats = _derive_time_features(hour, day_of_week)

    period = 'AM' if hour < 12 else 'PM'
    display_hour = hour if hour <= 12 else hour - 12
    display_hour = 12 if display_hour == 0 else display_hour

    return {
        'is_rush_hour': feats['is_rush_hour'],
        'is_weekend':   feats['is_weekend'],
        'day_name':     days[day_of_week],
        'time_str':     f"{display_hour}:00 {period}",
    }


if __name__ == '__main__':
    # Quick smoke test (requires trained model artifacts)
    try:
        price = predict_price(
            cab_type='Uber',
            ride_name='UberX',
            distance=2.5,
            surge_multiplier=1.0,
            source='Back Bay',
            destination='South Station',
            hour=8,
            day_of_week=0,   # Monday
        )
        print(f"Test prediction: ${price:.2f}")

        ctx = get_time_context(8, 0)
        print(f"Time context: {ctx}")

    except FileNotFoundError as e:
        print(f"[predict.py] {e}")
        print("Train the model first with: python src/train.py")
