"""
Train and evaluate different models, tune the best performing one, and save the final outputs.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import run_preprocessing
from feature_engineering import run_feature_engineering
from evaluate import compute_metrics, print_results_table

ROOT_DIR        = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH       = os.path.join(ROOT_DIR, 'data', 'rideshare_kaggle.csv')
MODEL_DIR       = os.path.join(ROOT_DIR, 'model')
MODEL_PATH      = os.path.join(MODEL_DIR, 'best_model.pkl')
ENCODERS_PATH   = os.path.join(MODEL_DIR, 'label_encoders.pkl')
FEAT_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names.pkl')

RANDOM_SEED = 42


def get_models():
    """
    Return a dictionary of model instances with default parameters.
    """
    return {
        'Linear Regression': LinearRegression(),
        'Decision Tree':     DecisionTreeRegressor(max_depth=10, random_state=RANDOM_SEED),
        'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
        'XGBoost':           XGBRegressor(n_estimators=100, learning_rate=0.1,
                                          random_state=RANDOM_SEED, verbosity=0),
    }


def train_all_models(models, X_train, y_train):
    """
    Train each model on the training dataset.
    """
    trained = {}

    for name, model in models.items():
        print(f"Training {name}...", end=' ', flush=True)
        model.fit(X_train, y_train)
        trained[name] = model
        print("done.")

    return trained


def compare_models(trained_models, X_test, y_test):
    """
    Evaluate trained models on test data and sort by R2 score.
    """
    rows = []

    for name, model in trained_models.items():
        metrics = compute_metrics(model, X_test, y_test, name)
        rows.append(metrics)

    results_df = pd.DataFrame(rows).sort_values('R²', ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]['Model']

    return results_df, best_model_name


def tune_best_model(trained_models, best_model_name, X_train, y_train):
    """
    Tune the best performing model using RandomizedSearchCV.
    """
    print(f"\nRunning hyperparameter tuning for: {best_model_name}")

    if 'XGBoost' in best_model_name:
        base_model = XGBRegressor(random_state=RANDOM_SEED, verbosity=0)
        param_grid = {
            'n_estimators':     [100, 200, 300],
            'max_depth':        [3, 5, 7, 9],
            'learning_rate':    [0.01, 0.05, 0.1, 0.2],
            'subsample':        [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'min_child_weight': [1, 3, 5],
        }

    elif 'Random Forest' in best_model_name:
        base_model = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth':    [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf':  [1, 2, 4],
        }

    else:
        print(f"Skipping tuning. No parameter grid defined for {best_model_name}")
        return trained_models[best_model_name]

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=20,
        scoring='r2',
        cv=3,
        verbose=1,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print(f"Best parameters: {search.best_params_}")
    print(f"Best cross-validation R²: {search.best_score_:.4f}")

    return search.best_estimator_


def save_artifacts(model, label_encoders, feature_names):
    """
    Save the model and preprocessing files to the model folder.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model,          MODEL_PATH)
    joblib.dump(label_encoders, ENCODERS_PATH)
    joblib.dump(feature_names,  FEAT_NAMES_PATH)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved encoders to {ENCODERS_PATH}")
    print(f"Saved feature names to {FEAT_NAMES_PATH}")


def main():
    print("=" * 50)
    print("Training Pipeline")
    print("=" * 50)

    if not os.path.exists(DATA_PATH):
        print(f"\nError: Data file not found at {DATA_PATH}")
        sys.exit(1)

    print("\nLoading data...")
    df = run_preprocessing(DATA_PATH)

    print("\nRunning feature engineering...")
    df, label_encoders = run_feature_engineering(df, save_encoders_path=ENCODERS_PATH)

    print("\nSplitting data...")
    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"Train size: {X_train.shape} | Test size: {X_test.shape}")

    print("\nTraining models...")
    models = get_models()
    trained_models = train_all_models(models, X_train, y_train)

    print("\nComparing models...")
    results_df, best_model_name = compare_models(trained_models, X_test, y_test)
    print_results_table(results_df)
    print(f"\nBest baseline model: {best_model_name}")

    print("\nHyperparameter tuning...")
    tuned_model = tune_best_model(trained_models, best_model_name, X_train, y_train)

    tuned_metrics = compute_metrics(tuned_model, X_test, y_test, f"{best_model_name} (Tuned)")
    print(f"\nTuned model metrics:")
    print(f"MAE:  {tuned_metrics['MAE']}")
    print(f"RMSE: {tuned_metrics['RMSE']}")
    print(f"R²:   {tuned_metrics['R²']}")

    feature_names = list(X.columns)
    save_artifacts(tuned_model, label_encoders, feature_names)

    print("\n" + "=" * 50)
    print("Training complete.")
    print("=" * 50)


if __name__ == '__main__':
    main()

