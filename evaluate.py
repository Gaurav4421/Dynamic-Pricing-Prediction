# evaluate.py
# functions for computing metrics, plotting results, and SHAP explanations
# used by train.py and the notebook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(model, X_test, y_test, model_name):
    """Returns MAE, RMSE, R² as a dict."""
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    return {
        'Model': model_name,
        'MAE':   round(mae,  4),
        'RMSE':  round(rmse, 4),
        'R²':    round(r2,   4),
    }


def print_results_table(results_df):
    """Prints a formatted comparison table to the console."""
    print("\n" + "-" * 55)
    print(f"  {'Model':<25} {'MAE':>7} {'RMSE':>7} {'R²':>7}")
    print("-" * 55)

    for _, row in results_df.iterrows():
        print(f"  {row['Model']:<25} {row['MAE']:>7.4f} {row['RMSE']:>7.4f} {row['R²']:>7.4f}")

    print("-" * 55)


def plot_actual_vs_predicted(model, X_test, y_test, model_name='Best Model', sample_size=3000):
    """Scatter plot of actual vs predicted. Points near the diagonal = good predictions."""
    y_pred = model.predict(X_test)

    # sample so the plot isn't too heavy
    idx = np.random.choice(len(y_test), size=min(sample_size, len(y_test)), replace=False)
    actual    = np.array(y_test)[idx]
    predicted = y_pred[idx]

    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, alpha=0.3, s=10, color='steelblue')

    # ideal prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    plt.title(f'Actual vs Predicted Price — {model_name}')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=10):
    """Horizontal bar chart of top N feature importances. Tree models only."""
    if not hasattr(model, 'feature_importances_'):
        print("[feature importance] This model does not support feature_importances_.")
        return

    importance_df = pd.DataFrame({
        'Feature':    feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True).tail(top_n)

    plt.figure(figsize=(9, 5))
    plt.barh(importance_df['Feature'], importance_df['Importance'],
             color='steelblue', edgecolor='white')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_df):
    """Side-by-side bar charts for MAE, RMSE, and R² across all models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics = ['MAE', 'RMSE', 'R²']
    colors  = ['#e74c3c', '#e67e22', '#2ecc71']

    for ax, metric, color in zip(axes, metrics, colors):
        ax.barh(results_df['Model'], results_df[metric], color=color, edgecolor='white')
        ax.set_title(f'Model Comparison — {metric}')
        ax.set_xlabel(metric)
        ax.invert_yaxis()

    plt.suptitle('Model Comparison Across All Metrics', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_shap_summary(model, X_test, sample_size=500):
    """
    SHAP summary plot for the test set.
    Uses TreeExplainer (faster than KernelExplainer for tree models).
    Samples 500 rows by default to keep it fast.
    """
    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    print(f"[SHAP] Computing values for {len(X_sample)} samples...")
    shap.summary_plot(shap_values, X_sample, show=True)


def get_shap_values_for_row(model, row_df):
    """
    Returns SHAP values for a single row.
    Used by the Streamlit app for per-prediction explanations.
    """
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(row_df)
    base_val  = explainer.expected_value

    return shap_vals[0], base_val


if __name__ == '__main__':
    # quick smoke test with dummy data
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    y_sr = pd.Series(y, name='price')

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_df, y_sr)

    metrics = compute_metrics(model, X_df, y_sr, 'Test RF')
    print("Smoke test metrics:", metrics)

    results = pd.DataFrame([metrics])
    print_results_table(results)

    plot_feature_importance(model, list(X_df.columns))
    plot_actual_vs_predicted(model, X_df, y_sr, 'Test RF')

    print("\nAll evaluate.py functions ran successfully.")
