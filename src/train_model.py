import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix

from config import DATA_PATH, MODEL_PATH, FIGURE_DIR, METRICS_DIR, TARGET, SELECTED_FEATURES
from data_preprocessing import load_data, inspect_data, scale_features, select_features

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(DATA_PATH)
    inspect_data(df)

    X_scaled, means, stds = scale_features(df)
    y = df[TARGET]
    gap_table, selected = select_features(X_scaled, y)
    print("\\nFeature-selection table:")
    print(selected.round(3))
    features = [f for f in SELECTED_FEATURES if f in selected.index]
    if not features:
        features = SELECTED_FEATURES
    print("\\nFeatures used:", features)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )
    model = LinearRegression()
    model.fit(X_train[features], y_train)

    train_risk = model.predict(X_train[features])
    test_risk = model.predict(X_test[features])
    threshold = 0.5
    train_pred = (train_risk >= threshold).astype(int)
    test_pred = (test_risk >= threshold).astype(int)

    # Save the complete inference bundle.
    bundle = {
        "model": model, "features": features, "means": means,
        "stds": stds, "threshold": threshold,
    }
    joblib.dump(bundle, MODEL_PATH)

    weights = pd.DataFrame({"measurement": features, "weight": model.coef_})
    weights.to_csv(METRICS_DIR / "model_weights.csv", index=False)

    metrics = {
        "train_mae": mean_absolute_error(y_train, train_risk),
        "test_mae": mean_absolute_error(y_test, test_risk),
        "train_rmse": mean_squared_error(y_train, train_risk) ** 0.5,
        "test_rmse": mean_squared_error(y_test, test_risk) ** 0.5,
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_precision": None,
        "test_recall": None,
        "test_f1": None,
    }
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["test_precision"] = tp / (tp + fp) if tp + fp else 0.0
    metrics["test_recall"] = tp / (tp + fn) if tp + fn else 0.0
    pr, re = metrics["test_precision"], metrics["test_recall"]
    metrics["test_f1"] = 2 * pr * re / (pr + re) if pr + re else 0.0
    pd.DataFrame([metrics]).to_csv(METRICS_DIR / "model_metrics.csv", index=False)
    pd.DataFrame(cm, index=["actual_survived","actual_died"],
                 columns=["predicted_survived","predicted_died"]).to_csv(
                     METRICS_DIR / "confusion_matrix.csv")

    # Store a fixed test split for repeatable evaluation and plots.
    test_results = X_test[features].copy()
    test_results["actual"] = y_test
    test_results["risk_score"] = test_risk
    test_results["predicted"] = test_pred
    test_results.to_csv(METRICS_DIR / "test_predictions.csv", index=False)

    plt.figure()
    plt.scatter(X_train["time"], train_risk, c=y_train)
    plt.axhline(threshold, color="black")
    plt.xlabel("Follow-up time (scaled)")
    plt.ylabel("Risk score")
    plt.title("Training patients: risk score")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "training_risk_scores.png", dpi=150)
    plt.close()

    plt.figure()
    plt.scatter(X_train["time"], X_train["ejection_fraction"], c=train_risk)
    plt.xlabel("Follow-up time (scaled)")
    plt.ylabel("Ejection fraction (scaled)")
    plt.title("Training patients: colour = risk score")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "training_risk_map.png", dpi=150)
    plt.close()

    print("\\nModel saved to:", MODEL_PATH)
    print(pd.DataFrame([metrics]).round(3).to_string(index=False))

if __name__ == "__main__":
    main()
