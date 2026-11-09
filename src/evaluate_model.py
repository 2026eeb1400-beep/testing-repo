import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from config import METRICS_DIR, FIGURE_DIR

def main():
    predictions_path = METRICS_DIR / "test_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError("Run `python src/train_model.py` first.")
    results = pd.read_csv(predictions_path)
    y_true = results["actual"]
    y_pred = results["predicted"]

    print("Classification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Survived", "Died"], cmap="Blues"
    )
    plt.title("Heart Failure Prediction - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    plt.figure()
    plt.scatter(results["time"], results["risk_score"], c=y_true)
    plt.axhline(0.5, color="black")
    plt.xlabel("Follow-up time (scaled)")
    plt.ylabel("Risk score")
    plt.title("Test patients: risk score")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "testing_risk_scores.png", dpi=150)
    plt.close()

    print("Evaluation figures saved in:", FIGURE_DIR)

if __name__ == "__main__":
    main()
