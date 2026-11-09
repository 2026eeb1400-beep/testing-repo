import argparse
import joblib
import pandas as pd
from config import MODEL_PATH, ALL_FEATURES

def main():
    parser = argparse.ArgumentParser(description="Predict heart-failure death-event risk.")
    parser.add_argument("csv", help="CSV file containing one or more patient records")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run `python src/train_model.py` first.")
    bundle = joblib.load(MODEL_PATH)
    data = pd.read_csv(args.csv)

    missing = set(ALL_FEATURES) - set(data.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    scaled = (data[ALL_FEATURES] - bundle["means"]) / bundle["stds"]
    risk = bundle["model"].predict(scaled[bundle["features"]])
    predicted = (risk >= bundle["threshold"]).astype(int)

    output = pd.DataFrame({
        "risk_score": risk,
        "predicted_DEATH_EVENT": predicted,
        "predicted_outcome": ["Died" if value else "Survived" for value in predicted],
    })
    print(output.to_string(index=False))

if __name__ == "__main__":
    main()
