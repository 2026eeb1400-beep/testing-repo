from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "heart_failure_clinical_records_dataset.csv"
MODEL_PATH = ROOT_DIR / "models" / "heart_failure_model.joblib"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"

TARGET = "DEATH_EVENT"
ALL_FEATURES = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking", "time",
]
SELECTED_FEATURES = [
    "age", "ejection_fraction", "serum_creatinine", "serum_sodium", "time"
]
