import pandas as pd
from sklearn.model_selection import train_test_split
from config import ALL_FEATURES, TARGET

def load_data(path):
    """Load the heart-failure clinical records CSV."""
    df = pd.read_csv(path)
    missing = set(ALL_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df

def inspect_data(df):
    print("Rows and columns:", df.shape)
    print("\\nMissing values:")
    print(df.isnull().sum())
    print("\\nPatients who died:", int(df[TARGET].sum()))
    print("Patients who survived:", len(df) - int(df[TARGET].sum()))
    print("\\nSummary statistics:")
    print(df.describe())

def scale_features(df):
    """Standardize the 12 input features using dataset mean and standard deviation."""
    X = df[ALL_FEATURES].copy()
    means = X.mean()
    stds = X.std().replace(0, 1)
    return (X - means) / stds, means, stds

def select_features(X_scaled, y, cutoff=0.10):
    """Select the five measurements used in the original notebook."""
    died = X_scaled[y == 1]
    survived = X_scaled[y == 0]
    p = y.mean()
    factor = (p * (1 - p)) ** 0.5
    gap_table = pd.DataFrame({
        "survived (average)": survived.mean(),
        "died (average)": died.mean(),
        "gap": died.mean() - survived.mean(),
    })
    gap_table["correlation (approx.)"] = gap_table["gap"] * factor
    r = gap_table["correlation (approx.)"]
    selected = gap_table[(r >= cutoff) | (r <= -cutoff)]
    return gap_table, selected

print("Data Processed.")