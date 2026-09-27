import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


DATA_PATH = Path("data/reference.csv")
MODEL_DIR = Path("models")

TARGET = "churn"


def train():
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df:
        raise ValueError(f"Training data is missing target column: {TARGET}")
    if df[TARGET].isna().any() or set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Training target must contain both binary classes without missing values")
    if df.empty or df.isna().any().any():
        raise ValueError("Training features must contain nonempty, complete data")
    if df[TARGET].value_counts().min() < 2 or len(df) < 10:
        raise ValueError("Training requires at least 10 rows and two examples per class")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
    }

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(model, MODEL_DIR / "baseline_model.joblib")
    model.get_booster().save_model(
        MODEL_DIR / "baseline_model.json"
    )
    with open(
        MODEL_DIR / "baseline_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\nSentinelML baseline model trained\n")

    for metric, value in metrics.items():
        print(f"{metric:10s}: {value:.4f}")

    print("\nSaved → models/baseline_model.joblib")


if __name__ == "__main__":
    train()
