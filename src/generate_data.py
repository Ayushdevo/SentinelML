import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification


RANDOM_STATE = 42
DATA_DIR = Path("data")


def build_dataset(n_samples=10000):
    if not isinstance(n_samples, int) or n_samples < 10:
        raise ValueError("n_samples must be an integer of at least 10")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_redundant=1,
        n_clusters_per_class=2,
        weights=[0.72, 0.28],
        class_sep=1.2,
        random_state=RANDOM_STATE,
    )

    columns = [
        "tenure",
        "monthly_charges",
        "usage_hours",
        "support_calls",
        "late_payments",
        "contract_score",
        "engagement_score",
        "service_count",
    ]

    df = pd.DataFrame(X, columns=columns)

    # Convert synthetic features into more interpretable ranges.
    df["tenure"] = np.clip(
        36 + df["tenure"] * 15, 1, 72
    ).round(0)

    df["monthly_charges"] = np.clip(
        70 + df["monthly_charges"] * 20, 15, 160
    ).round(2)

    df["usage_hours"] = np.clip(
        45 + df["usage_hours"] * 12, 0, 100
    ).round(2)

    df["support_calls"] = np.clip(
        2 + df["support_calls"] * 1.5, 0, 10
    ).round(0)

    df["late_payments"] = np.clip(
        1 + df["late_payments"], 0, 8
    ).round(0)

    df["contract_score"] = np.clip(
        50 + df["contract_score"] * 15, 0, 100
    ).round(2)

    df["engagement_score"] = np.clip(
        60 + df["engagement_score"] * 15, 0, 100
    ).round(2)

    df["service_count"] = np.clip(
        3 + df["service_count"], 1, 8
    ).round(0)

    df["churn"] = y

    return df


def create_production_data(reference):
    """
    Simulate a future production population where customer
    behaviour has changed.
    """

    if reference.empty:
        raise ValueError("Cannot simulate production data from an empty reference frame")
    production = reference.sample(
        n=2500,
        replace=True,
        random_state=100,
    ).copy()

    rng = np.random.default_rng(100)

    # Intentional distribution shifts.
    production["monthly_charges"] += rng.normal(
        loc=18,
        scale=7,
        size=len(production),
    )

    production["usage_hours"] *= rng.normal(
        loc=0.78,
        scale=0.08,
        size=len(production),
    )

    production["support_calls"] += rng.poisson(
        lam=1.2,
        size=len(production),
    )

    production["engagement_score"] -= rng.normal(
        loc=12,
        scale=5,
        size=len(production),
    )

    return production


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    dataset = build_dataset()

    reference = dataset.iloc[:7500].copy()
    production = create_production_data(dataset.iloc[7500:])

    reference.to_csv(DATA_DIR / "reference.csv", index=False)
    production.to_csv(DATA_DIR / "production.csv", index=False)

    print("SentinelML datasets generated.")
    print(f"Reference samples:  {len(reference)}")
    print(f"Production samples: {len(production)}")
    print(f"Churn rate: {reference['churn'].mean():.2%}")
