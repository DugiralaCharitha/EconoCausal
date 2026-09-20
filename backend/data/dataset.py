from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).parent / "customer_data.csv"


def load_customer_data() -> pd.DataFrame:
    """Load the generated customer dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Run generate_data.py first."
        )

    return pd.read_csv(DATA_PATH)


def validate_customer_data(df: pd.DataFrame) -> None:
    """Validate the required columns and basic data constraints."""

    required_columns = {
        "customer_id",
        "age",
        "income",
        "past_purchases",
        "website_visits",
        "customer_segment",
        "previous_discount",
        "discount",
        "purchase",
        "revenue",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs found.")

    if not df["age"].between(18, 70).all():
        raise ValueError("Age contains invalid values.")

    if not df["discount"].isin([0, 5, 10, 20]).all():
        raise ValueError("Discount contains invalid values.")

    if not df["purchase"].isin([0, 1]).all():
        raise ValueError("Purchase must contain only 0 or 1.")

    if (df["revenue"] < 0).any():
        raise ValueError("Revenue cannot be negative.")


def prepare_causal_data(df: pd.DataFrame):
    """Prepare features, treatment, and outcome for causal ML."""

    feature_columns = [
        "age",
        "income",
        "past_purchases",
        "website_visits",
        "customer_segment",
        "previous_discount",
    ]

    X = df[feature_columns].copy()
    T = df["discount"].copy()
    Y = df["purchase"].copy()

    X = pd.get_dummies(
        X,
        columns=["customer_segment"],
        dtype=int,
    )

    return X, T, Y
if __name__ == "__main__":
    customer_data = load_customer_data()

    validate_customer_data(customer_data)

    X, T, Y = prepare_causal_data(customer_data)

    print("Dataset validation successful.")
    print(f"Rows: {len(customer_data)}")
    print(f"Feature shape: {X.shape}")
    print(f"Treatment shape: {T.shape}")
    print(f"Outcome shape: {Y.shape}")
    print("\nFeatures:")
    print(X.head())