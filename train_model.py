"""
train_model.py
----------------
Generates a synthetic (but realistic-looking) credit-card-transaction
dataset, trains a RandomForest fraud-detection model, and saves the
model + scaler + a held-out TEST SET to disk.

Run this ONCE, before starting the Flask app:

    python train_model.py

It creates a `model/` folder containing:
    fraud_model.pkl   -> trained classifier
    scaler.pkl        -> StandardScaler fitted on the training data
    X_test.pkl        -> held-out test features (already scaled)
    y_test.pkl        -> held-out test labels
    feature_cols.pkl  -> ordered list of feature names the model expects

app.py loads these artifacts and uses the SAME X_test / y_test every
time it recomputes accuracy / recall / confusion matrix for a new
threshold -- nothing about those metrics is hard-coded.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

RANDOM_STATE = 42
N_SAMPLES = 20000
FRAUD_RATIO = 0.08          # ~8% fraud, mimics real-world class imbalance

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def generate_dataset(n_samples=N_SAMPLES, fraud_ratio=FRAUD_RATIO, seed=RANDOM_STATE):
    """
    Creates a synthetic dataset with 7 interpretable features (similar in
    spirit to the well-known 'Credit Card Fraud' synthetic dataset used in
    many student projects), instead of the anonymized PCA columns (V1..V28)
    found in the original Kaggle dataset -- these are much easier for a
    user to fill in through an HTML form.
    """
    rng = np.random.default_rng(seed)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # ---------------- Legitimate transactions ----------------
    legit = pd.DataFrame({
        "distance_from_home": rng.exponential(scale=10, size=n_legit),
        "distance_from_last_transaction": rng.exponential(scale=5, size=n_legit),
        "ratio_to_median_purchase_price": rng.normal(loc=1.0, scale=0.5, size=n_legit).clip(0.01),
        "repeat_retailer": rng.choice([0, 1], size=n_legit, p=[0.15, 0.85]),
        "used_chip": rng.choice([0, 1], size=n_legit, p=[0.3, 0.7]),
        "used_pin_number": rng.choice([0, 1], size=n_legit, p=[0.4, 0.6]),
        "online_order": rng.choice([0, 1], size=n_legit, p=[0.7, 0.3]),
    })
    legit["Class"] = 0

    # ---------------- Fraudulent transactions -----------------
    fraud = pd.DataFrame({
        "distance_from_home": rng.exponential(scale=60, size=n_fraud),
        "distance_from_last_transaction": rng.exponential(scale=40, size=n_fraud),
        "ratio_to_median_purchase_price": rng.normal(loc=4.0, scale=2.0, size=n_fraud).clip(0.01),
        "repeat_retailer": rng.choice([0, 1], size=n_fraud, p=[0.7, 0.3]),
        "used_chip": rng.choice([0, 1], size=n_fraud, p=[0.85, 0.15]),
        "used_pin_number": rng.choice([0, 1], size=n_fraud, p=[0.9, 0.1]),
        "online_order": rng.choice([0, 1], size=n_fraud, p=[0.15, 0.85]),
    })
    fraud["Class"] = 1

    data = pd.concat([legit, fraud], ignore_index=True)
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    return data


def main():
    print("Generating synthetic credit card transaction dataset...")
    data = generate_dataset()

    feature_cols = [
        "distance_from_home",
        "distance_from_last_transaction",
        "ratio_to_median_purchase_price",
        "repeat_retailer",
        "used_chip",
        "used_pin_number",
        "online_order",
    ]

    X = data[feature_cols]
    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)

    # Quick sanity check at the default threshold of 0.5
    probs = model.predict_proba(X_test_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)
    print("Accuracy @0.5:", accuracy_score(y_test, preds))
    print("Recall   @0.5:", recall_score(y_test, preds))
    print("Confusion matrix @0.5:\n", confusion_matrix(y_test, preds))

    joblib.dump(model, os.path.join(MODEL_DIR, "fraud_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(X_test_scaled, os.path.join(MODEL_DIR, "X_test.pkl"))
    joblib.dump(y_test.reset_index(drop=True), os.path.join(MODEL_DIR, "y_test.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.pkl"))

    print(f"\nSaved model artifacts to: {MODEL_DIR}")
    print("You can now run: python app.py")


if __name__ == "__main__":
    main()
