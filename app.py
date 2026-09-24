"""
app.py
-------
Flask backend for the Credit Card Fraud Detection web app.

Run:
    python app.py

Then open:
    http://127.0.0.1:5000/

Requires that `python train_model.py` has already been run once, so that
the `model/` folder contains the trained model, scaler and held-out
test set.
"""

import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR

app = Flask(__name__,
template_folder=BASE_DIR,
static_folder=BASE_DIR,
static_url_path="")

# ---------------------------------------------------------------------
# Load the trained model, scaler and held-out test set ONCE at startup
# ---------------------------------------------------------------------
model_path = os.path.join(MODEL_DIR, "fraud_model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(
        "Model artifacts not found. Please run `python train_model.py` "
        "first — it creates the 'model' folder used by this app."
    )

model = joblib.load(model_path)
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
X_test = joblib.load(os.path.join(MODEL_DIR, "X_test.pkl"))     # already scaled
y_test = joblib.load(os.path.join(MODEL_DIR, "y_test.pkl"))
FEATURE_COLS = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))

# Predicted fraud-probability for every transaction in the test set,
# computed ONCE. Every threshold change just re-thresholds these same
# probabilities — no retraining, nothing hard-coded.
TEST_PROBS = model.predict_proba(X_test)[:, 1]


@app.route("/")
def home():
    """Render the main page with an input field for every model feature."""
    return render_template("index.html", feature_cols=FEATURE_COLS)


@app.route("/predict", methods=["POST"])
def predict():
    """Predict a single, user-entered transaction at the chosen threshold."""
    payload = request.get_json(silent=True) or {}

    try:
        threshold = float(payload.get("threshold", 0.5))
        features = [float(payload[col]) for col in FEATURE_COLS]
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Please fill in every field with a valid number."}), 400

    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    probability = float(model.predict_proba(X_scaled)[0, 1])
    prediction = int(probability >= threshold)          # <-- required rule

    return jsonify({
        "prediction": "Fraud" if prediction == 1 else "Legitimate",
        "probability": round(probability, 4),
        "threshold_used": threshold,
    })


@app.route("/metrics", methods=["POST"])
def metrics():
    """
    Recompute accuracy, recall and the confusion matrix on the held-out
    TEST SET for whatever threshold the user currently has selected.
    """
    payload = request.get_json(silent=True) or {}
    try:
        threshold = float(payload.get("threshold", 0.5))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid threshold."}), 400

    if not (0.0 <= threshold <= 1.0):
        return jsonify({"error": "Threshold must be between 0 and 1."}), 400

    predictions = (TEST_PROBS >= threshold).astype(int)     # <-- required rule

    acc = accuracy_score(y_test, predictions)
    rec = recall_score(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return jsonify({
        "threshold": threshold,
        "accuracy": round(float(acc), 4),
        "recall": round(float(rec), 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
        "test_set_size": int(len(y_test)),
    })


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
