"""
score_customers.py
------------------
A small example showing that a saved model can be reused on new data without
retraining. Each saved file is a full Pipeline (preprocessing + classifier),
so we can hand it the raw columns and it does the scaling/encoding itself.

Run:
    python score_customers.py
"""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

ROOT = Path(__file__).resolve().parent.parent

# Pick which saved model to load. Random Forest was our best by AUC.
MODEL_TAG = "rforest"   # options: logreg, dtree, knn, gnb, rforest

# 1) Load the held-out customers (original columns + the true label).
holdout = pd.read_csv(ROOT / "test_data.csv")
true_label = holdout["Churn"]
inputs = holdout.drop(columns=["Churn"])

# 2) Load the chosen pipeline.
pipe = joblib.load(ROOT / "model" / f"clf_{MODEL_TAG}.joblib")
print(f"Loaded model: clf_{MODEL_TAG}.joblib")

# 3) Predict labels and churn probabilities.
guess = pipe.predict(inputs)
prob = pipe.predict_proba(inputs)[:, 1]

# 4) Report quality on these unseen customers.
print(f"Accuracy : {accuracy_score(true_label, guess):.4f}")
print(f"AUC      : {roc_auc_score(true_label, prob):.4f}")
print(f"F1       : {f1_score(true_label, guess):.4f}")

# 5) Peek at the first few predictions vs. the truth.
peek = pd.DataFrame({
    "actual_churn": true_label.values,
    "predicted_churn": guess,
    "churn_probability": prob.round(3),
})
print("\nFirst 5 predictions:")
print(peek.head().to_string(index=False))
