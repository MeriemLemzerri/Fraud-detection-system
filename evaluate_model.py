"""
Black-box evaluation of the deployed API.

Sends the labelled `sample_test.csv` (produced by train_pipeline.py) to a
running backend instance and reports the same classification metrics an
analyst would see -- useful as a smoke test after deployment, or to compare
against the offline numbers in assets/metrics.json.

Usage:
    python evaluate_model.py [--url http://localhost:8000]
"""
import argparse
import sys

import pandas as pd
import requests
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

FEATURES = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000", help="Backend base URL")
    parser.add_argument("--data", default="sample_test.csv", help="Labelled CSV to audit")
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        print(f"Could not find {args.data}. Run `python train_pipeline.py` first.")
        sys.exit(1)

    y_true = df["Class"].values
    X = df[FEATURES].values.tolist()

    print(f"Auditing {len(X):,} transactions against {args.url} ...")
    y_pred, y_prob = [], []
    for start in range(0, len(X), args.batch_size):
        batch = X[start:start + args.batch_size]
        try:
            resp = requests.post(f"{args.url}/api/v1/audit", json={"features": batch, "is_raw": False}, timeout=120)
        except requests.exceptions.ConnectionError:
            print(f"Could not reach {args.url}. Is the backend running? "
                  f"(uvicorn backend.main:app --host 0.0.0.0 --port 8000)")
            sys.exit(1)

        if resp.status_code != 200:
            print(f"API error [{resp.status_code}]: {resp.text}")
            sys.exit(1)

        metrics = resp.json()["metrics"]
        y_pred.extend(1 if m["is_fraud"] else 0 for m in metrics)
        y_prob.extend(m["fraud_probability"] for m in metrics)

    print("\n--- HYBRID VAE + RANDOM FOREST — LIVE API AUDIT ---")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Fraud"]))

    cm = confusion_matrix(y_true, y_pred)
    print("--- CONFUSION MATRIX ---")
    print(f"True Negatives  (correctly cleared):  {cm[0][0]}")
    print(f"False Positives (false alarms):       {cm[0][1]}")
    print(f"False Negatives (missed fraud):       {cm[1][0]}")
    print(f"True Positives  (caught fraud):       {cm[1][1]}")

    print(f"\nROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_true, y_prob):.4f}")


if __name__ == "__main__":
    main()
