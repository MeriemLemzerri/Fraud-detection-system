"""
Training pipeline for the Hybrid VAE + Random Forest fraud detector.

Two-stage design:
  1. A Variational Autoencoder (VAE) is trained ONLY on legitimate transactions.
     It learns the manifold of "normal" spending behaviour. The reconstruction
     error it produces for any transaction is a strong unsupervised anomaly
     signal, and it generalizes to fraud patterns the labels never saw.
  2. A Random Forest classifier is trained in a supervised fashion on the raw
     transaction features PLUS the VAE reconstruction error as an engineered
     feature. This lets a tree ensemble exploit the (limited) labelled fraud
     examples while still benefiting from the anomaly signal the VAE learned
     from the much larger pool of unlabelled/normal data.

This mirrors how autoencoder + gradient-boosted-tree hybrids are used in
production fraud stacks: the unsupervised model supplies a robust anomaly
prior, the supervised model calibrates a decision boundary against known
fraud precisely.

Outputs written to backend/:
  - vae_weights.weights.h5   VAE decoder/encoder weights
  - scaler_amount.pkl        RobustScaler fit on Amount
  - scaler_time.pkl          RobustScaler fit on Time
  - hybrid_rf.pkl            Trained RandomForestClassifier
  - threshold.json           Calibrated decision threshold + metadata
  - feature_names.json       Ordered feature names expected by the API

Outputs written to assets/ (used by the README and evaluate_model.py):
  - roc_curve.png, pr_curve.png, confusion_matrix.png, error_distribution.png,
    feature_importance.png
  - metrics.json
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("backend", exist_ok=True)
os.makedirs("assets", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load & prepare data
# ---------------------------------------------------------------------------
print("[1/7] Loading transaction ledger...")
if not os.path.exists("creditcard.csv"):
    raise FileNotFoundError(
        "creditcard.csv not found. Download it from "
        "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place "
        "it in the repository root."
    )

df = pd.read_csv("creditcard.csv")
print(f"    {len(df):,} transactions | {df['Class'].sum():,} fraudulent "
      f"({100 * df['Class'].mean():.3f}%)")

scaler_amount = RobustScaler()
scaler_time = RobustScaler()
df["scaled_amount"] = scaler_amount.fit_transform(df[["Amount"]].values)
df["scaled_time"] = scaler_time.fit_transform(df[["Time"]].values)

v_features = [f"V{i}" for i in range(1, 29)]
ordered_features = v_features + ["scaled_amount", "scaled_time"]
FEATURE_DIM = len(ordered_features)

# Split BEFORE any fraud-aware step touches the data, stratified so the RF
# validation/test sets keep a realistic fraud ratio.
train_val, test = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["Class"]
)
train, val = train_test_split(
    train_val, test_size=0.2, random_state=SEED, stratify=train_val["Class"]
)
print(f"    train={len(train):,}  val={len(val):,}  test={len(test):,}")

# The VAE only ever sees NORMAL transactions from the training split, so it
# never memorizes fraud patterns -- it purely models "normal".
vae_train = train[train["Class"] == 0][ordered_features].values.astype(np.float32)

# Persist a labelled sample_test.csv for evaluate_model.py / manual poking.
test.to_csv("sample_test.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Build & train the VAE
# ---------------------------------------------------------------------------
print("[2/7] Training Variational Autoencoder on legitimate transactions only...")


class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        # Clip before exp() -- unclipped log-variance can explode early in
        # training (Amount has heavy-tailed outliers even after robust
        # scaling) and silently produces NaN losses.
        z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAELossLayer(Layer):
    def call(self, inputs):
        encoder_inputs, outputs, z_mean, z_log_var = inputs
        z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
        reconstruction_loss = tf.reduce_mean(tf.square(encoder_inputs - outputs), axis=-1)
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
        )
        self.add_loss(tf.reduce_mean(reconstruction_loss + 0.05 * kl_loss))
        return outputs


latent_dim = 4
encoder_inputs = Input(shape=(FEATURE_DIM,), name="encoder_input")
h = Dense(16, activation="relu", name="encoder_hidden")(encoder_inputs)
z_mean = Dense(latent_dim, name="z_mean")(h)
z_log_var = Dense(latent_dim, name="z_log_var")(h)
z = Sampling(name="reparameterization")([z_mean, z_log_var])

decoder_hidden = Dense(16, activation="relu", name="decoder_hidden")
decoder_output = Dense(FEATURE_DIM, activation="linear", name="decoder_output")
h_p = decoder_hidden(z)
outputs = decoder_output(h_p)

vae_outputs = VAELossLayer(name="vae_loss")([encoder_inputs, outputs, z_mean, z_log_var])
vae = Model(encoder_inputs, vae_outputs, name="vae")
vae.compile(optimizer="adam")

early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, min_delta=1e-4, restore_best_weights=True
)
vae.fit(
    vae_train, vae_train, epochs=30, batch_size=512, validation_split=0.1,
    callbacks=[early_stopping], verbose=2,
)

# Deterministic reconstructor (uses z_mean, no sampling noise) for scoring.
h_det = decoder_hidden(z_mean)
outputs_det = decoder_output(h_det)
reconstructor = Model(encoder_inputs, outputs_det, name="vae_reconstructor")
reconstructor.save_weights("backend/vae_weights.weights.h5")


def reconstruction_error(matrix: np.ndarray) -> np.ndarray:
    preds = reconstructor.predict(matrix, batch_size=512, verbose=0)
    return np.mean(np.square(matrix - preds), axis=1)


# ---------------------------------------------------------------------------
# 3. Engineer the hybrid feature set: raw features + VAE anomaly score
# ---------------------------------------------------------------------------
print("[3/7] Engineering hybrid features (raw features + VAE reconstruction error)...")


def build_xy(split_df: pd.DataFrame):
    X_raw = split_df[ordered_features].values.astype(np.float32)
    err = reconstruction_error(X_raw)
    X_hybrid = np.hstack([X_raw, err.reshape(-1, 1)])
    y = split_df["Class"].values
    return X_hybrid, y, err


X_train, y_train, _ = build_xy(train)
X_val, y_val, _ = build_xy(val)
X_test, y_test, err_test = build_xy(test)

hybrid_feature_names = ordered_features + ["vae_reconstruction_error"]

# ---------------------------------------------------------------------------
# 4. Train the supervised Random Forest on the hybrid feature set
# ---------------------------------------------------------------------------
print("[4/7] Training Random Forest on hybrid feature set...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced_subsample",  # handles the ~0.17% fraud imbalance
    n_jobs=-1,
    random_state=SEED,
)
rf.fit(X_train, y_train)

with open("backend/hybrid_rf.pkl", "wb") as f:
    pickle.dump(rf, f)
with open("backend/scaler_amount.pkl", "wb") as f:
    pickle.dump(scaler_amount, f)
with open("backend/scaler_time.pkl", "wb") as f:
    pickle.dump(scaler_time, f)
with open("backend/feature_names.json", "w") as f:
    json.dump({"raw_features": ordered_features, "hybrid_features": hybrid_feature_names}, f, indent=2)

# ---------------------------------------------------------------------------
# 5. Calibrate the decision threshold on the VALIDATION split
# ---------------------------------------------------------------------------
print("[5/7] Calibrating decision threshold on validation split (F1-optimal)...")
val_probs = rf.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
f1_scores = np.divide(
    2 * precisions * recalls, precisions + recalls,
    out=np.zeros_like(precisions), where=(precisions + recalls) != 0,
)
best_idx = int(np.argmax(f1_scores[:-1])) if len(thresholds) else 0
best_threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5
print(f"    F1-optimal probability threshold on validation set: {best_threshold:.4f}")

with open("backend/threshold.json", "w") as f:
    json.dump(
        {
            "probability_threshold": best_threshold,
            "calibration_method": "F1-optimal on held-out validation split (20% of train_val)",
        },
        f,
        indent=2,
    )

# ---------------------------------------------------------------------------
# 6. Final evaluation on the untouched TEST split
# ---------------------------------------------------------------------------
print("[6/7] Evaluating on held-out test split...")
test_probs = rf.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)

report = classification_report(y_test, test_preds, target_names=["Normal", "Fraud"], output_dict=True)
cm = confusion_matrix(y_test, test_preds)
roc_auc = roc_auc_score(y_test, test_probs)
pr_auc = average_precision_score(y_test, test_probs)

print(classification_report(y_test, test_preds, target_names=["Normal", "Fraud"]))
print(f"    ROC-AUC: {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}")

metrics = {
    "threshold": best_threshold,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "confusion_matrix": {
        "true_negatives": int(cm[0][0]),
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "true_positives": int(cm[1][1]),
    },
    "fraud_precision": report["Fraud"]["precision"],
    "fraud_recall": report["Fraud"]["recall"],
    "fraud_f1": report["Fraud"]["f1-score"],
    "n_test": int(len(y_test)),
    "n_test_fraud": int(y_test.sum()),
}
with open("assets/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ---------------------------------------------------------------------------
# 7. Plots for the README
# ---------------------------------------------------------------------------
print("[7/7] Rendering evaluation plots to assets/...")

fpr, tpr, _ = roc_curve(y_test, test_probs)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}", color="#2563EB")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Hybrid VAE + RF")
plt.legend()
plt.tight_layout()
plt.savefig("assets/roc_curve.png", dpi=140)
plt.close()

p, r, _ = precision_recall_curve(y_test, test_probs)
plt.figure(figsize=(5, 4))
plt.plot(r, p, label=f"PR-AUC = {pr_auc:.4f}", color="#DC2626")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — Hybrid VAE + RF")
plt.legend()
plt.tight_layout()
plt.savefig("assets/pr_curve.png", dpi=140)
plt.close()

plt.figure(figsize=(4.5, 4))
import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Test Split)")
plt.tight_layout()
plt.savefig("assets/confusion_matrix.png", dpi=140)
plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(err_test[y_test == 0], bins=60, color="#2563EB", label="Normal", stat="density", log_scale=True)
sns.histplot(err_test[y_test == 1], bins=60, color="#DC2626", label="Fraud", stat="density", log_scale=True)
plt.xlabel("VAE Reconstruction Error (log scale)")
plt.title("Reconstruction Error Separation by Class")
plt.legend()
plt.tight_layout()
plt.savefig("assets/error_distribution.png", dpi=140)
plt.close()

feat_importance = sorted(zip(hybrid_feature_names, rf.feature_importances_), key=lambda x: -x[1])[:10]
names, vals = zip(*feat_importance)
plt.figure(figsize=(6, 4))
plt.barh(names[::-1], vals[::-1], color="#059669")
plt.xlabel("Random Forest Feature Importance")
plt.title("Top 10 Features (Hybrid Model)")
plt.tight_layout()
plt.savefig("assets/feature_importance.png", dpi=140)
plt.close()

print("\nDone. Model artifacts are in backend/, evaluation plots in assets/.")
