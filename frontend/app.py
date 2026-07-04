import os
import json

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Fintech Fraud Shield", layout="wide", page_icon="️")

st.markdown(
    """
    <style>
    .main-header { font-size:2.4rem; color: #1E3A8A; font-weight: 700; margin-bottom: 2px; }
    .sub-header { font-size:1.05rem; color: #4B5563; margin-bottom: 25px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">️ Fintech Fraud Shield</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Hybrid architecture: a Variational Autoencoder\'s reconstruction '
    "error (unsupervised anomaly signal) feeds a Random Forest classifier "
    "(supervised decision boundary), explained per-transaction with SHAP.</p>",
    unsafe_allow_html=True,
)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")
FEATURE_LABELS = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time", "vae_reconstruction_error"]
RAW_FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]


def _first_existing(*candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


HERE = os.path.dirname(__file__)
SAMPLE_CSV_PATH = _first_existing(
    os.path.join(HERE, "sample_test.csv"),
    os.path.join(HERE, "..", "sample_test.csv"),
)
ASSETS_DIR = _first_existing(
    os.path.join(HERE, "assets"),
    os.path.join(HERE, "..", "assets"),
)


@st.cache_data(show_spinner=False)
def load_sample_transactions():
    if not os.path.exists(SAMPLE_CSV_PATH):
        return None
    return pd.read_csv(SAMPLE_CSV_PATH)

@st.cache_data(ttl=5, show_spinner=False)
def check_backend_health():
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return resp.status_code == 200 and resp.json().get("model_loaded", False)
    except requests.exceptions.RequestException:
        return False


def render_shap_chart(shap_values, title="Top Feature Contributions (SHAP)"):
    shap_array = np.array(shap_values)
    top_idx = np.argsort(np.abs(shap_array))[-8:]
    colors = ["#DC2626" if shap_array[i] > 0 else "#2563EB" for i in top_idx]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh([FEATURE_LABELS[i] for i in top_idx], shap_array[top_idx], color=colors)
    ax.set_xlabel("Contribution to fraud probability (red = raises risk, blue = lowers it)")
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)


backend_ok = check_backend_health()
with st.sidebar:
    st.title("System Controls")
    if backend_ok:
        st.success("Backend online")
    else:
        st.error(f"Backend unreachable at {BACKEND_URL}")
    workspace_mode = st.radio(
        "Workspace",
        ["Single Transaction Simulator", "Bulk Ledger Audit", "Model Performance"],
    )

# ---------------------------------------------------------------------------
# Single Transaction Simulator
# ---------------------------------------------------------------------------
if workspace_mode == "Single Transaction Simulator":
    st.subheader("Real-time Transaction Risk Simulator")

    sample_df = load_sample_transactions()
    if sample_df is None:
        st.warning(
            "sample_test.csv not found. Run `python train_pipeline.py` to generate "
            "model assets and a labelled validation set."
        )
    else:
        if "current_txn" not in st.session_state:
            st.session_state.current_txn = sample_df[sample_df["Class"] == 0].sample(1).iloc[0]

        col_a, col_b, col_c = st.columns(3)
        if col_a.button(" Load random normal transaction"):
            st.session_state.current_txn = sample_df[sample_df["Class"] == 0].sample(1).iloc[0]
        if col_b.button(" Load random known-fraud transaction"):
            st.session_state.current_txn = sample_df[sample_df["Class"] == 1].sample(1).iloc[0]
        show_truth = col_c.checkbox("Show ground-truth label", value=True)

        txn = st.session_state.current_txn

        st.markdown("##### Adjust the transaction, then run it past the model")
        col1, col2 = st.columns(2)
        with col1:
            amount_input = st.number_input(
                "Transaction Amount ($)", min_value=0.0, value=float(txn["Amount"]), step=1.0
            )
        with col2:
            time_input = st.number_input(
                "Time (seconds since first transaction in dataset)",
                min_value=0.0, value=float(txn["Time"]), step=1.0,
            )

        with st.expander("Anonymized PCA features (V1–V28) carried over from the loaded transaction"):
            v_values = txn[[f"V{i}" for i in range(1, 29)]].astype(float)
            st.dataframe(v_values.to_frame("value").T, use_container_width=True)

        if show_truth:
            truth_label = "FRAUD" if txn["Class"] == 1 else "Normal"
            st.caption(f"Ground truth for the loaded transaction: **{truth_label}**")

        if st.button("Evaluate Transaction", type="primary", disabled=not backend_ok):
            vector = [float(txn[f"V{i}"]) for i in range(1, 29)] + [amount_input, time_input]
            with st.spinner("Scoring transaction..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/api/v1/audit",
                        json={"features": [vector], "is_raw": True},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        m = resp.json()["metrics"][0]
                        prob = m["fraud_probability"]
                        st.markdown("---")
                        gauge_col, detail_col = st.columns([1, 2])
                        with gauge_col:
                            st.metric("Fraud Probability", f"{prob * 100:.2f}%")
                            st.metric("Reconstruction Error", f"{m['reconstruction_error']:.4f}")
                            st.progress(min(prob, 1.0))
                        with detail_col:
                            if m["is_fraud"]:
                                st.error("️ ALERT: Transaction flagged as high fraud risk")
                            else:
                                st.success(" VERIFIED: Transaction looks legitimate")
                            if m.get("shap_values"):
                                render_shap_chart(m["shap_values"])
                            else:
                                st.caption("SHAP explanation only computed for the highest-risk transactions.")
                    else:
                        st.error(f"API error [{resp.status_code}]: {resp.text}")
                except requests.exceptions.RequestException as exc:
                    st.error(f"Could not reach backend: {exc}")

# ---------------------------------------------------------------------------
# Bulk Ledger Audit
# ---------------------------------------------------------------------------
elif workspace_mode == "Bulk Ledger Audit":
    st.subheader("Batch Transaction Audit")
    st.caption(
        "Upload a CSV with columns V1–V28, scaled_amount, scaled_time (as produced by "
        "train_pipeline.py's sample_test.csv), and an optional Class column for scoring."
    )
    uploaded_file = st.file_uploader("Upload a transaction ledger (CSV)", type=["csv"])

    if uploaded_file is not None and backend_ok:
        input_df = pd.read_csv(uploaded_file)
        feature_cols = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]
        missing = [c for c in feature_cols if c not in input_df.columns]
        if missing:
            st.error(f"Uploaded file is missing required columns: {missing}")
        else:
            with st.spinner(f"Auditing {len(input_df):,} transactions..."):
                try:
                    payload_matrix = input_df[feature_cols].values.tolist()
                    resp = requests.post(
                        f"{BACKEND_URL}/api/v1/audit",
                        json={"features": payload_matrix, "is_raw": False},
                        timeout=180,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        metrics_list = data["metrics"]
                        threshold = data["threshold"]

                        probs = [m["fraud_probability"] for m in metrics_list]
                        preds = [1 if m["is_fraud"] else 0 for m in metrics_list]

                        result_df = input_df.copy()
                        result_df["Fraud Probability"] = probs
                        result_df["Status"] = ["HIGH FRAUD RISK" if p else "Cleared" for p in preds]

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Transactions Processed", f"{len(preds):,}")
                        c2.metric("Flagged as Fraud", f"{sum(preds):,}")
                        c3.metric("Flag Rate", f"{100 * sum(preds) / len(preds):.2f}%")

                        st.markdown("---")
                        g_col, d_col = st.columns([3, 2])
                        with g_col:
                            st.write("Fraud Probability Distribution")
                            fig, ax = plt.subplots(figsize=(7, 4))
                            sns.histplot(probs, bins=40, kde=False, color="#2563EB", ax=ax)
                            ax.axvline(threshold, color="red", linestyle="--", label=f"Decision threshold ({threshold:.2f})")
                            ax.set_xlabel("Predicted fraud probability")
                            ax.legend()
                            st.pyplot(fig)
                            plt.close(fig)
                        with d_col:
                            st.write("Highest-Risk Transactions")
                            risk_ledger = result_df[result_df["Status"] == "HIGH FRAUD RISK"].sort_values(
                                "Fraud Probability", ascending=False
                            )
                            if not risk_ledger.empty:
                                st.dataframe(risk_ledger[["Fraud Probability", "Status"]], use_container_width=True)
                            else:
                                st.success("No transactions exceeded the fraud threshold.")

                        if "Class" in input_df.columns:
                            from sklearn.metrics import classification_report
                            st.markdown("---")
                            st.write("Validation report (Class column found in upload)")
                            report = classification_report(
                                input_df["Class"], preds, target_names=["Normal", "Fraud"], output_dict=True
                            )
                            st.dataframe(pd.DataFrame(report).T, use_container_width=True)

                        st.markdown("---")
                        st.write("Full Audit Output")
                        st.dataframe(result_df, use_container_width=True)
                    else:
                        st.error(f"API error [{resp.status_code}]: {resp.text}")
                except requests.exceptions.RequestException as exc:
                    st.error(f"Could not reach backend: {exc}")
    elif uploaded_file is not None and not backend_ok:
        st.error("Backend is unreachable, cannot run the audit.")

# ---------------------------------------------------------------------------
# Model Performance
# ---------------------------------------------------------------------------
else:
    st.subheader("Offline Evaluation (held-out test split)")
    assets_dir = ASSETS_DIR
    metrics_path = os.path.join(assets_dir, "metrics.json")

    if not os.path.exists(metrics_path):
        st.warning("No evaluation assets found. Run `python train_pipeline.py` to generate them.")
    else:
        with open(metrics_path) as f:
            metrics = json.load(f)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        c2.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
        c3.metric("Fraud Precision", f"{metrics['fraud_precision'] * 100:.1f}%")
        c4.metric("Fraud Recall", f"{metrics['fraud_recall'] * 100:.1f}%")

        st.caption(
            f"Evaluated on {metrics['n_test']:,} held-out transactions "
            f"({metrics['n_test_fraud']} fraudulent) never seen during training or threshold calibration."
        )

        col1, col2 = st.columns(2)
        for name, caption in [
            ("roc_curve.png", "ROC Curve"),
            ("pr_curve.png", "Precision-Recall Curve"),
            ("confusion_matrix.png", "Confusion Matrix"),
            ("error_distribution.png", "VAE Reconstruction Error by Class"),
            ("feature_importance.png", "Random Forest Feature Importance"),
        ]:
            path = os.path.join(assets_dir, name)
            if os.path.exists(path):
                target = col1 if hash(name) % 2 == 0 else col2
                target.image(path, caption=caption, use_container_width=True)
