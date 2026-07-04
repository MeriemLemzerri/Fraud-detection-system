"""
FastAPI inference service for the Hybrid VAE + Random Forest fraud detector.

Scoring pipeline per transaction:
  1. (optional) scale raw Amount/Time with the fitted RobustScalers
  2. run the transaction through the VAE reconstructor -> reconstruction error
     (an unsupervised anomaly score)
  3. concatenate [30 raw features, reconstruction error] -> feed to the
     Random Forest classifier -> calibrated fraud probability
  4. threshold the probability using the F1-optimal cut-off found on the
     validation split at training time
  5. for the highest-risk transactions in the request, compute SHAP values
     for the Random Forest with a TreeExplainer (exact, and orders of
     magnitude faster than KernelExplainer against a neural net)
"""
import os
import json
import pickle
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fraud-shield")

INPUT_DIM = 30
MAX_SHAP_PER_REQUEST = int(os.environ.get("MAX_SHAP_PER_REQUEST", 10))

ml_state = {}


def locate_asset(filename: str) -> str:
    for path in (filename, os.path.join("backend", filename), os.path.join("/app", filename)):
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find model asset '{filename}'. Run train_pipeline.py first "
        f"to generate backend/ artifacts."
    )


class ClippedSampling(Layer):
    """Deterministic path only needs z_mean, but the layer must exist to
    rebuild the exact graph shape used at training time when loading weights."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_reconstructor(input_dim: int = INPUT_DIM, latent_dim: int = 4) -> Model:
    encoder_inputs = Input(shape=(input_dim,))
    h = Dense(16, activation="relu")(encoder_inputs)
    z_mean = Dense(latent_dim)(h)
    decoder_hidden = Dense(16, activation="relu")
    decoder_output = Dense(input_dim, activation="linear")
    outputs_det = decoder_output(decoder_hidden(z_mean))
    return Model(encoder_inputs, outputs_det, name="vae_reconstructor")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model assets...")

    reconstructor = build_reconstructor()
    reconstructor.load_weights(locate_asset("vae_weights.weights.h5"))

    with open(locate_asset("scaler_amount.pkl"), "rb") as f:
        scaler_amount = pickle.load(f)
    with open(locate_asset("scaler_time.pkl"), "rb") as f:
        scaler_time = pickle.load(f)
    with open(locate_asset("hybrid_rf.pkl"), "rb") as f:
        hybrid_rf = pickle.load(f)
    with open(locate_asset("threshold.json"), "r") as f:
        threshold_cfg = json.load(f)
    with open(locate_asset("feature_names.json"), "r") as f:
        feature_names = json.load(f)

    shap_explainer = shap.TreeExplainer(hybrid_rf)

    ml_state.update(
        reconstructor=reconstructor,
        scaler_amount=scaler_amount,
        scaler_time=scaler_time,
        hybrid_rf=hybrid_rf,
        probability_threshold=threshold_cfg["probability_threshold"],
        feature_names=feature_names["hybrid_features"],
        shap_explainer=shap_explainer,
    )
    logger.info(
        "Model assets loaded. Decision threshold=%.4f",
        ml_state["probability_threshold"],
    )
    yield
    ml_state.clear()


app = FastAPI(
    title="Fintech Fraud Shield — Hybrid VAE + Random Forest API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class AuditPayload(BaseModel):
    features: List[List[float]] = Field(..., description="Rows of 30-dim transaction vectors")
    is_raw: bool = Field(False, description="Set true if Amount/Time (last two columns) are unscaled")

    @field_validator("features")
    @classmethod
    def validate_shape(cls, value):
        if not value:
            raise ValueError("features must contain at least one transaction")
        for row in value:
            if len(row) != INPUT_DIM:
                raise ValueError(f"each transaction must have exactly {INPUT_DIM} features, got {len(row)}")
        return value


class TransactionMetric(BaseModel):
    reconstruction_error: float
    fraud_probability: float
    is_fraud: bool
    shap_values: Optional[List[float]] = None


class AuditResponse(BaseModel):
    threshold: float
    metrics: List[TransactionMetric]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(ml_state)}


@app.get("/api/v1/model-info")
def model_info():
    return {
        "architecture": "VAE (unsupervised anomaly score) + Random Forest (supervised classifier)",
        "probability_threshold": ml_state["probability_threshold"],
        "feature_names": ml_state["feature_names"],
        "n_estimators": ml_state["hybrid_rf"].n_estimators,
    }


@app.post("/api/v1/audit", response_model=AuditResponse)
async def audit_transactions(payload: AuditPayload):
    try:
        input_matrix = np.array(payload.features, dtype=np.float32)

        if payload.is_raw:
            raw_amounts = input_matrix[:, 28].reshape(-1, 1)
            raw_times = input_matrix[:, 29].reshape(-1, 1)
            input_matrix[:, 28] = ml_state["scaler_amount"].transform(raw_amounts).flatten()
            input_matrix[:, 29] = ml_state["scaler_time"].transform(raw_times).flatten()

        reconstructed = ml_state["reconstructor"].predict(input_matrix, verbose=0)
        reconstruction_errors = np.mean(np.square(input_matrix - reconstructed), axis=1)

        hybrid_matrix = np.hstack([input_matrix, reconstruction_errors.reshape(-1, 1)])
        fraud_probs = ml_state["hybrid_rf"].predict_proba(hybrid_matrix)[:, 1]

        threshold = ml_state["probability_threshold"]
        is_fraud_flags = fraud_probs >= threshold

        # Only compute SHAP for the highest-risk rows in the batch -- it's
        # the expensive step, and reviewers/analysts care most about why the
        # riskiest transactions were flagged, not every single row.
        risk_order = np.argsort(-fraud_probs)
        shap_allowed = set(risk_order[:MAX_SHAP_PER_REQUEST].tolist()) & set(
            np.where(is_fraud_flags)[0].tolist()
        )

        shap_by_index = {}
        if shap_allowed:
            idx_list = sorted(shap_allowed)
            shap_raw = ml_state["shap_explainer"].shap_values(hybrid_matrix[idx_list])
            # shap>=0.45 returns a (n, features, n_classes) array for
            # classifiers; select the "fraud" class (index 1).
            if isinstance(shap_raw, list):
                shap_fraud = np.array(shap_raw[1])
            elif shap_raw.ndim == 3:
                shap_fraud = shap_raw[:, :, 1]
            else:
                shap_fraud = shap_raw
            for pos, idx in enumerate(idx_list):
                shap_by_index[idx] = shap_fraud[pos].tolist()

        results = []
        for i in range(len(input_matrix)):
            results.append(
                TransactionMetric(
                    reconstruction_error=float(reconstruction_errors[i]),
                    fraud_probability=float(fraud_probs[i]),
                    is_fraud=bool(is_fraud_flags[i]),
                    shap_values=shap_by_index.get(i),
                )
            )

        return AuditResponse(threshold=threshold, metrics=results)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Audit request failed")
        raise HTTPException(status_code=500, detail=str(exc))
