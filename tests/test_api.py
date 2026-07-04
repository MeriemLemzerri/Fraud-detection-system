"""
Tests for the FastAPI fraud-audit service.

These are integration-style tests that load the REAL model artifacts from
backend/ via FastAPI's TestClient (which triggers the lifespan startup
event). Run `python train_pipeline.py` at least once before running these
tests so the required files exist.
"""
import os
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.main import app  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
REQUIRED_ARTIFACTS = [
    "backend/vae_weights.weights.h5",
    "backend/hybrid_rf.pkl",
    "backend/scaler_amount.pkl",
    "backend/scaler_time.pkl",
    "backend/threshold.json",
    "backend/feature_names.json",
]

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(os.path.join(REPO_ROOT, p)) for p in REQUIRED_ARTIFACTS),
    reason="Model artifacts not found -- run `python train_pipeline.py` first.",
)

FEATURES = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_rows():
    df = pd.read_csv(os.path.join(REPO_ROOT, "sample_test.csv"))
    fraud = df[df["Class"] == 1].iloc[:3]
    normal = df[df["Class"] == 0].iloc[:3]
    return fraud, normal


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client):
    resp = client.get("/api/v1/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["feature_names"]) == 31  # 30 raw + reconstruction error
    assert 0.0 <= body["probability_threshold"] <= 1.0


def test_audit_rejects_wrong_feature_count(client):
    resp = client.post("/api/v1/audit", json={"features": [[0.1] * 29]})
    assert resp.status_code == 422


def test_audit_rejects_empty_batch(client):
    resp = client.post("/api/v1/audit", json={"features": []})
    assert resp.status_code == 422


def test_audit_normal_transaction_scores_low(client, sample_rows):
    _, normal = sample_rows
    X = normal[FEATURES].values.tolist()
    resp = client.post("/api/v1/audit", json={"features": X, "is_raw": False})
    assert resp.status_code == 200
    metrics = resp.json()["metrics"]
    assert len(metrics) == len(X)
    for m in metrics:
        assert m["fraud_probability"] < 0.5
        assert m["is_fraud"] is False


def test_audit_fraud_transaction_scores_high(client, sample_rows):
    fraud, _ = sample_rows
    X = fraud[FEATURES].values.tolist()
    resp = client.post("/api/v1/audit", json={"features": X, "is_raw": False})
    assert resp.status_code == 200
    metrics = resp.json()["metrics"]
    flagged = sum(1 for m in metrics if m["is_fraud"])
    # The hybrid model should catch the large majority of these known-fraud rows.
    assert flagged >= len(metrics) - 1


def test_audit_returns_shap_values_for_flagged_transactions(client, sample_rows):
    fraud, _ = sample_rows
    X = fraud[FEATURES].values.tolist()
    resp = client.post("/api/v1/audit", json={"features": X, "is_raw": False})
    metrics = resp.json()["metrics"]
    flagged = [m for m in metrics if m["is_fraud"]]
    assert any(m["shap_values"] is not None for m in flagged)
    for m in flagged:
        if m["shap_values"] is not None:
            assert len(m["shap_values"]) == 31


def test_audit_is_raw_mode_scales_amount_and_time(client):
    # A near-zero vector with a huge raw Amount should look anomalous once
    # the backend scales it, without us having to pre-scale it ourselves.
    vector = [0.0] * 28 + [5000.0, 50000.0]
    resp = client.post("/api/v1/audit", json={"features": [vector], "is_raw": True})
    assert resp.status_code == 200
    assert "reconstruction_error" in resp.json()["metrics"][0]
