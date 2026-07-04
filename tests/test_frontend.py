"""
Smoke tests for the Streamlit frontend using Streamlit's built-in AppTest
harness (no browser needed). These catch import errors, exceptions during
the initial script run, and exceptions triggered by widget interactions.

Requires a running backend at BACKEND_URL (defaults to http://localhost:8000)
for the tests that submit a scoring request; the rest run without it.
"""
import os

import pytest
import requests
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend", "app.py")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def _backend_available():
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=2).status_code == 200
    except requests.exceptions.RequestException:
        return False


def test_app_loads_without_exceptions():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception


def test_model_performance_tab_loads():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.radio[0].set_value("Model Performance").run(timeout=30)
    assert not at.exception


def test_bulk_audit_tab_loads():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.radio[0].set_value("Bulk Ledger Audit").run(timeout=30)
    assert not at.exception


@pytest.mark.skipif(not _backend_available(), reason="Backend not reachable for live scoring test")
def test_single_transaction_evaluate_flow():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.radio[0].set_value("Single Transaction Simulator").run(timeout=30)
    assert len(at.button) >= 3, "expected load-normal, load-fraud, and evaluate buttons"

    at.button[1].click().run(timeout=30)  # load random known-fraud transaction
    assert not at.exception

    at.button[2].click().run(timeout=60)  # Evaluate Transaction
    assert not at.exception
