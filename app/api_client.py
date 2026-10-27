import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:5000")

MODEL_NAMES = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}
FEATURE_NAMES = {"all": "all features", "no_period": "without period"}


class ApiError(Exception):
    pass


@st.cache_data(ttl=300, show_spinner=False)
def api_get(path, params=None):
    try:
        r = requests.get(API_URL + path, params=params, timeout=120)
    except requests.exceptions.ConnectionError:
        raise ApiError(f"Cannot reach the API at {API_URL}. "
                       "Start it in another terminal with: python api/app.py")
    if r.status_code != 200:
        try:
            message = r.json().get("error", r.text[:200])
        except ValueError:
            message = r.text[:200]
        raise ApiError(f"API error {r.status_code} for {path}: {message}")
    return r.json()


def require_api():
    """Stop the page with a clear message if the API is not running."""
    try:
        r = requests.get(API_URL + "/api/health", timeout=10)
        r.raise_for_status()
    except requests.exceptions.RequestException:
        st.error(f"Cannot reach the API at {API_URL}. "
                 "Start it in another terminal with: python api/app.py")
        st.stop()


def model_label(name, feature_set):
    return f"{MODEL_NAMES.get(name, name)} ({FEATURE_NAMES.get(feature_set, feature_set)})"


def fmt(value, digits=2):
    """Format a number for display; None or NaN becomes n/a."""
    if value is None or value != value:
        return "n/a"
    return f"{value:.{digits}f}"


def get_summary():
    return api_get("/api/summary")


def get_models():
    return api_get("/api/models")["models"]


def get_compare():
    return api_get("/api/compare")["models"]


def list_signals(**params):
    clean = {k: v for k, v in params.items() if v is not None}
    return api_get("/api/signals", clean)


def get_signal(signal_id):
    return api_get(f"/api/signals/{signal_id}")


def get_lightcurve(signal_id):
    return api_get(f"/api/signals/{signal_id}/lightcurve")