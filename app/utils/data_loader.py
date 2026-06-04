"""Data loading utilities for the hourly model app."""
import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

HOURLY_FEATURES_PARQUET = ROOT / "app" / "data" / "hourly_features.parquet"
MODEL_PKL = ROOT / "app" / "models" / "hourly_lgbm.pkl"
METRICS_JSON = ROOT / "app" / "models" / "metrics_hourly.json"
VALIDATION_CSV = ROOT / "app" / "outputs" / "forecasts" / "validation_hourly.csv"


def load_hourly_features() -> pd.DataFrame:
    return pd.read_parquet(HOURLY_FEATURES_PARQUET)


@st.cache_resource(show_spinner=False)
def load_model():
    with open(MODEL_PKL, "rb") as f:
        return pickle.load(f)


def load_metrics() -> dict:
    with open(METRICS_JSON, "r") as f:
        return json.load(f)


def load_validation() -> pd.DataFrame:
    return pd.read_csv(VALIDATION_CSV, parse_dates=["hour_bucket"])


def get_plant_names() -> dict:
    """Return {plant_id: plant_name} mapping from features file."""
    df = load_hourly_features()
    codes = sorted(df["ship_plant_code"].unique())
    return {int(c): f"Planta {c}" for c in codes}
