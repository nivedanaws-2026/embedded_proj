"""Baseline forecasting models: persistence and linear regression.

Persistence predicts the target equals the current observed moisture (optionally
projected by the local trend). Linear regression is fit on the flattened sliding
window (scaled inputs -> raw target). Baselines share the exact same windows and
time splits as the deep models so comparisons are fair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def persistence(measured: np.ndarray, horizon: int, trend: bool = False) -> np.ndarray:
    """Forecast at step i+horizon from the window ending at i."""
    n = len(measured)
    out = np.full(n, np.nan)
    if trend:
        slope = np.nan_to_num(np.diff(measured, prepend=measured[:1]), nan=0.0)
        out = measured + horizon * slope
    else:
        out = measured.copy()
    return out


def persistence_from_windows(X: np.ndarray, scaler, horizon: int,
                             trend: bool = False) -> np.ndarray:
    """Persistence forecast using the last observed moisture in each window.

    The measured-moisture feature (index 0 in ``X``) is scaled, so we invert it
    with the same :class:`FeatureScaler`. No raw series bookkeeping needed.
    """
    last_value = scaler.inverse_target(X[:, -1, 0], "soil_moisture_measured")
    if not trend:
        return last_value.astype(np.float32)
    prev = scaler.inverse_target(X[:, -2, 0], "soil_moisture_measured")
    return (last_value + horizon * (last_value - prev)).astype(np.float32)


def train_linear_regression(train_arr: dict) -> object:
    from sklearn.linear_model import LinearRegression

    X = train_arr["X"].reshape(len(train_arr["X"]), -1)
    model = LinearRegression()
    model.fit(X, train_arr["y"])
    return model


def predict_linear_regression(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X.reshape(len(X), -1))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": round(float(mae), 5), "rmse": round(rmse, 5),
            "r2": round(float(r2_score(y_true, y_pred)), 4)}