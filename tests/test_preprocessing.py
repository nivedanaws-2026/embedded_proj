import numpy as np
import pandas as pd

from src.config import ForecastConfig
from src.preprocessing.pipeline import (FeatureScaler, Preprocessor, find_duplicates,
                                        make_sequences, missing_report)


def test_timestamp_ordering(small_farm):
    df = small_farm.sample(frac=1, random_state=1)
    from src.preprocessing.pipeline import parse_and_order

    ordered = parse_and_order(df)
    assert (ordered["timestamp"].diff().dropna() > pd.Timedelta(0)).all()
    assert pd.api.types.is_datetime64_any_dtype(ordered["timestamp"])


def test_duplicate_and_missing_report(small_farm):
    df = small_farm.copy()
    dup = df.iloc[[0]].copy()
    df2 = pd.concat([df, dup], ignore_index=True)
    report = missing_report(df2)
    assert report["n_duplicates"] == 2  # the duplicated key has both copies flagged
    clean_report = missing_report(df)
    assert clean_report["n_duplicates"] == 0
    assert clean_report["n_rows"] == len(df)


def test_scaler_fit_on_train_only(small_farm):
    cfg = ForecastConfig(history=8, horizons=(3,))
    p = Preprocessor(history=cfg.history, horizons=cfg.horizons)
    prepared = p.prepare(small_farm)
    scaler = prepared.scaler
    tr = small_farm[small_farm["split"] == "train"]
    assert abs(scaler.mean_["temperature_c"] - tr["temperature_c"].mean()) < 1e-9


def test_no_future_leakage_in_windows(small_farm):
    cfg = ForecastConfig(history=8, horizons=(3,))
    p = Preprocessor(history=cfg.history, horizons=cfg.horizons)
    prepared = p.prepare(small_farm, split_col="split")
    X = prepared.test(3)["X"]
    assert X.ndim == 3 and X.shape[1] == cfg.history and X.shape[2] == 6
    assert np.isfinite(X).all()


def test_window_target_is_future_row(small_farm):
    """For every window, y must equal the measured value at row idx+horizon."""
    cfg = ForecastConfig(history=8, horizons=(3,))
    p = Preprocessor(history=cfg.history, horizons=cfg.horizons)
    prepared = p.prepare(small_farm)
    seq = prepared.test(3)
    m = small_farm["soil_moisture_measured"].values
    m_true = small_farm["soil_moisture_true"].values
    idx = seq["idx"] + 3
    assert np.allclose(seq["y"], m[idx], atol=1e-6)
    assert np.allclose(seq["y_true"], m_true[idx], atol=1e-6)


def test_make_sequences_shapes():
    Xr = np.random.default_rng(0).normal(size=(100, 4, 3)).astype(np.float32)
    yr = np.arange(100, dtype=np.float32)
    seq = make_sequences(Xr, yr, np.arange(100), history=4, horizon=2, stride=1)
    assert seq["X"].shape[0] == 100 - 4 - 2 + 1              # windows i = 3..97
    assert np.allclose(seq["y"], yr[5:])                     # y[i] = target[i+horizon]
    # first window contains rows 0..3, predicts row 5
    assert np.allclose(seq["X"][0][0], Xr[0])
    assert np.allclose(seq["X"][0][-1], Xr[3])
    assert seq["y"][0] == yr[5]