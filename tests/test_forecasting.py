import numpy as np
import torch

from src import config
from src.config import ForecastConfig
from src.forecasting.baselines import (persistence, persistence_from_windows,
                                       predict_linear_regression,
                                       train_linear_regression)
from src.forecasting.models import CNNLSTM, build_model
from src.forecasting.train import predict, train_model
from src.uncertainty.estimator import (EnsembleUncertainty, interval_metrics,
                                       reliability_status)


def test_model_io_shapes(tiny_forecast_cfg):
    model = build_model("cnn_lstm", 8, 6, hidden=8, channels=4)
    x = torch.randn(16, 8, 6)
    assert model(x).shape == (16, 1)
    lstm = build_model("lstm", 8, 6, hidden=8)
    assert lstm(x).shape == (16, 1)


def test_train_and_predict(tiny_forecast_cfg, prepared_small):
    cfg = tiny_forecast_cfg
    model = build_model("cnn_lstm", 8, prepared_small.train(3)["X"].shape[2],
                        hidden=cfg.hidden, channels=cfg.cnn_channels)
    res = train_model(model, prepared_small.train(3), prepared_small.val(3),
                      batch_size=64, epochs=1, lr=1e-3, patience=2, seed=config.SEED)
    assert res.val_loss >= 0 and res.best_epoch >= 0
    pred = predict(model, prepared_small.test(3)["X"])
    assert pred.shape == (prepared_small.test(3)["y"].shape[0],)
    assert np.isfinite(pred).all()


def test_persistence_baseline_is_sane(prepared_small):
    te = prepared_small.test(3)
    yp = persistence_from_windows(te["X"], prepared_small.scaler, 3)
    mae = np.mean(np.abs(yp - te["y"]))
    assert mae < 0.02  # 3h-ahead moisture changes are small


def test_linear_regression_shape(prepared_small):
    model = train_linear_regression(prepared_small.train(3))
    pred = predict_linear_regression(model, prepared_small.test(3)["X"])
    assert pred.shape == (prepared_small.test(3)["y"].shape[0],)


def test_ensemble_predict_interval(tiny_forecast_cfg, prepared_small):
    cfg = tiny_forecast_cfg
    ens = EnsembleUncertainty()
    ens.fit(prepared_small, 3, n_members=2, cfg=cfg, max_epochs=1)
    te = prepared_small.test(3)
    p = ens.predict_interval(te["X"])
    assert set(["mean", "std", "lo", "hi", "score"]).issubset(p)
    assert (p["hi"] >= p["mean"]).all() and (p["mean"] >= p["lo"]).all()
    m = interval_metrics(p["lo"], p["hi"], p["mean"], te["y"])
    assert 0.0 <= m["coverage"] <= 1.0 and m["interval_width"] > 0


def test_reliability_status_mapping(tiny_decision_cfg):
    assert reliability_status(0.1, 0.9, tiny_decision_cfg) == "reliable"
    assert reliability_status(0.9, 0.2, tiny_decision_cfg) == "low-reliability-sensor"
    assert reliability_status(0.9, 0.9, tiny_decision_cfg) == "high-uncertainty"