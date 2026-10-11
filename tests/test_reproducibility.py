import numpy as np

from src import config
from src.config import ForecastConfig
from src.preprocessing.pipeline import Preprocessor


def _prepare_twice(df, cfg):
    p = Preprocessor(history=cfg.history, horizons=cfg.horizons, stride=cfg.history)
    a = p.prepare(df)
    b = Preprocessor(history=cfg.history, horizons=cfg.horizons, stride=cfg.history).prepare(df)
    return a, b


def test_reproducible_preprocessing(small_farm):
    cfg = ForecastConfig(history=8, horizons=(3,), stride=1)
    a, b = _prepare_twice(small_farm, cfg)
    assert a.scaler.mean_ == b.scaler.mean_
    for h in cfg.horizons:
        assert np.array_equal(a.test(h)["X"], b.test(h)["X"])
        assert np.array_equal(a.test(h)["y"], b.test(h)["y"])


def test_seed_affects_data_generation():
    from src.data_generation.virtual_farm import simulate_farm
    df1 = simulate_farm("R1", "sandy_loam", "tomato", "hot_dry", seed=5)
    df2 = simulate_farm("R1", "sandy_loam", "tomato", "hot_dry", seed=5)
    df3 = simulate_farm("R1", "sandy_loam", "tomato", "hot_dry", seed=6)
    assert (df1["soil_moisture_true"] == df2["soil_moisture_true"]).all()
    assert (df1["soil_moisture_true"] != df3["soil_moisture_true"]).any()


def test_model_training_deterministic_given_seed(tiny_forecast_cfg, prepared_small):
    import torch
    from src.forecasting.models import build_model
    from src.forecasting.train import train_model

    def run():
        torch.manual_seed(99)
        config.set_global_seed(99)
        model = build_model("cnn_lstm", 8, prepared_small.train(3)["X"].shape[2],
                            hidden=8, channels=4)
        return train_model(model, prepared_small.train(3), prepared_small.val(3),
                           batch_size=64, epochs=1, lr=1e-3, patience=2, seed=99)

    r1, r2 = run(), run()
    assert r1.val_loss == r2.val_loss
    assert np.allclose(np.asarray(r1.history), np.asarray(r2.history), atol=1e-12)