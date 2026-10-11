"""Quick end-to-end smoke test (tiny config, one farm). Not part of experiments."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config
from src.config import ForecastConfig, DecisionConfig


def main():
    t0 = time.time()
    clean = pd.read_csv(config.FARM_CLEAN)
    faulty = pd.read_csv(config.FARM_FAULTY)
    g = clean[clean["farm_id"] == "F01"].copy()

    from src.preprocessing.pipeline import Preprocessor
    from src.forecasting.train import load_or_train, predict
    from src.forecasting.baselines import persistence_from_windows, evaluate
    from src.uncertainty.estimator import EnsembleUncertainty, interval_metrics
    from src.fault_detection.detector import SensorValidator, evaluate_detection
    from src.soil_water_simulator.simulator import SoilWaterSimulator
    from src.counterfactual.simulator import (candidates_from_config, WeatherForecast,
                                              build_schedule)
    from src.decision_engine.engine import DecisionEngine
    from src.explainability.explainer import explain_recommendation

    cfg = ForecastConfig(history=12, horizons=(3,), epochs=1, patience=2,
                         hidden=16, cnn_channels=8, ensemble_size=3, batch_size=64)
    prep = Preprocessor(history=cfg.history, horizons=cfg.horizons, stride=1)
    prepared = prep.prepare(g)
    print("prepared OK:", {h: {s: prepared.sequences[h][s]['X'].shape for s in prepared.sequences[h]} for h in cfg.horizons})

    model, res = load_or_train(prepared, 3, kind="cnn_lstm", cfg=cfg, run_id="smoke")
    te = prepared.test(3)
    yp = persistence_from_windows(te["X"], prepared.scaler, 3)
    ynn = predict(model, te["X"])
    print("persistence:", evaluate(te["y"], yp))
    print("cnn_lstm   :", evaluate(te["y"], ynn))

    ens = EnsembleUncertainty()
    ens.fit(prepared, 3, n_members=2, cfg=cfg, max_epochs=1, run_id="smoke_ens")
    p = ens.predict_interval(te["X"])
    print("intervals  :", interval_metrics(p["lo"], p["hi"], p["mean"], te["y"]))

    v = SensorValidator()
    fg = faulty[faulty["farm_id"] == "F01"]
    val = v.validate(fg)
    print("fault detect:", {k: val for k, val in
                            evaluate_detection(fg["sensor_fault_label"].values,
                                               val["sensor_fault_pred"].values).items()})
    print("reliability mean:", round(float(val["sensor_reliability"].mean()), 3))

    soil = config.load_soil_params().loc["sandy_loam"]
    crop = config.load_crop_params().loc["tomato"]
    sw = SoilWaterSimulator.from_params(soil, crop)

    # decision at one point in the test set
    test_g = g[g["split"] == "test"].reset_index(drop=True)
    i = 6
    theta0 = float(test_g["soil_moisture_measured"].iloc[i])
    env = WeatherForecast.make(g, int(test_g.index[i]), 24, mode="persistent")
    de = DecisionEngine(DecisionConfig())
    cands = candidates_from_config(de.cfg)
    rec = de.decide(theta0=theta0, target_low=float(g["target_low"].iloc[0]),
                    target_high=float(g["target_high"].iloc[0]), soil_water=sw,
                    candidates=cands, env=env, uncertainty_score=0.3, reliability=1.0)
    print("decision:", rec.chosen.candidate, "score", round(rec.chosen.total, 3), "fallback", rec.fallback)
    print(explain_recommendation(rec, theta0, float(g["target_low"].iloc[0]),
                                 float(g["target_high"].iloc[0]), "sandy_loam", "tomato"))
    print(f"\nOK in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()