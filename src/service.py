"""Shared runtime used by the Streamlit dashboard and the FastAPI endpoints.

Loads data + preprocessing once, trains/loads models and the deep ensemble
lazily, and offers high-level queries (forecast, counterfactual scenarios,
recommendation, explanation) with small caches. All data are SIMULATED.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.config import DecisionConfig, ForecastConfig
from src.counterfactual.simulator import (CandidateAction, CounterfactualSimulator,
                                          WeatherForecast, candidates_from_config)
from src.decision_engine.engine import DecisionEngine
from src.explainability.explainer import explain_recommendation, LABEL_TEXT
from src.fault_detection.detector import SensorValidator
from src.forecasting.train import load_or_train, predict
from src.soil_water_simulator.simulator import SoilWaterSimulator
from src.uncertainty.estimator import EnsembleUncertainty, reliability_status
from src.preprocessing.pipeline import Preprocessor


def make_soil_water(soil: str, crop: str) -> SoilWaterSimulator:
    soils = config.load_soil_params()
    crops = config.load_crop_params()
    return SoilWaterSimulator.from_params(soils.loc[soil], crops.loc[crop])


class Runtime:
    """Lazy, cached access to data, models and decisions for one data source."""

    def __init__(self, data_path: str | Path = config.FARM_CLEAN,
                 forecast_cfg: ForecastConfig | None = None,
                 decision_cfg: DecisionConfig | None = None,
                 run_id: str = "default"):
        self.data_path = Path(data_path)
        self.df = pd.read_csv(self.data_path)
        self.forecast_cfg = forecast_cfg or ForecastConfig()
        self.decision_cfg = decision_cfg or DecisionConfig()
        self.run_id = run_id
        self.prepared = Preprocessor(history=self.forecast_cfg.history,
                                     horizons=self.forecast_cfg.horizons,
                                     stride=1).prepare(self.df)
        self._fault_cache: dict[str, pd.DataFrame] = {}
        self._ensembles: dict[int, EnsembleUncertainty] = {}

    # ------------------------------------------------------------------ #
    def farm_ids(self) -> list[str]:
        return sorted(self.df["farm_id"].unique())

    def farm_df(self, farm_id: str) -> pd.DataFrame:
        return self.df[self.df["farm_id"] == farm_id].reset_index(drop=True)

    def model(self, kind: str = "cnn_lstm", horizon: int | None = None):
        horizon = horizon or self.forecast_cfg.horizons[-1]
        return load_or_train(self.prepared, horizon, kind=kind, cfg=self.forecast_cfg,
                             run_id=self.run_id)[0]

    def ensemble(self, horizon: int | None = None) -> EnsembleUncertainty:
        horizon = horizon or self.forecast_cfg.horizons[-1]
        if horizon not in self._ensembles:
            path = config.MODELS_DIR / f"ensemble_h{horizon}_{self.run_id}"
            if any(path.glob("member_*.pt")):
                n_features = self.prepared.train(horizon)["X"].shape[2]
                self._ensembles[horizon] = EnsembleUncertainty.load(
                    path, self.forecast_cfg.history, n_features, horizon)
            else:
                ens = EnsembleUncertainty()
                ens.fit(self.prepared, horizon, n_members=self.forecast_cfg.ensemble_size,
                        cfg=self.forecast_cfg, max_epochs=self.forecast_cfg.epochs)
                ens.save(path)
                self._ensembles[horizon] = ens
        return self._ensembles[horizon]

    def _window_at(self, farm_df: pd.DataFrame, i: int, horizon: int) -> np.ndarray:
        """Build the single history window ending at row ``i``."""
        scaler = self.prepared.scaler
        scaled = scaler.transform(farm_df[scaler.features])
        h = self.forecast_cfg.history
        lo = i - h + 1
        if lo < 0:  # document: pad by repeating the first observation
            pad = np.repeat(scaled[:1], -lo, axis=0)
            X = np.vstack([pad, scaled[:i + 1]])
        else:
            X = scaled[lo:i + 1]
        return X[-h:][None, :, :]

    def forecast(self, farm_id: str, i: int, horizon: int,
                 with_ensemble: bool = True) -> dict:
        farm_df = self.farm_df(farm_id)
        X = self._window_at(farm_df, i, horizon)
        m = self.model("cnn_lstm", horizon)
        mean = float(predict(m, X)[0])
        std = 0.0
        lo = hi = mean
        if with_ensemble:
            p = self.ensemble(horizon).predict_interval(X)
            mean, std, lo, hi = float(p["mean"][0]), float(p["std"][0]), \
                float(p["lo"][0]), float(p["hi"][0])
        um = min(1.0, max(0.0, std / 0.3))
        ts = farm_df["timestamp"].iloc[min(i + horizon, len(farm_df) - 1)]
        return {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                "std": round(std, 5), "uncertainty_score": round(um, 3),
                "target_time": str(ts), "horizon": horizon, "farm_id": farm_id}

    # ------------------------------------------------------------------ #
    def scenarios(self, farm_id: str, i: int, theta0: float,
                  candidates: list[CandidateAction],
                  mode: str = "persistent") -> tuple[dict, dict, pd.DataFrame]:
        """Roll out every candidate under the same weather; returns env, results."""
        farm_df = self.farm_df(farm_id)
        target_low = float(farm_df["target_low"].iloc[i])
        target_high = float(farm_df["target_high"].iloc[i])
        sw = make_soil_water(farm_df["soil_type"].iloc[0], farm_df["crop_type"].iloc[0])
        env = WeatherForecast.make(farm_df, i, self.decision_cfg.eval_horizon_h, mode=mode)
        sim = CounterfactualSimulator(sw, target_low, target_high,
                                      eval_horizon_h=self.decision_cfg.eval_horizon_h)
        results = [sim.simulate_candidate(theta0, env, c) for c in candidates]
        t0 = pd.to_datetime(farm_df["timestamp"].iloc[i + 1])
        times = pd.date_range(t0, periods=self.decision_cfg.eval_horizon_h, freq="h")
        traj = pd.DataFrame({r.candidate.label: r.trajectory["theta"].values for r in results})
        traj.index = times
        return {"target_low": target_low, "target_high": target_high,
                "soil": farm_df["soil_type"].iloc[0], "crop": farm_df["crop_type"].iloc[0],
                "rain": env["rainfall_mm"].tolist(), "et0": env["et0_mm"].tolist()}, \
            {r.candidate.label: r.metrics for r in results}, traj

    def recommend(self, farm_id: str, i: int,
                  candidates: list[CandidateAction] | None = None,
                  theta0: float | None = None,
                  fault_df: pd.DataFrame | None = None) -> dict:
        candidates = candidates or candidates_from_config(self.decision_cfg)
        farm_df = self.farm_df(farm_id)
        theta0 = theta0 if theta0 is not None else self._observed(farm_df, i)
        target_low = float(farm_df["target_low"].iloc[i])
        target_high = float(farm_df["target_high"].iloc[i])
        sw = make_soil_water(farm_df["soil_type"].iloc[0], farm_df["crop_type"].iloc[0])
        env = WeatherForecast.make(farm_df, i, self.decision_cfg.eval_horizon_h, mode="persistent")
        fc = self.forecast(farm_id, i, self.forecast_cfg.horizons[-1])
        rel = 1.0
        if fault_df is not None:
            rel = float(self._reliability(fault_df, farm_id, i))
        rec = DecisionEngine(self.decision_cfg).decide(
            theta0=theta0, target_low=target_low, target_high=target_high,
            soil_water=sw, candidates=candidates, env=env,
            uncertainty_score=fc["uncertainty_score"], reliability=rel)
        scen_env, scen_metrics, _ = self.scenarios(farm_id, i, theta0, candidates)
        explanation = explain_recommendation(rec, theta0, target_low, target_high,
                                             farm_df["soil_type"].iloc[0],
                                             farm_df["crop_type"].iloc[0])
        status = "fallback-observe" if rec.fallback else rec.status
        return {
            "farm_id": farm_id, "index": i, "theta0": round(theta0, 4),
            "target_low": target_low, "target_high": target_high,
            "forecast": fc, "sensor_reliability": round(rel, 3),
            "reliability_status": reliability_status(fc["uncertainty_score"], rel, self.decision_cfg),
            "recommended_action": rec.chosen.candidate.label,
            "recommended_amount_mm": rec.chosen.candidate.amount_mm,
            "recommended_delay_h": rec.chosen.candidate.delay_h,
            "total_score": round(rec.chosen.total, 4),
            "fallback": rec.fallback, "conservative": rec.conservative,
            "candidates": {s.candidate.label: {
                "amount_mm": s.candidate.amount_mm, "delay_h": s.candidate.delay_h,
                "total_score": round(s.total, 4), "feasible": s.feasible,
                "components": s.components, "metrics": s.metrics} for s in rec.alternatives},
            "selected_candidate": rec.chosen.candidate.label,
            "explanation": explanation.splitlines(),
            "context": scen_env,
        }

    @staticmethod
    def _observed(farm_df: pd.DataFrame, i: int) -> float:
        v = farm_df["soil_moisture_measured"].values
        j = i
        while j >= 0 and (v[j] != v[j]):  # NaN last-good
            j -= 1
        return float(v[max(0, j)])

    # ------------------------------------------------------------------ #
    def fault_analysis(self, farm_id: str, faulty_df: pd.DataFrame) -> pd.DataFrame:
        if farm_id not in self._fault_cache:
            g = faulty_df[faulty_df["farm_id"] == farm_id].reset_index(drop=True)
            self._fault_cache[farm_id] = SensorValidator().validate(g)
        return self._fault_cache[farm_id]

    def _reliability(self, faulty_df: pd.DataFrame, farm_id: str, i: int) -> float:
        out = self.fault_analysis(farm_id, faulty_df)
        return float(out["sensor_reliability"].iloc[min(i, len(out) - 1)])


@lru_cache(maxsize=1)
def get_runtime(data_path: str = str(config.FARM_CLEAN), run_id: str = "default",
                **cfg_kw) -> Runtime:
    fcfg = ForecastConfig(**(cfg_kw or {}))
    return Runtime(data_path=data_path, forecast_cfg=fcfg, run_id=run_id)


def decision_indexes(farm_df: pd.DataFrame, split: str = "test") -> list[int]:
    m = farm_df["split"].isin([split, "shift_test"]) & ((farm_df.index % 24) == 6)
    idx = list(np.where(m.values)[0])
    low = [i for i in idx if i >= 24]
    return low or idx