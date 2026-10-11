"""Evaluation: forecast comparison, fault-detection metrics, irrigation-policy
comparison (threshold vs forecast-rule vs full system) and ablations.

All comparisons use identical data splits, initial conditions, environmental
scenarios and horizons. Results are returned as records (and optionally written
to reports/). Simulations are clearly labelled as SIMULATED.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.counterfactual.simulator import (CandidateAction, WeatherForecast,
                                          build_schedule, candidates_from_config)
from src.decision_engine.engine import DecisionEngine
from src.fault_detection.detector import SensorValidator, evaluate_detection
from src.forecasting.baselines import (evaluate, persistence_from_windows,
                                       predict_linear_regression,
                                       train_linear_regression)
from src.forecasting.train import load_or_train, predict
from src.preprocessing.pipeline import PreparedData, make_sequences
from src.soil_water_simulator.simulator import SoilWaterSimulator


# --------------------------------------------------------------------------- #
# 1. Forecast comparison
# --------------------------------------------------------------------------- #
def evaluate_forecasts(prepared: PreparedData,
                       horizons=config.ForecastConfig().horizons,
                       train_nn: dict | None = None,
                       cfg: config.ForecastConfig | None = None,
                       run_id: str = "default") -> list[dict]:
    """MAE/RMSE across models and horizons on the untouched TEST split."""
    cfg = cfg or config.ForecastConfig()
    train_nn = train_nn or {"lstm": True, "cnn_lstm": True, "quantile_cnn_lstm": False}
    rows = []
    for h in horizons:
        tr, te = prepared.train(h), prepared.test(h)
        yt, ytrue = te["y"], te.get("y_true")
        rows.append({"model": "persistence", "horizon": h,
                     **evaluate(yt, persistence_from_windows(te["X"], prepared.scaler, h))})
        lr = train_linear_regression(tr)
        rows.append({"model": "linear_regression", "horizon": h,
                     **evaluate(yt, predict_linear_regression(lr, te["X"]))})
        for kind in ("lstm", "cnn_lstm", "quantile_cnn_lstm"):
            if not train_nn.get(kind, False):
                continue
            model, _ = load_or_train(prepared, h, kind=kind, cfg=cfg, run_id=run_id)
            ynn = predict(model, te["X"])
            row = {"model": kind, "horizon": h, **evaluate(yt, ynn)}
            if ytrue is not None:
                row["mae_vs_true"] = round(float(np.mean(np.abs(ynn - ytrue))), 5)
            rows.append(row)
    return rows


def evaluate_uncertainty(prepared: PreparedData, uncertainty, horizon: int = 6,
                         coverage: float = 0.90) -> dict:
    te = prepared.test(horizon)
    p = uncertainty.predict_interval(te["X"], coverage)
    from src.uncertainty.estimator import interval_metrics

    m = interval_metrics(p["lo"], p["hi"], p["mean"], te["y"], coverage)
    m["horizon"] = horizon
    m["mean_std"] = round(float(np.mean(p["std"])), 5)
    return m


# --------------------------------------------------------------------------- #
# 2. Fault detection evaluation
# --------------------------------------------------------------------------- #
def fault_detection_eval(faulty_df: pd.DataFrame,
                         validator: SensorValidator | None = None) -> list[dict]:
    validator = validator or SensorValidator()
    rows = []
    for fid, g in faulty_df.groupby("farm_id", sort=False):
        out = validator.validate(g)
        m = evaluate_detection(out["sensor_fault_label"].values,
                               out["sensor_fault_pred"].values)
        rows.append({"farm_id": fid, "soil": g["soil_type"].iloc[0],
                     "crop": g["crop_type"].iloc[0], **m})
    return rows


def fault_detection_eval_overall(faulty_df: pd.DataFrame,
                                 validator: SensorValidator | None = None) -> dict:
    validator = validator or SensorValidator()
    out = validator.validate(faulty_df)
    m = evaluate_detection(out["sensor_fault_label"].values,
                           out["sensor_fault_pred"].values,
                           true_types=out["sensor_fault_type"])
    m["n_fault_hours"] = int(out["sensor_fault_label"].sum())
    m["n_total_hours"] = int(len(out))
    m["fault_fraction"] = round(float(out["sensor_fault_label"].mean()), 4)
    m["reliability_mean"] = round(float(out["sensor_reliability"].mean()), 4)
    return m


# --------------------------------------------------------------------------- #
# 3. Irrigation policy comparison
# --------------------------------------------------------------------------- #
@dataclass
class PolicyResult:
    policy: str
    water_used_mm: float
    time_below_target: float        # fraction of hours below target_low
    stress_area_mmh: float
    runoff_mm: float
    drainage_mm: float
    n_irrigation_events: int
    final_theta: float


def _refill_amount(theta: float, soil_water: SoilWaterSimulator,
                   target_low: float, target_high: float) -> float:
    need = max(0.0, (target_high - theta)) * soil_water.Zr
    return float(np.clip(need, 4.0, 25.0))


def replay_policy(df: pd.DataFrame, from_idx: int, to_idx: int,
                  soil_water: SoilWaterSimulator, theta0: float,
                  decision_fn) -> PolicyResult:
    """Replay a policy over a test window using the ACTUAL weather.

    ``decision_fn(i, obs_theta, true_theta)`` returns None (no action) or a list
    of (offset_hours, amount_mm) irrigation events. All policies see only the
    measured observation ``obs_theta`` (faulted/NaN when the sensor fails), never
    the hidden true state -- except this replay's own water-balance state which
    is the simulator's physical state (used to score outcomes).
    """
    theta = float(theta0)
    target_low = float(df["target_low"].iloc[from_idx])
    target_high = float(df["target_high"].iloc[from_idx])
    measured = df["soil_moisture_measured"].values
    schedule: dict[int, float] = {}
    water, n_events, below, stress_area, run_, drain_ = 0.0, 0, 0, 0.0, 0.0, 0.0
    for i in range(from_idx, to_idx):
        act = decision_fn(i, float(measured[i]) if measured[i] == measured[i] else float(theta),
                          float(theta))
        if act:
            n_events += 1
            for off, amt in act:
                schedule[i + off] = schedule.get(i + off, 0.0) + amt
                water += amt
        irr = schedule.pop(i, 0.0)   # applied this hour (incl. offset-0 from this decision)
        r = soil_water.step(theta, rain_mm=float(df["rainfall_mm"].iloc[i]),
                            irr_mm=irr, et0_mm=float(df["et0_mm"].iloc[i]),
                            kc=float(df["kc"].iloc[i]))
        theta = r["theta"]
        if theta < target_low:
            below += 1
            stress_area += (target_low - theta) * soil_water.Zr
        run_ += r["runoff_mm"]
        drain_ += r["drainage_mm"]
    n = to_idx - from_idx
    return PolicyResult(policy="", water_used_mm=round(water, 1),
                        time_below_target=round(below / n, 4),
                        stress_area_mmh=round(stress_area, 1),
                        runoff_mm=round(run_, 1), drainage_mm=round(drain_, 1),
                        n_irrigation_events=n_events, final_theta=round(theta, 4))


def _precompute(farm_df: pd.DataFrame, prepared: PreparedData, model, uncertainty,
                horizon: int) -> dict[int, tuple[float, float]]:
    """Map absolute farm row index -> forecast (mean, std) with the model.

    Window ending at row ``i`` predicts row ``i+horizon``; this is exactly the
    information available at decision hour ``i``.
    """
    feats = config.MODEL_INPUTS
    scaled = prepared.scaler.transform(farm_df[feats])
    target = farm_df[config.TARGET].values.astype(np.float32)
    t_true = farm_df[config.TARGET_TRUE].values.astype(np.float32)
    seq = make_sequences(scaled, target, farm_df["timestamp"].values,
                         prepared.history, horizon, 1, t_true)
    mean = predict(model, seq["X"])
    std = np.zeros(len(mean))
    if uncertainty is not None:
        std = uncertainty.predict_interval(seq["X"])["std"]
    return {int(i): (float(m), float(s)) for i, m, s in zip(seq["idx"], mean, std)}


def _obs_reliability(df: pd.DataFrame) -> np.ndarray:
    if "sensor_reliability" in df.columns:
        return df["sensor_reliability"].values
    return np.ones(len(df))


def make_threshold_policy(soil_water: SoilWaterSimulator, target_low: float,
                          target_high: float):
    def fn(i: int, obs: float, true_theta: float):
        if i % 24 == 6 and obs < target_low:
            return [(0, _refill_amount(obs, soil_water, target_low, target_high))]
        return None
    return fn


def make_forecast_policy(precomputed: dict[int, tuple[float, float]],
                         soil_water: SoilWaterSimulator, target_low: float,
                         target_high: float):
    def fn(i: int, obs: float, true_theta: float):
        if i % 24 != 6:
            return None
        mean = precomputed.get(i, (np.nan, np.nan))[0]
        if np.isnan(mean) or mean < target_low:
            return None
        return [(0, _refill_amount(obs, soil_water, target_low, target_high))]
    return fn


def make_full_policy(engine: DecisionEngine, soil_water: SoilWaterSimulator,
                     candidates: list[CandidateAction], precomputed: dict,
                     reliability: np.ndarray):
    def fn(i: int, obs: float, true_theta: float):
        if i % 24 != 6:
            return None
        env = WeatherForecast.make(_DF_CONTEXT, i, engine.cfg.eval_horizon_h, mode="persistent")
        mean, std = precomputed.get(i, (np.nan, 0.0))
        unc = float(np.clip(std / 0.3, 0.0, 1.0)) if not np.isnan(mean) else 0.9
        rec = engine.decide(theta0=obs, target_low=float(_DF_CONTEXT["target_low"].iloc[i]),
                            target_high=float(_DF_CONTEXT["target_high"].iloc[i]),
                            soil_water=soil_water, candidates=candidates, env=env,
                            uncertainty_score=unc,
                            reliability=float(reliability[i]) if i < len(reliability) else 1.0)
        if rec.fallback:
            return None
        sched = build_schedule(rec.chosen.candidate, engine.cfg.eval_horizon_h,
                               soil_water.infil_cap_mm_h)
        return [(o, sched[o]) for o in range(len(sched)) if sched[o] > 0]
    return fn


_DF_CONTEXT: pd.DataFrame = pd.DataFrame({})


def run_policy_comparison(farm_id: str, df: pd.DataFrame, soil_water: SoilWaterSimulator,
                          prepared: PreparedData, model, uncertainty=None,
                          horizon: int = 6) -> list[PolicyResult]:
    """Compare threshold vs forecast-rule vs full system on one farm's test period."""
    global _DF_CONTEXT
    g = df[df["farm_id"] == farm_id].reset_index(drop=True)
    _DF_CONTEXT = g
    test = g[g["split"].isin(["test", "shift_test"])]
    start, end = int(test.index[0]), int(test.index[-1]) + 1
    target_low = float(g["target_low"].iloc[0])
    target_high = float(g["target_high"].iloc[0])
    theta0 = float(test["soil_moisture_true"].iloc[0])
    pre = _precompute(g, prepared, model, uncertainty, horizon)

    results = [replay_policy(g, start, end, soil_water, theta0,
                             make_threshold_policy(soil_water, target_low, target_high))]
    results[0].policy = "fixed_threshold"
    results.append(replay_policy(g, start, end, soil_water, theta0,
                                 make_forecast_policy(pre, soil_water, target_low, target_high)))
    results[-1].policy = "forecast_rule"
    engine = DecisionEngine()
    candidates = candidates_from_config(engine.cfg)
    rel = _obs_reliability(g)
    results.append(replay_policy(g, start, end, soil_water, theta0,
                                 make_full_policy(engine, soil_water, candidates, pre, rel)))
    results[-1].policy = "full_system_counterfactual"
    return results


# --------------------------------------------------------------------------- #
# 4. Ablations
# --------------------------------------------------------------------------- #
def run_ablations(farm_id: str, df: pd.DataFrame, soil_water: SoilWaterSimulator,
                  prepared: PreparedData, model, uncertainty,
                  faulty_df: pd.DataFrame | None = None) -> list[PolicyResult]:
    """Full system vs ablated variants (no uncertainty / no counterfactual /
    no fault handling)."""
    global _DF_CONTEXT
    g = df[df["farm_id"] == farm_id].reset_index(drop=True)
    _DF_CONTEXT = g
    test = g[g["split"].isin(["test", "shift_test"])]
    start, end = int(test.index[0]), int(test.index[-1]) + 1
    target_low = float(g["target_low"].iloc[0])
    target_high = float(g["target_high"].iloc[0])
    theta0 = float(test["soil_moisture_true"].iloc[0])
    pre = _precompute(g, prepared, model, uncertainty, 6)
    engine = DecisionEngine()
    candidates = candidates_from_config(engine.cfg)

    def run_full(reliability, unc_fn, start_i=None, end_i=None):
        s, e = (start, end) if start_i is None else (start_i, end_i)

        def fn(i, obs, tt):
            if i % 24 != 6:
                return None
            env = WeatherForecast.make(_DF_CONTEXT, i, engine.cfg.eval_horizon_h, mode="persistent")
            rel = float(reliability[i]) if np.ndim(reliability) else float(reliability)
            rec = engine.decide(theta0=obs, target_low=target_low, target_high=target_high,
                                soil_water=soil_water, candidates=candidates, env=env,
                                uncertainty_score=unc_fn(i), reliability=rel)
            if rec.fallback:
                return None
            sch = build_schedule(rec.chosen.candidate, engine.cfg.eval_horizon_h,
                                 soil_water.infil_cap_mm_h)
            return [(o, sch[o]) for o in range(len(sch)) if sch[o] > 0]
        return replay_policy(g, s, e, soil_water, theta0, fn)

    rel = _obs_reliability(g)
    rows = [run_full(rel, lambda i: float(np.clip(pre.get(i, (np.nan, 0.0))[1] / 0.3, 0, 1)))]
    rows[-1].policy = "full"
    rows.append(run_full(rel, lambda i: 0.0))            # no uncertainty awareness
    rows[-1].policy = "no_uncertainty"
    fm = make_forecast_policy(pre, soil_water, target_low, target_high)
    rows.append(replay_policy(g, start, end, soil_water, theta0, fm))  # threshold-on-forecast = no counterfactuals
    rows[-1].policy = "no_counterfactual"

    if faulty_df is not None:
        fg = faulty_df[faulty_df["farm_id"] == farm_id].reset_index(drop=True)
        fg_test = fg[fg["split"] == "test"]
        f_start, f_end = int(fg_test.index[0]), int(fg_test.index[-1]) + 1
        _DF_CONTEXT = fg
        pre_f = _precompute(fg, prepared, model, uncertainty, 6)
        rows.append(replay_policy(fg, f_start, f_end, soil_water, theta0,
                                  make_full_policy(engine, soil_water, candidates, pre_f,
                                                   np.ones(len(fg)))))  # faults ignored
        rows[-1].policy = "no_fault_detection"
        _DF_CONTEXT = g
    return rows


def aggregate_policy_results(all_results: list[list[PolicyResult]]) -> list[dict]:
    agg: dict[str, list[PolicyResult]] = {}
    for farm in all_results:
        for r in farm:
            agg.setdefault(r.policy, []).append(r)
    rows = []
    for policy, rs in sorted(agg.items()):
        ref = agg.get("fixed_threshold", [])
        base_water = float(np.mean([r.water_used_mm for r in ref])) if ref else 1.0
        rows.append({
            "policy": policy,
            "water_used_mm_mean": round(float(np.mean([r.water_used_mm for r in rs])), 1),
            "water_used_mm_std": round(float(np.std([r.water_used_mm for r in rs])), 1),
            "relative_water_to_threshold": round(float(np.mean(
                [r.water_used_mm for r in rs]) / max(base_water, 1e-9)), 3),
            "time_below_target_mean": round(float(np.mean([r.time_below_target for r in rs])), 4),
            "stress_area_mmh_mean": round(float(np.mean([r.stress_area_mmh for r in rs])), 1),
            "runoff_mm_mean": round(float(np.mean([r.runoff_mm for r in rs])), 1),
            "drainage_mm_mean": round(float(np.mean([r.drainage_mm for r in rs])), 1),
            "irrigation_events_mean": round(float(np.mean([r.n_irrigation_events for r in rs])), 1),
            "n_farms": len(rs),
        })
    return rows


def write_json_report(name: str, data) -> Path:
    p = config.REPORTS_DIR / f"{name}.json"
    p.write_text(json.dumps(data, indent=2, default=str))
    return p