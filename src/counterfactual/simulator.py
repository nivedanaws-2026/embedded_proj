"""Counterfactual irrigation simulation.

Before recommending any action, several candidate actions are rolled out against
the SAME weather assumptions, and the resulting trajectories are compared. A
counterfactual estimates what would happen under an action that was NOT actually
taken -- this module never assumes it is a factual forecast.

Environmental conditions for roll-outs are produced by a :class:`WeatherForecast`:

  * ``perfect``   - uses the actual future rows when available (in-sample eval only);
  * ``persistent``- repeats the last 24 h of rain/ET0 and the last known crop Kc
                    (realistic worst-case of no weather knowledge; documented).

The irrigation schedule mirrors the dataset generator (hourly application capped
by the infiltration rate / 10 mm/h), optionally delayed by ``delay_h`` hours.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.soil_water_simulator.simulator import SoilWaterSimulator


@dataclass
class CandidateAction:
    label: str
    amount_mm: float
    delay_h: int = 0


class WeatherForecast:
    @staticmethod
    def make(df: pd.DataFrame, t_index: int, horizon: int,
             mode: str = "persistent") -> dict[str, np.ndarray]:
        """Build rain / et0 / kc series of length ``horizon`` for a roll-out.

        ``t_index`` is the absolute row index of the current decision point in
        ``df`` (the horizon starts at t_index+1).
        """
        if mode == "perfect":
            end = min(len(df), t_index + 1 + horizon)
            rain = df["rainfall_mm"].values[t_index + 1:end]
            et0 = df["et0_mm"].values[t_index + 1:end]
            kc = df["kc"].values[t_index + 1:end]
        else:  # persistent: repeat the last 24 h of conditions
            lo = max(0, t_index - 23)
            seg = df.iloc[lo:t_index + 1]
            rain = _tile(seg["rainfall_mm"].values, horizon)
            et0 = _tile(seg["et0_mm"].values, horizon)
            kc = np.full(horizon, float(np.nanmean(seg["kc"].values)))
        # pad if fewer rows than horizon
        rain = _pad(rain, horizon)
        et0 = _pad(et0, horizon)
        kc = _pad(kc, horizon)
        return {"rainfall_mm": rain, "et0_mm": et0, "kc": kc}


def _tile(seg: np.ndarray, n: int) -> np.ndarray:
    if len(seg) == 0:
        return np.zeros(n)
    reps = int(np.ceil(n / len(seg)))
    return np.tile(seg, reps)[:n]


def _pad(a: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if len(a) >= n:
        return a[:n]
    return np.pad(a, (0, n - len(a)), constant_values=0.0)


def build_schedule(candidate: CandidateAction, horizon: int,
                   infil_cap_mm_h: float, max_rate: float = 10.0) -> np.ndarray:
    """Spread the candidate's water over early hours (respecting the max rate)."""
    arr = np.zeros(horizon)
    if candidate.amount_mm <= 0 or horizon <= candidate.delay_h:
        return arr
    step = min(max_rate, 0.8 * infil_cap_mm_h)
    remaining = float(candidate.amount_mm)
    i = int(candidate.delay_h)
    while remaining > 1e-9 and i < horizon:
        a = min(step, remaining)
        arr[i] = a
        remaining -= a
        i += 1
    return arr


@dataclass
class ScenarioResult:
    candidate: CandidateAction
    trajectory: pd.DataFrame
    metrics: dict


class CounterfactualSimulator:
    def __init__(self, soil_water: SoilWaterSimulator,
                 target_low: float, target_high: float,
                 eval_horizon_h: int = 24, seed: int = config.SEED):
        self.sw = soil_water
        self.target_low = target_low
        self.target_high = target_high
        self.eval_horizon = eval_horizon_h
        self.seed = seed

    def simulate_candidate(self, theta0: float, env: dict[str, np.ndarray],
                           candidate: CandidateAction) -> ScenarioResult:
        horizon = self.eval_horizon
        schedule = build_schedule(candidate, horizon, self.sw.infil_cap_mm_h)
        traj = self.sw.simulate(theta0, env["rainfall_mm"], schedule, env["et0_mm"], env["kc"])
        theta = traj["theta"].values
        below = (theta < self.target_low).mean()
        above = (theta > self.target_high).mean()
        deficit = np.clip(self.target_low - theta, 0.0, None)
        stress_area = float(deficit.sum()) * self.sw.Zr   # theta.h -> mm.h below target
        runoff, drainage = float(traj["runoff_mm"].sum()), float(traj["drainage_mm"].sum())
        applied = float(schedule.sum())
        mets = {
            "total_water_applied_mm": round(applied, 2),
            "final_theta": round(float(theta[-1]), 5),
            "min_theta": round(float(theta.min()), 5),
            "time_below_range": round(float(below), 4),
            "time_above_range": round(float(above), 4),
            "stress_area_mmh": round(stress_area, 3),
            "runoff_mm": round(runoff, 3),
            "drainage_mm": round(drainage, 3),
            "overwater_risk": round(float(above) + min(1.0, (runoff + drainage) / max(applied + 1e-9, 1) / 3), 4),
            "irrigation_schedule": schedule.tolist(),
        }
        return ScenarioResult(candidate=candidate, trajectory=traj, metrics=mets)


def candidates_from_config(cfg: config.DecisionConfig) -> list[CandidateAction]:
    return [CandidateAction(*c) for c in cfg.candidates]