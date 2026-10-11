"""Multi-objective irrigation decision engine.

For each candidate action the engine computes a normalized weighted cost:

    total = w_water*water + w_stress*stress + w_overwater*overwater
          + w_cost*operating_cost + w_risk*risk

Each component is normalized to a comparable scale (see component scores), and
hard constraints (stress, overwatering, drying out) render a candidate
infeasible. Among feasible candidates the lowest total is chosen; a documented
tie-break prefers less water, then an earlier action.

Uncertainty-aware operation:
  * with a reliable forecast + sensor, normal evaluation;
  * with high forecast uncertainty or a low-reliability sensor, a CONSERVATIVE
    evaluation is used: stricter stress and end-state constraints, a stronger
    risk term, and -- if nothing is safe -- a fallback recommendation that asks
    for further observation instead of blindly trusting a point prediction.

The engine only RECOMMENDS; it never drives a physical pump.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config
from src.counterfactual.simulator import (CandidateAction, CounterfactualSimulator,
                                          ScenarioResult, WeatherForecast)
from src.soil_water_simulator.simulator import SoilWaterSimulator


@dataclass
class ScoredScenario:
    candidate: CandidateAction
    metrics: dict
    components: dict
    total: float
    feasible: bool


@dataclass
class Recommendation:
    chosen: ScoredScenario
    alternatives: list[ScoredScenario]
    status: str                 # reliability status of the forecast/sensor
    conservative: bool          # True if conservative evaluation was used
    fallback: bool              # True if no action was safe -> observe only
    explanation: str = ""


class DecisionEngine:
    def __init__(self, cfg: config.DecisionConfig | None = None):
        self.cfg = cfg or config.DecisionConfig()

    # ------------------------------------------------------------------ #
    def component_scores(self, rr: ScenarioResult, theta0: float, target_low: float,
                         target_high: float, soil_water: SoilWaterSimulator,
                         uncertainty_score: float, reliability: float,
                         rainfall_total: float) -> dict:
        m = rr.metrics
        Zr = soil_water.Zr
        ideal_water = max(1.0, max(0.0, (target_high - theta0)) * Zr * 1.2)
        water = float(np.clip(m["total_water_applied_mm"] / ideal_water, 0, 1))
        stress_denom = max(1.0, (target_high - target_low) * Zr * self.cfg.eval_horizon_h)
        stress = float(np.clip(m["stress_area_mmh"] / stress_denom, 0, 1))
        overwater = float(np.clip(m["overwater_risk"], 0, 1))
        cost_norm = max(1.0, m["total_water_applied_mm"])
        cost = float(m["total_water_applied_mm"] / cost_norm)      # linear in water applied
        # risk: forecast uncertainty scaled by how unreliable the sensor is
        pen = max(0.0, self.cfg.min_reliability - reliability) / max(self.cfg.min_reliability, 1e-9)
        risk = float(np.clip(uncertainty_score * (0.5 + 0.5 * pen), 0, 1))
        return {"water": water, "stress": stress, "overwater": overwater,
                "cost": cost, "risk": risk}

    @staticmethod
    def total(c: dict, w) -> float:
        return (w.w_water * c["water"] + w.w_stress * c["stress"]
                + w.w_overwater * c["overwater"] + w.w_cost * c["cost"]
                + w.w_risk * c["risk"])

    # ------------------------------------------------------------------ #
    def _feasible(self, c: dict, rr: ScenarioResult, theta0: float, target_low: float,
                  conservative: bool) -> bool:
        m = rr.metrics
        hard_stress = self.cfg.eval_horizon_h * 0.5 * (m["time_below_range"])
        stress_lim = 0.25 if not conservative else 0.15
        if hard_stress > stress_lim * self.cfg.eval_horizon_h:
            return False
        if m["runoff_mm"] + m["drainage_mm"] > 0.35 * (m["total_water_applied_mm"] + 1e-6):
            return False
        end_margin = 0.0 if conservative else 0.005
        if m["final_theta"] < target_low - end_margin:
            return False
        if conservative and m["min_theta"] < 0.85 * target_low:
            return False
        return True

    # ------------------------------------------------------------------ #
    def evaluate(self, theta0: float, env: dict[str, np.ndarray],
                 candidates: list[CandidateAction], *, target_low: float,
                 target_high: float, soil_water: SoilWaterSimulator,
                 uncertainty_score: float, abstract: float = 0.0) -> tuple[list[ScoredScenario], bool]:
        conservative = abstract > 0.0
        sim = CounterfactualSimulator(soil_water, target_low, target_high,
                                      eval_horizon_h=self.cfg.eval_horizon_h)
        rainfall_total = float(env["rainfall_mm"].sum())
        scored = []
        for cand in candidates:
            rr = sim.simulate_candidate(theta0, env, cand)
            comp = self.component_scores(rr, theta0, target_low, target_high, soil_water,
                                         uncertainty_score, 1.0 - abstract, rainfall_total)
            if conservative:  # stronger risk when the input already looks unreliable
                comp["risk"] = max(comp["risk"], 0.5)
            total = self.total(comp, self.cfg)
            feasible = self._feasible(comp, rr, theta0, target_low, conservative)
            scored.append(ScoredScenario(candidate=cand, metrics=rr.metrics,
                                         components=comp, total=total, feasible=feasible))
        return scored, conservative

    # ------------------------------------------------------------------ #
    def choose(self, scored: list[ScoredScenario],
               reliability: float) -> tuple[ScoredScenario, bool, list[ScoredScenario]]:
        """Select the best feasible candidate (with documented tie-break)."""
        feasible = [s for s in scored if s.feasible]
        if not feasible:
            ranked = sorted(scored, key=lambda s: (s.components["risk"], s.total))
            return ranked[0], True, scored  # fallback: least-risky, but not confident

        def key(s: ScoredScenario):
            # total first, then less water, then smaller delay, then stable ordering
            return (s.total, s.metrics["total_water_applied_mm"], s.candidate.delay_h,
                    s.candidate.label)

        ranked = sorted(feasible, key=key)
        best = ranked[0]
        tied = [s for s in ranked
                if abs(s.total - best.total) <= self.cfg.tie_tolerance
                and s is not best]
        return best, False, tied

    # ------------------------------------------------------------------ #
    def decide(self, *, theta0: float, target_low: float, target_high: float,
               soil_water: SoilWaterSimulator, candidates: list[CandidateAction],
               env: dict[str, np.ndarray], uncertainty_score: float,
               reliability: float) -> Recommendation:
        """Full decision: normal or conservative evaluation + fallback handling."""
        abstract = 0.0
        conservative = False
        if uncertainty_score > self.cfg.high_uncertainty_quantile or reliability < self.cfg.min_reliability:
            abstract, conservative = max(uncertainty_score, 0.5), True
        scored, conservative = self.evaluate(theta0, env, candidates, target_low=target_low,
                                             target_high=target_high, soil_water=soil_water,
                                             uncertainty_score=uncertainty_score, abstract=abstract)
        best, fallback, tied = self.choose(scored, reliability)
        if fallback:
            return Recommendation(chosen=best, alternatives=scored,
                                  status="fallback-observe", conservative=conservative,
                                  fallback=True)
        status = "conservative" if conservative else "normal"
        return Recommendation(chosen=best, alternatives=scored, status=status,
                              conservative=conservative, fallback=False)