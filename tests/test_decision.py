import numpy as np

from src.counterfactual.simulator import (CandidateAction, WeatherForecast,
                                          candidates_from_config)
from src.decision_engine.engine import DecisionEngine, ScoredScenario


def test_more_water_is_not_automatically_chosen(small_farm, soil_water, tiny_decision_cfg):
    """The engine must select the option meeting crop needs with least cost."""
    de = DecisionEngine(tiny_decision_cfg)
    env = WeatherForecast.make(small_farm, 500, tiny_decision_cfg.eval_horizon_h, mode="persistent")
    cands = candidates_from_config(tiny_decision_cfg)
    rec = de.decide(theta0=0.19, target_low=0.17, target_high=0.21, soil_water=soil_water,
                    candidates=cands, env=env, uncertainty_score=0.1, reliability=1.0)
    assert rec.fallback is False
    if rec.chosen.candidate.label == "no_irrigation":
        return  # reaching target without water is acceptable and optimal
    # if it irrigates it must be the lowest-cost feasible one
    feasible = [s for s in rec.alternatives if s.feasible]
    assert rec.chosen in feasible
    assert min(s.total for s in feasible) >= rec.chosen.total - 1e-9 - 0.02


def test_dry_soil_high_uncertainty_triggers_fallback(small_farm, soil_water, tiny_decision_cfg):
    """Very dry start + unreliable forecast => no confident recommendation."""
    de = DecisionEngine(tiny_decision_cfg)
    env = WeatherForecast.make(small_farm, 500, tiny_decision_cfg.eval_horizon_h, mode="persistent")
    cands = candidates_from_config(tiny_decision_cfg)
    dry = 0.05  # far below sandy-loam field capacity / target
    rec = de.decide(theta0=dry, target_low=0.17, target_high=0.21, soil_water=soil_water,
                    candidates=cands, env=env, uncertainty_score=0.95, reliability=0.1)
    assert rec.conservative
    assert rec.fallback  # nothing safe under conservative constraints


def test_tie_break_prefers_less_water(tiny_decision_cfg):
    de = DecisionEngine(tiny_decision_cfg)
    mk = lambda label, amt, total: ScoredScenario(
        candidate=CandidateAction(label, amt), metrics={"total_water_applied_mm": amt,
                                                        "stress_area_mmh": 0.0,
                                                        "overwater_risk": 0.0,
                                                        "runoff_mm": 0.0, "drainage_mm": 0.0,
                                                        "final_theta": 0.2, "min_theta": 0.19,
                                                        "time_below_range": 0.0,
                                                        "time_above_range": 0.0,
                                                        "irrigation_schedule": []},
        components={"water": amt / 20, "stress": 0.0, "overwater": 0.0, "cost": amt / 20, "risk": 0.1},
        total=total, feasible=True)
    a = mk("small", 5.0, 1.0)
    b = mk("medium", 10.0, 1.0)
    best, fallback, tied = de.choose([a, b], reliability=1.0)
    assert not fallback
    assert best is a                       # equal score -> less water wins


def test_risk_term_favours_reliable_when_confident(tiny_decision_cfg, soil_water):
    de = DecisionEngine(tiny_decision_cfg)
    env = WeatherForecast.make(pd_small(), 300, 8, mode="persistent")
    cands = candidates_from_config(tiny_decision_cfg)
    rec_hi = de.decide(theta0=0.18, target_low=0.17, target_high=0.21, soil_water=soil_water,
                       candidates=cands, env=env, uncertainty_score=0.9, reliability=1.0)
    rec_lo = de.decide(theta0=0.18, target_low=0.17, target_high=0.21, soil_water=soil_water,
                       candidates=cands, env=env, uncertainty_score=0.1, reliability=1.0)
    # high-uncertainty forces conservative evaluation; low-uncertainty does not
    assert rec_hi.conservative
    assert not rec_lo.conservative


def pd_small():
    import pandas as pd
    import numpy as np
    n = 360
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "farm_id": "T", "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "rainfall_mm": np.zeros(n), "et0_mm": 0.2 * ((np.arange(n) % 24) < 12),
        "kc": np.ones(n), "soil_moisture_measured": 0.18 + rng.normal(0, 0.004, n),
        "target_low": 0.17, "target_high": 0.21,
    })