import numpy as np

from src.counterfactual.simulator import (CandidateAction, CounterfactualSimulator,
                                          WeatherForecast, build_schedule,
                                          candidates_from_config)
from src.config import DecisionConfig


def test_schedule_respects_max_rate():
    sched = build_schedule(CandidateAction("medium", 18.0, 0), horizon=8, infil_cap_mm_h=30.0)
    assert sched.sum() == 18.0
    assert sched.max() <= 10.0
    assert np.all(sched[:-1] >= sched[1:] - 1e-9)  # monotone (front-loaded)


def test_delayed_schedule():
    sched = build_schedule(CandidateAction("delay", 10.0, 6), horizon=24, infil_cap_mm_h=30.0)
    assert sched[:6].sum() == 0.0
    assert sched[6:].sum() == 10.0


def test_no_irrigation_schedule_empty():
    sched = build_schedule(CandidateAction("none", 0.0, 0), 24, 30.0)
    assert (sched == 0).all()


def test_candidates_from_config(tiny_decision_cfg):
    cands = candidates_from_config(tiny_decision_cfg)
    assert [c.label for c in cands] == ["no_irrigation", "small", "medium", "large", "delay_6h"]
    assert cands[0].amount_mm == 0.0
    assert cands[-1].delay_h == 6


def test_counterfactual_shares_environment(small_farm, soil_water):
    """All candidates must roll out under the same weather assumptions."""
    env = WeatherForecast.make(small_farm, 500, horizon=8, mode="persistent")
    assert env["rainfall_mm"].shape == (8,)
    assert env["et0_mm"].shape == (8,)
    assert env["kc"].shape == (8,)
    sim = CounterfactualSimulator(soil_water, 0.17, 0.21, eval_horizon_h=8)
    resA = sim.simulate_candidate(0.18, env, CandidateAction("none", 0.0))
    resB = sim.simulate_candidate(0.18, env, CandidateAction("medium", 8.0))
    assert len(resA.trajectory) == len(resB.trajectory) == 8
    assert resA.metrics["total_water_applied_mm"] == 0.0
    assert resB.metrics["total_water_applied_mm"] == 8.0
    # identical weather => with extra water the moisture should not be lower at end
    assert resB.metrics["final_theta"] >= resA.metrics["final_theta"] - 1e-9
    assert "stress_area_mmh" in resA.metrics and "runoff_mm" in resA.metrics


def test_weather_forecast_modes(small_farm):
    i = 500
    h = 24
    p = WeatherForecast.make(small_farm, i, h, mode="persistent")
    per = WeatherForecast.make(small_farm, i, min(h, len(small_farm) - i - 1), mode="perfect")
    assert len(per["rainfall_mm"]) == min(h, len(small_farm) - i - 1)
    # persistent daily cycle: 24 values always returned
    assert len(p["rainfall_mm"]) == h
    assert (p["kc"] == p["kc"][0]).all()