import numpy as np
import pandas as pd

from src.soil_water_simulator.simulator import SoilWaterSimulator


def test_moisture_stays_in_physical_bounds(soil_water):
    theta = 0.20
    for _ in range(500):
        r = soil_water.step(theta, rain_mm=np.random.exponential(2),
                            irr_mm=0.0, et0_mm=np.random.uniform(0, 0.6), kc=1.0)
        theta = r["theta"]
        assert soil_water.theta_wp * 0.5 - 1e-9 <= theta <= soil_water.theta_sat + 1e-9


def test_balance_identity(soil_water):
    """Total water should be conserved: S_new = S + infil - etc - drain."""
    theta, rain, irr, et0, kc = 0.20, 12.0, 5.0, 0.4, 1.1
    r = soil_water.step(theta, rain, irr, et0, kc)
    S0 = soil_water.storage(theta)
    S1 = soil_water.storage(r["theta"])
    inflow = min(rain + irr, soil_water.infil_cap_mm_h)
    assert abs(S1 - (S0 + inflow - r["etc_mm"] - r["drainage_mm"] - r["runoff_mm"])) < 1e-6


def test_runoff_when_over_infiltration_capacity(soil_water):
    theta = 0.18
    big = soil_water.infil_cap_mm_h * 3
    r = soil_water.step(theta, rain_mm=big, irr_mm=0, et0_mm=0.0, kc=1.0)
    assert r["runoff_mm"] > 0.9 * (big - soil_water.infil_cap_mm_h)


def test_drainage_above_field_capacity(soil_water):
    theta = soil_water.theta_fc + 0.02  # above field capacity
    r = soil_water.step(theta, rain_mm=0, irr_mm=0, et0_mm=0.0, kc=1.0)
    assert r["drainage_mm"] > 0.0
    # free drainage removes a fraction of the excess per step (rate drain_k)
    assert r["theta"] < theta and r["theta"] >= soil_water.theta_fc


def test_water_stress_reduces_etc(soil_water):
    low = soil_water.theta_wp + 0.002          # close to wilting point
    high = soil_water.theta_fc - 0.005         # well watered
    r_low = soil_water.step(low, 0, 0, et0_mm=0.8, kc=1.0)
    r_high = soil_water.step(high, 0, 0, et0_mm=0.8, kc=1.0)
    assert r_low["etc_mm"] < r_high["etc_mm"]
    assert r_low["ks"] < r_high["ks"]


def test_simulate_rollout_length_and_values(soil_water):
    n = 30
    traj = soil_water.simulate(0.20, np.ones(n), np.zeros(n), np.full(n, 0.3), np.ones(n))
    assert len(traj) == n and "theta" in traj.columns
    assert traj["theta"].between(soil_water.theta_wp * 0.5, soil_water.theta_sat).all()


def test_saturation_overflow_drains(soil_water):
    theta = soil_water.theta_sat
    r = soil_water.step(theta, rain_mm=50.0, irr_mm=0.0, et0_mm=0.0, kc=1.0)
    assert abs(r["theta"] - soil_water.theta_sat) < 1e-9    # never exceeds saturation
    assert r["drainage_mm"] >= 0
    assert r["runoff_mm"] >= 0