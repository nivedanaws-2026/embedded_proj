import numpy as np
import pandas as pd

from src.fault_detection.detector import (FaultConfig, MLAnomalyDetector,
                                          SensorValidator, evaluate_detection,
                                          impute_moisture)


def pd_series(x) -> pd.Series:
    return pd.Series(np.asarray(x, dtype=float))


def test_detects_all_injected_types(faulty_farm):
    v = SensorValidator(FaultConfig(max_jump=0.02, stuck_min_hours=8, drift_rate=0.001,
                                    theta_sat_limit=0.95))
    out = v.validate(faulty_farm)
    true = faulty_farm["sensor_fault_label"].values
    assert out["flag_missing"][30:46].all()
    assert out["flag_out_of_range"][170:173].all()
    assert out["flag_spike"][60]
    assert out["flag_stuck"][90:104].all()
    # drift takes a window to become obvious; require partial detection
    assert out["flag_drift"][130:160].sum() > 5
    m = evaluate_detection(true, out["sensor_fault_pred"].values)
    assert m["recall"] > 0.5


def test_heavy_rain_is_not_a_spike():
    """A large genuine infiltration should not be flagged if rain explains it."""
    n = 200
    m = np.full(n, 0.18)
    m[50] = 0.22                      # jump explained by rainfall that hour
    rain = np.zeros(n)
    rain[50] = 12.0
    irr = np.zeros(n)
    et0 = np.zeros(n)
    from src.fault_detection.detector import flags_spike
    fl = flags_spike(pd_series(m), pd_series(rain), pd_series(irr), FaultConfig())
    assert not fl[50]


def test_true_discontinuity_flagged_without_input():
    n = 200
    m = np.full(n, 0.18)
    m[80] = 0.25                      # rise with no rain/irrigation
    m = np.where(m < 0.03, 0.0, m).astype(float)
    m[80] = 0.25
    rain = np.zeros(n)
    irr = np.zeros(n)
    from src.fault_detection.detector import flags_conflict
    fl = flags_conflict(pd_series(m), pd_series(rain), pd_series(irr), FaultConfig(max_jump=0.03))
    assert fl[80]


def test_impute_is_logged_not_silent():
    s = pd_series([0.18, 0.19, np.nan, np.nan, 0.20])
    fixed, log = impute_moisture(s)
    assert (log["method"] != "unchanged").sum() == 2      # only the two NaN rows changed
    assert np.isfinite(fixed).all()
    assert (log["original"].isna() == (log["method"] != "unchanged")).all()


def test_reliability_decreases_with_faults(faulty_farm):
    v = SensorValidator()
    out = v.validate(faulty_farm)
    assert out["sensor_reliability"].mean() < 1.0
    assert (out.loc[out["sensor_fault_pred"] == 1, "sensor_reliability"].min() < 0.9)


def test_ml_detector_api(small_farm):
    det = MLAnomalyDetector(contamination=0.05)
    det.fit(small_farm)
    flags = det.predict_flags(small_farm)
    assert flags.dtype == bool and flags.shape[0] == len(small_farm)