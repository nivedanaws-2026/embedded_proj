"""Sensor validation and fault detection.

Checks implemented (configurable):
  * missing readings (NaN)
  * physically implausible values (out of [0, saturation] and absolute [0,1])
  * sudden spikes (hourly jump beyond an expected bound)
  * stuck-at readings (flat signal while physics says it should move)
  * gradual drift (long monotonic trend beyond noise)
  * discontinuities / conflicts (moisture rising with no rain or irrigation)

Genuine rainfall/irrigation/rapid-drying events are NOT labelled as faults by
default: spike checks are suppressed when rain or irrigation is present, unless
the jump is physically impossible.

An optional ML anomaly detector (IsolationForest) can be added on clean data.

Faults are reported as per-type flags plus a composite 0/1 flag and a rolling
``reliability`` in [0,1]. Nothing is silently overwritten: imputation is a
separate, logged step (see :func:`impute_moisture`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config

FAULT_TYPES = ["missing", "out_of_range", "spike", "stuck", "drift", "conflict", "anomaly"]


@dataclass
class FaultConfig:
    theta_sat_limit: float = 1.0         # absolute physical ceiling (m3/m3)
    theta_floor: float = 0.0
    max_jump: float = 0.03               # hourly |delta| beyond this -> spike candidate
    impossible_jump: float = 0.30        # hourly |delta| beyond this is always a fault
    stuck_min_hours: int = 8             # flat for >= this many hours -> stuck
    stuck_eps: float = 0.0               # tolerance for "identical" readings
    drift_min_hours: int = 24            # trend window for drift detection
    drift_rate: float = 0.004            # |slope| per hour beyond this -> drift candidate
    conflict_window: int = 6             # hours to check moisture rise without input
    use_ml_detector: bool = False
    ml_contamination: float = 0.03


# --------------------------------------------------------------------------- #
# Rule-based flags
# --------------------------------------------------------------------------- #
def flags_out_of_range(m: pd.Series, cfg: FaultConfig) -> np.ndarray:
    return ((m < cfg.theta_floor) | (m > cfg.theta_sat_limit)).values.astype(bool)


def flags_spike(m: pd.Series, rain: pd.Series, irr: pd.Series, cfg: FaultConfig) -> np.ndarray:
    d = m.diff().abs().values
    d[0] = 0.0
    hard = d > cfg.impossible_jump
    soft = (d > cfg.max_jump) & (rain.values == 0) & (irr.values == 0)
    return (hard | soft).astype(bool)


def flags_stuck(m: pd.Series, et0: pd.Series, rain: pd.Series, cfg: FaultConfig) -> np.ndarray:
    """Flat run of length >= stuck_min_hours while physics would change moisture."""
    eps = cfg.stuck_eps
    run = np.zeros(len(m), dtype=int)
    cnt = 1
    out = np.zeros(len(m), dtype=bool)
    vals = m.values
    for i in range(1, len(m)):
        if abs(vals[i] - vals[i - 1]) <= eps:
            cnt += 1
        else:
            cnt = 1
        run[i] = cnt
    for i in range(len(m) - 1, -1, -1):
        if run[i] >= cfg.stuck_min_hours:
            plausible_change = (et0.values[i] > 0.05) | (rain.values[i] > 0.1)
            if plausible_change:
                out[max(0, i - run[i] + 1):i + 1] = True
    return out


def flags_drift(m: pd.Series, rain: pd.Series, irr: pd.Series, cfg: FaultConfig) -> np.ndarray:
    """Detect slow persistent drift via rolling linear-trend slopes."""
    n = len(m)
    out = np.zeros(n, dtype=bool)
    w = cfg.drift_min_hours
    if n < w + 2:
        return out
    x = np.arange(w)
    for i in range(0, n - w + 1, 2):  # stride 2 for speed
        seg = m.values[i:i + w]
        if np.isnan(seg).any():
            continue
        slope = np.polyfit(x, seg, 1)[0]
        if abs(slope) > cfg.drift_rate and (rain.values[i:i + w].sum() < 1.0):
            out[i:i + w] = True
    return out


def flags_conflict(m: pd.Series, rain: pd.Series, irr: pd.Series, cfg: FaultConfig) -> np.ndarray:
    """Moisture rises by a clear amount with no rain and no irrigation recorded."""
    if cfg.conflict_window <= 0:
        return np.zeros(len(m), dtype=bool)
    d = m.diff(periods=cfg.conflict_window)
    inflow = (rain + irr).rolling(cfg.conflict_window).sum()
    cond = (d > cfg.max_jump) & (inflow < 0.5)
    return cond.fillna(False).values.astype(bool)


# --------------------------------------------------------------------------- #
# Optional ML anomaly detector
# --------------------------------------------------------------------------- #
class MLAnomalyDetector:
    """IsolationForest trained on clean data to flag unusual moisture dynamics."""

    def __init__(self, contamination: float = 0.03):
        self.contamination = contamination
        self.model = None

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        m = df["soil_moisture_measured"]
        feat = pd.DataFrame({
            "m": m,
            "m_d1": m.diff(1),
            "m_d3": m.diff(3),
            "m_roll": m.rolling(6, min_periods=1).std(),
        }).fillna(0.0).values
        return feat

    def fit(self, df: pd.DataFrame) -> "MLAnomalyDetector":
        from sklearn.ensemble import IsolationForest

        X = self._features(df)
        self.model = IsolationForest(contamination=self.contamination, random_state=config.SEED)
        self.model.fit(X)
        return self

    def predict_flags(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:  # pragma: no cover
            raise RuntimeError("fit() the detector first")
        X = self._features(df)
        return (self.model.predict(X) == -1).astype(bool)


# --------------------------------------------------------------------------- #
# Combined validator
# --------------------------------------------------------------------------- #
class SensorValidator:
    def __init__(self, cfg: FaultConfig | None = None):
        self.cfg = cfg or FaultConfig()

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Annotate ``df`` with per-type fault flags, composite flag and reliability."""
        df = df.copy()
        m = df["soil_moisture_measured"].astype(float)
        rain = df["rainfall_mm"].fillna(0.0)
        irr = df["irrigation_mm"].fillna(0.0)
        et0 = df["et0_mm"].fillna(0.0)
        flags = {
            "missing": m.isna().values,
            "out_of_range": flags_out_of_range(m, self.cfg),
            "spike": flags_spike(m, rain, irr, self.cfg),
            "stuck": flags_stuck(m, et0, rain, self.cfg),
            "drift": flags_drift(m, rain, irr, self.cfg),
            "conflict": flags_conflict(m, rain, irr, self.cfg),
        }
        for k, v in flags.items():
            df[f"flag_{k}"] = v
        if self.cfg.use_ml_detector:
            det = MLAnomalyDetector(self.cfg.ml_contamination)
            det.fit(df[df["sensor_fault_label"] == 0] if "sensor_fault_label" in df else df)
            df["flag_anomaly"] = det.predict_flags(df)
        else:
            df["flag_anomaly"] = False
        composite = np.logical_or.reduce([flags[k] for k in FAULT_TYPES if k in flags] +
                                         [df["flag_anomaly"].values])
        df["sensor_fault_pred"] = composite.astype(int)

        # reliability: rolling fraction of clean hours inside a window
        win = min(24, max(8, len(df) // 50))
        clean = 1.0 - composite.astype(float)
        df["sensor_reliability"] = pd.Series(clean).rolling(win, min_periods=1).mean().values
        return df

    def column_mapping(self) -> dict:
        return {k: f"flag_{k}" for k in FAULT_TYPES}


# --------------------------------------------------------------------------- #
# Imputation (explicit and logged - never silent)
# --------------------------------------------------------------------------- #
def impute_moisture(m: pd.Series, fault_flags: np.ndarray | None = None,
                    method: str = "forward_fill_limited") -> tuple[pd.Series, pd.DataFrame]:
    """Impute missing/faulty moisture and return a log of what was changed.

    Returns (corrected_series, log) where log has columns index, original,
    imputed, method. No silent replacement: each changed row is recorded.
    """
    s = m.copy()
    fix_mask = s.isna() if fault_flags is None else (fault_flags.astype(bool) | s.isna())
    log = pd.DataFrame({"index": range(len(s)),
                        "original": s.values,
                        "imputed": s.values,
                        "method": "unchanged"})
    changed = fix_mask
    if method == "forward_fill_limited":
        fill = s.ffill(limit=6)  # within 6h of the last good reading
        fill = fill.bfill(limit=6).fillna(s.median())
        s = pd.Series(np.where(changed, fill, s).astype(float), index=s.index)
    else:  # pragmatic median-based pull for remaining holes
        med = s.median()
        s = s.fillna(med)
    s = np.clip(s, config.MOISTURE_BOUNDS[0], config.MOISTURE_BOUNDS[1])
    log["imputed"] = s.values
    log["method"] = np.where(changed, f"imputed_{method}", "unchanged")
    return pd.Series(s.values, index=m.index), log


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_detection(true: np.ndarray, pred: np.ndarray,
                       true_types: pd.Series | None = None) -> dict:
    """Point-level metrics plus per-segment detection delay.

    * precision / recall / F1 / false-alarm rate computed row-wise;
    * detection delay = first corrected flag inside a ground-truth fault segment
      minus the segment's first hour (averaged over detected segments).
    """
    true = np.asarray(true, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int((true & pred).sum())
    fp = int((~true & pred).sum())
    fn = int((true & ~pred).sum())
    tn = int((~true & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0

    # segment-aware detection delay
    delays = []
    n = len(true)
    i = 0
    while i < n:
        if true[i]:
            j = i
            while j + 1 < n and true[j + 1]:
                j += 1
            seg = slice(i, j + 1)
            first = int(np.argmax(pred[seg])) if pred[seg].any() else -1
            if first >= 0:
                delays.append(first)
            i = j + 1
        else:
            i += 1
    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "false_alarm_rate": round(float(far), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "detection_delay_h": round(float(np.mean(delays)), 2) if delays else None,
        "n_fault_segments": len(delays),
    }