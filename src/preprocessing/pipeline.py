"""Data preprocessing: ordering, cleaning, leak-free scaling and windowing.

Design rules enforced here:
  * timestamps are parsed and rows set in chronological order per farm;
  * duplicates are detected (not silently dropped);
  * the scaler is fit on TRAINING data only (no leakage);
  * sliding windows use PAST observations to predict a FUTURE target, so no
    future observation is ever used as an input feature;
  * train/val/test splits are time-based, never random.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Loading / cleaning
# --------------------------------------------------------------------------- #
def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return parse_and_order(df)


def parse_and_order(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        sort_cols = ["farm_id", "timestamp"] if "farm_id" in df.columns else ["timestamp"]
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ["farm_id", "timestamp"] if c in df.columns]
    return df[df.duplicated(subset=keys, keep=False)] if keys else df.iloc[0:0]


def missing_report(df: pd.DataFrame) -> dict:
    n = len(df)
    return {
        "n_rows": int(n),
        "n_duplicates": int(len(find_duplicates(df))),
        "missing_counts": {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()},
        "missing_fraction": {c: round(float(df[c].isna().mean()), 5)
                             for c in df.columns if df[c].isna().any()},
    }


def regularize(df: pd.DataFrame, freq: str = "1h", max_gap: int = 3) -> pd.DataFrame:
    """Reindex each farm to a regular grid, interpolating gaps up to ``max_gap`` steps.

    Longer gaps are left as NaN so that downstream imputation is explicit and logged.
    """
    out = []
    for fid, g in df.groupby("farm_id", sort=False):
        g = g.set_index("timestamp").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq=freq)
        g = g.reindex(full)
        num = g.select_dtypes("number").columns
        g[num] = g[num].interpolate(method="time", limit=max_gap, limit_area="inside")
        g = g.rename_axis("timestamp").reset_index()
        g.insert(0, "farm_id", fid)
        out.append(g)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# Scaling (fit on train only)
# --------------------------------------------------------------------------- #
@dataclass
class FeatureScaler:
    features: list[str]
    mean_: dict = field(default_factory=dict)
    scale_: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame) -> "FeatureScaler":
        sub = X[self.features]
        self.mean_ = {c: float(sub[c].mean()) for c in self.features}
        std = sub.std(ddof=0).replace(0, 1.0)
        self.scale_ = {c: float(std[c]) if float(std[c]) != 0 else 1.0 for c in self.features}
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        sub = X[self.features]
        arr = np.empty((len(sub), len(self.features)), dtype=np.float32)
        for j, c in enumerate(self.features):
            arr[:, j] = (sub[c].values - self.mean_[c]) / self.scale_[c]
        return arr

    def inverse_target(self, values: np.ndarray, target: str) -> np.ndarray:
        return values * self.scale_[target] + self.mean_[target]

    def to_dict(self) -> dict:
        return {"features": self.features, "mean": self.mean_, "scale": self.scale_}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureScaler":
        return cls(features=d["features"], mean_=d["mean"], scale_=d["scale"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        return cls.from_dict(json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def make_sequences(scaled: np.ndarray, target_vals: np.ndarray, timestamps: np.ndarray,
                   history: int, horizon: int, stride: int = 1,
                   target_true: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Build supervised windows: inputs = rows [i-history+1 .. i], target = row i+horizon.

    ``scaled`` must already contain the target column at its final index for the
    measured-moisture input feature to be available in the history window.
    """
    n = len(scaled)
    xs, ys, yt, ts, idx = [], [], [], [], []
    for i in range(history - 1, n - horizon, stride):
        xs.append(scaled[i - history + 1:i + 1])
        ys.append(target_vals[i + horizon])
        if target_true is not None:
            yt.append(target_true[i + horizon])
        ts.append(timestamps[i + horizon])
        idx.append(i)
    out = {
        "X": np.asarray(xs, dtype=np.float32),
        "y": np.asarray(ys, dtype=np.float32),
        "idx": np.asarray(idx),
        "target_time": np.asarray(ts),
    }
    if target_true is not None:
        out["y_true"] = np.asarray(yt, dtype=np.float32)
    return out


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    scaler: FeatureScaler
    sequences: dict[int, dict[str, dict[str, np.ndarray]]]  # horizon -> split -> arrays
    report: dict
    history: int = 0

    def train(self, horizon: int) -> dict[str, np.ndarray]:
        return self.sequences[horizon]["train"]

    def val(self, horizon: int) -> dict[str, np.ndarray]:
        return self.sequences[horizon]["val"]

    def test(self, horizon: int) -> dict[str, np.ndarray]:
        return self.sequences[horizon]["test"]


class Preprocessor:
    def __init__(self, inputs: list[str] | None = None, target: str | None = None,
                 horizons: tuple = config.ForecastConfig().horizons,
                 history: int = config.ForecastConfig().history,
                 stride: int = 1):
        self.inputs = list(inputs or config.MODEL_INPUTS)
        self.target = target or config.TARGET
        self.horizons = tuple(horizons)
        self.history = history
        self.stride = stride

    def prepare(self, df: pd.DataFrame, split_col: str = "split",
                scaler_fit: str = "split") -> PreparedData:
        """Leak-free preparation.

        ``scaler_fit`` is "split" (fit on the train rows only) -- the default
        and the correct choice for forecasting. For an out-of-distribution test
        set with no train rows you may pass "all" (fit on the whole file); the
        script documents this choice for the shift experiment.
        """
        df = parse_and_order(df)
        report = missing_report(df)
        feats = self.inputs if self.target in self.inputs else self.inputs + [self.target]
        scaler = FeatureScaler(features=self.inputs)

        if scaler_fit == "all":
            scaler.fit(df[self.inputs])
            report["scaler_fit_on"] = "all rows (OOD test set: no training split present)"
        else:
            train_mask = df[split_col] == "train"
            scaler.fit(df.loc[train_mask, self.inputs])
            report["scaler_fit_on"] = "train split only"

        sequences: dict[int, dict[str, dict]] = {}
        for horizon in self.horizons:
            collect: dict[str, list[dict]] = {s: [] for s in ("train", "val", "test")}
            for _, g in df.groupby("farm_id", sort=False):
                scaled = scaler.transform(g[feats])
                target_vals = g[self.target].values.astype(np.float32)
                target_true = g[config.TARGET_TRUE].values.astype(np.float32) \
                    if config.TARGET_TRUE in g.columns else None
                seq = make_sequences(scaled, target_vals, g["timestamp"].values,
                                     self.history, horizon, self.stride, target_true)
                split_vals = g["split"].values
                if split_vals.dtype == object:  # keep only rows inside valid splits
                    split_vals = np.asarray([s if s in ("train", "val", "test") else "test"
                                             for s in split_vals], dtype=object)
                target_rows = seq["idx"] + horizon
                for s in ("train", "val", "test"):
                    mask = split_vals[target_rows] == s
                    if mask.any():
                        collect[s].append({k: seq[k][mask] for k in seq})
            sequences[horizon] = {}
            for s in ("train", "val", "test"):
                if collect[s]:
                    parts = collect[s]
                    sequences[horizon][s] = {k: np.concatenate([p[k] for p in parts], axis=0)
                                             for k in parts[0]}

        report["history"] = self.history
        report["horizons"] = list(self.horizons)
        report["inputs"] = self.inputs
        report["target"] = self.target
        report["n_windows"] = {h: {s: int(len(sequences[h][s]["X"])) for s in sequences[h]}
                               for h in self.horizons}
        for h in self.horizons:
            for s in sequences[h]:
                a = sequences[h][s]["X"]
                if a.size:
                    assert np.isfinite(a).all(), f"non-finite values in {h}h/{s} windows"
        logger.info("preprocessing report: %s", json.dumps({k: report[k] for k in ("n_rows", "n_duplicates", "n_windows")}))
        return PreparedData(scaler=scaler, sequences=sequences, report=report,
                            history=self.history)
