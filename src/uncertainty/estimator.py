"""Uncertainty estimation for soil-moisture forecasts.

Primary method: DEEP ENSEMBLE. ``ensemble_size`` independent CNN-LSTM members are
trained from different seeds; the ensemble mean is the point forecast and the
member spread captures epistemic uncertainty. To yield calibrated prediction
intervals we add a single multiplicative scale factor, fit on the VALIDATION set
so the empirical coverage of ``mean +/- k*std`` matches the requested coverage.

Alternative method: QUANTILE regression head (pinball loss) giving (loc, lo, hi)
directly.

Both estimators expose the same interface so the rest of the pipeline is
agnostic: ``fit(prepared, horizon)`` and
``predict_interval(X, coverage) -> (lo, hi, mean, std, score)``.

Assumptions (documented in reports): ensemble spread approximates model
uncertainty under the training distribution; we do NOT claim it catches every
out-of-distribution shift (that is explicitly tested in the stress experiments).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src import config
from src.forecasting.models import build_model, pinball_loss
from src.forecasting.train import predict

COVERAGE_DEFAULT = 0.90


@dataclass
class EnsembleUncertainty:
    """Deep-ensemble uncertainty estimator with validation-set calibration."""

    members: list = None
    history: int = 24
    n_features: int = 0
    horizon: int = 1
    kind: str = "cnn_lstm"
    cfg_kw: dict = None
    calibrated_scale: float = 1.0

    def _member_kind(self) -> str:
        return self.kind

    def fit(self, prepared, horizon: int, *, n_members: int,
            cfg: config.ForecastConfig | None = None,
            max_epochs: int | None = None, run_id: str = "ens") -> "EnsembleUncertainty":
        from src.forecasting.train import train_model

        cfg = cfg or config.ForecastConfig()
        max_epochs = max_epochs or cfg.epochs
        train_arr = prepared.train(horizon)
        val_arr = prepared.val(horizon)
        self.history = prepared.history
        self.n_features = train_arr["X"].shape[2]
        self.horizon = horizon
        self.members = []
        for m in range(n_members):
            model = build_model(self.kind, self.history, self.n_features,
                                hidden=cfg.hidden, channels=cfg.cnn_channels, dropout=cfg.dropout)
            res = train_model(model, train_arr, val_arr, batch_size=cfg.batch_size,
                              epochs=max_epochs, lr=cfg.lr, patience=cfg.patience,
                              seed=config.SEEDS["ensemble"] + m)
            self.members.append(model)
        self._calibrate(val_arr["X"], val_arr["y"], COVERAGE_DEFAULT)
        return self

    @torch.no_grad()
    def _raw_stats(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = np.stack([predict(m, X) for m in self.members], axis=0)
        return preds.mean(axis=0), preds.std(axis=0)

    def _calibrate(self, X: np.ndarray, y: np.ndarray, coverage: float) -> None:
        from scipy.stats import norm

        mean, std = self._raw_stats(X)
        resid_abs = np.abs(y - mean)
        z = float(norm.ppf(0.5 * (1 + coverage)))           # 1.6449 for 90%
        target = float(np.quantile(resid_abs, coverage))    # abs residual covering `coverage`
        med_std = float(np.median(std) + 1e-12)
        self.calibrated_scale = float(np.clip(target / (z * med_std), 0.1, 10.0))

    def predict_interval(self, X: np.ndarray, coverage: float = COVERAGE_DEFAULT) -> dict:
        from scipy.stats import norm

        mean, std = self._raw_stats(X)
        z = float(norm.ppf(0.5 * (1 + coverage)))
        half = z * self.calibrated_scale * std
        return {"mean": mean, "std": std, "lo": mean - half, "hi": mean + half,
                "score": self.score(std)}

    def score(self, std: np.ndarray, soil_range: float = 0.3) -> np.ndarray:
        """Normalized uncertainty score in [0,1] given a soil moisture range."""
        return np.clip(std / max(soil_range, 1e-6), 0.0, 1.0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.members):
            torch.save({"state_dict": m.state_dict(), "kind": self.kind}, path / f"member_{i}.pt")
        np.save(path / "scale.npy", np.array([self.calibrated_scale]))

    @classmethod
    def load(cls, path: str | Path, history: int, n_features: int, horizon: int) -> "EnsembleUncertainty":
        path = Path(path)
        obj = cls(history=history, n_features=n_features, horizon=horizon)
        obj.members = []
        for f in sorted(path.glob("member_*.pt")):
            ckpt = torch.load(f, weights_only=False)
            m = build_model(ckpt["kind"], history, n_features)
            m.load_state_dict(ckpt["state_dict"])
            obj.members.append(m)
        obj.calibrated_scale = float(np.load(path / "scale.npy")[0])
        return obj


class QuantileUncertainty:
    """Quantile-regression uncertainty (loc, q_lo, q_hi heads, pinball loss)."""

    def __init__(self, history: int = 24, n_features: int = 0,
                 horizon: int = 1, alpha: float = 0.10):
        self.history, self.n_features, self.horizon, self.alpha = history, n_features, horizon, alpha
        self.model = None
        self.calibrated_alpha = alpha

    def fit(self, prepared, horizon: int, *, cfg: config.ForecastConfig | None = None,
            max_epochs: int | None = None) -> "QuantileUncertainty":
        from src.forecasting.train import train_model

        cfg = cfg or config.ForecastConfig()
        train_arr = prepared.train(horizon)
        val_arr = prepared.val(horizon)
        self.history = prepared.history
        self.n_features = train_arr["X"].shape[2]
        self.horizon = horizon
        self.model = build_model("quantile_cnn_lstm", self.history, self.n_features,
                                 hidden=cfg.hidden, channels=cfg.cnn_channels, dropout=cfg.dropout)
        train_model(self.model, train_arr, val_arr, batch_size=cfg.batch_size,
                    epochs=max_epochs or cfg.epochs, lr=cfg.lr, patience=cfg.patience,
                    seed=config.SEEDS["ensemble"], quantile=True)
        return self

    @torch.no_grad()
    def predict_interval(self, X: np.ndarray, coverage: float = COVERAGE_DEFAULT) -> dict:
        out = []
        for i in range(0, len(X), 4096):
            xb = torch.tensor(X[i:i + 4096], dtype=torch.float32)
            pred = self.model(xb).numpy()
            out.append(pred)
        pred = np.concatenate(out)
        loc, q_lo, q_hi = pred[:, 0], pred[:, 1], pred[:, 2]
        mean = loc
        std = (q_hi - q_lo) / (2 * 1.2816)  # approx normal for 80% quantiles
        return {"mean": mean, "std": std, "lo": q_lo, "hi": q_hi,
                "score": np.clip(std / 0.3, 0.0, 1.0)}


def interval_metrics(lo: np.ndarray, hi: np.ndarray, mean: np.ndarray,
                     y: np.ndarray, coverage: float = COVERAGE_DEFAULT) -> dict:
    covered = float(np.mean((y >= lo) & (y <= hi)))
    width = float(np.mean(hi - lo))
    mae = float(np.mean(np.abs(y - mean)))
    return {"coverage": round(covered, 4), "target_coverage": coverage,
            "interval_width": round(width, 5), "mae": round(mae, 5),
            "calibrated": round(abs(covered - coverage), 4) <= 0.05}


def reliability_status(uncertainty_score: float, sensor_reliability: float,
                       decision_cfg) -> str:
    """Map uncertainty + sensor reliability to a reliability status."""
    if sensor_reliability < decision_cfg.min_reliability:
        return "low-reliability-sensor"
    if uncertainty_score > decision_cfg.high_uncertainty_quantile:
        return "high-uncertainty"
    if uncertainty_score > 0.5:
        return "moderate-uncertainty"
    return "reliable"