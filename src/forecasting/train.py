"""Training utilities: dataset wrapper, train/val loop, checkpointing."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src import config
from src.preprocessing.pipeline import PreparedData


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


@dataclass
class TrainResult:
    history: list[float]
    val_loss: float
    best_epoch: int
    n_epochs_run: int
    best_state: dict
    extra: dict = field(default_factory=dict)


def _validate(model: nn.Module, loader: torch.utils.data.DataLoader, loss_fn) -> float:
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for Xb, yb in loader:
            tot += loss_fn(model(Xb), yb).item() * len(yb)
            n += len(yb)
    return tot / max(1, n)


def train_model(model: nn.Module, train_arr: dict, val_arr: dict, *,
                batch_size: int, epochs: int, lr: float, patience: int,
                seed: int, quantile: bool = False) -> TrainResult:
    """Train with Adam + early stopping; returns best weights (on val loss)."""
    config.set_global_seed(seed)
    train_ds = SeqDataset(train_arr["X"], train_arr["y"])
    val_ds = SeqDataset(val_arr["X"], val_arr["y"])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1024, shuffle=False)

    alpha = getattr(model, "alpha", 0.10)
    loss_fn = (lambda p, y: _pinball(p, y, alpha)) if quantile else nn.MSELoss()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history, best_loss, best_state, best_epoch = [], float("inf"), None, -1
    bad = 0
    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            opt.step()
        vl = _validate(model, val_loader, loss_fn)
        history.append(vl)
        if vl < best_loss - 1e-5:
            best_loss, best_state, best_epoch, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return TrainResult(history=history, val_loss=best_loss, best_epoch=best_epoch,
                       n_epochs_run=len(history), best_state=best_state)


def _pinball(pred: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    loc, q_lo, q_hi = pred[:, 0], pred[:, 1], pred[:, 2]
    loc_loss = (loc - y).pow(2).mean()
    err_lo, err_hi = y - q_lo, y - q_hi
    return loc_loss + torch.maximum(alpha * err_lo, (alpha - 1) * err_lo).mean() + \
        torch.maximum((1 - alpha) * err_hi, -alpha * err_hi).mean()


def save_model(model: nn.Module, path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "kind": type(model).__name__,
                "meta": meta or {}}, path)


def load_model(path: str | Path, model_factory) -> nn.Module:
    ckpt = torch.load(path, weights_only=False)
    model = model_factory()
    model.load_state_dict(ckpt["state_dict"])
    return model


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), 4096):
        xb = torch.tensor(X[i:i + 4096], dtype=torch.float32)
        pred = model(xb)
        if pred.shape[1] == 3:
            pred = pred[:, 0]
        out.append(pred.numpy())
    return np.concatenate(out).ravel()


def horizon_arrays(prepared: PreparedData, horizon: int) -> dict[str, dict[str, np.ndarray]]:
    return {"train": prepared.train(horizon), "val": prepared.val(horizon),
            "test": prepared.test(horizon)}


def default_model_path(kind: str, horizon: int, run_id: str = "default") -> Path:
    return config.MODELS_DIR / f"{kind}_h{horizon}_{run_id}.pt"


def load_or_train(prepared: PreparedData, horizon: int, kind: str = "cnn_lstm",
                  cfg: config.ForecastConfig | None = None,
                  run_id: str = "default", force: bool = False) -> tuple[nn.Module, TrainResult | None]:
    """Load a saved model if present, otherwise train it (and save)."""
    cfg = cfg or config.ForecastConfig()
    path = default_model_path(kind, horizon, run_id)
    arrays = horizon_arrays(prepared, horizon)
    n_features = arrays["train"]["X"].shape[2]

    def factory():
        return build(kind, cfg, prepared, arrays)

    if path.exists() and not force:
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            return load_model(path, lambda: build(kind, cfg, prepared, arrays)), {
                "history": meta.get("val_history", [])}
        return load_model(path, lambda: build(kind, cfg, prepared, arrays)), None
    model = build(kind, cfg, prepared, arrays)
    result = train_model(model, arrays["train"], arrays["val"],
                         batch_size=cfg.batch_size, epochs=cfg.epochs, lr=cfg.lr,
                         patience=cfg.patience, seed=config.SEEDS["model"],
                         quantile=(kind == "quantile_cnn_lstm"))
    save_model(model, path, meta={"kind": kind, "horizon": horizon, "run_id": run_id})
    history_meta = {"kind": kind, "horizon": horizon, "run_id": run_id,
                    "val_history": result.history, "val_loss": result.val_loss,
                    "best_epoch": result.best_epoch, "n_epochs_run": result.n_epochs_run}
    path.with_suffix(".json").write_text(
        json.dumps(history_meta, indent=2, default=float))
    return model, result


def build(kind: str, cfg: config.ForecastConfig, prepared: PreparedData, arrays) -> nn.Module:
    from src.forecasting.models import build_model

    n_features = arrays["train"]["X"].shape[2]
    return build_model(kind, prepared.history, n_features, hidden=cfg.hidden,
                       channels=cfg.cnn_channels, dropout=cfg.dropout)