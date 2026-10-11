"""Central configuration: paths, seeds, feature definitions and model hyper-parameters.

Everything downstream imports from here so that experiments stay reproducible and
paths never drift. No data are ever invented here; this file only defines contract.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_SYNTHETIC = PROJECT_ROOT / "data" / "synthetic"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

for _d in (DATA_RAW, DATA_SYNTHETIC, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Dataset files (SIMULATED - see docs/data_dictionary.md)
FARM_CLEAN = DATA_RAW / "farm_clean.csv"
FARM_FAULTY = DATA_RAW / "farm_faulty.csv"
FARM_SHIFT = DATA_RAW / "farm_shift.csv"
SOIL_PARAMS = CONFIG_DIR / "soil_params.csv"
CROP_PARAMS = CONFIG_DIR / "crop_params.csv"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
# Fixed seeds for every stage so a run can be reproduced exactly.
SEEDS = {"data": 42, "split": 42, "model": 42, "ensemble": 42, "experiment": 42}


def set_global_seed(seed: int = SEED) -> None:
    """Seed python, numpy and torch (if available)."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - CPU CI
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- #
# Feature / target contract
# --------------------------------------------------------------------------- #
# Model INPUT features (never include the leakage columns below).
MODEL_INPUTS = [
    "soil_moisture_measured",
    "temperature_c",
    "humidity_pct",
    "rainfall_mm",
    "et0_mm",
    "irrigation_mm",
]

# Evaluation-only columns that MUST NOT be fed to the model (leakage / target).
LEAKAGE_COLUMNS = [
    "soil_moisture_true",
    "etc_mm",
    "drainage_mm",
    "runoff_mm",
    "water_stress_ks",
    "sensor_fault_label",
    "sensor_fault_type",
]

TARGET = "soil_moisture_measured"
TARGET_TRUE = "soil_moisture_true"

# Columns kept when we need to evaluate physical outcomes.
PHYSICAL_COLUMNS = [
    "etc_mm",
    "drainage_mm",
    "runoff_mm",
    "water_stress_ks",
    "soil_moisture_true",
    "irrigation_mm",
]

MOISTURE_BOUNDS = (0.0, 1.0)  # volumetric m3/m3 hard safety bound

# --------------------------------------------------------------------------- #
# Task / model hyper-parameters
# --------------------------------------------------------------------------- #


@dataclass
class ForecastConfig:
    history: int = 24       # hours of past observations used as input
    horizons: tuple = (1, 3, 6)   # forecast horizons (hours)
    stride: int = 1         # sliding-window stride for training sequences
    batch_size: int = 128
    epochs: int = 30
    lr: float = 1e-3
    patience: int = 6       # early-stopping patience
    hidden: int = 64        # LSTM hidden units
    cnn_channels: int = 32  # CNN filters
    dropout: float = 0.1
    ensemble_size: int = 5  # deep-ensemble members


@dataclass
class DecisionConfig:
    """Weights for the normalized multi-objective decision cost."""

    w_water: float = 1.0
    w_stress: float = 2.0
    w_overwater: float = 1.5
    w_cost: float = 0.5
    w_risk: float = 1.0
    eval_horizon_h: int = 24          # common evaluation horizon for candidates
    tie_tolerance: float = 0.02       # scores within this are treated as tied
    # Candidate action menu: (label, irrigation_amount_mm, delay_h)
    candidates: tuple = (
        ("no_irrigation", 0.0, 0),
        ("small", 5.0, 0),
        ("medium", 10.0, 0),
        ("large", 18.0, 0),
        ("delay_6h", 10.0, 6),
    )
    high_uncertainty_quantile: float = 0.80  # above this -> conservative fallback
    min_reliability: float = 0.6             # sensor reliability gate


# --------------------------------------------------------------------------- #
# Parameter tables
# --------------------------------------------------------------------------- #
def load_soil_params() -> pd.DataFrame:
    return pd.read_csv(SOIL_PARAMS).set_index("soil")


def load_crop_params() -> pd.DataFrame:
    return pd.read_csv(CROP_PARAMS).set_index("crop")


@dataclass
class ProjectConfig:
    seed: int = SEED
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    def __post_init__(self):
        set_global_seed(self.seed)
