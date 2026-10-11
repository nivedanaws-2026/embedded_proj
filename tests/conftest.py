import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.config import DecisionConfig, ForecastConfig  # noqa: E402
from src.soil_water_simulator.simulator import SoilWaterSimulator  # noqa: E402


@pytest.fixture(scope="session")
def small_farm() -> pd.DataFrame:
    """A short, self-contained farm slice (train/val/test by time)."""
    from src.data_generation.virtual_farm import simulate_farm

    df = simulate_farm("T01", "sandy_loam", "tomato", "temperate", seed=11).iloc[:600].copy()
    df["split"] = (["train"] * 420) + (["val"] * 90) + (["test"] * 90)
    return df.reset_index(drop=True)


@pytest.fixture(scope="session")
def soil_water() -> SoilWaterSimulator:
    soils = config.load_soil_params()
    crops = config.load_crop_params()
    return SoilWaterSimulator.from_params(soils.loc["sandy_loam"], crops.loc["tomato"])


@pytest.fixture()
def tiny_forecast_cfg() -> ForecastConfig:
    return ForecastConfig(history=8, horizons=(3,), epochs=1, patience=2,
                          hidden=8, cnn_channels=4, ensemble_size=2, batch_size=32)


@pytest.fixture(scope="session")
def tiny_decision_cfg() -> DecisionConfig:
    return DecisionConfig(eval_horizon_h=8, candidates=(
        ("no_irrigation", 0.0, 0),
        ("small", 3.0, 0),
        ("medium", 8.0, 0),
        ("large", 14.0, 0),
        ("delay_6h", 8.0, 6),
    ))


@pytest.fixture()
def faulty_farm() -> pd.DataFrame:
    """Concise labelled fault snippets for detector assertions."""
    rng = np.random.default_rng(7)
    n = 300
    base = 0.18 + 0.002 * np.sin(np.arange(n) / 12) + rng.normal(0, 0.004, n)
    m = base.copy()
    fault = np.zeros(n, dtype=int)
    types = ["none"] * n
    m[30:46] = np.nan                       # missing
    m[60] = 0.95                            # spike
    m[90:104] = m[90]                        # stuck
    m[130:160] = base[130:160] + np.linspace(0, 0.06, 30)   # upward drift
    m[170:173] = 1.2                        # out of range
    fault[30:46] = fault[60] = 1
    fault[90:104] = fault[130:160] = fault[170:173] = 1
    types[30:46] = ["missing"] * 16
    types[60] = "spike"
    types[90:104] = ["stuck"] * 14
    types[130:160] = ["drift"] * 30
    types[170:173] = ["out_of_range"] * 3
    _h = np.arange(n) % 24
    df = pd.DataFrame({
        "farm_id": "T01", "soil_type": "sandy_loam", "crop_type": "tomato",
        "climate": "temperate",
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "temperature_c": rng.normal(25, 3, n),
        "humidity_pct": rng.normal(60, 10, n),
        "rainfall_mm": np.zeros(n), "solar_rad_mj_m2_h": np.zeros(n),
        "wind_speed_m_s": rng.normal(3, 1, n),
        "et0_mm": (0.1 + 0.5 * ((_h >= 6) & (_h < 18))).astype(float),
        "growth_stage": "mid", "kc": 1.0, "irrigation_mm": np.zeros(n),
        "etc_mm": 0.2, "drainage_mm": 0.0, "runoff_mm": 0.0, "water_stress_ks": 1.0,
        "soil_moisture_true": base, "soil_moisture_measured": np.clip(m, 0.0, 1.0),
        "target_low": 0.17, "target_high": 0.21, "sensor_fault_label": fault,
        "sensor_fault_type": types, "data_origin": "simulated", "split": "test",
    })
    return df


@pytest.fixture()
def prepared_small(small_farm):
    from src.preprocessing.pipeline import Preprocessor

    cfg = ForecastConfig(history=8, horizons=(3,), stride=1)
    return Preprocessor(history=cfg.history, horizons=cfg.horizons, stride=1).prepare(small_farm)