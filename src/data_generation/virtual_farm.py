"""Virtual-farm dataset generator.

ALL DATA PRODUCED HERE ARE SIMULATED, not field measurements. Fixed seeds make
runs reproducible. The water balance (root zone, mm/h) is:

    dS = P_eff + I_eff - ETc - Drainage - Runoff

This module is the canonical generator; ``scripts/generate_datasets.py`` is a
thin CLI wrapper that writes into the configured data folders.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src import config

HOURS = 24 * 120  # 120-day season, hourly

_DEFAULT_SOILS = [
    # soil, theta_wp, theta_fc, theta_sat, drain_k(1/h), infil_cap(mm/h)
    ("sandy_loam", 0.10, 0.22, 0.43, 0.020, 30.0),
    ("loam", 0.12, 0.27, 0.46, 0.010, 15.0),
    ("clay_loam", 0.17, 0.33, 0.50, 0.004, 6.0),
    ("sand", 0.05, 0.12, 0.40, 0.050, 50.0),  # unseen soil, shift set only
]
_DEFAULT_CROPS = [
    # crop, root_depth_m, p_depletion, target_low_frac, target_high_frac, kc_ini, kc_mid, kc_end,
    # len_ini_d, len_dev_d, len_mid_d
    ("tomato", 0.50, 0.40, 0.60, 0.90, 0.60, 1.15, 0.80, 25, 30, 40),
    ("maize", 0.80, 0.55, 0.55, 0.90, 0.30, 1.20, 0.60, 20, 35, 40),
    ("wheat", 0.70, 0.55, 0.50, 0.90, 0.40, 1.15, 0.30, 20, 30, 40),
]

CLIMATES = {  # mean T, diurnal amplitude, daily wet-day prob, mean storm mm, cloudiness
    "hot_dry": dict(Tm=29, Ta=7, pw=0.05, rain=12, cloud=0.15),
    "temperate": dict(Tm=24, Ta=6, pw=0.15, rain=10, cloud=0.30),
    "humid_wet": dict(Tm=27, Ta=4, pw=0.20, rain=10, cloud=0.50),
}


def _load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prefer the on-disk parameter tables (configs/) if present, else defaults."""
    if config.SOIL_PARAMS.exists() and config.CROP_PARAMS.exists():
        soils = pd.read_csv(config.SOIL_PARAMS).set_index("soil")
        crops = pd.read_csv(config.CROP_PARAMS).set_index("crop")
        return soils, crops
    soils = pd.DataFrame(_DEFAULT_SOILS, columns=[
        "soil", "theta_wp", "theta_fc", "theta_sat", "drain_k_per_h", "infil_cap_mm_h"]
    ).set_index("soil")
    crops = pd.DataFrame(_DEFAULT_CROPS, columns=[
        "crop", "root_depth_m", "p_depletion", "target_low_frac_of_TAW",
        "target_high_frac_of_TAW", "kc_ini", "kc_mid", "kc_end",
        "len_ini_d", "len_dev_d", "len_mid_d"]).set_index("crop")
    return soils, crops


SOILS, CROPS = _load_tables()


def weather(rng: np.random.Generator, clim: str, start_doy: int = 120,
            heatwave: bool = False) -> pd.DataFrame:
    c = CLIMATES[clim]
    n = HOURS
    t = np.arange(n)
    hod = t % 24
    day = t // 24
    seasonal = 2.0 * np.sin(2 * np.pi * (start_doy + day - 80) / 365)
    daily_anom = np.repeat(rng.normal(0, 1.5, n // 24 + 1), 24)[:n]
    T = (c["Tm"] + seasonal + daily_anom
         + c["Ta"] * np.sin(2 * np.pi * (hod - 9) / 24) + rng.normal(0, .3, n))
    if heatwave:
        T = T + 5.0 * ((day >= 70) & (day < 85))
    cloud = np.clip(np.repeat(rng.beta(2, 2, n // 24 + 1) * c["cloud"] * 2, 24)[:n], 0, 0.95)
    Rs = np.clip(3.0 * np.sin(np.pi * (hod - 6) / 12), 0, None) * (1 - 0.75 * cloud ** 1.5)
    es = 0.6108 * np.exp(17.27 * T / (T + 237.3))
    rh = np.clip(100 * (0.55 * es * (1.0 if clim != "hot_dry" else .8)) / es
                 + 12 * np.cos(2 * np.pi * (hod - 15) / 24) * (-1)
                 + 25 * cloud + rng.normal(0, 2, n) + (0 if clim != "hot_dry" else -10), 20, 100)
    wind = np.clip(rng.gamma(4, .6, n) * (1 + 0.3 * np.sin(2 * np.pi * (hod - 14) / 24)), 0, 12)
    rain = np.zeros(n)
    for d in range(n // 24):
        if rng.random() < c["pw"]:
            total = rng.gamma(1.5, c["rain"] / 1.5)
            dur = rng.integers(1, 7)
            s = rng.integers(0, 24 - dur)
            w = rng.dirichlet(np.ones(dur))
            rain[d * 24 + s:d * 24 + s + dur] += total * w
            rh[d * 24 + s:d * 24 + 24] = np.clip(rh[d * 24 + s:d * 24 + 24] + 15, 20, 100)
            Rs[d * 24 + s:d * 24 + s + dur] *= .4
    # ET0: Priestley-Taylor, hourly (simplification; documented in docs/)
    delta = 4098 * es / (T + 237.3) ** 2
    gamma = 0.066
    ET0 = np.clip(0.65 * 1.26 * delta / (delta + gamma) * (0.65 * Rs) / 2.45
                  * (1 + 0.04 * wind) * (1.15 - rh / 200), 0, None)
    return pd.DataFrame(dict(
        temperature_c=T, humidity_pct=rh, rainfall_mm=rain,
        solar_rad_mj_m2_h=Rs, wind_speed_m_s=wind, et0_mm=ET0))


def kc_stage(crop: str, day: int) -> tuple[str, float]:
    p = CROPS.loc[crop]
    a, b, c = p.len_ini_d, p.len_ini_d + p.len_dev_d, p.len_ini_d + p.len_dev_d + p.len_mid_d
    if day < a:
        return "initial", p.kc_ini
    if day < b:
        return "development", p.kc_ini + (p.kc_mid - p.kc_ini) * (day - a) / (b - a)
    if day < c:
        return "mid", p.kc_mid
    return "late", p.kc_mid + (p.kc_end - p.kc_mid) * min(1, (day - c) / max(1, 120 - c))


def simulate_farm(farm_id: str, soil: str, crop: str, clim: str, seed: int,
                  heatwave: bool = False, start_doy: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s = SOILS.loc[soil]
    cp = CROPS.loc[crop]
    w = weather(rng, clim, start_doy, heatwave)
    n = HOURS
    Zr = cp.root_depth_m * 1000.0
    TAW = (s.theta_fc - s.theta_wp) * Zr
    S = (s.theta_wp + rng.uniform(0.55, 0.85) * (s.theta_fc - s.theta_wp)) * Zr
    thr = rng.uniform(0.50, 0.65)  # synthetic "farmer" trigger -> action diversity
    rows = []
    queue = 0.0
    for h in range(n):
        d = h // 24
        hod = h % 24
        stage, kc = kc_stage(crop, d)
        theta = S / Zr
        if hod == 6:  # daily decision at 06:00 using noisy rule + random events
            frac = (theta - s.theta_wp) / (s.theta_fc - s.theta_wp)
            if frac < thr + rng.normal(0, .05):
                queue += rng.choice([10., 15., 20., 25.])
            elif rng.random() < 0.02:
                queue += rng.choice([6., 10.])
        irr = min(queue, 0.8 * s.infil_cap_mm_h, 10.0)
        queue -= irr
        inflow = w.rainfall_mm[h] + irr
        infil = min(inflow, s.infil_cap_mm_h)
        runoff = inflow - infil
        Dr = max(0.0, (s.theta_fc * Zr) - S)
        Ks = 1.0 if Dr <= cp.p_depletion * TAW else max(0.0, (TAW - Dr) / ((1 - cp.p_depletion) * TAW))
        etc = kc * w.et0_mm[h] * Ks
        S_new = S + infil - etc
        drain = 0.0
        if S_new > s.theta_fc * Zr:
            drain = s.drain_k_per_h * (S_new - s.theta_fc * Zr)
            S_new -= drain
        if S_new > s.theta_sat * Zr:  # saturation overflow -> extra drainage
            drain += S_new - s.theta_sat * Zr
            S_new = s.theta_sat * Zr
        S = max(S_new, s.theta_wp * Zr * 0.5)  # air-dry floor
        theta_true = S / Zr
        rows.append((h, stage, round(kc, 3), irr, etc, drain, runoff, Ks, theta_true))
    sim = pd.DataFrame(rows, columns=["h", "growth_stage", "kc", "irrigation_mm", "etc_mm",
                                      "drainage_mm", "runoff_mm", "water_stress_ks", "soil_moisture_true"])
    df = pd.concat([w, sim.drop(columns="h")], axis=1)
    df.insert(0, "timestamp", pd.date_range("2025-05-01", periods=n, freq="h"))
    df.insert(0, "farm_id", farm_id)
    df.insert(1, "soil_type", soil)
    df.insert(2, "crop_type", crop)
    df.insert(3, "climate", clim)
    noise = rng.normal(0, 0.004, n)
    df["soil_moisture_measured"] = np.clip(df.soil_moisture_true + noise, 0.0, s.theta_sat).round(4)
    df["target_low"] = (s.theta_wp + cp.target_low_frac_of_TAW * (s.theta_fc - s.theta_wp)).round(4)
    df["target_high"] = (s.theta_wp + cp.target_high_frac_of_TAW * (s.theta_fc - s.theta_wp)).round(4)
    df["sensor_fault_label"] = 0
    df["sensor_fault_type"] = "none"
    df["data_origin"] = "simulated"
    n_tr, n_va = int(.70 * n), int(.15 * n)
    df["split"] = ["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va)
    for c in ["temperature_c", "humidity_pct", "rainfall_mm", "solar_rad_mj_m2_h", "wind_speed_m_s",
              "et0_mm", "etc_mm", "drainage_mm", "runoff_mm", "water_stress_ks", "soil_moisture_true"]:
        df[c] = df[c].round(4)
    return df


def inject_faults(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Inject labelled sensor faults (missing / spike / stuck / drift / out_of_range)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    col = out.columns.get_loc("soil_moisture_measured")
    lab = out.columns.get_loc("sensor_fault_label")
    typ = out.columns.get_loc("sensor_fault_type")
    plan = [("missing", 4), ("spike", 5), ("stuck", 3), ("drift", 2), ("out_of_range", 3)]
    taken = np.zeros(n, bool)
    for kind, k in plan:
        for _ in range(k):
            for _try in range(50):
                if kind == "missing":
                    L = rng.integers(3, 25)
                elif kind == "spike":
                    L = 1
                elif kind == "stuck":
                    L = rng.integers(12, 49)
                elif kind == "drift":
                    L = rng.integers(36, 97)
                else:
                    L = rng.integers(1, 4)
                s0 = rng.integers(int(.1 * n), n - L - 1)  # keep first 10% clean for warm-up
                if not taken[max(0, s0 - 6):s0 + L + 6].any():
                    break
            else:
                continue
            sl = slice(s0, s0 + L)
            taken[sl] = True
            if kind == "missing":
                out.iloc[sl, col] = np.nan
            elif kind == "spike":
                out.iloc[s0, col] = np.clip(out.iloc[s0, col] + rng.choice([-1, 1]) * rng.uniform(.12, .2), 0, 1.0)
            elif kind == "stuck":
                out.iloc[sl, col] = out.iloc[s0, col]
            elif kind == "drift":
                ramp = np.linspace(0, rng.choice([-1, 1]) * rng.uniform(.05, .10), L)
                out.iloc[sl, col] = np.clip(out.iloc[sl, col].values + ramp, 0, 1.0)
            else:
                out.iloc[sl, col] = rng.choice([0.0, 1.2, -0.05])
            out.iloc[sl, lab] = 1
            out.iloc[sl, typ] = kind
    return out


def generate_all(out_dir: Path | None = None, seed: int = 1000) -> dict[str, pd.DataFrame]:
    """Generate clean, faulty and distribution-shift datasets. Writes CSVs if out_dir given."""
    out_dir = Path(out_dir) if out_dir else config.DATA_RAW
    out_dir.mkdir(parents=True, exist_ok=True)
    clims = list(CLIMATES)
    frames = []
    k = 0
    for soil in ["sandy_loam", "loam", "clay_loam"]:
        for crop in ["tomato", "maize", "wheat"]:
            frames.append(simulate_farm(f"F{k+1:02d}", soil, crop, clims[k % 3], seed=seed + k))
            k += 1
    clean = pd.concat(frames, ignore_index=True)
    faulty = pd.concat([inject_faults(f, seed=seed + 1000 + i) for i, f in enumerate(frames)], ignore_index=True)
    shift = pd.concat([
        simulate_farm("S01", "sand", "tomato", "hot_dry", seed=seed + 2001, heatwave=True),
        simulate_farm("S02", "sand", "maize", "hot_dry", seed=seed + 2002, heatwave=True),
    ], ignore_index=True)
    shift["split"] = "shift_test"
    shift["data_origin"] = "simulated_shift"

    clean.to_csv(out_dir / "farm_clean.csv", index=False)
    faulty.to_csv(out_dir / "farm_faulty.csv", index=False)
    shift.to_csv(out_dir / "farm_shift.csv", index=False)
    SOILS.reset_index().to_csv(config.SOIL_PARAMS, index=False)
    CROPS.reset_index().to_csv(config.CROP_PARAMS, index=False)
    return {"clean": clean, "faulty": faulty, "shift": shift}


if __name__ == "__main__":  # pragma: no cover
    res = generate_all()
    print({k: v.shape for k, v in res.items()})
