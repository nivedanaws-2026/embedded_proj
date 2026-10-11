"""Run the full experimental suite and write reproducible reports.

Steps (see README / docs):
  1. data loading + preprocessing
  2. forecast comparison (persistence / linear regression / LSTM / CNN-LSTM)
  3. uncertainty estimation (deep ensemble) + calibration metrics
  4. sensor fault detection evaluation
  5. irrigation-policy comparison (fixed threshold vs forecast-rule vs full system)
  6. ablation study (no uncertainty / no counterfactual / no fault detection)
  7. stress test on the OOD shift set (unseen soil + heat wave)

Usage:
    python experiments/run_experiments.py            # full run (may take time)
    python experiments/run_experiments.py --quick    # 1 epoch, 2 ensemble members
    python experiments/run_experiments.py --epochs 20 --n-farms 3

All report JSONs and figures are written into reports/. Results are SIMULATED.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.config import ForecastConfig  # noqa: E402
from src.evaluation.experiments import write_json_report  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_soil_water(farm_df: pd.DataFrame):
    from src.soil_water_simulator.simulator import SoilWaterSimulator

    soils = config.load_soil_params()
    crops = config.load_crop_params()
    return SoilWaterSimulator.from_params(soils.loc[farm_df["soil_type"].iloc[0]],
                                          crops.loc[farm_df["crop_type"].iloc[0]])


def step1_preprocess(quick: bool):
    from src.preprocessing.pipeline import Preprocessor

    cfg = ForecastConfig()
    common = dict(history=cfg.history, horizons=cfg.horizons, stride=1)
    clean = pd.read_csv(config.FARM_CLEAN)
    if quick:
        clean = clean[clean["farm_id"] <= "F03"].copy()
    prep = Preprocessor(**common).prepare(clean)
    shift = pd.read_csv(config.FARM_SHIFT)
    prep_shift = Preprocessor(**common).prepare(shift, scaler_fit="all")
    return clean, prep, shift, prep_shift, cfg


def step2_forecasts(clean, prep, cfg, quick: bool, run_id: str):
    from src.evaluation.experiments import evaluate_forecasts

    train_nn = {"lstm": True, "cnn_lstm": True,
                "quantile_cnn_lstm": False}
    rows = evaluate_forecasts(prep, train_nn=train_nn, cfg=cfg, run_id=run_id)
    from src.evaluation.experiments import write_json_report
    p = write_json_report("forecasts", rows)
    log(f"forecast metrics: {len(rows)} rows -> {p.name}")
    print(pd.DataFrame(rows).to_string(index=False))
    return rows


def step3_uncertainty(prep, cfg, quick: bool, run_id: str):
    from src.evaluation.experiments import evaluate_uncertainty, write_json_report
    from src.uncertainty.estimator import EnsembleUncertainty

    horizon = cfg.horizons[-1]
    n_members = 2 if quick else cfg.ensemble_size
    ens = EnsembleUncertainty()
    t0 = time.time()
    ens.fit(prep, horizon, n_members=n_members, cfg=cfg,
            max_epochs=1 if quick else cfg.epochs)
    log(f"ensemble trained ({n_members} members) in {time.time()-t0:.0f}s")
    ens.save(config.MODELS_DIR / f"ensemble_h{horizon}_{run_id}")
    m = evaluate_uncertainty(prep, ens, horizon, coverage=0.90)
    m["ensemble_size"] = n_members
    p = write_json_report("uncertainty", m)
    log(f"uncertainty: {m}")
    return ens, m


def step4_faults():
    from src.evaluation.experiments import (fault_detection_eval,
                                            fault_detection_eval_overall,
                                            write_json_report)

    faulty = pd.read_csv(config.FARM_FAULTY)
    overall = fault_detection_eval_overall(faulty)
    per_farm = fault_detection_eval(faulty)
    p = write_json_report("faults", {"overall": overall, "per_farm": per_farm})
    log(f"fault detection overall: {overall}")
    print(pd.DataFrame(per_farm).to_string(index=False))
    return overall, per_farm


def step5_policies(clean, prep, model, uncertainty, n_farms, quick: bool):
    from src.evaluation.experiments import (aggregate_policy_results,
                                            run_policy_comparison, write_json_report)

    farms = sorted(clean["farm_id"].unique())[:n_farms]
    all_res = []
    for fid in farms:
        sw = make_soil_water(clean[clean["farm_id"] == fid])
        res = run_policy_comparison(fid, clean, sw, prep, model, uncertainty, horizon=6)
        all_res.append(res)
    agg = aggregate_policy_results(all_res)
    p = write_json_report("policies", {"per_farm": [
        [r.__dict__ for r in res] for res in all_res], "aggregate": agg})
    log(f"policy comparison: {len(farms)} farms")
    print(pd.DataFrame(agg).to_string(index=False))
    return agg


def step6_ablations(clean, prep, model, uncertainty):
    from src.evaluation.experiments import (aggregate_policy_results,
                                            run_ablations, write_json_report)

    faulty = pd.read_csv(config.FARM_FAULTY)
    sw = make_soil_water(clean[clean["farm_id"] == "F01"])
    res = run_ablations("F01", clean, sw, prep, model, uncertainty, faulty_df=faulty)
    agg = aggregate_policy_results([res])
    write_json_report("ablations", {"per_farm": [r.__dict__ for r in res], "aggregate": agg})
    log("ablation study (F01, incl. faulted sensor data)")
    print(pd.DataFrame(agg).to_string(index=False))
    return agg


def step7_shift(shift, prep_shift, model, uncertainty, cfg, quick: bool):
    from src.evaluation.experiments import (aggregate_policy_results,
                                            evaluate_uncertainty, run_policy_comparison,
                                            write_json_report)
    from src.forecasting.train import predict

    te = prep_shift.test(cfg.horizons[-1])
    mean = predict(model, te["X"])
    m = evaluate_uncertainty(prep_shift, uncertainty, cfg.horizons[-1], coverage=0.90)
    m["model"] = "cnn_lstm_transfer"
    m["mae"] = round(float(np.mean(np.abs(mean - te["y"]))), 5)
    m["mae_vs_true"] = round(float(np.mean(np.abs(mean - te["y_true"]))), 5)

    farms = sorted(shift["farm_id"].unique())
    all_res = []
    for fid in farms:
        sw = make_soil_water(shift[shift["farm_id"] == fid])
        res = run_policy_comparison(fid, shift, sw, prep_shift, model, uncertainty, horizon=6)
        all_res.append(res)
    agg = aggregate_policy_results(all_res)
    out = {"forecast": m, "policies_aggregate": agg,
           "per_farm": [[r.__dict__ for r in res] for res in all_res]}
    write_json_report("shift_stress", out)
    log("OOD shift stress test (sand soil + heat wave)")
    print(pd.DataFrame(agg).to_string(index=False))
    return out


def make_figures() -> None:
    """Small matplotlib figures from the reports (also referenced by the report)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = config.REPORTS_DIR / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for name in ("forecasts", "policies", "faults"):
        f = config.REPORTS_DIR / f"{name}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        fig, ax = plt.subplots(figsize=(8, 4))
        if name == "forecasts":
            rows = data if isinstance(data, list) else []
            df = pd.DataFrame(rows)
            px = df.pivot_table(index="horizon", columns="model", values="mae")
            px.plot.bar(ax=ax, rot=0)
            ax.set_ylabel("MAE (m3/m3)"); ax.set_title("Forecast MAE by horizon and model (SIMULATED)")
        elif name == "policies":
            agg = data["aggregate"] if isinstance(data, dict) else data
            df = pd.DataFrame(agg).set_index("policy")
            df["pct_below"] = 100.0 * df["time_below_target_mean"]
            df[["water_used_mm_mean"]].plot.bar(ax=ax, rot=30, legend=True, color="tab:blue")
            ax.set_ylabel("mean water used (mm)")
            ax2 = ax.twinx()
            df[["pct_below"]].plot.bar(ax=ax2, rot=30, legend=True, color="tab:orange", alpha=0.6)
            ax2.set_ylabel("% hours below target")
            ax.set_title("Policy comparison: water use vs time below target (SIMULATED)")
        else:
            d = data["overall"] if isinstance(data, dict) else {}
            keys = ["precision", "recall", "f1", "false_alarm_rate"]
            ax.bar(keys, [float(d.get(k, 0)) for k in keys]);
            ax.set_title("Fault detection metrics (SIMULATED)")
        fig.tight_layout()
        fig.savefig(figdir / f"{name}.png", dpi=120)
        plt.close(fig)
    log(f"figures -> {figdir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Full irrigation-system experiments (SIMULATED data).")
    ap.add_argument("--quick", action="store_true", help="1 epoch, 2 ensemble members, 3 farms")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-farms", type=int, default=9)
    ap.add_argument("--skip-shift", action="store_true")
    args = ap.parse_args()
    quick = args.quick
    config.set_global_seed(config.SEED)

    t = time.time()
    log("step 1 preprocessing")
    clean, prep, shift, prep_shift, cfg = step1_preprocess(quick)
    if args.epochs:
        cfg.epochs = args.epochs
    if quick:
        cfg.epochs = 1
    run_id = f"{'quick' if quick else 'full'}_e{cfg.epochs}"

    log("step 2 forecast comparison")
    rows = step2_forecasts(clean, prep, cfg, quick, run_id)

    log("step 3 uncertainty")
    ens, unc = step3_uncertainty(prep, cfg, quick, run_id)

    log("step 4 fault detection")
    step4_faults()

    log("step 5 policy comparison")
    from src.forecasting.train import load_or_train
    model, _ = load_or_train(prep, 6, kind="cnn_lstm", cfg=cfg, run_id=run_id)
    n_farms = 3 if quick else args.n_farms
    step5_policies(clean, prep, model, ens, n_farms, quick)

    log("step 6 ablations")
    step6_ablations(clean, prep, model, ens)

    if not args.skip_shift:
        log("step 7 shift stress")
        step7_shift(shift, prep_shift, model, ens, cfg, quick)

    make_figures()
    write_json_report("summary", {
        "quick": quick, "seed": config.SEED, "elapsed_s": round(time.time() - t, 1),
        "config": {"history": cfg.history, "horizons": list(cfg.horizons), "epochs": cfg.epochs,
                   "ensemble_size": cfg.ensemble_size}})
    log(f"DONE in {time.time()-t:.0f}s. Reports in {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()