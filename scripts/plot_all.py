"""Run the saved simulation reports through the plotting pipeline.

Prints a compact simulation-output summary to the terminal and writes every
graph into reports/figures/. Purely a reporting/demo entry point - the
heavy lifting was already done by experiments/run_experiments.py.

Usage:
    python scripts/plot_all.py [--run-id full_e30]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402


def load(name: str) -> dict | list | None:
    p = config.REPORTS_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def short_summary() -> None:
    """Print the key numbers of the simulation run to the terminal."""
    sep = "=" * 78
    print(sep)
    print("SIMULATION OUTPUT SUMMARY (all values are SIMULATED)")
    print(sep)

    f = load("forecasts")
    if f:
        df = pd.DataFrame(f)
        print("\n[1] Forecast MAE/RMSE  (test split, SIMULATED)")
        print(df.pivot_table(index="horizon", columns="model", values="mae").round(5).to_string())
    u = load("uncertainty")
    if u:
        print("\n[2] Deep-ensemble uncertainty (horizon 6)")
        print(f"    mean-std={u['mean_std']:.5f}  interval-width={u['interval_width']:.5f}"
              f"  coverage={u['coverage']:.3f} vs target={u['target_coverage']:.2f}"
              f"  calibrated={u['calibrated']}")
    fa = load("faults")
    if fa:
        o = fa["overall"]
        print("\n[3] Sensor-fault detection (overall)")
        print(f"    precision={o['precision']:.3f} recall={o['recall']:.3f} f1={o['f1']:.3f}"
              f" false-alarm-rate={o['false_alarm_rate']:.4f} delay={o['detection_delay_h']:.1f}h"
              f" reliability-mean={o['reliability_mean']:.3f}")
    for name in ("policies", "ablations", "shift_stress"):
        d = load(name)
        if not d:
            continue
        print(f"\n[{name}] aggregate (SIMULATED)")
        agg = d["policies_aggregate"] if name == "shift_stress" else d.get("aggregate")
        print(pd.DataFrame(agg).round(4).to_string(index=False))
    if load("shift_stress"):
        print("\n[7] OOD shift note: engine time-below rose on unseen soil/heat wave "
              "(model uncertainty rises OOD) - expected and reported.")
    print(sep)
    print("Figures written to:", config.REPORTS_DIR / "figures")


def plot_all(run_id: str) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.counterfactual.simulator import candidates_from_config
    from src.service import Runtime

    figdir = config.REPORTS_DIR / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    rt = Runtime(run_id=run_id)
    farm_id = "F01"
    i = 600

    def save(fig, name: str) -> None:
        p = figdir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)

    plt.rcParams.update({"figure.dpi": 110})

    # --- 1. forecast MAE by horizon/model ---------------------------------
    f = load("forecasts")
    if f:
        df = pd.DataFrame(f)
        px = df.pivot_table(index="horizon", columns="model", values="mae")
        fig, ax = plt.subplots(figsize=(8, 4))
        px.plot.bar(ax=ax, rot=0)
        ax.set_ylabel("MAE (m3/m3)")
        ax.set_title("Forecast MAE by horizon and model (SIMULATED)")
        save(fig, "forecasts")

    # --- 2. uncertainty: coverage bar + interval width ----------------------
    u = load("uncertainty")
    if u:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        vals = [u["coverage"], u["target_coverage"]]
        bars = ax.bar(["coverage", "target (90%)"], vals,
                      color=["tab:blue", "tab:gray"])
        ax.axhline(0.9, color="k", lw=1, ls=":")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("fraction of targets inside interval")
        ax.set_title(f"Uncertainty calibration  (mean std={u['mean_std']:.4f}, "
                     f"width={u['interval_width']:.3f})")
        save(fig, "uncertainty")

    # --- 3. fault detection per farm + overall -------------------------------
    fa = load("faults")
    if fa:
        pf = pd.DataFrame(fa["per_farm"])
        fig, ax = plt.subplots(figsize=(9, 4))
        pf.plot(x="farm_id", y=["precision", "recall", "f1"], kind="bar",
                ax=ax, rot=0)
        ax.set_ylabel("score")
        ax.set_title("Sensor-fault detection per farm (SIMULATED faults)")
        save(fig, "faults")

    # --- 4. policy comparison -------------------------------------------------
    for nm, title in (("policies", "Policy comparison (9 farms)"),
                      ("shift_stress", "OOD shift-stress comparison (sand+heat wave)")):
        d = load(nm)
        if not d:
            continue
        agg = d["policies_aggregate"] if nm == "shift_stress" else d.get("aggregate")
        df = pd.DataFrame(agg).set_index("policy")
        df["pct_below"] = 100.0 * df["time_below_target_mean"]
        fig, ax = plt.subplots(figsize=(8, 4))
        df[["water_used_mm_mean", "pct_below"]].plot.bar(ax=ax, rot=30)
        ax.set_ylabel("water used (mm) / % time below target")
        ax.set_title(title + "  (SIMULATED)")
        save(fig, nm)

    # --- 5. ablations -----------------------------------------------------------
    ab = load("ablations")
    if ab:
        df = pd.DataFrame(ab["aggregate"]).set_index("policy")
        fig, ax = plt.subplots(figsize=(8, 4))
        df[["water_used_mm_mean"]].plot.bar(ax=ax, rot=30, color="tab:green")
        ax.set_ylabel("water used (mm)")
        ax.set_title("Ablation: contribution of each component (F01, SIMULATED)")
        save(fig, "ablations")

    # --- 6. example farm time series with target band & faults ------------------
    farm = rt.farm_df(farm_id)
    farm["timestamp"] = pd.to_datetime(farm["timestamp"])
    win = farm.iloc[300:1100]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(win["timestamp"], win["soil_moisture_measured"], lw=1, label="measured")
    ax.axhspan(win["target_low"].iloc[0], win["target_high"].iloc[0],
               color="green", alpha=0.10, label="target band")
    ax.set_ylabel("vWC (m3/m3)")
    ax.set_title(f"Farm {farm_id}: moisture under the target band (SIMULATED)")
    ax.legend(loc="lower left")
    ax.tick_params(axis="x", rotation=30)
    save(fig, "farm_example")

    # --- 7. forecast + uncertainty band at one decision point ---------------------
    horizon = rt.forecast_cfg.horizons[-1]
    fc = rt.forecast(farm_id, i, horizon)
    lo = max(0, i - rt.forecast_cfg.history)
    seg = farm.iloc[lo:i + 1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(seg["timestamp"], seg["soil_moisture_measured"], lw=1.2, label="history (measured)")
    ax.axhspan(farm["target_low"].iloc[i], farm["target_high"].iloc[i],
               color="green", alpha=0.10)
    t0 = pd.to_datetime(farm["timestamp"].iloc[i])
    t1 = pd.to_datetime(fc["target_time"])
    ax.annotate(f"forecast {fc['mean']:.3f}\nband [{fc['lo']:.3f},{fc['hi']:.3f}]",
                xy=(t1, fc["mean"]), xytext=(t1, fc["hi"] + 0.01),
                arrowprops=dict(arrowstyle="->"), ha="center")
    ax.axvline(t0, color="k", ls=":", lw=1)
    ax.set_ylabel("vWC (m3/m3)")
    ax.set_title(f"24h history + {horizon}h forecast with 90% interval (SIMULATED)")
    ax.tick_params(axis="x", rotation=35)
    save(fig, "forecast_interval")

    # --- 8. counterfactual scenario trajectories at a decision point ----------------
    theta0 = float(farm["soil_moisture_measured"].iloc[i])
    candidates = candidates_from_config(rt.decision_cfg)
    ctx, metrics, traj = rt.scenarios(farm_id, i, theta0, candidates, mode="persistent")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for col in traj.columns:
        ax.plot(traj.index, traj[col], label=col, lw=1.4)
    ax.axhline(ctx["target_low"], color="red", ls="--", lw=1)
    ax.axhline(ctx["target_high"], color="green", ls="--", lw=1)
    ax.legend(fontsize=8)
    ax.set_ylabel("vWC (m3/m3)")
    ax.set_title("Counterfactual roll-outs under identical weather (SIMULATED)")
    ax.tick_params(axis="x", rotation=30)
    save(fig, "counterfactuals")

    # --- 9. decision scores for the same point ---------------------------------------
    rec = rt.recommend(farm_id, i, candidates=candidates, theta0=theta0)
    rows = []
    for label, a in rec["candidates"].items():
        rows.append({"action": label, "total_score": a["total_score"],
                     "water": a["components"]["water"], "stress": a["components"]["stress"],
                     "overwater": a["components"]["overwater"], "risk": a["components"]["risk"]})
    sdf = pd.DataFrame(rows).set_index("action")
    fig, ax = plt.subplots(figsize=(8, 4))
    sdf.plot.bar(ax=ax, rot=30)
    chosen = rec["selected_candidate"]
    ax.get_xticklabels()[list(sdf.index).index(chosen) if chosen in sdf.index else 0].set_color("red")
    ax.set_ylabel("normalized score (lower = better)")
    ax.set_title(f"Decision scores - selected: {chosen} (SIMULATED)")
    save(fig, "decision_scores")

    # --- 10. training/val loss curves from cached checkpoints ----------------------------
    loss_figs = 0
    for kind in ("cnn_lstm", "lstm"):
        for h in rt.forecast_cfg.horizons:
            mp = config.MODELS_DIR / f"{kind}_h{h}_{run_id}.json"
            if not mp.exists():
                continue
            meta = json.loads(mp.read_text())
            hist = meta.get("val_history", [])
            if not hist:
                continue
            fig, ax = plt.subplots(figsize=(6.5, 3.4))
            ax.plot(range(1, len(hist) + 1), hist)
            ax.set_xlabel("epoch")
            ax.set_ylabel("val loss (MSE)")
            ax.set_title(f"{kind.upper()} h={h} - early stopping (SIMULATED)")
            save(fig, f"train_curve_{kind}_h{h}")
            loss_figs += 1
    if loss_figs == 0:
        print("(no cached training histories - run the full experiment to get loss curves)")

    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="full_e30")
    args = ap.parse_args()
    short_summary()
    made = plot_all(args.run_id)
    print(f"Plotted {len(made)} figures:")
    for p in made:
        print("  -", p)


if __name__ == "__main__":
    main()