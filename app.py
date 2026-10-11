"""Streamlit dashboard for the AI smart-irrigation system (SIMULATED data).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.counterfactual.simulator import candidates_from_config
from src.service import get_runtime

st.set_page_config(page_title="AI Smart Irrigation (SIMULATED)", layout="wide",
                   page_icon=":droplet:")

MODEL_NAMES = {"lstm": "LSTM", "cnn_lstm": "CNN-LSTM", "persistence": "Persistence"}


# ------------------------------------------------------------------ cache -- #
@st.cache_resource(show_spinner="Building preprocessing + lazy models (SIMULATED)...")
def runtime_for(run_id: str):
    return get_runtime(config.FARM_CLEAN, run_id=run_id)


def available_run_ids() -> list[str]:
    seen, ordered = set(), []
    for p in sorted(config.MODELS_DIR.glob("*_h*.pt"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        parts = p.stem.split("_")
        for i, tok in enumerate(parts):
            if tok == "h" and i + 1 < len(parts):
                rid = "_".join(parts[i + 2:]) or "default"
                if rid not in seen:
                    seen.add(rid)
                    ordered.append(rid)
                break
    return ordered + [r for r in ["default"] if r not in seen]


def load_report(name: str) -> dict | list | None:
    p = config.REPORTS_DIR / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


PAGES = ["Overview", "Data & Sensors", "Deep Learning Prediction",
         "Counterfactual Simulation", "Explainable Recommendations",
         "Experiments & Evaluation"]

with st.sidebar:
    st.header(":droplet: Control panel")
    page = st.radio("Page", PAGES)
    run_id = st.selectbox("Model run id", available_run_ids(), index=0)
    st.caption("Models are cached under `models/*_h<horizon>_<run_id>.pt`; "
               "an unknown run_id trains on demand (SIMULATED).")
    quick = st.checkbox("Use sensor-fault data + reliability", value=False,
                        help="Feed fault-injected data so the dashboard reports sensor reliability.")

r = runtime_for(run_id)

fault_df = None
if quick:
    fault_df = pd.read_csv(config.FARM_FAULTY)

with st.sidebar:
    farms = r.farm_ids()
    farm_id = st.selectbox("Farm", farms)
    farm_df = r.farm_df(farm_id)
    n = len(farm_df)
    idx_default = int(np.median([i for i in range(24, n - 24, 24)]))
    norm_index = st.slider("Decision index (row)", 24, n - 25, int(idx_default))

st.title("AI Smart-Irrigation Decision Support — SIMULATED")
st.caption(f"Data source: `{config.FARM_CLEAN.name}` · {len(r.df):,} rows · farms "
           f"{', '.join(r.farm_ids())} · **all values are simulated** · the system only recommends, never actuates.")

# =============================================================== Overview === #
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Farms", len(r.farm_ids()))
    c2.metric("Sensors (columns)", len(farm_df.columns))
    c3.metric("Forecast history", f"{r.forecast_cfg.history} h")
    c4.metric("Horizons", ", ".join(map(str, r.forecast_cfg.horizons)))

    st.subheader("System pipeline")
    st.markdown("""
| Stage | Technique | Notes |
|---|---|---|
| Preprocessing | time-aware splits, z-score scaling, sequence windows | leakage columns excluded |
| Fault detection | rule-based sensor validation + error flags + reliability | optional ML (IsolationForest) |
| Forecasting | CNN-LSTM / LSTM / persistence / linear regression | trained on train split only |
| Uncertainty | deep ensemble + calibration (+ quantile CNN-LSTM) | std -> uncertainty score |
| Counterfactuals | soil-water balance rollouts per candidate | vs. target band |
| Decision | weighted multi-objective scoring + constraints | explanation always attached |
""")
    st.subheader("Decision candidates (default)")
    cand_df = pd.DataFrame([{"action": c.label, "amount_mm": c.amount_mm,
                             "kind": getattr(c, "candidate_kind", "based_on_theta"), "delay_h": c.delay_h}
                            for c in candidates_from_config(r.decision_cfg)])
    st.dataframe(cand_df, use_container_width=True)

    s = load_report("summary") or {}
    if s:
        st.subheader("Latest experiment run")
        st.json(s)
    st.info("Everything shown here is produced by reproducibly simulated physics "
            "(soil-water balance) and synthetic sensor faults — **never real field data**.")

# ===================================================== Data & Sensors ====== #
elif page == "Data & Sensors":
    st.subheader(f"`{farm_id}` — spread of a window + sensor reliability")
    win = st.slider("Window (hours)", 120, min(n - 25, 2000), 480)
    lo = max(0, norm_index - win // 2)
    seg = farm_df.iloc[lo: lo + win]
    c_hi = seg["target_high"] if "target_high" in seg else pd.Series([0.3] * len(seg))

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seg["timestamp"], y=seg["soil_moisture_measured"],
                             name="measured", line=dict(width=1.5)))
    if "soil_moisture_true" in seg.columns:
        fig.add_trace(go.Scatter(x=seg["timestamp"], y=seg["soil_moisture_true"],
                                 name="true", line=dict(width=1.5, dash="dot")))
    if "target_low" in seg.columns:
        fig.add_trace(go.Scatter(x=seg["timestamp"], y=seg["target_low"], name="target_low",
                                 line=dict(color="red", width=1)))
        fig.add_trace(go.Scatter(x=seg["timestamp"], y=seg["target_high"], name="target_high",
                                 line=dict(color="green", width=1)))
    fig.update_layout(yaxis_title="volumetric water content (m3/m3)", height=380)
    st.plotly_chart(fig, use_container_width=True)

    if "sensor_fault_label" in farm_df.columns:
        flagged = seg[seg["sensor_fault_label"] == 1]
        st.metric("Fault hours in window", f"{len(flagged)} / {len(seg)}")
        if len(flagged):
            st.write("Flagged timestamps:", flagged[["timestamp", "sensor_fault_type"]].head(20))
    with st.expander("Latest rows (all columns)"):
        st.dataframe(seg.tail(50), use_container_width=True)

    with st.expander("Sensor reliability (fault model)"):
        if fault_df is not None:
            out = r.fault_analysis(farm_id, fault_df)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=out["timestamp"], y=out["sensor_reliability"],
                                      name="reliability"))
            ax_f = out.loc[out["sensor_fault_label"] == 1]
            fig2.add_trace(go.Scatter(x=ax_f["timestamp"], y=ax_f["sensor_reliability"],
                                      mode="markers", name="flagged fault"))
            fig2.update_layout(height=260, yaxis_title="reliability [0,1]")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.write("Fault-injected dataset not loaded (side panel). Load to see reliability.")

# ================================================ Deep Learning Prediction = #
elif page == "Deep Learning Prediction":
    st.subheader(f"`{farm_id}` — learned forecasting with uncertainty band")
    kind = st.selectbox("Model", ["cnn_lstm", "lstm"], format_func=lambda k: MODEL_NAMES[k])
    horizon = st.selectbox("Horizon (h)", sorted(r.forecast_cfg.horizons), index=len(r.forecast_cfg.horizons) - 1)
    i = int(st.slider("Current time index (row)", 24, n - horizon, norm_index))

    fc = r.forecast(farm_id, i, horizon)
    st.write("**Point forecast at t+%d h:** `%.4f`  (90%% band [%.4f, %.4f], std=%.5f)  uncertainty score **%.2f**"
             % (horizon, fc["mean"], fc["lo"], fc["hi"], fc["std"], fc["uncertainty_score"]))

    import plotly.graph_objects as go

    hist = r.forecast_cfg.history
    lo = max(0, i - hist - 8)
    seg = farm_df.iloc[lo: i + 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seg["timestamp"], y=seg["soil_moisture_measured"],
                             name="measured (24h history)", line=dict(width=1.5)))
    if "target_low" in seg.columns:
        fig.add_trace(go.Scatter(x=seg["timestamp"], y=seg["target_low"], name="target_low",
                                 line=dict(color="red", width=1)))
    tgt = pd.to_datetime(fc["target_time"])
    fig.add_trace(go.Scatter(x=[tgt], y=[fc["mean"]], mode="markers", name="forecast mean",
                             marker=dict(size=12, color="black")))
    fig.add_trace(go.Scatter(x=[tgt], y=[fc["mean"]], name="90% band",
                             error_y=dict(type="data", symmetric=False, array=[fc["hi"] - fc["mean"]],
                                          arrayminus=[fc["mean"] - fc["lo"]]),
                             mode="markers", marker=dict(size=0)))
    fig.add_vline(x=pd.to_datetime(seg["timestamp"].iloc[-1]).isoformat(), line_dash="dash",
                  line_color="gray")
    fig.update_layout(yaxis_title="vWC (m3/m3)", height=360)
    st.plotly_chart(fig, use_container_width=True)

    # training curve from sidecar json
    meta_path = config.MODELS_DIR / f"{kind}_h{horizon}_{run_id}.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        hist_vals = meta.get("val_history", [])
        if hist_vals:
            st.subheader("Validation-loss (early stopping)")
            st.line_chart(pd.DataFrame({"val_loss": hist_vals}))
            st.caption(f"best epoch {meta.get('best_epoch')}, n epochs run {meta.get('n_epochs_run')}")
        else:
            st.caption("Validation history not saved for this cached checkpoint. Retrain to record it.")
    else:
        st.caption("Training curve unavailable (model loaded from cache). Retrain to record it.")

    rows = load_report("forecasts")
    if rows:
        mdf = pd.DataFrame(rows[rows["model"] == kind]) if isinstance(rows, list) else pd.DataFrame()
        if len(mdf):
            st.subheader("Test-set metrics (from experiments)")
            st.dataframe(mdf.set_index("horizon"), use_container_width=True)

# ============================================== Counterfactual Simulation === #
elif page == "Counterfactual Simulation":
    st.subheader(f"`{farm_id}` — counterfactual irrigation rollouts under the same weather")
    i = int(st.slider("Decision index (row)", 24, n - 2, min(norm_index, n - 2)))
    mode = st.selectbox("Forecast mode", ["persistent", "perfect"])

    defaults = candidates_from_config(r.decision_cfg)
    amounts = {}
    for c in defaults:
        if c.label == "no_irrigation":
            continue
        amounts[c.label] = st.slider(f"{c.label} amount (mm)", 0, 30, int(c.amount_mm), key=f"amt_{c.label}")
    cands = []
    from src.counterfactual.simulator import CandidateAction
    for c in defaults:
        if c.label == "no_irrigation":
            cands.append(c)
        else:
            cands.append(CandidateAction(c.label, "based_on_theta", amounts[c.label], c.delay_h))

    ctx, metrics, traj = r.scenarios(farm_id, i, theta0=float(farm_df["soil_moisture_measured"].iloc[i]),
                                     candidates=cands, mode=mode)

    import plotly.graph_objects as go

    fig = go.Figure()
    for col in traj.columns:
        fig.add_trace(go.Scatter(x=traj.index, y=traj[col], name=col, mode="lines"))
    hline = go.layout.Shape(type="line", x0=traj.index[0], x1=traj.index[-1],
                            y0=ctx["target_low"], y1=ctx["target_low"], line=dict(color="red", dash="dash"))
    hline2 = go.layout.Shape(type="line", x0=traj.index[0], x1=traj.index[-1],
                             y0=ctx["target_high"], y1=ctx["target_high"], line=dict(color="green", dash="dash"))
    fig.add_shape(hline); fig.add_shape(hline2)
    fig.update_layout(yaxis_title="vWC (m3/m3)", title="Predicted moisture band for each candidate",
                      height=400)
    st.plotly_chart(fig, use_container_width=True)

    mdf = pd.DataFrame(metrics).T
    st.subheader("Metrics per candidate")
    st.dataframe(mdf.round(3), use_container_width=True)
    bar = go.Figure()
    bar.add_trace(go.Bar(x=mdf.index, y=mdf["water_used_mm"], name="water used (mm)"))
    bar.add_trace(go.Bar(x=mdf.index, y=mdf["stress_area"], name="stress area"))
    bar.update_layout(height=320)
    st.plotly_chart(bar, use_container_width=True)

    with st.expander("Weather context (same for all candidates)"):
        ctx_df = pd.DataFrame(ctx)
        st.line_chart(ctx_df[["rain", "et0"]], height=200)
    st.info("Each roll-out uses the identical weather path; only the irrigation action differs.")

# ============================================ Explainable Recommendations = #
elif page == "Explainable Recommendations":
    st.subheader(f"`{farm_id}` — decision with explanation, alternatives and risk")
    i = int(st.slider("Decision index (row)", 24, n - 2, min(norm_index, n - 2)))
    use_fault = st.checkbox("Feed the sensor-reliability model (fault-injected farm)", value=quick)
    theta0 = st.number_input("Measured water content (m3/m3) — blank = last valid obs",
                             value=None, min_value=0.0, max_value=0.6, step=0.001)
    fdf_ = fault_df if use_fault else None
    if use_fault and fdf_ is None:
        fdf_ = pd.read_csv(config.FARM_FAULTY)

    res = r.recommend(farm_id, i, theta0=theta0, fault_df=fdf_)

    st.markdown(f"#### Recommended: **`{res['recommended_action']}`**  "
                f"({res['recommended_amount_mm']} mm, delay {res['recommended_delay_h']} h)  "
                f"total score {res['total_score']:.3f}")
    st.markdown(f"Observed θ0 = **{res['theta0']:.4f}** · target band "
                f"[{res['target_low']}, {res['target_high']}] · sensor reliability "
                f"**{res['sensor_reliability']:.2f}** · engine status **`{res['reliability_status']}`**")

    for line in res["explanation"]:
        st.markdown("- " + line)

    if res["fallback"]:
        st.warning("Conservative fallback triggered: the uncertainty/reliability gate blocked "
                   "every candidate, so **no irrigation action is recommended now.**")

    st.subheader("Alternative candidates (full decision table)")
    rows = []
    for label, a in res["candidates"].items():
        comps = a.get("components", {})
        best = " 🎯" if label == res["selected_candidate"] else ""
        rows.append({"action": label + best, "amount_mm": a["amount_mm"], "delay_h": a["delay_h"],
                     "feasible": a["feasible"], "total_score": a["total_score"],
                     "water": round(comps.get("water", 0), 3), "stress": round(comps.get("stress", 0), 3),
                     "overwater": round(comps.get("overwater", 0), 3), "risk": round(comps.get("risk", 0), 3)})
    st.dataframe(pd.DataFrame(rows).set_index("action"), use_container_width=True)

    st.subheader("Predicted moisture trajectory for the selected candidate")
    selected = [a for a in candidates_from_config(r.decision_cfg) if a.label == res["selected_candidate"]]
    if selected:
        ctx, metrics, traj = r.scenarios(farm_id, i, res["theta0"], selected, mode="persistent")
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=traj.index, y=traj.iloc[:, 0], mode="lines", name=traj.columns[0]))
        fig.add_shape(type="line", x0=traj.index[0], x1=traj.index[-1], y0=res["target_low"],
                      y1=res["target_low"], line=dict(color="red", dash="dash"))
        fig.add_shape(type="line", x0=traj.index[0], x1=traj.index[-1], y0=res["target_high"],
                      y1=res["target_high"], line=dict(color="green", dash="dash"))
        fig.update_layout(yaxis_title="vWC (m3/m3)", height=280)
        st.plotly_chart(fig, use_container_width=True)

# ============================================ Experiments & Evaluation ===== #
else:  # Experiments & Evaluation
    st.subheader("Experiment reports (`reports/*.json`, SIMULATED)")

    def show(name: str, title: str = None, table_keys: list | None = None):
        data = load_report(name)
        if data is None:
            st.caption(f"No `{name}.json` yet — run `python experiments/run_experiments.py`.")
            return
        st.markdown(f"### {title or name}")
        if isinstance(data, dict):
            if "aggregate" in data:
                st.dataframe(pd.DataFrame(data["aggregate"]), use_container_width=True)
            if "per_farm" in data:
                st.caption("Per-farm detail")
                flat = []
                for g in data["per_farm"]:
                    flat += [r if isinstance(r, dict) else (getattr(r, "__dict__", r)) for r in g]
                st.dataframe(pd.DataFrame(flat), use_container_width=True)
            if "overall" in data:
                st.json(data["overall"])
            elif "forecast" in data:
                st.json(data["forecast"])
        elif isinstance(data, list):
            st.dataframe(pd.DataFrame(data), use_container_width=True)

    show("forecasts", "Forecast MAE/RMSE")
    show("uncertainty", "Uncertainty calibration")
    show("faults", "Sensor-fault detection")
    show("policies", "Policy comparison")
    show("ablations", "Ablation study (F01)")
    show("shift_stress", "OOD shift-stress test")

    st.markdown("### Figures")
    for f in sorted((config.REPORTS_DIR / "figures").glob("*.png")):
        st.image(str(f), caption=f.name, use_container_width=True)