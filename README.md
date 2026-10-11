# AI-Predictive Smart Irrigation System (Software-Only, SIMULATED Data)

A research-grade reference implementation of an AI-powered smart-irrigation
decision-support system built on a **simulated virtual farm**. It combines
deep-learning soil-moisture forecasting, sensor fault detection, uncertainty
quantification, counterfactual irrigation simulation and an explainable
recommendation engine.

> **IMPORTANT — all data are SIMULATED.** Every CSV is produced by a
> reproducible generator (`scripts/generate_datasets.py`, seed `1000`, verified
> byte-identical to `data/raw/*.csv`). Nothing here is real field data, and the
> system is a recommendation layer only: it never drives any hardware.

## What it does

| Component | Modules | Purpose |
|---|---|---|
| Data generation | `scripts/generate_datasets.py`, `src/data_generation/virtual_farm.py` | hourly soil-water balance over 120 days, 9 farms (3 soils × 3 crops); faulty sensor version; OOD shift set (sand soil + heat wave) |
| Preprocessing | `src/preprocessing/pipeline.py` | time-ordered train/val/test splits, z-score scaling, sequence windows; leakage columns excluded |
| Fault detection | `src/fault_detection/detector.py` | rules (missing/out-of-range/spike/stuck/drift/conflict) + optional IsolationForest, per-hour reliability, delay-aware evaluation |
| Forecasting | `src/forecasting/` | CNN-LSTM, LSTM, quantile CNN-LSTM, linear regression, persistence baselines; early stopping + checkpointing |
| Uncertainty | `src/uncertainty/estimator.py` | deep ensemble (MC-style std) with multiplicative calibration + quantile intervals, coverage metrics |
| Counterfactuals | `src/counterfactual/simulator.py` | forward roll-outs of the soil-water simulator per irrigation candidate under identical weather |
| Decision engine | `src/decision_engine/engine.py` | weighted multi-objective score, feasibility constraints, uncertainty/reliability gates, conservative fallback |
| Explainability | `src/explainability/explainer.py` | human-readable reason for every recommendation |
| Evaluation | `src/evaluation/experiments.py` | forecast/fault/policy/ablation/shift reports |
| Interfaces | `app.py`, `src/api/main.py` | Streamlit dashboard + FastAPI service |

## Layout

```
configs/            soil & crop parameters, project.yaml
data/raw/           farm_clean.csv | farm_faulty.csv | farm_shift.csv
data/processed/     prepared artifacts
docs/               literature table + report skeleton
experiments/        run_experiments.py (7-step reproducible harness)
models/             cached checkpoints + training histories
reports/            JSON metrics + figures
src/                all source modules (config, pipeline, models, ...)
tests/              pytest suite
scripts/            dataset generator
```

## Quickstart

```bash
pip install -r requirements.txt

# full reproducibility guard
python -m pytest -q                          # 38+ tests

# fast sanity run of the whole pipeline
python experiments/run_experiments.py --quick

# full experimental suite (expected: several minutes on CPU)
python experiments/run_experiments.py        # or --epochs 20

# interactive dashboard
streamlit run app.py

# REST service (OpenAPI docs at /docs)
uvicorn src.api.main:app --reload
```

## Reproducibility & data provenance

- `SOIL`/`CROP` tables and the balance equations in
  `src/data_generation/virtual_farm.py` mirror the generator in
  `scripts/generate_datasets.py`; regeneration with seed `1000` reproduces the
  committed CSVs exactly (validated to max abs diff `0.0`).
- All splits are made **before** any scaling/modeling; leakage columns
  (`soil_moisture_true`, `etc_mm`, `drainage_mm`, `runoff_mm`, `water_stress_ks`,
  `sensor_fault_*`) are never model inputs.
- The counterfactual simulator uses the same physics as the data generator, so
  roll-outs are consistent with how the numbers were produced.

## Ethics & safety

This is an offline, decision-support research prototype on synthetic data.
Irrigation actions are only *recommended*; a real deployment requires
hardware-in-the-loop validation, on-site calibration and a review of local
regulatory and water-management constraints.

See `docs/literature.md` (verifiable sources) and `docs/report_skeleton.md`
(structure for the accompanying report/thesis).