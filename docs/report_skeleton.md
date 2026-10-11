# Report skeleton — AI-Predictive Smart Irrigation System

Fill the placeholders. Every figure/metric references `reports/*.json` and
`reports/figures/*.png` produced by `python experiments/run_experiments.py`.

## 1. Introduction & objectives
- Context: smart irrigation as decision support; the role of forecasting + uncertainty + fault awareness.
- Objectives: (a) predict soil moisture, (b) quantify uncertainty, (c) detect sensor faults,
  (d) recommend irrigation via counterfactual roll-outs, (e) evaluate under one reproducible setup.
- **Data disclaimer**: all experiments run on SIMULATED data (virtual farm), so numbers are
  illustrative of methodology, not field performance.

## 2. Related work
- Use `docs/literature.md` table [1]–[12]. Map each: ET0/soil-water physics, forecasting,
  uncertainty, fault detection, decision support.

## 3. Virtual farm & data
- Generator (seed 1000), soil/crop parameters (`configs/`), balance equation (P_eff + I_eff − ETc − D − R).
- Datasets: `farm_clean`, `farm_faulty` (sensor-fault types + segments), `farm_shift` (OOD sand + heat wave).
- Feature/target contract and leakage policy. Splits (time-ordered train/val/test).

## 4. Methods
### 4.1 Preprocessing            (src/preprocessing)
### 4.2 Sensor-fault detection   (src/fault_detection) — rules + IsolationForest, reliability, delay metric
### 4.3 Forecasting              (src/forecasting) — persistence/linear baselines, LSTM, CNN-LSTM, (quantile) variant
### 4.4 Uncertainty              (src/uncertainty) — deep ensemble, calibration, interval metrics
### 4.5 Counterfactual simulation(src/counterfactual) — candidate actions, shared weather, target band
### 4.6 Decision engine          (src/decision_engine) — multi-objective score, constraints, gates, fallback
### 4.7 Explainability           (src/explainability)

## 5. Evaluation & results (SIMULATED)
5.1 Forecast accuracy → `reports/forecasts.json` (MAE/RMSE by model × horizon; best epoch; vs. true)
5.2 Uncertainty → `reports/uncertainty.json` (coverage vs. target, mean std, calibration status)
5.3 Fault detection → `reports/faults.json` (precision/recall/F1 per farm, FAR, delay, reliability)
5.4 Policy comparison → `reports/policies.json` (water use, % below target, drainage, events)
5.5 Ablations → `reports/ablations.json` (full vs. no-counterfactual vs. no-uncertainty vs. no-fault-detection)
5.6 OOD shift stress → `reports/shift_stress.json` (degradation on unseen soil/heat wave)
Figures: `reports/figures/forecasts.png`, `policies.png`, `faults.png`.

## 6. Discussion
- Strengths, limitations of simulation-based evaluation, calibration gap, OOD behavior, model staleness, hardware safety.

## 7. Conclusions & future work
- Hardware-in-the-loop, real-field validation, online learning, multi-horizon scheduling, conformal calibration.

## Bibliography
- See `docs/literature.md`.