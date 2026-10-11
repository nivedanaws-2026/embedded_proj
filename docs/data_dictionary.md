# Irrigation project datasets (SIMULATED)

All data here come from a virtual-farm simulator (`generate_datasets.py`, fixed seeds, pandas/numpy only).
They are **not field measurements** and must be reported as simulated data. Re-run: `python generate_datasets.py`.

| File | Rows | Purpose |
|---|---|---|
| farm_clean.csv | 25,920 | 9 farms (3 soils x 3 crops), hourly, 120 days. Training/validation/test of forecasting & decisions |
| farm_faulty.csv | 25,920 | Same data; `soil_moisture_measured` corrupted with labelled faults (missing, spike, stuck, drift, out_of_range) |
| farm_shift.csv | 5,760 | Distribution-shift test: unseen soil (`sand`), heat wave (+5 C, days 70-85) |
| soil_params.csv | 4 | theta_wp, theta_fc, theta_sat, drainage coefficient, infiltration cap |
| crop_params.csv | 3 | Root depth, depletion fraction, moisture targets, Kc, stage lengths |

## Columns (farm_*.csv)
farm_id, soil_type, crop_type, climate, timestamp (hourly) | temperature_c, humidity_pct, rainfall_mm, solar_rad_mj_m2_h, wind_speed_m_s,
et0_mm (Priestley-Taylor, simplified) | growth_stage, kc | irrigation_mm (applied per hour, max 10 mm/h) | etc_mm, drainage_mm, runoff_mm,
water_stress_ks (1 = no stress) | soil_moisture_true (volumetric, m3/m3; simulator state, hidden in real deployment) |
soil_moisture_measured (true + N(0,0.004) noise; NaN where faulty) | target_low, target_high (crop moisture range) |
sensor_fault_label (0/1), sensor_fault_type | data_origin | split (train 70% / val 15% / test 15% by time; `shift_test` in shift file).

## Model use
- Inputs: measured moisture, temperature, humidity, rainfall, et0, irrigation_mm (past window only). Target: soil_moisture_true or measured at +1/3/6 h.
- Do NOT feed `soil_moisture_true`, `etc_mm`, `drainage_mm`, `runoff_mm`, `water_stress_ks` or fault labels as model inputs (leakage); use them for evaluation only.
- First 10% of each faulty series is kept clean as warm-up. Faults do not overlap.

## Simulator assumptions / limitations
- Water balance: dS = P_eff + I_eff - ETc - drainage - runoff (root zone, mm/h); drainage = k*(S - FC) above field capacity; runoff if inflow > infiltration cap.
- Irrigation policy that generated the history is a noisy threshold "farmer" rule, giving diverse actions for learning.
- Weather is stochastic, not from a real station; ET0 is a simplified formula; no validation against a reference model (e.g. AquaCrop) or field data.
- Results on these data show behaviour on simulated farms only, not real agronomic performance.

## Optional real data (verify access/licences yourself before use)
Real soil-moisture/weather sources to consider for validation: International Soil Moisture Network (ISMN), NASA POWER (weather/solar), 
FAO CLIMWAT/ETo references, and open datasets on Kaggle/Zenodo/UCI. Document source, units, interval and missing values if used.
