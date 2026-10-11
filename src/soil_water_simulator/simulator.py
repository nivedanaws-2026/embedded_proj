"""Virtual soil-water balance simulator (independent of the deep-learning models).

Root-zone water balance, mm/h:

    dS / dt = P_eff + I_eff - ETc - Drainage - Runoff

State is tracked as stored water (mm) and exposed as volumetric soil moisture
(theta = S / root depth). The scheme mirrors ``data_generation.virtual_farm`` so
that counterfactual roll-outs are consistent with the dataset's physics:

  * effective infiltration = min(inflow, infiltration capacity); the excess is runoff;
  * evaporation/transpiration is stressed (Ks<1) once depletion exceeds the
    crop's readily-available-water fraction;
  * free drainage occurs above field capacity; saturation overflow is drained too;
  * an air-dry floor prevents physically impossible drying.

SIMPLIFICATIONS (documented; NOT validated against AquaCrop or field data):
  * single root-zone bucket, no deep-reserve layer, no capillary rise;
  * no crop-cover dynamics beyond the daily Kc used in data generation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SoilWaterSimulator:
    theta_wp: float
    theta_fc: float
    theta_sat: float
    drain_k_per_h: float
    infil_cap_mm_h: float
    root_depth_m: float
    p_depletion: float

    @classmethod
    def from_params(cls, soil_row: pd.Series, crop_row: pd.Series) -> "SoilWaterSimulator":
        return cls(theta_wp=float(soil_row["theta_wp"]),
                   theta_fc=float(soil_row["theta_fc"]),
                   theta_sat=float(soil_row["theta_sat"]),
                   drain_k_per_h=float(soil_row["drain_k_per_h"]),
                   infil_cap_mm_h=float(soil_row["infil_cap_mm_h"]),
                   root_depth_m=float(crop_row["root_depth_m"]),
                   p_depletion=float(crop_row["p_depletion"]))

    @property
    def Zr(self) -> float:
        """Root-zone depth in mm."""
        return self.root_depth_m * 1000.0

    @property
    def TAW(self) -> float:
        """Total available water in the root zone (mm)."""
        return (self.theta_fc - self.theta_wp) * self.Zr

    def storage(self, theta: float) -> float:
        return theta * self.Zr

    def moisture(self, S: float) -> float:
        return S / self.Zr

    def step(self, theta: float, rain_mm: float, irr_mm: float, et0_mm: float,
             kc: float, dt_h: float = 1.0) -> dict:
        """Advance one time step; returns new theta and flux components (mm)."""
        S = self.storage(float(theta))
        inflow = rain_mm + irr_mm
        infil = min(inflow, self.infil_cap_mm_h)
        runoff = inflow - infil
        Dr = max(0.0, self.theta_fc * self.Zr - S)               # depletion below FC
        if Dr <= self.p_depletion * self.TAW:
            Ks = 1.0
        else:
            Ks = max(0.0, (self.TAW - Dr) / ((1 - self.p_depletion) * self.TAW))
        etc = kc * et0_mm * Ks * dt_h
        S_new = S + infil * dt_h - etc
        drain = 0.0
        if S_new > self.theta_fc * self.Zr:                       # free drainage above FC
            drain = self.drain_k_per_h * (S_new - self.theta_fc * self.Zr)
            S_new -= drain
        if S_new > self.theta_sat * self.Zr:                      # saturation overflow
            drain += S_new - self.theta_sat * self.Zr
            S_new = self.theta_sat * self.Zr
        S_new = max(S_new, self.theta_wp * self.Zr * 0.5)         # air-dry floor
        return {"theta": S_new / self.Zr, "etc_mm": etc, "drainage_mm": drain,
                "runoff_mm": runoff, "infiltration_mm": infil, "ks": Ks}

    def simulate(self, theta0: float, rainfall_mm: np.ndarray, irrigation_mm: np.ndarray,
                 et0_mm: np.ndarray, kc: np.ndarray, dt_h: float = 1.0) -> pd.DataFrame:
        """Roll the water balance forward over a shared series of conditions."""
        n = len(rainfall_mm)
        out = np.empty((n, 6))
        theta = float(theta0)
        for i in range(n):
            r = self.step(theta, rain_mm=float(rainfall_mm[i]), irr_mm=float(irrigation_mm[i]),
                          et0_mm=float(et0_mm[i]), kc=float(kc[i]), dt_h=dt_h)
            out[i] = [theta, r["etc_mm"], r["drainage_mm"], r["runoff_mm"], r["infiltration_mm"], r["ks"]]
            theta = r["theta"]
        return pd.DataFrame(out, columns=["theta", "etc_mm", "drainage_mm",
                                          "runoff_mm", "infiltration_mm", "ks"])