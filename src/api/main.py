"""FastAPI backend for the smart-irrigation system.

NOT a production deployment surface: all data are SIMULATED, models are loaded
lazily from models/ (trained on demand if missing), and the system only ever
recommends actions — it never drives hardware.
"""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config
from src.counterfactual.simulator import (CandidateAction, candidates_from_config)
from src.service import Runtime, get_runtime, decision_indexes


# ----------------------------------------------------------------- Schema -- #
class IdIndex(BaseModel):
    farm_id: str
    index: int
    horizon: int = Field(default=6, ge=1, le=24)


class RecommendIn(BaseModel):
    farm_id: str
    index: int
    faulted: bool = Field(default=False, description="Use the fault-injected farm data")
    theta0_mm: Optional[float] = None


class IntervalOut(BaseModel):
    mean: float
    lo: float
    hi: float
    std: float
    uncertainty_score: float
    target_time: str
    horizon: int
    farm_id: str


def create_app(run_id: str = "default") -> FastAPI:
    app = FastAPI(
        title="AI Smart-Irrigation Decision Support (SIMULATED)",
        version="0.1.0",
        description="Forecasting, sensor-fault detection, uncertainty and counterfactual "
                    "irrigation recommendations on a virtual farm. All data are simulated.",
    )
    runtime: Runtime | None = None

    def rt() -> Runtime:
        nonlocal runtime
        if runtime is None:
            runtime = get_runtime(config.FARM_CLEAN, run_id=run_id)
        return runtime

    def fdf(farm_id: str) -> "object":
        ids = rt().farm_ids()
        if farm_id not in ids:
            raise HTTPException(404, f"unknown farm {farm_id}; available: {ids}")
        return rt().farm_df(farm_id)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "data_path": config.FARM_CLEAN,
                "dataset_rows": len(rt().df), "n_farms": len(rt().farm_ids())}

    @app.get("/farms")
    def farms() -> dict:
        return {"farms": rt().farm_ids()}

    @app.get("/farm/{farm_id}/head")
    def farm_head(farm_id: str, rows: int = 100) -> dict:
        g = fdf(farm_id)
        cols = ["timestamp", "split", "soil_moisture_measured", "soil_moisture_true",
                "temperature_c", "humidity_pct", "rainfall_mm", "et0_mm", "target_low",
                "target_high", "water_stress_ks", "sensor_fault_label"]
        available = [c for c in cols if c in g.columns]
        return {"farm_id": farm_id, "columns": available,
                "rows": g.head(rows)[available].to_dict(orient="records")}

    @app.get("/farm/{farm_id}/decision-indexes")
    def decision_indexes_endpoint(farm_id: str, split: str = "test") -> dict:
        return {"farm_id": farm_id, "split": split,
                "indexes": decision_indexes(fdf(farm_id), split)}

    @app.post("/forecast", response_model=IntervalOut)
    def forecast(q: IdIndex) -> IntervalOut:
        g = fdf(q.farm_id)
        if not 0 <= q.index < len(g):
            raise HTTPException(422, f"index {q.index} out of range [0,{len(g)})")
        return IntervalOut(**rt().forecast(q.farm_id, q.index, q.horizon))

    @app.post("/recommend")
    def recommend(q: RecommendIn) -> dict:
        g = fdf(q.farm_id)
        if not 0 <= q.index < len(g):
            raise HTTPException(422, f"index {q.index} out of range [0,{len(g)})")
        candidates: list[CandidateAction] = candidates_from_config(
            rt().decision_cfg)
        fault_df = None
        if q.faulted:
            if not config.FARM_FAULTY.exists():
                raise HTTPException(404, "faulty dataset not present")
            import pandas as pd
            fault_df = pd.read_csv(config.FARM_FAULTY)
        return rt().recommend(q.farm_id, q.index, candidates=candidates,
                              theta0=q.theta0_mm, fault_df=fault_df)

    @app.get("/reports")
    def list_reports() -> dict:
        files = sorted(config.REPORTS_DIR.glob("*.json"))
        return {"reports": [f.name for f in files]}

    return app


app = create_app()