"""FastAPI service: delivery-delay risk scoring + fast-lane drift check.

Serves the healed champion (promoted by the self-healing loop) with the
deployed v2 percentile policy. The batch endpoint carries its own drift
tripwire (feature PSI vs reference snapshot) - the service monitors itself.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import CAT, NUM, load_champion, predict_df
from src.monitoring import feature_drift_report

ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(title="ml-reliability",
              description="Delivery-delay risk scoring with self-monitoring",
              version="0.1.0")

CONFIG = json.load(open(ROOT / "evaluation" / "calibration_config.json"))
Q_RATE = CONFIG["policy_v2_percentile"]["q"]

# ---- load artifacts at startup ----
pre, model = load_champion(ROOT / "models" / "champion.pt")
iso = joblib.load(ROOT / "models" / "calibrator.joblib")
_ref = pd.read_parquet(ROOT / "data" / "processed" / "reference_snapshot.parquet")

MONITOR_FEATURES = ["est_window_days", "dist_mean_km", "dist_max_km", "total_price",
                    "total_freight", "freight_ratio", "mean_weight_g", "mean_volume_l",
                    "seller_prior_rate", "seller_prior_orders", "n_items", "n_sellers"]


class OrderIn(BaseModel):
    est_window_days: float = Field(..., gt=0)
    dist_mean_km: float = Field(0.0, ge=0)
    dist_max_km: float = Field(0.0, ge=0)
    total_price: float = Field(..., gt=0)
    total_freight: float = Field(0.0, ge=0)
    max_price: float = Field(0.0, ge=0)
    mean_weight_g: float = Field(0.0, ge=0)
    mean_volume_l: float = Field(0.0, ge=0)
    seller_prior_orders: int = Field(0, ge=0)
    seller_prior_rate: float = Field(0.0, ge=0, le=1)
    n_items: int = Field(1, ge=1)
    n_sellers: int = Field(1, ge=1)
    freight_ratio: float = Field(0.0, ge=0)
    payment_installments: int = Field(1, ge=1)
    purchase_month: int = Field(1, ge=1, le=12)
    purchase_dow: int = Field(0, ge=0, le=6)
    purchase_hour: int = Field(0, ge=0, le=23)
    top_category: str = "other"
    payment_type: str = "missing"
    customer_state: str = "missing"


def to_frame(order: OrderIn) -> pd.DataFrame:
    return pd.DataFrame([order.model_dump()])


def calibrated_risk(frame: pd.DataFrame) -> np.ndarray:
    return iso.predict(predict_df(frame, pre, model))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "pytorch_mlp (healed champion, promoted 2018-08)",
        "policy": {"rule": "top-q percentile", "q": Q_RATE},
        "monitor_features": len(MONITOR_FEATURES),
    }


@app.post("/v1/score")
def score(order: OrderIn):
    t0 = time.perf_counter()
    risk = float(calibrated_risk(to_frame(order))[0])
    return {"risk": round(risk, 4),
            "note": "percentile flagging is batch-relative - use /v1/score/batch",
            "latency_ms": int((time.perf_counter() - t0) * 1000)}


@app.post("/v1/score/batch")
def score_batch(orders: list[OrderIn]):
    if len(orders) < 50:
        raise HTTPException(400, f"batch endpoint needs >=50 orders for stable percentiles, got {len(orders)}")
    t0 = time.perf_counter()
    frame = pd.DataFrame([o.model_dump() for o in orders])
    p = calibrated_risk(frame)

    # deployed policy: flag top q% of the batch
    n_flag = max(1, int(round(Q_RATE * len(p))))
    flags = np.zeros(len(p), dtype=bool)
    flags[np.argsort(-p)[:n_flag]] = True

    # fast-lane drift tripwire
    drift = feature_drift_report(_ref, frame, MONITOR_FEATURES)

    return {
        "n": len(frame),
        "mean_risk": round(float(p.mean()), 4),
        "flagged": int(flags.sum()),
        "q_rate": Q_RATE,
        "flagged_indices": np.where(flags)[0].tolist()[:20],
        "drift": drift,
        "drift_summary": {
            "alarms": [d["feature"] for d in drift if d["status"] == "alarm"],
            "warns": [d["feature"] for d in drift if d["status"] == "warn"],
        },
        "latency_ms": int((time.perf_counter() - t0) * 1000),
    }