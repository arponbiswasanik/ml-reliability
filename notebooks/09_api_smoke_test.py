"""09 - API smoke test: real-data batches, normal vs shifted.

v2: hand-invented single-point batches were statistically degenerate (delta
functions alarm PSI by construction). Real stream orders instead:
  normal  = 100 real August orders (the healed champion's regime)
  shifted = same orders, distances x2, seller_prior_rate inflated
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import CAT, NUM

BASE = "http://127.0.0.1:8000"

def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))

def get(path):
    return json.load(urllib.request.urlopen(BASE + path))

print(json.dumps(get("/health"), indent=2))

feat = pd.read_parquet("data/processed/features.parquet")
stream = feat[feat.split == "stream"]
batch_df = stream.sample(100, random_state=42)[CAT + NUM]
batch = batch_df.where(pd.notna(batch_df), None).to_dict("records")

print("\n/v1/score/batch (100 REAL August orders - expect few/no alarms):")
r1 = post("/v1/score/batch", batch)
print(json.dumps(r1["drift_summary"], indent=2))
print(f"mean_risk: {r1['mean_risk']} | flagged: {r1['flagged']}/{r1['n']}")

shifted = []
for row in batch:
    s = dict(row)
    if s.get("dist_mean_km") is not None:
        s["dist_mean_km"] = float(s["dist_mean_km"]) * 2
        s["dist_max_km"] = float(s["dist_max_km"]) * 2
    if s.get("seller_prior_rate") is not None:
        s["seller_prior_rate"] = min(1.0, float(s["seller_prior_rate"]) * 2 + 0.15)
    shifted.append(s)

print("\n/v1/score/batch (same orders SHIFTED - expect distance/seller alarms):")
r2 = post("/v1/score/batch", shifted)
print(json.dumps(r2["drift_summary"], indent=2))
print(f"mean_risk: {r2['mean_risk']} | flagged: {r2['flagged']}/{r2['n']}")

print(f"\nmean risk: normal {r1['mean_risk']} -> shifted {r2['mean_risk']} "
      f"({r2['mean_risk']/max(r1['mean_risk'],1e-9):.2f}x)")