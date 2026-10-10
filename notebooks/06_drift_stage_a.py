"""06 - Stage A: natural drift detection over test+stream windows.

Two-tier monitoring:
  fast lane (label-free):    feature PSI + calibrated-score PSI vs reference
  slow lane (label-based):   realized vs predicted late rate (arrives with
                             delivery outcomes - label latency is real)
Reference = calibration slice (the last world the deployed model knew).
Calendar features excluded from PSI: cyclical by construction -> alarm fatigue.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import load_champion, predict_df
from src.monitoring import PSI_WARN, PSI_ALARM, classify, feature_drift_report, psi
import joblib

MONITOR_FEATURES = ["est_window_days", "dist_mean_km", "dist_max_km", "total_price",
                    "total_freight", "freight_ratio", "mean_weight_g", "mean_volume_l",
                    "seller_prior_rate", "seller_prior_orders", "n_items", "n_sellers"]
LABEL_GAP_ALARM = 0.03   # |realized - predicted| >= 3pp -> calibration drift alarm

# ---------- data + timestamps ----------
feat = pd.read_parquet("data/processed/features.parquet")
labels = pd.concat([pd.read_parquet(f"data/processed/{s}_labels.parquet")
                    for s in ["train", "test", "stream"]])
feat = feat.merge(labels[["order_id", "order_purchase_timestamp"]], on="order_id")

train = feat[feat.split == "train"].reset_index(drop=True)
n_cal = int(len(train) * 0.15)
reference = train.iloc[-n_cal:]                       # same slice as step 05
walk = feat[feat.split.isin(["test", "stream"])].sort_values("order_purchase_timestamp")
print(f"reference: {len(reference):,} rows | walk: {len(walk):,} rows "
      f"({walk.order_purchase_timestamp.min().date()} -> {walk.order_purchase_timestamp.max().date()})")

# ---------- champion + calibrator ----------
pre, model = load_champion()
iso = joblib.load("models/calibrator.joblib")

p_ref_cal = iso.predict(predict_df(reference, pre, model))
print(f"reference calibrated risk: mean {p_ref_cal.mean():.3f} | "
      f"realized {reference['is_late'].mean():.3f}")

# ---------- window walk (14-day) ----------
t0, t1 = walk["order_purchase_timestamp"].min(), walk["order_purchase_timestamp"].max()
edges = pd.date_range(t0, t1 + pd.Timedelta(days=14), freq="14D")

windows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    w = walk[(walk.order_purchase_timestamp >= lo) & (walk.order_purchase_timestamp < hi)]
    if len(w) < 300:
        continue
    p_cal = iso.predict(predict_df(w, pre, model))
    realized, predicted = float(w["is_late"].mean()), float(p_cal.mean())
    gap = realized - predicted
    drift = feature_drift_report(reference, w, MONITOR_FEATURES)
    s_psi = psi(p_ref_cal, p_cal)
    alarms = [d["feature"] for d in drift if d["status"] == "alarm"]
    warns = [d["feature"] for d in drift if d["status"] == "warn"]
    label_alarm = abs(gap) >= LABEL_GAP_ALARM
    windows.append({
        "window": f"{lo.date()}->{hi.date()}", "n": len(w),
        "realized_late": round(realized, 4), "predicted_late": round(predicted, 4),
        "gap_pp": round(gap * 100, 1), "label_alarm": bool(label_alarm),
        "score_psi": round(s_psi, 4), "score_status": classify(s_psi),
        "feature_alarms": alarms, "feature_warns": warns,
        "top_feature_psi": drift[0],
    })
    print(f"{lo.date()}->{hi.date()}  n={len(w):>5,} | realized {realized:5.1%} "
          f"vs predicted {predicted:5.1%} (gap {gap:+.1%}{' ALARM' if label_alarm else ''}) | "
          f"score PSI {s_psi:.3f} [{classify(s_psi)}] | "
          f"feat alarm {alarms or '-'} | warn {warns[:3] or '-'}")

# ---------- plot ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
xs = range(len(windows))
ax1.plot(xs, [w["realized_late"] for w in windows], "o-", label="realized late rate")
ax1.plot(xs, [w["predicted_late"] for w in windows], "s--", label="predicted (calibrated)")
ax1.axhline(LABEL_GAP_ALARM, color="gray", ls=":", alpha=0.5)
ax1.set_ylabel("late rate")
ax1.set_title("Stage A: natural drift over 14-day windows (test + stream)")
ax1.legend()
ax1.grid(alpha=0.3)
ax2.plot(xs, [w["score_psi"] for w in windows], "o-", label="score PSI")
ax2.plot(xs, [w["top_feature_psi"]["psi"] for w in windows], "s--",
         label=f"max feature PSI ({windows[0]['top_feature_psi']['feature'] if windows else ''}..)")
ax2.axhline(PSI_WARN, color="orange", ls=":", label=f"warn {PSI_WARN}")
ax2.axhline(PSI_ALARM, color="red", ls=":", label=f"alarm {PSI_ALARM}")
ax2.set_ylabel("PSI vs reference")
ax2.set_xticks(list(xs))
ax2.set_xticklabels([w["window"].split("->")[0] for w in windows], rotation=45, ha="right")
ax2.legend()
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("evaluation/stage_a_drift.png", dpi=150, bbox_inches="tight")

with open("evaluation/stage_a_drift.json", "w") as f:
    json.dump({"reference": "calibration slice (train tail 15%)",
               "label_gap_alarm_pp": LABEL_GAP_ALARM * 100,
               "windows": windows}, f, indent=2)
print("\nsaved: evaluation/stage_a_drift.json, evaluation/stage_a_drift.png")