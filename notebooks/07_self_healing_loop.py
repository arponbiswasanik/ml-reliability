"""07 - Self-healing loop: drift-triggered retraining with promotion gate.

Monthly walk over the stream period. Each month:
  1. score incoming orders with current champion (rolling reference = last
     60 days of labeled data - fixes the frozen-calibrator sin of stage A)
  2. fast+slow drift check; slow-lane alarm + >=500 fresh labels -> retrain
  3. challenger (trailing 90d) vs champion on recent labeled window
  4. promotion gate: challenger must beat champion (PR-AUC) -> MLflow
     registry version bump + hot-swap. Rejections are logged, not retried.

No human in the loop. MLflow SQLite backend; registry = audit trail.
API notes (current MLflow): filesystem backend deprecated -> sqlite;
log_model uses name= (not artifact_path=) and requires input_example for
the default pt2 traced-graph serialization.
"""
import json
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import CAT, NUM, train_mlp, predict_df
from src.monitoring import classify, feature_drift_report, psi

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

LABEL_GAP_ALARM = 0.03
PSI_ALARM = 0.20
MONITOR_FEATURES = ["est_window_days", "dist_mean_km", "dist_max_km", "total_price",
                    "total_freight", "freight_ratio", "mean_weight_g", "mean_volume_l",
                    "seller_prior_rate", "seller_prior_orders", "n_items", "n_sellers"]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ml-reliability")

feat = pd.read_parquet("data/processed/features.parquet")
labels = pd.concat([pd.read_parquet(f"data/processed/{s}_labels.parquet")
                    for s in ["train", "test", "stream"]])
feat = feat.merge(labels[["order_id", "order_purchase_timestamp"]], on="order_id")
feat = feat.sort_values("order_purchase_timestamp").reset_index(drop=True)

stream = feat[feat.split == "stream"].copy()
stream["month"] = stream.order_purchase_timestamp.dt.to_period("M")

# initial champion: the step-05 artifact (trained on fit slice, calib Feb-May)
pre, model = None, None
try:
    from src.model import load_champion
    pre, model = load_champion()
except Exception as e:
    print(f"warm-start unavailable ({type(e).__name__}) - training fresh champion")
if pre is None:
    pre, model = train_mlp(feat[feat.split == "train"].iloc[:-int(len(feat[feat.split == 'train']) * 0.15)])

def calibrate(ref_df, pre, model):
    """Isotonic fit on labeled reference; returns the calibrator."""
    p_raw = predict_df(ref_df, pre, model)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(p_raw, ref_df["is_late"].values)
    return iso

# initial reference: trailing labeled history before the stream begins
hist = feat[feat.order_purchase_timestamp < stream.order_purchase_timestamp.min()]
reference = hist.iloc[-4000:].copy()
iso = calibrate(reference, pre, model)

log = []
months = sorted(stream["month"].unique())
print(f"stream months: {[str(m) for m in months]} | initial champion from step 05\n")

for m in months:
    w = stream[stream.month == m].copy()
    incoming = w.iloc[: len(w) // 2]      # first half: what the model scores live
    fresh = w.iloc[len(w) // 2:]          # second half: labels arriving during the month

    # ---------- 1. live scoring + drift check ----------
    p_raw = predict_df(incoming, pre, model)
    p_cal = iso.predict(p_raw)
    realized = incoming["is_late"].mean()
    predicted = p_cal.mean()
    gap = realized - predicted
    drift = feature_drift_report(reference, incoming, MONITOR_FEATURES)
    alarms = [d["feature"] for d in drift if d["status"] == "alarm"]
    slow_alarm = abs(gap) >= LABEL_GAP_ALARM
    entry = {"month": str(m), "n": len(incoming), "realized": round(float(realized), 4),
             "predicted": round(float(predicted), 4), "gap_pp": round(float(gap) * 100, 1),
             "slow_alarm": bool(slow_alarm), "feature_alarms": alarms,
             "model_version": None, "action": "none", "promoted": False}

    # fresh labeled data available for retraining (labels arrive with lag)
    fresh_labeled = pd.concat([reference, fresh]).iloc[-6000:]
    n_fresh = len(fresh)

    # ---------- 2. retrain trigger ----------
    if slow_alarm and n_fresh >= 500:
        challenger_pre, challenger = train_mlp(fresh_labeled, seed=42)

        # ---------- 3. promotion gate: recent labeled window ----------
        val = pd.concat([reference, fresh]).iloc[-2500:]
        p_champ = iso.predict(predict_df(val, pre, model))
        ch_iso = calibrate(fresh_labeled.iloc[-1500:], challenger_pre, challenger)
        p_chal = ch_iso.predict(predict_df(val, challenger_pre, challenger))
        champ_pr = average_precision_score(val["is_late"], p_champ)
        chal_pr = average_precision_score(val["is_late"], p_chal)

        entry.update({"champ_pr": round(float(champ_pr), 4),
                      "chal_pr": round(float(chal_pr), 4)})

        # ---------- 4. promotion + registry ----------
        if chal_pr > champ_pr:
            with mlflow.start_run(run_name=f"retrain-{m}"):
                mlflow.log_param("trigger", "slow-lane drift alarm")
                mlflow.log_param("train_rows", len(fresh_labeled))
                mlflow.log_metric("champ_pr_auc", champ_pr)
                mlflow.log_metric("challenger_pr_auc", chal_pr)
                mlflow.log_metric("gap_pp", entry["gap_pp"])
                X_ex = challenger_pre.transform(fresh_labeled[CAT + NUM]).toarray().astype(np.float32)[:2]
                mlflow.pytorch.log_model(challenger, name="model",
                                         registered_model_name="delivery_risk_mlp",
                                         input_example=torch.from_numpy(X_ex),
                                         serialization_format="pickle")
            pre, model = challenger_pre, challenger
            iso = calibrate(fresh_labeled.iloc[-1500:], pre, model)
            entry.update({"action": "retrain+promote", "promoted": True})
            print(f"{m}: DRIFT ALARM (gap {entry['gap_pp']:+.1f}pp) -> retrained "
                  f"(PR {champ_pr:.3f} -> {chal_pr:.3f}) -> PROMOTED, hot-swapped")
        else:
            entry["action"] = "retrain_rejected"
            print(f"{m}: DRIFT ALARM (gap {entry['gap_pp']:+.1f}pp) -> challenger "
                  f"PR {chal_pr:.3f} <= champion {champ_pr:.3f} -> REJECTED, champion stands")
    else:
        print(f"{m}: realized {realized:5.1%} vs predicted {predicted:5.1%} "
              f"(gap {gap:+.1%}) | {'slow-lane ALARM but insufficient fresh labels' if slow_alarm else 'quiet'}")

    # rolling reference absorbs the month's labeled data
    reference = pd.concat([reference, incoming, fresh]).iloc[-8000:]
    iso = calibrate(reference, pre, model)
    log.append(entry)

promotions = [e for e in log if e["promoted"]]
print(f"\n=== summary: {len(promotions)} promotions | "
      f"{sum(1 for e in log if e['action'] == 'retrain_rejected')} rejections ===")

with open("evaluation/self_healing_log.json", "w") as f:
    json.dump({"label_gap_alarm_pp": LABEL_GAP_ALARM * 100,
               "months": log,
               "summary": {"promotions": len(promotions),
                           "rejections": sum(1 for e in log if e["action"] == "retrain_rejected")}},
              f, indent=2)
print("saved: evaluation/self_healing_log.json")