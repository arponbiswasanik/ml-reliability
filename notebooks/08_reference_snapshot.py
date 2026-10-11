"""08 - Persist the drift reference snapshot for the API's fast lane."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MONITOR_FEATURES = ["est_window_days", "dist_mean_km", "dist_max_km", "total_price",
                    "total_freight", "freight_ratio", "mean_weight_g", "mean_volume_l",
                    "seller_prior_rate", "seller_prior_orders", "n_items", "n_sellers"]

feat = pd.read_parquet("data/processed/features.parquet")
labels = pd.concat([pd.read_parquet(f"data/processed/{s}_labels.parquet")
                    for s in ["train", "test", "stream"]])
feat = feat.merge(labels[["order_id", "order_purchase_timestamp"]], on="order_id")

# same definition as the loop's final rolling reference
hist = feat.sort_values("order_purchase_timestamp").reset_index(drop=True)
ref = hist.iloc[-8000:][["order_id"] + MONITOR_FEATURES].copy()

ref.to_parquet("data/processed/reference_snapshot.parquet", index=False)
print(f"reference snapshot: {len(ref):,} rows -> data/processed/reference_snapshot.parquet")