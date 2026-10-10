"""02 - Label construction + temporal splits.

Label: is_late = (delivered > estimated). Predict AT PURCHASE whether the
delivery promise breaks. See leakage map in step description.
Censoring guard: final-14-day purchases dropped (snapshot may cut outcomes).
"""
from pathlib import Path

import pandas as pd

RAW, OUT = Path("data/raw"), Path("data/processed")

orders = pd.read_csv(RAW / "olist_orders_dataset.csv", parse_dates=[
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date"])

n0 = len(orders)
df = orders[orders["order_delivered_customer_date"].notna()].copy()
print(f"Delivered only:              {len(df):,} / {n0:,}")

bad = df["order_delivered_customer_date"] < df["order_purchase_timestamp"]
print(f"Invalid (delivered < purchase): {bad.sum()} -> dropped")
df = df[~bad]

df["is_late"] = (df["order_delivered_customer_date"]
                 > df["order_estimated_delivery_date"]).astype(int)
df["est_window_days"] = ((df["order_estimated_delivery_date"]
                          - df["order_purchase_timestamp"]).dt.total_seconds() / 86400).round(1)

pre2017 = df["order_purchase_timestamp"] < "2017-01-01"
print(f"2016 fragment (gap period):  {pre2017.sum():,} -> dropped "
      f"(ingest gap + ~0.3% of volume; documented)")
df = df[~pre2017]

cutoff = df["order_purchase_timestamp"].max() - pd.Timedelta(days=14)
censored = df["order_purchase_timestamp"] >= cutoff
print(f"Final-14-day purchases:      {censored.sum():,} -> dropped (censoring guard)")
df = df[~censored].sort_values("order_purchase_timestamp").reset_index(drop=True)

print(f"\nLabeled dataset: {len(df):,} orders | late rate {df['is_late'].mean():.1%}")

df["month"] = df["order_purchase_timestamp"].dt.to_period("M")
monthly = df.groupby("month").agg(n=("is_late", "size"), late_rate=("is_late", "mean"))
print("\nMonthly volume + late rate (the natural-drift preview):")
print(monthly.to_string(float_format=lambda x: f"{x:.3f}"))

n = len(df)
n_train, n_test = int(n * 0.80), int(n * 0.10)
train = df.iloc[:n_train]
test = df.iloc[n_train:n_train + n_test]
stream = df.iloc[n_train + n_test:]
for name, part in [("train", train), ("test", test), ("stream", stream)]:
    print(f"{name:>6}: {len(part):,} | {part['order_purchase_timestamp'].min().date()} -> "
          f"{part['order_purchase_timestamp'].max().date()} | late {part['is_late'].mean():.1%}")

keep = ["order_id", "customer_id", "order_purchase_timestamp",
        "order_estimated_delivery_date", "est_window_days", "is_late"]
train[keep].to_parquet(OUT / "train_labels.parquet", index=False)
test[keep].to_parquet(OUT / "test_labels.parquet", index=False)
stream[keep].to_parquet(OUT / "stream_labels.parquet", index=False)
print(f"\nSaved: train/test/stream_labels.parquet")