"""03 - Feature engineering: 8-table joins, point-in-time seller history.

All features knowable AT PURCHASE (leakage map, step 02). Excluded by design:
order_approved_at, carrier/customer delivery dates, review data.
Customer history skipped: near-unique customer IDs -> no signal.
Seller history approximation: expanding over purchase time; labels of very
recent past orders may be in-flight at current purchase (documented).
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW, OUT = Path("data/raw"), Path("data/processed")

# ---------- load ----------
splits = {}
for name in ["train", "test", "stream"]:
    splits[name] = pd.read_parquet(OUT / f"{name}_labels.parquet")
    splits[name]["split"] = name
orders = pd.concat(splits.values(), ignore_index=True)
n_expected = len(orders)

customers = pd.read_csv(RAW / "olist_customers_dataset.csv")
items = pd.read_csv(RAW / "olist_order_items_dataset.csv")
products = pd.read_csv(RAW / "olist_products_dataset.csv")
sellers = pd.read_csv(RAW / "olist_sellers_dataset.csv")
geo = pd.read_csv(RAW / "olist_geolocation_dataset.csv")
payments = pd.read_csv(RAW / "olist_order_payments_dataset.csv")

# ---------- geolocation: outlier guard + zip aggregation ----------
BRAZIL = {"lat": (-33.75, 5.30), "lng": (-73.99, -34.79)}
n0 = len(geo)
geo = geo[geo.geolocation_lat.between(*BRAZIL["lat"]) &
          geo.geolocation_lng.between(*BRAZIL["lng"])]
print(f"geolocation: {n0 - len(geo):,} rows outside Brazil bounding box -> dropped")
zip_geo = (geo.groupby("geolocation_zip_code_prefix")
              .agg(lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median")))

cust = (customers.merge(zip_geo, left_on="customer_zip_code_prefix", right_index=True, how="left")
        [["customer_id", "customer_state", "lat", "lng"]]
        .rename(columns={"lat": "cust_lat", "lng": "cust_lng"}))
sell = (sellers.merge(zip_geo, left_on="seller_zip_code_prefix", right_index=True, how="left")
        [["seller_id", "seller_state", "lat", "lng"]]
        .rename(columns={"lat": "seller_lat", "lng": "seller_lng"}))

# ---------- order-level item aggregation ----------
order_items = items.groupby("order_id").agg(
    n_items=("product_id", "size"), n_sellers=("seller_id", "nunique"),
    total_price=("price", "sum"), total_freight=("freight_value", "sum"),
    max_price=("price", "max"))
order_items["freight_ratio"] = order_items["total_freight"] / order_items["total_price"].clip(lower=1)

# ---------- seller<->customer distance ----------
def haversine(lat1, lng1, lat2, lng2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = np.radians(lat2 - lat1), np.radians(lng2 - lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

it = items[["order_id", "seller_id"]].merge(sell, on="seller_id", how="left")
oc = orders[["order_id", "customer_id"]].merge(cust, on="customer_id", how="left")
it = it.merge(oc, on="order_id", how="left")
it["dist_km"] = haversine(it.seller_lat, it.seller_lng, it.cust_lat, it.cust_lng)
dist = it.groupby("order_id").agg(dist_mean_km=("dist_km", "mean"),
                                  dist_max_km=("dist_km", "max"))

# ---------- products: dims + dominant category ----------
products["volume_l"] = (products.product_length_cm * products.product_height_cm
                        * products.product_width_cm / 1000)
prod = products[["product_id", "product_category_name", "product_weight_g", "volume_l"]]
it2 = (items[["order_id", "product_id", "price"]]
       .merge(prod, on="product_id", how="left").sort_values("price", ascending=False))
cat = it2.groupby("order_id").agg(mean_weight_g=("product_weight_g", "mean"),
                                  mean_volume_l=("volume_l", "mean"))
top_cats = it2["product_category_name"].value_counts().head(30).index
it2["category"] = it2["product_category_name"].where(
    it2["product_category_name"].isin(top_cats), "other")
dominant = it2.groupby("order_id")["category"].first().rename("top_category")

# ---------- payments ----------
pay = payments.groupby("order_id").agg(
    n_payments=("payment_sequential", "size"),
    payment_installments=("payment_installments", "max"),
    payment_value=("payment_value", "sum"))
pay_type = (payments.sort_values("payment_sequential")
            .groupby("order_id")["payment_type"].first().rename("payment_type"))

# ---------- point-in-time seller history ----------
sq = (it[["order_id", "seller_id"]].drop_duplicates()
      .merge(orders[["order_id", "order_purchase_timestamp", "is_late"]], on="order_id")
      .sort_values("order_purchase_timestamp"))
g = sq.groupby("seller_id")["is_late"]
sq["seller_prior_orders"] = g.expanding().count().reset_index(level=0, drop=True) - 1
sq["seller_prior_late"] = g.expanding().sum().reset_index(level=0, drop=True) - sq["is_late"]
sq["seller_prior_rate"] = np.where(sq["seller_prior_orders"] > 0,
                                   sq["seller_prior_late"] / sq["seller_prior_orders"], np.nan)
sh = sq.groupby("order_id").agg(seller_prior_orders=("seller_prior_orders", "max"),
                                seller_prior_rate=("seller_prior_rate", "max"))

# ---------- temporal features ----------
ts = orders["order_purchase_timestamp"]
orders["purchase_month"], orders["purchase_dow"], orders["purchase_hour"] = ts.dt.month, ts.dt.dayofweek, ts.dt.hour

# ---------- assemble ----------
feat = (orders[["order_id", "customer_id", "split", "is_late", "est_window_days",
                "purchase_month", "purchase_dow", "purchase_hour"]]
        .merge(order_items, on="order_id", how="left")
        .merge(dist, on="order_id", how="left")
        .merge(cat, on="order_id", how="left")
        .merge(dominant, on="order_id", how="left")
        .merge(pay, on="order_id", how="left")
        .merge(pay_type, on="order_id", how="left")
        .merge(sh, on="order_id", how="left")
        .merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left"))

assert len(feat) == n_expected, f"row loss: {n_expected} -> {len(feat)}"

# ---------- fills: train-only statistics (no future info) ----------
tr = feat["split"] == "train"
train_prior = feat.loc[tr, "is_late"].mean()
num_cols = ["mean_weight_g", "mean_volume_l", "dist_mean_km", "dist_max_km",
            "seller_prior_rate"]
fills = {c: feat.loc[tr, c].median() for c in num_cols}
fills["seller_prior_rate"] = train_prior          # new sellers -> global prior
nulls = feat[num_cols].isna().sum()
print("\nNulls before fill:\n", nulls[nulls > 0].to_string() if nulls.any() else "  none")
feat = feat.fillna({**fills, "top_category": "missing", "payment_type": "missing",
                    "customer_state": "missing"})

feat = feat.drop(columns=["customer_id"])
feat.to_parquet(OUT / "features.parquet", index=False)

print(f"\nfeatures.parquet: {feat.shape[0]:,} x {feat.shape[1]}")
print(feat["split"].value_counts().to_string())

# ---------- signal preview (learnability probe, ticket-triage style) ----------
for col in ["dist_mean_km", "est_window_days", "seller_prior_rate"]:
    b = pd.qcut(feat[col], 5, duplicates="drop")
    print(f"\nlate rate by {col} quintile:")
    print(feat.groupby(b, observed=True)["is_late"].agg(["size", "mean"])
          .to_string(float_format=lambda x: f"{x:.3f}"))