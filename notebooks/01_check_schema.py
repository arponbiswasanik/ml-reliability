"""01 - Schema verification for Olist: fail loud, then report."""
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")

EXPECTED = {
    "olist_orders_dataset.csv": ["order_id", "customer_id", "order_status",
                                 "order_purchase_timestamp",
                                 "order_delivered_customer_date",
                                 "order_estimated_delivery_date"],
    "olist_customers_dataset.csv": ["customer_id", "customer_zip_code_prefix",
                                    "customer_state"],
    "olist_order_items_dataset.csv": ["order_id", "product_id", "seller_id",
                                       "price", "freight_value"],
    "olist_products_dataset.csv": ["product_id", "product_category_name"],
    "olist_sellers_dataset.csv": ["seller_id", "seller_zip_code_prefix",
                                  "seller_state"],
    "olist_geolocation_dataset.csv": ["geolocation_zip_code_prefix",
                                       "geolocation_lat", "geolocation_lng"],
    "olist_order_payments_dataset.csv": ["order_id", "payment_type"],
    "olist_order_reviews_dataset.csv": ["order_id"],
}

# --- fail loud: files ---
missing_files = [f for f in EXPECTED if not (RAW / f).exists()]
if missing_files:
    raise SystemExit(f"MISSING FILES: {missing_files}\n"
                     f"Present: {sorted(p.name for p in RAW.glob('*.csv'))}")

# --- fail loud: columns ---
frames = {}
for fname, need in EXPECTED.items():
    df = pd.read_csv(RAW / fname)
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"{fname}: MISSING COLUMNS {miss}")
    frames[fname] = df
    print(f"{fname:42} {df.shape[0]:>9,} rows x {df.shape[1]:>2} cols")

# --- orders: the table our label comes from ---
orders = frames["olist_orders_dataset.csv"]
orders["ts"] = pd.to_datetime(orders["order_purchase_timestamp"])
print(f"\nPurchase window: {orders['ts'].min()} -> {orders['ts'].max()}")
print(f"\nOrder status:\n{orders['order_status'].value_counts().to_string()}")

delivered = orders["order_delivered_customer_date"].notna()
print(f"\nDelivered orders: {delivered.sum():,} / {len(orders):,} ({delivered.mean():.1%})")
print(f"Unique order_id: {orders['order_id'].nunique():,} / {len(orders):,}")

# --- join integrity ---
items = frames["olist_order_items_dataset.csv"]
customers = frames["olist_customers_dataset.csv"]
print(f"\nOrders with >=1 item row:      {orders['order_id'].isin(set(items['order_id'])).mean():.1%}")
print(f"Orders resolving to a customer:{orders['customer_id'].isin(set(customers['customer_id'])).mean():.1%}")

n_items_per_order = items.groupby("order_id").size()
print(f"Items per order: median {n_items_per_order.median():.0f}, "
      f"p95 {n_items_per_order.quantile(0.95):.0f}, max {n_items_per_order.max()}")