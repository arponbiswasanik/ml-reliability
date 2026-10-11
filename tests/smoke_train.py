"""CI smoke test: the training path runs end-to-end on synthetic data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import CAT, NUM, train_mlp, predict_df

rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({
    "est_window_days": rng.uniform(5, 60, n),
    "dist_mean_km": rng.uniform(0, 3000, n),
    "dist_max_km": rng.uniform(0, 3500, n),
    "total_price": rng.uniform(20, 500, n),
    "total_freight": rng.uniform(0, 80, n),
    "max_price": rng.uniform(10, 400, n),
    "mean_weight_g": rng.uniform(100, 8000, n),
    "mean_volume_l": rng.uniform(0.5, 40, n),
    "seller_prior_orders": rng.integers(0, 200, n),
    "seller_prior_rate": rng.uniform(0, 0.4, n),
    "n_items": rng.integers(1, 5, n),
    "n_sellers": rng.integers(1, 3, n),
    "freight_ratio": rng.uniform(0, 0.5, n),
    "payment_installments": rng.integers(1, 10, n),
    "purchase_month": rng.integers(1, 13, n),
    "purchase_dow": rng.integers(0, 7, n),
    "purchase_hour": rng.integers(0, 24, n),
    "top_category": rng.choice(["other", "furniture", "electronics"], n),
    "payment_type": rng.choice(["credit_card", "boleto"], n),
    "customer_state": rng.choice(["SP", "RJ", "MG"], n),
})
z = (0.002 * df["dist_mean_km"] + 3.0 * df["seller_prior_rate"]
     - 0.01 * df["est_window_days"] + rng.normal(0, 1, n))
df["is_late"] = (z > 1.5).astype(int)

pre, model = train_mlp(df, epochs=2)          # 2 epochs: CI speed, not quality
p = predict_df(df.iloc[:200], pre, model)

assert len(p) == 200, "prediction count mismatch"
assert np.all((p >= 0) & (p <= 1)), "probabilities out of range"
assert p.min() != p.max(), "degenerate predictions"
print(f"smoke train OK: {len(df)} rows, 2 epochs, "
      f"pred range [{p.min():.3f}, {p.max():.3f}]")