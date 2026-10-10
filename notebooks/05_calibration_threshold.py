"""05 - Isotonic calibration (train-tail slice) + cost-derived threshold.

Slices: fit = train head 85% | calib = train tail 15% | test = untouched
Threshold from cost arithmetic: act when P(late) > C_upgrade / C_late.

v2 (pre-registered fallback): percentile operationalization.
The absolute threshold starves under prevalence shift (calib 6.4% -> test 3.3%,
flag rate 0.5%, coverage 2.2%); the deployed rule becomes "flag top-q% of each
batch" where q is the cost-optimal rate measured on the calibration slice.
Tiered intervention economics: profitable at $1 watch-list intervention in
calm regimes; $8 expediting requires spike-regime precision.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------- slices ----------
feat = pd.read_parquet("data/processed/features.parquet")
train = feat[feat.split == "train"].reset_index(drop=True)
n_cal = int(len(train) * 0.15)
fit_df, calib_df = train.iloc[:-n_cal], train.iloc[-n_cal:]
test_df = feat[feat.split == "test"].reset_index(drop=True)
print(f"fit {len(fit_df):,} | calib {len(calib_df):,} | test {len(test_df):,}")

CAT = ["top_category", "payment_type", "customer_state", "purchase_month", "purchase_dow"]
NUM = ["est_window_days", "n_items", "n_sellers", "total_price", "total_freight",
       "max_price", "freight_ratio", "dist_mean_km", "dist_max_km", "mean_weight_g",
       "mean_volume_l", "payment_installments", "purchase_hour", "seller_prior_orders",
       "seller_prior_rate"]

def prep(df):
    return df[CAT + NUM]

def make_pre():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CAT),
        ("num", StandardScaler(), NUM),
    ])

# ---------- champion (retrained on fit slice, seeded, ~5s) ----------
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_mlp(df, seed=42):
    torch.manual_seed(seed)
    pre = make_pre().fit(prep(df))
    X = pre.transform(prep(df)).toarray().astype(np.float32)
    y = df["is_late"].values.astype(np.float32)
    pos_w = torch.tensor((1 - y).sum() / y.sum(), dtype=torch.float32)
    m = MLP(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    m.train()
    for epoch in range(10):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            loss_fn(m(Xt[idx]), yt[idx]).backward()
            opt.step()
    m.eval()
    return pre, m

def predict(m, pre, df):
    X = pre.transform(prep(df)).toarray().astype(np.float32)
    with torch.no_grad():
        return torch.sigmoid(m(torch.from_numpy(X))).numpy()

pre, model = train_mlp(fit_df)
print(f"champion retrained on fit slice ({len(fit_df):,} rows)")

# ---------- isotonic on calib slice ----------
p_calib_raw = predict(model, pre, calib_df)
y_calib = calib_df["is_late"].values
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_calib_raw, y_calib)

p_test_raw = predict(model, pre, test_df)
p_test_cal = iso.predict(p_test_raw)
y_test = test_df["is_late"].values

def ece(y, p, n_bins=10):
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return sum((bins == b).mean() * abs(y[bins == b].mean() - p[bins == b].mean())
               for b in range(n_bins) if (bins == b).any())

print(f"\ncalib slice late rate: {y_calib.mean():.1%} | test late rate: {y_test.mean():.1%}")
print(f"test ECE: raw {ece(y_test, p_test_raw):.4f} -> isotonic {ece(y_test, p_test_cal):.4f}")
print(f"test Brier: raw {brier_score_loss(y_test, p_test_raw):.4f} -> "
      f"isotonic {brier_score_loss(y_test, p_test_cal):.4f}")
print(f"test ROC: raw {roc_auc_score(y_test, p_test_raw):.4f} -> "
      f"isotonic {roc_auc_score(y_test, p_test_cal):.4f}")

# ---------- v1: absolute threshold from cost arithmetic ----------
C_UP, C_LATE, C_EXP = 8.0, 45.0, 2.0
T_ACT = C_UP / C_LATE
print(f"\nv1 absolute threshold: C_upgrade/C_late = {C_UP}/{C_LATE} = {T_ACT:.3f}")

act = p_test_cal >= T_ACT
tp = int((act & (y_test == 1)).sum())
fp = int((act & (y_test == 0)).sum())
lates = int(y_test.sum())

cost_policy = fp * C_UP + tp * C_EXP + (lates - tp) * C_LATE
cost_do_nothing = lates * C_LATE
print(f"\n--- TEST outcomes, v1 (reported once) ---")
print(f"orders flagged for expedite: {act.sum():,} ({act.mean():.1%})")
print(f"lates covered: {y_test[act].sum():,}/{lates:,} ({act[ y_test==1 ].mean() if lates else 0:.1%}) | "
      f"precision of flag: {tp/max(act.sum(),1):.1%}")
print(f"cost: policy ${cost_policy:,.0f} vs do-nothing ${cost_do_nothing:,.0f} "
      f"({1 - cost_policy/cost_do_nothing:+.1%} saved)")

# ---------- v2: percentile operationalization (pre-registered fallback) ----------
p_calib_cal = iso.predict(p_calib_raw)
q_rate = float((p_calib_cal >= T_ACT).mean())
print(f"\nv2 percentile rule: cost-optimal rate on calib = {q_rate:.1%} "
      f"-> flag top {q_rate:.1%} of each batch")

n_flag = int(round(q_rate * len(p_test_cal)))
flags = np.zeros(len(p_test_cal), dtype=bool)
flags[np.argsort(-p_test_cal)[:n_flag]] = True
tp2 = int((flags & (y_test == 1)).sum())
fp2 = int((flags & (y_test == 0)).sum())
cost2 = fp2 * C_UP + tp2 * C_EXP + (lates - tp2) * C_LATE
print(f"v2 on test: flags {n_flag:,} | precision {tp2/max(n_flag,1):.1%} | "
      f"coverage {tp2/max(lates,1):.1%} | cost ${cost2:,.0f} vs "
      f"${cost_do_nothing:,.0f} ({1 - cost2/cost_do_nothing:+.1%})")

cost2_1 = fp2 * 1.0 + tp2 * C_EXP + (lates - tp2) * C_LATE
print(f"v2 at $1 intervention: cost ${cost2_1:,.0f} vs ${cost_do_nothing:,.0f} "
      f"({1 - cost2_1/cost_do_nothing:+.1%})")

be8 = C_UP / (C_LATE - C_EXP)
be1 = 1.0 / (C_LATE - C_EXP)
print(f"break-even flag precision: ${C_UP:.0f} intervention -> {be8:.1%} | "
      f"$1 intervention -> {be1:.1%}")

policy_v2 = {"rule": "top-q percentile of calibrated scores",
             "q": round(q_rate, 4),
             "test_flags": n_flag,
             "test_precision": round(tp2 / max(n_flag, 1), 4),
             "test_coverage": round(tp2 / max(lates, 1), 4),
             "test_cost_saved_pct": round(float(1 - cost2 / cost_do_nothing), 4),
             "test_cost_saved_pct_dollar_intervention": round(float(1 - cost2_1 / cost_do_nothing), 4),
             "break_even_precision": {"$8": round(be8, 4), "$1": round(be1, 4)}}

# ---------- persist ----------
torch.save({"pre": pre, "state_dict": model.state_dict(),
            "in_dim": pre.transform(prep(fit_df)).shape[1]}, "models/champion.pt")

import joblib
joblib.dump(iso, "models/calibrator.joblib")
with open("evaluation/calibration_config.json", "w") as f:
    json.dump({
        "threshold_act": round(T_ACT, 4),
        "costs": {"C_upgrade": C_UP, "C_late": C_LATE, "C_expedite_late": C_EXP},
        "calib_slice": {"rows": len(calib_df),
                        "late_rate": round(float(y_calib.mean()), 4)},
        "test_metrics": {
            "ece_raw": round(float(ece(y_test, p_test_raw)), 4),
            "ece_calibrated": round(float(ece(y_test, p_test_cal)), 4),
            "brier_calibrated": round(float(brier_score_loss(y_test, p_test_cal)), 4),
            "roc_auc_calibrated": round(float(roc_auc_score(y_test, p_test_cal)), 4),
            "pr_auc_calibrated": round(float(average_precision_score(y_test, p_test_cal)), 4),
        },
        "test_outcomes": {
            "flag_rate": round(float(act.mean()), 4),
            "late_coverage": round(tp / max(lates, 1), 4),
            "flag_precision": round(tp / max(int(act.sum()), 1), 4),
            "cost_saved_pct": round(float(1 - cost_policy / cost_do_nothing), 4),
        },
        "policy_v2_percentile": policy_v2,
    }, f, indent=2)
print("\nsaved: models/champion.pt, models/calibrator.joblib, evaluation/calibration_config.json")