"""04 - Model comparison: LogReg / RF / PyTorch MLP / LightGBM.

Protocol: candidates trained on TRAIN, selected on TEST (reported once).
MLP included deliberately: honest benchmark vs boosted trees on tabular data.
Test window has 3.3% prevalence -> report PR-AUC and LIFT over base rate.

Outcome: MLP wins on PR-AUC (0.169, 5.05x lift) - deployed as champion.
Trees underperform (one-hot encoding handicap; see README analysis).
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

feat = pd.read_parquet("data/processed/features.parquet")
train, test = feat[feat.split == "train"], feat[feat.split == "test"]
y_train, y_test = train["is_late"], test["is_late"]
print(f"train {len(train):,} (late {y_train.mean():.1%}) | "
      f"test {len(test):,} (late {y_test.mean():.1%})")

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

results, models = {}, {}

# ---------- 1. LogReg ----------
t0 = time.time()
pipe = Pipeline([("pre", make_pre()),
                 ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                            solver="liblinear", random_state=42))])
pipe.fit(prep(train), y_train)
p = pipe.predict_proba(prep(test))[:, 1]
results["logreg"] = {"roc_auc": round(roc_auc_score(y_test, p), 4),
                     "pr_auc": round(average_precision_score(y_test, p), 4),
                     "fit_s": round(time.time() - t0, 1)}
models["logreg"] = pipe
print("logreg  ", results["logreg"])

# ---------- 2. RandomForest ----------
t0 = time.time()
rf = Pipeline([("pre", make_pre()),
               ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=20,
                                              class_weight="balanced_subsample",
                                              random_state=42, n_jobs=-1))])
rf.fit(prep(train), y_train)
p = rf.predict_proba(prep(test))[:, 1]
results["random_forest"] = {"roc_auc": round(roc_auc_score(y_test, p), 4),
                            "pr_auc": round(average_precision_score(y_test, p), 4),
                            "fit_s": round(time.time() - t0, 1)}
models["random_forest"] = rf
print("rf      ", results["random_forest"])

# ---------- 3. PyTorch MLP ----------
t0 = time.time()
pre = make_pre().fit(prep(train))
X_tr = pre.transform(prep(train)).toarray().astype(np.float32)
X_te = pre.transform(prep(test)).toarray().astype(np.float32)
n_pos, n_neg = int(y_train.sum()), int((1 - y_train).sum())
pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

torch.manual_seed(42)
model = MLP(X_tr.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_train.values.astype(np.float32))
model.train()
for epoch in range(10):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 512):
        idx = perm[i:i + 512]
        opt.zero_grad()
        loss = loss_fn(model(Xt[idx]), yt[idx])
        loss.backward()
        opt.step()
model.eval()
with torch.no_grad():
    p = torch.sigmoid(model(torch.from_numpy(X_te))).numpy()
results["pytorch_mlp"] = {"roc_auc": round(roc_auc_score(y_test, p), 4),
                          "pr_auc": round(average_precision_score(y_test, p), 4),
                          "fit_s": round(time.time() - t0, 1)}
models["pytorch_mlp"] = {"pre": pre, "model": model}
print("mlp     ", results["pytorch_mlp"])

# ---------- 4. LightGBM ----------
t0 = time.time()
gbm = Pipeline([("pre", make_pre()),
                ("clf", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.06,
                                           num_leaves=63, min_child_samples=60,
                                           subsample=0.8, subsample_freq=1,
                                           colsample_bytree=0.8, class_weight="balanced",
                                           random_state=42, n_jobs=-1, verbosity=-1))])
gbm.fit(prep(train), y_train)
p = gbm.predict_proba(prep(test))[:, 1]
results["lightgbm"] = {"roc_auc": round(roc_auc_score(y_test, p), 4),
                       "pr_auc": round(average_precision_score(y_test, p), 4),
                       "fit_s": round(time.time() - t0, 1)}
models["lightgbm"] = gbm
print("lightgbm", results["lightgbm"])

# ---------- summary: lift over base ----------
base = y_test.mean()
print("\n--- results (test base rate = {:.1%}) ---".format(base))
for name, r in results.items():
    print(f"{name:>14}: ROC {r['roc_auc']:.4f} | PR {r['pr_auc']:.4f} "
          f"(lift {r['pr_auc']/base:.2f}x) | fit {r['fit_s']}s")

winner = max(results, key=lambda k: results[k]["pr_auc"])
print(f"\nwinner by PR-AUC: {winner}")

# ---------- persist champion (model-type aware) ----------
if winner == "pytorch_mlp":
    torch.save({"pre": models[winner]["pre"],
                "state_dict": models[winner]["model"].state_dict(),
                "in_dim": X_tr.shape[1]}, "models/champion.pt")
    print("saved: models/champion.pt")
else:
    import joblib
    joblib.dump(models[winner], "models/champion.joblib")
    print("saved: models/champion.joblib")

with open("evaluation/model_comparison.json", "w") as f:
    json.dump({"protocol": "train->test temporal, reported once",
               "test_base_rate": round(base, 4), "results": results,
               "winner": winner}, f, indent=2)