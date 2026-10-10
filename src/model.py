"""Champion model: MLP definition, training, loading, prediction.

Canonical home for the model code (notebooks 04-05 are frozen experiment
records; this module is what monitoring, retraining and the API import).
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAT = ["top_category", "payment_type", "customer_state", "purchase_month", "purchase_dow"]
NUM = ["est_window_days", "n_items", "n_sellers", "total_price", "total_freight",
       "max_price", "freight_ratio", "dist_mean_km", "dist_max_km", "mean_weight_g",
       "mean_volume_l", "payment_installments", "purchase_hour", "seller_prior_orders",
       "seller_prior_rate"]

def make_pre():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CAT),
        ("num", StandardScaler(), NUM),
    ])

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_mlp(df, seed=42, epochs=10, lr=1e-3, wd=1e-4):
    """Train on a DataFrame with CAT+NUM+is_late columns. Returns (pre, model)."""
    torch.manual_seed(seed)
    pre = make_pre().fit(df[CAT + NUM])
    X = pre.transform(df[CAT + NUM]).toarray().astype(np.float32)
    y = df["is_late"].values.astype(np.float32)
    pos_w = torch.tensor((1 - y).sum() / y.sum(), dtype=torch.float32)
    m = MLP(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            loss_fn(m(Xt[idx]), yt[idx]).backward()
            opt.step()
    m.eval()
    return pre, m

def load_champion(path="models/champion.pt"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # contains sklearn pre
    pre = ckpt["pre"]
    m = MLP(ckpt["in_dim"])
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return pre, m

def predict_df(df, pre, model):
    """Raw (uncalibrated) probabilities for a features-format DataFrame."""
    X = pre.transform(df[CAT + NUM]).toarray().astype(np.float32)
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).numpy()