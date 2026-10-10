"""Drift primitives: Population Stability Index + feature drift reports.

Thresholds (credit-risk convention): PSI < 0.10 stable, 0.10-0.20 moderate
shift (warn), >= 0.20 major shift (alarm).
"""
import numpy as np

PSI_WARN, PSI_ALARM = 0.10, 0.20

def psi(reference, current, bins=10, eps=1e-6):
    """PSI between two samples of one numeric feature.

    Quantile bins derived from the reference; end bins open (-inf, +inf) so
    out-of-range current values land in the edge bins, not the void.
    Proportions clipped by eps: zero-count bins are the classic PSI footgun
    (log 0 / div 0); clipping bounds their contribution.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:          # degenerate (constant) reference
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, bins=edges)[0] / len(ref)
    c = np.histogram(cur, bins=edges)[0] / len(cur)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(((c - r) * np.log(c / r)).sum())

def classify(value):
    if value >= PSI_ALARM:
        return "alarm"
    if value >= PSI_WARN:
        return "warn"
    return "ok"

def feature_drift_report(reference, current, features):
    """PSI per feature + status classification."""
    rows = []
    for f in features:
        s = psi(reference[f].values, current[f].values)
        rows.append({"feature": f, "psi": round(s, 4), "status": classify(s)})
    return sorted(rows, key=lambda r: -r["psi"])