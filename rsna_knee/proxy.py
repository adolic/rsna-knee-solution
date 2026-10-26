"""Kaggle-proxy AUC: the local selection metric (it tracked the public leaderboard better than plain macro AUC).

Per target: AUC over positive / negative pairs from the same stratum only (scanner model x field strength,
labels/proxy-strata.csv), each pair weighted by the product of the two studies' label confidences |2 * soft - 1|^2
(labels/soft.csv). Macro = weighted mean over the 12 targets: medial / lateral / PF OA x1.5, contusion / fracture x0.5,
others x1. Hard labels are the evaluation's own."""
from functools import lru_cache

import numpy as np
import pandas as pd

from .paths import LABELS, TARGETS

TARGET_WEIGHTS = {"Medial OA": 1.5, "Lateral OA": 1.5, "PF OA": 1.5, "Contusion": 0.5, "Fracture": 0.5}


@lru_cache(maxsize=1)
def reference():
    strata = pd.read_csv(LABELS / "proxy-strata.csv", index_col=0)["stratum"]
    soft = pd.read_csv(LABELS / "soft.csv", index_col=0).reindex(strata.index)[list(TARGETS)]
    _, codes = np.unique(strata.to_numpy(), return_inverse=True)
    return {u: i for i, u in enumerate(strata.index)}, codes, ((2 * soft.to_numpy(np.float64) - 1) ** 2).round(7)


def stratified_weighted_auc(y, p, groups, w):
    """Within-group weighted AUC, groups pooled by weighted pair mass; ties count half."""
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    ok = np.isfinite(y) & np.isfinite(p) & (w > 0)
    y, p, g, w = y[ok], p[ok], np.asarray(groups)[ok], w[ok]
    if not len(y):
        return None
    wp, wn = w * (y >= 0.5), w * (y < 0.5)
    order = np.lexsort((p, g))
    g, p, wp, wn = g[order], p[order], wp[order], wn[order]
    starts = np.r_[0, np.flatnonzero((np.diff(g) != 0) | (np.diff(p) != 0)) + 1]   # tie blocks of identical (group, score)
    block_g = g[starts]
    bp, bn = np.add.reduceat(wp, starts), np.add.reduceat(wn, starts)
    cum = np.cumsum(bn)
    group_start = np.r_[0, np.flatnonzero(np.diff(block_g) != 0) + 1]
    offset = np.repeat(np.r_[0, cum[group_start[1:] - 1]], np.diff(np.r_[group_start, len(bn)]))
    below = cum - bn - offset                       # negative mass strictly below, same group
    numerator = float(np.sum(bp * (below + 0.5 * bn)))
    gp, gn = np.add.reduceat(bp, group_start), np.add.reduceat(bn, group_start)
    denominator = float(np.sum(gp * gn))
    return numerator / denominator if denominator > 0 else None


def proxy_auc(study_uids, hard, prob):
    """(weighted macro proxy AUC, per-target list). Every study must be in the reference and every target defined."""
    lookup, codes, weights = reference()
    rows = np.array([lookup.get(str(u), -1) for u in study_uids])
    if (rows < 0).any():
        raise ValueError("proxy reference does not cover every study")
    per = [stratified_weighted_auc(np.asarray(hard)[:, t], np.asarray(prob)[:, t], codes[rows], weights[rows, t]) for t in range(12)]
    if any(v is None for v in per):
        raise ValueError("all 12 proxy AUCs must be defined")
    w = np.array([TARGET_WEIGHTS.get(t, 1.0) for t in TARGETS])
    return float(np.dot(w, per) / w.sum()), per
