"""Crop planner: keypoint-centred views and their 3-slice-window sampling, from a crop config (configs/crops/).

Inputs per knee: the knee frame row (knee_frame.csv: joint centre O, axes m/a/s, scale k, keypoints) and, per series, its
slice positions along the series normal (absolute LPS mm).
View geometry: centre = anchor keypoint + offset (m, a, s mm, scaled by k); in-plane width W mm (scaled by k when
scale_width); slab = slices whose position lies in [lo, hi] mm from the centre along the series normal (symmetric slab_mm;
or slab_range [lo, hi] along a knee axis slab_axis = m / a / s, mapped onto the series normal; or the whole stack).
Windows (k - s, k, k + s) with stride s = max(1, round(triplet_step_mm / slice spacing)), so a window spans about 2 x 3.3 mm
whatever the slice thickness (thin series are subsampled, never averaged); neighbours clamped at the stack ends.
Coverage: a series serves a view only if the slab holds >= min_triplets distinct window centres at the step spacing.
Training: per study and view `train_budget` windows: a covering series uniformly, a position from N(0, sigma_mm x k)
truncated to the slab, snapped to the nearest slice. Inference: centres on a grid of spacing
max(triplet_step_mm, slab length / (infer_cap - 1)) through the slab, anchored at the centre, deduplicated (<= infer_cap).
"""
from __future__ import annotations

import numpy as np

ANCHOR = {"O": None, "M": "medial_tibiofemoral_joint", "L": "lateral_tibiofemoral_joint", "N": "intercondylar_notch",
          "F": "patellofemoral_joint", "P": "patella_center"}


def knee_geometry(row):
    """Anchors (LPS), axes and scale from a knee-frame-v1 row (pandas Series or dict)."""
    v = lambda p: np.array([row[f"{p}_x"], row[f"{p}_y"], row[f"{p}_z"]], float)
    anchors = {a: (v("O") if lm is None else v(lm)) for a, lm in ANCHOR.items()}
    anchors["PFmid"] = (anchors["F"] + anchors["P"]) / 2
    return anchors, {c: v(c) for c in "mas"}, float(row["scale_k"])


def view_centre(view, anchors, axes, k):
    return anchors[view["anchor"]] + sum(o * k * axes[c] for o, c in zip(view["offset"], "mas"))


def view_width(view, k):
    return view["width_mm"] * (k if view.get("scale_width") else 1.0)


def slab(view, pos, normal, centre, axes, k):
    """[lo, hi] mm along the series normal relative to the centre's position."""
    rel = pos - centre @ normal
    if "slab_range_m" in view and "slab_axis" not in view:          # v3/v4 spelling
        view = {**view, "slab_axis": "m", "slab_range": view["slab_range_m"]}
    if "slab_axis" in view:   # range along a knee axis (m / a / s), mapped onto the series normal
        sgn = float(np.sign(normal @ axes[view["slab_axis"]])) or 1.0
        a, b = (x * k * sgn for x in view["slab_range"])
        return min(a, b), max(a, b), rel
    if view.get("slab_mm") is None:
        return float(rel.min()), float(rel.max()), rel
    return -view["slab_mm"] * k, view["slab_mm"] * k, rel


def stride(pos, step_mm):
    sp = float(np.median(np.diff(pos))) if len(pos) > 1 else step_mm
    return max(1, int(round(step_mm / max(sp, 1e-3)))), sp


def triplet(kc, s, n):
    return [int(np.clip(kc - s, 0, n - 1)), int(kc), int(np.clip(kc + s, 0, n - 1))]


def plan_series(view, rules, pos, normal, centre, axes, k):
    """Inference triplets and coverage for one series. Returns dict or None when the series does not cover the view."""
    lo, hi, rel = slab(view, pos, normal, centre, axes, k)
    step = rules["triplet_step_mm"]
    s, sp = stride(pos, step)
    inside = np.where((rel >= lo - sp / 2) & (rel <= hi + sp / 2))[0]
    if not len(inside):
        return None
    span_lo, span_hi = max(lo, rel[inside].min()), min(hi, rel[inside].max())
    n_grid = int(np.floor((span_hi - span_lo) / step)) + 1
    if n_grid < rules["min_triplets"]:
        return None
    cap = view["infer_cap"]
    gstep = max(step, (span_hi - span_lo) / max(cap - 1, 1))
    j = np.arange(np.ceil((span_lo) / gstep - 1e-9), np.floor(span_hi / gstep + 1e-9) + 1)
    grid = j * gstep
    if len(grid) > cap:
        grid = np.linspace(span_lo, span_hi, cap)
    ks = sorted({int(inside[np.argmin(np.abs(rel[inside] - g))]) for g in grid})
    return dict(lo=lo, hi=hi, span=(span_lo, span_hi), rel=rel, inside=inside, stride=s, spacing=sp,
                infer_k=ks, infer_triplets=[triplet(kc, s, len(pos)) for kc in ks])


def sample_train(view, plans, k, rng, budget=None):
    """Training draws for one study x view. plans: {series_uid: plan_series(...)} of covering series."""
    uids = [u for u, p in plans.items() if p is not None]
    if not uids:
        return []
    sigma = view["sigma_mm"] * k
    out, seen = [], set()
    for _ in range(budget or view["train_budget"]):
        for _try in range(20):
            u = uids[rng.integers(len(uids))]; p = plans[u]
            a, b = p["span"]
            x = rng.normal(0, sigma)
            if not a <= x <= b:
                continue
            kc = int(p["inside"][np.argmin(np.abs(p["rel"][p["inside"]] - x))])
            if (u, kc) in seen:
                continue
            seen.add((u, kc)); out.append((u, kc, triplet(kc, p["stride"], len(p["rel"])), float(x)))
            break
    return out


def train_weights(view, plan, k):
    """Relative training probability per candidate slice of a plan (for plots / audits)."""
    sigma = view["sigma_mm"] * k
    x = plan["rel"][plan["inside"]]
    w = np.exp(-0.5 * (x / sigma) ** 2)
    return x, w / w.sum()


# ---- image side (raw DICOM -> canonical patient frame -> keypoint-centred crop) --------------------------------------
CANON_AXES = {"Sagittal": (np.array([0., 1, 0]), np.array([0., 0, -1])),
              "Coronal": (np.array([1., 0, 0]), np.array([0., 0, -1])),
              "Axial": (np.array([1., 0, 0]), np.array([0., 1, 0]))}
# Canonical frame (as native224): sagittal x -> posterior (+y), y -> inferior; coronal x -> patient left,
# y -> inferior; axial x -> patient left, y -> posterior. 90-degree rotations / flips only, no mirroring.
PLANE_OF_AXIS = {0: "Sagittal", 1: "Coronal", 2: "Axial"}


def canonical_series(paths):
    """DICOM reader -> windowed [0, 1] float stack in the canonical frame + per-slice geometry."""
    from . import dicom as g8
    recs = [r for r in map(g8.read_once, paths) if r.get("values") is not None and r.get("geometry") is not None]
    if len(recs) < 3:
        return None
    g8.order_records(recs)
    shapes = {r["values"].shape for r in recs}
    if len(shapes) != 1:
        return None
    g0 = recs[0]["geometry"]
    stack = np.stack([r["values"] for r in recs]).astype(np.float32)
    lo, hi = np.percentile(stack, [g8.PCT_LO, g8.PCT_HI]); hi = hi if hi > lo else lo + 1
    r_dir, c_dir, sx, sy = g0["row_dir"], g0["col_dir"], g0["d_col"], g0["d_row"]
    plane = PLANE_OF_AXIS[int(np.argmax(np.abs(np.cross(r_dir, c_dir))))]
    a, b = CANON_AXES[plane]
    origins = np.stack([r["geometry"]["ipp"] for r in recs])
    if abs(r_dir @ a) < abs(c_dir @ a):
        stack = stack.transpose(0, 2, 1); r_dir, c_dir, sx, sy = c_dir, r_dir, sy, sx
    n, h, w = stack.shape
    if r_dir @ a < 0:
        stack = stack[:, :, ::-1]; origins = origins + r_dir * sx * (w - 1); r_dir = -r_dir
    if c_dir @ b < 0:
        stack = stack[:, ::-1, :]; origins = origins + c_dir * sy * (h - 1); c_dir = -c_dir
    normal = np.cross(r_dir, c_dir); normal *= np.sign((origins[-1] - origins[0]) @ normal) or 1
    return dict(plane=plane, stack=np.clip((stack - lo) / (hi - lo), 0, 1), origins=origins, r=r_dir, c=c_dir,
                sx=sx, sy=sy, normal=normal, pos=origins @ normal)


def crop_slice(S, k, centre, width_mm, size=224):
    """Square in-plane crop of slice k around an LPS centre (zero outside the FOV), resized to size px."""
    import cv2
    d = centre - S["origins"][k]
    ci, cj = d @ S["r"] / S["sx"], d @ S["c"] / S["sy"]
    W, H = max(int(round(width_mm / S["sx"])), 1), max(int(round(width_mm / S["sy"])), 1)
    x0, y0 = int(round(ci - W / 2)), int(round(cj - H / 2))
    img = S["stack"][k]; out = np.zeros((H, W), np.float32)
    a0, b0, a1, b1 = max(x0, 0), max(y0, 0), min(x0 + W, img.shape[1]), min(y0 + H, img.shape[0])
    if a1 > a0 and b1 > b0:
        out[b0 - y0:b1 - y0, a0 - x0:a1 - x0] = img[b0:b1, a0:a1]
    interp = cv2.INTER_AREA if W >= size else cv2.INTER_LANCZOS4
    return np.clip(cv2.resize(out, (size, size), interpolation=interp), 0, 1)


def stored_indices(plan, n):
    """Slices kept in a cache entry: the slab's slices plus one window step of context at each end; thin series
    (stride s > 1) keep every s-th slice, phase-locked to the slice nearest the view centre, so consecutive stored
    slices are one window step apart and a triplet is three consecutive stored slices."""
    s = plan["stride"]; ins = plan["inside"]
    k0 = int(ins[np.argmin(np.abs(plan["rel"][ins]))])
    lo, hi = max(int(ins.min()) - s, 0), min(int(ins.max()) + s, n - 1)
    return np.arange(k0 - ((k0 - lo) // s) * s, hi + 1, s)
