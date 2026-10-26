"""ResNet50 single-slice OA specialist at inference (medial / lateral / PF OA logits).

Per knee: the six OA views of crop-config-v5 (medial / lateral coronal + sagittal, PF axial + sagittal), every series of the
view's plane that covers it, the stored slices (crops.stored_indices) inside the slab +- 2 mm, each slice cut at the view
width to 320 px from the series' 1-99 % windowed stack (as the training crops, crop_cache.py source "dicom"), uint8,
repeated to 3 channels. Input: the oriented series of pipeline.orient and the knee frame row (pipeline.knee_row).
Logits of the 5 fold models averaged.
"""
from __future__ import annotations

import numpy as np
import cv2

from . import crops as gc

OA_VIEWS = ["med_cor", "med_sag", "lat_cor", "lat_sag", "pf_ax", "pf_sag"]


def series_geom(S):
    """normal / slice positions as crops.canonical_series (cached on the series dict)."""
    if "oa_pos" not in S:
        n = np.cross(S["r"], S["c"]); n = n * (np.sign((S["origins"][-1] - S["origins"][0]) @ n) or 1)
        S["oa_normal"], S["oa_pos"] = n, S["origins"] @ n
    return S["oa_normal"], S["oa_pos"]


def crop(S, k, centre, width_mm, size):
    """crops.crop_slice on the windowed slice (window first, then zero padding outside the FOV)."""
    lo, hi = S["native_pct"]; hi = hi if hi > lo else lo + 1
    img = np.clip((S["stack"][k] - lo) / (hi - lo), 0, 1).astype(np.float32)
    d = centre - S["origins"][k]
    ci, cj = d @ S["r"] / S["sx"], d @ S["c"] / S["sy"]
    W, H = max(int(round(width_mm / S["sx"])), 1), max(int(round(width_mm / S["sy"])), 1)
    x0, y0 = int(round(ci - W / 2)), int(round(cj - H / 2))
    out = np.zeros((H, W), np.float32)
    a0, b0, a1, b1 = max(x0, 0), max(y0, 0), min(x0 + W, img.shape[1]), min(y0 + H, img.shape[0])
    if a1 > a0 and b1 > b0:
        out[b0 - y0:b1 - y0, a0 - x0:a1 - x0] = img[b0:b1, a0:a1]
    interp = cv2.INTER_AREA if W >= size else cv2.INTER_LANCZOS4
    return (np.clip(cv2.resize(out, (size, size), interpolation=interp), 0, 1) * 255).round().astype(np.uint8)


def knee_bag(series, row, cfg, size=320, margin=2.0):
    """series: {uid: oriented S (with "fluid")} of this knee. Returns (images u8 [N, 3, size, size], view ids, fluid)."""
    anchors, axes, k = gc.knee_geometry(row)
    names = list(cfg["views"]); imgs, views, fluid = [], [], []
    for name in OA_VIEWS:
        v = cfg["views"][name]; c = gc.view_centre(v, anchors, axes, k); W = gc.view_width(v, k)
        for u, S in series.items():
            if S["plane"] != v["plane"]:
                continue
            normal, pos = series_geom(S)
            p = gc.plan_series(v, cfg["rules"], pos, normal, c, axes, k)
            if p is None:
                continue
            idx = gc.stored_indices(p, len(pos))
            rel = np.round(pos[idx] - c @ normal, 2); lo, hi = round(p["lo"], 2), round(p["hi"], 2)
            for i in idx[(rel >= lo - margin) & (rel <= hi + margin)]:
                imgs.append(crop(S, int(i), c, W, size)); views.append(names.index(name)); fluid.append(int(S["fluid"]))
    if not imgs:
        return None
    a = np.stack(imgs)[:, None].repeat(3, 1)
    return a, np.asarray(views, np.int64), np.asarray(fluid, np.int64)


def pf_logit(models, bag, device, chunk=128):
    """Mean over the fold models of the PF OA logit (index 2 of the 3 OA outputs)."""
    import torch
    imgs, view, fluid = bag
    b = dict(images=torch.from_numpy(imgs).to(device), view=torch.from_numpy(view).to(device),
             fluid=torch.from_numpy(fluid).to(device), lengths=torch.tensor([len(view)]),
             depth=torch.zeros(len(view), device=device))
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for m in models:
            out.append(m(b, chunk=chunk).float().cpu().numpy()[0])
    return float(np.mean(out, 0)[2]), np.mean(out, 0)
