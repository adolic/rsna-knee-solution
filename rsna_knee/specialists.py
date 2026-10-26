"""Canvas-crop specialists at inference (meniscus, ACL, OA family): crops cut from the study's own knee512 canvases.

Per knee and specialist row (target + views of the row's crop config): every canvas series of the view's plane, slices
re-indexed into the reader's order (ascending along the normal whose largest component is positive), crops.plan_series /
stored_indices, crops of the stored slices (bicubic remap of the canvas onto the view square, canonical in-plane axes, zero
outside); windows = the plan's inference grid, 3 consecutive stored crops, clipped at the ends; view id = the row's view id.
The row's logit = mean over the fold models of the row's head. A row may read several crop configs (OA family: compartment
crops + zoomed cartilage crops): row_bag_parts concatenates the parts' windows into one bag, as the trainer does.
"""
from __future__ import annotations

import numpy as np
import cv2

from . import crops as gc


def series_geo(geo):
    """Canvas geometry (pipeline.Geo512) -> the reader's slice order (as crop_cache.series_geo)."""
    normal = np.asarray(geo.normal, float)
    normal = normal * (np.sign(normal[int(np.argmax(np.abs(normal)))]) or 1.0)
    order = np.argsort(geo.origin @ normal, kind="stable")
    origin = geo.origin[order]
    return dict(origin=origin, u=geo.u, v=geo.v, normal=normal, pos=origin @ normal, order=order)


def canon_axes(S, plane):
    a, b = gc.CANON_AXES[plane]
    cand = [S["u"] / np.linalg.norm(S["u"]), S["v"] / np.linalg.norm(S["v"])]
    i = int(abs(cand[1] @ a) > abs(cand[0] @ a))
    r, c = cand[i], cand[1 - i]
    return r * (np.sign(r @ a) or 1.0), c * (np.sign(c @ b) or 1.0)


def crop_canvas(img, S, k, centre, W, size, r, c):
    t = (np.arange(size) + .5) / size * W - W / 2
    P = centre[None, None, :] + t[None, :, None] * r + t[:, None, None] * c
    d = P - S["origin"][k]
    ci = (d @ S["u"]) / (S["u"] @ S["u"]); cj = (d @ S["v"]) / (S["v"] @ S["v"])
    return cv2.remap(np.ascontiguousarray(img), ci.astype(np.float32), cj.astype(np.float32), cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def row_bag(canv, row, cfg, views, view_ids, size, nslices=3, step=1):
    """canv: {uid: (uint8 [n, 512, 512], Geo512, plane, fluid)} of this knee. -> (images u8 [N, 3, size, size], view ids,
    fluid) or None."""
    anchors, axes, k = gc.knee_geometry(row)
    imgs, vids, flu = [], [], []
    half = nslices // 2; offsets = [step * d for d in range(-half, half + 1)]
    for name, vid in zip(views, view_ids):
        v = cfg["views"][name]; c = gc.view_centre(v, anchors, axes, k); W = gc.view_width(v, k)
        for u, (vol, geo, plane, fluid) in canv.items():
            if plane != v["plane"]:
                continue
            S = series_geo(geo)
            p = gc.plan_series(v, cfg["rules"], S["pos"], S["normal"], c, axes, k)
            if p is None:
                continue
            idx = gc.stored_indices(p, len(S["pos"]))
            r_ax, c_ax = canon_axes(S, plane)
            crops = np.stack([crop_canvas(vol[int(S["order"][i])], S, int(i), c, W, size, r_ax, c_ax) for i in idx])
            infer = sorted({int(np.argmin(np.abs(idx - kk))) for kk in p["infer_k"]})
            for j in infer:
                imgs.append(crops[[int(np.clip(j + d, 0, len(idx) - 1)) for d in offsets]])
                vids.append(int(vid)); flu.append(int(fluid))
    if not imgs:
        return None
    return np.stack(imgs), np.asarray(vids, np.int64), np.asarray(flu, np.int64)


def row_bag_parts(canv, row, parts, configs, size, nslices=3, step=1):
    """parts: [{config, views, view_ids}] (config = key into configs) -> one bag over all parts, or None if every part is empty."""
    bags = [row_bag(canv, row, configs[q["config"]], q["views"], q["view_ids"], size, nslices, step) for q in parts]
    bags = [b for b in bags if b is not None]
    if not bags:
        return None
    return tuple(np.concatenate([b[i] for b in bags]) for i in range(3))


def row_logit(models, bag, device, chunk=128, head=0):
    """Mean over the fold models of the row's head output."""
    import torch
    imgs, view, fluid = bag
    b = dict(images=torch.from_numpy(imgs).to(device), view=torch.from_numpy(view).to(device),
             fluid=torch.from_numpy(fluid).to(device), lengths=torch.tensor([len(view)]),
             depth=torch.zeros(len(view), device=device))
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for m in models:
            out.append(float(m(b, chunk=chunk).float().cpu().numpy()[0, head]))
    return float(np.mean(out)), out
