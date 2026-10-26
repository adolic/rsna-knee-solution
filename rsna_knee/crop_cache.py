"""Crop caches for the crop specialists: keypoint-centred square views (configs/crops/crop-config-*.json) of every knee.

Per knee (the cache's knee_frame.csv), view of the config and series of the view's plane that covers it (crops.plan_series:
the slab around the view centre must hold >= min_triplets windows): the stored slices (crops.stored_indices: the slab plus
one window step each side; thin series every s-th slice) cut at the view's width.
  source "canvas"  (meniscus, ACL and OA-family specialists): cut from the knee512 canvases with a bicubic remap onto the
                   view square, canonical in-plane axes, slices in the reader's order; exactly what inference does
                   (specialists.row_bag on the same canvases).
  source "dicom"   (ResNet50 OA specialist): cut from the decoded DICOM series (per-series 1-99 % window, canonical patient
                   frame; crops.canonical_series / crop_slice), as inference does from its oriented series.
Output folder: studies/<study>.npy (uint8 [N, size, size], every entry of the study concatenated), studies/<study>.json,
index.csv (one row per (knee, view, series): study_uid, knee, side, view, plane, series_uid, fluid, start, count, rel_mm
(json: stored slices' position from the view centre along the series normal), slab_lo / slab_hi, sigma_mm, infer (json:
stored-slice indices of the inference window centres), width_mm, native_spacing_mm, stride, knee_scale), manifest.json."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .crops import CANON_AXES, canonical_series, crop_slice, knee_geometry, plan_series, stored_indices, view_centre, view_width


def series_geo(path):
    """A canvas series in the reader's slice order: positions along the normal whose largest component is positive."""
    z = np.load(path)
    normal = z["normal"].astype(float)
    normal = normal * (np.sign(normal[int(np.argmax(np.abs(normal)))]) or 1.0)
    order = np.argsort(z["origin"] @ normal, kind="stable")
    origin = z["origin"][order]
    return dict(origin=origin, u=z["u"], v=z["v"], normal=normal, pos=origin @ normal, order=order)


def canon_axes(S, plane):
    a, b = CANON_AXES[plane]
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


def _row(study, kr, name, v, e, start, idx, rel, p, k, infer, W):
    return dict(study_uid=study, knee=kr["knee"], side=kr["side"], view=name, plane=v["plane"], series_uid=e["series_uid"],
                fluid=int(e["fluid"]), start=start, count=len(idx), rel_mm=json.dumps([round(float(x), 2) for x in rel]),
                slab_lo=round(p["lo"], 2), slab_hi=round(p["hi"], 2), sigma_mm=round(v["sigma_mm"] * k, 2), infer=json.dumps(infer),
                width_mm=round(W, 2), native_spacing_mm=round(p["spacing"], 3), stride=p["stride"], knee_scale=round(k, 4))


def build_study(job):
    study, cfg, out, knees, entries, size, source, paths = job
    dst = Path(out) / "studies" / f"{study}.npy"
    if dst.exists() and dst.with_suffix(".json").exists():
        return json.loads(dst.with_suffix(".json").read_text())
    dicom = {}
    if source == "dicom":
        for e in entries:
            if e["series_uid"] not in dicom:
                dicom[e["series_uid"]] = canonical_series([Path(p) for p in paths.get(e["series_uid"], [])])
    chunks, rows, start = [], [], 0
    for kr in knees:
        anchors, axes, k = knee_geometry(kr)
        for name, v in cfg["views"].items():
            c = view_centre(v, anchors, axes, k); W = view_width(v, k)
            for e in entries:
                if e["knee"] != kr["knee"] or e["plane"] != v["plane"]:
                    continue
                if source == "dicom":
                    S = dicom.get(e["series_uid"])
                    if S is None:
                        continue
                    p = plan_series(v, cfg["rules"], S["pos"], S["normal"], c, axes, k)
                    if p is None:
                        continue
                    idx = stored_indices(p, len(S["pos"]))
                    imgs = np.stack([(crop_slice(S, int(i), c, W, size) * 255).round().astype(np.uint8) for i in idx])
                else:
                    S = series_geo(e["geometry"])
                    p = plan_series(v, cfg["rules"], S["pos"], S["normal"], c, axes, k)
                    if p is None:
                        continue
                    idx = stored_indices(p, len(S["pos"]))
                    vol = np.load(e["series"], mmap_mode="r"); r, cc = canon_axes(S, v["plane"])
                    imgs = np.stack([crop_canvas(vol[int(S["order"][i])], S, int(i), c, W, size, r, cc) for i in idx])
                rel = S["pos"][idx] - c @ S["normal"]
                infer = sorted({int(np.argmin(np.abs(idx - kk))) for kk in p["infer_k"]})
                chunks.append(imgs); rows.append(_row(study, kr, name, v, e, start, idx, rel, p, k, infer, W)); start += len(idx)
    arr = np.concatenate(chunks) if chunks else np.zeros((0, size, size), np.uint8)
    tmp = dst.with_name(".tmp-" + dst.name); np.save(tmp, arr); tmp.replace(dst)
    meta = dict(study_uid=study, slices=int(len(arr)), entries=rows)
    dst.with_suffix(".json").write_text(json.dumps(meta))
    return meta


def build(config, cache, out, size, source="canvas", dicom_folder=None, split="train", studies=None, workers=8, views=None):
    """config: crop config JSON path; cache: a knee512 cache (rsna_knee.cache: frames, canvases, series-to-knee index)."""
    full = json.loads(Path(config).read_text()); cache, out = Path(cache), Path(out)
    # views: a subset of the config's views to cut (e.g. the six OA views of crop-config-v5); the manifest keeps the full
    # config because the models' view ids index its view list
    cfg = dict(full, views={k: v for k, v in full["views"].items() if not views or k in views})
    (out / "studies").mkdir(parents=True, exist_ok=True)
    kf = pd.read_csv(cache / "knee_frame.csv"); kf = kf[kf.available == 1]
    idx = pd.read_csv(cache / "knee512/index.csv")
    planes = {v["plane"] for v in cfg["views"].values()}
    idx = idx[idx.plane.isin(planes)]
    paths = {}
    if source == "dicom":
        folder = Path(dicom_folder)
        for s, u in zip(idx.study_uid, idx.series_uid):
            paths.setdefault(s, {})[u] = [str(p) for p in sorted((folder / f"{split}_series" / s / u).glob("*.dcm"))]
    by_kf = {s: g for s, g in kf.groupby("study_uid")}; by_ix = {s: g for s, g in idx.groupby("study_uid")}
    jobs = []
    for st in (studies or sorted(by_kf)):
        if st not in by_kf or st not in by_ix:
            continue
        ent = [dict(knee=r.knee, plane=r.plane, series_uid=r.series_uid, fluid=r.fluid,
                    geometry=str(cache / "knee512/geometry" / f"{r.file}.npz"), series=str(cache / "knee512/series" / f"{r.file}.npy"))
               for r in by_ix[st].itertuples()]
        jobs.append((st, cfg, str(out), [r._asdict() for r in by_kf[st].itertuples(index=False)], ent, size, source, paths.get(st, {})))
    metas = []
    with ProcessPoolExecutor(workers) as ex:
        for i, m in enumerate(ex.map(build_study, jobs, chunksize=2)):
            metas.append(m)
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} studies", flush=True)
    index = pd.DataFrame([e for m in metas for e in m["entries"]])
    index.to_csv(out / "index.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(dict(name=out.name, image_size=size, source=source, views=sorted(cfg["views"]), config=full,
                                                       studies=len(metas), entries=len(index)), indent=1) + "\n")
    print(f"{len(metas)} studies, {len(index)} entries -> {out}", flush=True)
    return index
