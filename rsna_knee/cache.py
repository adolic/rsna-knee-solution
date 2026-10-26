"""Preprocessed caches built from raw DICOM with the inference pipeline itself (pipeline.py), so training and inference
see identical inputs.

Per study (a Kaggle-style folder: <split>.csv, <split>_series.csv with Fluid_Sensitive, <split>_series/<study>/<series>/*.dcm):
  decode + orient every series -> native224 (whole FOV, 224 px; keypoint localizer input) -> 5 keypoint models ->
  knee frame per knee (two-knee studies: one frame per knee) -> knee512 canvases (0.3125 mm/px, 512 px around the knee
  centre, left knees mirrored).
Layout of a cache folder:
  native224/series/<series>.npy      uint8 [slices, 224, 224]          native224/geometry/<series>.npz   origin, u, v, normal
  knee512/series/<file>.npy          uint8 [slices, 512, 512]          knee512/geometry/<file>.npz       origin, u, v, normal
  knee512/index.csv                  one row per (knee, series): study_uid, knee, side, series_uid, plane, fluid, slices, file
  knee_frame.csv                     one row per knee: frame origin O, axes m / a / s, scale k, the 5 landmarks (LPS mm)
  rows/<study>.json                  per-study rows (resume marker; merged into the two CSVs by merge())
<file> = <study>__<knee>__<series>. Studies already in rows/ are skipped, so an interrupted build resumes."""
from __future__ import annotations

import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

_W = {}


def _init(models, devices, counter, lock, axial_every, native_only=False):
    import torch
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    from . import keypoint_model, pipeline
    with lock:
        wid = counter.value; counter.value += 1
    device = torch.device(devices[wid % len(devices)])
    kps = []
    for f in ([] if native_only else models):
        m = keypoint_model.Localizer(pretrained=False).to(device)
        m.load_state_dict(torch.load(f, map_location=device)); kps.append(m.eval())
    _W.update(P=pipeline, kps=kps, device=device, axial_every=axial_every, native_only=native_only)


def _save(path, **arrays):
    tmp = path.with_name(".tmp-" + path.name)
    with open(tmp, "wb") as fh:
        np.save(fh, arrays.pop("_array")) if "_array" in arrays else np.savez(fh, **arrays)
    tmp.replace(path)


def build_study(args):
    """args: (out dir, study uid, [(series uid, [paths], fluid)], knee512?) -> receipt dict; writes the study's files."""
    out, uid, series, do512 = args
    out = Path(out); P = _W["P"]; t0 = time.monotonic()
    try:
        oriented, native, rec = P.prepare_series([(u, [Path(p) for p in ps], f) for u, ps, f in series])
        if _W["axial_every"] > 1:   # thinner axial stacks (OAI 3D DESS, 1.5 mm) subsampled to ~3 mm like the RSNA axials
            for S in oriented.values():
                if S["plane"] == "Axial":
                    k = np.arange(0, len(S["stack"]), _W["axial_every"])
                    S["stack"], S["origins"] = S["stack"][k], S["origins"][k]
        for u, (vol, g, plane) in native.items():
            _save(out / "native224/series" / f"{u}.npy", _array=vol)
            _save(out / "native224/geometry" / f"{u}.npz", **{k: np.asarray(v) for k, v in g.items()})
        if _W["native_only"]:   # bootstrap: native224 only (keypoint training input), no frames / canvases yet
            res = dict(study=uid, status="ok", index=[], frames=[], seconds=time.monotonic() - t0)
            tmp = out / "rows-native" / f".tmp-{uid}.json"; tmp.write_text(json.dumps(res)); tmp.replace(out / "rows-native" / f"{uid}.json")
            return dict(study=uid, status="ok", seconds=res["seconds"], error=None)
        frames = P.knee_frames(native, _W["kps"], _W["device"]) if native else []
        idx, kf = [], []
        for knee, row, fallback in frames:
            kf.append(dict(study_uid=uid, **row, available=1, frame_fallback=int(fallback),
                           n_series=len(native), planes="".join(sorted({p[0] for _, _, p in native.values()}))))
            if not do512:
                continue
            for u, (vol, geo, plane, fluid) in P.knee_canvases(oriented, row).items():
                f = f"{uid}__{knee}__{u}"
                _save(out / "knee512/series" / f"{f}.npy", _array=vol)
                _save(out / "knee512/geometry" / f"{f}.npz", origin=geo.origin, u=geo.u, v=geo.v, normal=geo.normal)
                idx.append(dict(study_uid=uid, knee=knee, side=row["side"], series_uid=u, plane=plane, fluid=int(fluid),
                                slices=int(len(vol)), file=f))
        res = dict(study=uid, status="ok" if frames else "no-knee", index=idx, frames=kf, seconds=time.monotonic() - t0,
                   **{k: v for k, v in rec.items() if k != "decode_seconds"})
    except Exception as error:
        res = dict(study=uid, status="error", error=f"{type(error).__name__}: {error}", trace=traceback.format_exc()[-1200:],
                   index=[], frames=[], seconds=time.monotonic() - t0)
    tmp = out / "rows" / f".tmp-{uid}.json"; tmp.write_text(json.dumps(res, default=float)); tmp.replace(out / "rows" / f"{uid}.json")
    return dict(study=uid, status=res["status"], seconds=res["seconds"], error=res.get("error"))


def build(out, jobs, keypoint_models, devices, workers, knee512=True, axial_every=1, native_only=False):
    """jobs: [(study uid, [(series uid, [paths], fluid)])]; studies with rows/<study>.json (native_only: rows-native/) are skipped."""
    out = Path(out); marker = "rows-native" if native_only else "rows"
    for d in ("native224/series", "native224/geometry", "knee512/series", "knee512/geometry", "rows", "rows-native"):
        (out / d).mkdir(parents=True, exist_ok=True)
    todo = [(str(out), uid, series, knee512) for uid, series in jobs if not (out / marker / f"{uid}.json").exists()]
    print(f"{len(jobs) - len(todo)} studies already built, {len(todo)} to build", flush=True)
    ctx = mp.get_context("spawn"); counter, lock = ctx.Value("i", 0), ctx.Lock(); t0 = time.time(); bad = 0
    with ctx.Pool(workers, initializer=_init, initargs=([str(m) for m in keypoint_models], devices, counter, lock, axial_every, native_only)) as pool:
        for n, r in enumerate(pool.imap_unordered(build_study, todo), 1):
            bad += r["status"] != "ok"
            if r["status"] != "ok":
                print(json.dumps(r), flush=True)
            if n % 50 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} studies, {bad} without a knee frame, {time.time() - t0:.0f}s", flush=True)
    return None if native_only else merge(out)


def merge(out):
    """rows/*.json -> knee512/index.csv and knee_frame.csv (sorted, atomic)."""
    out = Path(out); idx, kf = [], []
    for f in sorted((out / "rows").glob("*.json")):
        r = json.loads(f.read_text()); idx += r["index"]; kf += r["frames"]
    cols = ["study_uid", "knee", "side", "series_uid", "plane", "fluid", "slices", "file"]
    pd.DataFrame(idx, columns=cols).to_csv(out / "knee512/.tmp-index.csv", index=False)
    (out / "knee512/.tmp-index.csv").replace(out / "knee512/index.csv")
    pd.DataFrame(kf).to_csv(out / ".tmp-knee_frame.csv", index=False); (out / ".tmp-knee_frame.csv").replace(out / "knee_frame.csv")
    print(f"merged: {len(kf)} knees, {len(idx)} knee512 series -> {out}", flush=True)
    return len(kf), len(idx)


def jobs_from_folder(root, split="train", studies=None):
    """Kaggle-style folder -> [(study uid, [(series uid, [dicom paths], fluid flag)])] in study order."""
    root = Path(root)
    ids = list(studies) if studies else pd.read_csv(root / f"{split}.csv", dtype={"StudyInstanceUID": str}).StudyInstanceUID.tolist()
    meta = pd.read_csv(root / f"{split}_series.csv", dtype=str)
    by = {u: g for u, g in meta.groupby("StudyInstanceUID")}
    out = []
    for uid in ids:
        g = by.get(uid)
        if g is None:
            continue
        out.append((uid, [(s, [str(p) for p in sorted((root / f"{split}_series" / uid / s).glob("*.dcm"))], int(float(f or 0)))
                          for s, f in zip(g.SeriesInstanceUID, g.Fluid_Sensitive.fillna("0"))]))
    return out
