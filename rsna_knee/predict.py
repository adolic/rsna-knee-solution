"""Inference: RSNA knee MRI studies (raw DICOM) -> 12 probabilities per study, exactly as the submitted Kaggle kernel.

Per study (one worker process per GPU slot, models loaded once per worker):
  decode -> native224 -> keypoint localizer x5 -> knee frame per knee -> knee512 canvases (pipeline.py)
  -> per tile member: the canvases of its acquisition, one tile bag (draw key = member position), FP16 forward with the
     attention temperature(s) of the recipe (tau_mix: logits averaged over several taus; octant TTA: the bag re-scored
     without each knee-frame octant), probability
  -> per knee: sigmoid of the NaN-aware mean of plane means of member logits; study = mean of its knees
  -> canvas-crop specialists (meniscus, ACL, OA family) and the ResNet50 OA specialist, blended per target after the
     ensemble: (1 - w) * ensemble logit + w * standardised specialist logit.
Failure policy: a study that raises gets the training prior and is counted; more than 5 % of such studies abort the run.

Input (--input):
  * a Kaggle-style folder: test.csv (or --studies), test_series.csv and test_series/<study>/<series>/*.dcm
    (--split train reads train.csv / train_series.csv / train_series/ instead), or
  * one study folder <study>/<series>/*.dcm: the fluid-sensitive flag per series then comes from --series-csv
    (StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive) or, without it, from the DICOM SeriesDescription
    (fluid_from_description, a heuristic: the models were trained with the competition's flags).
usage: python -m rsna_knee.predict --models DATA/models --input DATA/raw/rsna --out predictions/ [--devices cuda:0,cuda:1]
"""
from __future__ import annotations
import os as __o
__o.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less allocator fragmentation, same outputs

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import MODELS, SUBMISSION, TARGETS

TARGETS = list(TARGETS)
UID = "StudyInstanceUID"
MAX_FALLBACK_SHARE = 0.05
_W = {}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"); temp.replace(path)


def cell_of(name):
    return name.rsplit("_f", 1)[0]


# ------------------------------------------------------------------------------------------------- tile scoring
def _cached_token_head(P, tm, imgs, view, fluid, device, chunk):
    """Backbone tokens of the bag computed once; returns head(rows) = the member's own forward (mixer, attention,
    readout) on a subset of the bag's rows, reusing the cached tokens."""
    torch = _W["torch"]
    images = torch.from_numpy(imgs).to(device); v = torch.from_numpy(view).to(device); fl = torch.from_numpy(fluid).to(device)
    pos = torch.zeros(len(imgs), 15, device=device)
    tok = tm.tokens(images, v, fl, torch.tensor([len(imgs)]), chunk, None, pos)

    def head(rows):
        r = torch.from_numpy(rows).to(device)
        tm.tokens = lambda *a, **k: tok[r]   # the forward's own token call returns the cached tokens of these rows
        try:
            return tm(dict(images=images[:0], view=v[r], fluid=fl[r], lengths=torch.tensor([len(rows)]), pos=pos[r]),
                      chunk=chunk).float().cpu().numpy()[0]
        finally:
            del tm.tokens
    return head


def _octants(P, F, picks, geos):
    """Knee-frame octant (signs of medial, anterior, superior of the tile centre) of every tile of the bag."""
    h = (P.gt.TILE - 1) / 2
    q = np.array([P.gt.to_knee(F, geos[f].lps(k, x0 + h, y0 + h)) for f, k, x0, y0 in picks])
    return (q[:, 0] > 0).astype(int) * 4 + (q[:, 1] > 0).astype(int) * 2 + (q[:, 2] > 0).astype(int)


def mix_tile_logits(P, tm, imgs, view, fluid, device, taus, chunk, F, picks, geos, weight):
    """For every attention tau in `taus` (tensors shaped as the member's tau): the member's forward on the whole bag and,
    with octant weight > 0, on each leave-one-octant-out bag; returns the mean over the taus of
    (1 - weight) * whole + weight * mean(leave-one-octant-out logits)."""
    torch = _W["torch"]
    oc = _octants(P, F, picks, geos) if weight else None
    res = []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        head = _cached_token_head(P, tm, imgs, view, fluid, device, chunk)
        for tau in taus:
            tm.att_tau = tau
            whole = head(np.arange(len(imgs)))
            outs = [head(np.flatnonzero(oc != c)) for c in range(8) if 0 < int((oc != c).sum()) < len(imgs)] if weight else []
            res.append(whole if not outs else (1 - weight) * whole + weight * np.mean(outs, 0))
    return np.mean(res, 0)


def octant_tile_logits(P, tm, imgs, view, fluid, device, tau, chunk, F, picks, geos, weight):
    """One tau: (1 - weight) * whole bag + weight * mean of the leave-one-octant-out bags (empty / full octants skipped)."""
    torch = _W["torch"]
    oc = _octants(P, F, picks, geos)
    tm.att_tau = tau
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        head = _cached_token_head(P, tm, imgs, view, fluid, device, chunk)
        whole = head(np.arange(len(imgs)))
        outs = [head(np.flatnonzero(oc != c)) for c in range(8) if 0 < int((oc != c).sum()) < len(imgs)]
    return whole if not outs else (1 - weight) * whole + weight * np.mean(outs, 0)


def nanmean0(arrs):
    """Mean over the first axis ignoring NaN; NaN where every entry is NaN."""
    a = np.stack(arrs); n = np.isfinite(a).sum(0)
    return np.where(n > 0, np.nansum(a, 0) / np.maximum(n, 1), np.nan)


# ----------------------------------------------------------------------------------------------- worker side
def worker_init(models_dir, devices, counter, lock):
    """Load every model once per worker. A failure is stored (a raising initializer makes the pool respawn workers
    forever) and every study of this worker then reports it as a counted fallback."""
    try:
        import importlib
        import torch
        torch.set_num_threads(1)
        import cv2
        cv2.setNumThreads(1)
        from . import pipeline, keypoint_model, tile_model, crop_model, specialists, oa_specialist
        root = Path(models_dir)
        with lock:
            wid = counter.value; counter.value += 1
        device = torch.device(devices[wid % len(devices)])
        man = json.loads((root / "manifest.json").read_text()); recipe = man["recipe"]
        speed, inf = recipe.get("speed", {}), recipe.get("inference", {})

        def fp16(m):
            if speed.get("half_weights"):    # FP16 weights (compute is FP16 autocast either way): half the GPU memory
                m.half()
            if speed.get("channels_last"):
                m.to(memory_format=torch.channels_last)
            return m
        kps = []
        for f in man["keypoint_models"]:
            m = keypoint_model.Localizer(pretrained=False).to(device)
            m.load_state_dict(torch.load(root / f, map_location=device)); kps.append(m.eval())
        members, packs = [], {}
        for mem in man["members"]:
            if mem["checkpoint"] not in packs:   # one pack file per cell: FP16 weights per member key
                packs.clear(); packs[mem["checkpoint"]] = torch.load(root / mem["checkpoint"], map_location="cpu", weights_only=False)
            ck = packs[mem["checkpoint"]][mem["checkpoint_key"]]
            tm = tile_model.TileModel(ck["cfg"], pretrained=False); tm.load_state_dict(ck["model"])
            tm = fp16(tm.to(device).eval())
            override = inf.get("cell_tau", {}).get(mem["cell"], inf.get("attention_tau"))
            values = mem.get("attention_tau", [1.0] * 12) if override is None else ([float(override)] * 12 if np.isscalar(override) else list(override))
            assert len(values) == 12
            members.append((mem, tm, torch.tensor(values, device=device, dtype=torch.float32).view(1, 12, 1)))
        packs.clear()
        oa = None
        if man.get("oa"):   # ResNet50 OA specialist (5 folds) on keypoint-centred OA crops
            om = json.loads((root / man["oa"] / "oa-manifest.json").read_text()); oms = []
            for f in om["checkpoints"]:
                ck = torch.load(root / man["oa"] / f, map_location="cpu", weights_only=False)
                m = crop_model.CropModel(ck["cfg"], pretrained=False); m.load_state_dict(ck["model"]); oms.append(fp16(m.to(device).eval()))
            oa = (oms, om)
        spec = None
        if man.get("specialists"):   # canvas-crop specialists (meniscus, ACL, OA family), 5 folds each
            sman = json.loads((root / man["specialists"] / "spec-manifest.json").read_text()); loaded = []
            for sp in sman["specialists"]:
                ms = []
                for f in sp["checkpoints"]:
                    ck = torch.load(root / man["specialists"] / f, map_location="cpu", weights_only=False)
                    m = crop_model.CropModel(ck["cfg"], pretrained=False); m.load_state_dict(ck["model"]); ms.append(fp16(m.to(device).eval()))
                loaded.append((sp, ms))
            spec = (loaded, sman)
        _W.update(P=pipeline, SM=specialists, OM=oa_specialist, torch=torch, device=device, kps=kps, members=members,
                  man=man, inf=inf, wid=wid, oa=oa, spec=spec)
    except Exception as error:
        _W.update(init_error=f"{type(error).__name__}: {error}", init_trace=traceback.format_exc()[-1500:], wid=-1)


def predict_study(job):
    """job: (row index, uid, [(series uid, [paths], fluid)]) -> (index, probabilities | None, receipt)."""
    index, uid, series = job
    if "init_error" in _W:
        return index, None, dict(sample_index=index, study=uid, status="fallback", error="worker init: " + _W["init_error"],
                                 trace=_W["init_trace"], seconds=0.)
    P, torch, device, man, inf = _W["P"], _W["torch"], _W["device"], _W["man"], _W["inf"]
    t0 = time.monotonic()
    receipt = dict(sample_index=index, study=uid, worker=_W["wid"], device=str(device))
    try:
        oriented, native, rec = P.prepare_series([(u, [Path(p) for p in ps], f) for u, ps, f in series])
        receipt.update(rec)
        if not native:
            raise ValueError("no usable native224 series")
        t1 = time.monotonic()
        frames = P.knee_frames(native, _W["kps"], device)
        receipt.update(knees=[k for k, *_ in frames], frame_fallback=int(sum(fb for *_, fb in frames)),
                       keypoint_seconds=time.monotonic() - t1)
        per_knee, tiles, used = [], [], []
        t2 = time.monotonic()
        spec_knees, spec_wins = {}, []
        own = bool(inf.get("own_draws"))
        for knee, row, _ in frames:
            canv = P.knee_canvases(oriented, row)
            if not canv:
                continue
            if _W["spec"] is not None:   # canvas-crop specialists on this knee's canvases (errors recorded, never fatal)
                ts = time.monotonic()
                try:
                    SM = _W["SM"]; loaded, sman = _W["spec"]
                    for sp, ms in loaded:
                        for r_ in sp["rows"]:
                            if r_.get("parts"):   # several crop configs in one bag (OA family model)
                                bag = SM.row_bag_parts(canv, row, r_["parts"], sp["crop_configs"], sp["size"])
                            else:
                                bag = SM.row_bag(canv, row, sp["crop_config"], r_["views"], r_["view_ids"], sp["size"])
                            if bag is None:
                                continue
                            spec_knees.setdefault(r_["target"], []).append(SM.row_logit(ms, bag, device, head=int(r_.get("head", 0)))[0])
                            spec_wins.append(int(len(bag[1])))
                except Exception as error:
                    receipt.update(spec_error=f"{type(error).__name__}: {error}", spec_trace=traceback.format_exc()[-800:])
                receipt["spec_seconds"] = receipt.get("spec_seconds", 0.) + time.monotonic() - ts
            F = P.gt.frame(row)
            groups = {}
            for position, (mem, tm, tau) in enumerate(_W["members"]):
                acq = mem.get("acquisition")
                sub = {u: c for u, c in canv.items()   # acquisition: None (all series), [plane] or [plane, fluid flag]
                       if acq is None or (c[2] == acq[0] and (len(acq) < 2 or acq[1] is None or int(c[3]) == int(acq[1])))}
                if not sub:
                    continue
                draws = []
                for key in ([position] if own else mem.get("draw_keys", [0])):
                    key = 0 if key == 0 else (int(key), int(index))
                    imgs, view, fluid, picks_ = P.tile_bag(F, sub, mem["tiles_per_series"], key, mem.get("fluid_mult", 1.0),
                                                           mem.get("other_mult", 1.0), mem.get("slices_per_tile", 1))
                    tiles.append(len(imgs))
                    chunk_ = int(inf.get("cell_chunk", {}).get(mem["cell"], 128))
                    w_oct = inf.get("cell_octant", {}).get(mem["cell"])
                    tmix = inf.get("tau_mix")
                    if tmix:
                        logits = mix_tile_logits(P, tm, imgs, view, fluid, device, [torch.full_like(tau, float(t)) for t in tmix], chunk_,
                                                 F, picks_, {u: c[1] for u, c in sub.items()}, float(w_oct or 0.0))
                    elif w_oct:
                        logits = octant_tile_logits(P, tm, imgs, view, fluid, device, tau, chunk_, F, picks_,
                                                    {u: c[1] for u, c in sub.items()}, float(w_oct))
                    else:
                        logits = P.tile_logits(tm, imgs, view, fluid, device, tau, chunk=chunk_)
                    draws.append(1 / (1 + np.exp(-logits.astype(np.float64))))
                p = np.clip(np.mean(draws, 0), 1e-7, 1 - 1e-7)
                plane = mem.get("plane") or (acq[0] if acq else "all")   # members -> plane means -> group mean
                logit = np.log(p / (1 - p))
                groups.setdefault(mem.get("group", "planes"), {}).setdefault(plane, []).append(logit)
                used.append(f"{knee}:{mem['name']}")
                if os.environ.get("RNK_RECEIPT_LOGITS") == "1":   # parity checks: every member's logits in the receipt
                    receipt.setdefault("member_debug", {})[f"{knee}:{mem['name']}"] = dict(plane=plane, logit=logit.tolist(), tau=tau.flatten().tolist())
            if not groups:
                continue
            gp = {}
            for g, by in groups.items():
                v = nanmean0([nanmean0(L) for L in by.values()])
                if not np.isfinite(v).all():
                    raise ValueError(f"no member feeds targets {np.flatnonzero(~np.isfinite(v)).tolist()} for knee {knee}")
                gp[g] = 1 / (1 + np.exp(-v))
            w = {g: man["group_weights"][g] for g in gp}
            per_knee.append(sum(w[g] * gp[g] for g in gp) / sum(w.values()))
        receipt.update(knees_predicted=len(per_knee), tiles=tiles, members_used=used, model_seconds=time.monotonic() - t2,
                       spec_logits={t: float(np.mean(v)) for t, v in spec_knees.items()} if _W["spec"] is not None else None,
                       spec_windows=spec_wins)
        if _W["oa"] is not None:
            t3 = time.monotonic()
            try:
                OM = _W["OM"]; oms, om = _W["oa"]
                geoms = {u: P.Geo224(g) for u, (_, g, _) in native.items()}
                centres = {u: float((g.origin[g.n // 2] + 111.5 * g.u + 111.5 * g.v)[0]) for u, g in geoms.items()}
                groups_k = P.knee_groups(centres)
                vals, wins = [], []
                for knee, row, _ in frames:
                    ser = {u: oriented[u] for u in groups_k.get(knee, []) if u in oriented}
                    bag = OM.knee_bag(ser, row, om["crop_config"], om["size"], om["margin_mm"])
                    if bag is None:
                        continue
                    vals.append(OM.pf_logit(oms, bag, device)[1]); wins.append(int(len(bag[1])))   # medial, lateral, PF OA logits
                mean3 = np.mean(vals, 0) if vals else None
                receipt.update(oa_logits=[float(x) for x in mean3] if vals else None, oa_windows=wins)
            except Exception as error:
                receipt.update(oa_logits=None, oa_error=f"{type(error).__name__}: {error}", oa_trace=traceback.format_exc()[-800:])
            receipt.update(oa_seconds=time.monotonic() - t3)
        if not per_knee:
            raise ValueError("no knee canvas")
        prob = np.mean(per_knee, 0)
        if prob.shape != (12,) or not np.isfinite(prob).all():
            raise ValueError("invalid model output")
        receipt.update(status="ok", seconds=time.monotonic() - t0)
        return index, prob, receipt
    except Exception as error:   # counted fallback (prior); too many abort the run in main()
        receipt.update(status="fallback", error=f"{type(error).__name__}: {error}", trace=traceback.format_exc()[-1500:],
                       seconds=time.monotonic() - t0)
        return index, None, receipt


# --------------------------------------------------------------------------------------------------- inputs
FLUID_WORDS = ("FS", "FATSAT", "FAT SAT", "SPIR", "SPAIR", "STIR", "TIRM", "DIXON W", "WATER", "T2", "PD FS", "PDFS", "DESS", "MEDIC", "MERGE")


def fluid_from_description(desc):
    """Heuristic fluid-sensitive flag from a SeriesDescription (only used without a series CSV): fat-suppressed or T2-type
    sequences are fluid sensitive; plain T1 / PD without fat suppression are not."""
    d = f" {str(desc).upper().replace('_', ' ').replace('-', ' ')} "
    if " T1 " in d and not any(w in d for w in ("FS", "FAT", "SPIR", "SPAIR", "STIR")):
        return 0
    return int(any(f" {w} " in d or w in d.split() for w in FLUID_WORDS) or "FS" in d.replace(" ", "")[-4:])


def jobs_from_folder(root, split, studies=None):
    """Kaggle-style folder -> (study ids in order, jobs)."""
    root = Path(root)
    ids = list(studies) if studies else pd.read_csv(root / f"{split}.csv", dtype={UID: str})[UID].tolist()
    meta = pd.read_csv(root / f"{split}_series.csv", dtype=str)
    images = root / f"{split}_series"
    jobs = []
    for i, uid in enumerate(ids):
        rows = meta[meta[UID] == uid]
        jobs.append((i, uid, [(u, [str(p) for p in sorted((images / uid / u).glob("*.dcm"))], int(f))
                              for u, f in zip(rows.SeriesInstanceUID, rows.Fluid_Sensitive.fillna("0").astype(float).astype(int))]))
    return ids, jobs


def jobs_from_study(study_dir, series_csv=None):
    """One study folder <study>/<series>/*.dcm -> ([uid], [job]); fluid flags from series_csv or SeriesDescription."""
    import pydicom
    study_dir = Path(study_dir); uid = study_dir.name
    flags = {}
    if series_csv:
        m = pd.read_csv(series_csv, dtype=str); flags = dict(zip(m.SeriesInstanceUID, m.Fluid_Sensitive.astype(float).astype(int)))
    series = []
    for d in sorted(p for p in study_dir.iterdir() if p.is_dir()):
        files = sorted(d.glob("*.dcm")) or sorted(f for f in d.iterdir() if f.is_file())
        if not files:
            continue
        if d.name in flags:
            fl = flags[d.name]
        else:
            fl = fluid_from_description(getattr(pydicom.dcmread(files[0], stop_before_pixels=True), "SeriesDescription", ""))
            print(f"series {d.name}: fluid-sensitive flag {fl} from SeriesDescription (heuristic)", flush=True)
        series.append((d.name, [str(f) for f in files], fl))
    return [uid], [(0, uid, series)]


# ----------------------------------------------------------------------------------------------- post blends
def blend(predictions, logit_r, weights, have, stats_fn, min_test):
    """Per target: (1 - w) * ensemble logit + w * specialist logit standardised to the ensemble logit (test-set statistics
    when at least min_test studies have both, else the model pack's offline out-of-fold statistics)."""
    used = {}
    for c, (name, w) in enumerate(weights.items()):
        j = TARGETS.index(name)
        k = np.log(np.clip(predictions[:, j], 1e-7, 1 - 1e-7) / (1 - np.clip(predictions[:, j], 1e-7, 1 - 1e-7)))
        r = logit_r[:, c]; h = have & np.isfinite(r)
        if h.sum() >= min_test:
            mr, sr, mk, sk = r[h].mean(), r[h].std(), k[h].mean(), k[h].std(); src = "test set"
        else:
            st = stats_fn(name); mr, sr, mk, sk = st["r_mean"], st["r_std"], st["k_mean"], st["k_std"]; src = "offline OOF"
        z = (r - mr) / max(sr, 1e-6) * sk + mk
        predictions[:, j] = np.where(h, 1 / (1 + np.exp(-((1 - w) * k + w * z))), predictions[:, j])
        used[name] = dict(weight=float(w), stats_source=src, stats=[float(mr), float(sr), float(mk), float(sk)], blended=int(h.sum()))
    return used


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=str(MODELS), help="model pack folder (manifest.json)")
    ap.add_argument("--input", required=True, help="Kaggle-style folder or one study folder")
    ap.add_argument("--split", default="test", help="Kaggle-style folder: read <split>.csv / <split>_series.csv / <split>_series/")
    ap.add_argument("--studies", nargs="*", help="only these study ids (Kaggle-style folder)")
    ap.add_argument("--series-csv", default=None, help="one study folder: per-series Fluid_Sensitive flags")
    ap.add_argument("--out", default=str(SUBMISSION), help="output folder: predictions.csv, receipts.jsonl, verification.json (default: SETTINGS.json SUBMISSION_DIR)")
    ap.add_argument("--devices", default=None, help="comma list, default every visible CUDA device")
    ap.add_argument("--workers", type=int, default=None, help="worker processes (default: 2 per GPU)")
    a = ap.parse_args(argv)
    started = time.monotonic()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    output, receipts_path, verification_path = out / "predictions.csv", out / "receipts.jsonl", out / "verification.json"
    models = Path(a.models); man = json.loads((models / "manifest.json").read_text())
    if man["target_columns"] != TARGETS or not man.get("members"):
        raise ValueError("Unexpected model manifest")
    for rel, digest in man.get("files", {}).items():   # every checkpoint as released
        if sha256(models / rel) != digest:
            raise ValueError(f"changed model file: {rel}")
    inp = Path(a.input)
    if (inp / f"{a.split}_series.csv").exists():
        ids, jobs = jobs_from_folder(inp, a.split, a.studies)
    else:
        ids, jobs = jobs_from_study(inp, a.series_csv)
    import torch
    devices = a.devices.split(",") if a.devices else [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if not devices:
        raise RuntimeError("No CUDA device (FP16 inference needs a GPU)")
    recipe = man["recipe"]
    workers = a.workers or max(1, min(max(2, min(4, os.cpu_count() or 2)), int(recipe.get("speed", {}).get("workers_per_gpu", 2)) * len(devices)))
    ctx = mp.get_context("spawn")
    counter, lock = ctx.Value("i", 0), ctx.Lock()
    prior = np.asarray(man["prior"], np.float64)
    predictions = np.full((len(ids), 12), np.nan); fell_back = np.zeros(len(ids), bool)
    oa_logit = np.full((len(ids), 3), np.nan)
    spec_cfg = dict(recipe.get("spec", {})); spec_targets = list(spec_cfg.get("weights", {}))
    spec_logit = np.full((len(ids), max(len(spec_targets), 1)), np.nan)
    stats = dict(fallback=0, two_knee=0, frame_fallback=0, unreadable_slices=0, tiles=0, study_seconds=0.)
    with ctx.Pool(workers, initializer=worker_init, initargs=(str(models), devices, counter, lock)) as pool, receipts_path.open("w") as rh:
        for done, (i, prob, rec) in enumerate(pool.imap_unordered(predict_study, jobs), 1):
            if prob is None:
                stats["fallback"] += 1; prob = prior; fell_back[i] = True
            else:
                if rec.get("oa_logits") is not None:
                    oa_logit[i] = rec["oa_logits"]
                for c_, t_ in enumerate(spec_targets):
                    spec_logit[i, c_] = (rec.get("spec_logits") or {}).get(t_, np.nan)
            predictions[i] = prob
            stats["two_knee"] += int(len(rec.get("knees", [])) > 1); stats["frame_fallback"] += rec.get("frame_fallback", 0)
            stats["unreadable_slices"] += rec.get("unreadable_slices", 0); stats["tiles"] += sum(rec.get("tiles", []))
            stats["study_seconds"] += rec["seconds"]
            rh.write(json.dumps(rec, default=str) + "\n")
            if done % 25 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} studies, fallback {stats['fallback']}, {time.monotonic() - started:.0f}s", flush=True)
    if stats["fallback"] > MAX_FALLBACK_SHARE * max(len(ids), 20):
        raise RuntimeError(f"Too many fallback studies: {stats['fallback']} of {len(ids)}")
    if man.get("oa"):   # ResNet50 OA specialist: medial / lateral / PF OA logits -> recipe weights per target
        cfg = recipe["oa"]; col = {"Medial OA": 0, "Lateral OA": 1, "PF OA": 2}
        stats["oa_targets"] = blend(predictions, oa_logit[:, [col[t] for t in cfg["weights"]]], cfg["weights"], ~fell_back,
                                    lambda t: cfg["offline_stats"][t], int(cfg.get("min_test_studies", 100)))
    if spec_targets:   # canvas-crop specialists, after the OA blend
        stats["spec_targets"] = blend(predictions, spec_logit, spec_cfg["weights"], ~fell_back,
                                      lambda t: spec_cfg["offline_stats"][t], int(spec_cfg.get("min_test_studies", 100)))
    frame = pd.DataFrame(predictions, columns=TARGETS); frame.insert(0, UID, ids)
    v = frame[TARGETS].to_numpy(np.float64)
    if not np.isfinite(v).all() or not ((v >= 0) & (v <= 1)).all():
        raise ValueError("invalid probabilities")
    frame.to_csv(output, index=False, float_format="%.17g")
    import cv2, pydicom, timm
    atomic_json(verification_path, dict(status="passed", studies=len(ids), failures=stats["fallback"], stats=stats, workers=workers,
                                        devices=devices, models=str(models), recipe=recipe, output_sha256=sha256(output),
                                        elapsed_seconds=time.monotonic() - started,
                                        versions=dict(torch=torch.__version__, timm=timm.__version__, numpy=np.__version__,
                                                      opencv=cv2.__version__, pydicom=pydicom.__version__)))
    print(f"wrote {output} ({len(ids)} studies, {stats['fallback']} fallback, {time.monotonic() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
