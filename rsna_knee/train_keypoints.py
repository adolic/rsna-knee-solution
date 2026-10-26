"""Train the keypoint localizer (one fold of the scanner-grouped 5-fold split of the 417 annotated knees).

Data: native224 volumes of the RSNA cache (rsna_knee.cache: whole field of view, 224 px, canonical patient frame) and
labels/keypoint-targets.csv: for every annotated knee and every series, the 5 landmarks (medial / lateral tibiofemoral
joint, intercondylar notch, patellofemoral joint, patella centre) projected to the series' 224 px grid (i, j) with a weight
(1 confident, 0.5 low confidence, 0 unobservable / outside the field of view / series that moved during the exam).
Input: three slices from the 5-30 / 30-70 / 70-95 % bands of the stack (random in training, band centres otherwise).
Mirror augmentation (p 0.5) flips the patient left-right axis (coronal / axial: image x; sagittal: slice order); landmark
identities do not change, only the knee-side label, so the model must find the medial side from anatomy.
Model (keypoint_model.py): EfficientNet-B0 + FPN to stride 4, plane-specific heatmap heads, soft-argmax, knee-side logit.
Loss: L1 of the soft-argmax coordinate (mm) + cross-entropy against a Gaussian heatmap (sigma 6 px) + 0.2 x side BCE.
AdamW lr 6e-4, wd 1e-4, cosine with a 1-epoch warm-up, bf16, 40 epochs; the last epoch is kept (last.pt).
usage: python -m rsna_knee.train_keypoints --fold 0 --cache DATA/cache/rsna --out RUNS/kp-f0   (folds 0-4: the five models)"""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .keypoint_model import GRID, Localizer
from .paths import LABELS

LANDMARKS = ["medial_tibiofemoral_joint", "lateral_tibiofemoral_joint", "intercondylar_notch", "patellofemoral_joint", "patella_center"]
PLANES = ["Sagittal", "Coronal", "Axial"]
BANDS = [(.05, .30), (.30, .70), (.70, .95)]


def series_table(targets):
    """One row per series with (5, 2) targets and (5,) weights."""
    rows = []
    for uid, g in targets.groupby("series_uid", sort=True):
        g = g.set_index("landmark").reindex(LANDMARKS)
        rows.append(dict(series_uid=uid, case_id=g.case_id.dropna().iloc[0], plane=g.plane.dropna().iloc[0],
                         knee_side=g.knee_side.dropna().iloc[0], keypoint_fold=int(g.keypoint_fold.dropna().iloc[0]),
                         n_slices=int(g.n_slices.dropna().iloc[0]), pixel_mm=float(g.pixel_mm.dropna().iloc[0]),
                         xy=np.nan_to_num(g[["i", "j"]].to_numpy(float), nan=0.), w=np.nan_to_num(g.weight.to_numpy(float), nan=0.)))
    return pd.DataFrame(rows)


def triplet_indices(n, rng=None):
    out = []
    for lo, hi in BANDS:
        a, b = lo * (n - 1), hi * (n - 1)
        x = rng.uniform(a, b) if rng is not None else (a + b) / 2
        out.append(int(np.clip(round(x), 0, n - 1)))
    return out


class Triplets(torch.utils.data.Dataset):
    def __init__(self, table, series_dir, train, seed=0, mirror_p=.5):
        self.t, self.dir = table.reset_index(drop=True), Path(series_dir)
        self.train, self.seed, self.mirror_p = train, seed, mirror_p
        self.epoch = 0

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        r = self.t.iloc[i]
        rng = np.random.default_rng((self.seed, self.epoch, i)) if self.train else None
        idx = triplet_indices(r.n_slices, rng)
        mirror = bool(self.train and rng.random() < self.mirror_p)
        if mirror and r.plane == "Sagittal":
            idx = [r.n_slices - 1 - k for k in idx][::-1]    # mirrored stack, same band layout
        vol = np.load(self.dir / f"{r.series_uid}.npy", mmap_mode="r")
        img = np.stack([np.asarray(vol[k], np.float32) / 255. for k in idx])
        xy, w = r.xy.copy(), r.w.copy()
        if mirror and r.plane != "Sagittal":
            img = img[:, :, ::-1].copy(); xy[:, 0] = 223. - xy[:, 0]
        left = float(r.knee_side == "L") if not mirror else float(r.knee_side == "R")
        if self.train:
            img, xy, w = self.augment(img, xy, w, rng)
        return dict(image=torch.from_numpy(np.ascontiguousarray(img)), xy=torch.from_numpy(xy).float(),
                    w=torch.from_numpy(w).float(), plane=PLANES.index(r.plane), left=torch.tensor(left),
                    pixel_mm=torch.tensor(r.pixel_mm, dtype=torch.float32))

    @staticmethod
    def augment(img, xy, w, rng):
        s = math.exp(rng.uniform(math.log(.85), math.log(1.18)))
        a = math.radians(rng.uniform(-12, 12))
        t = rng.uniform(-20, 20, 2)
        c = 111.5
        M = np.array([[s * math.cos(a), -s * math.sin(a), 0], [s * math.sin(a), s * math.cos(a), 0]])
        M[:, 2] = np.array([c, c]) + t - M[:, :2] @ np.array([c, c])
        img = np.stack([cv2.warpAffine(x, M, (224, 224), flags=cv2.INTER_LINEAR, borderValue=0.) for x in img])
        xy = xy @ M[:, :2].T + M[:, 2]
        w = w * ((xy >= 0) & (xy <= 223)).all(1)
        g = math.exp(rng.uniform(-.35, .35))                  # gamma, contrast, noise
        img = np.clip(img, 0, 1) ** g
        img = img * rng.uniform(.85, 1.15) + rng.uniform(-.08, .08)
        if rng.random() < .3:
            img = img + rng.normal(0, rng.uniform(.01, .04), img.shape)
        return np.clip(img, 0, 1).astype(np.float32), xy, w


def gaussian_targets(xy, sigma=1.5):
    c = (torch.arange(GRID, device=xy.device).float() + .5) * 4 - .5
    d2 = (c.view(1, 1, 1, GRID) - xy[..., 0, None, None]) ** 2 + (c.view(1, 1, GRID, 1) - xy[..., 1, None, None]) ** 2
    g = torch.exp(-d2 / (2 * (sigma * 4) ** 2))
    return g / g.flatten(2).sum(-1).clamp_min(1e-8)[..., None, None]


@torch.no_grad()
def evaluate(model, ds, device):
    """Weighted median in-plane error (mm) per landmark on band-centre triplets (+ mirrored copy averaged)."""
    model.eval(); errs = {lm: [] for lm in LANDMARKS}
    for i in range(len(ds)):
        b = ds[i]; img = b["image"][None].to(device); plane = torch.tensor([b["plane"]], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(img, plane)
        xy = model.coords(logits)[0].float().cpu().numpy()[0]
        d = np.hypot(*(xy - b["xy"].numpy()).T) * float(b["pixel_mm"])
        for j, lm in enumerate(LANDMARKS):
            if b["w"][j] > 0:
                errs[lm].append(d[j])
    model.train()
    return {lm: float(np.median(v)) if v else None for lm, v in errs.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True); ap.add_argument("--out", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4); ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--mirror", type=float, default=.5)
    ap.add_argument("--limit", type=int, default=0, help="smoke tests: first N training series only")
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); device = torch.device("cuda")
    series_dir = Path(args.cache) / "native224/series"
    targets = pd.read_csv(LABELS / "keypoint-targets.csv")
    table = series_table(targets)
    table = table[[ (series_dir / f"{u}.npy").exists() for u in table.series_uid ]]
    tr, va = table[table.keypoint_fold != args.fold], table[table.keypoint_fold == args.fold]
    if args.limit:
        tr, va = tr.head(args.limit), va.head(max(args.limit // 4, 8))
    (out / "config.json").write_text(json.dumps(dict(vars(args), train_series=len(tr), val_series=len(va),
                                                     backbone="efficientnet_b0.ra_in1k"), indent=2) + "\n")
    ds = Triplets(tr, series_dir, train=True, seed=args.seed, mirror_p=args.mirror); val = Triplets(va, series_dir, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=min(args.batch, len(ds)), shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
    model = Localizer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1., (s + 1) / len(dl)) * .5 * (1 + math.cos(math.pi * min(s / steps, 1))))
    history = []
    for epoch in range(1, args.epochs + 1):
        ds.epoch = epoch; model.train(); t0 = time.time(); agg = []
        for b in dl:
            img, xy, w = b["image"].to(device, non_blocking=True), b["xy"].to(device), b["w"].to(device)
            plane, left, mm = b["plane"].to(device), b["left"].to(device), b["pixel_mm"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, side = model(img, plane)
            pred, prob = model.coords(logits)
            l1 = ((pred - xy).abs().sum(-1) * mm[:, None] * w).sum() / w.sum().clamp_min(1)
            ce = (-(gaussian_targets(xy) * torch.log(prob.clamp_min(1e-9))).flatten(2).sum(-1) * w).sum() / w.sum().clamp_min(1)
            loss = l1 + ce + .2 * F.binary_cross_entropy_with_logits(side, left)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step(); sched.step()
            agg.append([loss.item(), l1.item()])
        row = dict(epoch=epoch, loss=float(np.mean(agg, 0)[0]), l1_mm=float(np.mean(agg, 0)[1]), seconds=round(time.time() - t0, 1))
        if len(va) and (epoch % 5 == 0 or epoch == args.epochs):
            row["val_median_mm"] = evaluate(model, val, device)
        history.append(row); print(json.dumps(row), flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=1) + "\n")
    torch.save(model.state_dict(), out / "last.pt")
    (out / "result.json").write_text(json.dumps(dict(status="completed", fold=args.fold, epochs=args.epochs, final=history[-1]), indent=1) + "\n")


if __name__ == "__main__":
    main()
