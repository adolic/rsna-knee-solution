"""Crop-cache bags (rsna_knee.crop_cache): one sample = one knee's bag of 3-slice windows over the chosen views.

Training bag:
  * series dropout from the 3rd epoch: per plane with p 0.4, one of the plane's series (if it has > 1) is removed from every
    view of that plane (never emptying a view);
  * per view `budget` windows: a covering series uniformly, a position from N(0, sigma) truncated to the slab, snapped to the
    nearest stored slice; window = stored slices j-1, j, j+1 (j-step .. j+step; step 0 = the centre slice three times);
  * orientation "mirror": random patient left-right mirror per study (p 0.5): coronal / axial windows flipped horizontally,
    sagittal windows in reversed slice (channel) order.
Validation bag: every inference window of every view and series (the crop config's inference grid; with infer_all every
stored slice inside the slab +- 2 mm), no augmentation. Per window also its fluid flag and its depth (signed offset of the
centre slice from the view centre, mm / knee scale)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .labels import side_for

PLANES = ("Sagittal", "Coronal", "Axial")


def load_index(cache):
    idx = pd.read_csv(Path(cache) / "index.csv")
    idx["rel_mm"] = idx.rel_mm.map(lambda s: np.asarray(json.loads(s), np.float32))
    idx["infer"] = idx.infer.map(lambda s: np.asarray(json.loads(s), np.int64))
    return idx


def knee_entries(idx):
    """study -> index rows of the knee its study label describes (two-knee studies: labels/two-knee-sides.csv)."""
    return {study: g[g.knee == side_for(study, g.knee)].reset_index(drop=True) for study, g in idx.groupby("study_uid")}


class Bags(torch.utils.data.Dataset):
    def __init__(self, cache, cfg, studies, entries, soft, weights, training, seed=0, orientation="mirror",
                 use_views=None, budgets=None, nslices=3, step=1, infer_all=False):
        self.cache, self.cfg, self.studies, self.entries = Path(cache), cfg, list(studies), entries
        self.soft, self.weights, self.training, self.seed = soft, weights, training, seed
        self.views = list(cfg["views"])                      # global view ids (embedding index) stay fixed
        self.use_views = list(use_views or self.views)
        self.budgets = budgets or {v: cfg["views"][v]["train_budget"] for v in self.views}
        self.nslices, self.step, self.infer_all, self.orientation = nslices, step, infer_all, orientation
        self.epoch = 0

    def __len__(self):
        return len(self.studies)

    def _layout(self, study):
        if not hasattr(self, "_cache"):
            self._cache = {}
        if study not in self._cache:
            e = self.entries[study]; e = e[e.view.isin(self.use_views)]
            rows = list(e.itertuples())
            by_view = {v: [r for r in rows if r.view == v] for v in self.use_views}
            planes = {}
            for r in rows:
                planes.setdefault(r.plane, {}).setdefault(r.view, set()).add(r.series_uid)
            self._cache[study] = (rows, by_view, planes)
        return self._cache[study]

    def windows(self, rng, study):
        picks = []
        rows_all, by_view, planes = self._layout(study)
        if not self.training:
            for r in rows_all:
                if self.infer_all:
                    picks += [(r, int(j)) for j in np.flatnonzero((r.rel_mm >= r.slab_lo - 2) & (r.rel_mm <= r.slab_hi + 2))]
                else:
                    picks += [(r, int(j)) for j in r.infer]
            return picks
        drop = set()
        if self.epoch >= 2:
            for plane in PLANES:
                views = planes.get(plane, {})
                ser = sorted(set().union(*views.values())) if views else []
                if len(ser) > 1 and rng.random() < .4:
                    cand = ser[int(rng.integers(len(ser)))]
                    if all(len(s - {cand}) > 0 for s in views.values()):
                        drop.add(cand)
        for view in self.use_views:
            rows = [r for r in by_view[view] if r.series_uid not in drop]
            if not rows:
                continue
            seen = set()
            for _ in range(self.budgets[view]):
                for _try in range(20):
                    r = rows[int(rng.integers(len(rows)))]
                    x = rng.normal(0, r.sigma_mm)
                    if not r.slab_lo <= x <= r.slab_hi:
                        continue
                    ok = np.flatnonzero((r.rel_mm >= r.slab_lo - 2) & (r.rel_mm <= r.slab_hi + 2))
                    j = int(ok[np.argmin(np.abs(r.rel_mm[ok] - x))])
                    if (r.Index, j) in seen:
                        continue
                    seen.add((r.Index, j)); picks.append((r, j)); break
        return picks

    def __getitem__(self, i):
        study = self.studies[i]
        e = self.entries[study]
        rng = np.random.default_rng((self.seed, self.epoch, i)) if self.training else np.random.default_rng(0)
        picks = self.windows(rng, study)
        vol = np.load(self.cache / "studies" / f"{study}.npy", mmap_mode="r")
        mirror = (str(e.side.iloc[0]) == "L") if self.orientation == "canonical" else (self.training and rng.random() < .5)
        half = self.nslices // 2
        offsets = [self.step * d for d in range(-half, half + 1)]
        imgs = np.empty((len(picks), self.nslices, *vol.shape[1:]), np.uint8)
        view_id = np.empty(len(picks), np.int64); fluid = np.empty(len(picks), np.int64); depth = np.empty(len(picks), np.float32)
        for n, (r, j) in enumerate(picks):
            sign = {"Sagittal": 1. if r.side == "R" else -1., "Coronal": -1., "Axial": 1.}[r.plane]
            depth[n] = sign * float(r.rel_mm[j]) / float(r.knee_scale)
            w = vol[[r.start + int(np.clip(j + d, 0, r.count - 1)) for d in offsets]]
            if mirror:
                w = w[::-1] if r.plane == "Sagittal" else w[:, :, ::-1]
            imgs[n] = w; view_id[n] = self.views.index(r.view); fluid[n] = r.fluid
        return dict(images=torch.from_numpy(imgs), view=torch.from_numpy(view_id), fluid=torch.from_numpy(fluid),
                    depth=torch.from_numpy(depth), labels=torch.from_numpy(self.soft[i]), weights=torch.from_numpy(self.weights[i]),
                    study_uid=study)


def collate(samples):
    return dict(images=torch.cat([s["images"] for s in samples]), view=torch.cat([s["view"] for s in samples]),
                fluid=torch.cat([s["fluid"] for s in samples]), depth=torch.cat([s["depth"] for s in samples]),
                lengths=torch.tensor([len(s["view"]) for s in samples]),
                labels=torch.stack([s["labels"] for s in samples]), weights=torch.stack([s["weights"] for s in samples]),
                study_uid=[s["study_uid"] for s in samples])
