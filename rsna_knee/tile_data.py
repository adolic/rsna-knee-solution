"""Tile bags on knee512 caches (rsna_knee.cache): one sample = one knee's bag of 224 px tiles (70 mm at 0.3125 mm/px).

  * knee: the knee the study label belongs to ("one", or for a two-knee study the knee in labels/two-knee-sides.csv);
  * view: which series of the cache a model sees (all | sagittal | coronal | axial | corfluid = coronal fluid-sensitive);
  * training: per series `tiles_per_series` x fluid_mult (fluid-sensitive series) or x other_mult (others) tiles, centred
    on 3D points drawn from the keypoint-anchor mixture (tiles.py, bone-biased), the knee frame jittered per study
    (N(0, 2 mm) translation, N(0, 2 deg) rotation); series dropout per plane from the 3rd epoch (p 0.4, never emptying a
    plane); optional cap of series per knee (max_train_series);
  * validation / inference: the same mixture with `val_tiles_per_series` per series and a fixed seed (rng 0), the bag
    the inference kernel draws for draw key 0;
  * each tile: slices_per_tile adjacent slices (k-1, k, k+1) as channels, plane id, fluid flag and its knee-frame
    position (centre + corners, 15 numbers; unused by the final models, position "none").
Canvases are canonical (left knees mirrored at build time); the study-level mirror augmentation lives in the model."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .labels import side_for
from .tiles import TILE, SeriesGeo, frame, sample_tiles, to_knee

PLANES = ("Sagittal", "Coronal", "Axial")
VIEWS = {"all": lambda r: np.ones(len(r), bool), "sagittal": lambda r: r.plane == "Sagittal",
         "coronal": lambda r: r.plane == "Coronal", "axial": lambda r: r.plane == "Axial",
         "corfluid": lambda r: (r.plane == "Coronal") & (r.fluid == 1)}


def knees(cache, view="all"):
    """study -> (cache folder, knee frame row, index rows of that knee and view) for every study of the cache with a frame."""
    cache = Path(cache)
    idx = pd.read_csv(cache / "knee512/index.csv"); idx = idx[VIEWS[view](idx)]
    kf = pd.read_csv(cache / "knee_frame.csv"); kf = kf[kf.available == 1]
    by = {s: g for s, g in idx.groupby("study_uid")}
    out = {}
    for study, g in kf.groupby("study_uid"):
        row = g[g.knee == side_for(study, g.knee)]
        rows = by.get(study)
        if not len(row) or rows is None:
            continue
        rows = rows[rows.knee == row.iloc[0].knee]
        if len(rows):
            out[study] = (cache, row.iloc[0].to_dict(), rows.reset_index(drop=True))
    return out


def jittered(F, rng):
    """Knee frame with a small random rigid perturbation (keypoint-error tolerance)."""
    ang = np.deg2rad(rng.normal(0, 2, 3)); cx, cy, cz = np.cos(ang); sx, sy, sz = np.sin(ang)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return dict(F, O=F["O"] + rng.normal(0, 2, 3), R=F["R"] @ (Rx @ Ry @ Rz).T)


class TileBags(torch.utils.data.Dataset):
    def __init__(self, studies, knee_map, soft, weights, training, seed=0, tiles_per_series=8, val_tiles_per_series=96,
                 slices_per_tile=3, fluid_mult=1.0, other_mult=1.0, jitter=True, max_train_series=0):
        self.studies, self.knee_map = list(studies), knee_map
        self.soft, self.weights, self.training, self.seed = soft, weights, training, seed
        self.tiles_per_series, self.val_tiles_per_series, self.jitter = tiles_per_series, val_tiles_per_series, jitter
        self.slices_per_tile, self.fluid_mult, self.other_mult = slices_per_tile, fluid_mult, other_mult
        self.max_train_series = max_train_series
        self.epoch = 0

    def __len__(self):
        return len(self.studies)

    def _geos(self, study):
        if not hasattr(self, "_g"):
            self._g = {}
        if study not in self._g:
            cache, row, rows = self.knee_map[study]
            self._g[study] = (cache, frame(row), {r.file: SeriesGeo(cache / "knee512/geometry" / f"{r.file}.npz") for r in rows.itertuples()}, rows)
            if len(self._g) > 64:
                self._g.pop(next(iter(self._g)))
        return self._g[study]

    def __getitem__(self, i):
        study = self.studies[i]
        cache, F, geos, rows = self._geos(study)
        rng = np.random.default_rng((self.seed, self.epoch, i)) if self.training else np.random.default_rng(0)
        plane_of = dict(zip(rows.file, rows.plane)); fluid_of = dict(zip(rows.file, rows.fluid))
        use = dict(geos)
        if self.training and self.epoch >= 2:   # series dropout per plane, never emptying a plane
            for plane in PLANES:
                files = [f for f in use if plane_of[f] == plane]
                if len(files) > 1 and rng.random() < .4:
                    use.pop(files[int(rng.integers(len(files)))])
        if self.training and self.max_train_series and len(use) > self.max_train_series:
            for f in [list(use)[j] for j in rng.permutation(len(use))[:len(use) - self.max_train_series]]:
                use.pop(f)
        Fp = jittered(F, rng) if (self.training and self.jitter) else F
        n = self.tiles_per_series if self.training else self.val_tiles_per_series
        if self.fluid_mult == self.other_mult == 1.0:
            drawn = sample_tiles(F, use, n, rng)
        else:
            drawn = {}
            for fl, mult in ((1, self.fluid_mult), (0, self.other_mult)):
                part = {f: g for f, g in use.items() if int(fluid_of[f]) == fl}
                if part and round(n * mult) > 0:
                    drawn.update(sample_tiles(F, part, round(n * mult), rng))
        picks = [(f, t[0], t[1], t[2]) for f, ts in drawn.items() for t in ts]
        vols = {f: np.load(cache / "knee512/series" / f"{f}.npy", mmap_mode="r") for f in use}
        S = self.slices_per_tile; half = S // 2
        count = {}   # I/O: a slice holding >= 3 tiles is read once whole, otherwise only the tile rows
        for f, k, _, _ in picks:
            for d in range(-half, half + 1):
                kk = int(np.clip(k + d, 0, geos[f].n - 1)); count[(f, kk)] = count.get((f, kk), 0) + 1
        slices = {fk: np.array(vols[fk[0]][fk[1]]) for fk, c in count.items() if c >= 3}
        imgs = np.empty((len(picks), S, TILE, TILE), np.uint8)
        for m, (f, k, x0, y0) in enumerate(picks):
            for c, d in enumerate(range(-half, half + 1)):
                kk = int(np.clip(k + d, 0, geos[f].n - 1))
                imgs[m, c] = slices[(f, kk)][y0:y0 + TILE, x0:x0 + TILE] if (f, kk) in slices else vols[f][kk, y0:y0 + TILE, x0:x0 + TILE]
        pos = np.empty((len(picks), 15), np.float32)
        for m, (f, k, x0, y0) in enumerate(picks):
            G = geos[f]; h = (TILE - 1) / 2
            pts = G.lps(k, np.array([x0 + h, x0, x0 + TILE - 1, x0, x0 + TILE - 1]), np.array([y0 + h, y0, y0, y0 + TILE - 1, y0 + TILE - 1]))
            pos[m] = to_knee(Fp, pts).reshape(-1)
        return dict(images=torch.from_numpy(imgs), view=torch.tensor([PLANES.index(plane_of[p[0]]) for p in picks]),
                    fluid=torch.tensor([int(fluid_of[p[0]]) for p in picks]), pos=torch.from_numpy(pos),
                    labels=torch.from_numpy(self.soft[i]), weights=torch.from_numpy(self.weights[i]), study_uid=study)


def collate(samples):
    return dict(images=torch.cat([s["images"] for s in samples]), view=torch.cat([s["view"] for s in samples]),
                fluid=torch.cat([s["fluid"] for s in samples]), pos=torch.cat([s["pos"] for s in samples]),
                lengths=torch.tensor([len(s["view"]) for s in samples]),
                labels=torch.stack([s["labels"] for s in samples]), weights=torch.stack([s["weights"] for s in samples]),
                study_uid=[s["study_uid"] for s in samples])
