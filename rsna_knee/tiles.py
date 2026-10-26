"""Tile sampling on the knee-centred 512 canvases (0.3125 mm/px; see pipeline.py step 5).

A training tile = one stored slice, a TILE x TILE square (224 px = 70 mm) at a sampled centre. Centres are drawn in
3D, in the knee frame of the stage-1 keypoints (knee_frame.csv), so every plane samples the same anatomy:
  frame: origin O = tibiofemoral midpoint; m = medial (lateral TF -> medial TF), s = superior, a = anterior (towards
  the patella); k = knee scale (|medial TF - lateral TF| / 45 mm). Offsets and spreads are in mm x k.
  anchors (base keypoint, offset (m, a, s), spread sigma (m, a, s), weight) target the RSNA findings, with extra weight
  on the bones (femoral condyles, tibial plateaus, patella) as asked; "background" is uniform over the knee box.
A 3D point is projected into a series (nearest slice along its normal; points more than half a slice spacing beyond
the first / last slice are redrawn), the tile centre is the projection, clamped so the tile stays on the canvas.
Every tile carries its knee-frame coordinates (centre and four corners, divided by k), used as the token position.
"""
import numpy as np
import pandas as pd

MM, CANVAS, TILE = 0.3125, 512, 224
KP = {"MTF": "medial_tibiofemoral_joint", "LTF": "lateral_tibiofemoral_joint", "N": "intercondylar_notch",
      "PFJ": "patellofemoral_joint", "P": "patella_center"}
# name: (base, offset (m, a, s) mm, sigma (m, a, s) mm, weight)
ANCHORS = {
    "medial compartment": ("MTF", (0, 0, 0), (8, 14, 8), .09),
    "lateral compartment": ("LTF", (0, 0, 0), (8, 14, 8), .09),
    "notch / cruciates": ("N", (0, -8, -6), (6, 12, 12), .07),
    "patellofemoral": ("PFJ", (0, 0, 0), (14, 8, 10), .08),
    "medial femoral condyle": ("MTF", (0, -5, 18), (10, 16, 10), .08),
    "lateral femoral condyle": ("LTF", (0, -5, 18), (10, 16, 10), .08),
    "medial tibial plateau": ("MTF", (0, -3, -16), (10, 14, 9), .08),
    "lateral tibial plateau": ("LTF", (0, -3, -16), (10, 14, 9), .08),
    "patella": ("P", (0, 0, 0), (12, 6, 12), .06),
    "suprapatellar pouch": ("P", (0, -10, 35), (18, 8, 12), .06),
    "posteromedial (Baker's)": ("MTF", (-5, -35, 0), (12, 10, 18), .06),
    "medial collateral": ("MTF", (18, 0, 0), (5, 12, 18), .05),
    "background": (None, None, None, .12),
}
BONE = ("medial femoral condyle", "lateral femoral condyle", "medial tibial plateau", "lateral tibial plateau", "patella")


def frame(row):
    """Knee frame (dict) from a knee_frame.csv row."""
    g = lambda p: np.array([row[f"{p}_x"], row[f"{p}_y"], row[f"{p}_z"]], float)
    return dict(O=g("O"), R=np.stack([g("m"), g("a"), g("s")]), k=float(row["scale_k"]),
                kp={n: g(c) for n, c in KP.items()})


def to_knee(F, X):
    """LPS points [..., 3] -> knee-frame coordinates (m, a, s) in units of k (mm / k)."""
    return ((X - F["O"]) @ F["R"].T) / F["k"]


def draw_points(F, n, rng, names=None, weights=None):
    """n 3D points (LPS) and their anchor names, drawn from the anchor mixture."""
    names = names or list(ANCHORS)
    w = np.array(weights if weights is not None else [ANCHORS[a][3] for a in names], float); w /= w.sum()
    pick = rng.choice(len(names), n, p=w)
    pts = np.empty((n, 3))
    for i, j in enumerate(pick):
        base, off, sig, _ = ANCHORS[names[j]]
        if base is None:      # background: uniform in the knee box (m +-50, a +-50, s +-60 mm)
            local = rng.uniform([-50, -50, -60], [50, 50, 60])
            pts[i] = F["O"] + (local * F["k"]) @ F["R"]
        else:
            local = (np.array(off) + rng.normal(0, sig)) * F["k"]
            pts[i] = F["kp"][base] + local @ F["R"]
    return pts, [names[j] for j in pick]


class SeriesGeo:
    def __init__(self, path):
        z = np.load(path)
        self.origin, self.u, self.v, self.normal = z["origin"], z["u"], z["v"], z["normal"]
        self.n = len(self.origin)
        self.pos = self.origin @ self.normal
        self.step = float(np.median(np.abs(np.diff(self.pos)))) if self.n > 1 else 1.

    def project(self, X):
        """(slice index, column, row, inside) for an LPS point."""
        p = X @ self.normal
        k = int(np.argmin(np.abs(self.pos - p)))
        inside = abs(self.pos[k] - p) <= self.step / 2 + 1e-6
        d = X - self.origin[k]
        return k, d @ self.u / (self.u @ self.u), d @ self.v / (self.v @ self.v), inside

    def lps(self, k, i, j):
        return self.origin[k] + np.asarray(i)[..., None] * self.u + np.asarray(j)[..., None] * self.v


def tile_box(ci, cj):
    """Top-left corner of the TILE square centred at (ci, cj), clamped to the canvas."""
    h = TILE // 2
    return int(np.clip(round(ci) - h, 0, CANVAS - TILE)), int(np.clip(round(cj) - h, 0, CANVAS - TILE))


def sample_tiles(F, geos, per_series, rng, max_tries=20):
    """Training tiles for one knee: {series key: [(slice, x0, y0, anchor, centre_knee(3), corners_knee(4x3))]}."""
    out = {}
    for key, G in geos.items():
        tiles = []
        for _ in range(per_series):
            for _try in range(max_tries):
                X, (name,) = draw_points(F, 1, rng)
                k, ci, cj, inside = G.project(X[0])
                if inside and 0 <= ci < CANVAS and 0 <= cj < CANVAS:
                    break
            else:
                continue
            x0, y0 = tile_box(ci, cj)
            c = G.lps(k, x0 + (TILE - 1) / 2, y0 + (TILE - 1) / 2)
            corners = G.lps(k, np.array([x0, x0 + TILE - 1, x0, x0 + TILE - 1]), np.array([y0, y0, y0 + TILE - 1, y0 + TILE - 1]))
            tiles.append((k, x0, y0, name, to_knee(F, c), to_knee(F, corners)))
        out[key] = tiles
    return out


def grid_tiles(G, stride=144, every=1):
    """Validation tiles: a 3 x 3 grid (stride 144 px: 0, 144, 288 -> covers 512) on every `every`-th slice."""
    starts = list(range(0, CANVAS - TILE + 1, stride))
    if starts[-1] != CANVAS - TILE:
        starts.append(CANVAS - TILE)
    return [(k, x0, y0) for k in range(0, G.n, every) for y0 in starts for x0 in starts]


def load_knee(cache, kf_row, index):
    """(frame, {file: SeriesGeo}, index rows) for one knee of the cache."""
    F = frame(kf_row)
    rows = index[(index.study_uid == kf_row["study_uid"]) & (index.knee == kf_row["knee"])]
    return F, {r.file: SeriesGeo(cache / "geometry" / f"{r.file}.npz") for r in rows.itertuples()}, rows
