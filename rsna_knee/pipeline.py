"""Per-study preprocessing and tile scoring from raw DICOM; the same functions build the training caches (cache.py), so
training and inference inputs are identical.

  1. decode      every instance once (dicom.read_once), order along the slice normal, orient to the canonical patient frame
                 (90-degree rotations / flips only).
  2. native224   per series, the whole field of view -> 224 px, per-series 1-99 % window (keypoint localizer input).
  3. keypoints   5 localizer models (keypoint_model.py): band-centre + 7 random slice triplets per series, each also mirrored;
                 knees split by series field-of-view centre x (two-knee studies); robust 3D fusion across the series with a
                 moved-series check. Random triplets use a fixed seed per study.
  4. knee frame  O = tibiofemoral midpoint, axes m (medial) / s (superior) / a (anterior), scale k; side from anatomy.
  5. knee512     each series on a 512 x 512 canvas at 0.3125 mm/px around O, 0.5-99.5 % window, left knees mirrored.
  6. tiles       keypoint-anchor mixture (tiles.py), `tiles_per_series` per series, one bag per draw key.
  7. model       FP16 autocast; per-target attention temperature; two-knee studies: mean of the two knees.
Fallback (counted, never silent): a knee whose landmarks cannot be fused in 3D (single-plane study) uses each series' own
in-plane points at its middle slice.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch

from . import dicom as g8
from . import tiles as gt

PLANE_OF_AXIS = {0: "Sagittal", 1: "Coronal", 2: "Axial"}
PLANES = ("Sagittal", "Coronal", "Axial")
AXES = {"Sagittal": (np.array([0., 1., 0.]), np.array([0., 0., -1.])),
        "Coronal": (np.array([1., 0., 0.]), np.array([0., 0., -1.])),
        "Axial": (np.array([1., 0., 0.]), np.array([0., 1., 0.]))}
LANDMARKS = ["medial_tibiofemoral_joint", "lateral_tibiofemoral_joint", "intercondylar_notch", "patellofemoral_joint",
             "patella_center"]
BANDS = [(.05, .30), (.30, .70), (.70, .95)]
RESIDUAL_MM, GAP_MM, MIN_SERIES = 6.0, 100.0, 2
KP_SEED, N_RANDOM = 20261002, 7
MM, SIZE, PCT = 0.3125, 512, (0.5, 99.5)


# ----------------------------------------------------------------------------------------------------- 1. decode
def decode_series(paths, pool=None):
    recs = list(pool.map(g8.read_once, paths)) if pool is not None else [g8.read_once(p) for p in paths]
    ok = [r for r in recs if r.get("values") is not None and r.get("geometry") is not None]
    return ok, len(recs) - len(ok)


def orient(recs):
    """Ordered, oriented stack plus what native224 needs. None if unusable."""
    if len(recs) < 3:
        return None
    g8.order_records(recs)
    g0 = recs[0]["geometry"]
    if len({r["values"].shape for r in recs}) != 1:
        return None
    orients = np.stack([np.r_[r["geometry"]["row_dir"], r["geometry"]["col_dir"]] for r in recs])
    stack = np.stack([r["values"] for r in recs])
    if stack.dtype != np.float32:   # the reader already returns float32: no second copy
        stack = stack.astype(np.float32)
    for r in recs:
        r["values"] = None
    r_dir, c_dir, sx, sy = g0["row_dir"], g0["col_dir"], g0["d_col"], g0["d_row"]
    plane = PLANE_OF_AXIS[int(np.argmax(np.abs(np.cross(r_dir, c_dir))))]
    a, b = AXES[plane]
    origins = np.stack([r["geometry"]["ipp"] for r in recs]).astype(np.float64)
    if abs(r_dir @ a) < abs(c_dir @ a):
        stack = stack.transpose(0, 2, 1); r_dir, c_dir, sx, sy = c_dir, r_dir, sy, sx
    h, w = stack.shape[1:]
    if r_dir @ a < 0:
        origins = origins + r_dir * sx * (w - 1); r_dir = -r_dir; stack = stack[:, :, ::-1]
    if c_dir @ b < 0:
        origins = origins + c_dir * sy * (h - 1); c_dir = -c_dir; stack = stack[:, ::-1, :]
    # one partition pass for the knee512 (0.5 / 99.5) and native224 (1 / 99) windows: same order statistics, same values
    lo, hi, nlo, nhi = map(float, np.percentile(stack, [*PCT, g8.PCT_LO, g8.PCT_HI]))
    if hi <= lo:
        hi = lo + 1.
    return dict(stack=stack, origins=origins, r=r_dir, c=c_dir, sx=float(sx), sy=float(sy), plane=plane, lo=lo, hi=hi,
                native_pct=(nlo, nhi),
                native_ok=bool(np.abs(orients - orients[0]).max() <= 1e-3), n_files=len(recs))


# -------------------------------------------------------------------------------------------------- 2. native224
def native224(S):
    """Whole field of view -> 224 px, 1-99 % window. Returns (u8 volume, geometry) or None."""
    if not S["native_ok"]:
        return None
    stack = S["stack"]
    low, high = S["native_pct"] if "native_pct" in S else map(float, np.percentile(stack, [g8.PCT_LO, g8.PCT_HI]))
    if high <= low:
        low, high = float(stack.min()), float(stack.max()) + (1. if stack.max() <= stack.min() else 0.)
    origins, r_dir, c_dir, sx, sy = S["origins"].copy(), S["r"], S["c"], S["sx"], S["sy"]
    n, h, w = stack.shape
    square = abs(sx - sy) / max(sx, sy) > 0.01
    if square:
        s = min(sx, sy)
        nw, nh = int(round(w * sx / s)), int(round(h * sy / s))
        origins = origins + r_dir * ((s - sx) / 2) + c_dir * ((s - sy) / 2)
        sx = sy = s; h, w = nh, nw
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    origins = origins - r_dir * sx * left - c_dir * sy * top
    scale = side / 224
    interp = cv2.INTER_AREA if side >= 224 else cv2.INTER_CUBIC
    out = np.empty((n, 224, 224), np.float32)
    padded = np.zeros((side, side), np.float32)
    for k in range(n):
        x = np.clip((stack[k] - low) / (high - low), 0., 1.)
        if square:
            x = cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)
        padded[top:top + h, left:left + w] = x
        out[k] = cv2.resize(padded, (224, 224), interpolation=interp)
    origins = origins + (r_dir * sx + c_dir * sy) * (scale / 2 - .5)
    u, v = r_dir * sx * scale, c_dir * sy * scale
    normal = np.cross(u, v); normal /= np.linalg.norm(normal)
    normal *= np.sign((origins[-1] - origins[0]) @ normal)
    if normal[int(np.argmax(np.abs(normal)))] < 0:
        return None
    if np.any(np.diff(origins @ normal) <= 0):
        return None
    u8 = np.ascontiguousarray((np.clip(out, 0., 1.) * 255.).round().astype(np.uint8))
    return u8, dict(origin=origins, u=u, v=v, normal=normal)


class Geo224:
    """native224 series geometry (LPS <-> column, row, fractional slice) from an in-memory geometry dict."""
    def __init__(self, g):
        self.origin, self.u, self.v, self.normal = g["origin"], g["u"], g["v"], g["normal"]
        self.pos = self.origin @ self.normal
        self.uv = np.stack([self.u, self.v], 1)
        self.pixel_mm = float(np.linalg.norm(self.u))
        self.n = len(self.origin)

    def project(self, p):
        p = np.asarray(p, float)
        s = p @ self.normal
        k = float(np.interp(s, self.pos, np.arange(self.n)))
        if s < self.pos[0]:
            k = (s - self.pos[0]) / max(self.pos[1] - self.pos[0], 1e-6)
        elif s > self.pos[-1]:
            k = self.n - 1 + (s - self.pos[-1]) / max(self.pos[-1] - self.pos[-2], 1e-6)
        nearest = int(np.clip(round(k), 0, self.n - 1))
        ij, *_ = np.linalg.lstsq(self.uv, p - self.origin[nearest], rcond=None)
        off = max(self.pos[0] - s, s - self.pos[-1], 0.)
        return float(ij[0]), float(ij[1]), k, float(off)

    def to_lps(self, i, j, k):
        k0 = int(np.clip(np.floor(k), 0, self.n - 2)); t = k - k0
        o = self.origin[k0] * (1 - t) + self.origin[k0 + 1] * t
        return o + i * self.u + j * self.v


# -------------------------------------------------------------------------------------------------- 3. keypoints
def triplet_indices(n, rng=None):
    out = []
    for lo, hi in BANDS:
        a, b = lo * (n - 1), hi * (n - 1)
        x = rng.uniform(a, b) if rng is not None else (a + b) / 2
        out.append(int(np.clip(round(x), 0, n - 1)))
    return out


@torch.no_grad()
def kp_predict(model, vol, plane, device, rng, n_random=N_RANDOM, dtype=torch.float16):
    n = len(vol)
    sets = [triplet_indices(n)] + [triplet_indices(n, rng) for _ in range(n_random)]
    imgs, flip = [], []
    for idx in sets:
        img = np.stack([vol[k].astype(np.float32) / 255. for k in idx])
        imgs.append(img); flip.append(False)
        if plane == "Sagittal":
            m = [n - 1 - k for k in idx][::-1]
            imgs.append(np.stack([vol[k].astype(np.float32) / 255. for k in m]))
        else:
            imgs.append(img[:, :, ::-1].copy())
        flip.append(True)
    x = torch.from_numpy(np.stack(imgs)).to(device)
    with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
        logits, side = model(x, torch.full((len(imgs),), PLANES.index(plane), device=device))
    xy, _ = model.coords(logits)
    xy = xy.float().cpu().numpy(); flip = np.array(flip)
    if plane != "Sagittal":
        xy[flip, :, 0] = 223. - xy[flip, :, 0]
    left = torch.sigmoid(side).float().cpu().numpy(); left[flip] = 1 - left[flip]
    return dict(xy=xy.mean(0), spread=xy.std(0).mean(-1), left=float(left.mean()))


def kp_ensemble(models, vol, plane, device, rng):
    state = rng.bit_generator.state
    outs = []
    for m in models:
        rng.bit_generator.state = state
        outs.append(kp_predict(m, vol, plane, device, rng))
    return dict(xy=np.mean([o["xy"] for o in outs], 0), left=float(np.mean([o["left"] for o in outs])))


def fuse(geoms, preds, li, huber=5.0):
    A, b = [], []
    for uid, g in geoms.items():
        i, j = preds[uid]["xy"][li]
        o = g.origin[g.n // 2]
        for vec, val in ((g.u, i), (g.v, j)):
            nv = vec / np.linalg.norm(vec)
            A.append(nv); b.append(nv @ o + val * np.linalg.norm(vec))
    A, b = np.array(A), np.array(b)
    if np.linalg.matrix_rank(A) < 3:
        return None
    w = np.ones(len(b))
    for _ in range(6):
        X = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)[0]
        res = np.abs(A @ X - b)
        w = np.sqrt(np.where(res <= huber, 1., huber / np.maximum(res, 1e-6)))
    return X


def _residuals(geoms, preds, fused):
    out = {}
    for u, g in geoms.items():
        r = []
        for li, X in enumerate(fused):
            if X is not None:
                fi, fj, _, _ = g.project(X)
                r.append(np.hypot(preds[u]["xy"][li, 0] - fi, preds[u]["xy"][li, 1] - fj) * g.pixel_mm)
        out[u] = float(np.median(r)) if r else np.nan
    return out


def _planes(geoms):
    return {PLANE_OF_AXIS[int(np.argmax(np.abs(g.normal)))] for g in geoms.values()}


def fuse_knee(geoms, preds):
    """3D fusion of one knee with the moved-series set-aside loop. Returns the 5 fused points and the set-aside count."""
    main = dict(geoms); aside = {}
    while len(_planes(main)) >= 2:
        fused = [fuse(main, preds, li) for li in range(5)]
        res = _residuals(main, preds, fused)
        worst = max(res, key=lambda u: res[u] if np.isfinite(res[u]) else -1)
        if not res[worst] > RESIDUAL_MM:
            break
        rest = {u: g for u, g in main.items() if u != worst}
        if len(_planes(rest)) < 2:
            break
        aside[worst] = main.pop(worst)
    if len(_planes(aside)) >= 2 and len(aside) > len(main):
        main, aside = aside, main
    fused = [fuse(main, preds, li) for li in range(5)] if len(_planes(main)) >= 2 else [None] * 5
    return fused, len(aside)


def knee_groups(centres):
    order = sorted(centres, key=centres.get)
    xs = np.array([centres[s] for s in order])
    if len(xs) >= 2 * MIN_SERIES:
        gaps = np.diff(xs); cut = int(np.argmax(gaps)) + 1
        if gaps[cut - 1] >= GAP_MM and cut >= MIN_SERIES and len(xs) - cut >= MIN_SERIES:
            return {"R": order[:cut], "L": order[cut:]}
    return {"one": order}


# ------------------------------------------------------------------------------------------------- 4. knee frame
def knee_row(points):
    """Knee frame of one knee from its 5 fused landmarks: a row usable by tiles.frame (+ side)."""
    P = np.stack(points)
    M, L, Pa = P[0], P[1], P[4]
    O = (M + L) / 2
    m = M - L; k = np.linalg.norm(m) / 45.0; m = m / np.linalg.norm(m)
    z = np.array([0, 0, 1.0])
    s = z - (m @ z) * m; s /= np.linalg.norm(s)
    a = np.cross(s, m); a *= np.sign((Pa - O) @ a)
    row = {"side": "R" if M[0] > L[0] else "L", "scale_k": round(float(k), 4)}
    for name, v in (("O", O), ("m", m), ("a", a), ("s", s)):
        for i, c in enumerate("xyz"):
            row[f"{name}_{c}"] = round(float(v[i]), 6)
    for j, lm in enumerate(LANDMARKS):
        for i, c in enumerate("xyz"):
            row[f"{lm}_{c}"] = round(float(P[j, i]), 4)
    return row


def knee_frames(native, models, device, seed=KP_SEED):
    """native: {series: (u8 224 volume, geometry, plane)}. Returns [(knee, row, fallback)] for every knee."""
    geoms = {u: Geo224(g) for u, (_, g, _) in native.items()}
    centres = {u: float((g.origin[g.n // 2] + 111.5 * g.u + 111.5 * g.v)[0]) for u, g in geoms.items()}
    rng = np.random.default_rng(seed)
    preds = {u: kp_ensemble(models, vol, plane, device, rng) for u, (vol, _, plane) in native.items()}
    out = []
    for knee, uids in knee_groups(centres).items():
        kg = {u: geoms[u] for u in uids}
        fused, moved = fuse_knee(kg, preds)
        fallback = any(X is None for X in fused)
        if fallback:   # single-plane knee: each series' own in-plane points at its middle slice, averaged
            fused = [np.mean([g.to_lps(*preds[u]["xy"][li], (g.n - 1) / 2) for u, g in kg.items()], 0) for li in range(5)]
        row = knee_row(fused)
        row.update(knee=knee, moved_series=moved)
        out.append((knee, row, fallback))
    return out


# ---------------------------------------------------------------------------------------------------- 5. knee512
def render(S, O, mirror):
    """knee512 canvas of one series around O. Returns (uint8 [n, 512, 512], geometry dict) or None."""
    st, origins = S["stack"], S["origins"]; n, h, w = st.shape
    normal = np.cross(S["r"], S["c"]); normal /= np.linalg.norm(normal)
    pos = origins @ normal; po = O @ normal
    k0 = int(np.argmin(np.abs(pos - po)))
    d = O - origins[k0]; ci, cj = d @ S["r"] / S["sx"], d @ S["c"] / S["sy"]
    if not (0 <= ci < w and 0 <= cj < h and min(pos) - 10 <= po <= max(pos) + 10):
        return None
    nw, nh = max(int(round(w * S["sx"] / MM)), 1), max(int(round(h * S["sy"] / MM)), 1)
    ex, ey = w / nw, h / nh
    left = int(round((ci + .5) / ex - .5 - (SIZE - 1) / 2)); top = int(round((cj + .5) / ey - .5 - (SIZE - 1) / 2))
    interp = cv2.INTER_AREA if (nw < w or nh < h) else cv2.INTER_LANCZOS4
    out = np.zeros((n, SIZE, SIZE), np.uint8)
    a0, b0, a1, b1 = max(left, 0), max(top, 0), min(left + SIZE, nw), min(top + SIZE, nh)
    for k in range(n):
        x = cv2.resize(np.clip((st[k] - S["lo"]) / (S["hi"] - S["lo"]), 0, 1), (nw, nh), interpolation=interp)
        if a1 > a0 and b1 > b0:
            out[k, b0 - top:b1 - top, a0 - left:a1 - left] = (np.clip(x[b0:b1, a0:a1], 0, 1) * 255).round().astype(np.uint8)
    org = origins + S["r"] * S["sx"] * ((left + .5) * ex - .5) + S["c"] * S["sy"] * ((top + .5) * ey - .5)
    u, v = S["r"] * S["sx"] * ex, S["c"] * S["sy"] * ey
    if mirror:
        if S["plane"] == "Sagittal":
            out, org, normal = out[::-1], org[::-1], -normal
        else:
            out = out[:, :, ::-1]; org = org + u * (SIZE - 1); u = -u
    return np.ascontiguousarray(out), dict(origin=org, u=u, v=v, normal=normal)


class Geo512(gt.SeriesGeo):
    """tiles.SeriesGeo from an in-memory geometry dict."""
    def __init__(self, g):
        self.origin, self.u, self.v, self.normal = g["origin"], g["u"], g["v"], g["normal"]
        self.n = len(self.origin)
        self.pos = self.origin @ self.normal
        self.step = float(np.median(np.abs(np.diff(self.pos)))) if self.n > 1 else 1.


# ------------------------------------------------------------------------------------------------------ 6. tiles
def tile_bag(F, canvases, tiles_per_series, draw_key, fluid_mult=1.0, other_mult=1.0, slices_per_tile=1):
    """data.TileBags validation 'mix' (no jitter, no dropout). canvases: ordered {name: (u8 [n,512,512], Geo512, plane,
    fluid)}. fluid_mult / other_mult: tiles per series x mult on fluid-sensitive / other series, drawn in that order from
    one rng (as tile_data.TileBags); slices_per_tile 3: adjacent slices k-1, k, k+1 (clipped) as channels.
    Returns images [N,S,224,224] uint8, view, fluid."""
    rng = np.random.default_rng(draw_key) if draw_key else np.random.default_rng(0)
    geos = {f: c[1] for f, c in canvases.items()}
    if fluid_mult == other_mult == 1.0:
        drawn = gt.sample_tiles(F, geos, tiles_per_series, rng)
    else:
        drawn = {}
        for fl, mult in ((1, fluid_mult), (0, other_mult)):
            part = {f: g for f, g in geos.items() if int(canvases[f][3]) == fl}
            if part and round(tiles_per_series * mult) > 0:
                drawn.update(gt.sample_tiles(F, part, round(tiles_per_series * mult), rng))
    picks = [(f, t[0], t[1], t[2]) for f, ts in drawn.items() for t in ts]
    T, S = gt.TILE, slices_per_tile; half = S // 2
    imgs = np.empty((len(picks), S, T, T), np.uint8)
    for n, (f, k, x0, y0) in enumerate(picks):
        vol = canvases[f][0]
        for c, d in enumerate(range(-half, half + 1)):
            imgs[n, c] = vol[int(np.clip(k + d, 0, geos[f].n - 1)), y0:y0 + T, x0:x0 + T]
    view = np.array([PLANES.index(canvases[f][2]) for f, *_ in picks])
    fluid = np.array([int(canvases[f][3]) for f, *_ in picks])
    return imgs, view, fluid, picks


@torch.no_grad()
def tile_logits(model, imgs, view, fluid, device, tau, chunk=128, dtype=torch.float16):
    """One study bag -> 12 logits (FP16 autocast; attention logits x tau per target)."""
    model.att_tau = tau
    batch = dict(images=torch.from_numpy(imgs).to(device), view=torch.from_numpy(view).to(device),
                 fluid=torch.from_numpy(fluid).to(device), lengths=torch.tensor([len(imgs)]),
                 pos=torch.zeros(len(imgs), 15, device=device))
    with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
        return model(batch, chunk=chunk).float().cpu().numpy()[0]


# ---------------------------------------------------------------------------------------------------- study API
def prepare_series(series_list, pool=None):
    """series_list: [(series_uid, [dicom paths], fluid flag)] in metadata order -> oriented series + native224."""
    t = time.monotonic()
    oriented, native, bad = {}, {}, 0
    for uid, paths, fluid in series_list:
        recs, b = decode_series(paths, pool); bad += b
        S = orient(recs)
        if S is None:
            continue
        S["fluid"] = int(fluid)
        oriented[uid] = S
        nat = native224(S)
        if nat is not None:
            native[uid] = (nat[0], nat[1], S["plane"])
    return oriented, native, dict(series=len(series_list), oriented=len(oriented), native=len(native),
                                  unreadable_slices=bad, decode_seconds=time.monotonic() - t)


def knee_canvases(oriented, row):
    O = np.array([row["O_x"], row["O_y"], row["O_z"]])
    out = {}
    for uid, S in oriented.items():
        r = render(S, O, row["side"] == "L")
        if r is not None:
            out[uid] = (r[0], Geo512(r[1]), S["plane"], S["fluid"])
    return out
