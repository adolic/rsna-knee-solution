"""Train a crop specialist (crop_model.py on crop-cache windows, crop_data.py) for one fold.

Recipe: AdamW (head lr 4.243e-4, backbone --backbone-lr, wd 0.02), cosine with 3 % warm-up, backbone frozen for the first
--freeze-backbone-epochs then ramped, gradient clip 1 (backbone / head), EMA 0.999 with buffers, dropout 0.2, bf16, weighted
BCE with pos_weight sqrt((1-p)/p) in [0.5, 4]; for single-bag models hard-example mining in the 3 epochs after N-5 (weighted
resampling of misranked studies + a pairwise rank loss). Selection: the best mean AUC of the run's targets on the fold's
hard labels (validation from the 6th epoch); best.pt + selected-predictions.npz.
Modes:
  plain     one bag per study over --views, one output per --targets (ResNet50 OA specialist; ACL specialist)
  combo     --combo JSON {rows: [{target, views, cache, train_cache?, head?, extra?: [{cache, views}]}]}: one binary row per
            (study, structure), each from its own crop cache, shared backbone, one output per head; global view ids over the
            rows' (cache, view) pairs (--compartment-token keeps them, so the model knows which structure it scores);
            "extra" parts add windows from more caches to the same row (OA family: compartment + zoomed cartilage crops);
            with --train-labels-csv the rows with a train_cache train on it (OAI pretraining of the meniscus model)
Case weights: --isolated-weight (this compartment's OA positive, no other OA positive), --context-neg-weight (negative while
another OA target is positive), --cooccur-neg TARGET:OTHER=W (TARGET negative while OTHER positive, e.g. ACL:Contusion=3).
usage: python -m rsna_knee.train_crops --out RUNS/<name> --fold 1 --cache DATA/cache/crops-v8aclw --views aclw_sag aclw_cor aclw_ax --targets ACL ...
       (configs/specialists.json; scripts/70_train_specialists.py)"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .crop_data import Bags, collate, knee_entries, load_index
from .crop_model import CropModel
from .labels import labels, split
from .paths import CACHE

OA = ("Medial OA", "Lateral OA", "PF OA")
PLANE_ID = {"Coronal": 0, "Sagittal": 1, "Axial": 2}
AUG = dict(enabled=True, spatial=dict(probability=.75, rotation_degrees=3., translate_fraction=.02, scale=[.95, 1.]),
           intensity=dict(probability=.8, gain=[.9, 1.1], shift=[-.04, .04]), flips=False)


class Remap(torch.utils.data.Dataset):
    """A row's windows with their cache-local view ids mapped to the global (cache, view) ids."""
    def __init__(self, ds, view_map):
        self.ds, self.view_map, self.training = ds, view_map, ds.training
        self.soft, self.weights = ds.soft, ds.weights

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        s = self.ds[i]; s["view"] = self.view_map[s["view"]]
        return s


class Merge(torch.utils.data.Dataset):
    """A row reading several caches: one bag per study, the parts' windows concatenated (labels from the first part)."""
    def __init__(self, parts):
        self.parts, self.training = parts, parts[0].training
        self.soft, self.weights = parts[0].soft, parts[0].weights

    def __len__(self):
        return len(self.parts[0])

    def __getitem__(self, i):
        ss = [d[i] for d in self.parts]; s = dict(ss[0])
        for k in ("images", "view", "fluid", "depth"):
            s[k] = torch.cat([x[k] for x in ss])
        return s


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lr_mult(step, epochs, spe, warm=.03, floor=.1):
    total = spe * epochs; w = int(total * warm)
    if w and step < w:
        return max(1e-6, step / w)
    prog = min(max((step - w) / max(1, total - w), 0), 1)
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * prog))


def pairwise_rank(z, y):
    dz = z[:, None, :] - z[None, :, :]; m = ((y[:, None, :] - y[None, :, :]) >= .5).float()
    return (F.softplus(-dz) * m).sum() / m.sum().clamp_min(1.)


def to_cuda(b):
    return {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def loader(ds, bs, workers, sampler=None):
    return torch.utils.data.DataLoader(ds, batch_size=bs if ds.training else 1, shuffle=ds.training and sampler is None,
                                       sampler=sampler, drop_last=ds.training, num_workers=workers, pin_memory=True,
                                       collate_fn=collate, prefetch_factor=2 if workers else None)


@torch.no_grad()
def predict(model, ds, workers, view_map=None):
    model.eval(); out = []
    for b in loader(ds, 1, workers):
        b = to_cuda(b)
        if view_map is not None:
            b["view"] = view_map[b["view"]]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(b, chunk=128).float().cpu().numpy())
    return np.concatenate(out)


def cache_path(c):
    return Path(c) if Path(c).is_absolute() else CACHE / c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--cache", help="crop cache (plain mode)"); ap.add_argument("--views", nargs="+"); ap.add_argument("--targets", nargs="+")
    ap.add_argument("--budget", type=int, default=2, help="training windows per view per study")
    ap.add_argument("--backbone", default="convnext-tiny-in12k")
    ap.add_argument("--nslices", type=int, default=3); ap.add_argument("--step", type=int, default=1, help="0 = one slice repeated as 3 channels")
    ap.add_argument("--init-backbone", help="checkpoint whose backbone.* weights initialise the backbone (OAI pretraining)")
    ap.add_argument("--orientation", default="mirror", choices=["mirror", "canonical"])
    ap.add_argument("--input-size", type=int, default=224, help="model input px (resampled on the GPU)")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=4.243e-4); ap.add_argument("--backbone-lr", type=float, default=6e-5)
    ap.add_argument("--freeze-backbone-epochs", type=int, default=3); ap.add_argument("--backbone-warmup-epochs", type=float, default=2.0)
    ap.add_argument("--infer-all-slices", action="store_true", help="validate on every in-slab stored slice (single-slice models)")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--memory-fraction", type=float, default=0.95)
    ap.add_argument("--no-mining", action="store_true"); ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--isolated-weight", type=float, default=1.0); ap.add_argument("--context-neg-weight", type=float, default=1.0)
    ap.add_argument("--cooccur-neg", nargs="*", default=[], metavar="TARGET:OTHER=W")
    ap.add_argument("--combo", help="combo rows JSON (cache names relative to DATA/cache)")
    ap.add_argument("--compartment-token", action="store_true", help="combo: keep the global view ids (else plane ids only)")
    ap.add_argument("--train-labels-csv", help="combo: soft labels of the rows' train_cache knees (OAI pretraining)")
    ap.add_argument("--limit-train", type=int, default=0); ap.add_argument("--limit-val", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_float32_matmul_precision("high")
    cfg = dict(dropout=.2, channels_last=True, backbone=args.backbone, grad_checkpointing=False, early_checkpointing=False,
               checkpoint_chunk=128, in_chans=args.nslices, depth_embedding=False, orientation=args.orientation, input_zoom=1.0,
               input_size=args.input_size, augmentation=AUG)
    kw = dict(orientation=args.orientation, nslices=args.nslices, step=args.step)
    lim = lambda xs, n: xs[:n] if n else xs
    if args.combo:
        ROWS = json.loads(Path(args.combo).read_text())["rows"]
        parts = lambda r: [dict(cache=r["cache"], views=r["views"]), *r.get("extra", [])]
        caches, crops_c, ents_c = {}, {}, {}
        for r in ROWS:
            for c in filter(None, (*(q["cache"] for q in parts(r)), r.get("train_cache") if args.train_labels_csv else None)):
                if c not in caches:
                    caches[c] = cache_path(c); crops_c[c] = json.loads((caches[c] / "manifest.json").read_text())["config"]
                    ents_c[c] = knee_entries(load_index(caches[c]))
        tr, va = split(args.fold, sorted(set.intersection(*(set(ents_c[q["cache"]]) for r in ROWS for q in parts(r)))))
        okc = lambda u: all(ents_c[q["cache"]][u].view.isin(q["views"]).any() for r in ROWS for q in parts(r))   # no empty bags
        tr, va = lim([u for u in tr if okc(u)], args.limit_train), lim([u for u in va if okc(u)], args.limit_val)
        P = pd.read_csv(args.train_labels_csv, index_col=0) if args.train_labels_csv else None
        gviews = list(dict.fromkeys((q["cache"], v) for r in ROWS for q in parts(r) for v in q["views"]))
        HEADS = list(dict.fromkeys(r.get("head", "shared") for r in ROWS)); hix = lambda r: HEADS.index(r.get("head", "shared"))

        def onehead(a, r):
            o = np.zeros((len(a), len(HEADS)), np.float32); o[:, hix(r)] = a[:, 0]; return o
        gmap = lambda c, key: torch.tensor([gviews.index((key, v)) if (key, v) in gviews else 0 for v in crops_c[c]["views"]])
        trs, vals, tr_sets, extra_bags = [], [], [], []
        targets = [r["target"] for r in ROWS]
        for r in ROWS:
            t, c = r["target"], r["cache"]; tc = r.get("train_cache") if P is not None else None
            kc = dict(kw, use_views=r["views"], budgets={v: args.budget for v in crops_c[tc or c]["views"]})
            if tc:   # pretraining rows: every knee of the train cache that has a soft label
                trr = lim(sorted(set(ents_c[tc]) & set(P.index)), args.limit_train)
                d = Bags(caches[tc], crops_c[tc], trr, ents_c[tc], onehead(P.loc[trr, [t]].to_numpy(np.float32), r),
                         onehead(np.ones((len(trr), 1), np.float32), r), True, args.seed, **kc)
            else:
                d = Bags(caches[c], crops_c[c], tr, ents_c[c], onehead(labels(tr, "soft", [t]), r), onehead(labels(tr, "weights", [t]), r),
                         True, args.seed, **kc)
            trs.append(d); xt = [Remap(d, gmap(tc or c, c))]
            for q in r.get("extra", []):
                kq = dict(kw, use_views=q["views"], budgets={v: args.budget for v in crops_c[q["cache"]]["views"]})
                dq = Bags(caches[q["cache"]], crops_c[q["cache"]], tr, ents_c[q["cache"]], d.soft, d.weights, True, args.seed, **kq)
                extra_bags.append(dq); xt.append(Remap(dq, gmap(q["cache"], q["cache"])))
            tr_sets.append(Merge(xt) if len(xt) > 1 else xt[0])
            xv = []
            for q in parts(r):
                kq = dict(kw, use_views=q["views"], budgets={v: args.budget for v in crops_c[q["cache"]]["views"]})
                xv.append(Remap(Bags(caches[q["cache"]], crops_c[q["cache"]], va, ents_c[q["cache"]], labels(va, "hard", [t]),
                                     labels(va, "weights", [t]), False, infer_all=args.infer_all_slices, **kq), gmap(q["cache"], q["cache"])))
            vals.append(Merge(xv) if len(xv) > 1 else xv[0])
        train = torch.utils.data.ConcatDataset(tr_sets)
        train.soft = np.concatenate([d.soft for d in trs]); train.weights = np.concatenate([d.weights for d in trs]); train.training = True
        plane_map = (torch.arange(len(gviews), device="cuda") if args.compartment_token else
                     torch.tensor([PLANE_ID[crops_c[c]["views"][v]["plane"]] for c, v in gviews], device="cuda"))
        cfg["n_views"] = max(12, len(gviews)); cfg["n_targets"] = len(HEADS)
        tr_for_weights = tr
    else:
        ROWS, targets = None, args.targets
        cache = cache_path(args.cache); crop = json.loads((cache / "manifest.json").read_text())["config"]
        ent = knee_entries(load_index(cache)); tr, va = split(args.fold, sorted(ent))
        has = lambda u: ent[u].view.isin(args.views).any()   # no empty bags
        tr, va = lim([u for u in tr if has(u)], args.limit_train), lim([u for u in va if has(u)], args.limit_val)
        kw = dict(kw, use_views=args.views, budgets={v: args.budget for v in crop["views"]})
        train = Bags(cache, crop, tr, ent, labels(tr, "soft", targets), labels(tr, "weights", targets), True, args.seed, **kw)
        val = Bags(cache, crop, va, ent, labels(va, "hard", targets), labels(va, "weights", targets), False, infer_all=args.infer_all_slices, **kw)
        cfg["n_targets"] = len(targets)
        if len(crop["views"]) > 12:
            cfg["n_views"] = len(crop["views"])
        trs = [train]; tr_for_weights = tr
    if args.isolated_weight != 1 or args.context_neg_weight != 1:   # discordant OA cases (training labels at 0.5)
        oa_pos = labels(tr_for_weights, "soft", list(OA)) >= .5

        def discordant(t):
            i = OA.index(t); other = np.delete(oa_pos, i, 1).any(1)
            return np.where(oa_pos[:, i] & ~other, args.isolated_weight, 1.) * np.where(~oa_pos[:, i] & other, args.context_neg_weight, 1.)
        if ROWS:
            for d, r in zip(trs, ROWS):
                if r["target"] in OA and len(d.weights) == len(tr_for_weights):
                    d.weights = (d.weights * discordant(r["target"])[:, None]).astype(np.float32)
            train.weights = np.concatenate([d.weights for d in trs])
        else:
            train.weights = train.weights.copy()
            for j, t in enumerate(targets):
                if t in OA:
                    train.weights[:, j] *= discordant(t).astype(np.float32)
    for spec in args.cooccur_neg:   # hard negatives: TARGET negative while OTHER positive
        tgt, rest = spec.split(":"); other, w = rest.rsplit("=", 1)
        lab = labels(tr_for_weights, "soft", [tgt, other]) >= .5; m = ~lab[:, 0] & lab[:, 1]
        if ROWS:
            for r, d in zip(ROWS, trs):
                if r["target"] == tgt:
                    d.weights = d.weights.copy(); d.weights[m, hix(r)] *= float(w)
            train.weights = np.concatenate([d.weights for d in trs])
        else:
            train.weights = train.weights.copy(); train.weights[m, targets.index(tgt)] *= float(w)
    prev = ((train.soft * (train.weights > 0)).sum(0) / np.maximum((train.weights > 0).sum(0), 1) if ROWS else train.soft.mean(0)).clip(1e-4, 1 - 1e-4)
    pos_w = torch.tensor(np.sqrt((1 - prev) / prev).clip(.5, 4), dtype=torch.float32, device="cuda")
    norm = float(train.weights.mean())
    model = CropModel(cfg)
    if args.init_backbone:
        ck = torch.load(args.init_backbone, map_location="cpu", weights_only=False); sd = ck.get("shared", ck.get("model", ck))
        bsd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        miss = model.backbone.load_state_dict(bsd, strict=False)
        if miss.missing_keys:
            raise ValueError(f"backbone init missing {miss.missing_keys[:3]}")
    model = model.cuda().to(memory_format=torch.channels_last)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(.999), use_buffers=True)
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone")]; ids = {id(p) for p in bb}
    head = [p for p in model.parameters() if id(p) not in ids]
    opt = torch.optim.AdamW([dict(params=bb, lr=args.backbone_lr), dict(params=head, lr=args.lr)], weight_decay=.02)
    tl = loader(train, args.batch, args.workers); spe = len(tl)
    unfreeze, ramp = args.freeze_backbone_epochs * spe, max(args.backbone_warmup_epochs * spe, 1)
    bb_mult = lambda s: lr_mult(s, args.epochs, spe) * (min(max((s - unfreeze) / ramp, 0.), 1.) if args.backbone_warmup_epochs else 1.)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [bb_mult, lambda s: lr_mult(s, args.epochs, spe)])
    (out / "config.json").write_text(json.dumps(dict(vars(args), model=cfg, train_studies=len(tr), val_studies=len(va), targets=targets), indent=1) + "\n")
    hard = labels(va, "hard", targets)
    mine_from = args.epochs - 5
    history, best, mined, t0 = [], None, None, time.time()
    for epoch in range(1, args.epochs + 1):
        for d in trs + (extra_bags if ROWS else []):
            d.epoch = epoch - 1
        model.train(); te = time.time(); ls, n = 0., 0
        mining = mined is not None and mine_from < epoch <= mine_from + 3 and not ROWS
        frozen = epoch <= args.freeze_backbone_epochs
        for p in bb:
            p.requires_grad_(not frozen)
        for b in (mined if mining else tl):
            b = to_cuda(b)
            if ROWS:
                b["view"] = plane_map[b["view"]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(b).float()
            loss = (F.binary_cross_entropy_with_logits(z, b["labels"], reduction="none", pos_weight=pos_w) * b["weights"]).mean() / norm
            if mining:
                loss = loss + .5 * pairwise_rank(z, b["labels"])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(bb, 1.); torch.nn.utils.clip_grad_norm_(head, 1.)
            opt.step(); sched.step(); ema.update_parameters(model)
            ls += loss.detach(); n += 1
        rec = dict(epoch=epoch, loss=float(ls) / max(n, 1), train_seconds=round(time.time() - te, 1), mining=mining, backbone_frozen=frozen)
        if epoch >= min(6, args.epochs):
            if ROWS:
                logits = np.concatenate([predict(ema.module, v, args.workers, view_map=plane_map)[:, [hix(r)]] for r, v in zip(ROWS, vals)], 1)
            else:
                logits = predict(ema.module, val, args.workers)
            prob = 1 / (1 + np.exp(-logits))
            per = {t: float(roc_auc_score(hard[:, i], prob[:, i])) if 0 < hard[:, i].sum() < len(hard) else float("nan") for i, t in enumerate(targets)}
            rec.update(per_target=per, mean_auc=float(np.nanmean(list(per.values()))))
            if best is None or rec["mean_auc"] > best["mean_auc"]:
                best = dict(rec); torch.save(dict(model=ema.module.state_dict(), cfg=cfg), out / "best.pt")
                np.savez(out / "selected-predictions.npz", study_uids=np.asarray(va), logits=logits.astype(np.float32), targets=np.asarray(targets))
        rec["elapsed_seconds"] = round(time.time() - t0, 1)
        history.append(rec); (out / "history.json").write_text(json.dumps(history, indent=1) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "per_target"}), flush=True)
        if epoch == mine_from and not ROWS and not args.no_mining:
            mined = mine(ema.module, train, args)
    (out / "result.json").write_text(json.dumps(dict(status="completed", mean_auc=best["mean_auc"] if best else None,
                                                     best_epoch=best["epoch"] if best else None, per_target=best["per_target"] if best else None,
                                                     checkpoint_sha256=sha(out / "best.pt") if best else None,
                                                     elapsed_seconds=round(time.time() - t0, 1)), indent=1) + "\n")


@torch.no_grad()
def mine(model, train, args):
    scan = Bags(train.cache, train.cfg, train.studies, train.entries, train.soft, train.weights, False, orientation=train.orientation,
                use_views=train.use_views, budgets=train.budgets, nslices=train.nslices, step=train.step)
    p = 1 / (1 + np.exp(-predict(model, scan, args.workers))); y = train.soft >= .5; hard = np.zeros_like(p, bool)
    for t in range(p.shape[1]):
        pos, neg = p[y[:, t], t], p[~y[:, t], t]
        if len(pos) and len(neg):
            hard[:, t] = (y[:, t] & (p[:, t] < np.median(neg))) | (~y[:, t] & (p[:, t] > np.median(pos)))
    w = 1. + 3. * np.minimum(hard.sum(1), 3) / 3.
    s = torch.utils.data.WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(w), replacement=True,
                                               generator=torch.Generator().manual_seed(args.seed + 10))
    return loader(train, args.batch, args.workers, s)


if __name__ == "__main__":
    main()
