"""Train one tile model (one cell x one fold): RSNA fine-tuning on the soft labels, or pretraining on pseudo-labelled
external knees (OAI / fastMRI) with validation on the RSNA fold.

Model (tile_model.py): shared backbone per 224 px tile (3 adjacent slices as channels) -> 256-d token + plane / fluid
embeddings -> one transformer layer over the knee's tiles -> per-target gated attention pooling -> 12 logits.
Recipe: AdamW (head lr 4.243e-4, backbone lr --backbone-lr, wd 0.02), cosine with 3 % warm-up, backbone frozen for the
first --freeze-backbone-epochs and ramped over --backbone-warmup-epochs, gradient clip 1 (backbone and head separately),
EMA 0.999 (with buffers) for validation and checkpoints, bf16 autocast, weighted BCE with pos_weight sqrt((1-p)/p)
clipped to [0.5, 4] from the training prevalence, labels as soft targets with per-cell weights.
Validation on the fold's hard labels from the 6th epoch (every 2nd epoch and the last five); the checkpoint with the best
Kaggle-proxy AUC (proxy.py) is best.pt, its predictions selected-predictions.npz. Multi-GPU: torchrun (DDP).
usage: python -m rsna_knee.train_tiles --out RUNS/<name> --fold 1 --cache DATA/cache/rsna --view coronal --backbone convnext-tiny-in12k ...
       (configs/cells.json lists every cell's arguments; scripts/40_train_cells.py runs them)"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from . import tiles as gt
from .labels import TARGET_COLUMNS, labels, split
from .proxy import proxy_auc
from .tile_data import TileBags, collate, knees
from .tile_model import TileModel

AUG = dict(enabled=True, spatial=dict(probability=.75, rotation_degrees=3., translate_fraction=.02, scale=[.95, 1.]),
           intensity=dict(probability=.8, gain=[.9, 1.1], shift=[-.04, .04]), flips=False)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lr_mult(step, epochs, spe, warm=.03, floor=.1):
    total = epochs * spe; w = max(int(total * warm), 1)
    if step < w:
        return (step + 1) / w
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * min((step - w) / max(total - w, 1), 1)))


def to_cuda(b):
    return {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def loader(ds, bs, workers, sampler=None):
    return torch.utils.data.DataLoader(ds, batch_size=bs if ds.training else 1, shuffle=ds.training and sampler is None,
                                       drop_last=ds.training, sampler=sampler, num_workers=workers, pin_memory=True,
                                       collate_fn=collate, prefetch_factor=2 if workers else None, persistent_workers=False)


@torch.no_grad()
def predict(model, ds, workers):
    model.eval(); out = []
    for b in loader(ds, 1, workers):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(to_cuda(b), chunk=128).float().cpu().numpy())
    return np.concatenate(out)


def predict_dist(model, ds, workers, rank, world):
    """predict() split over the DDP ranks (study i -> rank i % world), gathered back in order on every rank."""
    if world == 1:
        return predict(model, ds, workers)
    import torch.distributed as dist
    idx = list(range(rank, len(ds), world)); part = torch.utils.data.Subset(ds, idx); part.training = False
    got = [None] * world; dist.all_gather_object(got, (idx, predict(model, part, workers)))
    out = None
    for ix, lg in got:
        out = np.zeros((len(ds), lg.shape[1]), lg.dtype) if out is None else out; out[ix] = lg
    return out


def save_preds(path, uids, logits, labels_hard):
    tmp = path.with_name(".tmp-" + path.name)
    np.savez(tmp, study_uids=np.asarray(uids), logits=logits.astype(np.float32),
             probabilities=(1 / (1 + np.exp(-logits.astype(np.float64)))).astype(np.float32), labels=labels_hard.astype(np.uint8))
    tmp.replace(path)


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--cache", required=True, help="RSNA knee512 cache (rsna_knee.cache)")
    ap.add_argument("--view", default="all", choices=("all", "sagittal", "coronal", "axial", "corfluid"))
    ap.add_argument("--backbone", default="convnext-tiny-in12k")
    ap.add_argument("--tiles-per-series", type=int, default=12); ap.add_argument("--val-tiles-per-series", type=int, default=96)
    ap.add_argument("--slices-per-tile", type=int, choices=(1, 3), default=3)
    ap.add_argument("--fluid-mult", type=float, default=1.0); ap.add_argument("--other-mult", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=3, help="one optimiser step per N batches")
    ap.add_argument("--lr", type=float, default=4.243e-4); ap.add_argument("--backbone-lr", type=float, default=6e-5)
    ap.add_argument("--freeze-backbone-epochs", type=int, default=2); ap.add_argument("--backbone-warmup-epochs", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--memory-fraction", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--mirror-p", type=float, default=0.5, help="study-level left-right mirror (plane aware; labels unchanged)")
    ap.add_argument("--ema-decay", type=float, default=.999)
    ap.add_argument("--early-stop", type=int, default=0, help="stop when the proxy AUC has not improved for N epochs (0 = off)")
    ap.add_argument("--save-last", action="store_true", help="also save the final EMA weights (last.pt)")
    ap.add_argument("--save-best-macro", action="store_true", help="also keep best-macro.pt (macro-AUC selection)")
    ap.add_argument("--max-train-series", type=int, default=0, help="training bags use at most N series per knee (memory)")
    ap.add_argument("--init-from", default="", help="start from this run's best.pt (or a .pt): fine-tuning after pretraining")
    ap.add_argument("--train-cache", nargs="*", default=[], help="pretraining: train on these caches' knees (OAI / fastMRI)")
    ap.add_argument("--train-labels-file", default="", help="pretraining: soft labels CSV (StudyInstanceUID + 12 targets [+ extra columns])")
    ap.add_argument("--train-weights-file", default="", help="pretraining: per-cell loss weights (same columns; NaN = 0)")
    ap.add_argument("--train-view", default=None, help="view of the --train-cache knees (default --view)")
    ap.add_argument("--limit-train", type=int, default=0, help="smoke tests: train on the first N training knees only")
    ap.add_argument("--limit-val", type=int, default=0, help="smoke tests: validate on the first N fold studies only")
    return ap.parse_args(argv)


def auc_or_nan(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def selection_metrics(va, hard, prob):
    """(per-target AUC, macro, proxy); the proxy needs every target defined: on tiny smoke-test folds it falls back to macro."""
    per = {t: auc_or_nan(hard[:, i], prob[:, i]) for i, t in enumerate(TARGET_COLUMNS)}
    macro = float(np.nanmean(list(per.values())))
    try:
        proxy = float(proxy_auc(va, hard, prob)[0])
    except ValueError:
        proxy = macro
    return per, macro, proxy


def main(argv=None):
    args = parse(argv)
    world = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0")); main0 = rank == 0
    if world > 1:
        import torch.distributed as dist
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"])); dist.init_process_group("nccl")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_float32_matmul_precision("high")
    label_cols = list(pd.read_csv(args.train_labels_file, nrows=0, index_col=0).columns) if args.train_labels_file else list(TARGET_COLUMNS)
    if label_cols[:12] != list(TARGET_COLUMNS):
        raise ValueError("--train-labels-file must start with the 12 target columns in order")
    cfg = dict(dropout=.2, channels_last=True, n_targets=len(label_cols), extra_targets=label_cols[12:], backbone=args.backbone,
               backbone2="", in_chans=3, position="none", plane_backbones=False, topk=0, pool="attn", pos_dropout=0.0, cell_grid=0,
               augmentation=dict(AUG, mirror=args.mirror_p) if args.mirror_p else AUG)
    km = knees(args.cache, args.view); tr, va = split(args.fold, sorted(km))
    if args.limit_val:
        va = va[:args.limit_val]
    bags = dict(tiles_per_series=args.tiles_per_series, val_tiles_per_series=args.val_tiles_per_series,
                slices_per_tile=args.slices_per_tile, fluid_mult=args.fluid_mult, other_mult=args.other_mult)
    if args.train_cache:   # pretraining on pseudo-labelled external knees; validation unchanged (RSNA fold)
        km_t = {}
        for c in args.train_cache:
            km_t.update(knees(c, args.train_view or args.view))
        lab = pd.read_csv(args.train_labels_file, index_col=0)[label_cols]
        tr = sorted(set(km_t) & set(lab.index))
        tr = tr[:args.limit_train] if args.limit_train else tr
        raw = lab.reindex(tr).to_numpy(np.float32); soft = np.nan_to_num(raw, nan=0.0)
        wts = (pd.read_csv(args.train_weights_file, index_col=0)[label_cols].reindex(tr).to_numpy(np.float32) if args.train_weights_file
               else np.ones_like(soft))
        wts = np.where(np.isnan(raw) | np.isnan(wts), 0.0, wts).astype(np.float32)
        train = TileBags(tr, km_t, soft, wts, True, args.seed, max_train_series=args.max_train_series, **bags)
    else:
        tr = tr[:args.limit_train] if args.limit_train else tr
        train = TileBags(tr, km, labels(tr, "soft"), labels(tr, "weights"), True, args.seed, max_train_series=args.max_train_series, **bags)
    hard_eval = labels(va, "hard")
    val = TileBags(va, km, hard_eval, labels(va, "weights"), False, **bags)
    print(json.dumps(dict(stage="data", train=len(tr), val=len(va), outputs=len(label_cols))), flush=True)
    prev = ((train.soft * (train.weights > 0)).sum(0) / np.maximum((train.weights > 0).sum(0), 1)).clip(1e-4, 1 - 1e-4)
    pos_w = torch.tensor(np.sqrt((1 - prev) / prev).clip(.5, 4), dtype=torch.float32, device="cuda")
    norm = float(train.weights.mean())
    model = TileModel(cfg).cuda().to(memory_format=torch.channels_last)
    if args.init_from:   # fine-tuning: start from a trained run's weights (same architecture; extra pretraining heads dropped)
        src = Path(args.init_from); src = src / "best.pt" if src.is_dir() else src
        sd = torch.load(src, map_location="cuda", weights_only=False)["model"]
        own = model.state_dict()
        for k, v in list(sd.items()):
            if k in own and v.shape != own[k].shape:   # a pretraining head with extra outputs -> its first 12 rows
                dim = next(d for d in range(v.dim()) if v.shape[d] != own[k].shape[d])
                sd[k] = v.narrow(dim, 0, own[k].shape[dim])
        miss = model.load_state_dict(sd, strict=False)
        if miss.missing_keys or miss.unexpected_keys:
            raise ValueError(f"init-from mismatch: {(miss.missing_keys + miss.unexpected_keys)[:5]}")
        print(json.dumps(dict(stage="init_from", checkpoint=str(src))), flush=True)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema_decay), use_buffers=True)
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone")]; ids = {id(p) for p in bb}
    head = [p for p in model.parameters() if id(p) not in ids]
    opt = torch.optim.AdamW([dict(params=bb, lr=args.backbone_lr), dict(params=head, lr=args.lr)], weight_decay=.02)
    sampler = (torch.utils.data.distributed.DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
               if world > 1 else None)
    tl = loader(train, args.batch, args.workers, sampler); spe = max(len(tl) // args.accum, 1)
    fwd = (torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
           if world > 1 else model)
    unfreeze, ramp = args.freeze_backbone_epochs * spe, max(args.backbone_warmup_epochs * spe, 1)
    bb_mult = lambda s: lr_mult(s, args.epochs, spe) * (min(max((s - unfreeze) / ramp, 0.), 1.) if args.backbone_warmup_epochs else 1.)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [bb_mult, lambda s: lr_mult(s, args.epochs, spe)])
    if main0:
        (out / "config.json").write_text(json.dumps(dict(vars(args), model=cfg, train_studies=len(tr), val_studies=len(va), world_size=world), indent=1) + "\n")
    history, best, best_macro, t0 = [], None, None, time.time()
    log = open(out / "train.log", "a") if main0 else open(os.devnull, "w")
    for epoch in range(1, args.epochs + 1):
        train.epoch = epoch - 1; model.train(); te = time.time(); ls, n = 0., 0
        if sampler is not None:
            sampler.set_epoch(epoch)
        frozen = epoch <= args.freeze_backbone_epochs
        for p in bb:   # DDP keeps requires_grad; frozen backbone grads are dropped before the step instead
            p.requires_grad_(not frozen or world > 1)
        opt.zero_grad(set_to_none=True)
        for j, b in enumerate(tl):
            b = to_cuda(b)
            sync = fwd.no_sync() if world > 1 and (j + 1) % args.accum != 0 else contextlib.nullcontext()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = fwd(b).float()
                loss = (F.binary_cross_entropy_with_logits(z, b["labels"], reduction="none", pos_weight=pos_w) * b["weights"]).mean() / norm
                (loss / args.accum).backward()
            ls += loss.detach(); n += 1
            if (j + 1) % args.accum == 0:   # effective batch = batch x accum x world studies
                if frozen and world > 1:
                    for p in bb:
                        p.grad = None
                torch.nn.utils.clip_grad_norm_(bb, 1.); torch.nn.utils.clip_grad_norm_(head, 1.)
                opt.step(); sched.step(); ema.update_parameters(model)
                opt.zero_grad(set_to_none=True)
        rec = dict(epoch=epoch, loss=float(ls) / max(n, 1), train_seconds=round(time.time() - te, 1), backbone_frozen=frozen)
        if epoch >= min(6, args.epochs) and (epoch % 2 == 0 or epoch >= args.epochs - 4):
            tv = time.time(); logits = predict_dist(ema.module, val, args.workers, rank, world)[:, :12]; prob = 1 / (1 + np.exp(-logits))
            per, macro, proxy = selection_metrics(va, hard_eval, prob)
            rec.update(per_target=per, macro_auc=macro, proxy_auc=proxy, validation_seconds=round(time.time() - tv, 1))
            if main0:
                save_preds(out / f"epoch-{epoch:02d}.npz", va, logits, hard_eval)
            if best is None or rec["proxy_auc"] > best["proxy_auc"]:
                best = dict(rec)
                if main0:
                    torch.save(dict(model=ema.module.state_dict(), cfg=cfg), out / "best.pt")
                    save_preds(out / "selected-predictions.npz", va, logits, hard_eval)
            if args.save_best_macro and (best_macro is None or rec["macro_auc"] > best_macro["macro_auc"]):
                best_macro = dict(rec)
                if main0:
                    torch.save(dict(model=ema.module.state_dict(), cfg=cfg), out / "best-macro.pt")
        rec["elapsed_seconds"] = round(time.time() - t0, 1)
        history.append(rec)
        if main0:
            (out / "history.json").write_text(json.dumps(history, indent=1) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "per_target"}), file=log, flush=True)
        print(json.dumps({k: v for k, v in rec.items() if k != "per_target"}), flush=True)
        if args.early_stop and best is not None and "proxy_auc" in rec and epoch - best["epoch"] >= args.early_stop:
            print(json.dumps(dict(stage="early_stop", epoch=epoch, best_epoch=best["epoch"])), file=log, flush=True)
            break
    if world > 1:
        dist.barrier(); dist.destroy_process_group()
    if not main0:
        return
    if args.save_last:
        torch.save(dict(model=ema.module.state_dict(), cfg=cfg, epoch=epoch), out / "last.pt")
    result = dict(status="completed", epochs_run=epoch)
    if best is not None:
        result.update(proxy_auc=best["proxy_auc"], macro_auc=best["macro_auc"], best_epoch=best["epoch"], per_target=best["per_target"],
                      checkpoint_sha256=sha(out / "best.pt"))
    result.update(elapsed_seconds=round(time.time() - t0, 1), peak_vram_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
