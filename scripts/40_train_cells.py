"""Train the tile models of configs/cells.json, stage by stage (each run is one rsna_knee.train_tiles call):

  A   sagD corA corD axiA                         ImageNet init, RSNA labels, 5 folds   (also the round-1 teachers)
  P1  sagFMpre corFMpre axOAIpre allfmpre allcfmpre  pretraining on DATA/pseudo/round1.csv (OAI + fastMRI), once
  B   sagFM corFM axOAI2 allfm allfmold allcfm      init from P1, RSNA labels, 5 folds
  P2  allrexpre                                   pretraining on DATA/pseudo/round2.csv
  C   allrex                                      init from P2, RSNA labels, 5 folds
Runs go to DATA/runs/<cell>-f<fold> (pretraining: DATA/runs/<name>); a run with result.json is skipped.
usage: python scripts/40_train_cells.py --stage A [--cells sagD] [--folds 0 1 2 3 4] [--dry-run] [--smoke] [--gpus 1]
  --smoke: 2 epochs on 24 training knees, validation on 40 fold studies, into DATA/runs/smoke/ (checks the wiring only)
  --gpus N > 1: torchrun with N processes; batch x accum x N must equal the recipe's effective batch (6 studies)"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from rsna_knee.paths import CACHE, CONFIGS, DATA, PSEUDO, RUNS  # noqa: E402

STAGES = ("A", "P1", "B", "P2", "C")


def run_args(name, spec, common, fold, out, smoke):
    a = dict(common, **{k: v for k, v in spec.items() if not k.startswith("_") and k not in ("stage", "train_cache", "train_labels", "init_from")})
    args = ["--out", str(out), "--fold", str(fold), "--cache", str(CACHE / "rsna")]
    if "train_cache" in spec:   # pretraining: external knees, validation on RSNA fold 1
        args += ["--train-cache", *[str(CACHE / c) for c in spec["train_cache"]],
                 "--train-labels-file", str(PSEUDO / f"{spec['train_labels'].replace('pseudo-', '')}.csv")]
    if "init_from" in spec:
        args += ["--init-from", str((RUNS / "smoke" if smoke else RUNS) / spec["init_from"] / "best.pt")]
    for k, v in a.items():
        if isinstance(v, bool):
            args += [f"--{k.replace('_', '-')}"] if v else []
        else:
            args += [f"--{k.replace('_', '-')}", str(v)]
    if smoke:
        args += ["--epochs", "2", "--limit-train", "24", "--limit-val", "40", "--workers", "2", "--val-tiles-per-series", "24"]
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=STAGES); ap.add_argument("--cells", nargs="*")
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gpus", type=int, default=1)
    a = ap.parse_args()
    cfg = json.loads((CONFIGS / "cells.json").read_text())
    pre = a.stage.startswith("P")
    table = cfg["pretrain"] if pre else cfg["cells"]
    names = [n for n, s in table.items() if s["stage"] == a.stage and (not a.cells or n in a.cells)]
    root = RUNS / "smoke" if a.smoke else RUNS
    jobs = []
    for n in names:
        for fold in ([1] if pre else a.folds):
            out = root / (n if pre else f"{n}-f{fold}")
            if (out / "result.json").exists():
                print(f"skip {out.name} (done)"); continue
            jobs.append((out, run_args(n, table[n], cfg["common"], fold, out, a.smoke)))
    for out, args in jobs:
        cmd = ([sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={a.gpus}", "-m", "rsna_knee.train_tiles"] if a.gpus > 1
               else [sys.executable, "-m", "rsna_knee.train_tiles"]) + args
        print(" ".join(cmd), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
