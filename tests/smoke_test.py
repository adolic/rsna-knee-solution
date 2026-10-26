"""Smoke test of the whole training pipeline on a few studies (minutes, one GPU): every stage runs end to end on tiny data
and writes its outputs; it checks the wiring, not the scores. Needs DATA/raw/rsna and DATA/models (keypoint models for the
cache build); OAI / fastMRI stages are skipped unless --fastmri-archive is given (the first studies of the archive).
Everything is written under DATA/smoke (pass RSNA_KNEE_DATA=<scratch dir> to keep the real data folder clean; raw/rsna,
models and pretrained are then linked from --data).
Steps: pretrained backbones -> RSNA cache (70 studies) -> keypoint training (5 epochs, 256 series) -> stage-A tile cell
(2 epochs) -> crop caches + the 4 crop specialists (6 epochs) -> specialist pack + model pack -> inference with that pack ->
[fastMRI preparation + round-1 pseudo labels on 3 studies].
usage: RSNA_KNEE_DATA=/tmp/smoke python tests/smoke_test.py --data /path/to/real/data [--fastmri-archive knee_DICOMs_batch2.tar.xz]"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from rsna_knee.paths import DATA  # noqa: E402

PY = sys.executable


def step(name, cmd, results):
    t = time.time(); print(f"\n=== {name}\n{' '.join(map(str, cmd))}", flush=True)
    r = subprocess.run(list(map(str, cmd)), cwd=REPO)
    results.append((name, r.returncode == 0, time.time() - t))
    if r.returncode:
        raise SystemExit(f"FAILED: {name}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True, help="the real data folder (raw/rsna, models, pretrained)")
    ap.add_argument("--fastmri-archive"); ap.add_argument("--devices", default="cuda:0")
    a = ap.parse_args()
    real = Path(a.data).resolve()
    for sub in ("raw/rsna", "models", "pretrained"):
        dst = DATA / sub; dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() and (real / sub).exists() and real != DATA:
            os.symlink(real / sub, dst)
    import pandas as pd
    f = pd.read_csv(REPO / "labels/folds.csv")
    studies = sorted(f.StudyInstanceUID[f.fold == 1])[:40] + sorted(f.StudyInstanceUID[f.fold != 1])[:30]
    res = []
    step("pretrained backbones (download + sha256)", [PY, "scripts/00_download_pretrained.py"], res)
    step("RSNA cache: native224 + keypoint frames + knee512 (70 studies)",
         [PY, "scripts/20_build_cache.py", "--input", DATA / "raw/rsna", "--split", "train", "--out", DATA / "cache/rsna", "--workers", "4", "--studies", *studies], res)
    step("keypoint localizer (5 epochs, 256 series)", [PY, "scripts/10_train_keypoints.py", "--folds", "0", "--smoke"], res)
    step("stage-A tile cell corA (2 epochs, 24 knees)", [PY, "scripts/40_train_cells.py", "--stage", "A", "--cells", "corA", "--folds", "1", "--smoke"], res)
    step("crop caches", [PY, "scripts/70_train_specialists.py", "--stage", "caches", "--caches", "crops-v5-oa320", "crops-v6iso", "crops-v7men",
                         "crops-v8aclw", "crops-v8zoom", "--workers", "8"], res)
    step("crop specialists r50oa acl oafam (6 epochs)", [PY, "scripts/70_train_specialists.py", "--stage", "train", "--runs", "r50oa", "acl", "oafam", "--folds", "1", "--smoke"], res)
    step("meniscus specialist (6 epochs, no OAI pretraining)", [PY, "-m", "rsna_knee.train_crops", "--out", DATA / "runs/smoke/spec-men-f1", "--fold", "1",
         "--combo", REPO / "configs/combos/men.json", "--compartment-token", "--epochs", "6", "--limit-train", "24", "--limit-val", "40", "--workers", "2"], res)
    step("specialist pack", [PY, "scripts/70_train_specialists.py", "--stage", "pack", "--folds", "1", "--smoke"], res)
    step("model pack from the smoke runs", [PY, "scripts/90_pack_models.py", "--out", DATA / "smoke-pack", "--cells", "corA", "--folds", "1",
         "--runs", DATA / "runs/smoke", "--keypoints", *[DATA / f"models/keypoints/kp-f{k}.pt" for k in range(5)],
         "--oa-from", DATA / "specialist-pack", "--spec-from", DATA / "specialist-pack"], res)
    step("inference with the smoke pack (3 studies)", [PY, "-m", "rsna_knee.predict", "--models", DATA / "smoke-pack", "--input", DATA / "raw/rsna",
         "--split", "train", "--studies", *studies[:3], "--out", DATA / "smoke-pred", "--devices", a.devices, "--workers", "1"], res)
    if a.fastmri_archive:
        step("fastMRI preparation (3 studies)", [PY, "scripts/12_prepare_fastmri.py", "--archives", a.fastmri_archive, "--out", DATA / "raw/fastmri-dicom",
                                                  "--limit-studies", "3"], res)
        step("round-1 teacher pack from the released pack", [PY, "scripts/90_pack_models.py", "--teacher", "round1", "--from-pack", DATA / "models",
                                                              "--out", DATA / "teachers/round1"], res)
        step("round-1 pseudo labels (3 fastMRI studies)", [PY, "scripts/50_pseudo_labels.py", "--round", "1", "--sources", f"{DATA / 'raw/fastmri-dicom'}:fastmri",
                                                           "--devices", a.devices], res)
    print("\n=== summary")
    for name, ok, sec in res:
        print(f"  {'ok  ' if ok else 'FAIL'} {sec:7.0f}s  {name}")


if __name__ == "__main__":
    main()
