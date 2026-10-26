"""Pseudo labels for the external knees (OAI, fastMRI): a teacher ensemble scores every knee with the inference pipeline,
then the probabilities are calibrated for training.

  round 1: the four stage-A cells (5 folds each) at attention tau 2 + the ResNet50 OA specialist for PF OA (weight 0.3)
  round 2: the stage-A + stage-B ensemble at tau 12 + OA specialist (medial 0.3, PF 0.4) + canvas-crop specialists
(configs/cells.json "teachers"; the specialist blends are standardised over the scored set). Kept: knees with all three
planes (sagittal, coronal, axial series) and one knee per study (two-knee studies dropped). Calibration: the trainer's
positive-class weight w = sqrt((1 - p) / p) clipped to [0.5, 4] (p = training prevalence of labels/soft.csv) shifts the
logits up by log w, so pseudo label = sigmoid(logit - log w) undoes that shift for training on soft targets.
Steps: (1) teacher pack (scripts/90_pack_models.py --teacher roundN), (2) rsna_knee.predict on each source folder,
(3) calibrated CSV DATA/pseudo/round<N>.csv (StudyInstanceUID + 12 targets) for scripts/40_train_cells.py --stage P<N>.
usage: python scripts/50_pseudo_labels.py --round 1 --sources DATA/raw/oai-dicom:oai DATA/raw/fastmri-dicom:fastmri
       [--teacher-pack DIR] [--devices cuda:0,cuda:1] [--limit N]"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from rsna_knee.labels import labels  # noqa: E402
from rsna_knee.paths import DATA, LABELS, TARGETS  # noqa: E402


def calibrate(prob):
    studies = pd.read_csv(LABELS / "soft.csv", index_col=0).index
    prev = labels(studies, "soft").astype(float).mean(0).clip(1e-4, 1 - 1e-4)
    logw = np.log(np.sqrt((1 - prev) / prev).clip(.5, 4))
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) - logw)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, choices=(1, 2), required=True)
    ap.add_argument("--sources", nargs="+", required=True, help="FOLDER:SPLIT (Kaggle-style folders from 11_ / 12_prepare_*.py)")
    ap.add_argument("--teacher-pack", default=None, help="default DATA/teachers/round<N> (built from DATA/runs if missing)")
    ap.add_argument("--devices", default="cuda:0"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="default DATA/pseudo/round<N>.csv")
    a = ap.parse_args()
    pack = Path(a.teacher_pack or DATA / "teachers" / f"round{a.round}")
    if not (pack / "manifest.json").exists():
        subprocess.run([sys.executable, str(REPO / "scripts/90_pack_models.py"), "--teacher", f"round{a.round}", "--out", str(pack)], check=True)
    parts = []
    for src in a.sources:
        folder, split = src.rsplit(":", 1)
        out = DATA / "pseudo" / f"round{a.round}-{split}"
        studies = pd.read_csv(Path(folder) / f"{split}.csv").StudyInstanceUID.tolist()
        cmd = [sys.executable, "-m", "rsna_knee.predict", "--models", str(pack), "--input", folder, "--split", split, "--out", str(out),
               "--devices", a.devices] + (["--studies", *studies[:a.limit]] if a.limit else [])
        print(" ".join(cmd[:12]), "...", flush=True); subprocess.run(cmd, cwd=REPO, check=True)
        p = pd.read_csv(out / "predictions.csv", index_col=0)[list(TARGETS)]
        meta = pd.read_csv(Path(folder) / f"{split}_series.csv")
        three = meta.groupby("StudyInstanceUID").Anatomical_Plane.apply(lambda x: {"Sagittal", "Coronal", "Axial"} <= set(x))
        rec = pd.read_json(out / "receipts.jsonl", lines=True).set_index("study")
        one = rec.apply(lambda r: r.status == "ok" and isinstance(r.knees, list) and len(r.knees) == 1, axis=1)
        keep = p.index[three.reindex(p.index).fillna(False).astype(bool) & one.reindex(p.index).fillna(False).astype(bool)]
        print(f"{split}: keep {len(keep)} of {len(p)} knees (all three planes, one knee)", flush=True)
        p = p.loc[keep]
        parts.append(pd.DataFrame(calibrate(p.to_numpy(float)), index=p.index, columns=list(TARGETS)))
    lab = pd.concat(parts); lab.index.name = "StudyInstanceUID"
    dst = Path(a.out or DATA / "pseudo" / f"round{a.round}.csv"); dst.parent.mkdir(parents=True, exist_ok=True)
    lab.to_csv(dst, float_format="%.6f")
    print(f"{len(lab)} knees -> {dst}; mean pseudo label per target: {lab.mean().round(3).to_dict()}")


if __name__ == "__main__":
    main()
