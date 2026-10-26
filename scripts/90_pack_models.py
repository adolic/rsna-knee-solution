"""Pack trained runs into an inference model pack (the format rsna_knee.predict reads and the released pack uses).

  tiles/pack_<cell>.pt   {<cell>_f<fold>: {"cfg", "model": best.pt EMA weights in FP16}} for every requested cell
  keypoints/kp-f<k>.pt   the keypoint localizer fold models (last.pt)
  oa/, specialists/      the ResNet50 OA specialist and the canvas-crop specialists (scripts/70_train_specialists.py output,
                         or copied from another pack with --oa-from / --spec-from)
  manifest.json          members in configs/cells.json "inference_order" (the order fixes each member's tile draw), the
                         inference recipe, the training prior and every file's sha256
Uses: the final submission (--recipe configs/submission_recipe.json, all 11 cells), and the pseudo-label teachers
(--teacher round1 | round2: the cells, attention tau and OA / specialist weights of configs/cells.json "teachers").
usage: python scripts/90_pack_models.py --out DATA/models [--cells ...] [--teacher round1] [--oa-from PACK] [--spec-from PACK]"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from rsna_knee.labels import labels  # noqa: E402
from rsna_knee.paths import CONFIGS, LABELS, RUNS, TARGETS  # noqa: E402

ACQ = {"all": None, "sagittal": ["Sagittal", None], "coronal": ["Coronal", None], "axial": ["Axial", None], "corfluid": ["Coronal", 1]}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--cells", nargs="*"); ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--runs", default=str(RUNS)); ap.add_argument("--recipe", default=str(CONFIGS / "submission_recipe.json"))
    ap.add_argument("--teacher", choices=("round1", "round2"), help="pack a pseudo-label teacher instead of the submission")
    ap.add_argument("--keypoints", nargs="*", help="keypoint run dirs (default RUNS/kp-f0 ... kp-f4)")
    ap.add_argument("--oa-from"); ap.add_argument("--spec-from")
    ap.add_argument("--from-pack", help="take the cells' members, keypoints, OA and specialists from this existing pack instead of runs")
    a = ap.parse_args()
    cfg = json.loads((CONFIGS / "cells.json").read_text()); runs = Path(a.runs); out = Path(a.out)
    for d in ("tiles", "keypoints"):
        (out / d).mkdir(parents=True, exist_ok=True)
    recipe = json.loads(Path(a.recipe).read_text())
    if a.teacher:
        t = cfg["teachers"][a.teacher]; cells = t["cells"]
        recipe = dict(name=f"pseudo-label teacher {a.teacher}", speed=recipe["speed"],
                      inference=dict(own_draws=True, attention_tau=float(t["tau"]), cell_chunk=recipe["inference"].get("cell_chunk", {})),
                      oa=dict(recipe["oa"], weights=t["oa_weights"]), **({"spec": recipe["spec"]} if t["specialists"] else {}))
    else:
        cells = a.cells or cfg["inference_order"]
    files, members = {}, []
    if a.from_pack:   # e.g. a pseudo-label teacher made from the released pack
        src = Path(a.from_pack); sm = json.loads((src / "manifest.json").read_text())
        for m in sm["members"]:
            if m["cell"] in cells and m["fold"] in a.folds:
                members.append(m)
                if m["checkpoint"] not in files:
                    shutil.copy2(src / m["checkpoint"], out / m["checkpoint"]); files[m["checkpoint"]] = sha(out / m["checkpoint"])
        a.keypoints = a.keypoints or []
        for f in sm["keypoint_models"]:
            shutil.copy2(src / f, out / f); files[f] = sha(out / f)
        a.oa_from, a.spec_from = a.oa_from or src, a.spec_from or src
        cells = []
    for cell in [c for c in cfg["inference_order"] if c in cells]:
        spec = cfg["cells"][cell]; common = cfg["common"]; pack = {}
        for k in a.folds:
            ck = torch.load(runs / f"{cell}-f{k}" / "best.pt", map_location="cpu", weights_only=False)
            pack[f"{cell}_f{k}"] = dict(cfg=ck["cfg"], model={n: v.half() if v.is_floating_point() else v for n, v in ck["model"].items()})
            m = dict(name=f"{cell}_f{k}", cell=cell, fold=k, group="planes", checkpoint=f"tiles/pack_{cell}.pt", checkpoint_key=f"{cell}_f{k}",
                     acquisition=ACQ[spec["view"]], tiles_per_series=96, fluid_mult=spec.get("fluid_mult", common["fluid_mult"]),
                     other_mult=spec["other_mult"], slices_per_tile=common["slices_per_tile"], attention_tau=[1.0] * 12, draw_keys=[0])
            if spec.get("plane"):
                m["plane"] = spec["plane"]
            members.append(m)
        torch.save(pack, out / f"tiles/pack_{cell}.pt"); files[f"tiles/pack_{cell}.pt"] = sha(out / f"tiles/pack_{cell}.pt")
        print(f"packed {cell}: {len(pack)} folds", flush=True)
    kps = [f for f in files if f.startswith("keypoints/")]
    for i, d in enumerate(a.keypoints if a.keypoints is not None else [runs / f"kp-f{k}" for k in range(5)]):
        dst = out / f"keypoints/kp-f{i}.pt"; shutil.copy2(Path(d) if Path(d).is_file() else Path(d) / "last.pt", dst); files[f"keypoints/kp-f{i}.pt"] = sha(dst); kps.append(f"keypoints/kp-f{i}.pt")
    oa = spec_dir = None
    for sub, src in (("oa", a.oa_from), ("specialists", a.spec_from)):
        if src:
            shutil.copytree(Path(src) / sub, out / sub, dirs_exist_ok=True)
            for f in sorted((out / sub).glob("*.pt")):
                files[f"{sub}/{f.name}"] = sha(f)
            oa, spec_dir = (sub, spec_dir) if sub == "oa" else (oa, sub)
    if a.teacher and not cfg["teachers"][a.teacher]["specialists"]:
        spec_dir = None
    studies = list(np.loadtxt(LABELS / "soft.csv", delimiter=",", skiprows=1, usecols=0, dtype=str))
    prior = labels(studies, "soft").mean(0).round(6).tolist()   # fallback prediction for a study that cannot be read
    man = dict(schema="rsna-knee-models-v1", target_columns=list(TARGETS), prior=prior, group_weights={"planes": 1.0},
               keypoint_models=kps, members=members, oa=oa, specialists=spec_dir, recipe=recipe, files=files)
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(f"{len(members)} tile members, {len(files)} files -> {out}")


if __name__ == "__main__":
    main()
