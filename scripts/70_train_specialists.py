"""Crop specialists (configs/specialists.json): caches, training and their part of the model pack.

  --stage caches   crop caches from the knee512 caches (DATA/cache/rsna, DATA/cache/oai) -> DATA/cache/crops-*
  --stage menpre   meniscus OAI pretraining on DATA/pseudo/round1.csv (once, fold-1 validation)
  --stage train    r50oa, men, acl, oafam x 5 folds -> DATA/runs/spec-<name>-f<k>
  --stage pack     DATA/specialist-pack/{oa,specialists}: checkpoints + manifests (crop configs, rows and view ids as the
                   trainer assigned them) for scripts/90_pack_models.py --oa-from / --spec-from
usage: python scripts/70_train_specialists.py --stage caches [--studies ...] | --stage train [--runs acl] [--folds 0 1 2 3 4] [--smoke]"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from rsna_knee.paths import CACHE, CONFIGS, DATA, PSEUDO, RAW_OAI, RAW_OAI_DICOM, RAW_RSNA, RUNS, SPECIALIST_PACK  # noqa: E402

SPEC = json.loads((CONFIGS / "specialists.json").read_text())
SPEC_TARGET_RUNS = {"men": "men", "acl": "acl", "oafam": "oafam"}   # pack names of the canvas-crop specialists


def build_caches(names, studies, workers):
    from rsna_knee import crop_cache
    for name in names:
        c = SPEC["caches"][name]
        src = {"rsna": (RAW_RSNA, "train"), "oai": (RAW_OAI_DICOM, "oai")}[c["from"]]
        crop_cache.build(CONFIGS / "crops" / f"{c['config']}.json", CACHE / c["from"], CACHE / name, c["size"], source=c["source"],
                         dicom_folder=src[0], split=src[1], studies=studies, workers=workers, views=c.get("views"))


def run_args(name, fold, out, smoke):
    r = SPEC["runs"][name]; a = ["--out", str(out), "--fold", str(fold)]
    for k, v in r.items():
        if k in ("once", "train_labels", "init_backbone"):
            continue
        if k == "combo":
            v = str(CONFIGS / v)
        if isinstance(v, bool):
            a += [f"--{k.replace('_', '-')}"] if v else []
        elif isinstance(v, list):
            a += [f"--{k.replace('_', '-')}", *map(str, v)]
        else:
            a += [f"--{k.replace('_', '-')}", str(v)]
    if "train_labels" in r:
        a += ["--train-labels-csv", str(PSEUDO / f"{r['train_labels'].replace('pseudo-', '')}.csv")]
    if "init_backbone" in r:
        a += ["--init-backbone", str((RUNS / "smoke" if smoke else RUNS) / f"spec-{r['init_backbone']}" / "best.pt")]
    if smoke:
        a += ["--epochs", "6", "--limit-train", "24", "--limit-val", "40", "--workers", "2"]
    return a


def view_ids(rows_cfg):
    """Rows as the trainer numbers views: combo -> global (cache, view) ids in row / part order; plain -> config view order."""
    gviews, out = [], []
    parts = lambda r: [dict(cache=r["cache"], views=r["views"]), *r.get("extra", [])]
    for r in rows_cfg:
        for q in parts(r):
            for v in q["views"]:
                if (q["cache"], v) not in gviews:
                    gviews.append((q["cache"], v))
    heads = list(dict.fromkeys(r.get("head", "shared") for r in rows_cfg))
    for r in rows_cfg:
        ps = [dict(config=SPEC["caches"][q["cache"]]["config"].replace("crop-config-", ""), views=q["views"],
                   view_ids=[gviews.index((q["cache"], v)) for v in q["views"]]) for q in parts(r)]
        row = dict(target=r["target"], head=heads.index(r.get("head", "shared")))
        row.update(dict(views=ps[0]["views"], view_ids=ps[0]["view_ids"]) if len(ps) == 1 else dict(parts=ps))
        out.append(row)
    return out


def pack(folds, runs=RUNS):
    import shutil
    dst = SPECIALIST_PACK
    for d in ("oa", "specialists"):
        (dst / d).mkdir(parents=True, exist_ok=True)
    cfg = lambda n: json.loads((CONFIGS / "crops" / f"crop-config-{n}.json").read_text())
    r = SPEC["runs"]["r50oa"]; c = SPEC["caches"][r["cache"]]
    for k in folds:
        shutil.copy2(runs / f"spec-r50oa-f{k}" / "best.pt", dst / "oa" / f"r50-f{k}.pt")
    (dst / "oa" / "oa-manifest.json").write_text(json.dumps(dict(schema="rsna-knee-oa-pf-v1", crop_config=cfg(c["config"].replace("crop-config-", "")),
        size=c["size"], margin_mm=2.0, views=r["views"], checkpoints=[f"r50-f{k}.pt" for k in folds]), indent=1) + "\n")
    specs = []
    for name in ("men", "acl", "oafam"):
        rr = SPEC["runs"][name]
        for k in folds:
            shutil.copy2(runs / f"spec-{name}-f{k}" / "best.pt", dst / "specialists" / f"{name}-f{k}.pt")
        if "combo" in rr:
            rows_cfg = json.loads((CONFIGS / rr["combo"]).read_text())["rows"]
            used = json.loads((CONFIGS / "submission_recipe.json").read_text())["spec"]["weights"]   # rows the blend uses
            rows = [x for x in view_ids(rows_cfg) if x["target"] in used]
            confs = sorted({SPEC["caches"][q]["config"].replace("crop-config-", "") for r_ in rows_cfg for q in [r_["cache"], *[e["cache"] for e in r_.get("extra", [])]]})
            size = SPEC["caches"][rows_cfg[0]["cache"]]["size"]
            entry = dict(name=name, checkpoints=[f"{name}-f{k}.pt" for k in folds], size=size, rows=rows)
            if any("parts" in x for x in rows):
                entry["crop_configs"] = {n: cfg(n) for n in confs}
            else:
                entry["crop_config"] = cfg(confs[0])
        else:
            cc = cfg(SPEC["caches"][rr["cache"]]["config"].replace("crop-config-", "")); vlist = list(cc["views"])
            entry = dict(name=name, checkpoints=[f"{name}-f{k}.pt" for k in folds], size=SPEC["caches"][rr["cache"]]["size"], crop_config=cc,
                         rows=[dict(target=t, head=i, views=rr["views"], view_ids=[vlist.index(v) for v in rr["views"]]) for i, t in enumerate(rr["targets"])])
        specs.append(entry)
    (dst / "specialists" / "spec-manifest.json").write_text(json.dumps(dict(schema="rsna-knee-spec-v1", specialists=specs), indent=1) + "\n")
    print(f"specialist pack -> {dst} (blend weights and offline statistics live in the recipe, configs/submission_recipe.json)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("caches", "menpre", "train", "pack"))
    ap.add_argument("--caches", nargs="*"); ap.add_argument("--runs", nargs="*"); ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--studies", nargs="*"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.stage == "caches":
        return build_caches(a.caches or list(SPEC["caches"]), a.studies, a.workers)
    if a.stage == "pack":
        return pack(a.folds, RUNS / "smoke" if a.smoke else RUNS)
    names = ["menpre"] if a.stage == "menpre" else (a.runs or ["r50oa", "men", "acl", "oafam"])
    root = RUNS / "smoke" if a.smoke else RUNS
    for name in names:
        for fold in ([1] if SPEC["runs"][name].get("once") else a.folds):
            out = root / (f"spec-{name}" if SPEC["runs"][name].get("once") else f"spec-{name}-f{fold}")
            if (out / "result.json").exists():
                print(f"skip {out.name} (done)"); continue
            cmd = [sys.executable, "-m", "rsna_knee.train_crops", *run_args(name, fold, out, a.smoke)]
            print(" ".join(cmd), flush=True)
            if not a.dry_run:
                subprocess.run(cmd, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
