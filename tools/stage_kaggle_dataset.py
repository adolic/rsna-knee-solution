"""Stage the public Kaggle dataset: code/ (the rsna_knee package + configs + labels) and models/ (a model pack), the licence
files (LICENSE, THIRD_PARTY_NOTICES.md, LICENSES/) and dataset-metadata.json for `kaggle datasets create -p <out>` (run that
yourself after reviewing the folder). --no-labels leaves labels/ out (RSNA MIRA licence: no redistribution without RSNA's
written permission); inference does not need it.
usage: python tools/stage_kaggle_dataset.py --models DATA/models --out /path/to/staging --id <kaggle-user>/rsna-knee-6th-place-models
       [--no-labels] [--description FILE.md]"""
import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LICENSE_ID = "CC-BY-NC-4.0"   # the competition's winner licence (rules section 2.5.a.2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--out", required=True); ap.add_argument("--id", required=True)
    ap.add_argument("--no-labels", action="store_true"); ap.add_argument("--description")
    a = ap.parse_args()
    out = Path(a.out); (out / "code").mkdir(parents=True, exist_ok=True)
    if a.no_labels and (out / "code" / "labels").exists():
        raise SystemExit(f"{out / 'code' / 'labels'} exists from an earlier staging: move it out of the folder first")
    for d in ("rsna_knee", "configs") + (() if a.no_labels else ("labels",)):
        shutil.copytree(REPO / d, out / "code" / d, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(REPO / "kaggle" / "inference_notebook.py", out / "code" / "inference_notebook.py")
    for f in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(REPO / f, out / f)
    shutil.copytree(REPO / "LICENSES", out / "LICENSES", dirs_exist_ok=True)
    shutil.copytree(a.models, out / "models", dirs_exist_ok=True)
    meta = dict(title="RSNA knee 6th place: code and models", id=a.id, licenses=[dict(name=LICENSE_ID)], keywords=["pre-trained model"])
    if a.description:
        meta["description"] = Path(a.description).read_text()
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    print(f"staged {out} ({sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 2**30:.1f} GiB)")


if __name__ == "__main__":
    main()
