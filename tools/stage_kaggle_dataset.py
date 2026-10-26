"""Stage the public Kaggle dataset: code/ (the rsna_knee package + configs + labels) and models/ (a model pack), plus
dataset-metadata.json for `kaggle datasets create -p <out>` (run that yourself after reviewing the folder).
usage: python tools/stage_kaggle_dataset.py --models DATA/models --out /path/to/staging --id <kaggle-user>/rsna-knee-6th-place-models"""
import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--out", required=True); ap.add_argument("--id", required=True)
    a = ap.parse_args()
    out = Path(a.out); (out / "code").mkdir(parents=True, exist_ok=True)
    for d in ("rsna_knee", "configs", "labels"):
        shutil.copytree(REPO / d, out / "code" / d, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(REPO / "kaggle" / "inference_notebook.py", out / "code" / "inference_notebook.py")
    shutil.copytree(a.models, out / "models", dirs_exist_ok=True)
    (out / "dataset-metadata.json").write_text(json.dumps(dict(title="RSNA knee 6th place: code and models", id=a.id,
        licenses=[dict(name="CC-BY-SA-4.0")]), indent=1) + "\n")
    print(f"staged {out} ({sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 2**30:.1f} GiB)")


if __name__ == "__main__":
    main()
