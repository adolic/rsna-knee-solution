"""The keypoint localizer: 5 fold models (scanner-grouped split of the 417 annotated knees, labels/keypoint-targets.csv).
Needs the native224 volumes of the RSNA studies: python scripts/20_build_cache.py --input DATA/raw/rsna --out DATA/cache/rsna --native-only
usage: python scripts/10_train_keypoints.py [--folds 0 1 2 3 4] [--smoke]   -> DATA/runs/kp-f<k>/last.pt"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from rsna_knee.paths import CACHE, RUNS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4]); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    for k in a.folds:
        out = (RUNS / "smoke" if a.smoke else RUNS) / f"kp-f{k}"
        if (out / "result.json").exists():
            print(f"skip {out.name} (done)"); continue
        cmd = [sys.executable, "-m", "rsna_knee.train_keypoints", "--fold", str(k), "--cache", str(CACHE / "rsna"), "--out", str(out)]
        cmd += ["--epochs", "5", "--limit", "256"] if a.smoke else []
        print(" ".join(cmd), flush=True); subprocess.run(cmd, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
