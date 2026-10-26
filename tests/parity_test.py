"""Parity: the package reproduces the submitted Kaggle kernel's probabilities on 12 RSNA training studies (fold-1 validation
studies; tests/reference/kernel-predictions-12.csv = the kernel's own output on them, FP16, same models).
Needs the competition data (DATA/raw/rsna) and the released model pack (DATA/models). Passes at |difference| <= 1e-6
(observed: 0.0, bit-identical, on an RTX 4090 with torch 2.13; other GPUs / library versions may differ in the last digits).
usage: python tests/parity_test.py [--devices cuda:0]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from rsna_knee import predict  # noqa: E402
from rsna_knee.paths import DATA, MODELS, RAW_RSNA  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--devices", default="cuda:0"); ap.add_argument("--tolerance", type=float, default=1e-6)
    a = ap.parse_args()
    ref = pd.read_csv(REPO / "tests/reference/kernel-predictions-12.csv", index_col=0)
    out = DATA / "tests" / "parity"
    predict.main(["--models", str(MODELS), "--input", str(RAW_RSNA), "--split", "train", "--studies", *ref.index,
                  "--out", str(out), "--devices", a.devices, "--workers", "1"])
    got = pd.read_csv(out / "predictions.csv", index_col=0).loc[ref.index, ref.columns]
    d = float(np.abs(got.to_numpy() - ref.to_numpy()).max())
    print(f"max |probability difference| vs the Kaggle kernel: {d:.3g} over {ref.size} values")
    sys.exit(0 if d <= a.tolerance else 1)


if __name__ == "__main__":
    main()
