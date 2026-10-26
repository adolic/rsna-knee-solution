"""Kaggle submission notebook (offline): the released dataset holds this repository's package (code/) and the model pack
(models/). Attach the dataset and the competition data to a GPU notebook (T4 x2), internet off, and run this script.
Expected run time on the hidden test set: about 6-6.5 hours (55 tile models with tau mix and octant TTA, plus specialists).
Output: /kaggle/working/submission.csv"""
import shutil
import sys
from pathlib import Path

DATASET = next(p for p in Path("/kaggle/input").glob("*") if (p / "models" / "manifest.json").exists())
COMP = next(p for p in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"), Path("/kaggle/input/rsna-knee-abnormality-detection")]
            if (p / "test.csv").exists())
sys.path.insert(0, str(DATASET / "code"))
from rsna_knee import predict  # noqa: E402

if __name__ == "__main__":
    predict.main(["--models", str(DATASET / "models"), "--input", str(COMP), "--split", "test", "--out", "/kaggle/working/out"])
    shutil.copy2("/kaggle/working/out/predictions.csv", "/kaggle/working/submission.csv")
    print("wrote /kaggle/working/submission.csv")
