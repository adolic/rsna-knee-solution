"""Kaggle submission notebook (offline): the released dataset holds this repository's package (code/) and the model pack
(models/). Attach the dataset and the competition data to a GPU notebook (T4 x2), internet off, and run this script.
Expected run time on the hidden test set: about 6-6.5 hours (55 tile models with tau mix and octant TTA, plus specialists).
Output: /kaggle/working/submission.csv"""
import shutil
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
# attached inputs live at /kaggle/input/<slug> or, in the newer layout, /kaggle/input/{datasets,competitions}/<owner>/<slug>
DATASET = next(m.parent.parent for pat in ("*/models/manifest.json", "*/*/models/manifest.json", "*/*/*/models/manifest.json")
               for m in INPUT.glob(pat))
COMP = next(t.parent for pat in ("*/test.csv", "*/*/test.csv", "*/*/*/test.csv") for t in INPUT.glob(pat)
            if "rsna-knee-abnormality-detection" in str(t))
sys.path.insert(0, str(DATASET / "code"))
from rsna_knee import predict  # noqa: E402

if __name__ == "__main__":
    predict.main(["--models", str(DATASET / "models"), "--input", str(COMP), "--split", "test", "--out", "/kaggle/working/out"])
    shutil.copy2("/kaggle/working/out/predictions.csv", "/kaggle/working/submission.csv")
    print("wrote /kaggle/working/submission.csv")
