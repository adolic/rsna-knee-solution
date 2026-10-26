"""Training labels and splits (labels/, shipped with the repository).

  soft.csv      final soft labels (one row per RSNA training study, 12 targets in [0, 1]); training targets
  hard.csv      hard companions (soft >= 0.5, the 58 expert-labelled "Gold" studies keep their expert values); validation
  weights.csv   per-cell loss weights (2 on the Gold studies, 1 elsewhere)
  folds.csv     5-fold patient-grouped split (fold = the validation fold of each study), is_gold
  two-knee-sides.csv   the knee whose findings the study label describes, for the few studies that scan both knees
How the labels were made: docs/LABELS.md."""
from functools import lru_cache

import numpy as np
import pandas as pd

from .paths import LABELS, TARGETS

TARGET_COLUMNS = TARGETS


@lru_cache(maxsize=None)
def _table(name):
    return pd.read_csv(LABELS / name, index_col=0)


def split(fold, studies):
    """(train, val) study lists of one fold, restricted to `studies`."""
    f = _table("folds.csv")["fold"].reindex(list(studies))
    if f.isna().any():
        raise ValueError("studies outside labels/folds.csv")
    return sorted(f.index[f != fold]), sorted(f.index[f == fold])


def labels(studies, kind, targets=TARGET_COLUMNS):
    """kind: soft | hard | weights -> float32 [n, len(targets)]."""
    t = _table({"soft": "soft.csv", "hard": "hard.csv", "weights": "weights.csv"}[kind]).reindex(list(studies))[list(targets)]
    if t.isna().any().any():
        raise ValueError(f"{kind} labels miss studies")
    return t.to_numpy(np.float32)


@lru_cache(maxsize=1)
def _sides():
    return pd.read_csv(LABELS / "two-knee-sides.csv").set_index("study_uid").knee.to_dict()


def side_for(study, knees):
    """The knee ('one', 'L' or 'R') whose inputs train the study-level label: 'one' unless the study scans both knees,
    then the knee named in two-knee-sides.csv (default left)."""
    knees = set(knees)
    if "one" in knees:
        return "one"
    want = _sides().get(study, "L")
    return want if want in knees else ("L" if "L" in knees else sorted(knees)[0])
