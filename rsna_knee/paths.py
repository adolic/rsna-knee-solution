"""Repository and data locations. Everything large lives under DATA (env RSNA_KNEE_DATA, default <repo>/data):

  DATA/raw/rsna/            the Kaggle competition download (train.csv, train_series.csv, train_series/<study>/<series>/*.dcm, ...)
  DATA/raw/oai/             OAI baseline (V00) MRI image archives + MOAKS kMRI files (NIMH Data Archive, data use agreement)
  DATA/raw/fastmri/         fastMRI knee DICOM tar archives (NYU fastMRI data sharing agreement)
  DATA/pretrained/          ImageNet / RadImageNet backbone weights (scripts/00_download_pretrained.py)
  DATA/cache/               preprocessed caches (native224, keypoints, knee frames, knee512 canvases, crops)
  DATA/runs/                training runs (one directory per run: config.json, best.pt, predictions)
  DATA/models/              an inference model pack (scripts/90_pack_models.py, or the released Kaggle dataset)
Labels and small tables ship with the repository (LABELS)."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LABELS = REPO / "labels"
CONFIGS = REPO / "configs"
DATA = Path(os.environ.get("RSNA_KNEE_DATA", REPO / "data")).resolve()
RAW_RSNA = DATA / "raw" / "rsna"
RAW_OAI = DATA / "raw" / "oai"
RAW_FASTMRI = DATA / "raw" / "fastmri"
PRETRAINED = DATA / "pretrained"
CACHE = DATA / "cache"
RUNS = DATA / "runs"
MODELS = DATA / "models"
TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
