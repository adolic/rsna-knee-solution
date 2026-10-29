"""Repository and data locations, read from SETTINGS.json at the repository root (the only place paths are set).
Everything large lives under DATA (SETTINGS DATA_DIR, default <repo>/data; the env RSNA_KNEE_DATA overrides it):

  DATA/raw/rsna/            the Kaggle competition download (train.csv, train_series.csv, train_series/<study>/<series>/*.dcm, ...)
  DATA/raw/oai/             OAI baseline (V00) MRI image archives + MOAKS kMRI files (NIMH Data Archive, data use agreement)
  DATA/raw/fastmri/         fastMRI knee DICOM tar archives (NYU fastMRI data sharing agreement)
  DATA/pretrained/          ImageNet / RadImageNet backbone weights (scripts/00_download_pretrained.py)
  DATA/cache/               preprocessed caches (native224, keypoints, knee frames, knee512 canvases, crops)
  DATA/runs/                training runs (one directory per run: config.json, best.pt, predictions)
  DATA/models/              an inference model pack (scripts/90_pack_models.py, or the released Kaggle dataset)
Labels and small tables ship with the repository (LABELS)."""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LABELS = REPO / "labels"
CONFIGS = REPO / "configs"
_DEFAULTS = {"DATA_DIR": "data", "RAW_RSNA_DIR": "raw/rsna", "RAW_OAI_DIR": "raw/oai", "RAW_OAI_DICOM_DIR": "raw/oai-dicom",
             "RAW_FASTMRI_DIR": "raw/fastmri", "RAW_FASTMRI_DICOM_DIR": "raw/fastmri-dicom", "PRETRAINED_DIR": "pretrained",
             "CACHE_DIR": "cache", "CHECKPOINT_DIR": "runs", "TEACHER_DIR": "teachers", "PSEUDO_LABEL_DIR": "pseudo",
             "MOAKS_DIR": "moaks", "SPECIALIST_PACK_DIR": "specialist-pack", "MODEL_DIR": "models", "SUBMISSION_DIR": "predictions"}
_settings_file = REPO / "SETTINGS.json"   # absent only in stripped copies (e.g. an old Kaggle dataset version): defaults apply
SETTINGS = {**_DEFAULTS, **(json.loads(_settings_file.read_text()) if _settings_file.is_file() else {})}
DATA = Path(os.environ.get("RSNA_KNEE_DATA") or (REPO / SETTINGS["DATA_DIR"])).resolve()
_d = lambda key: (DATA / SETTINGS[key]).resolve()  # noqa: E731  (absolute entries are kept as they are)
RAW_RSNA = _d("RAW_RSNA_DIR")
RAW_OAI = _d("RAW_OAI_DIR")
RAW_OAI_DICOM = _d("RAW_OAI_DICOM_DIR")
RAW_FASTMRI = _d("RAW_FASTMRI_DIR")
RAW_FASTMRI_DICOM = _d("RAW_FASTMRI_DICOM_DIR")
PRETRAINED = _d("PRETRAINED_DIR")
CACHE = _d("CACHE_DIR")
RUNS = _d("CHECKPOINT_DIR")
TEACHERS = _d("TEACHER_DIR")
PSEUDO = _d("PSEUDO_LABEL_DIR")
MOAKS = _d("MOAKS_DIR")
SPECIALIST_PACK = _d("SPECIALIST_PACK_DIR")
MODELS = _d("MODEL_DIR")
SUBMISSION = _d("SUBMISSION_DIR")
TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
