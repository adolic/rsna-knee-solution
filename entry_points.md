# Entry points

All paths come from `SETTINGS.json` (through `rsna_knee/paths.py`). Run from the repository root with the packages in
`requirements.txt` installed. `DATA` below means `SETTINGS.json` `DATA_DIR` (default `./data`).

## Predict with the released models (the submitted solution)

1. Download the model pack (public Kaggle dataset) into `MODEL_DIR`:
   `kaggle datasets download centralyds/rsna-knee-6th-place-models -p DATA --unzip`
2. Download the competition data into `RAW_RSNA_DIR`: `bash scripts/01_download_rsna.sh`
3. Predict:
   * the hidden-test layout (test.csv, test_series.csv, test_series/): `python -m rsna_knee.predict --input DATA/raw/rsna --split test`
   * one study folder (`<study>/<series>/*.dcm`): `python -m rsna_knee.predict --input /path/to/study`

   Reads DICOM from `--input` and models from `MODEL_DIR`; writes `predictions.csv` (the submission format),
   `receipts.jsonl` and `verification.json` to `SUBMISSION_DIR`.
4. On Kaggle (offline): attach the competition and the model dataset and run `kaggle/inference_notebook.py`.

## Train from scratch (about 250 GPU-hours on one RTX 4090)

| Step | Command | Reads | Writes |
|---|---|---|---|
| 0 | `python scripts/00_download_pretrained.py` | Hugging Face (pinned revisions) | `PRETRAINED_DIR` |
| 1 | `bash scripts/01_download_rsna.sh` | Kaggle API | `RAW_RSNA_DIR` |
| 2 | `python scripts/11_prepare_oai.py --image03 image03.txt --archives DIR --out DATA/raw/oai-dicom` | OAI archives (NIMH Data Archive) | `RAW_OAI_DICOM_DIR` |
| 3 | `python scripts/12_prepare_fastmri.py --archives knee_DICOMs_batch1.tar.xz knee_DICOMs_batch2.tar.xz --out DATA/raw/fastmri-dicom` | fastMRI archives (NYU) | `RAW_FASTMRI_DICOM_DIR` |
| 4 | `python scripts/20_build_cache.py --input DATA/raw/rsna --out DATA/cache/rsna --native-only` | `RAW_RSNA_DIR` | `CACHE_DIR` |
| 5 | `python scripts/10_train_keypoints.py` | `CACHE_DIR`, `labels/keypoint-targets.csv` | `CHECKPOINT_DIR/kp-f*` |
| 6 | `python scripts/20_build_cache.py --input DATA/raw/rsna --out DATA/cache/rsna --keypoints DATA/runs/kp-f*/last.pt` (OAI: `--split oai --axial-every 2`; fastMRI: `--split fastmri`) | raw DICOM, keypoint models | `CACHE_DIR` |
| 7 | `python scripts/70_train_specialists.py --stage caches`, then `--stage train --runs r50oa acl oafam` | `CACHE_DIR`, `labels/` | `CHECKPOINT_DIR/spec-*` |
| 8 | `python scripts/40_train_cells.py --stage A` | `CACHE_DIR`, `labels/` | `CHECKPOINT_DIR` |
| 9 | `python scripts/50_pseudo_labels.py --round 1 --sources DATA/raw/oai-dicom:oai DATA/raw/fastmri-dicom:fastmri` | external DICOM, round-1 teacher | `PSEUDO_LABEL_DIR`, `TEACHER_DIR` |
| 10 | `python scripts/40_train_cells.py --stage P1`, `--stage B`; `python scripts/70_train_specialists.py --stage menpre`, `--stage train --runs men` | caches, pseudo labels | `CHECKPOINT_DIR` |
| 11 | `python scripts/50_pseudo_labels.py --round 2 ...`, `python scripts/40_train_cells.py --stage P2`, `--stage C` | caches, round-2 teacher | `PSEUDO_LABEL_DIR`, `CHECKPOINT_DIR` |
| 12 | `python scripts/70_train_specialists.py --stage pack`, then `python scripts/90_pack_models.py --out DATA/models --oa-from DATA/specialist-pack --spec-from DATA/specialist-pack` | `CHECKPOINT_DIR` | `SPECIALIST_PACK_DIR`, `MODEL_DIR` |

Optional: `python scripts/60_moaks_label_models.py` (MOAKS label models: targets, train, zero-shot, blend; reads the NDA
file `oai_kmrisemiquant01.txt`, writes `MOAKS_DIR`). Their effect is already in `labels/`.

## Checks

* `python tests/smoke_test.py --data DATA`: every stage on a few studies, in minutes.
* `python tests/parity_test.py`: inference against the submitted Kaggle kernel's own output (12 studies, identical).
