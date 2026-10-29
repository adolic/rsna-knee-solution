# RSNA Knee Abnormality Detection — 6th place solution

12 findings per knee MRI study (ACL, MCL, medial / lateral meniscus, medial / lateral / patellofemoral OA, effusion,
synovitis, Baker's cyst, contusion, fracture) from raw DICOM. 6th place on the private leaderboard (6,382 teams; public 0.969).

**In one paragraph.** A keypoint localizer finds five landmarks of the knee in every series and fuses them into a 3D knee
frame. Every series is resampled to 0.3125 mm/px on a 512 px canvas around the joint. Models see bags of 70 mm tiles (3
adjacent slices each) sampled around anatomical anchors and pool them per finding with attention. The final ensemble has
55 tile models: 11 recipes × 5 folds of ConvNeXt-T, CAFormer-S18 and ReXNet-R300, per plane and over all planes. Seven of
the recipes start from pseudo-label pretraining on ~18,500 external knees (OAI + fastMRI), in two noisy-student rounds.
Small crop specialists (meniscus, ACL, OA family, a ResNet50 OA model) are blended in per finding. The training labels were
extracted from the radiology reports and refined with image models trained only on expert MOAKS reads of OAI knees.

| | |
|---|---|
| Inference | `rsna_knee/predict.py`: raw DICOM → 12 probabilities per study, bit-identical to the submitted Kaggle kernel |
| Released models | one model pack (4.1 GB): 55 tile models, 5 keypoint models, 15 crop specialists, 5 ResNet50 OA models |
| Labels | `labels/` — final soft labels, hard labels, loss weights, folds, keypoint annotations, proxy-metric strata |
| Training | `scripts/` — every stage from download to the model pack, configured by `configs/` |
| Docs | [docs/PIPELINE.md](docs/PIPELINE.md) (method), [docs/LABELS.md](docs/LABELS.md) (labels), [docs/VERIFICATION.md](docs/VERIFICATION.md) (what was tested) |

## Quick start: inference with the released models

```bash
pip install -r requirements.txt
export RSNA_KNEE_DATA=/path/to/data           # large files live here (default: ./data)
# the released model pack: Kaggle dataset centralyds/rsna-knee-6th-place-models (folder models/)
kaggle datasets download centralyds/rsna-knee-6th-place-models -p $RSNA_KNEE_DATA --unzip   # -> $RSNA_KNEE_DATA/models
python -m rsna_knee.predict --input /path/to/rsna-knee-abnormality-detection --split test --out predictions/
python -m rsna_knee.predict --input /path/to/one_study_folder --out predictions/      # <study>/<series>/*.dcm
```

* A Kaggle-style folder (`test.csv`, `test_series.csv` with `Fluid_Sensitive`, `test_series/<study>/<series>/*.dcm`) is
  scored like the competition. A single study folder works too. Its per-series fluid-sensitive flags then come from
  `--series-csv` or, as a fallback, from the DICOM series descriptions.
* GPU required (FP16). On one RTX 4090 a study takes about 13–17 s; the Kaggle hidden test ran in about 6–6.5 h on 2 × T4.
* Outputs: `predictions.csv`, `receipts.jsonl` (per-study timing, knees, fallbacks) and `verification.json`.
* On Kaggle (offline): attach the released dataset and run `kaggle/inference_notebook.py`.

## Environment

| | |
|---|---|
| Hardware (training, reference runs) | Intel Core i7-8700K (6 cores / 12 threads), 64 GB RAM, 1 × NVIDIA RTX 4090 24 GB. During the competition some cells were also trained on rented RTX 3090 / RTX 4090 machines (vast.ai) with the same code. |
| Hardware (submission) | Kaggle GPU notebook, 2 × T4, about 6–6.5 h for the hidden test |
| OS | Arch Linux, kernel 7.2; NVIDIA driver 615.71, CUDA 13.0, cuDNN 9.20 (also tested on the Kaggle GPU image) |
| Software | Python 3.11.15; exact package versions in `requirements.txt` (`pip install -r requirements.txt`) |
| Paths | `SETTINGS.json` is the only place input/output paths are set (`DATA_DIR`, raw data, caches, `CHECKPOINT_DIR`, `MODEL_DIR`, `SUBMISSION_DIR`, ...); the environment variable `RSNA_KNEE_DATA` overrides `DATA_DIR` |
| Commands | [entry_points.md](entry_points.md) lists every command, what it reads and what it writes |

Side effects and assumptions: nothing under the raw-data folders is ever modified; every stage writes only to its own
output folder and skips finished runs, so a stage can be resumed by re-running it. Training needs the external OAI and
fastMRI data (each under its own data use agreement, see below); inference needs only the competition data and the
released model pack.

## Reproducing the training

Each stage is resumable and skips finished runs. Times are for one RTX 4090 (24 GB); stages A and B parallelise over GPUs.

| # | Stage | Command | Time |
|---|---|---|---|
| 0 | backbone weights (pinned Hugging Face revisions, sha256-checked) | `python scripts/00_download_pretrained.py` | minutes |
| 1 | RSNA competition data (Kaggle API) | `bash scripts/01_download_rsna.sh` | download |
| 2 | OAI baseline MRI → study folders (NIMH Data Archive, data use certification) | `python scripts/11_prepare_oai.py --image03 image03.txt --archives DIR --out DATA/raw/oai-dicom` | hours |
| 3 | fastMRI knee DICOMs → study folders (NYU fastMRI agreement) | `python scripts/12_prepare_fastmri.py --archives knee_DICOMs_batch1.tar.xz knee_DICOMs_batch2.tar.xz --out DATA/raw/fastmri-dicom` | hours |
| 4 | native224 volumes (keypoint training input) | `python scripts/20_build_cache.py --input DATA/raw/rsna --out DATA/cache/rsna --native-only` | ~1 h |
| 5 | keypoint localizer, 5 folds | `python scripts/10_train_keypoints.py` | ~1 h |
| 6 | caches: knee frames + knee512 canvases (RSNA, OAI, fastMRI) | `python scripts/20_build_cache.py --input DATA/raw/rsna --out DATA/cache/rsna --keypoints DATA/runs/kp-f*/last.pt` (OAI: `--split oai --axial-every 2`; fastMRI: `--split fastmri`) | ~10 h |
| 7 | crop specialists: caches, ResNet50 OA, ACL, OA family | `python scripts/70_train_specialists.py --stage caches` then `--stage train --runs r50oa acl oafam` | ~15 h |
| 8 | stage A: 4 plane cells × 5 folds (these are also the round-1 teachers) | `python scripts/40_train_cells.py --stage A` | ~20 h |
| 9 | round-1 pseudo labels on OAI + fastMRI | `python scripts/50_pseudo_labels.py --round 1 --sources DATA/raw/oai-dicom:oai DATA/raw/fastmri-dicom:fastmri` | ~25 h |
| 10 | P1 pretraining (5 recipes) → stage B fine-tuning (6 cells × 5 folds); meniscus specialist | `scripts/40_train_cells.py --stage P1`, `--stage B`; `scripts/70_train_specialists.py --stage menpre`, `--stage train --runs men` | ~90 h |
| 11 | round-2 pseudo labels → P2 pretraining → stage C (ReXNet) | `scripts/50_pseudo_labels.py --round 2 ...`, `scripts/40_train_cells.py --stage P2`, `--stage C` | ~70 h |
| 12 | model pack | `python scripts/70_train_specialists.py --stage pack` then `python scripts/90_pack_models.py --out DATA/models --oa-from DATA/specialist-pack --spec-from DATA/specialist-pack` | minutes |

The round-1 pseudo labels use the ResNet50 OA specialist (stage 7). Round 2 uses the full stage A + B ensemble with every
specialist (stages 7 and 10). Total ≈ 250 GPU-hours.

The MOAKS label models (`scripts/60_moaks_label_models.py`: targets → train → zero-shot → blend) document how the shipped
labels were refined. The labels in `labels/` already include their effect, so retraining does not need them.

`python tests/smoke_test.py --data DATA` runs every stage on a few studies in minutes. `python tests/parity_test.py`
checks inference against the Kaggle kernel's own output.

## Data and licences

* **Licence:** the code and the released model weights are © 2026 CentralyDS, licensed under
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (`LICENSE`), the licence the competition rules name for
  winning solutions. You may share and adapt them for non-commercial purposes with attribution ("rsna-knee-solution by
  CentralyDS").
* **Not covered by that licence:**
  * `labels/` (see below);
  * the pretrained backbones the models were fine-tuned from (ConvNeXt-T, CAFormer-S18, ReXNet-R300, EfficientNet-B0:
    Apache-2.0; RadImageNet ResNet50: MIT);
  * the Python dependencies.

  Their terms and the citations OAI and fastMRI require are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
* **Research use only:** the models are not a medical device. Do not use them for diagnosis or patient care.
* **RSNA competition data:** © RSNA. Download it yourself from Kaggle under the competition rules and the RSNA MIRA
  licence. `labels/` holds labels derived from it:
  * soft labels from the reports;
  * expert labels of the 58 "Gold" studies as given in `train.csv`;
  * keypoint annotations;
  * scanner strata.

  RSNA keeps all rights in the data. These files are not licensed by us, and their use is subject to the same terms.
* **OAI:** NIMH Data Archive (Collection 2343), data use certification required. `rsna_knee/moaks.py` reads the NDA
  structure `oai_kmrisemiquant01` (MOAKS reads) directly. Nothing from OAI is redistributed here.
* **fastMRI:** NYU fastMRI data sharing agreement. Nothing from fastMRI is redistributed here.
* **External data in the models:**
  * 30 tile models were pretrained with pseudo labels on OAI + fastMRI images;
  * 10 models (axOAI2 and the meniscus specialist) were pretrained on OAI images only;
  * every model used labels derived from the competition data.

  The table is in THIRD_PARTY_NOTICES.md. Check that your use complies with those terms.

## What differs from the submitted models

The released model pack *is* the submission: the same checkpoints in the same member order, and its outputs match the
kernel exactly. A retrain from this repository reproduces the method; it does not reproduce those exact weights:

* **Labels:** only the final label version is shipped. The submitted cells were trained during the competition on
  successive label versions: most cells on v7.16, `allfm` on v7.18, `allfmold` on v7.15, the specialists on v7.14 or
  v7.10, and the pretraining runs validated on v7.15. The changes between versions are listed in docs/LABELS.md.
  `allfmold` is the `allfm` recipe on an older label version. Retrained on the final labels it becomes a second seed.
* **Preprocessing:** the original training caches ran the keypoint models in BF16 with one random stream across all
  studies. This repository builds caches with the inference pipeline itself (FP16, a fixed seed per study), so training
  and inference inputs match by construction. Knee frames differ from the original caches by under 1 mm.
* **Specialist crops:** the ResNet50 OA and OA-family crops were originally cut from raw DICOM by a separate builder.
  Here the specialists are cut from the canvases (as inference does), and the ResNet50 crops from the decoded series
  (as inference does).
* **Run-to-run choices:** fold 0 of the OA-family model froze its backbone for 3 epochs, folds 1–4 did not; the config
  uses the folds 1–4 setting. Pretraining runs used one GPU, except the ReXNet round-2 pretraining, which was also run
  with DDP (supported, `--gpus N`).

## Repository map

```
rsna_knee/          package: dicom.py, pipeline.py (preprocessing), tiles.py (sampling), tile_model.py, crop_model.py,
                    keypoint_model.py, cache.py, crop_cache.py, tile_data.py, crop_data.py, train_tiles.py, train_crops.py,
                    train_keypoints.py, predict.py, labels.py, proxy.py (selection metric), moaks.py
scripts/            00-90: data, caches, training stages, pseudo labels, packing
configs/            cells.json (every tile recipe, stages, teachers), specialists.json, submission_recipe.json (inference),
                    crops/ (crop view configs), combos/ (specialist row definitions)
labels/             soft / hard / weights / folds / keypoint targets / proxy strata / two-knee sides
kaggle/             offline Kaggle notebook
tests/              parity_test.py (vs the kernel), smoke_test.py (every stage on a few studies)
tools/              release helpers (pack assembly from the Kaggle bundles, crop-config scrub, dataset staging)
SETTINGS.json       all input/output paths; entry_points.md: every command; directory_structure.txt
LICENSE, LICENSES/  CC BY-NC 4.0 (code + weights); THIRD_PARTY_NOTICES.md: third-party licences, data terms, citations
```
