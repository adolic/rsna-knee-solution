# What was tested

Training everything from scratch takes about 250 GPU-hours, so every stage was checked separately. Where possible, a stage
was compared with the artefacts of the original competition runs; otherwise it was run end to end on a small subset.
Unless noted: one RTX 4090, torch 2.13, timm 1.0.28.

## Exact reproductions

| Check | Result |
|---|---|
| **Inference vs the submitted Kaggle kernel** (12 RSNA fold-1 studies, raw DICOM, released model pack; `tests/parity_test.py`) | all 650 member × study logit vectors and all 144 final probabilities identical (max \|difference\| 0.0) |
| Pretrained backbones (`scripts/00_download_pretrained.py`: ConvNeXt-T, CAFormer-S18, ReXNet-R300, RadImageNet ResNet50) | 4 / 4 files reproduce the sha256 the models were trained from |
| Selection metric (`rsna_knee/proxy.py`) vs the competition-time implementation | identical value (0.979135…) on a fold of out-of-fold predictions |
| MOAKS targets from the NDA file (`rsna_knee/moaks.py`) vs the label files the MOAKS models were trained on | effusion/synovitis 1,660 knees, OA 2,399, meniscus/Baker's/ACL 2,581: max \|difference\| 0.0, same missing-value pattern |
| MOAKS zero-shot predictor (`60_moaks_label_models.py --stage predict`) with the original effusion/synovitis model | 40 studies, every head: max \|difference\| 5e-7 (CSV rounding) |
| fastMRI preparation (`12_prepare_fastmri.py`, first 3 studies of batch 2) vs the original series selection | 12 / 12 series selected, identical planes and fluid flags |
| OAI selection (`11_prepare_oai.py`, from `image03.txt`) vs the original OAI cache | covers 9,437 / 9,437 cached knees (+3 knees) |
| Specialist manifest rows and view ids (`70_train_specialists.py --stage pack`) vs the released manifest | identical |

## Equivalent (by design not bit-identical) reproductions

| Check | Result |
|---|---|
| Cache builder (`20_build_cache.py`, 4 studies incl. a two-knee study) vs the original training caches | native224: 25 / 25 series identical. Same knees and series. Knee frames within 0.85 mm, because the original caches ran keypoints in BF16 with one random stream, while the builder uses the inference pipeline (FP16, seed per study). Canvases shift accordingly. |
| Crop caches vs the original caches (6 studies) | meniscus crops: 77 / 77 entries, median pixel correlation 0.98. ResNet50 OA crops: 0.987. Differences come from the knee frames above. |
| **Tile trainer on the full data** (corD, fold 1, 25 epochs, `rsna_knee.train_tiles` on the original fold-1 caches) vs the original run | training loss per epoch within 0.001 of the original (e.g. 0.7405 / 0.7119 / 0.7044 vs 0.7401 / 0.7113 / 0.7048). Validation AUC tracks the original (epoch 6: macro 0.917 vs 0.913). Best proxy AUC 0.969 at epoch 18 (macro 0.938) vs the original's selected 0.971 (macro 0.935); each is scored on its own label version (final here, an earlier one originally). The run stopped after 24 of 25 epochs, when the scratch disk filled up. |
| Pseudo labels (round-1 teacher built from the released pack's plane cells, 3 fastMRI knees) | Same QC as the original: 1 of 3 kept (two lack an axial series). Values within 0.03–0.07 of the original round 1, from a different teacher checkpoint generation. |

## Smoke runs (wiring, on tiny data)

`tests/smoke_test.py` runs every stage on 70 RSNA studies (plus 3 fastMRI studies): pretrained download → cache build →
keypoint training (5 epochs) → stage-A tile cell (2 epochs) → crop caches → the four crop specialists (6 epochs) → specialist
pack → model pack → inference with that pack (the OA and specialist blends active, 0 fallbacks) → fastMRI preparation →
round-1 teacher pack → pseudo labels. Every stage completes. Also run separately, through the same scripts:
* P1 pretraining on the OAI cache with round-1 pseudo labels, then a stage-B fine-tune initialised from it
  (`40_train_cells.py --stage P1 / B --smoke`);
* the MOAKS label-model training on the OAI cache, with zero-shot prediction (2 epochs);
* the native-only bootstrap cache.

## Not re-run

* The full ~250 GPU-hour training.
* OAI archive extraction (`11_prepare_oai.py` beyond the series selection): the raw OAI images were no longer available
  on the development machine. The selection, the member filter (wrong-plane / projection images) and the folder layout
  follow the original reader exactly.
* Kaggle offline notebook (`kaggle/inference_notebook.py`): it calls the same `predict.main` that the parity test runs.
