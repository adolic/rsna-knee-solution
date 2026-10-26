# Method

## 1. Preprocessing (`rsna_knee/pipeline.py`, `cache.py`)

The same code builds the training caches and runs at inference.

1. **Decode.** One read per DICOM instance: modality LUT, MONOCHROME1 inversion, two known malformed-pixel repairs.
   Slices are ordered along the median slice normal, then oriented to a canonical patient frame using only 90° turns and
   flips.
2. **native224.** Each series' whole field of view → 224 px, 1–99 % intensity window. This is the keypoint input.
3. **Keypoints.** An EfficientNet-B0 heatmap localizer (5 fold models) predicts 5 landmarks per series:
   * the landmarks are the medial and lateral tibiofemoral joint, intercondylar notch, patellofemoral joint and patella centre;
   * each series is scored on a band-centre slice triplet plus 7 random triplets, each also mirrored;
   * the predictions are fused into 3D points per knee by robust least squares across series, and series that moved
     during the exam (residual > 6 mm) are set aside;
   * two-knee studies are split by the series' field-of-view centres.
4. **Knee frame.** Origin O = midpoint of the two tibiofemoral points; axes medial / anterior / superior; scale
   k = |medial − lateral TF| / 45 mm. The side comes from anatomy, never from DICOM laterality tags.
5. **knee512.** Every series is resampled once to 0.3125 mm/px on a 512 × 512 canvas centred on O, with a 0.5–99.5 %
   window. Left knees are mirrored so that every knee looks like a right knee.

## 2. Tile model (`tile_model.py`, `tile_data.py`, `tiles.py`)

* **Tiles.** 224 px squares (70 mm), each holding 3 adjacent slices as channels. Tile centres are 3D points drawn from a
  mixture of anatomical anchors in the knee frame: compartments, condyles, plateaus, notch, patellofemoral joint,
  suprapatellar pouch, posteromedial corner, MCL, plus a uniform background. Each point is projected into a series.
* **Bag.** Training draws 12–24 tiles per series. Fluid-sensitive and other series can be weighted differently
  (`other_mult`). Validation and inference draw 96 tiles per series.
* **Model.** A backbone per tile (ConvNeXt-T, CAFormer-S18 or ReXNet-R300) → 256-d token + plane and fluid embeddings → one
  transformer layer over the knee's tokens → per-finding gated attention pooling → 12 logits.
* **Training recipe:**
  * AdamW: head lr 4.2e-4, backbone 6e-5 (1e-4 for ReXNet), wd 0.02; cosine schedule.
  * Backbone frozen for the first 2–3 epochs, then ramped up over 2 epochs.
  * EMA 0.999, bf16, effective batch 6 studies.
  * Weighted BCE on the soft labels with pos_weight √((1−p)/p).
  * Augmentation: study-level left-right mirror (p 0.5), knee-frame jitter (2 mm / 2°), series dropout per plane, one
    affine + gain/shift draw per study.
  * 25 epochs; the checkpoint with the best proxy AUC on the fold is kept.

## 3. The 55 tile models (`configs/cells.json`)

| Cell | Backbone | Series seen | Tiles / series | Initialisation | Plane group |
|---|---|---|---|---|---|
| sagD | ConvNeXt-T | sagittal | 24 (non-fluid ×0.25) | ImageNet-12k | sagittal |
| corA | ConvNeXt-T | coronal | 16 (non-fluid ×0.5) | ImageNet-12k | coronal |
| corD | ConvNeXt-T | coronal fluid-sensitive | 16 | ImageNet-12k | coronal |
| axiA | ConvNeXt-T | axial | 16 (×0.5) | ImageNet-12k | axial |
| sagFM | ConvNeXt-T | sagittal | 24 (×0.25) | round-1 pretraining, OAI + fastMRI | sagittal |
| corFM | ConvNeXt-T | coronal | 16 (×0.5) | round-1 pretraining, OAI + fastMRI | coronal |
| axOAI2 | ConvNeXt-T | axial | 16 (×0.5) | round-1 pretraining, OAI | axial |
| allfm, allfmold | ConvNeXt-T | all series | 12 (×0.5) | round-1 pretraining, OAI + fastMRI | all-plane |
| allcfm | CAFormer-S18 | all series | 12 (×0.5) | round-1 pretraining, OAI + fastMRI | own group |
| allrex | ReXNet-R300 | all series | 12 (×0.5), ≤ 8 series in training | round-2 pretraining, OAI + fastMRI | own group |

Each cell is trained on all 5 folds.

## 4. Pseudo-label pretraining (`scripts/50_pseudo_labels.py`)

The external knees are ~9,400 OAI baseline knees (sagittal IW TSE FS, coronal IW TSE, axial DESS reformat) and ~9,100
fastMRI clinical exams. Only knees with all three planes are kept.

* **Round 1.** The teacher is the four ImageNet-initialised cells (sagD, corA, corD, axiA; 5 folds each) at attention
  tau 2, own tile draws, with PF OA blended with the ResNet50 OA specialist. The probabilities are calibrated for training
  (logit − log pos_weight).
* **Pretraining.** 30 epochs on the external knees (early stop 5), validated on RSNA fold 1. The best checkpoint
  initialises all five folds of the RSNA fine-tune.
* **Round 2.** The teacher is the round-1 fine-tuned ensemble with all specialists at tau 12. It pretrains the ReXNet-R300.

## 5. Crop specialists (`crop_cache.py`, `crop_data.py`, `train_crops.py`, `configs/specialists.json`)

Small models on keypoint-centred square crops (`configs/crops/`), 3-slice windows, a shared backbone and attention pooling
over the windows:

| Specialist | Input | Training |
|---|---|---|
| meniscus | zoomed horns / body per meniscus (160 px from the canvases) | backbone pretrained 8 epochs on OAI crops with round-1 pseudo labels |
| ACL | notch crops at tile scale (224 px) | ACL-negative knees with a contusion weighted ×3 |
| OA family | compartment crops plus zoomed cartilage crops | one shared head for both tibiofemoral compartments; isolated positives ×3, context negatives ×2 |
| ResNet50 OA | six OA views, single slices at 320 px from the decoded series | RadImageNet ResNet50 initialisation |

## 6. Inference recipe (`configs/submission_recipe.json`, `predict.py`)

1. **Tile draws.** Each member draws its own tile bag (draw key = member position), 96 tiles per series.
2. **Attention temperature mix.** Each member's logits are averaged over attention tau 12 and 32 (sharper pooling than in
   training). The backbone runs once; only the pooling head re-runs.
3. **Octant TTA** on the all-plane cells: the head is also run with the tiles of each knee-frame octant left out, and
   those logits are averaged.
4. **Combination.** Per finding: a NaN-aware mean of member logits within each plane group, then the mean over the groups
   (sagittal, coronal, axial, all-plane ConvNeXt, CAFormer, ReXNet), then sigmoid. Two-knee studies: mean of the two
   knees.
5. **Specialist blends**, in logit space with the specialist logit standardised to the ensemble over the test set:
   * ResNet50 OA: medial OA 0.3, PF OA 0.4.
   * Meniscus 0.3 per side, ACL 0.2, OA family: lateral OA 0.1, PF OA 0.1.
6. **Precision.** FP16 weights and autocast throughout.
7. **Fallback.** A study that cannot be read gets the training prior; this is counted, and more than 5 % aborts the run.

## 7. What moved the public score

Each row is the first submission that reached a new best public score:

| Step | Public |
|---|---|
| Plane cells (sagD, corA, corD, axiA) × 5 folds, one shared tile draw, attention tau 1 | 0.960 |
| Own tile draw per member, tau 2 | 0.963 |
| + cells pretrained on OAI pseudo labels, ResNet50 OA blend for PF OA | 0.964 |
| + all-plane ConvNeXt pretrained on the full OAI set | 0.965 |
| + CAFormer-S18 cell, canvas-crop specialists | 0.966 |
| attention tau 8 | 0.967 |
| + per-plane and all-plane cells pretrained on OAI + fastMRI | 0.968 |
| tau 32, + ReXNet-R300 (round-2 pseudo labels), + octant TTA; cells retrained on the refined labels | 0.969 |

The final submission adds the tau 12/32 mix and the second allfm fold set; it was also 0.969. Public differences of
0.001–0.002 between these submissions are within noise. Among 0.969 submissions the selection was made on held-out
cross-validation and a public/private split simulation.
