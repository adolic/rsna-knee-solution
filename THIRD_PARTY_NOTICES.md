# Third-party notices

The code and the released model weights are © 2026 Antonio Dolic and licensed under CC BY-NC 4.0 (`LICENSE`). This file
lists what that licence does not cover: the pretrained weights the models start from, the data they were trained on, and
the Python dependencies. It also gives the citations those sources require. The terms below apply in addition to
CC BY-NC 4.0.

**Research use only.** The models are not a medical device. Do not use them, or anything they produce, for diagnosis or
patient care. The RSNA MIRA licence (§7) and the NYU fastMRI agreement both forbid that use of anything generated with
their data.

## 1. Pretrained weights the released models were fine-tuned from

Every released model starts from one of these checkpoints and was then trained further by us, so the released weights are
**modified** versions of them (Apache-2.0 §4(b)). A copy of the Apache License 2.0 is in `LICENSES/Apache-2.0.txt`. The
MIT notice for the ResNet50 is in `LICENSES/MIT-RadImagenet-ResNet50.txt`.

| Checkpoint (Hugging Face, revision) | Licence | Released models fine-tuned from it | Cite |
|---|---|---|---|
| `timm/convnext_tiny.in12k_ft_in1k` (`main`) | Apache-2.0. Weights trained in timm by Ross Wightman (timm: "Copyright 2019 Ross Wightman"). | 45 tile models (cells sagD, corA, corD, axiA, sagFM, corFM, axOAI2, allfm, allfmold) and 15 crop specialists (acl, men, oafam) | Liu et al., *A ConvNet for the 2020s*, CVPR 2022 |
| `timm/caformer_s18.sail_in22k_ft_in1k` (`cc6467f00ce5f7c89ad66e02900bc17e6df2c135`) | Apache-2.0. Weights by the paper authors (sail-sg/metaformer). | 5 tile models (allcfm) | Yu et al., *MetaFormer Baselines for Vision*, arXiv:2210.13452, 2022 |
| `timm/rexnetr_300.sw_in12k_ft_in1k` (`524b9020be9649c58f7b4c48e52def228b97ea08`) | Apache-2.0. Weights trained in timm by Ross Wightman. | 5 tile models (allrex) | Han et al., *Rethinking Channel Dimensions for Efficient Model Design*, CVPR 2021 |
| `timm/efficientnet_b0.ra_in1k` (fetched by timm; revision not pinned) | Apache-2.0. Weights trained in timm by Ross Wightman. | 5 keypoint localizers | Tan & Le, *EfficientNet*, ICML 2019 |
| `convergedmachine/RadImagenet`, `resnet50.pth` (`7fa62dccee2b14e49188d11c4b0e66d88c29b41e`) | MIT, according to the model card's tag. No copyright holder is named. | 5 ResNet50 OA models | Mei et al., *RadImageNet*, Radiol Artif Intell 2022;4(5):e210315 |

* **ImageNet:** the timm backbones were pretrained on ImageNet. The ImageNet terms allow "non-commercial research and
  educational purposes" only, and timm's author advises assuming those terms carry over to the weights. A non-commercial
  release is consistent with that.
* **RadImageNet:** the ResNet50's own training arguments show it was trained from scratch on RadImageNet (165 classes)
  with the torchvision reference script. It is not the official BMEII-AI release, and its provenance is undocumented.
  The RadImageNet images need a licence from RadImageNet for commercial use.
* **Architecture credits:** ConvNeXt (facebookresearch/ConvNeXt, MIT, Meta Platforms), MetaFormer (sail-sg/metaformer,
  Apache-2.0), ReXNet (clovaai/rexnet, MIT, NAVER Corp.). No code from these repositories is included here: the models
  are built with timm.
* timm: Ross Wightman, *PyTorch Image Models*, 2019, doi:10.5281/zenodo.4414861.

## 2. Data each released model was trained on

| Models | Count | Images used in training | Labels used in training |
|---|---:|---|---|
| Tile cells sagD, corA, corD, axiA | 20 | RSNA | RSNA-derived labels (`labels/`) |
| Tile cell axOAI2 | 5 | RSNA; OAI (pretraining on pseudo labels) | as above, plus OAI pseudo labels |
| Tile cells sagFM, corFM, allfm, allfmold, allcfm, allrex | 30 | RSNA; **OAI + fastMRI** (pretraining on pseudo labels) | as above, plus OAI / fastMRI pseudo labels |
| Meniscus specialist (men) | 5 | RSNA; OAI (backbone pretraining) | as above, plus OAI pseudo labels |
| ACL and OA-family specialists (acl, oafam), ResNet50 OA models | 15 | RSNA | RSNA-derived labels |
| Keypoint localizers | 5 | RSNA | our own landmark annotations of 417 RSNA knees |

The RSNA-derived soft labels also contain blends with image models trained only on the OAI MOAKS expert reads
(effusion, synovitis, medial OA, medial meniscus, Baker's cyst). No OAI or fastMRI image, subject ID, MOAKS read or
pseudo label is included in this repository or in the model pack.

## 3. RSNA competition data

* **Source:** RSNA Knee Abnormality Detection (Kaggle). Data © Radiological Society of North America (RSNA).
* **Terms:** the competition rules (§2.4) and the RSNA MIRA Dataset Research Use Agreement V2.0
  (http://rsna.org/mira-license):
  * research use only (§1);
  * no distribution or reproduction of any portion without RSNA's prior written permission (§4);
  * no derivative works (§6);
  * no diagnosis or patient care (§7);
  * no re-identification (§9).
  The competition data is not redistributed here. Download it yourself from Kaggle.
* **`labels/`:** derived from the competition data. It holds report-derived soft labels, the expert labels of the 58
  "Gold" studies as given in `train.csv`, our keypoint annotations of RSNA series and scanner strata from the DICOM
  headers. RSNA keeps all rights in the underlying data. These files are **not** licensed under CC BY-NC 4.0 or any
  other licence by us. Their use is governed by the competition rules and the MIRA licence. The rules exempt such
  input data from the winner licence (§2.5.a.4).
* **Cite:**
  > Po-Hao "Howard" Chen, Naveen Subhas, Robyn Ball, Pieter Baeyens, Errol Colak, Ali Emami, Hillary Garner, Jacob Kazam,
  > Hui-Ming Lin, Luciano Prevedello, Daniel Schneider, Jason Sho, Ryan Holbrook, and María Cruz. RSNA Knee Abnormality
  > Detection. https://www.kaggle.com/competitions/rsna-knee-abnormality-detection, 2026. Kaggle.
* **Thanks** to RSNA, the organising team, the annotating radiologists and MD.ai for creating the competition dataset.

## 4. Osteoarthritis Initiative (OAI), NIMH Data Archive, with MOAKS reads

* **Source:** NDA Collection 2343 (DOI 10.15154/64dx-3822): baseline knee MRI and the MOAKS semi-quantitative reads
  (`oai_kmrisemiquant01`), including the FNIH OA Biomarkers Consortium project (read project 22).
* **Terms:** the NDA Data Use Certification. Nothing from OAI is redistributed here. `rsna_knee/moaks.py` and
  `scripts/11_prepare_oai.py` read files you obtain from NDA yourself.
* **NDA acknowledgement** (NDA's statement for OAI, verbatim):
  > Data and/or research tools used in the preparation of this manuscript were obtained and analyzed from the controlled
  > access datasets distributed from the Osteoarthritis Initiative (OAI), a data repository housed within the NIMH Data
  > Archive (NDA). OAI is a collaborative informatics system created by the National Institute of Mental Health and the
  > National Institute of Arthritis, Musculoskeletal and Skin Diseases (NIAMS) to provide a worldwide resource to quicken
  > the pace of biomarker identification, scientific investigation and OA drug development. Dataset identifier(s): NDA
  > Collection 2343, DOI 10.15154/64dx-3822.
* **Methods line:** "Data used in the preparation of this work were obtained from the Osteoarthritis Initiative (OAI)
  database."
* **OAI funding** (OAI Knowledgebase):
  > The OAI is a public-private partnership comprised of five contracts (N01-AR-2-2258; N01-AR-2-2259; N01-AR-2-2260;
  > N01-AR-2-2261; N01-AR-2-2262) funded by the National Institutes of Health, a branch of the Department of Health and
  > Human Services, and conducted by the OAI Study Investigators. Private funding partners include Merck Research
  > Laboratories; Novartis Pharmaceuticals Corporation, GlaxoSmithKline; and Pfizer, Inc. The results are the sole opinion
  > of the authors and not of the NIH or the other funding partners.
* **FNIH** (required because the MOAKS targets include read project 22):
  > Data from the FNIH OA Biomarkers Consortium Project was funded by: AbbVie; Amgen; Arthritis Foundation; Artialis;
  > Bioiberica; BioVendor; DePuy; Flexion Therapeutics; GSK; IBEX; IDS; Merck Serono; Quidel; Rottapharm | Madaus; Sanofi;
  > Stryker; the Pivotal OAI MRI Analyses (POMA) study, NIH HHSN2682010000 21C; and the Osteoarthritis Research Society
  > International.
* **MOAKS:** Hunter DJ, Guermazi A, Lo GH, et al. Evolution of semi-quantitative whole joint assessment of knee OA: MOAKS
  (MRI Osteoarthritis Knee Score). Osteoarthritis Cartilage. 2011;19(8):990–1002. doi:10.1016/j.joca.2011.05.004.

## 5. NYU fastMRI

* **Source:** NYU fastMRI knee DICOMs (fastmri.med.nyu.edu), used for pseudo-label pretraining of 30 tile models
  (section 2).
* **Terms:** the NYU fastMRI Dataset Sharing Agreement:
  * internal research or educational purposes only;
  * no distribution of the data or of data files derived from it;
  * no diagnosis or patient care;
  * no use of the NYU name or marks in publicity.
  Nothing from fastMRI is redistributed here. `scripts/12_prepare_fastmri.py` reads archives you obtain from NYU yourself.
* **Data source:** NYU fastMRI.
* **Methods statement** (adapted from NYU's template):
  > Data used in the preparation of this work were obtained from the NYU fastMRI Initiative database
  > (fastmri.med.nyu.edu) [Knoll et al. 2020; Zbontar et al. 2018]. As such, NYU fastMRI investigators provided data but
  > did not participate in analysis or writing of this report. A listing of NYU fastMRI investigators, subject to updates,
  > can be found at fastmri.med.nyu.edu. The primary goal of fastMRI is to test whether machine learning can aid in the
  > reconstruction of medical images.
* **Cite:**
  * Knoll F, Zbontar J, Sriram A, Muckley MJ, et al. fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee
    Images for Accelerated MR Image Reconstruction Using Machine Learning. Radiol Artif Intell. 2020;2(1):e190007.
    doi:10.1148/ryai.2020190007.
  * Zbontar J, Knoll F, Sriram A, et al. fastMRI: An Open Dataset and Benchmarks for Accelerated MRI. arXiv:1811.08839,
    2018.

## 6. Python dependencies

These packages are installed from PyPI (`requirements.txt`) and not redistributed here. All have OSI-approved licences that
do not limit commercial use.

| Package | Licence |
|---|---|
| torch | BSD-3-Clause (bundled components: Apache-2.0, BSD-2-Clause, BSL-1.0, MIT) |
| timm | Apache-2.0 |
| numpy, pandas, scikit-learn | BSD-3-Clause |
| opencv-python-headless | Apache-2.0 |
| pydicom | MIT |
| pillow | MIT-CMU (HPND) |
| safetensors, huggingface_hub, kaggle | Apache-2.0 |

## 7. No endorsement

RSNA, Kaggle, the NIH, the OAI investigators, the FNIH, NYU and the authors of the pretrained models have not reviewed or
endorsed this work.
