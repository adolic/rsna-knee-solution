# Labels

The competition gives a radiology report for every training study, but expert image labels for only 58 of them (the
"Gold" studies). The training labels were therefore built from the reports and refined in many audited steps. The repository
ships the final version, the one the last models were trained and validated on:

| File | Content |
|---|---|
| `labels/soft.csv` | soft labels in [0, 1], 4,407 studies × 12 findings: the training targets |
| `labels/hard.csv` | hard labels for validation (soft ≥ 0.5; the Gold studies keep their expert values) |
| `labels/weights.csv` | per-cell loss weights: 2 on the 58 Gold studies, 1 elsewhere |
| `labels/folds.csv` | 5 folds grouped by patient (fold = the study's validation fold), `is_gold` |
| `labels/two-knee-sides.csv` | for the few studies that scan both knees: the knee the report describes |
| `labels/keypoint-targets.csv` | keypoint annotations of 417 knees, projected to every series (training data of the localizer) |
| `labels/proxy-strata.csv` | scanner model × field strength per study (the selection metric's strata) |

## How the soft labels were made

1. **Report reading.** Several independent LLM extractions of every report gave per-finding probabilities, combined by
   consensus. The per-finding rubric follows the host's definitions, for example OA = cartilage loss of more than half its
   thickness, and ambiguous findings negative.
2. **Teacher lifts** for findings that reports often leave unsaid (for example an effusion visible but not mentioned):
   an early image model's out-of-fold probability raised such cells.
3. **Rules on report wording**, each validated on the Gold studies and on held-out ranking checks before release:
   * PF OA: deep cartilage loss → positive, low-grade only → halved, unspecified depth → floor 0.8.
   * Lateral meniscus: intrasubstance degeneration without a tear → ≤ 0.2.
   * Fracture: a strong traumatic bone bruise with no fracture wording → floor 0.45.
   * ACL low-grade partial tear / sprain and MCL low-grade sprain → halved.
   * Site-specific wording, for example one German template's "akzentuierte Gelenkflüssigkeit" (accentuated fluid, not
     an effusion) → 0.25.
4. **Image adjudication.** Confident model–label disagreements were read on the model's own input tiles: 33
   effusion / Baker's / meniscus cells and 10 contusion cells were corrected.
5. **Per-site audits of synovitis and effusion.** Reporting style varies per site: some templates never mention synovitis,
   others mention it whenever there is fluid. Site-level rules removed most of that bias.
6. **Expert MOAKS reads (OAI).** Tile models trained only on the MOAKS reads of the OAI baseline knees, never on RSNA
   labels, were applied zero-shot to every RSNA study and blended in:

| Finding | MOAKS head | Weight |
|---|---|---|
| Effusion | P(whole-knee effusion grade ≥ 2) | 0.50 |
| Synovitis | P(Hoffa synovitis grade ≥ 2) | 0.25 |
| Medial OA | P(any medial subregion with cartilage loss over ≥ 10 % of it) | 0.25 |
| Medial meniscus | P(any medial meniscal tear) | 0.25 |
| Baker's cyst | P(popliteal cyst) | 0.25 |

   Each step was released only when it improved the Gold-58 AUC and held-out ranking checks. On the 58 Gold studies the
   zero-shot MOAKS models reached 0.968 effusion AUC and 0.957 medial meniscus AUC, compared with 0.953 for the report
   labels on the meniscus.

The Gold studies keep their expert labels throughout (weight 2). The MOAKS step is reproducible with
`scripts/60_moaks_label_models.py`, from the NIMH Data Archive file `oai_kmrisemiquant01.txt`:

* `rsna_knee/moaks.py` rebuilds the three target sets exactly as used: every baseline read counts equally; MOAKS cartilage
  codes are split into area and full-thickness grades.
* The blend is applied with `--stage blend --base <labels before the step>`.
* The earlier steps (report reading, rules, audits) are not part of this repository; their result is `labels/soft.csv`.

## The selection metric

`rsna_knee/proxy.py` is the checkpoint-selection metric: AUC over positive/negative pairs from the same scanner stratum
only, each pair weighted by the two cells' label confidence |2·soft − 1|². Targets are weighted OA ×1.5 and contusion /
fracture ×0.5. It tracked the public leaderboard better than plain macro AUC (Spearman 0.94 vs 0.86 over 61 submissions).
