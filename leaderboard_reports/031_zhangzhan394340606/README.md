---
license: apache-2.0
tags:
  - chemprop
  - cyp
  - admet
  - openadmet
---

# OpenADMET CYP Challenge 2026 — Model Report

**Team:** Zhan Zhang (Independent Researcher, Xi'an, China)
**Hugging Face:** `zhangzhan394340606`
**Tracks:** Direct Inhibition (regression); Time-Dependent Inhibition (classification)

## Data

Challenge training data and public CYP potency datasets. No test labels were used;
public structures overlapping the blind set were removed before training. No
proprietary data.

## Models

Multi-task Chemprop D-MPNN ensemble with per-isoform specialists and fine-tuned
pretrained SMILES transformers. Five-fold group cross-validation; ensemble and
distribution calibration. TDI: LightGBM classifiers with threshold tuning.

## Validation

Official metrics and the official submission validator; group/cluster CV, not
random splits. No test labels or groundtruth were used.
