<div align="center">

# 🧬 OpenADMET CYP Inhibition Challenge

### Predicting Cytochrome P450 Inhibition & Time-Dependent Inhibition

[![Challenge](https://img.shields.io/badge/challenge-OpenADMET%20CYP-blueviolet)](https://huggingface.co/spaces/openadmet/cyp-challenge)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![MA--ST--RAE](https://img.shields.io/badge/MA--ST--RAE-0.755-orange)]()
[![TDI MCC](https://img.shields.io/badge/CYP3A4%20TDI%20MCC-0.354-orange)]()

Predictions for four CYP isoforms — **CYP3A4 · CYP2D6 · CYP2C9 · CYP1A2** —
scored against the challenge's own official evaluation code, not a
reimplementation.

</div>

---

## 📊 Results

Validated on a held-out, **structure-aware split**, scored with the challenge's
[official `evaluation/` code](https://github.com/OpenADMET/CYP-Challenge-Tutorial).

### Direct Inhibition (Regression)

| Isoform | ST-RAE ↓ | MAE ↓ | R² ↑ |
|:--|:--:|:--:|:--:|
| 🟢 CYP3A4 | **0.532** | 0.573 | 0.561 |
| 🟢 CYP2C9 | **0.616** | 0.482 | 0.404 |
| 🟡 CYP1A2 | 0.907 | 0.690 | 0.179 |
| 🔴 CYP2D6 | 0.966 | 0.597 | 0.105 |
| **🎯 Macro (MA-ST-RAE)** | **0.755** | 0.585 | 0.312 |

### Time-Dependent Inhibition (Classification)

| Isoform | MCC ↑ | Threshold |
|:--|:--:|:--:|
| 🟢 CYP3A4 | **0.354** | 0.45 |
| 🟡 CYP2D6 | 0.126 | 0.45 |

> 🟢 solid · 🟡 weak-but-real signal · 🔴 struggling — see [Limitations](#-known-limitations)

---

## 🗂️ Data

| Source | Config | Rows | Used for |
|:--|:--|:--:|:--|
| `openadmet/cyp-challenge-train-test` | `default` | ~4,900 | pIC50 + credible intervals |
| `openadmet/cyp-challenge-train-test` | `tdi` | ~4,900 | CYP3A4/CYP2D6 TDI labels |
| Official blinded set | — | 750 | Final predictions |

TDI labels are used **as provided**, not re-derived — consistent with the
official rule: positive if the TDI-arm shift exceeds 2-fold relative to
direct inhibition, with inferred positives for low-activity compounds where
a shift can't be measured directly.

---

## 🔬 Method

```
SMILES ──► Morgan FP (2048) + MACCS (167) + RDKit descriptors (13)
              │
              ▼
   Structure-aware split (KMeans on fingerprints, whole clusters held out)
              │
       ┌──────┴──────┐
       ▼             ▼
  Regression      Classification
  LGBM+XGB+RF     RandomForest
  (per isoform)   (OOF-tuned threshold)
       │             │
       ▼             ▼
  Official scorer (evaluation/ from CYP-Challenge-Tutorial)
```

**Why structure-aware, not random, splitting?** The real test set was built
by hit-expansion — top hits plus purchased chemisimilars — so a random split
would overstate performance. Whole fingerprint clusters are held out instead.

**Why the official scorer?** The published metric (Macro-Averaged
Soft-Threshold RAE) measures distance to a compound's *credible interval*,
not a flat error threshold. An early approximation of this metric gave
misleadingly optimistic numbers; every result above uses the real code.

---

## 🧪 What Didn't Work

Kept here as a record, not just the wins — these are real, validated
negative results:

<details>
<summary><b>❌ Pseudo-labeling from single-concentration screening data</b> (17.5k extra measurements)</summary>

<br>

Linear calibration from `log2fc_estimate` → pIC50 on overlapping compounds,
applied to ~3,000 new compounds per isoform.

- **CYP2D6:** 99% of new compounds fell outside the calibration model's
  fitted range — a selection-bias artifact, since compounds only reach DRC
  follow-up *because* they showed activity at screening.
- **CYP1A2:** actively hurt performance even at low sample weight
  (MAE 0.690 → 0.999, R² went negative).
- **Reverted.**
</details>

<details>
<summary><b>❌ Hyperparameter tuning + feature selection</b> (Optuna, 40 trials/isoform)</summary>

<br>

MA-ST-RAE 0.758 vs. baseline 0.755 — essentially no change. Every isoform's
search converged on keeping the full feature set; dimensionality wasn't
the bottleneck.
</details>

<details>
<summary><b>❌ Structural-alert features for CYP2D6 TDI</b> (bioactivation SMARTS patterns)</summary>

<br>

Furans, thiophenes, anilines, and similar reactive-metabolite motifs added
as binary flags. No meaningful MCC improvement over fingerprints alone.
</details>

**Takeaway:** performance is currently bottlenecked by the *feature
representation*, not model choice, tuning, or extra noisy data — motivating
a pretrained-embedding experiment (CheMeleon) as the next direction.

---

## ⚠️ Known Limitations

- CYP2D6 and CYP1A2 are the weakest endpoints throughout — smallest training
  sets, and CYP2D6 additionally uses a different assay technology than the
  other three isoforms.
- All reported numbers use the real official scorer; an earlier flat-threshold
  ST-RAE approximation used during development is **not** reflected above.

---

## ▶️ Reproducing

```bash
conda env create -f environment.yaml   # from OpenADMET/CYP-Challenge-Tutorial
conda activate oadmet_cyp_tutorial
pip install datasets lightgbm xgboost loguru
```

See `notebooks/` for the full pipeline: data loading → TDI join →
featurization → structure-aware split → training → official evaluation →
submission generation.

---

<div align="center">

Built for the [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge)

</div>
