# OpenADMET CYP Inhibition Blind Challenge — Methods report
Entrant: Igor Eduardo, Cortexa Social · Date: 2026-10-18 · Proprietary data: none · External training data: none (model C starts from the public pretrained CheMeleon foundation-model weights)

## Data
Official training files only (openadmet/cyp-challenge-train-test): direct-inhibition pIC50, TDI labels, Emax and single-concentration screens (the latter two used only as auxiliary training targets/features). No external training data.

## Validation
5-fold scaffold-grouped (Murcko) cross-validation on common folds shared by all models; all numbers below are out-of-fold. Blend weights were picked on the same out-of-fold predictions (grid step 0.1), so blended scores are slightly optimistic.

## Direct inhibition (regression) — pIC50 per isoform
Models (all CPU, free compute):
- (a) LightGBM per isoform on ECFP4 (2048 bits, r=2) + RDKit 2D descriptors;
- (b) the same plus auxiliary features derived from the Emax / single-concentration training data;
- (A) Chemprop v2 multitask D-MPNN (4 pIC50 tasks) + RDKit 2D normalized features, 3 seeds;
- (B) as A plus 8 auxiliary Emax / single-concentration tasks, 3 seeds;
- (C) Chemprop v2 fine-tuned from the CheMeleon foundation model, pIC50 + auxiliary tasks, 2 seeds.
Final = per-isoform weighted blend (weights below). Per isoform, the blend with the better out-of-fold RAE is kept: for CYP1A2, CYP2C9 and CYP3A4 that is the blend with 2 CheMeleon seeds; for CYP2D6 the earlier blend with 1 CheMeleon seed (RAE 0.932 vs 0.935) is kept.

| Isoform | LGBM | LGBM+aux | CP-A | CP-B | CP-C (CheMeleon) | Interim RAE | **Final blend RAE** | Final MAE | Final Spearman |
|---|---|---|---|---|---|---|---|---|---|
| CYP1A2 | 0.884 | 0.912 | 0.888 | 0.877 | 0.873 | 0.874 | **0.830** | 0.624 | 0.563 |
| CYP2C9 | 0.798 | 0.751 | 0.801 | 0.789 | 0.750 | 0.751 | **0.714** | 0.423 | 0.691 |
| CYP2D6 | 0.991 | 1.005 | 1.003 | 0.992 | 0.955 (1 seed) | 0.980 | **0.932** | 0.588 | 0.438 |
| CYP3A4 | 0.608 | 0.598 | 0.675 | 0.660 | 0.604 | 0.590 | **0.567** | 0.508 | 0.800 |

Blend weights (LGBM / LGBM+aux / CP-A / CP-B / CP-C): 1A2 0.2/0.1/0.1/0.2/0.4 · 2C9 0/0.4/0/0.1/0.5 · 2D6 0.2/0.1/0.2/0/0.5 (1-seed CP-C) · 3A4 0.2/0.3/0/0/0.5.

## TDI (classification) — CYP2D6, CYP3A4
LightGBM classifier (class-balanced) on fingerprint/descriptor features plus leak-free out-of-fold blended pIC50 predictions (test compounds receive blind-set blend predictions). Per endpoint, the better of the interim and v2 models (by OOF MCC) is used.

Correction vs v1: v1 used in-sample pIC50 predictions as features for training compounds, which inflated its internal MCC (0.258 / 0.503). All numbers below are leakage-free.

**CYP3A4 threshold choice:** max-MCC on OOF picks a low threshold (≈0.13–0.14) with MCC ≈0.49 but predicts ≈60% test positives vs ≈21% training prevalence. Threshold **0.5** was chosen for robustness versus the base rate (CV MCC ≈0.43; predicted test positive rate ≈34%, closer to training prevalence).

| Endpoint | n (pos) | Interim MCC | v2 MCC | Used | Threshold | Predicted test positive rate |
|---|---|---|---|---|---|---|
| CYP2D6 TDI | 1497 (324) | 0.133 | 0.117 | interim | 0.42 | 7.1% |
| CYP3A4 TDI | 3584 (764) | 0.417 | 0.490 max / **0.435 at thr 0.5** | v2 | **0.5** | **34.0%** |

Re-running the TDI step on the 2-CheMeleon-seed blend gave CYP2D6 MCC 0.106 and CYP3A4 MCC 0.490 (threshold 0.17, 55.5% predicted test positives); neither beat the models above for model selection. The classification file keeps interim CYP2D6 and v2 CYP3A4 with threshold 0.5.

## Limitations
Small per-isoform training sets (1.3–2.3k); CYP2D6 remains hard (RAE ≈ 0.93, TDI MCC 0.13). The CheMeleon variant used for the selected CYP3A4 TDI features has one seed (pre-seed2 blend). CYP3A4 TDI at threshold 0.5 trades some OOF MCC (0.435 vs 0.490 at thr ≈0.14) for a predicted positive rate nearer the training base rate.

## AI use
Code and this report were drafted with AI assistance and reviewed by the entrant.
