# wbot (rank 7, Tier 4, MA-ST-RAE 0.389) - distilled notes

Source: Google Doc pasted by user (verbatim HTML NOT saved - ~100KB mostly class markup). Content below is a condensed digest, not a full copy.

NOTE: leaderboard score (0.389) differs from the "public-score arithmetic" in the report (0.36617 macro ST-RAE, spliced from best public columns per isoform; not a private-holdout estimate).

## Data
- Primary targets: organizer pIC50 + credible intervals. 6,525 measured compound-isoform obs / 4,905 molecules (1A2 1,412; 2C9 1,285; 2D6 1,493; 3A4 2,335).
- Never treat missing measurement on another isoform as inactive.
- External: ChEMBL 37 (111,666 CYP records, 9 isoforms; used std value/units/relation, not pchembl, to keep 21,139 inequality rows), PubChem AID 1851 qHTS (85,715 rows; inactives without potency kept as censored evidence), ToxCast (direct pooling into IC50 head audited and REJECTED: assay/substrate mismatch).
- Enamine catalog (500k sampled) used as UNLABELED data for self-supervised property tasks (clogP, charge), not labels.
- Blind set = 75 congeneric series x 10; 75 parents with 152 measured labels, all kept in training.

## Splits / metrics
- Structure-grouped 5-fold: connected components (nearest FCFP4 neighbor >=0.40, merged at Tanimoto >=0.60). Analogue/parent-retaining protocol; held-out still less similar to train than blind queries.
- ST-RAE = sum interval-error(pred) / sum interval-error(mean). Interval error = distance outside credible interval.
- Admit: process highly adaptive, public leaderboard used for column selection; no unbiased OOF for final pipeline.

## Features
- Monroe (GRIT) 720-d frozen embedding -> rank-16 SVD (96 for 2C9 TabPFN/TabICL); Uni-Mol2 768-d -> 16; CHARGE4 (pH7.4 charge, basic/acidic centres, formal charge); heme SMARTS (17) + 44 CYP features; Boltz-2 affinity/binding prob + z_cross (256-d pair-rep pool); predicted log2FC and native D-MPNN heads fed to tabular models.
- Tables: F0 36, F1 38, F2 54, N5 62, H264, P296, P296+H4/H28, 179-feat CYP2C9 TFM.
- Not helpful/redundant: Jazzy, Mordred, MolFormerXL (not selected), AIMNet2/QM (didn't transfer), Gnina docking/heme distance (provenance bug; no blind coverage).

## Models
- D-MPNN (Chemprop-style): depth 6, width 2048, mean pool, 300-unit FFN, dropout 0.1, bs 64, 16 epochs, LR 1e-4->1e-3->1e-4. 28 tasks: 4 pIC50 (w1.0), 4 log2FC, 6 ChEMBL IC50, 5 AID1851 pAC50, 5 max response, 4 calc props (w0.3/0.25). Loss = distance outside interval (MAE when bounds coincide). Seed + fold averaging (e.g. 2D6: 24 seeds x 5 folds).
- Ridge (H28: ridge on P296 + 28 native D-MPNN heads, weights 1/(1+u-l), alpha 100 2C9 / 300 3A4): 3A4 native ST-RAE 0.464 -> 0.388 vs plain ridge.
- ExtraTrees (ET20: 300 trees, leaf 20); ECFP ET (256 trees, leaf 2).
- TabPFN v2 + TabICL v2 on feature tables (equal average).
- Negative/neutral: CheMeleon-init D-MPNN, ADMET-AI pretraining, Monroe fine-tune, wider embeddings, LightGBM/CatBoost, KNN, causal-chemprop, classifier hurdles, selectivity priors, MMP delta learning.

## Data interventions
- SC25: weak upper bounds for molecules with log2FC >= -0.25 and no pIC50 (1A2 335 recipients; 2D6 2,693 bounds at pIC50 3.97; 3A4 162 at 4.30103).
- 2D6 F1 D-MPNN: 1,493 measured + 2,693 screen bounds + 2,848 qHTS inactive bounds = 7,034 cells.
- 2C9: up to 456 synthetic low-potency labels ~ Normal(2.5, 0.7) per realization.

## Ensemble per isoform
- 1A2: 8-seed SC25 D-MPNN + 1/6*(H28 ridge - ET), affine placement, soft floor.
- 2C9: two branches 50/50 (D-MPNN 20.6% + 179-feat TabPFN/TabICL 79.4%; four D-MPNNs 2/15 each + N5 TabICL + H28 ridge 1/3).
- 2D6: equal thirds qHTS D-MPNN, ET20, descriptor ridge, affine placement.
- 3A4: 80% eight-group neural/TFM + 20% (H28 ridge, 2 ECFP ET, interval D-MPNN).

## ST-RAE-specific post-processing (like rank 1)
- Affine placement T(p) = mu* + sigma*(p-m)/s with chosen moments: 1A2 (4.45, 1.0656), 2C9 (5.0, 0.8622), 2D6 (3.75, 1.0725), 3A4 (4.95, 0.9916). Developed with public-board feedback.
- Soft floor F(p)=c+a(p-c) for p<c: 1A2 c=3.25 a=0.05; 3A4 c=3.25 a=0.25 (raises low predictions toward threshold; same idea as rank 1's upward shift).
- Localized hand offsets on 70 (1A2), 0 (2C9), 12 (2D6), 23 (3A4) blind predictions.
- Public scores per isoform: 1A2 0.3695, 2C9 0.3316, 2D6 0.4239, 3A4 0.3396.
