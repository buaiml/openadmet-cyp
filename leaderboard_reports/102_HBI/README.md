---
tags:
- chemistry
- openadmet
- cyp-inhibition
- research
---

# HBI CYP Challenge V2 — preregistration and methods

This is HBI's public methods record for the 2026 OpenADMET CYP inhibition blind challenge. This repository currently contains a preregistration, not a released model package. At initial publication, no V2 model has been fitted, no prediction file has been submitted, and no challenge leaderboard result or hidden test outcome has been used. The local protocol and implementation were frozen before publication. Subsequent progress will be added below without replacing the original preregistration commit.

## Integrity and scope

- Local freeze time: 2026-09-20T00:52:37.617225+00:00.
- SHA256 of the local integrity manifest: `2b0207399ebf8882f55fc87053532692e9c09fd4cc1ad1c5809cb774115e46a1`.
- SHA256 of the normative protocol JSON: `932f5764035069360320669b53e567aa4cbff7ff729ed58aa0dac9cd7f3a2b14`.
- The accompanying `protocol.json` gives the full fixed recipe. `implementation_hashes.json` records implementation hashes without local filesystem paths.
- The first public commit is the external timestamp for these decisions. The local manifest itself contains filesystem paths and is retained privately.
- All 99 artifacts in HBI's V1 preservation ledger were verified unchanged. Previously exposed V1 outcomes are development material, never a fresh validation set.
- This entry uses public challenge training data only. No proprietary training data or pretrained model weights are used. Code and model weights are not being represented as open source by this methods-only publication.

## Data and endpoints

Source: [OpenADMET CYP challenge dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test), pinned revision `3ac9c5dbb83eec5780ec7fa511908698cfe1396d`.

Four direct-inhibition regressions predict the organizer's pIC50 estimates: CYP1A2, CYP2C9, CYP2D6 and CYP3A4. Two classifications predict the organizer's Boolean time-dependent-inhibition calls: CYP2D6 and CYP3A4. We use the official TDI labels, including inferred positives and assigned negatives; they are a different target from HBI V1's stricter paired-arm-only TDI analysis.

The direct training file contains 4,905 compounds and the TDI training file 6,145. Emax and single-concentration files are acquired for provenance but are not used in this fixed entry. The 750-compound blinded file supplies only official identifiers and structures. No hidden outcomes are acquired. All five source files are checked against pinned SHA256 digests.

The publisher selected blind compounds by similarity to training hits. Results therefore measure blind analog expansion within this assay program, not independent laboratory, unrelated-scaffold, clinical, or customer validation. Submission does not provide access to withheld labels.

We retain the source notices: the Hugging Face card identifies Apache-2.0, while the publisher's assay/data announcement identifies CC BY 4.0. No source material is relicensed by this methods record.

## Chemistry and split

HBI's frozen parent-standardization procedure supplies 12 physicochemical descriptors and 2,048-bit chiral Morgan fingerprints of radius 2. Unsupported training structures are excluded and documented. An unsupported blind structure stops the complete-entry workflow; we do not invent a prediction.

Before development splitting, all training structures sharing a standardized non-stereochemical canonical-tautomer identity with any blind structure are excluded. Related analogs remain allowed by the challenge design.

The identity-only union of eligible direct and TDI training compounds is grouped by parent and stereo/tautomer identity, atom-typed Murcko scaffold for cyclic compounds, and connected components linked by nonchiral radius-2 Morgan Tanimoto similarity of at least 0.70. Acyclic compounds do not receive one universal empty-scaffold group. Intact components are assigned deterministically toward 60% train, 15% calibration, 15% policy selection and 10% development test, with seed 425. Actual fractions and dominant groups are reported. We do not reroll or relax grouping after examining outcomes.

Direct regression requires finite point estimates and enclosing finite confidence bounds. For repeated exact-parent observations, the lexically first eligible molecule identifier is retained, independently of target magnitude. TDI requires the official nonmissing Boolean call; conflicting calls within a parent are excluded. Each endpoint partition must contain at least 10 records, and all classification partitions must contain both classes, or the run stops and preserves the failure.

## Fixed model menu and selection

For each endpoint, compare a training-mean constant control and three learned recipes:

1. Physicochemical linear model: standardized descriptors, Ridge alpha 10 for regression or logistic regression C=1 for classification.
2. Morgan linear model: Ridge alpha 10 or logistic regression C=1, without feature scaling.
3. Extra Trees: 256 trees, minimum leaf size 2, maximum feature fraction 0.5 and seed 425.

Logistic classifiers allow 3,000 iterations. Each learned classifier receives sigmoid calibration using logistic regression C=1 fitted only on the calibration partition. This is assay-model calibration, not a clinical toxicity probability.

The policy partition selects the lowest soft-threshold RAE for regression and the highest Matthews correlation coefficient (MCC) for classification. TDI thresholds are restricted to 0.10 through 0.90 in increments of 0.05. Threshold ties prefer the value nearest 0.50 and then the higher value. Recipe ties follow the listed fixed order. The constant control is reported but cannot become a submitted model. Selection is saved before development-test diagnostics are computed. Development-test results cannot replace the selected recipe.

Soft-threshold RAE follows the [official tutorial](https://github.com/OpenADMET/CYP-Challenge-Tutorial) at revision `858ae63ce79934113bccdb7fc65467de5f7b1935`: sum distances outside the published intervals, divided by the same sum for the cohort-mean reference prediction. A zero denominator is undefined. Also report regression MAE and Spearman correlation, and classification MCC, AUROC, average precision and threshold-specific accuracy. The challenge server supplies official blind scores.

## Final fitting and submission

After selection, each regression recipe is refitted on all eligible public development records. Each classifier is refitted on all eligible records except calibration, then recalibrated on the separate calibration partition. The previously selected threshold is unchanged. These final refitted artifacts do not inherit the pre-refit diagnostic performance.

Both output files must contain exactly the 750 official identifiers and unchanged SMILES. Regression supplies four finite, nonconstant pIC50 columns; classification supplies two nonconstant Boolean columns. No variation is fabricated to satisfy the portal. Prediction files, final model artifacts, protocol, sources and implementation are hashed before upload or feedback access. No leaderboard feedback is used for selection or revision of this entry.

## Product claims remain separate

Challenge predictions do not authorize replacing assays, declaring safety, or automatically rejecting compounds. HBI's research flag criteria require at least 80% observed precision, 30 resolved flags, 20 flagged chemical groups, complete flag outcomes, and both 95% Wilson and group-bootstrap lower bounds of at least 60%. Chemical groups are diversity proxies, not guaranteed independent experiments. There is no minimum 10% flag fraction per endpoint.

Useful panel decision coverage needs one real jointly measured cohort, qualified endpoint policies, actual stop rules and a baseline-versus-recommended assay plan. NRG's assay list and stop rules are not yet available; panel savings and useful action coverage are therefore not estimable. PXR, cross-source transfer and pretrained-model comparisons are separate studies and are not included in the CYP entry's training.

## Progress after preregistration

The fixed training run completed after preregistration commit `86bc500814eefd757ae068478c2cd0eab3555684`. All six endpoints were fitted without changing the published recipe. Both submission files passed the 750-row identity, schema, finite-value and nonconstant-output checks.

| Endpoint | Selected recipe | Development test compounds | Diagnostic metric |
|---|---|---:|---|
| CYP1A2_pIC50_direct_inhibition | extra_trees | 128 | ST-RAE 0.944; MAE 0.744 |
| CYP2C9_pIC50_direct_inhibition | physchem_linear | 122 | ST-RAE 0.590; MAE 0.454 |
| CYP2D6_is_TDI | morgan_linear | 140 | MCC -0.017; AUROC 0.495 |
| CYP2D6_pIC50_direct_inhibition | physchem_linear | 140 | ST-RAE 1.097; MAE 0.598 |
| CYP3A4_is_TDI | morgan_linear | 369 | MCC 0.141; AUROC 0.696 |
| CYP3A4_pIC50_direct_inhibition | physchem_linear | 242 | ST-RAE 0.727; MAE 0.697 |

These are internal development diagnostics for the selected pre-refit models. They are not the accuracy of the final refitted challenge artifacts and are not independent validation. Regression MAE is in pIC50 units; ST-RAE and MCC are not percentages. Official blind performance is not yet available to this methods record.

Predictions were frozen at 2026-09-20T00:57:58.699030+00:00. Regression file SHA256: `93db8cab57730eea84871078d36e80fb06172a9096f5b640455bf4f2c1b351fc`. Classification file SHA256: `00eb4621367639478be8ebaaa1404134fd0aaf4e140e39ef9b54c9a0c09500d8`.

The identity-only split contained 6,145 unique parents in 5,195 chemical groups. Partition counts were 3,687 train, 922 calibration, 922 policy and 614 development test before endpoint-specific missing-label filtering. No exact or tautomer identity crossed partitions. The largest group was 0.85% of parents. Only 11 parents were acyclic; none fell in the policy or development-test partition, so these diagnostics do not establish acyclic performance.

Both entries were received by the challenge portal on September 20, 2026 UTC under alias HBI. The portal states that organizer processing can take up to two hours. Receipt is not a verified official score or completed server validation. A first leaderboard check occurred only after both frozen entries were submitted; no HBI score was verified. No hidden label was accessed, and no leaderboard result informed the entry. Neither file will be revised using leaderboard feedback.
