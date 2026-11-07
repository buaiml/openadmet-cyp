# OpenADMET CYP 2026: v3 fixed graph-model blend

Participant: **yemeng123**. Track: **Regression Prediction / Direct Inhibition**.
Training completed 2026-10-21; report prepared 2026-10-22.

This report describes the exact v3 CSV identified below. Earlier README and METHOD_REPORT_V2.md describe historical predictions, not v3.

## Scope and disclosure

This is exploratory computational research using organizer-released experimental data. There were no new wet-lab experiments. Design, implementation, analysis and report preparation used OpenAI Codex assistance. No clinical efficacy, safety, prospective experimental validation or guaranteed ranking improvement is claimed.

Aggregate official v1/v2 scores had already been viewed before the v3 protocol was frozen. The v2 per-endpoint scores are also known. The v3 training program does not read leaderboard data, and no hidden test labels were accessed. Nevertheless, this is iterative exploratory development informed by prior results, not a wholly untouched prospective evaluation. No official v3 score was available when the protocol, predictions and this report were prepared.

## Data and structures

Only the primary official inhibition data at dataset revision `146282d2ce1532f1803679a834a9bec9dac1e482` were used: 4,905 training molecules and 750 blinded molecules. Available labels number 1,412 (CYP1A2), 1,285 (CYP2C9), 1,493 (CYP2D6) and 2,335 (CYP3A4). No external training data, proprietary data or pretrained prediction models were used.

Targets are supplied pIC50 values, defined as -log10(IC50 in mol/L). Missing labels are masked in the multitask loss; molecules without any observed endpoint are omitted only from fitting. Values are neither clipped nor imputed. Supplied confidence bounds are used in ST-RAE evaluation, not the graph-model loss. Low-activity fitted values remain uncertain; this is not a fitted censoring model.

Input fragments, formal charges and stereochemistry are preserved with default RDKit parsing/sanitization. No salt stripping or tautomer normalization is applied. Canonical uniqueness and exact train/blind non-overlap checks passed. Murcko scaffolds ignore chirality; all acyclic molecules share one group. These checks do not eliminate all near-analogue similarity.

## Exact prediction recipe

For all four endpoints, the final prediction is fixed in advance as:

`v3 = 0.50 * v2 + 0.25 * graph(seed=20261021) + 0.25 * graph(seed=20261022)`.

The v2 component uses the frozen per-endpoint recipes and full-data predictions described in [the v2 report](https://huggingface.co/yemeng123/openadmet-cyp-2026-baseline/blob/0068385d8717ef0daed978b7c7418606a68c949f/METHOD_REPORT_V2.md). Its source and evidence are separately preserved at that revision. No endpoint-specific mixing weights or post-hoc clipping are introduced in v3. Graph-only scores are diagnostic and cannot replace the sole eligible blend after seeing results.

Each graph model uses Chemprop 2.3.1 `BondMessagePassing(d_h=128, depth=3, dropout=0.2)`, mean aggregation, and a four-task `RegressionFFN(input_dim=128, hidden_dim=128, n_layers=1, dropout=0.2)`. Chemprop's default molecule featurization is used. All other module defaults follow the recorded version.

Both seeds train for exactly 50 epochs, without early stopping or checkpoint selection. Training batch size is 64; inference batch size is 128; neither drops the last batch. Optimizer is Adam, weight_decay=0, with its remaining defaults. Learning rate is 0.0001 in epoch 1, 0.001 in epoch 2, then `0.001 * 0.1**((epoch-2)/48)` through epoch 50. Gradient norm is clipped to 5. Data-loader workers=0 and torch CPU threads=4.

Each endpoint is standardized with its fitting-fold mean and population standard deviation. Task weight is `total observed labels / (4 * observed labels for this endpoint)`, computed from fitting data only. The manual loss sums squared standardized errors multiplied by their observed-label mask and task weight, then divides by the number of observed labels in the batch. Missing targets are zero-filled only inside this masked arithmetic, never treated as observed measurements. Predictions are inverse-transformed using the same fitting-fold statistics.

## Fixed validation procedure

The v2 development/confirmation partition and three global outer scaffold folds are reused unchanged. Each graph outer model excludes every validation molecule from all four tasks. Comparisons use archived v2 nested out-of-fold predictions generated on these same folds. The old confirmation partition is evaluated only after the outer-fold gate passes. The graph models used for the final blinded predictions are fitted on all observed training data only after both gates pass.

**Both development folds and confirmation data have been used previously. Neither is a fresh independent test.** The archived v2 nested predictions reflect inner-fold recipe selection; v3's graph architecture and mixing rule were fixed before this run. No search over graph hyperparameters or weights occurred in this run. Repeated use of these partitions across versions still limits generalization claims.

ST-RAE uses the official tutorial revision `832ae6d6447b07c2c71059de34d07f1130388247`. Its numerator measures prediction distance outside each supplied measurement interval; its denominator uses the correspondingly thresholded constant at the evaluation-label mean. This mean is only a metric reference, never a model input. Independent arithmetic matched the pinned scorer.

## Recorded local results

ST-RAE is dimensionless; lower is better. These are internal scores, not official blinded scores. MAE/RMSE in the evidence bundle use pIC50 units.

| Evaluation | Endpoint | n | v2 ST-RAE | Graph mean ST-RAE | v3 blend ST-RAE |
|---|---|---:|---:|---:|---:|
| Pooled outer folds | CYP1A2 | 1127 | 0.837118 | 0.850614 | 0.809750 |
| Pooled outer folds | CYP2C9 | 1008 | 0.717222 | 0.727444 | 0.675112 |
| Pooled outer folds | CYP2D6 | 1188 | 0.930873 | 0.937772 | 0.891059 |
| Pooled outer folds | CYP3A4 | 1837 | 0.585584 | 0.623736 | 0.562499 |
| Reused confirmation | CYP1A2 | 285 | 0.803151 | 0.849227 | 0.792484 |
| Reused confirmation | CYP2C9 | 277 | 0.653399 | 0.664894 | 0.602669 |
| Reused confirmation | CYP2D6 | 305 | 0.910937 | 0.901807 | 0.863787 |
| Reused confirmation | CYP3A4 | 498 | 0.524812 | 0.622522 | 0.512061 |

Pooled outer-fold macro ST-RAE: **0.767700 to 0.734605**, a **4.31%** reduction. All three outer folds improve on the four-endpoint macro score. Reused-confirmation macro ST-RAE: **0.723075 to 0.692750**, a **4.19%** reduction.

The prespecified outer-fold gate requires at least 3% macro ST-RAE improvement, improvement in at least two outer folds, no endpoint more than 5% worse, and macro RMSE no more than 1% worse. Confirmation requires macro ST-RAE improvement, no endpoint more than 5% worse, and macro RMSE no more than 1% worse. All checks passed.

A paired scaffold bootstrap of fixed outer-fold predictions (1,000 replicates, seed 20261021) gives a blend-minus-v2 macro ST-RAE 95% interval **[-0.043210, -0.023295]**. This conditions on fixed predictions; it does not refit models or represent all training and model-selection uncertainty. Two random seeds are training replicates, not independent biological studies.

Post-hoc diagnostics retain an unresolved CYP2D6 weakness: among 90 outer-fold observations with pIC50 >=6, predictions remain biased downward and MAE changes from 1.4647 (v2) to 1.4722 (v3). The overall improvement does not fix this subgroup. These diagnostics did not change the frozen recipe.

Blinded molecules have higher median nearest-training Morgan similarity than outer validation molecules in this comparison, but the training reference sets differ in size. This does not establish that the blind test is easier, rule out distribution shift, or calibrate confidence. No calibrated prediction intervals are supplied.

## Submission identity and checks

Filename: `submission_regression_v3.csv`.

**SHA256: `db51259612c49684043b578534574b5766bb8c97c413ac2b56cfeac2903edc09`**.

The CSV has 750 rows, original blinded SMILES and molecule IDs, and four finite prediction columns. The official local format validator passed. Independent checks verified source/artifact hashes, scaffold isolation, training-only scaling/masking, seed aggregation, evaluation arithmetic, both release gates and reconstruction of every submitted prediction. Each seed checkpoint was reloaded using `weights_only=True` and checked against all its query rows during the original run.

These are local verification results. Platform receipt, platform validation and official scoring remain separate steps. No official v3 result or final rank is asserted here.

## Source and reproducibility limits

[Download v3 source and evidence](./v3-source-evidence.zip). This archive includes the frozen v3 training/checking scripts, exact frozen v2 prediction dependencies, fixed splits, recorded training/evaluation evidence and a SHA256 manifest. The frozen dependencies include the previous v2 prediction CSV and the graph seed prediction components; these are computational outputs, not experimental labels. Fitted weights, account credentials, private contact fields and machine-specific submission-status records are excluded.

The archive provides a pinned-source downloader and a fresh-directory reproduction path using the frozen v2 component. Rebuilding v2 itself is a separate procedure described in its report and source bundle. A complete independent retraining from a newly installed environment has **not** been verified. Packaging checks and earlier v1 reproduction do not establish such reproduction for v3.

Recorded graph environment: torch 2.10.0+cu126, Chemprop 2.3.1, RDKit 2026.3.6, NumPy 2.5.3, pandas 3.0.6 and scikit-learn 1.9.1; NVIDIA GeForce RTX 4060 Laptop GPU. Seeds are fixed and deterministic algorithms requested with warnings enabled, but GPU scatter operations can remain nondeterministic. Bitwise-identical predictions across new runs/platforms are not promised.

Source is public for inspection without a newly asserted open-source license; the competition open-code checkbox remains unchecked. Original data/tutorial terms remain authoritative. No external or proprietary training data were used, and no future-publication pledge is made.

## Official sources

- [Organizer dataset at the frozen revision](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test/tree/146282d2ce1532f1803679a834a9bec9dac1e482)
- [Official scoring tutorial at the frozen revision](https://github.com/OpenADMET/CYP-Challenge-Tutorial/tree/832ae6d6447b07c2c71059de34d07f1130388247)
- [Competition](https://huggingface.co/spaces/openadmet/cyp-challenge)
