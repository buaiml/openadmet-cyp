# Method report: OpenADMET CYP Inhibition Blind Challenge

## Entry scope

This entry addresses both the direct-inhibition regression track and the time-dependent-inhibition classification track. It uses only the official released challenge data and no proprietary training data.

## Molecular representation

Every structure is parsed with RDKit and represented by 217 two-dimensional descriptors plus a 2,048-bit, radius-2 Morgan fingerprint with chirality enabled. Invalid structures cause the pipeline to stop. Descriptor generation is performed once over the released labelled and blind structure pool; a label-free pooled median is available for non-finite values. All 217 descriptors were finite for all 6,895 structures used in this run, so imputation did not affect the fitted models or blind predictions.

Generic Bemis–Murcko scaffolds provide the grouped stress-test labels. Local models use chiral ECFP4 Tanimoto similarity.

## Regression models

Each measured CYP endpoint is fitted separately because the training response matrix is sparse. The final ensemble contains:

1. a base LightGBM regressor trained with equal row weights;
2. a second, otherwise identical LightGBM regressor that moderately upweights more precise measurements;
3. an eight-neighbour Tanimoto-weighted local regression estimate, shrunk toward the endpoint training mean as maximum similarity falls.

For lower and upper response confidence limits, the second model uses:

```text
width = conf_high - conf_low
raw = 1 / max(width, 0.10)^0.5
sample_weight = clip(raw / mean(raw), 0.30, 3.00)
```

Weights are derived only from each CV training fold; the full-data fit uses all labelled rows for the endpoint. The selected `(base, CI-weighted, local)` blend is:

| Endpoint | Base | CI-weighted | Local |
|---|---:|---:|---:|
| CYP1A2 | 0.500 | 0.375 | 0.125 |
| CYP2C9 | 0.375 | 0.625 | 0.000 |
| CYP2D6 | 0.625 | 0.125 | 0.250 |
| CYP3A4 | 0.375 | 0.625 | 0.000 |

The final global prediction for each component is the mean of models seeded 42, 314 and 2718. Predictions are clipped to the observed endpoint response range.

## TDI models

Separate LightGBM classifiers and ten-neighbour local Tanimoto estimates were screened for CYP2D6 and CYP3A4. Blend weight and classification threshold were selected on seeded stratified random OOF predictions by Matthews correlation coefficient, then applied unchanged to a generic-scaffold stress test.

The final CYP2D6 TDI model is the local component alone at threshold 0.26. ExtraTrees, ExtraTrees/local blends, pooled-isoform models and continuous-potency-shift models failed to improve the frozen-threshold scaffold result. The final CYP3A4 model is LightGBM alone at threshold 0.18. Full-data LightGBM probability predictions average the same three seeds.

## Validation design

Four complementary views are retained:

- **Random tuning OOF:** used for recipe/threshold selection. Scores are selection-optimistic and are not unbiased performance estimates.
- **Generic-scaffold stress:** applies the selected parameters unchanged to grouped folds. It tests structural robustness but is not fully nested.
- **Strict close-analogue repeats:** retains potent leaders in training while holding out only associated molecules with leader Tanimoto at least 0.40. Five repeats contain 202–241 validation molecules each.
- **Exact-750 coverage repeats:** retains 75 potent leaders and holds out 750 rows per repeat while approximately matching the blind set's nearest-neighbour density. Nominated leader relationships are often weak, so this is a size/density check rather than a pure close-leader test.

All ten analogue splits have zero train/validation connectivity-key overlap. Their construction uses the released blind SMILES nearest-neighbour distribution for label-free calibration but never uses blind labels. Repeats overlap, so paired tests and standard deviations are exploratory rather than independent-sample uncertainty estimates.

The local validation metric is the arithmetic mean across endpoints: MA-ST-RAE for regression and macro MCC for TDI. The public leaderboard additionally uses bootstrap resampling; local point estimates should not be interpreted as leaderboard scores.

## Regression results

The exact-main five-fold comparison was:

| Endpoint | Previous random | Final random | Previous scaffold | Final scaffold |
|---|---:|---:|---:|---:|
| CYP1A2 | 0.80794 | 0.79524 | 0.80687 | 0.79118 |
| CYP2C9 | 0.69151 | 0.67893 | 0.68999 | 0.67776 |
| CYP2D6 | 0.90939 | 0.90875 | 0.91119 | 0.91094 |
| CYP3A4 | 0.51203 | 0.49368 | 0.51991 | 0.50517 |
| **MA-ST-RAE** | **0.73022** | **0.71915** | **0.73199** | **0.72126** |

Lower is better. Every endpoint was non-worse in both views; the largest gain was CYP3A4.

The principal challenge-matched comparison used the preconfirmed recipe with a 0.500/0.500 CYP2C9 global blend, followed by an exact CYP2C9 refinement:

| Regime | Previous recipe | CI-weighted recipe | Difference | CI wins |
|---|---:|---:|---:|---:|
| Strict close analogue, five repeats | 0.7532 ± 0.0687 | 0.7431 ± 0.0673 | -0.0101 | 4/5 |
| Exact-750 coverage, five repeats | 0.7268 ± 0.0169 | 0.7138 ± 0.0174 | -0.0130 | 5/5 |
| Combined ten comparisons | 0.7400 | 0.7284 | -0.0115 | 9/10 |

Changing CYP2C9 from 0.500/0.500 to the final 0.375/0.625 base/CI blend improved its mean analogue-split endpoint score by another 0.00083 across ten comparisons. Removing the 0.125 local term from CYP1A2 worsened its combined analogue score by 0.00213, so that local term was retained.

## TDI results

| Endpoint / summary | Random tuning OOF | Generic-scaffold stress |
|---|---:|---:|
| CYP2D6 MCC | 0.157 | 0.122 |
| CYP3A4 MCC | 0.413 | 0.396 |
| Macro MCC | 0.285 | 0.259 |

For CYP2D6, the retained local model's scaffold MCC was 0.12168, compared with 0.11969 for ExtraTrees and 0.11819 for the tested ExtraTrees/local blend under random-selected, frozen thresholds. CYP2D6 remains the weakest endpoint.

## Data, validation and provenance

The regression and TDI training files are model inputs. Emax and single-concentration files are audited but are not used for fitting. No exact blind SMILES occur in either fitted training table.

Both submission files pass the organizers' official validators plus stricter checks for exact columns, exact ordered `Molecule_Name`/`SMILES` pairs, finite nonconstant regression predictions and both Boolean classes for each TDI endpoint. Input files, outputs, entry source and requirements are SHA-256 hashed in `reports/run_metadata.json`, which also records the exact dataset/tutorial commits and runtime versions.

## Limitations

The model is ligand based and does not explicitly represent CYP protein structure, reactive metabolites or probe-substrate effects. OOF folds use one deterministic seed per fold, whereas final global inference averages three full-data seeds; CV therefore approximates but does not exactly reproduce the submitted seed ensemble. The estimates are comparative model-development evidence, not clinical validation. Predictions require experimental confirmation and must not be used alone for clinical drug–drug interaction decisions.
