# MolLIBRA-GP6 for the OpenADMET CYP Blind Challenge

## Submission summary

This initial CPU entry adapts the surrogate-model portion of MolLIBRA-G. The
test compounds are fixed by the challenge, so graph mutation and the molecular
diversity archive are intentionally not used.

Six molecular views are evaluated independently:

1. Morgan radius 2
2. Morgan radius 3
3. Feature Morgan radius 2
4. RDKit path fingerprint
5. Hashed atom-pair fingerprint
6. MACCS keys

Each view uses a fixed Tanimoto-kernel Gaussian process. A deterministic,
target-balanced subset limits validation fits to 600 and final fits to 900
training rows. This keeps exact Cholesky inference practical on a 16 GB CPU.

Validation scores are converted to convex ensemble weights. Every critic is
kept between 5% and 35%, a direct stability fix for the earlier experiment in
which one critic captured effectively all weight.

## Holdout results

One fixed seed-42 80/20 holdout is used for this initial submission. TEST
labels remain blinded.

| Regression endpoint | Validation MAE |
|---|---:|
| CYP1A2 | 0.6326 |
| CYP2C9 | 0.5829 |
| CYP2D6 | 0.6180 |
| CYP3A4 | 0.6921 |

| Classification endpoint | ROC AUC | MCC | Predicted-positive rate |
|---|---:|---:|---:|
| CYP2D6 TDI | 0.5757 | 0.0854 | 0.267 |
| CYP3A4 TDI | 0.7090 | 0.2679 | 0.264 |

Classification thresholds are chosen only on the holdout set. To avoid an
unstable MCC optimum that calls most compounds positive, the predicted-positive
rate must remain within 0.75–1.25 times the holdout prevalence.

## Reproducibility and disclosure

- Seed: 42
- Hardware: Apple M1 CPU, 16 GB RAM
- Runtime for final generation: 18.84 seconds
- Proprietary data: none
- Open-source code: this repository
- Full data and output SHA-256 values:
  `results/openadmet/openadmet_validation_report.json`

## Limitations

- The initial weights use one random holdout rather than repeated scaffold CV.
- Fixed-kernel GPs are bounded to 900 rows per critic.
- CYP2D6 TDI discrimination is weak on the holdout and should be treated as a
  transparent baseline, not a production predictor.
- No leaderboard feedback was used to choose features, weights or thresholds.
