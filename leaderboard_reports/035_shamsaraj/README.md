# OpenADMET CYP Inhibition Blind Challenge 2026 — method reports

Method reports for the three submission tracks.

**Author:** Jamal Shamsara

**Final results** (leaderboard released 2026-11-04, restricted to entrants with a method report):

| Track | Report | Submitted file | Final score | Rank |
|---|---|---|---|---|
| Direct inhibition (regression) | [**regression.md**](regression.md) | `25_0_per_endpoint_best_regression_submission.csv` | MA-ST-RAE **0.4593** | 35 / 103 |
| Time-dependent inhibition (classification) | [**classification.md**](classification.md) | `10_25_majority_vote_classification_submission.csv` | MA-MCC **0.3232** | 32 / 62 |
| CYP3A4 structure prediction (pose) | [**pose.md**](pose.md) | `24_11_frozen_structures.zip` | LDDT-PLI **0.3608** | 34 / 38 |

Where an earlier version of these reports quoted a different number (regression 0.4749,
classification 0.3846), that was the interim-subset scoring of the same predictions; the
final board scores the full 750-compound test set.

The pose track's ground-truth structures were released after the challenge closed, so
[**pose.md** §11](pose.md) additionally scores **every** pose submission — including two that
were built and never uploaded — by ligand RMSD against the deposited structures. The
per-compound table is [`pose_rmsd_table.csv`](pose_rmsd_table.csv).

Each report describes the **single best-scoring submission** for its track, not the full
exploration behind it. Where a number is quoted it is the organisers' own blind score unless
it is explicitly labelled as internal.

## Common ground

All three tracks were approached from one principle, which the challenge's design makes
unavoidable: **a local score is a hypothesis about the blind score, and most of them are
wrong.** The published training data and the blinded test set differ in label distribution,
assay condition and chemical space, so a model selected purely on cross-validation tends to
be mis-centred on the test set even when it ranks compounds correctly. Every selection
decision reported here was therefore either (a) made on a locally validated protocol whose
agreement with blind placement had itself been measured, or (b) made on already-scored blind
results and labelled as such.

Nothing in these reports used the blinded labels. Where a submission reuses a column from an
earlier submission of ours, that is stated and the earlier blind score is given, because the
choice was informed by it.

## Software

RDKit, scikit-learn, PyTorch, Chemprop (D-MPNN), XGBoost / HistGradientBoosting, OpenEye
(Spruce, POSIT, Omega, QUACPAC), AutoDock Vina 1.2.7 + Meeko, xTB (GFN2), PoseBusters, and
the open-source [`mlmolprop`](https://github.com/shamsaraj/mlmolprop) package for the shared
descriptor, split and metric machinery.

## Contact

Jamal Shamsara
