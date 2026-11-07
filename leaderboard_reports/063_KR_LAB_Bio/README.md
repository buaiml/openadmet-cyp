---
license: cc-by-4.0
language:
  - en
tags:
  - chemistry
  - drug-discovery
  - admet
  - cytochrome-p450
  - pic50-regression
  - openadmet
pretty_name: CYP Direct Inhibition pIC50 Method Report (OpenADMET Blind Challenge)
---

# Predicting CYP Direct Inhibition pIC50: One Pooled Gradient-Boosting Model over Four Isoforms, with Public Auxiliary Measurements

**OpenADMET CYP Inhibition Blind Challenge - Method Report**
Regression (direct inhibition) track - Leaderboard alias **KR LAB Bio** - KR LAB Bio Inc., Republic of Korea
Draft v1, 2026-10-30. No proprietary data was used. No test-set labels were used at any step.

A companion report covers our classification (TDI) submission:
https://huggingface.co/krho/openadmet-cyp-tdi-method-report

---

## 1. Summary

We predict pIC50 for direct inhibition of CYP1A2, CYP2C9, CYP2D6 and CYP3A4 on the 750
blinded test compounds. The submission is a **single LightGBM regressor** trained on
(compound, isoform) rows with an isoform one-hot, so the four endpoints share one model, plus
34,586 public auxiliary measurements entered as extra training rows at a reduced sample
weight behind their own source flag. Cross-validated macro-averaged ST-RAE under a Murcko
scaffold split is **0.7227** (bootstrap standard deviation 0.0102), from per-isoform ST-RAE
of 0.801 (CYP1A2), 0.676 (CYP2C9), 0.916 (CYP2D6) and 0.496 (CYP3A4).

This is a deliberately modest entry, and we would rather say so than dress it up. Our effort
in this challenge went into the classification track; the regression model is the best of
twenty configurations we screened in one sitting, with no ensembling, no pretrained
representations in the submitted model, and no architecture search. We report it because the
measurement we did make is worth more than the rank it buys, and because a method report is
the condition for appearing on the final leaderboard at all.

The finding we think is worth reading is **where ST-RAE allows a model to gain, and where it
does not** (section 3). ST-RAE divides a prediction's distance outside the measured credible
interval by the same quantity for a constant mean predictor. We measured both terms per
isoform. On CYP3A4 our model lands inside the interval **2.47 times** as often as the mean
predictor does. On the other three isoforms the ratio is 1.05, 1.17 and 1.06. In other words
our macro-averaged score is carried almost entirely by one of four equally weighted
endpoints, and the reason is a property of the measurement intervals rather than of the
model: CYP3A4's intervals have an upper quartile width of 1.668 log units against 0.358 for
CYP2D6. A model cannot be rewarded for precision the assay did not have, and it cannot be
rewarded for precision the assay had in such abundance that a constant answer already lands
inside. We did not know this when we started, and knowing it would have changed where we
spent the week.

Two configuration choices account for most of the gap between the organisers' tutorial
baseline (MA-ST-RAE 0.8257) and our submission (0.7227), and both are choices we rejected for
the classification track:

1. **Pooling the four isoforms into one model** rather than fitting one regressor per
   isoform: 0.7615 to 0.7508 with fingerprints held fixed. The same pooling hurts TDI
   classification, so our two submissions differ here on purpose.
2. **Public auxiliary measurements as extra training rows**: 0.7508 to 0.7227, the single
   largest improvement we found. The same data did not improve TDI classification in any
   configuration we measured, because those assays do not share the challenge's definition
   of a time-dependent shift.

One change we measured and did **not** adopt is recorded in section 9, because the way it
failed is instructive. Averaging three LightGBM seeds improves MA-ST-RAE by 0.0037 with a
paired t of 61.95 and a consistent sign in 97.8% of bootstrap samples. The effect is
unambiguously real and unambiguously too small: it misses the threshold we registered before
measuring it, and it changes no rank on the leaderboard as it stood (section 8). We submit
the single-seed model.

## 2. Task and data

| Item | Value |
|---|---|
| Predicted endpoints | `CYP1A2`, `CYP2C9`, `CYP2D6`, `CYP3A4` pIC50 direct inhibition |
| Metric | Macro-averaged ST-RAE, lower is better, 1.0 equals a constant mean predictor |
| Test set | 750 compounds, blinded |
| Training compounds | 4,901 with at least one dose-response pIC50 |
| Measured (compound, isoform) pairs | 6,521 |
| Compounds measured on 1 / 2 / 3 / 4 isoforms | 3,592 / 1,039 / 229 / 41 |

Per isoform, the training set and the shape of its measurements:

| Isoform | Compounds | pIC50 mean | pIC50 sd | pIC50 range | Interval width q25 / median / q75 |
|---|---|---|---|---|---|
| CYP1A2 | 1,412 | 4.955 | 1.030 | 1.91 to 7.95 | 0.221 / 0.328 / 0.579 |
| CYP2C9 | 1,285 | 4.581 | 0.782 | 2.10 to 7.47 | 0.324 / 0.526 / 0.887 |
| CYP2D6 | 1,490 | 4.782 | 0.914 | 1.95 to 7.53 | 0.208 / 0.272 / 0.358 |
| CYP3A4 | 2,334 | 4.096 | 1.093 | 1.91 to 7.19 | 0.197 / 0.379 / 1.668 |

**Auxiliary public data.** We assembled 85,491 measurements over 29,798 compounds from
PubChem AID 1851, ChEMBL 37 CYP assays, and the Octant blog release, after removing any
compound that overlaps the blinded test set. The submitted model uses the **34,586 rows whose
activity relation is an equality** (19,961 compounds), dropping the 50,225 censored rows
reported as a bound. Keeping the censored rows behind a censored flag scored 0.7250 against
0.7227 for dropping them, so we drop them. Per isoform the retained rows are CYP3A4 13,145,
CYP1A2 7,814, CYP2C9 7,356 and CYP2D6 6,271; by source, PubChem 18,138, ChEMBL 15,423 and
Octant 1,025.

Auxiliary rows enter a fold's training side only, and any auxiliary row whose compound
matches a held-out compound is dropped from that fold, so the auxiliary measurements for
compounds the challenge training set already contains cannot leak across the split.

**Test set construction, from the organisers' own description.** Hit expansion was performed
on the top 25 hits from CYP1A2, CYP2C9 and CYP3A4 by purchasing the top 10 ECFP4-nearest
catalogue analogues per hit, and the resulting 750 compounds were then assayed against all
four CYPs. Two consequences matter here. The test set is 75 parent series of 10 analogues
rather than a diverse set, and its compounds sit closer to the training data than a scaffold
split implies, so our cross-validated figure should be pessimistic. On the classification
track we were able to check that expectation against the leaderboard: cross-validation read
MA-MCC 0.3092 and the blinded test set gave 0.3384 at the same operating point, about 0.029
in our favour. We expect the same direction here and do not know the size.

## 3. What ST-RAE rewards, and on which isoform

This section is the measurement we would keep if we could keep only one.

ST-RAE treats a prediction inside the measured credible interval as zero error, and outside
it charges only the distance to the nearest bound. The denominator applies the same rule to
a constant predictor equal to the mean of the observed values. So the score answers a
specific question: **how much more often, and by how much, does the model land inside the
interval than a constant answer does.**

We computed both terms separately on the out-of-fold predictions of the submitted
configuration.

| Isoform | ST-RAE | R² | MAE | Inside interval, model | Inside interval, mean predictor | Ratio | Denominator sum | Our error sum |
|---|---|---|---|---|---|---|---|---|
| CYP1A2 | 0.801 | 0.303 | 0.642 | 0.238 | 0.227 | 1.05 | 694.0 | 555.9 |
| CYP2C9 | 0.676 | 0.384 | 0.468 | 0.429 | 0.366 | 1.17 | 403.2 | 272.7 |
| CYP2D6 | 0.916 | 0.134 | 0.593 | 0.200 | 0.189 | 1.06 | 622.2 | 569.8 |
| CYP3A4 | 0.496 | 0.591 | 0.540 | 0.378 | **0.153** | **2.47** | 1222.8 | 606.9 |

Three readings follow, and the third is the one that would have changed our plan.

**The macro average is carried by one endpoint.** CYP3A4 contributes 0.496 to a four-way
mean in which the other three contribute 0.801, 0.676 and 0.916. Removing CYP3A4 would move
our macro ST-RAE from 0.7227 to 0.798, which is roughly the organisers' tutorial baseline.
Every gain we can point to is a CYP3A4 gain.

**R² and ST-RAE disagree, and the disagreement is informative.** CYP2D6 has positive R²
(0.134) and an ST-RAE of 0.916: the model does carry real signal, and the metric barely
credits it. The mechanism is in the interval widths. CYP2D6's intervals are the narrowest in
the dataset (median 0.272, upper quartile 0.358) while our mean absolute error there is
0.593. Predictions therefore almost never land inside, the soft threshold almost never
activates, and ST-RAE collapses towards plain relative absolute error. The constant predictor
suffers the same way, so the ratio approaches one. **A narrow interval does not make an
endpoint harder to predict; it makes an endpoint harder to be rewarded for predicting.**

**Where a model can actually gain is where the intervals are wide and the values are
dispersed.** CYP3A4 has both: the widest upper-quartile interval (1.668) and the largest
pIC50 spread (sd 1.093, 40.4% of compounds below pIC50 4). Wide intervals give the model
somewhere to land; a dispersed distribution keeps the constant predictor from landing there
too, since the mean of a bimodal active-inactive distribution sits between the modes. That
combination is why the ratio is 2.47 on CYP3A4 and near unity elsewhere. CYP2C9 shows the
other failure mode: its intervals are comfortably wide (median 0.526) and its values are the
least dispersed (sd 0.782), so the constant predictor is already inside 36.6% of the time and
the available headroom is small.

The practical consequence is that a macro-averaged metric over four equally weighted
endpoints concealed, from us, that three of the four offered almost no headroom to a model of
our accuracy. Had we measured this on day one, we would have spent the week either on CYP3A4
specifically or on a different track.

## 4. Features

Two blocks, identical to what our classification baseline uses:

- **217 RDKit descriptors** via `useful_rdkit_utils.RDKitDescriptors`, matching the
  organisers' tutorial baseline. Missing values are filled with the training median.
- **Morgan count fingerprint**, radius 2, 2048 bits, kept as counts, with bits present in
  fewer than five training compounds pruned. 2,046 bits survive.

For the pooled model the design matrix additionally carries a 4-wide isoform one-hot, a
4-wide source one-hot (challenge, ChEMBL, PubChem, Octant) and a censored flag, giving 2,272
columns in total. Auxiliary rows carry their own source position; test rows and challenge
training rows carry the challenge position and a zero censored flag.

We tested three pretrained molecular representations (ChemBERTa MLM, ChemBERTa MTR,
MoLFormer) on this task. None improved it, and the two configurations that replace
descriptors with embeddings are the worst two of our twenty (0.8268 and 0.8375). Section 7
gives the numbers. **The submitted model uses no pretrained representation.**

## 5. Model

One `LGBMRegressor` over all (compound, isoform) rows, with these settings:

```
n_estimators=800, learning_rate=0.03, num_leaves=63, min_child_samples=10,
colsample_bytree=0.4, subsample=0.8, subsample_freq=1, reg_lambda=1.0, random_state=0
```

Challenge rows carry sample weight 1.0; auxiliary equality rows carry 0.3. The weight was
not tuned: 0.3 was chosen once as a plausible discount for a different assay protocol and
never revisited, which we note as an untested free parameter rather than a result.

The isoform one-hot is what makes pooling work: the model can share structure-activity
signal across endpoints while still shifting its prediction per isoform. We did not attempt
a multi-output formulation, a shared-trunk neural network, or any form of isoform-specific
calibration on top of the pooled predictions.

## 6. Validation protocol

Out-of-fold predictions over all 4,901 training compounds are collected into one pseudo test
set and scored the way the leaderboard scores a submission: per endpoint, 1,000 bootstrap
resamples at fixed seed 0, macro-averaged per bootstrap sample into the reported MA row.
Our metric implementation was checked against the organisers'
`custom_scoring_functions.py` on 200 random inputs and on the training data, with a maximum
absolute difference of 0 and identical bootstrap index matrices.

Folds are **5-fold grouped by Murcko scaffold**, which is the pessimistic choice: random
folds score 0.7453 and Butina clustering 0.7494 on the same configuration that scores 0.7508
under scaffolds. We report the scaffold figure throughout.

Repeating the submitted configuration under three different fold assignments gives 0.7227,
0.7287 and 0.7246, a spread of 0.006 that is within the 0.0102 bootstrap standard deviation.
Differences below about 0.010 in the table of section 7 should not be read as real.

## 7. What each configuration choice is worth

Twenty configurations, all scored under the same 5-fold scaffold split and the same bootstrap
protocol. The submitted configuration is the first row.

| Configuration | MA-ST-RAE | sd | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 |
|---|---|---|---|---|---|---|
| **Pooled + fingerprint + auxiliary, censored dropped (submitted)** | **0.7227** | 0.0102 | 0.801 | 0.676 | 0.916 | 0.496 |
| Same, fold seed 2 | 0.7246 | 0.0101 | 0.793 | 0.682 | 0.917 | 0.505 |
| Same, censored auxiliary rows kept | 0.7250 | 0.0101 | 0.807 | 0.677 | 0.916 | 0.499 |
| Same, fold seed 1 | 0.7287 | 0.0099 | 0.796 | 0.679 | 0.931 | 0.508 |
| Same, PubChem only, censored kept | 0.7304 | 0.0101 | 0.807 | 0.683 | 0.922 | 0.507 |
| Same, plus ChemBERTa MTR embeddings | 0.7335 | 0.0099 | 0.819 | 0.692 | 0.911 | 0.511 |
| Same, ChEMBL only | 0.7349 | 0.0105 | 0.806 | 0.688 | 0.937 | 0.508 |
| Same, plus ChemBERTa MLM embeddings | 0.7364 | 0.0099 | 0.807 | 0.701 | 0.919 | 0.516 |
| Pooled + fingerprint, no auxiliary, random folds | 0.7453 | 0.0110 | 0.823 | 0.670 | 0.973 | 0.513 |
| Pooled + fingerprint, no auxiliary, Butina folds | 0.7494 | 0.0109 | 0.821 | 0.695 | 0.964 | 0.517 |
| Pooled + fingerprint, no auxiliary, scaffold folds | 0.7508 | 0.0111 | 0.826 | 0.689 | 0.975 | 0.513 |
| Per-isoform + fingerprint | 0.7615 | 0.0110 | 0.840 | 0.710 | 0.968 | 0.527 |
| Per-isoform, descriptors only, tuned | 0.7979 | 0.0114 | 0.866 | 0.754 | 0.997 | 0.574 |
| Tutorial baseline, random folds | 0.8253 | 0.0123 | 0.910 | 0.771 | 1.029 | 0.590 |
| Tutorial baseline, Butina folds | 0.8257 | 0.0121 | 0.899 | 0.781 | 1.026 | 0.595 |
| Tutorial baseline, scaffold folds | 0.8257 | 0.0127 | 0.905 | 0.783 | 1.029 | 0.586 |
| Embeddings replacing descriptors, MTR | 0.8268 | 0.0095 | 0.889 | 0.806 | 0.964 | 0.649 |
| Tutorial baseline, scaffold seed 2 | 0.8270 | 0.0124 | 0.887 | 0.779 | 1.038 | 0.604 |
| Tutorial baseline, scaffold seed 1 | 0.8275 | 0.0123 | 0.897 | 0.776 | 1.050 | 0.586 |
| Embeddings replacing descriptors, MLM | 0.8375 | 0.0098 | 0.879 | 0.822 | 0.997 | 0.651 |

Reading the ladder from the bottom:

- **Hyperparameters**: 0.8257 to 0.7979, the largest single step after the auxiliary data.
  The tutorial's `LGBMRegressor()` defaults are not a strong baseline.
- **Morgan fingerprint added to descriptors**: 0.7979 to 0.7615.
- **Pooling the four isoforms**: 0.7615 to 0.7508. Small, and the opposite sign to what the
  same choice does on TDI classification.
- **Auxiliary public measurements**: 0.7508 to 0.7227. Both sources contribute and neither is
  sufficient alone: ChEMBL only reaches 0.7349, PubChem only 0.7304, both together 0.7227.
  The gain lands mostly on CYP2D6 (0.975 to 0.916), the isoform with the fewest challenge
  labels and the one the metric rewards least, which is an unlucky combination.
- **Pretrained embeddings**: nothing. Appended to descriptors and fingerprints they cost
  0.011 to 0.014; replacing descriptors they cost 0.104 to 0.115.

## 8. What we did not do, and why we stopped

We deliberately did not pursue three directions that would normally be next. In each case
the reason is a measurement rather than a preference.

**No ensembling.** Our classification work found the same pattern five separate times: a
component that improves a single model contributes nothing once it is added to an ensemble,
with paired t statistics indistinguishable from zero. Before building a regression ensemble
we checked what one could buy. On the live leaderboard snapshot of 2026-10-30 09:30 UTC, 231
of 276 entries show no method report link and 8 more show an invalid one, leaving 37 entries
that would be eligible for a final leaderboard on the stated requirement. Among those 37,
**the interval from 0.6404 to 0.7497 contains no entry at all.** Our 0.7227 sits 16th of 37,
and so would any score down to 0.6405: the next place up requires 0.6404, which is 0.08
better than our submission and more than ten times the largest gain any configuration change
in section 7 produced. Against the full 276-entry board the same improvement would move us
from 122nd to 108th, so the conclusion depends on whether the final board is restricted to
entries with a report; we read the requirement as saying it is, and note that as a reading
rather than a certainty. Either way the available gain is not an ensembling gain. We
registered a pass criterion for a seed-averaged variant before measuring it and report the
outcome in section 9.

**No metric-aware objective.** ST-RAE's soft threshold suggests training against distance to
the interval rather than to the point estimate. We did not try it. The reason it is unlikely
to pay is section 3: on three of four isoforms our errors are large compared with the
interval widths, so an interval-aware loss reduces to the absolute-error loss we already use,
and on the fourth the model is already inside 37.8% of the time.

**No attempt to exploit the test set's series structure.** The 75 parent series of 10
analogues invite within-series smoothing. On the classification track we measured whether the
series structure could be recovered from structure alone and it could not: an ECFP4
similarity graph over the test set gives 213 components at threshold 0.55 and a single
238-compound component at 0.40, because the parent compounds resemble each other and the
series interpenetrate before they separate. The same graph underlies both tracks.

## 9. Pre-registered check: does averaging model seeds help

Registered before measuring, as our classification work did for every submission change:
hold the fold assignment at scaffold seed 0, vary only the LightGBM `random_state` over 0, 1
and 2, average the out-of-fold predictions, and compare against the single seed with the
paired bootstrap. Adopt the average only if the mean paired difference in MA-ST-RAE is at
least 0.005 in the averaged model's favour **and** at least 97.5% of the 1,000 paired
differences point the same way. This is a variance reduction rather than new information, so
it is not the pattern section 8 describes, and it costs three training runs.

**Result: the criterion fails on effect size, not on significance.** Averaging three model
seeds moves MA-ST-RAE from 0.7227 to 0.7190, a paired improvement of **0.0037** with
**t = 61.95** and the same sign in **97.8%** of the 1,000 bootstrap samples. The sign
condition passes and the size condition does not: 0.0037 against the registered 0.005. Per
isoform the averaged model gives 0.800, 0.672, 0.906 and 0.496, so the gain is concentrated
on CYP2D6, the endpoint section 3 shows the metric rewards least.

**We submit the single-seed model.** The pre-registered threshold exists so that a decision
is not made after seeing the number, and we are keeping to it. We also think it is the right
call on the merits: 0.0037 is a third of the bootstrap standard deviation and, by section 8,
changes no rank on either reading of the board. Reporting a tiny effect with a huge t
statistic is worth more here than acting on it, because it shows how little the paired test's
precision has to do with whether a change matters. At this one fold assignment the three
individual model seeds give point ST-RAE of 0.7224, 0.7287 and 0.7274, and their average
lands at 0.7186, below the best of the three: the ensemble is doing real variance reduction.
It is simply too small to buy anything.

A side effect of this run is a reproducibility check we wanted anyway. Its single-seed arm
reproduced MA-ST-RAE 0.7227 and bootstrap standard deviation 0.0102 to four decimal places
against the configuration file written days earlier, confirming that the code path producing
the submitted predictions is the one that produced the cross-validated figure we report.

## 10. Reproducibility and scope

Fixed seeds throughout. Fold assignments are 5-fold grouped Murcko scaffolds at seed 0.
Feature computation takes 73 seconds for 35,391 compounds; one cross-validation run of the
submitted configuration takes 195 seconds; the final fit over all training data plus
auxiliary rows takes 41,107 rows by 2,272 columns and under 60 seconds. All computation ran
on commodity hardware with a peak of 1.5 GB.

The submitted predictions were produced by a script that trains the configuration above on
every available training row with no hold-out and predicts the 750 test compounds, then runs
the organisers' own `activity_validation.py` and, separately, the checks the submission Space
itself applies: 750 rows, the four required value columns, all values finite, and each
column's standard deviation above the Space's minimum of 0.01. Submitted column standard
deviations are 0.635, 0.629, 0.444 and 0.808.

No proprietary data, no undisclosed measurements, and no test-set labels enter any step. The
auxiliary measurements are public (PubChem AID 1851, ChEMBL 37, the Octant blog release) and
are disclosed here in full, including which rows were dropped.

Scope limits we want on the record. The auxiliary sample weight of 0.3 was never tuned. We
did not calibrate the pooled model's predictions per isoform. We did not ensemble. We screened
twenty configurations in one sitting and selected the best by cross-validated macro ST-RAE,
which with a bootstrap standard deviation of 0.0102 and twenty candidates risks selecting
partly on noise; the three-fold-seed spread of 0.006 in section 6 is the honest bound on how
much of our margin over the next configuration is real.

## 11. Contact

KR LAB Bio Inc., Republic of Korea. Leaderboard alias: KR LAB Bio.
