---
license: cc-by-4.0
tags:
  - cheminformatics
  - admet
  - cyp450
  - benchmark
---

# OpenADMET CYP inhibition blind challenge, method report

**Team:** Mado Digital
**Tracks entered:** direct inhibition regression, and time dependent inhibition
classification
**Date:** 31 October 2026

## Data

Public data only. No proprietary data was used. Training came from the
challenge's released training set and public ChEMBL records. We hold no
unpublished CYP measurements.

## Direct inhibition, regression

pIC50 predicted for CYP1A2, CYP2C9, CYP2D6 and CYP3A4. Two arms were fitted
independently and blended per endpoint.

**Arm 1, gradient boosting.** Morgan fingerprints at radius 2 and 2048 bits plus
RDKit descriptors, into LightGBM 4.7.0, one configuration for all four
endpoints. 800 estimators, learning rate 0.03, colsample 0.35, subsample 0.85,
reg_lambda 1.0, leaves by `min(48, max(8, n_fit // 25))`.

**Arm 2, message passing neural network.** Directed message passing over bonds,
chemprop style, depth 3, hidden 300, plus 19 RDKit descriptors into a shared
trunk with four pIC50 heads. Dropout 0.1, batch size 64, schedule from 1e-4 up
to 1e-3 and back to 1e-4 over at most 120 epochs with patience 25, gradient
clipping at 5.0, trained multitask across the four endpoints.

Auxiliary training targets were used at weight 0.3: TDI pIC50, Emax, single
concentration response and the TDI boolean. These are **training set labels used
as auxiliary supervision**, which is ordinary multitask learning. They are not
read at prediction time, and the blinded test set does not carry them. That
distinction is what separates this arm from the excluded one described at the
end of this report.

**The split.** Bemis-Murcko scaffold folds over the 6,146 molecule training
union, scaffold families kept whole, five folds. Both arms import the identical
split rather than re-deriving it, and both record the same split digest
`01029922db48c7fe`. The stopping epoch for the network was chosen on an inner
scaffold bucket carved from the training folds only.

**Blending.** A per endpoint convex weight on the gradient boosting arm, fitted
on outer out of fold predictions only. Weights were 0.46 for CYP1A2, 0.37 for
CYP2C9, 0.33 for CYP2D6 and 0.46 for CYP3A4.

Everything that could leak was fitted inside the folds: the descriptor scaler,
the per endpoint loss normaliser, the head bias initialisation, the prediction
clip range and the auxiliary target scaler.

### Out of fold results, ST-RAE, lower is better

| endpoint | n | blend | GNN | GBM |
|---|---|---|---|---|
| CYP1A2 | 1412 | **0.76806** | 0.79842 | 0.79807 |
| CYP2C9 | 1285 | **0.61140** | 0.63996 | 0.67746 |
| CYP2D6 | 1493 | **0.86874** | 0.88386 | 0.92301 |
| CYP3A4 | 2335 | **0.47782** | 0.50646 | 0.51750 |
| macro | | **0.6815** | 0.70717 | 0.72901 |

The blend beats the better single arm on every endpoint. Against the GNN, which
was the stronger arm, the macro improvement is 0.02562 with a standard deviation
of 0.00307 across folds and a win rate of 1.0.

For scale, a predictor that returns the training mean scores an ST-RAE of
exactly 1.0 on this metric, which is the self check recorded in our run. So the
blend's 0.6815 is about 32 percent better than predicting the mean.

### The number above is optimistic, and here is by how much

**The gradient boosting configuration was chosen on the same out of fold
predictions it is scored on.** Twelve configurations were searched and 0.72901
is the minimum of the twelve, so it is a selected minimum rather than an
independent estimate. Our own run metadata records this rather than hiding it:
the grid ran from 0.72901 at best to 0.78198 at worst, a spread of 0.05297, and
the selected configuration beat the runner up by only **0.01033**.

A margin of 0.01 against a spread of 0.05 means the choice among the top
configurations is close to arbitrary, and the quoted figure should be read as a
ceiling on that arm rather than a clean out of fold number. The blend weights
were also fitted on out of fold predictions, so the blend's 0.6815 inherits the
same caveat.

What we did not do is select anything on the test set. The test predictions are
the mean of the fold models, and no statistic of the blinded 750 was computed at
any point.

## Time dependent inhibition, classification

True or false for a greater than two fold IC50 shift after preincubation,
scored for CYP3A4 and CYP2D6.

**Threshold rule, fixed before the test scores were seen.** The cut is the
argmax of MCC over all distinct cut points on the outer out of fold scores, and
it is applied unchanged to the test scores. The two sets are on one scale
because both come from the same five four fold models.

| | CYP2D6 | CYP3A4 |
|---|---|---|
| cut | -0.5028 | -1.8252 |
| out of fold MCC at that cut | 0.1409 | 0.3944 |
| bootstrap median MCC, 1000 draws | 0.1388 | 0.3923 |
| 95 percent interval | 0.0718 to 0.1409 | 0.3672 to 0.3944 |
| out of fold positive rate | 0.1583 | 0.4852 |
| test positive rate | 0.0973 | 0.5560 |

The positive rate sits above prevalence on CYP3A4 because MCC maximisation
favours recall at this signal level. It is a property of the rule and not a
calibration to the test set.

**One residual difference, measured rather than corrected.** An out of fold
score is a single model's output while a test score is the mean of five, and
averaging correlated models compresses the spread slightly. Comparing the out of
fold and test positive rates above is what that is worth here. Correcting it
would require computing a statistic of the blinded set, which we declined to do.

## What the labels can support, and where we fall against it

We estimated the reliability of the released TDI labels by simulating a repeat
of the same measurement and thresholding it by the same rule.

| | CYP2D6 | CYP3A4 |
|---|---|---|
| test-retest MCC, a lower bound on a model predicting the assay expectation | 0.577 | 0.518 |
| oracle MCC, an upper bound on any model predicting that expectation exactly | 0.725 | 0.655 |
| ours, out of fold | **0.1409** | **0.3944** |

Our CYP3A4 classifier is below the test-retest floor. Our CYP2D6 classifier is
far below it and we do not regard it as useful. We are reporting it because
withholding the weaker of two endpoints would misrepresent the method, and
because a blind challenge is the right place to find out.

Accuracy also varies strongly with distance from the labelling threshold. In the
band within half a standard deviation of the cut, a majority class baseline
scores an MCC of -0.1452 on CYP2D6. Compounds near the threshold are close to
unlearnable from these labels.

## What we are not claiming

Nothing here is a prospective result. No compound was made or measured on the
strength of these predictions. The models are conventional and the architectures
are standard. The only thing we would ask a reader to weigh is that the
thresholds, the blend weights and the arm selection were all fixed on out of
fold data, and that we have reported the endpoint we expect to do badly.

## Code and disclosures

**Proprietary training data: none.** All training data is public.

**Code: not publicly available at this time.** We are a two person company
without funding and the code stays private for now. The open source box on the
submission form is therefore unchecked. Everything needed to judge the method is
in this report: the architectures, the hyperparameters, the blend weights, the
threshold rule and the out of fold numbers including the endpoint we expect to
do badly.

One detail that belongs in the open regardless. An auxiliary gradient boosting
arm was built and measured and then excluded, because it reads measured Emax and
single concentration columns that the blinded test set does not carry. It was
kept in our repository with that reason recorded against it rather than deleted,
and it took no part in these predictions.
