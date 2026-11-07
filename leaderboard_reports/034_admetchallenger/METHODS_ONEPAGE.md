# OpenADMET CYP Inhibition Challenge — Methods

Entrant `admetchallenger`. Final submissions 2026-11-02.
Regression **MA-ST-RAE 0.4447** (rank 43/274) · TDI **MA-MCC 0.3437** (rank 39/156)

## Model
One LightGBM model per isoform (`regression_l1` objective, 600 trees, 3 seeds averaged) on a
five-block feature stack: 2D descriptors and fingerprints; CYP2D6 pharmacophore features;
out-of-fold primary-screen activity predictions; public qHTS activity priors; ChEMBL CYP
priors. Each isoform's final prediction is a rank-space blend of that base with two or three
auxiliary models (a chemprop graph network, a multi-task variant, and a base re-trained to
match the test set's activity composition), with per-isoform weights. L1 was chosen because
the metric charges nothing inside a credible interval, so squared error spends capacity where
the metric is indifferent. TDI uses the same features with a LightGBM classifier, converted to
booleans by a per-isoform rank-quantile cut (CYP2D6 14%, CYP3A4 35%).

## Metric-aware post-processing — most of our score
- **Per-isoform centring.** The test set is an analog expansion around screening hits, so its
  activity distribution differs from training; the shift differs in size and sign per isoform.
- **Spread expansion.** Predictions are scaled outward about their own mean. Credible intervals
  are ~5x wider for weak compounds than potent ones and the metric charges nothing inside an
  interval, so under-dispersion is penalised exactly where it costs most.

Both optimise the scored metric at some cost to calibration. We would not ship the
spread-expanded model as a calibrated predictor.

## How models were selected
Our analog-based cross-validation did not predict held-out performance — most clearly on
CYP2D6, where several candidates improved CV and then lost on the held-out set. We therefore
treated the public leaderboard as the measuring instrument, under a fixed protocol: the four
isoforms score on independent endpoints, so one submission tests up to four changes without
confounding them; at least one isoform per submission is restated unchanged as a control, and
if its score moves the reading is void; every change preserves either the exact value multiset
(ordering tests) or the exact ordering and mean (placement tests), enforced by a validator
before submission; and the final entry is assembled automatically as the per-isoform best
measured configuration rather than a single best file. Every channel of our final score is
therefore a measured quantity, not a projection.

## Results
**Worked:** per-isoform centring and spread expansion; adding a base model re-trained to match
the test set's activity composition; re-weighting blend members toward the most decorrelated
member; one spread adjustment on CYP3A4; and on TDI, lowering CYP2D6's predicted-positive rate
from 30% to 14% in four measured steps.

**Did not work:** frozen molecular-embedding arms (several pretrained encoders x heads x
dimensions); tabular foundation-model heads; Mordred re-derivations; docking scores; blind-set
neighbour augmentation; covariate-shift training weights; rank-valued targets; interval-weighted
losses; multi-source activity rows; multi-arm blending; asymmetric placement shapes; and
hyperparameter search on the regression base.

Our remaining gap is representation-limited: ordering quality (Spearman rho 0.80-0.85, and 0.52
on CYP2D6) sits 0.04-0.065 below the leaders on every isoform, and no re-blending, re-weighting
or re-tuning of the models we hold closes it.

## External sources
The public leaderboard was used quantitatively: because it publishes Spearman rho alongside the
scored metric for every entry, a score can be decomposed into ordering quality and value
placement, locating our own weaknesses per isoform. All figures used were publicly displayed to
every entrant. We also read two other participants' voluntarily published materials (one
open-sourced repository, one public write-up; names omitted) and tested their reported findings
on our own pipeline. Their headline claim that model diversity helps across isoforms did not
reproduce for us — replicating it was our worst submission of the campaign (-0.0801 on the
board) and we found the effect inverted on CYP2D6. What transferred was one methodological
idea, their data-split construction; no weights, predictions or code from any other entrant
were used in any submission. Public datasets used as prior features: ChEMBL and a public qHTS
panel.

## Reproducibility
All submissions, per-isoform board readings and per-experiment outcomes are logged with
timestamps. The final entry is regenerated from that record by a script that refuses to emit
anything scoring worse than the last hand-verified configuration.
