# OpenADMET CYP Inhibition Blind Challenge — method report

Entry covering the **Direct Inhibition (Regression)** and **TDI (Classification)** tracks.
Structure-prediction track not entered.

**Submitted files.** Regression track: `submission_regression_dmpnn.parquet` — the D-MPNN blend
described in section 16, *not* the ensemble of sections 1–13. TDI track:
`submission_classification.parquet`, unchanged from section 9 and uploaded separately.

**External data disclosure.** Frozen embeddings from the publicly available CheMeleon pretrained
molecular representation, used as baseline input features; the graph model itself is trained from
scratch with no CYP-supervised public checkpoint. No proprietary data. Public ChEMBL bioactivity
data were evaluated during development and **rejected** (section 5); they are not used in the
submitted regression file.


## How to read this report

This is a chronological record of the work, not a retrospective summary, and later sections
correct earlier ones. Two things a reader must know before using any number:

1. **Sections 1–13 report ST-RAE on a non-official denominator.** Section 14 documents the error
   and the correction. Within-endpoint comparisons in those sections remain valid (the rescaling
   is monotone); their absolute values and macro averages do not transfer to the official scale.
   Official-scale figures appear in sections 14 onward.
2. **The submitted regression file is not the ensemble described in sections 1–13.** Section 16
   records the switch to a D-MPNN blend validated under a separate four-split protocol, and why.

Sections 5, 10, 15 and 17 are negative results, retained deliberately: four of the ideas that
looked most promising did not survive validation, and one correction (section 14) invalidated a
metric used throughout the first half of the project. Three further errors of our own are
recorded in section 10.

## 1. Data

All five released tables were used. SMILES were standardised with RDKit
(`Cleanup` -> `LargestFragmentChooser` -> `Uncharger`, canonical output); all 6,896 input strings
parsed. After standardisation there are 6,145 unique training structures and 750 test structures,
**no duplicates within either set and no structural overlap between them**.

Direct-inhibition labels are sparse: 6,525 pIC50 values over 4,905 compounds (CYP1A2 1,412 /
CYP2C9 1,285 / CYP2D6 1,493 / CYP3A4 2,335), and 3,596 of those compounds carry only a single
endpoint. The TDI table's 1,240 extra compounds add no direct-inhibition labels — where the two
tables overlap the values are bit-identical.

A decisive constraint on feature design: the single-concentration, Emax and TDI-arm tables cover
**0 of the 750 test compounds**. They can therefore only ever be auxiliary *targets*, never input
features. Any model using them as features would be unusable at submission time.

## 2. Metric

ST-RAE was reimplemented from the challenge FAQ (`cyp_metrics.py`): per-compound error is the
distance from the prediction to the nearest bound of the fitted dose-response credible interval,
zero inside it; relative normalisation by the sum of |y - mean(y)|; macro-averaged over the four
isoforms. The published definition does not fix the denominator unambiguously, but the denominator
is a constant for a fixed evaluation set, so **model ranking depends only on the summed soft error**
and the ambiguity does not affect optimisation.

Two properties of the real intervals drive everything downstream. Median interval width is 0.22-0.43
log units above pIC50 4 but **1.2-2.6 log units below it**, so inactive compounds carry a very wide
zero-error zone. Consequently a *constant* prediction already scores macro ST-RAE **0.594** — the
meaningful bar is far below 1.0, and MAE/R2 are poor proxies for the ranking metric.

## 3. Validation design

The test set is not an i.i.d. sample. Per the challenge description it is an analog expansion of the
top 25 hits from CYP1A2, CYP2C9 and CYP3A4 — 25 x 3 x 10 = exactly 750 compounds. This is directly
visible in the data: the 750 test compounds collapse onto only **134 distinct nearest training
neighbours**, many with exactly ten analogs apiece, and 70 seeds account for 631 test compounds.
Those seeds have median max-pIC50 6.35 against 4.70 for the training set overall.

Attempting to reproduce that regime internally **failed, informatively**. The training set is a
diversity library: potent training compounds have a median of *zero* analogs at Tanimoto >= 0.5, and
within-train nearest-neighbour similarity is 0.467 (0.434 for potent compounds) against 0.587 for
test-to-train. With 5,367 Murcko scaffolds across 6,145 compounds (5,016 singletons), scaffold
splitting is indistinguishable in difficulty from random splitting (realised neighbour similarity
0.439 vs 0.437). No internal split can place held-out compounds as close to the training data as the
real test compounds sit.

Selection therefore used **random 5-fold CV over the 4,905 labelled compounds**, which is
conservative with respect to the test regime, reported alongside a 522-compound close-analog subset
(labelled neighbour at Tanimoto >= 0.6) as the nearest available proxy.

## 4. Models

Three families, all on a shared 6,568-column feature set (Morgan count fingerprints r=2 and r=3,
atom-pair counts, Avalon, MACCS, 203 RDKit 2D descriptors with tails clipped at the 0.1/99.9
percentile; columns with fewer than 10 non-zero entries dropped).

1. **Interval-censored LightGBM** (per endpoint). Custom objective whose gradient is -1 below the
   lower bound, +1 above the upper bound and **exactly zero inside the credible interval**, making
   the training loss the metric numerator. Macro OOF ST-RAE 0.4436 against
   0.4511 for L2 — and notably *worse* MAE and R2 (0.617/0.457 vs 0.543/0.584 on
   CYP3A4), which is the point: it spends no capacity on precision the metric does not reward.
2. **L2 LightGBM** (per endpoint), retained for blend diversity. Macro 0.4511.
3. **Masked multi-task network** (PyTorch): shared 6,568-1,024-512 trunk, four pIC50 heads, each
   compound contributing loss only on its measured endpoints, same censored hinge plus a small
   (0.1) pull toward the point estimate for optimisation stability. 5 folds x 2 seeds, macro
   0.4711.

Hyperparameters were tuned in a later round (section 5c); the figures quoted in this section are for
the original untuned configuration. Sample weighting by inverse interval width is addressed there
too — correctly implemented, it hurts.

**Ensemble**: fixed global weights 0.50 / 0.25 / 0.25 (interval-GBM / L2-GBM / network). Weights
fitted per endpoint on out-of-fold data gained only 0.0006 macro (0.4231 vs 0.4237), so the fixed,
unbiased weights were deployed. GBM fold models and network fold/seed models are averaged for test
prediction.

**Calibration: none applied.** Linear recalibration p' = a + b*p, fitted fold-wise and evaluated on
held-out folds, changed macro ST-RAE by -0.0002. Despite a visible compression of predicted range
relative to the training label range, no shift or variance inflation survived the held-out test, and
an untestable correction was not applied.

## 5. Two negative results

**Auxiliary supervision does not help.** Adding the single-concentration log2 fold-change (4,375
compounds), TDI-arm pIC50 and Emax tables as twelve auxiliary heads: fold-0 macro ST-RAE 0.4730 at
weight 0, 0.4742 at 0.3, 0.4965 at 1.0. Dropped. Covering more compounds than the dose-response set
did not translate into transferable signal.

**Nearest-neighbour label transfer is actively harmful.** A tuned similarity-weighted kNN (power and
threshold swept) reached macro ST-RAE 0.5916 — barely better than the 0.594 constant baseline.
Stratified by neighbour similarity it is worse than the GBM in *every* stratum and degrades as
similarity rises: 0.603 at neighbour similarity below 0.4 versus **0.726 at >= 0.6**. The
close-analog pairs in this dataset are precisely the activity-cliff pairs the challenge says it built
in. Since the test set is an analog expansion, this is the regime it occupies, and local label
transfer was excluded from the ensemble. The global model is by contrast nearly flat across strata
(0.43-0.48), which is the more reassuring property for this test set.

### 5b. External data (ChEMBL) — the one intervention that worked

External data is explicitly permitted by the challenge rules. All ChEMBL activities with a
standardised pChEMBL value were pulled for the four CYP targets (CYP1A2 `CHEMBL3356`, CYP2C9
`CHEMBL3397`, CYP2D6 `CHEMBL289`, CYP3A4 `CHEMBL340`): 67,563 records (31,629 AC50, 20,423 Potency,
14,862 IC50, 569 Ki), collapsing to 25,124 unique standardised structures. The PubChem qHTS CYP panel
is already deposited in ChEMBL as AC50/Potency, so no separate PubChem pull was needed.

**Overlap with the challenge sets was checked before use**: 192 training compounds and exactly
**1 test compound** appear in ChEMBL. All 193 were removed from the external set, leaving 24,931
compounds — both to keep cross-validation honest and to avoid deriving any part of the submission
from a test-set label, permitted or not.

**The scales do not match.** On the 192 shared training compounds, ChEMBL pChEMBL versus Octant
pIC50 gives Pearson 0.654 / 0.273 / 0.706 / 0.504 (1A2 / 2C9 / 2D6 / 3A4) with ChEMBL systematically
more potent by +0.58 / +0.32 / +0.46 / +0.73 log units and differences scattering 0.55-0.82 log
units — wider than the model's own MAE. External values were therefore **not** used as additional
labels for the same endpoint. Instead a separate LightGBM was trained per endpoint on the external
data alone (held-out R2 0.480 / 0.242 / 0.332 / 0.540; Spearman 0.671 / 0.496 / 0.553 / 0.651) and
its four predictions were appended as features to the challenge models, letting the challenge model
learn the scale mapping. Because the external model never sees any challenge compound, its
predictions are out-of-sample for every fold.

This improved **every endpoint** of the interval-GBM — 1A2 -0.0016, 2C9 -0.0084, 2D6 -0.0187,
3A4 -0.0081, macro 0.4436 -> 0.4344 — with the largest gain on the weakest endpoint. All three
ensemble members were retrained on the same features (interval-GBM 0.4344, L2-GBM 0.4413, network
0.4686) and the blend improved from 0.4237 to **0.4172**.

Notably CYP1A2 gained almost nothing (-0.0002 in the blend) despite having the second-largest
external set (11,046 compounds) and a capable external model — held-out R2 0.480, second only to
CYP3A4's 0.540, and the best rank correlation of the four (Spearman 0.671 vs 0.651 for CYP3A4).
External volume and external model quality alone do not predict transfer.

Feature-pipeline alignment was verified bit-for-bit: challenge molecules pushed through the external
featurisation path reproduce the challenge feature matrix exactly. 205 non-finite descriptor cells
across 18 of 24,931 external compounds (Gasteiger-charge, BCUT2D, Ipc on unusual atom types) are left
as missing values, which LightGBM handles natively.

### 5c. Hyperparameter tuning — capacity was set far too high

The initial models used one untuned configuration (31 leaves, lr 0.05, feature_fraction 0.25). A
16-config random search followed by two focused rounds (~30 configurations total, all scored by
out-of-fold macro ST-RAE) showed capacity was the binding constraint in the wrong direction: the
score improves monotonically as trees get smaller, 127 -> 31 -> 15 -> 7 -> 5 leaves, bottoming at
**5 leaves, lr 0.02, feature_fraction 0.08, lambda_l2 1, bagging 0.6**. The worst configuration in
the search (127 leaves, lr 0.08, min_data_in_leaf 5) scored 0.4636 against 0.4203 for the best — a
spread wider than everything else attempted in this project combined.

That near-additive optimum is consistent with the sample size: with ~1,300-2,300 labels per endpoint
against 6,568 features, deep interaction structure cannot be estimated, and the models that try
overfit. Tuning alone took the single interval-GBM from 0.4344 to 0.4203 — on its own matching the
entire untuned three-model ensemble — and the L2 member from 0.4413 to 0.4347. The rebuilt ensemble
at the original fixed 0.50/0.25/0.25 weights scores **0.4132**; re-optimising the blend weights gave
0.4128, so the fixed weights were kept.

**Interval-width sample weighting, now correctly implemented inside the custom objective (LightGBM
ignores Dataset weights under a custom objective), makes things worse**: 0.4331 weighted versus
0.4256 unweighted on an identical configuration. The metric's zero-error zone already discounts
wide-interval compounds, and weighting on top of it double-counts that discount.

## 6. Results — direct inhibition

Out-of-fold, 5-fold, 6,525 labels. Lower ST-RAE is better. Final tuned ensemble including the
external ChEMBL stacked features.

| Endpoint | n | ST-RAE | Constant baseline | ST-RAE (close analogs) | MAE | R² | Spearman ρ | Kendall τ |
|---|---|---|---|---|---|---|---|---|
| CYP1A2 | 1412 | 0.495 | 0.624 | 0.533 | 0.624 | 0.291 | 0.553 | 0.388 |
| CYP2C9 | 1285 | 0.327 | 0.521 | 0.432 | 0.461 | 0.394 | 0.629 | 0.453 |
| CYP2D6 | 1493 | 0.568 | 0.663 | 0.583 | 0.569 | 0.185 | 0.448 | 0.308 |
| CYP3A4 | 2335 | 0.262 | 0.567 | 0.228 | 0.539 | 0.582 | 0.790 | 0.596 |
| MACRO | 6525 | 0.413 | 0.594 | 0.444 | 0.548 | 0.363 | 0.605 | 0.436 |

!`Out-of-fold predictions against the credible interval`

CYP3A4 is the strongest endpoint (ST-RAE 0.262, rho 0.790) and the only one that improves on the
close-analog subset (0.228). CYP2D6 remains weakest (R2 0.185, rho 0.448) with predictions spanning
4.06-5.82 against observed 1.95-7.54. Progression of the macro score across the three rounds of work:
0.4237 (internal data, untuned) -> 0.4172 (plus external ChEMBL signal) -> **0.4132** (plus
hyperparameter tuning).

## 7. Results — TDI classification

Labels were reconstructed from the four stated rules (positive / negative / inferred positive /
assigned negative) and agree with the published `is_TDI` columns on **100% of overlapping compounds**
(CYP2D6 n=1,475; CYP3A4 n=2,175), confirming the rules were implemented as intended. Training then
used the authoritative columns, which cover more compounds.

Classifiers are LightGBM on the shared features plus the four out-of-fold predicted
direct-inhibition pIC50 values (out-of-fold for training rows, ensemble output for test rows, so no
leakage).

| Isoform | n labels | OOF AUC | Deployed threshold | OOF MCC at threshold | Test positives |
|---|---|---|---|---|---|
| CYP3A4 | 3584 | 0.858 | 0.289 | 0.438 | 319 / 750 |
| CYP2D6 | 1497 | 0.619 | 0.237 | 0.163 | 161 / 750 |

External ChEMBL features were tested here too and **rejected**: CYP3A4 improved (AUC 0.859, MCC
0.455) but CYP2D6 degraded (AUC 0.608, MCC 0.118), a net -0.028 MCC across the two isoforms.
Selecting the winner per isoform would be fitting out-of-fold noise, so the classification
submission uses the models without external features for both.

Thresholding needed care. The MCC-optimal threshold on the full out-of-fold set implies **71%**
positives on the test set for CYP3A4, because the training label set is dominated by compounds
inactive in both arms while the test set is actives-enriched. Rather than transplant a threshold
across that shift, the expected test positive rate was estimated by transferring the conditional TDI
rate across bins of direct-inhibition pIC50 — the rate is 0.489 among compounds with direct
pIC50 > 4 but 0.013 in the most inactive bin — and integrating over the test set's predicted-activity
distribution. That yields 0.425 for CYP3A4 and 0.214 for CYP2D6; thresholds were
rank-matched to those rates. The deployed CYP3A4 threshold costs 0.043 OOF MCC relative to the
OOF-optimal value and should be substantially more robust under the shift.

## 8. Limitations

- Held-out estimates come from a split that cannot reproduce the test set's neighbour density; the
  direction of the resulting bias is not determined. Higher similarity should help a global model,
  while the deliberate activity cliffs and actives enrichment cut the other way.
- CYP2D6 regression and TDI are both weak enough that their contribution to the macro averages is
  close to a well-calibrated constant.
- Predicted ranges are compressed relative to training labels (CYP3A4 test predictions span
  2.41-6.22, mean 4.687, against a training label maximum of 7.19). If the analog series are more
  potent than the model projects, errors will be one-sided. No correction was applied because none
  improved held-out ST-RAE.
- Model-selection decisions (objective, blend weights, TDI thresholds) were made on out-of-fold
  predictions; blend weights were checked for selection bias but the objective choice was not nested.
- External ChEMBL labels are aggregated medians across heterogeneous assays, substrates and labs,
  and sit ~0.5 log units more potent than the Octant platform. The stacking design tolerates that
  offset but inherits the underlying measurement noise.
- One test compound exists in ChEMBL; it was excluded from external training, so no submitted
  prediction derives from a known test label.
- Roughly 30 hyperparameter configurations were scored on the same out-of-fold folds and the best
  selected, so 0.4132 is mildly optimistic as an unbiased estimate. The mitigating evidence is that
  the regularisation trend is monotone and mechanistically expected rather than a single spike, with
  four independent configurations clustering within 0.0015 of the optimum.
- Assay-level effects (the fluorescence panel for 1A2/2C9/3A4 versus the Echo-MS panel for 2D6) were
  not modelled explicitly.

## 9. Reproduction

Environment: Python 3.11, RDKit 2026.03.1, LightGBM 4.7.0, PyTorch 2.10.0, scikit-learn 1.9.1; CPU
only, roughly 45 min end to end on 10 cores. `cyp_metrics.py` holds the metric, `mtl.py`
the multi-task model. Fold assignment is seeded (`KFold(5, shuffle=True, random_state=0)`); GBM and
network seeds are fixed.

External data: ChEMBL REST `/activity.json` filtered to `pchembl_value__isnull=false` per target,
paginated at 1,000 records (~14 min for 67,563 records).

A bug worth flagging for anyone reusing this: fold assignment must cover compounds labelled in the
TDI table but *not* in the direct-inhibition table. Restricting folds to direct-inhibition-labelled
compounds silently discarded 1,238 CYP3A4 TDI labels and cost 0.11 AUC (0.746 -> 0.858).

---

## 10. Locked-protocol round: what survived independent selection

Sections 4-9 selected models using cross-validation folds that were reused across rounds. To test how
much of the reported gain was selection bias, 20% of labelled compounds (981, with 254-462 labels per
endpoint) were locked away; all subsequent hyperparameters, blend weights and thresholds were chosen
using 5-fold CV inside the remaining 3,924 development compounds only, and the holdout was scored once
per variant.

**Most of the tuning gain was selection bias.** The 5-leaf configuration beat the original 31-leaf
default by 0.0132 on development CV but by only **0.0012** on the locked holdout (0.4176 vs 0.4187),
splitting 2-2 across endpoints with sign-flipping differences of about 0.013. The honest conclusion is
no detectable advantage. A further nested search over 30 configurations (num_leaves, max_depth,
min_data_in_leaf, feature_fraction) reproduced this: a development-CV spread of 0.4271-0.4463 collapsed
to a holdout spread of ~0.001, with the development-CV-selected configuration scoring 0.4182. **Tree
hyperparameter tuning on this dataset is exhausted.**

Note that holdout figures average five fold-models while development-CV figures come from single
models, so the two columns are not comparable in absolute terms; only within-column contrasts are valid.

### Model families — all worse, and all making the same mistakes

| Model | Locked holdout macro ST-RAE |
|---|---|
| Interval-censored LightGBM (incumbent) | **0.4176** |
| Extra Trees (max_features 0.05, min_samples_leaf 5) | 0.4627 |
| Tanimoto-kernel SVR (r3/4096, C=1, eps=0.1) | 0.4866 |

The SVR grid was extended downward in C after C=1 won monotonically; C=0.3 and below are worse, so the
grid was not truncated. More informative than the ranking is the **error correlation: signed residuals
correlate 0.888-0.984 between families (mean 0.964 GBM-ExtraTrees, 0.934 GBM-SVR)**. The three model
classes make nearly identical mistakes, so blending cannot help: development-CV-optimal weights were
0.9/0.1/0.0 and the blend scored 0.4177 against 0.4176 for the GBM alone. Under the pre-agreed
criterion — blend only if errors complement — the blend was not adopted. This is the strongest evidence
in the project that the limit is the information content of the representation relative to the labels,
not the functional form of the model.

### CYP2D6 expansion — careful curation made transfer worse

850 CYP2D6 assays were pulled with metadata and filtered for protocol compatibility (recombinant
single-protein format, non-microsomal, ChEMBL confidence >= 8), cutting 12,853 activities to 2,299 over
2,053 structures. Duplicate structures agree well (median spread 0.01 log units across 179
multi-measured; 12 disagree by more than 1 log unit). Concordance against Octant could not be shown to
improve — only 13 curated compounds overlap the challenge set (Pearson 0.741) versus 76 uncurated
(0.706) — and the +0.49 log-unit bias is unchanged.

CYP2D6, locked holdout:

| Variant | ST-RAE |
|---|---|
| Uncurated external stack feature (incumbent) | **0.5723** |
| Curated external stack feature | 0.5816 |
| No external data | 0.5854 |
| Plus 600 curated rows as labels (offset -0.493) | 0.5913 |
| Plus 2,036 curated rows as labels | 0.6024 |

Curation raises the external model's own held-out R2 from 0.332 to 0.537 while making transfer *worse* —
the second time in this project that external model quality failed to predict transfer (CYP1A2 was the
first). Public data does not relieve the 1,493-compound CYP2D6 bottleneck, and using it as direct labels
is actively harmful even with a scale offset and subsampling.

## 11. Uncertainty quantification

Split-conformal intervals were calibrated on development out-of-fold absolute residuals and their
coverage verified on the locked holdout (`conformal_coverage.csv`):

| Nominal | Empirical (mean over 4 isoforms) |
|---|---|
| 80% | 80.4% |
| 90% | 89.9% |

Per-endpoint 90% half-widths are 1.33-1.59 log units (CYP2C9 narrowest at 0.97 but undercovering at
85.0%, the one endpoint where the stated level should not be trusted).

**Two standard difficulty signals were tested and both failed.** Nearest-neighbour Tanimoto similarity
does not predict absolute error (Spearman -0.155 to +0.114 across endpoints, *positive* for CYP1A2), and
neither does disagreement among the five fold-models (-0.141 to +0.027, with mean error flat and
non-monotone across disagreement tertiles). A Mondrian conformal variant conditioned on similarity was
built and rejected: its widths are non-monotone in similarity. Global per-endpoint quantiles are used
instead. The only variable found to track error is the experimental credible-interval width (Spearman
+0.107 to +0.464), which is a property of the label and therefore unavailable at prediction time.

This has a direct consequence for how the reliability report must be read: **a test compound with few
close measured neighbours is not thereby less reliably predicted.** The report deliberately contains no
similarity-derived "evidence tier", because such a tier would encode a relationship this dataset does not
support. The neighbour columns are factual context for a chemist, not a confidence score.

### Per-compound reliability report

`reliability_report.csv` — 3,000 rows (750 test compounds x 4 isoforms), 23 columns: predicted
pIC50; calibrated 80% and 90% bounds; the three nearest experimentally measured training compounds with
their Tanimoto similarity and measured pIC50; counts of measured neighbours at Tanimoto >= 0.5 and
>= 0.6; fold-model spread (reported as a diagnostic, explicitly *not* predictive of error); and, for the
two TDI isoforms, the isotonic-calibrated TDI probability alongside the submitted boolean call.

Median nearest measured-neighbour similarity per endpoint is 0.47-0.54, and 268-394 of the 750 compounds
have no measured neighbour at Tanimoto >= 0.5 for a given isoform (CYP2D6 worst at 394) — a consequence
of the sparse training matrix, since a compound counts as a neighbour only if it was measured against
that isoform.

### TDI probability calibration

Isotonic regression fitted inside folds (`tdi_calibration.csv`) improves probability quality
without improving discrimination, as expected: CYP3A4 expected calibration error falls 0.0379 -> 0.0110
and Brier 0.1231 -> 0.1203, CYP2D6 ECE 0.0206 -> 0.0185, while best-threshold MCC moves -0.0034 and
-0.0065 respectively. Platt scaling is worse than no calibration for CYP3A4 (ECE 0.0738). Calibrated
probabilities are reported for interpretation; the submitted boolean calls are unchanged, since
calibration gave no MCC benefit.

## 12. Submission status after this round

**No submission file was changed.** Nothing tested in this round improved the locked-holdout score: tree
retuning, Extra Trees, Tanimoto-kernel SVR, three-way blending, curated CYP2D6 data as features or as
labels, and TDI calibration all failed to beat the incumbent. The regression submission continues to use
the 5-leaf configuration, which the locked holdout shows is neither better nor worse than the original
default (0.4176 vs 0.4187, a difference within noise). Changing it would be churn without evidence.

---

## 13. Learned representation: the one change that worked

Every model family in section 10 consumed the same fixed fingerprints, and their errors correlated
0.89-0.99 — the signature of a representation ceiling rather than a model-class ceiling. The remaining
untested hypothesis was that a model which *learns* its representation from the molecular graph would
make different mistakes. It does.

A Chemprop 2.3.1 D-MPNN (BondMessagePassing d_h=300 depth=3, mean aggregation, 4-task regression FFN
1024x3, dropout 0.1, 50 epochs, 5 folds on the development set) is individually the **worst** model
tested — locked-holdout macro ST-RAE 0.4704, and on CYP2D6 it reaches 0.7219, worse than that
endpoint's constant baseline. But its signed residuals are markedly decorrelated from the
interval-censored LightGBM (`model_family_error_correlation.csv`):

| Family | Mean error correlation with interval-LightGBM | Locked holdout |
|---|---|---|
| CatBoost (depth 3, rsm 0.1) | 0.970 | 0.4266 |
| Extra Trees | 0.964 | 0.4627 |
| Tanimoto-kernel SVR | 0.934 | 0.4866 |
| **D-MPNN (learned graph representation)** | **0.771** | 0.4704 |

Blending on that decorrelation, with weights chosen on development folds and the holdout scored once:

| Ensemble | Locked holdout |
|---|---|
| Interval-LightGBM alone | 0.4176 |
| Previous submission weights (0.50/0.25/0.25/0) | 0.4113 |
| Interval-LightGBM + D-MPNN, 0.75/0.25 | 0.3984 |
| **Selected: 0.6 interval-GBM / 0.1 L2-GBM / 0.1 MLP / 0.2 D-MPNN** | **0.3980** |

This is **-0.0133 against the previous submission, the largest validated gain in the project** — larger
than external ChEMBL data (-0.0066) and than hyperparameter tuning (which was zero once nested).
Crucially it was measured on the locked holdout, so unlike the earlier tuning result it is not an
artefact of reused splits. The two-member blend is statistically indistinguishable from the four-member
one (0.3984 vs 0.3980), so the D-MPNN supplies essentially all of the gain; the L2-GBM and MLP members
are retained only because development CV selected small non-zero weights for them.

Gains concentrate where the D-MPNN is competent: CYP1A2 -0.018, CYP2C9 -0.019, CYP3A4 -0.015, CYP2D6
flat at -0.001 (it is too weak there to contribute).

### Final locked-holdout performance

| Endpoint | n | ST-RAE | Constant baseline | MAE | R² | Spearman ρ | Kendall τ |
|---|---|---|---|---|---|---|---|
| CYP1A2 | 304 | 0.469 | 0.614 | 0.595 | 0.322 | 0.535 | 0.376 |
| CYP2C9 | 254 | 0.306 | 0.531 | 0.473 | 0.399 | 0.641 | 0.469 |
| CYP2D6 | 297 | 0.573 | 0.663 | 0.557 | 0.180 | 0.440 | 0.301 |
| CYP3A4 | 462 | 0.244 | 0.559 | 0.525 | 0.596 | 0.814 | 0.619 |
| MACRO | 1317 | 0.398 | 0.592 | 0.538 | 0.374 | 0.608 | 0.441 |

Submission regenerated with the four-member ensemble (`submission_regression.parquet`,
`validation_metrics.csv`), validated and read-back verified. The TDI classification submission is
unchanged.

### What this implies for CheMeleon

Since a D-MPNN trained from scratch on 4,905 compounds already pays, a *pretrained* molecular encoder
fine-tuned on this data is now well-motivated rather than speculative. OpenADMET publish a CheMeleon
multi-task CYP model (`openadmet/cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1`, `from_chemeleon: true`,
4 tasks, FFN 1024x3) trained on ChEMBL-curated pIC50 with no validation split — i.e. on public data, not
on the challenge's Octant measurements, so it carries no leakage into the Octant holdout. It would need
OpenADMET's Anvil framework to load. Note the expected overlap: its training signal is ChEMBL pIC50,
which is already distilled into this pipeline's four stacked features, so part of its information is
redundant; the non-redundant part is the pretrained representation itself.

---

## 14. CORRECTION: the ST-RAE denominator was wrong in sections 1-13

Sections 1-13 used a non-standard denominator. The challenge's ST-RAE normalises the summed soft
error by **the same soft-error function applied to a constant mean predictor**, so predicting the mean
scores exactly 1.0 by construction. This project instead used the textbook RAE denominator,
sum|y - mean(y)|, under which a constant predictor scored 0.594 rather than 1.0. Every ST-RAE figure
in sections 1-13 is therefore on a project-internal scale and is **not comparable to the challenge
leaderboard**. `cyp_metrics.py` now implements the official definition (verified: a constant-mean
predictor returns exactly 1.0 on all four endpoints); the old variant is retained as
`st_rae_meandev` only for reading earlier logs.

Final locked-holdout performance on the **official** scale:

| Endpoint | ST-RAE (official) | Constant baseline | MAE | R² | Spearman ρ |
|---|---|---|---|---|---|
| CYP3A4 | 0.427 | 1.0 | 0.525 | 0.596 | 0.814 |
| CYP2C9 | 0.564 | 1.0 | 0.473 | 0.399 | 0.641 |
| CYP1A2 | 0.729 | 1.0 | 0.595 | 0.322 | 0.535 |
| CYP2D6 | 0.867 | 1.0 | 0.557 | 0.180 | 0.440 |
| **MACRO** | **0.647** | 1.0 | 0.538 | 0.374 | 0.608 |

**No modelling decision changes.** Within an endpoint the denominator is a constant for a fixed
evaluation set, so per-endpoint rankings are untouched; only the macro weighting across endpoints
shifts. Re-scored officially, the submitted ensemble still wins by a wider margin than previously
reported: 0.6468 against 0.6696 for the previous submission (-0.0228, where the internal scale showed
-0.0133) and 0.6804 for the interval-GBM alone. The submitted file is unaffected.

For a fixed evaluation set the official score is exactly the internal score divided by that
endpoint's internal constant-baseline value, so the per-endpoint inflation factors are 1.56 (CYP1A2),
1.84 (CYP2C9), 1.51 (CYP2D6) and 1.75 (CYP3A4). The rescaling is therefore *smallest* for CYP2D6 and
CYP1A2, and their combined share of the macro sum slightly **falls**, from 65.4% internally to 61.7%
officially. CYP2D6 (0.867) and CYP1A2 (0.729) remain the dominant contributors to the macro average
because they are genuinely the weakest endpoints (R² 0.180 and 0.322), not because of any denominator
effect. The ranking of where effort should go is unchanged by the correction.

Earlier artefacts that quote the internal scale — `leave_group_out.csv`,
`group_novelty_by_model.csv`, `learning_curve.csv`, `train_vs_heldout.csv` — retain their original
values. Comparisons *within* an endpoint in those files remain valid (monotone rescaling); their
macro averages do not transfer to the official scale. The conformal intervals in
`reliability_report.csv` are in pIC50 units and are unaffected.

## 15. Potency-tail compression is not correctable at prediction time

Sections 11–12 recorded that strong inhibitors are under-predicted by more than a log unit
(CYP2D6 −1.40, i.e. 24×), and that the blinded test set is deliberately enriched in actives
(25 hits × 3 isoforms × 10 analogues). That combination looks like an exploitable bias. It is not,
and the reason is worth recording because it generalises.

**The scoring function is symmetric.** `rae_soft_threshold_absolute_error` computes
`clip(pred − hi, 0, ∞) + clip(lo − pred, 0, ∞)` — the distance to the nearer edge of the credible
band, costing the same in either direction. There is no asymmetry that rewards shifting
predictions upward; only a reduction in genuine bias can help.

**The bias is conditional on the truth, not on the prediction.** A correction applied at
prediction time can condition only on the prediction. Measured on the 3,924-compound development
partition's out-of-fold predictions (1,108 / 1,031 / 1,196 / 1,873 labels for
CYP1A2 / 2C9 / 2D6 / 3A4; median residual `y − ŷ`, sextile bins):

| Endpoint | Conditioned on true y (low → high) | Conditioned on prediction (low → high) |
|---|---|---|
| CYP1A2 | −1.44 → +0.82 | −0.26 → −0.04 |
| CYP2C9 | −0.78 → +0.54 | −0.25 → +0.07 |
| CYP2D6 | −0.80 → +0.94 | +0.11 → +0.10 |
| CYP3A4 | −1.09 → +0.49 | −0.67 → +0.02 |

Conditioned on the prediction, the top bin is unbiased to within 0.10 log units on all four
endpoints. This is regression dilution: "strong compounds are under-predicted" and "high
predictions are unbiased" are simultaneously true, and only the second is actionable. Shifting
high predictions upward moves a near-zero median away from zero.

**Where bias does exist, the metric already forgives it.** The only material
prediction-conditional bias is at the *low* end — CYP3A4's bottom sextile sits 0.67 log units
above truth. But low-potency compounds carry wide credible bands: that bin has a median band
width of 2.18 log units, 74.4% of its predictions already fall inside the band, and it
contributes 7.0% of the endpoint's total soft error. By contrast each of the top three bins
carries 19–22% of the error with band widths of 0.15–0.31. The error budget is concentrated
exactly where the model is unbiased.

**Nested test.** Three pre-specified correction forms — identity, isotonic regression of y on ŷ,
and a two-parameter low-end hinge `ŷ − δ·max(0, τ − ŷ)` with (τ, δ) chosen by minimising soft
error on the fitting partition — were fitted per endpoint inside the five development folds and
evaluated out-of-fold. Macro nested-dev ST-RAE: identity **0.6669**, low-hinge 0.6674,
isotonic 0.7078. Isotonic is markedly worse on every endpoint; the hinge is neutral, selecting
δ = 0 outright on CYP2D6 and CYP3A4. Nothing beat identity, so **the locked holdout was not
scored for this experiment** — the one-shot budget is not spent on a candidate development has
already rejected, and the submitted macro 0.6468 stands unchanged.

Evidence: `potency_correction_test.csv` (bias by bin under both conditionings, per-bin error
budget with band widths, nested-CV scores for all three forms).

This closes the question raised in §12. The compression is real, it is visible in two independent
pipelines, and it is a property of what the features can resolve — not a calibration defect. The
remedy would have to be a model that separates potent analogues from their weak neighbours, not a
transformation of the predictions it already makes.

## 16. Submitted file: the D-MPNN candidate, not the ensemble of §1–13

The regression track is submitted from the second agent's D-MPNN pipeline, run to completion in
this workspace from their portable bundle (plan/split hash
`ff38465ea25a6b5467be2a45802a7ca490ac988ca2e6538a23b914d05078c37d`). The four-member ensemble
described in §1–13 is retained as the project's own result but is not the submitted file.

Grounds for the switch:

- Their predeclared gate passed on all five conditions: mean macro ST-RAE gain **+0.0488**
  (threshold 0.02), **4 of 4** split wins, no endpoint deteriorating, CYP2D6 strong-inhibitor MAE
  improving by 0.0101. A paired interval across the four splits, declared before fitting, gives
  **95% CI [+0.0352, +0.0623]** — the gate passes on effect size, not on a threshold artefact.
- Validation is across four scaffold-disjoint outer splits with three group-exclusive inner folds
  under a frozen plan with sealed jobs, versus the single locked holdout used here.
- Preflight in this environment was exact: 25/25 contract tests, and baseline reproduction against
  their retained v3 at max absolute difference **3.55e-15** over all 3,000 predictions
  (Python 3.11 here against their 3.12). Every candidate prediction was replay-verified from
  saved models.
- Both files pass the official `activity_validation.py`. The candidate predicts 80 compounds at
  pIC50 >= 6 against this project's 25, which is better matched to a test set built as
  10-analogue expansions around 25 confirmed hits per isoform — though §15 establishes that high
  predictions are already unbiased conditional on the prediction, so wider tail coverage is not
  automatically more accurate.

Recorded against it: the candidate's CYP2D6 column is effectively their Extra Trees baseline
(final graph weight 0.05), and Extra Trees was the weakest member on CYP2D6 in this project's own
comparison (dev OOF 0.892 against 0.872 for the interval LightGBM), receiving zero weight when
seven members were given freely fitted nonnegative weights. The two scores are not directly
comparable — 0.6632 is a four-split scaffold-disjoint mean on their features, 0.6468 a single
locked holdout on these.

A prediction recorded in §14's review of their package — that the gate was "set at the size of the
effect" and likely to yield a misleading negative — was **wrong**. It transferred this project's
effect size (+0.0198 dev, +0.0230 holdout, measured against a three-member blend) onto their
single-model-per-endpoint baseline without examining it. The review had named the responsible
mechanism, that a weaker baseline inflates the apparent graph gain, and failed to apply it to its
own conclusion.

The TDI classification track is unchanged from §9.

## 17. Censored bounds from the single-concentration screen: rejected

Another entrant's public report states that a model trained on dose-response labels alone has
never seen a CYP2D6 non-inhibitor and over-predicts every unselected compound, and that treating
the single-readout rows as interval labels rather than point labels moved their leaderboard
CYP2D6 from 1.46 to 0.746. This project had rejected the single-concentration table earlier, but
only as *auxiliary multi-task heads* (0.4730 -> 0.4965); censored bounds on the primary target
were untested. They are now tested, and they fail here.

**The data.** `cyp-challenge-single-concentration-TRAIN.csv` is 17,504 rows: 4,376 compounds x
4 enzymes, one readout each, all at a single concentration of 4.95e-5 M. The censoring point is
therefore pIC50 = 4.305. 11,509 compound-enzyme pairs carry a screen readout but no dose-response
label, against 5,995 that have both — nearly twice the supervision.

**Rule calibration (pre-specified before fitting).** Among pairs with both, the violation rate of
"no inhibition implies pIC50 <= 4.305":

| log2fc cut | labelled rows | violating |
|---|---|---|
| >= −0.10 | 212 | 0.00% |
| >= −0.25 | 636 | 0.47% |
| >= −0.40 | 997 | 2.71% |
| >= −0.50 | 1,183 | 5.41% |

Variant A right-censors at log2fc >= −0.25 (0.47% violation); variant B adds left-censoring at
log2fc <= −2.0 (1.92% violation). Censored rows enter the interval objective with the loss zero
anywhere below 4.305, are never scored, and were excluded whenever their compound sat in the
locked holdout or in the fold being scored.

**Result.** Macro blend ST-RAE, development out-of-fold: 0.6669 -> **0.6899** (A, +0.0230) and
0.6909 (B, +0.0240). Both are decisively worse, so no variant was promoted; the confirmatory
holdout figure (0.6468 -> 0.6629) agrees. At member level the augmented interval-LightGBM
degrades from 0.786 to 0.928 on CYP1A2 and 0.872 to 0.995 on CYP2D6, and is near-neutral on
CYP3A4 — which received only 128 censored rows, because CYP3A4 has the most dose-response labels
and its unlabelled compounds mostly still inhibit.

**Mechanism.** The augmentation shifts predictions down by 0.108 / 0.094 / 0.124 / 0.016 log
units on CYP1A2 / 2C9 / 2D6 / 3A4. The censored rows are not a random sample — they are the
inactive tail that the primary screen declined to promote to dose-response. Adding them removes a
selection bias that **the evaluation set shares**: the held-out labels, and the blinded test set
built as 10-analogue expansions around confirmed hits, are drawn from the same actives-enriched
selection. Correcting the bias therefore moves predictions away from the distribution being
scored.

This does not refute the other entrant's result. It identifies a precondition. Their fix targets a
model that over-predicts unselected compounds; section 15 established that this project's
predictions are unbiased *conditional on the prediction*, with top-bin median residuals within
0.10 log units on all four endpoints. A global downward correction helps a model with that
pathology and hurts one without it.

Evidence: `censoring_rule_calibration.csv`, `censored_bounds_test.csv`.
