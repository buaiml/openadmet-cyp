---
tags:
- chemistry
- tabular-regression
- reproducibility
language:
- en
---

# Current regression submission: reproducibility report

Submission snapshot: **2 November 2026**; documentation revised **3 November 2026**. This self-contained version covers the latest accepted regression file recorded in this checkout.

| Submission | Accepted | Recorded public macro score |
|---|---|---|
| [Geometry ensemble](submissions/regression_best_ensemble_heme_geometry_concat_20261029.csv) | 29 October 2026 | ST-RAE **0.423257** |

The outputs are direct-inhibition pIC50 for **CYP1A2, CYP2C9, CYP2D6 and CYP3A4** on 750 molecules. `pIC50 = −log10(IC50 in mol/L)`, so higher predictions mean stronger inhibition. The score is a saved public-subset result, not a new retrieval or final full-test estimate. Here A2/C9/D6/A4 abbreviate the four CYPs, and OOF means out-of-fold.

## 1. Data and provenance

The [public challenge dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test) is pinned to revision `146282d2ce1532f1803679a834a9bec9dac1e482`. The [data manifest](assets/cache/workbench_reproduction/data_manifest.json) records the equivalent public S3 exports and external CYP files, including URLs, byte counts and SHA-256 hashes. Counts below refer to processed/aligned inputs.

| Data | Role in regression | Local artifact |
|---|---|---|
| 4,905 challenge compounds; 6,525 measured direct pIC50 values | Endpoint targets: A2 1,412; C9 1,285; D6 1,493; A4 2,335. Credible intervals define scoring. | `.cache/workbench_reproduction/raw/inhibition.csv`; `processed/challenge_aux.parquet` |
| Single-concentration screen, 4,375 measurements per CYP | Four log2FC auxiliary heads and potent-hit validation anchors | `.cache/workbench_reproduction/raw/single_concentration.csv` |
| Challenge TDI-condition pIC50, direct Emax and TDI Emax | Auxiliary continuous heads for shared encoders and D6 specialist views; they do not change the four submitted targets | `.cache/workbench_reproduction/raw/{tdi,emax}.csv` |
| ChEMBL 37 and Veith/PubChem AID 1851, five CYPs | Separate source/assay auxiliary heads; heterogeneous endpoints are not pooled into direct challenge pIC50 | `processed/union18.parquet` (31,670 rows), `processed/union_specialist.parquet` (32,877 rows) |
| Public Monroe, CheMeleon, ADME Chemprop and ESMFold2-Fast weights | Frozen representations, encoder initialization or complex features, as specified below | Component checkpoints and recorded hashes |
| Cofold representations and saved CYP–ligand poses | D6 512-channel pair features; heme-relative geometry for all four endpoints | `runs/internal_eval/d6-esmfold2-20261016/features.npz`; `cyp_activity_atlas/data/{geometry.json,geometry_atoms.npz,geometry_sources.json}` |
| 750 blinded molecules, structures/identifiers only at submission time | Prediction inputs, external-data exclusion, similarity matching and cohort-level placement | `.cache/workbench_reproduction/processed/blinded.parquet` |

Processed paths in the table are relative to `.cache/workbench_reproduction/`. Parent/InChIKey-connectivity grouping collapses salt/stereo aliases for exclusion without necessarily removing stereochemistry from model inputs. AUX30 excludes one blind-matching auxiliary row, retaining **32,876 rows**. Geometry uses **6,525 training + 3,000 blind poses**, including **422 accepted repairs** in the submitted snapshot. The subsequent Vina geometry experiment is outside this submission.

No hidden per-molecule outcomes were used in submitted fitting. **Published leaderboard-derived population information does enter placement.** Foundation-model pretraining histories have not been exhaustively audited for challenge-structure overlap. Separately trained Novartis-surrogate, Tox21 and substrate experimental arms were excluded; the public ADME initialization listed below remains included. Later test-label diagnostics in report 82 and report 83 did not replace this accepted file. Data details: [Workbench assembly](reports/06_workbench_reproduction.md) and [AUX30](reports/20_external_auxiliary_chemeleon_encoders.md).

## 2. Internal validation and metrics

The design follows the challenge's hit-expansion mechanism: the blind compounds were purchased analogues of potent CYP1A2/2C9/3A4 screening hits, with CYP2D6 measured on that same compound panel. Consequently, validation retains potent anchors in training and holds out their labelled analogues. Whole-series holdout supplies a separate, harder stress test.

The main regression development registry selects **150 labelled analogues per endpoint**, using the strongest quarter of primary-screen compounds as eligible anchors because the literal top-25 hit set provides too few labelled neighbours. Morgan radius-2 similarity to training is matched approximately to the blinded structures; anchor families stay together across **five outer folds**. The split hash is `303c137aeaea4fce65c43ee842b867b494ea61607248e91f1c3eca1f82bfa747`. A later parent audit removed one D6 query with a training counterpart, leaving the **599 observations** used for recent ensemble comparisons. The earlier Workbench reproduction's 529-molecule holdout is a different panel and its scores are not interchangeable.

Similarity uses Morgan radius-2/ECFP4 fingerprints, following [Rogers and Hahn (2010)](https://doi.org/10.1021/ci100050t), with [RDKit’s implementation](https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints). Whole-series stress tests use [Bemis–Murcko frameworks](https://doi.org/10.1021/jm9602928). The analogue selection is motivated by the [organizers’ selection design](https://openadmet.github.io/octant-cyp-inhib-blog-post/); its exact matching procedure is local. Held-out parents are removed from challenge-derived auxiliaries and external training sources. Multitask models exclude the union of endpoint holdouts, with learned preprocessing confined to each assigned fit pool.

**Limits:** development labels were reused repeatedly. Historical mixer weights used OOF predictions from other outer folds whose base learners could have trained on the evaluated fold, and the old optimizer used an ordinary RAE denominator. Weight selection was therefore not fully nested through the base learners. Later comparisons correct the scorer and parent overlap but inherit this mixer. Geometry subsets were chosen retrospectively on the same 599 observations. Bootstrap intervals are conditional on predictions and omit model-refit/selection uncertainty; cohort placement is not validated by native internal scores. These are development comparisons, not unbiased estimates of final performance. See [validation design](reports/03_internal_validation.md) and [evaluation corrections](reports/48_evaluation_redo.md).

### Regression metric

For observation `i`, let `[L_i, U_i]` be the supplied credible interval and `p_i` the prediction. The interval distance is `d_i(p_i) = max(L_i − p_i, 0) + max(p_i − U_i, 0)`. For each endpoint:

`ST-RAE = Σ_i w_i d_i(p_i) / Σ_i w_i d_i(y_bar)`, with `y_bar = Σ_i w_i y_i / Σ_i w_i`.

Ordinary evaluation uses `w_i = 1`; resampling can change these weights. Predictions within an interval incur zero error. Crucially, the mean baseline is also interval-thresholded, so an ordinary RAE denominator changes the metric. Lower is better; macro ST-RAE gives each of the four endpoints equal weight. The local implementation is [`progress_evaluation.st_rae`](workbench_reproduction/progress_evaluation.py), following the [official challenge convention](https://github.com/OpenADMET/CYP-Challenge-Tutorial).

## 3. Final models, training and calibration

The core has **six prediction arms**, followed by three fixed additions. Each tabular head uses only measured challenge labels for its endpoint. Kumo-L is v1.0.0 with eight estimators and native q500 median predictions; TabPFN-3.5 uses three estimators and training-only scaling.

| Arm | Training and inference recipe |
|---|---|
| **Workbench direct** | Four task views: `union18` (4 direct + 4 log2FC + 5 ChEMBL + 5 Veith heads), `aux8` (4 direct + 4 log2FC), D6-only six-head specialist, and D6 single-task. Each is a five-scaffold-fold Chemprop D-MPNN ensemble. Hidden size 700, depth 5, dropout 0.1, two-layer 300-wide FFN; MAE, batch 64, up to 400 epochs, patience 40. Auxiliary task weight 0.3 relative to primary. Average the first two views for A2/C9/A4, all four for D6. |
| **Workbench → Kumo** | 1,200 aligned hidden coordinates (four 300D view means) + 33 semantic predictions, without PCA. Training semantic outputs are scaffold-OOF; blind outputs average five GNN folds. Hidden means use the matched extraction policy for training and inference. Fit one Kumo head per endpoint. |
| **Workbench → TabPFN** | Same 1,233D representation and endpoint label pools; separate TabPFN heads. |
| **Monroe + geometry → Kumo** | Frozen 720D Monroe foundation embedding, replacing the original Monroe-only arm. Append A2: centroid distance + XYZ (**724D total**); C9/A4: all 19 geometry descriptors (**739D**); D6: 12 coordinate-histogram fractions (**732D**). Features use a named-heme coordinate frame; histogram quartiles come from endpoint training heavy atoms. Four full-label heads use the same original embedding bank for training and blind rows. |
| **Challenge-multitask CheMeleon → Kumo** | Fine-tune the pretrained encoder jointly on four direct-inhibition targets, masking missing labels. Final encoder has a 10-epoch cap, patience 10, and 10% grouped development split. Freeze its full 2,048D fingerprint; fit endpoint Kumo heads. |
| **AUX30 CheMeleon → Kumo** | Fine-tune with 30 continuous heads: 4 each of direct pIC50, log2FC, TDI pIC50, direct Emax and TDI Emax, plus 5 ChEMBL and 5 Veith heads. MAE, batch 64, 30-epoch cap, patience 10; 29,588 optimization / 3,288 connectivity-grouped development rows. Freeze 2,048D fingerprints; fit endpoint Kumo heads. |
| **D6 cofold addition** | One all-1,493-label Kumo regressor on ECFP4-2048 + 32 physicochemical controls + 512 frozen ESMFold2-Fast complex channels. Channels summarize protein–ligand pair states with mean/SD, three folding loops and a fixed seed. |
| **A2/A4 specialists** | Same cached 1,233D Workbench representation, three fixed-seed heads averaged: Kumo for A2; TabPFN for A4. No encoder retraining. |
| **ADME Chemprop addition** | Start from public `surrogate_adme__s42.pt`, revision `0f8201d06c4656c3f97bc244ce2dc18ac2881be9`. Fine-tune all encoder weights with 12 heads: four each of direct pIC50, log2FC and TDI-condition pIC50. Masked standardized MAE; encoder/head learning rates 1e-5/1e-3, batch 128, weight decay 1e-4. Average five full-data refits on 6,145 rows (6,141 parents), using previously selected epoch counts **3, 1, 2, 2, 3**. |

### Regression method provenance and feature construction

The four-view starting point follows the [published Workbench entry](https://supercowpowers.github.io/workbench/blogs/cyp_challenge/) at the [pinned upstream revision](https://github.com/SuperCowPowers/workbench/tree/20e09bd049f2724b2e9fb2a8b4085a941ee21b10/ml_pipelines/OpenADMET/cyp). Its graph models use the D-MPNN approach of [Yang et al. (2019)](https://doi.org/10.1021/acs.jcim.9b00237). The hybrid stages first learn molecular features, then condition a tabular foundation model on endpoint-labelled rows; they do not retrain the entire foundation model for each CYP head. [Kumo’s release](https://huggingface.co/blog/nvidia/kumo-tabular) and the [TabPFN-3.5 model card](https://huggingface.co/Prior-Labs/tabpfn_3_5) describe those tabular predictors.

[CheMeleon](https://arxiv.org/abs/2506.15792) supplies descriptor-pretrained D-MPNN weights, adapted here following the [official fine-tuning workflow](https://chemprop.readthedocs.io/en/main/chemeleon_foundation_finetuning.html). [Monroe](https://arxiv.org/abs/2608.18982) supplies a pretrained molecular graph transformer. The ADME arm uses the [OpenADMET weight release](https://huggingface.co/openadmet/chemprop-foundation-pretraining-weights/tree/0f8201d06c4656c3f97bc244ce2dc18ac2881be9). The task lists, stopping rules, endpoint weights and fixed additions in this report are the project’s extensions of those releases.

Geometry is computed from saved cofold poses: translate the ligand heavy atoms to the heme Fe origin, rotate them using the named heme nitrogen frame, and compute centroid distance (1), centroid coordinates (3), coordinate SDs (3), and four-bin coordinate histograms (12), giving 19 descriptors. The A2, C9, D6 and A4 subsets are selected separately. Histogram edges use training atoms only. The D6 complex arm instead pools learned pair states from the [ESMFold2-Fast release](https://huggingface.co/biohub/ESMFold2-Fast); its 512 channels are not these geometric descriptors. Pair-state pooling and the heme-relative feature construction are local procedures, documented in [report 51](reports/51_cyp2d6_esmfold2.md) and [report 77](reports/77_heme_geometry_kumo.md).

The six-arm weights below precede the fixed additions. They sum to one, with a 0.02 minimum; historical OOF selection chose endpoint-specific weights and L2 shrinkage 0.03. Exact precision is retained in [selected_weights.json](runs/submission/workbench-kumo-tabpfn-monroe-chemeleon-aux30-optimized-20261006/selected_weights.json).

| CYP | WB direct | WB Kumo | WB TabPFN | Monroe + geometry | CheMeleon MT | AUX30 |
|---|---:|---:|---:|---:|---:|---:|
| 1A2 | .132679 | .318102 | .271680 | .020000 | .020000 | .237539 |
| 2C9 | .199906 | .172127 | .331985 | .153814 | .020000 | .122168 |
| 2D6 | .292746 | .020000 | .020000 | .297039 | .020000 | .350215 |
| 3A4 | .020000 | .313179 | .486637 | .020000 | .140184 | .020000 |

Let `B` be this six-arm raw pIC50 blend, `C` the D6 cofold head, `S` the endpoint specialist mean and `A` the ADME mean. The final raw predictions are:

- **1A2/3A4:** `0.90 × (0.80 × B + 0.20 × S) + 0.10 × A`.
- **2C9:** `0.90 × B + 0.10 × A`.
- **2D6:** `0.90 × (0.90 × B + 0.10 × C) + 0.10 × A`.

All mixing precedes population placement. For each endpoint, transform its 750 raw predictions `r` using `max(1, μ + σ × (r − mean(r)) / sample_SD(r))`:

| CYP | Target μ | Target σ |
|---|---:|---:|
| 1A2 | 4.412 | 1.169409 |
| 2C9 | 4.830 | .922638 |
| 2D6 | 3.570 | .900000 |
| 3A4 | 4.880 | 1.082472 |

The sample SD uses denominator `750 − 1`. The floor is applied after the affine transform, so the final moments can differ slightly from the table. Replacing or adding molecules to the cohort changes its mean/SD and can therefore change individual placed predictions.

These inherited moments contain **published leaderboard-derived information**. This cohort-level affine placement preserves ordering except for floor ties; it is not training-only calibration and is not validated by the native internal scores.

### Regression evidence for the submitted replacement

On the 599 reused internal observations, replacing the Monroe arm with the selected geometry concatenations changes the complete ensemble’s native macro ST-RAE from **0.514789 to 0.512784**. The reduction is 0.002004, with a 5,000-draw paired scaffold-bootstrap 95% interval **[−0.000159, +0.004199]**. This interval conditions on fitted models and selected subsets; it includes zero and excludes refitting/selection uncertainty. [Saved comparison](runs/submission/regression-heme-geometry-concat-20261029/internal_comparison.json).

The saved public leaderboard comparison changes macro ST-RAE from **0.423968 to 0.423257**. Submitted endpoint scores are **0.418144 / 0.354800 / 0.536596 / 0.383488** for A2/C9/D6/A4. These public-subset scores include population placement and cannot be directly compared with the native internal scores. [Saved public comparison](runs/submission/regression-heme-geometry-concat-20261029/leaderboard_scored_comparison.json).

<!-- REPORT81_INTERNAL_PLOTS_START -->
## 4. Internal evaluation: what each component contributes

The following figures use saved predictions on **the same 599 audited endpoint observations**: 150 each for CYP1A2/2C9/3A4 and 149 for CYP2D6, across five outer folds. These are endpoint observations, not 599 distinct compounds. All scores are recomputed with the interval-thresholded numerator **and denominator**. Except for the explicitly labelled placement diagnostic, predictions stay in native pIC50 units. None of the later HF test-label analyses enters these plots.

Read the figures as a retrospective explanation of the submitted recipe. They retain the historical mixer’s indirect dependence on development labels and the retrospective geometry selection. Error bars are **pointwise 95% paired scaffold-bootstrap intervals**, using 5,000 shared family draws; they condition on saved fits and are not adjusted for the multiple displayed comparisons. An interval excluding zero is therefore not independent confirmation of a component’s value.

Each scatter plot uses the same axis limits and error-color scale. The horizontal segment is the measured compound’s credible interval; a prediction anywhere on that segment incurs zero interval error. The dashed line shows equality with the point estimate. Color indicates distance outside the interval, capped at 2 pIC50 units for display. **Lower ST-RAE is better; higher Spearman ρ is better.**

[Figure assets and per-figure summary CSVs](reports/figures/81_regression/) include both PNG and editable SVG versions. The [source hashes and checks](reports/figures/81_regression/provenance.json), [complete endpoint metrics](reports/figures/81_regression/all_endpoint_metrics.csv), and [aligned prediction table](reports/figures/81_regression/audited_plot_predictions.parquet) make the plots auditable. Regenerate them without fitting models:

```bash
/rwthfs/rz/cluster/home/py110287/micromamba/envs/lle/bin/python \
  -m workbench_reproduction.report81_regression_figures
```

### 4.1. Different training datasets and assay views

![Measured training labels by CYP and source](reports/figures/81_regression/01_data_sources.png)

**What to learn:** ChEMBL and Veith supply many more labels than the challenge’s direct curves, but their assay definitions differ. They supervise separate auxiliary tasks, rather than becoming additional challenge pIC50 labels. Counts are taken from the processed data manifest, include cross-source compound overlap, and must not be added to estimate unique molecules. This first plot describes training support; the next plot measures internal prediction performance.

![Internal ST-RAE for the four Workbench dataset views and their deployed average](reports/figures/81_regression/02_dataset_views.png)

**What to learn:** the broader `union18` view has lower macro ST-RAE than challenge-only `aux8` (**0.553 vs 0.571**), but it is slightly worse on D6 (**0.749 vs 0.734**). Averaging the views gives **0.542 macro**, with D6 improving to **0.703** after adding its two specialist views. More auxiliary data helps unevenly across endpoints; no single view wins everywhere. This comparison changes the supervised task set, so it does not isolate the causal effect of ChEMBL or Veith individually. A dash means that a D6-only view does not produce that endpoint; its four-endpoint macro is undefined.

### 4.2. The six core ensemble arms

The table summarizes each standalone arm on the common panel. “Monroe + geometry” uses the retrospectively selected endpoint subsets. A weaker standalone arm can still help a blend if its errors differ from those of the stronger arms; §4.4 measures that contribution directly.

| Standalone arm | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | Macro ST-RAE |
|---|---:|---:|---:|---:|---:|
| Workbench direct | 0.613 | 0.449 | 0.703 | 0.403 | 0.542 |
| Workbench → Kumo | 0.582 | 0.464 | 0.809 | 0.331 | 0.546 |
| Workbench → TabPFN | 0.590 | 0.455 | 0.874 | 0.317 | 0.559 |
| Monroe + geometry → Kumo | 0.697 | 0.541 | 0.704 | 0.473 | 0.604 |
| Challenge-MT CheMeleon → Kumo | 0.736 | 0.562 | 0.806 | 0.404 | 0.627 |
| AUX30 CheMeleon → Kumo | 0.639 | 0.481 | 0.724 | 0.421 | 0.566 |

#### Workbench direct

![Workbench direct predictions against measured pIC50, one panel per CYP](reports/figures/81_regression/03_workbench_direct.png)

This is the molecular-model ensemble before any tabular head. It is the strongest standalone core arm by macro ST-RAE in this comparison. The D6 cloud is compressed relative to the measured range: predictions capture some ordering but underrepresent the most extreme point estimates.

#### Workbench → Kumo

![Workbench Kumo predictions against measured pIC50](reports/figures/81_regression/04_workbench_kumo.png)

Replacing the direct readout with Kumo on the cached hidden/semantic representation improves A2 (**0.613 → 0.582**) and A4 (**0.403 → 0.331**), while worsening C9 and D6. The improvement is endpoint-specific, not a universal benefit of a tabular foundation head.

#### Workbench → TabPFN

![Workbench TabPFN predictions against measured pIC50](reports/figures/81_regression/05_workbench_tabpfn.png)

The same representation with TabPFN gives the best standalone A4 ST-RAE among the six core arms (**0.317**), but the weakest D6 result (**0.874**). This asymmetry helps explain why the recipe assigns large TabPFN weight to A4 and a small weight to D6.

#### Monroe + heme geometry → Kumo

![Monroe plus selected heme geometry predictions against measured pIC50](reports/figures/81_regression/06_monroe_geometry.png)

This arm adds a different molecular representation and pose-derived descriptors. Its standalone macro score (**0.604**) is weaker than the Workbench arms. Its D6 score (**0.704**) is competitive with direct Workbench, making endpoint-level behavior more informative than the macro ranking alone. The separate paired geometry comparison in §4.5 distinguishes this entire arm from the incremental value of geometry.

#### Challenge-multitask CheMeleon → Kumo

![Challenge-multitask CheMeleon predictions against measured pIC50](reports/figures/81_regression/07_chemeleon_mt.png)

This encoder is adapted only on the four direct targets. Its macro ST-RAE is **0.627**. The plot exposes the residual spread and compressed prediction ranges that remain after adaptation; adding this representation does not automatically improve the final blend.

#### AUX30 CheMeleon → Kumo

![AUX30 CheMeleon predictions against measured pIC50](reports/figures/81_regression/08_aux30.png)

Broader continuous auxiliary supervision lowers macro ST-RAE relative to the challenge-multitask CheMeleon arm (**0.627 → 0.566**), particularly on A2, C9 and D6. A4 worsens slightly (**0.404 → 0.421**). This comparison changes the complete adaptation recipe, including tasks and training data; it is not a controlled ablation of one external source.

### 4.3. The fixed additions

#### D6 chemistry + cofold representation

![Paired ST-RAE reductions for D6 cofold controls and the 10% ensemble addition](reports/figures/81_regression/09_d6_cofold.png)

Each row is a different matched comparison on the same **149 D6 observations**. Chemistry plus real cofold features improves on chemistry alone by **0.076 ST-RAE**, but the comparisons against ligand-only and shuffled-cofold controls are less conclusive. Adding the cofold head to the already weighted ensemble at 10% gives a much smaller D6 reduction (**0.0063**), with an interval crossing zero. Strong standalone control results do not imply an equally large incremental ensemble gain.

#### A2/A4 specialists and seed averaging

![Paired specialist-head and seed-average effects for A2 and A4](reports/figures/81_regression/10_specialists.png)

The first four rows for each endpoint compare individual refits or their three-seed mean with the original corresponding Workbench tabular head. The last row instead measures adding the mean to the then-current complete ensemble at 20%. The standalone mean does not beat the original head by the point estimate, yet the ensemble additions improve A2 by **0.0039** and A4 by **0.0068**. Seed averaging and blending with other model families answer different questions.

#### ADME-pretrained Chemprop fine-tuning

![ADME-initialized fine-tuned Chemprop predictions against measured pIC50](reports/figures/81_regression/11_adme.png)

The standalone 12-head ADME model has macro ST-RAE **0.615**. Its 10% addition nevertheless improves the previous complete ensemble from **0.517438 to 0.514789**, driven by A2 and C9; D6 and A4 worsen slightly at this step. These OOF predictions come from the assigned outer-fold fits. They do not validate the five-full-data-model averaging used for blind deployment exactly.

### 4.4. Ensembling: stages, coefficients and component contributions

![Internal endpoint and macro ST-RAE through successive ensemble stages](reports/figures/81_regression/12_ensemble_stages.png)

The fixed original three-arm average already improves macro ST-RAE over direct Workbench (**0.542 → 0.536**). Equal weighting of all six changes the macro only slightly (**0.535**). The historical weighted six-arm mixer reaches **0.522**, then cofold, specialists, ADME and geometry bring the final native internal score to **0.512784**. The weighted stages retain the known meta-training dependence; this progression is a reconstruction of development results, not an unbiased learning curve.

Read the endpoint columns as well as the macro: additions can help one CYP while hurting another. For example, the final ensemble’s A4 score (**0.338**) is weaker than standalone TabPFN (**0.317**), even though its macro is better.

![Effective prediction coefficients after all submitted fixed additions](reports/figures/81_regression/15_effective_weights.png)

This is the **deployment recipe**, after multiplying through the downstream blend factors. The ADME coefficient is 10% everywhere; specialists contribute 18% on A2/A4, and cofold contributes 9% on D6. These coefficients describe arithmetic prediction mixing, not feature importance or a measured performance benefit. Internal reconstructions use the historical **fold-specific** weights, whereas this figure shows the all-data submission weights.

![Paired change when each prediction arm is removed from the final raw ensemble](reports/figures/81_regression/17_arm_contributions.png)

Here each arm is removed from the final native internal blend, with all remaining weights renormalized and **no refitting**. For arm prediction `p` and its effective fold-specific coefficient `w`, the comparison uses `prediction_without_arm = (prediction_final − w × p) / (1 − w)`. Positive reductions favor retaining the arm. Endpoint-restricted additions have zero coefficient outside their active CYPs; the displayed macro still weights all four endpoints equally.

On this reused panel, retaining the challenge-multitask CheMeleon arm worsens the macro point estimate by about **0.0036**. Retaining AUX30 has a smaller negative macro point estimate (**0.0024**) with an interval spanning zero. WB/TabPFN and the specialists have the clearest positive conditional point changes. These are diagnostics of the frozen weights and available predictions, not reasons to claim a new optimally selected ensemble or to alter the accepted submission.

### 4.5. Geometry replacement and the final native predictions

![Paired effect of selected geometry versus matched Monroe and of replacing the ensemble component](reports/figures/81_regression/14_geometry_increment.png)

The first four rows compare the selected geometry-augmented arm with **fresh matched Monroe heads**, endpoint by endpoint. D6 has the largest point reduction (**0.0372**). A4’s selected concatenation is slightly worse than matched Monroe (**−0.0019**). Selection among geometry subsets used these same labels, so the endpoint intervals omit selection uncertainty.

The last row measures the actual complete-ensemble replacement. Its macro reduction is only **0.002004**, with the same saved 95% interval **[−0.000159, +0.004199]** reproduced from the original bootstrap seed. Small effective component weights and downstream attenuation explain why a larger standalone-arm change can have a small final effect.

![Final complete ensemble predictions against measured pIC50 before placement](reports/figures/81_regression/16_final_ensemble.png)

The final native internal scores are **0.588 / 0.435 / 0.690 / 0.338** for A2/C9/D6/A4. A4 has the strongest rank ordering (ρ **0.901**), while D6 remains substantially compressed and less well ordered (ρ **0.599**). The macro score of **0.512784** describes this OOF prediction construction before population placement, with the validation limitations above.

### 4.6. Placement: ranking can survive while error changes sharply

![Counterfactual population placement applied to internal OOF predictions, with distributions and ST-RAE](reports/figures/81_regression/13_placement.png)

The top row compares measured values, native OOF predictions and an explicit **placement diagnostic**. For each full endpoint OOF cohort, we apply the frozen published target mean/SD and floor at 1, recomputing the input mean/sample SD from those 149 or 150 OOF predictions. No internal labels fit the transform. The bottom row recomputes ST-RAE on the same observations.

This operation changes macro ST-RAE from **0.5128 to 1.1938**. D6 changes most: its predicted mean moves **4.824 → 3.570**, its sample SD expands **0.388 → 0.900**, and ST-RAE increases **0.690 → 2.926**. Spearman remains unchanged for all four endpoints in this diagnostic, illustrating that rank preservation does not preserve interval error.

**Interpretation:** the internal analogue panel and blinded panel have different populations. Applying the blinded-population target moments to the internal cohort is not expected to recover its outcome distribution. This diagnostic illustrates placement sensitivity; it is **not** the submitted 750-molecule mapping, a fitted internal calibrator, an estimate of blind performance, or a comparison with the saved public score of 0.423257.

<!-- REPORT81_INTERNAL_PLOTS_END -->

## Reproduction from this private bundle

This repository contains the full project source snapshot and the regression-specific frozen data, representations, checkpoints, predictions and report figures. It excludes unrelated experimental artifacts, classification submissions, optimizer duplicates, installed environments, authentication files and Git history. Frozen complex features and pose geometry are the inputs used here; the 26 GB ESMFold2 generator is not required to replay this accepted fixed submission.

Hugging Face reserves `.cache/`, so cached artifacts are stored under `assets/cache/`. The replay entry point creates local compatibility links automatically.

For the guaranteed replay path, use Python 3.11 and the pinned lightweight dependencies:

```bash
python -m pip install -r requirements-replay.txt
python reproduce.py all
```

`reproduce.py` checks every file against `MANIFEST.json`, rebuilds the 750-row CSV from six component predictions, four saved geometry heads, cofold/specialist/ADME additions and population placement, and requires the accepted SHA-256 exactly. It also regenerates the 17 figures and checks endpoint metrics and bootstrap tables against the report. Outputs go under `reproduced/`; the frozen originals remain intact. `python reproduce.py verify`, `submission`, and `figures` run individual stages. No network or GPU is required for artifact replay.

The historical training/refit commands below are preserved as provenance. Their original machine paths appear in `environment/original_path_map.json`, and their original environments are recorded under `environment/`. They are not the portable replay interface. Refitting neural or tabular models on different hardware/software is not promised to produce byte-identical predictions. Exactness here means reconstruction of the accepted fixed submission and numerical report results from the supplied frozen artifacts.

## Reproduction and artifact identity

These are **artifact-dependent rebuilds**, with commands checked against the current entry points. Run-specific environments and checkpoint identities take precedence over a new installation of the latest package versions. Historical Chemprop components use different environments; see [workbench_reproduction/README.md](workbench_reproduction/README.md). Frozen source hashes can reject later source edits, so restore the recorded source revision together with its artifacts before replaying a historical recipe. The commands below rebuild local prediction files and contain no upload stage.

For a read-only identity check on the existing files, use `sha256sum` and compare with the accepted hashes below. Receipt status records acceptance for processing; the linked scored-comparison snapshots establish the subsequently observed scores. The report revision recomputed internal metrics and figures from saved predictions, checked component reconstruction and existing artifact metadata, and did not rerun model fitting or retrieve a new leaderboard snapshot.

### Regression rebuild

From the repository root, with the saved component predictions, embedding banks, geometry snapshot and historical source/checkpoint dependencies restored:

```bash
module load CUDA/12.6.0
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.cache/structured-data-models
REPORT_PYTHON=.cache/kumo-eval-venv/bin/python
$REPORT_PYTHON -m workbench_reproduction.geometry_kumo_submit prepare
$REPORT_PYTHON -m workbench_reproduction.geometry_kumo_submit evaluate
$REPORT_PYTHON -m workbench_reproduction.geometry_kumo_submit fit
$REPORT_PYTHON -m workbench_reproduction.geometry_kumo_submit build
```

Use one allocated/available GPU; `0` selects the first visible device. `prepare` verifies the frozen recipe and feature/source hashes; `evaluate` reproduces the saved retrospective internal comparison; `fit` produces four all-label geometry/Kumo heads; `build` replaces that component, applies all saved additions and placement, and validates the CSV. This rebuild reuses earlier arms; it does not retrain them. End-to-end training requires the component instructions in [Workbench reproduction](workbench_reproduction/README.md), [six-arm ensemble](reports/21_aux30_six_model_ensemble.md), [D6 cofold](reports/57_regression_cofold_submission.md), [specialists](reports/60_a2a4_specialists20_submission.md), [ADME refits](reports/74_regression_adme10_submission.md) and [geometry](reports/77_heme_geometry_kumo.md).

Expected output: [`regression_best_ensemble_heme_geometry_concat_20261029.csv`](submissions/regression_best_ensemble_heme_geometry_concat_20261029.csv), **750 rows**, columns `SMILES`, `Molecule_Name`, then `CYP1A2_pIC50_direct_inhibition`, `CYP2C9_pIC50_direct_inhibition`, `CYP2D6_pIC50_direct_inhibition`, `CYP3A4_pIC50_direct_inhibition`. Values are finite pIC50 predictions with the floor applied. The [recipe](runs/submission/regression-heme-geometry-concat-20261029/recipe.json), [fit completion](runs/submission/regression-heme-geometry-concat-20261029/fit_completion.json), [integrity audit](runs/submission/regression-heme-geometry-concat-20261029/integrity_audit.json) and [receipt](submissions/regression_best_ensemble_heme_geometry_concat_20261029_receipt.json) record identity, membership and row-order checks. Reconstruction and replacement identities are checked to `1e-12`; that arithmetic tolerance is not a guarantee of identical predictions after refitting on different hardware or software.

Accepted CSV SHA-256: `541f3023cd5b7be9873d3727431f596784069b727d8af8d3afeb56b90f22c6ab`.

## References and method provenance

Literature and web references were checked on **3 November 2026**. These sources identify borrowed methods, data and released software. Exact experiment identities are established by the local hashes, checkpoints and recipes; live documentation may describe newer software.

- **Challenge data and assay design.** OpenADMET, [CYP challenge dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test) (use revision `146282d2ce1532f1803679a834a9bec9dac1e482`); Orr, Simpkins and MacDermott-Opeskin (2026), [Lowering Inhibitions, One CYP at a Time](https://openadmet.github.io/octant-cyp-inhib-blog-post/); [official tutorial and validators](https://github.com/OpenADMET/CYP-Challenge-Tutorial). Sources for the assay, analogue-expansion design and evaluation conventions.

- **Borrowed Workbench recipe.** SuperCowPowers, [The CYP Challenge](https://supercowpowers.github.io/workbench/blogs/cyp_challenge/), with the [upstream pipeline pinned to `20e09bd…`](https://github.com/SuperCowPowers/workbench/tree/20e09bd049f2724b2e9fb2a8b4085a941ee21b10/ml_pipelines/OpenADMET/cyp). Source of the historical four-view Chemprop ensemble and inherited population placement; the present hybrid ensemble extends it.

- **D-MPNN molecular models.** Yang et al. (2019), [Analyzing Learned Molecular Representations for Property Prediction](https://doi.org/10.1021/acs.jcim.9b00237), *Journal of Chemical Information and Modeling* 59, 3370–3388. Architecture underlying the Chemprop components; run-specific settings come from this repository.

- **CheMeleon pretraining and transfer.** Burns, Zalte and Green (2025), [Descriptor-based Foundation Models for Molecular Property Prediction](https://arxiv.org/abs/2506.15792); [released weights, Zenodo 15460715](https://doi.org/10.5281/zenodo.15460715); [official Chemprop fine-tuning workflow](https://chemprop.readthedocs.io/en/main/chemeleon_foundation_finetuning.html). Basis for the 2,048D encoder, with local challenge/AUX30 adaptation.

- **Monroe molecular representation.** Banaszewski and Fitzgibbon (2026), [Monroe: A Molecular Foundation Model for In-Context Probabilistic Inference](https://arxiv.org/abs/2608.18982); [source at the recorded revision `57238ed…`](https://github.com/blazejba/Monroe/blob/57238edfffea03808abe761a00cd9a75fa41bb95/README.md). Basis for the 720D representation; CYP-specific adaptation and Kumo heads are local extensions.

- **Kumo tabular foundation model.** NVIDIA, Qu et al. (2026), [NVIDIA Kumo Tabular Sets a New Accuracy-Efficiency Frontier for Tabular Prediction](https://huggingface.co/blog/nvidia/kumo-tabular); [official KumoTabular API](https://nvidia.github.io/structured-data-models/api/generated/sdm.models.KumoTabular). Software/model reference for the saved Kumo-L v1.0.0 components.

- **TabPFN version used.** Prior Labs (2026), [TabPFN-3.5 model card](https://huggingface.co/Prior-Labs/tabpfn_3_5). Version-specific reference for the shared regression/classification checkpoint and in-context inference; local regression runs use three estimators.

- **External CYP assays.** Veith et al. (2009), [Comprehensive characterization of cytochrome P450 isozyme selectivity across chemical libraries](https://doi.org/10.1038/nbt.1581), *Nature Biotechnology* 27, 1050–1055; [PubChem AID 1851](https://pubchem.ncbi.nlm.nih.gov/bioassay/1851); [ChEMBL database](https://www.ebi.ac.uk/chembl/). The local manifest identifies the exact ChEMBL 37 and Veith exports used.

- **Circular fingerprints and implementation.** Rogers and Hahn (2010), [Extended-Connectivity Fingerprints](https://doi.org/10.1021/ci100050t), *Journal of Chemical Information and Modeling* 50, 742–754; [RDKit Morgan fingerprint documentation](https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints). Basis for ECFP4/Morgan chemistry controls and Tanimoto similarity.

- **Scaffold grouping.** Bemis and Murcko (1996), [The Properties of Known Drugs. 1. Molecular Frameworks](https://doi.org/10.1021/jm9602928), *Journal of Medicinal Chemistry* 39, 2887–2893. Basis for scaffold identity; the anchor-preserving split and parent exclusions are local validation choices.

- **Complex foundation model.** Biohub, [ESMFold2-Fast release](https://huggingface.co/biohub/ESMFold2-Fast) and [official complex-input implementation and citation](https://github.com/Biohub/esm#citations); Candido et al. (2026), [Language Modeling Materializes a World Model of Protein Biology](https://www.biorxiv.org/content/10.64898/2026.06.03.729735), preprint. Local code defines the pair-state pooling and heme-coordinate features.

- **ADME initialization.** OpenADMET, [Chemprop foundation pretraining weights](https://huggingface.co/openadmet/chemprop-foundation-pretraining-weights/tree/0f8201d06c4656c3f97bc244ce2dc18ac2881be9) and [pretraining experiment description](https://openadmet.ghost.io/throwing-everything-and-the-kitchen-sink-at-chemeleon/). Source of `surrogate_adme__s42.pt`; the 12-head CYP fine-tuning and 10% addition are local.
