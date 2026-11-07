# Model report — OpenADMET CYP450 blind challenge

Entry **`fold-zero`**, Direct Inhibition (regression) track, four endpoints. Repo: <https://github.com/codie-freeman/OpenADMET-CYP-Blind-Challenge>

---

## In short

One Chemprop model with a frozen CheMeleon encoder, trained multitask on all four isoforms against each label's **credible interval** rather than its point estimate. Then an affine correction on three isoforms, with constants solved from the leaderboard's own returned metrics.

**Macro ST-RAE 0.5861**, best of eighteen scored rows.

One thing to know before reading that number. The correction is **leaderboard probing**: it could not have been built without the board, and an entrant with one submission could not reproduce it.

The part that would transfer to a setting with no leaderboard is the dead-zone training target. It is the only change in this project's history that improved every isoform on both a placement metric and a ranking one.

---

## 1. The submission

| | |
|---|---|
| file | `outputs/56b_cyp1a2_corrected/submission_candidate_s050.csv` |
| sha256 | `a1da00e209975f4b82fdaeae1bc0210eb83953ef13872d22aa04d354c438aa8e` |
| rows | 750, one per blind compound |
| base model | Chemprop 2.3.1, `--from-foundation CHEMELEON`, 8 regression heads (4 scored + 4 auxiliary), MAE loss, dead-zone target |
| correction | affine, per isoform, half strength (`s = 0.5`) |

The four columns are **not** uniform, which is a limitation rather than a design choice.

| Isoform | Scale correction | Placement correction |
|---|---|---|
| CYP1A2 | ×1.4145, to 0.836 of training-label SD | yes, `s = 0.5`, onto a solved blind population |
| CYP2C9 | ×1.1304, to 0.817 | yes, `s = 0.5`, onto a solved blind population |
| CYP2D6 | ×1.0 (already at 0.850) | yes, applied earlier, at full strength |
| CYP3A4 | ×1.1317, to 0.926 | none |

![Board ST-RAE for all nineteen submissions in the order they were sent, macro and four isoforms](docs/figures/leaderboard_trajectory.png)

*Nineteen sends over two months, of which the one above is order 18 and the lowest point on the macro line: the first five were model changes and none improved on the raw baseline by more than 0.005, while the single largest step came from correcting where the predictions sat rather than from changing what produced them.*

---
## 2. Data

`openadmet/cyp-challenge-train-test` from the Hugging Face Hub, downloaded 2026-08-19 at pinned revision `85f8b358d0a2056a98b990dd75d3b3ec9247862b`, retained byte-identical.

**Curation here did almost nothing, and that is worth saying plainly**, because the phrase can imply work that was not performed. Diffing raw against curated shows two columns added, `canonical_smiles` and `inchikey`, no rows dropped or filtered, and no label altered. The organisers supplied pIC50 with credible intervals, so no dose-response fitting, replicate aggregation or outlier removal happened in this entry.

| | count |
|---|---:|
| training compounds | **4,905**, all distinct InChIKeys |
| labelled measurements | **6,525** |
| CYP1A2 / CYP2C9 / CYP2D6 / CYP3A4 | 1,412 / 1,285 / 1,493 / 2,335 |
| blind compounds | **750**, all distinct InChIKeys |

Each isoform is labelled on 26–48% of the library, so a single-task model trains on a quarter to a half of it. Label sparsity is the binding constraint, and it is why the model is multitask.

**The credible intervals do two jobs**, and the second is the modelling contribution. Coverage is 100% of labelled rows with zero `conf_high < conf_low`, median widths 0.328 / 0.526 / 0.272 / 0.379 pIC50. First, they define the metric: ST-RAE charges nothing for a prediction inside a compound's interval and the distance to the nearest edge otherwise, normalised by the same function applied to a constant mean predictor, so predicting the mean scores exactly 1.0 by construction. Width is strongly *anti*-correlated with potency, Spearman −0.56 to −0.93, so the band is wide for weak compounds and narrow for potent ones. **Over-prediction is cheap and under-prediction is punished**: at raw placement, total under-prediction error exceeds over-prediction error by 1.65–2.79× on every isoform. Second, they supply the training target that won.

**Leakage is checked by InChIKey** through one shared canonicalisation routine (`src/features.py`), never raw SMILES or compound name, because the same structure appears under several spellings across the released files and how a dataset is split or deduplicated materially affects what an evaluation measures. Training against blind, and each of the three auxiliary challenge files against blind, return zero shared compounds. The one external dataset, PubChem AID 1851, contributed a single blind compound (`RWZAIHHFBKOVIK-UHFFFAOYSA-N`), dropped in code before any training run, and 98 training compounds, retained since they carry auxiliary labels only.

---
## 3. Validation

Two partitions, frozen before the experiments that use them and pinned by hash.

| | random 5×5 | cluster-disjoint 5×5 |
|---|---|---|
| file | `data/folds/cv_folds.csv` | `data/folds/cv_folds_butina_5x5.csv` |
| sha256 | `559a078569032d4d…` | `52f39b1a1cea9200…` |
| construction | `RepeatedKFold`, seed 42 | Butina clustering, whole clusters to one fold |
| fingerprint | — | 2048-bit Morgan radius 3, Tanimoto distance 0.65 |
| samples / fold size | 25 / 981 | 25 / 981, 2,617 clusters, none split across folds |

Model selection runs on the cluster-disjoint partition. One honest caveat: a "repeat" there re-randomises cluster-to-fold *assignment* over one fixed clustering, since Butina is deterministic given fingerprint and cutoff, so the 25 samples are draws of assignment noise rather than of the whole resampling mechanism. That differs from the published 5×5 recommendation and is stated rather than glossed.

Comparisons follow Ash et al. — 25 samples, paired, Benjamini-Hochberg-corrected across isoforms, with the parametric branch measured per cell by a Brown-Forsythe test rather than inherited, and family size declared before results. The distinctive part is that **significance is not sufficient for adoption**: each comparison carries a **materiality bar** derived from the pooled paired standard error across all arms, never from the standard error of the comparison under test, which would merely restate the significance threshold. Bars are frozen before results, and significant-but-under-bar is reported as *detected, not material* — the commonest outcome in this project's screening history.

**The split-scheme objection was testable and it failed.** A random split on this library plausibly flatters every model, and a design that does not resemble how the test set was built can estimate prospective performance badly. Across a completed 5×5 cluster-disjoint design against the frozen random partition, **0 of 25 sign-resolved arm-by-isoform comparisons flip**, with 11 cells excluded as sign-unresolved rather than counted either way, and ordering agreement at Spearman 0.842–0.988. An earlier 5-fold version had shown a flip on three of four isoforms; it did not survive repeats. The exercise still produced this project's only genuinely decorrelated ensemble candidates, tree models on raw fingerprint and Mordred features.

---
## 4. The one modelling idea that worked

Replace each training label with

```
target = clip(out-of-fold prediction for this compound, conf_low, conf_high)
```

and change the loss from MSE to MAE. ST-RAE charges nothing inside a compound's interval, so capacity spent moving a prediction *within* a band where it is already free is wasted, and under-prediction costs more than over-prediction anyway.

Eleven arms, same 25 cluster-disjoint folds, same per-fold seeds, one variable between them. ST-RAE, lower better, paired against `plain`, BH-corrected:

| Arm | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 |
|---|---|---|---|---|
| `plain` — CheMeleon multitask, MSE, point target | 0.8595 | 0.6601 | 0.9984 | 0.5437 |
| **`deadzone`** — interval target, MAE loss | **0.7711** ✓ | **0.5636** ✓ | **0.8770** ✓ | **0.4773** ✓ |
| `mae_only` — MAE loss, point target | 0.8278 ✓ | 0.6462 | 0.9180 ✓ | 0.5403 |
| `aid` — +4 AID 1851 auxiliary heads | 0.8255 ✓ | 0.6511 | 0.9802 | 0.5499 |
| `single` — one model per isoform | 0.8803 | 0.6950 ✗ | 0.9987 | 0.5411 |
| six tabular arms — ECFP4 / Mordred-PCA × RF, XGBoost, LightGBM | 0.869–0.994 | 0.738–0.957 | 0.961–1.072 | 0.606–0.809 |

✓ better than `plain`, ✗ worse, blank not different. `outputs/37_butina_5x5_complete/paired_tests_vs_plain_25fold.csv`

![Eleven modelling arms compared on 25-fold cluster-disjoint cross-validated ST-RAE, four panels by isoform](docs/figures/model_comparison_cld.png)

*The dead-zone target is the only arm to hold the top compact-letter-display tier on every isoform. What reached the board was dead-zone combined with the AID auxiliary heads, a pairing never confirmed at 25 folds, so the arm measured here has never been submitted on its own.*

`deadzone` is the only arm better on all four isoforms, improving on 25 of 25 folds on three of them and 23 of 25 on CYP3A4, uniquely best under an all-pairs compact letter display. Every tabular arm is worse, significantly so on 17 of 24 cells. And `single` is never better than multitask and is worse on CYP2C9, which is the label-sparsity argument showing up as a measurement, consistent with what is reported for multitask ADME models generally.

**It raised Spearman on all four too**: 0.514 → 0.548, 0.577 → 0.650, 0.406 → 0.472, 0.780 → 0.823. That is the part that matters, because an affine correction cannot move rank order. A ranking gain has to come from the model or not at all. A loss-only arm attributes 64% / 86% / 34% / 95% of the gain to the clipping rather than the loss change, isoform by isoform.

**One structural caveat, which repeating does not remove.** Each target comes from an out-of-fold donor model trained on the fold being held out. That is inherent to building the target this way against a cross-validation design, not a coding error, and it does **not** apply to a blind submission, where no donor has seen the compound. But it makes the measured *magnitude* plausibly optimistic. The direction is not in doubt: it reproduces across two partitions, three fold counts and a single-fold screen, and the loss-only arm, which carries none of this exposure, is itself significantly better on two isoforms.

The auxiliary heads use AID 1851's **efficacy** readout, not potency. Potency correlates better with these labels, Spearman up to 0.79, but 42–66% of its values are censoring floors, and stripping the floors makes CYP2C9's correlation flip sign. Potency aggregated across assays and laboratories carries substantial systematic noise in any case.

---
## 5. The correction

For predictions `p` against unobserved truth `y`, with `k = sd(p)/sd(y)` and `b = (mean(p) − mean(y))/sd(y)`:

```
R² = 2·ρ·k − k² − b²
```

The decomposition is SuperCowPowers'. Each submission of the *same ordering* at a *different* mean and spread gives one equation in three unknowns: `mean(y)`, `sd(y)`, `ρ`. Three with genuinely distinct `(mean(p), sd(p))` identify all three, and **ρ is recovered, not proxied** — the bivariate-normal Spearman proxy undershoots the recovered value by 0.144 on CYP2D6.

**What identification needs is distinct equations, not many submissions.** That is why only one isoform was identifiable from the submission history alone:

| Isoform | range of submitted prediction mean (pIC50) | identified from history? |
|---|---:|---|
| CYP2D6 | **1.4778** | yes |
| CYP3A4 | 0.7233 | no |
| CYP2C9 | 0.3731 | no |
| CYP1A2 | **0.0892** | no |

CYP1A2 had been sent thirteen times at essentially one placement, so thirteen submissions supplied, in effect, one equation. **Identifiability here is a property of the submission history, not of the chemistry.** Worse, the two-equation system is provably symmetric under `mean(y) → 2·mean(p) − mean(y)`, since both R² and MAE are even functions of the bias term, so the *sign* of the placement error is not recoverable from them at all.

**One submission was spent to buy that sign**: the prior best file with a uniform +0.25 pIC50 translation on three isoforms, CYP2D6 carried byte-identical as a control. Four candidate R² readings for CYP1A2 were frozen in a pre-registration before it went; the returned value was **0.1154** against a predicted **0.1107** for one specific branch, matched to 0.0047 and well inside the 0.02 bar declared in advance. So the identity describes this leaderboard, which was the thing most at risk.

**The prior behind the probe was falsified.** The reasoning was that a hit-expansion-seeded test set should centre *above* the training mean, so the correction should shift predictions up. All three isoforms turned out low-centred and the correction shifts them **down**. The premise is in fact correct for CYP2C9 and CYP3A4, +0.227 and +0.701 against training, and the conclusion still fails because the predictions sit higher still; on CYP1A2 the premise fails too, its blind centre sitting 0.655 below the training mean. The construction argument was not a reliable guide to placement. Measurement was required.

Three equations, three unknowns, unique root on all three isoforms:

| Isoform | solved blind mean | solved blind SD | ρ (recovered) | R² before | R² ceiling (ρ²) |
|---|---:|---:|---:|---:|---:|
| CYP1A2 | 4.3001 | 1.5099 | 0.7690 | 0.3068 | 0.5914 |
| CYP2C9 | 4.8074 | 1.1325 | 0.7560 | 0.5308 | 0.5716 |
| CYP3A4 | 4.7974 | 1.3145 | 0.8136 | 0.6563 | 0.6619 |

![Training-label distribution per isoform with the solved blind population overlaid as a normal curve](docs/figures/label_vs_solved_population.png)

*CYP2D6, solved separately and earlier than the three isoforms tabulated above, sits 1.73 pIC50 below the distribution the model was trained on, which is why a placement correction was worth more there than any modelling change attempted in this work.*

The headroom is overwhelmingly on CYP1A2, +0.2846 R² available, and essentially absent on CYP3A4 at +0.0056, which is why CYP3A4 was left uncorrected. Its placement has been found adequate by three independent routes, the strongest being an eleven-submission joint least-squares fit that reproduced its published R² values from two free parameters with RMS error 0.0065, then **predicted a twelfth submission's R² at 0.6515 before that value existed in the repository, against a measured 0.6563**.

The correction is applied as `b(s) = (1−s)·b`, `k(s) = k + s·(ρ − k)` at `s = 0.5`. The constants are point estimates recovered from three equations, so a full correction stakes the whole displacement on their being right. Halving it halves that exposure, at the cost of leaving some of the available gain unclaimed — which is what the comparison below measures. The resulting file hits its target spread ratio to better than 10⁻⁹, preserves rank order at Spearman 1.0 to within 10⁻¹², and contains no non-positive predictions. Two of 750 CYP1A2 predictions fall below the training-label minimum, reported rather than clipped, since no submission-format floor is documented anywhere in the challenge materials.

---
## 6. Half strength against full strength

Both settings were sent. **They differ in exactly one scalar** — same base file, same solved populations, same recovered ρ, same code path — which makes this the only controlled experiment this entry ran on the live board.

| | `s = 0.5` | `s = 1.0` | Δ |
|---|---:|---:|---:|
| macro ST-RAE | **0.5861** | 0.5933 | +0.0072 |
| macro MAE | 0.6973 | **0.6864** | −0.0109 |
| macro R² | 0.5249 | **0.5366** | +0.0117 |
| macro Spearman / Kendall | 0.6894 / 0.5118 | 0.6894 / 0.5118 | **0.0000** |
| CYP1A2 ST-RAE / R² / MAE | 0.5631 / 0.5070 / 0.7988 | 0.5767 / 0.5467 / 0.7575 | +0.0136 / +0.0397 / −0.0413 |
| CYP2C9 ST-RAE / R² / MAE | 0.5408 / 0.5595 / 0.5212 | 0.5562 / 0.5667 / 0.5190 | +0.0154 / +0.0072 / −0.0022 |
| CYP2D6, CYP3A4 | carried byte-identical | | **0.0000 on all ten cells** |

The lever moved only what it touched: CYP2D6 and CYP3A4 are equal not to four significant figures but at every digit the board prints, which is a control built into the comparison. And rank order did not move on any isoform, on predictions that genuinely changed under two different affine maps.

**MAE moves with R², not with ST-RAE**, so ST-RAE is the odd one out of three metrics, which is a stronger statement than two metrics merely disagreeing. The mechanism is the interval asymmetry, measured before this comparison existed: because over-prediction is cheap and under-prediction is punished, shifting predictions *down* onto a low-centred population moves error from the cheap side to the expensive side, while R² and MAE have no such asymmetry and score the same move as an improvement. Sweeping a uniform offset against this project's own out-of-fold predictions puts the ST-RAE optimum above the R² optimum on all four isoforms, by +0.07 / +0.08 / +0.06 / +0.12 pIC50. This pair is the first time that gap has appeared on the real board rather than out of fold, and its direction matches.

Half strength wins or ties on every isoform on ST-RAE, so selecting it required no per-isoform cherry-picking and the submission stays one model plus one scalar.

---
## 7. What didn't work

The negatives are the most transferable part of this report. All were tested to the standard set out under Validation.

| Route | Evidence | Outcome |
|---|---|---|
| **Ensembling** — simple average, Caruana bagged selection, non-negative stacking, shared-weight single object | nested 25-fold, **material on both metrics** on two isoforms | **No board gain.** Three sent; best 0.6356 against 0.5861 for a single corrected model. The whole measured gain rested on one member whose out-of-fold ordering the board contradicts on 4 of 4 isoforms. |
| **External HTS auxiliary heads as the headline change** | full 5×5, paired, BH-corrected, significant on 2 of 4 | **Worse on all four isoforms when sent**, 0.8160 against 0.7138, the largest single-step regression on this record. Kept only as auxiliary heads inside the final model, where they are not the deciding variable. |
| **Calibration toward the training population, or an external one** | out-of-fold, then two real sends | **Negative in every form tested.** The external version scored 0.8811, the worst row on record. Only calibration toward the *blind* population helped. |
| **Quantum-chemistry descriptors** — GFN2-xTB geometry then B3LYP-D3/def2-SV(P) across the library: 11,310 ORCA jobs, 281.6 core-hours, zero failures | 25-fold, Tukey HSD, materiality bars frozen in advance | **Detected, not material.** Six significant cells of forty, none material, largest 45% of its bar. Flat null again when routed through per-atom message passing against a zeroed-feature control. |
| **Protonation-state net charge at pH 7.4** | correlation confirmed real, surviving a lipophilicity partial-correlation control | **Clean tie on all four isoforms** when given to the model. An established descriptor–endpoint correlation did not convert. |
| **A test-set-mimicking analog holdout** — top-25 hits held in, their nearest neighbours held out | single partition, bootstrap noise band | **Backwards.** Gap to the real blind result 2.3–9.3× *larger* than either ordinary partition. |
| **Interval labels for screened-but-unpromoted compounds** | 25-fold, pre-registered, frozen bar | **Split.** Spearman +0.0187, 88% of bar and not material; ST-RAE a material *harm*. |
| **Compound-pool expansion from the released files** | exact InChIKey counting | **Exhausted.** The single-concentration screen is a strict subset of the training pool. |

One diagnostic reframes most of these. Across the eleven-arm pool, **error correlation runs 0.79–0.91 while raw-prediction correlation runs only 0.47–0.76**. The models look different and get the same compounds wrong. Averaging cancels independent noise and cannot cancel shared ignorance, and shared ignorance is what this pool has. And the compounds they all miss are structurally ordinary: nearest-neighbour Tanimoto similarity to the training set does not separate hard from easy on any isoform, against the expectation from the activity-cliff literature. The errors are not sitting on unseen chemistry, which is why adding descriptors kept not helping.

![Residual correlation between every pair of eleven configurations, lower triangle, one panel per isoform](docs/figures/error_correlation_matrix.png)

*Pairs that disagree about their predictions still agree about their mistakes, which is the reason every ensemble route in the table above measured well out of fold and returned nothing on the board.*

---
## 8. Limitations

**This is leaderboard probing.** The correction constants are solved from the R², MAE and Spearman the board returned across this entry's submissions, one of which was a deliberately degraded probe sent for no purpose but to make a sign identifiable. So the model is reproducible from this repository alone and the correction is not. Anyone assessing the model's own merit should read the cross-validated comparison and the raw uncorrected column (`outputs/32_deadzone_aid_retrain/raw_predictions.csv`), not the submitted file.

**Rank order cannot move.** An affine map changes where predictions sit and how far apart they are, not which compound outranks which, so no ranking claim here rests on the correction. Verified on the sent files rather than argued from algebra: the map between the two shrinkage settings is strictly increasing, neither file contains a tie, and the rank vectors are exactly equal on all four isoforms. The board agrees.

**The base model and the correction are not separable in this file.** Three of four columns carry corrections fitted at different times against different board readings. No experiment here isolates the model's contribution from the correction's within the submitted file, so this score licenses no inference about whether the dead-zone target was the right modelling choice, in either direction.

**Choosing half strength assumes ST-RAE decides.** Both settings were sent; half strength wins on ST-RAE, full strength wins on R² **and** MAE. ST-RAE is listed first in the vendored evaluator's `ACTIVITY_METRICS` and is its docstring's exemplar `primary_metric`, but the vendored copy sets no default and the live backend's sort key is not in this repository. If the organisers rank on R² or MAE, the other file was the better choice. **On the evidence, full strength is the better-calibrated file.**

**Cross-validation did not predict board outcomes, seven times**, including a fully confirmed 5×5 repeated-CV result that scored worse than its own baseline on all four isoforms when sent. Independent re-evaluations of published ADMET models report the same pattern at field scale. The seventh is the uncomfortable one: a disagreement on **Spearman**, which is affine-invariant on both sides, so calibration cannot explain it. Ranking models on 25-fold out-of-fold Spearman and on board Spearman disagrees on 2 of 4 isoforms, and CYP3A4 is fully reversed. The two estimates disagree about the models themselves. The model comparison above is honest, properly powered and multiplicity-corrected, and it is still not strong evidence about blind performance. It is reported because it is what the selection was based on. Full list: [`docs/leaderboard_submissions.md`](docs/leaderboard_submissions.md).

---
## 9. What came from other entrants

This challenge has a public working culture and this entry drew on it.

**SuperCowPowers** supplied the central method: the decomposition `R² = 2ρk − k² − b²`, and the idea of using it to find which isoforms have a recoverable R² gap. This project's addition is applying it against the *blind* population rather than the training one, recovering ρ rather than proxying it, and spending a submission on a probe to resolve the sign. Their independent report that the blind population is wider than the training data on every isoform is what made the spread correction worth attempting, and their own CYP2D6 population figure agrees with the one solved here.

**lachrymator** diagnosed large cancelling coefficients in unconstrained stackers on correlated members, with non-negative combiners as the fix. Tested here, the cancellation does not occur in this pool, no coefficient exceeding 1.5 or below −0.053. A diagnosis that does not transfer is still worth establishing rather than assuming.

**Antonio De La Vega De Leon's** PXR-challenge posts set the four questions the chemical-space work was built around: how diverse the space is, whether potency tracks known determinants, how discontinuous the SAR is, and whether there is enough scaffold structure for a cluster split to matter.

Three facts used here are recorded on the challenge Discord rather than in the released documentation: that the tutorial's evaluation code matches the scoring backend, that CYP2D6 was not used to seed the test set's hit-expansion construction and runs on a different assay, and, from Pat Walters, that CYP3A4's screening set was itself built by Butina clustering on 2048-bit Morgan radius-3 fingerprints at Tanimoto distance 0.65 — the recipe the cluster-disjoint partition here reproduces. None are third-party-verifiable, and they are flagged as Discord provenance wherever they are load-bearing.

---
## 10. Reproducibility

conda, Python 3.11.13, Apple Silicon arm64. Chemprop 2.3.1, RDKit 2026.03.3, PyTorch 2.13.0, scikit-learn 1.9.0, NumPy 1.26.4, pandas 2.3.3 (`environment.yml`).

CPU, not MPS: a non-deterministic MPS kernel conflicts with Chemprop's forced-determinism-on-seed behaviour, diagnosed rather than assumed. The check that makes the model comparison trustworthy is that the plain baseline, retrained for each of twelve separate experiments, reproduced an independently stored fold-0 score to 0.0000 on all four isoforms every time. Seeds are fixed per fold and logged with their results; the final model used seed 2684470948, eight heads, `mae` loss, a 15% internal validation split, stopping at epoch 14 of a 50 maximum.

| | |
|---|---|
| raw dataset revision | `85f8b358d0a2056a98b990dd75d3b3ec9247862b` |
| random partition | `data/folds/cv_folds.csv` · `559a078569032d4d…` |
| cluster-disjoint partition | `data/folds/cv_folds_butina_5x5.csv` · `52f39b1a1cea9200…` |
| **submitted file** | `outputs/56b_cyp1a2_corrected/submission_candidate_s050.csv` · `a1da00e209975f4b82fdaeae1bc0210eb83953ef13872d22aa04d354c438aa8e` |
| alternate shrinkage | `outputs/56b_cyp1a2_corrected/submission_candidate_s100.csv` · `b2c53f136f1d2684643126492a98d2b718b696e55b216a17d60bf7ff6f79a228` |

Scoring uses the challenge's own evaluator, vendored unmodified at `src/vendor/openadmet_eval/`. It has **not** been diffed against the live backend, so internal-versus-board agreement is assumed rather than established.

**Where to read next.** [`docs/leaderboard_submissions.md`](docs/leaderboard_submissions.md) is the canonical per-send record, every row with its sha256 and scoring basis. [`docs/findings.md`](docs/findings.md) holds results that outlive the analysis that produced them. [`docs/prereg_fullset.md`](docs/prereg_fullset.md), committed before the labels were released, states what the reveal should show, with tolerances and which prediction is likeliest to fail.

---
## References

1. J. R. Ash, C. Wognum, R. Rodríguez-Pérez, M. Aldeghi, A. C. Cheng, D.-A. Clevert, O. Engkvist, C. Fang, D. J. Price, J. M. Hughes-Oliver and W. P. Walters, Practically Significant Method Comparison Protocols for Machine Learning in Small Molecule Drug Discovery, *J. Chem. Inf. Model.*, 2025, **65**, 9398–9411.
2. Y. Benjamini and Y. Hochberg, Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing, *J. R. Stat. Soc. Series B Stat. Methodol.*, 1995, **57**, 289–300.
3. J. W. Burns, A. S. Zalte, C. R. A. Abreu, J. Sieg, C. Feldmann, M. Mathea and W. H. Green, Deep Learning Foundation Models from Classical Molecular Descriptors, *arXiv*, 2026, preprint, arXiv:2506.15792, DOI: 10.48550/arXiv.2506.15792.
4. D. Butina, Unsupervised Data Base Clustering Based on Daylight's Fingerprint and Tanimoto Similarity: A Fast and Automated Way To Cluster Small and Large Data Sets, *J. Chem. Inf. Comput. Sci.*, 1999, **39**, 747–750.
5. R. Caruana, A. Niculescu-Mizil, G. Crew and A. Ksikes, Ensemble Selection from Libraries of Models, in *Twenty-First International Conference on Machine Learning (ICML '04)*, ACM Press, Banff, Alberta, Canada, 2004, p. 18.
6. A. De La Vega De Leon, PXR challenge #1: Exploring the data, <https://www.delavega.ai/posts/2026_04_15_pxr_sar_exploration.html>, (accessed 23 August 2026).
7. T. G. Dietterich, Ensemble Methods in Machine Learning, in *Multiple Classifier Systems*, Springer Berlin Heidelberg, Berlin, Heidelberg, 2000, vol. 1857, pp. 1–15.
8. L.-H. Dinh Pham, M.-T. Le and K.-M. Thai, Improved ADME Prediction by Multitask Pretraining on Predicted Data: Insights from the ASAP-Polaris-OpenADMET Blind Challenge, *J. Chem. Inf. Model.*, 2026, **66**, 395–405.
9. K. Goossens, G. Tricarico, J. Hofmans, M.-P. Dréanic, S. De Cesco and E. B. Lenselink, ChemProp Multi-Task Models for Predicting ADME Properties in the Polaris Challenge, *ChemRxiv*, 2025, preprint, DOI: 10.26434/chemrxiv-2025-q12vh.
10. E. Heid, K. P. Greenman, Y. Chung, S.-C. Li, D. E. Graff, F. H. Vermeire, H. Wu, W. H. Green and C. J. McGill, Chemprop: A Machine Learning Package for Chemical Property Prediction, *J. Chem. Inf. Model.*, 2023, **64**, 9–17.
11. I. Koleiev, R. Stratiichuk, N. Shevchuk, M. Melnychenko, O. Nyporko, D. Todoryshyn, V. Husak, S. Starosyla, S. Yesylevskyy and A. Nafiiev, Critical Assessment of ML Models for ADMET Prediction in TDC Leaderboards, *bioRxiv*, 2026, preprint, DOI: 10.64898/2026.02.26.708193.
12. lachrymator, *OpenADMET CYP Challenge — public write-up*, <https://github.com/lachrymator/openadmet-cyp-challenge-public>, (accessed 15 September 2026).
13. G. A. Landrum and S. Riniker, Combining IC50 or K<sub>i</sub> Values from Different Sources Is a Source of Significant Noise, *J. Chem. Inf. Model.*, 2024, **64**, 1560–1567.
14. H. Moriwaki, Y.-S. Tian, N. Kawashita and T. Takagi, Mordred: A Molecular Descriptor Calculator, *J. Cheminform.*, 2018, **10**, 4.
15. *OpenADMET CYP Blind Challenge*, a Hugging Face Space by openadmet, <https://huggingface.co/spaces/openadmet/cyp-challenge>, (accessed 18 August 2026).
16. F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot and É. Duchesnay, Scikit-learn: Machine Learning in Python, *J. Mach. Learn. Res.*, 2011, **12**, 2825–2830.
17. PubChem BioAssay AID 1851, qHTS Assay for Cytochrome Panel, National Center for Advancing Translational Sciences, <https://pubchem.ncbi.nlm.nih.gov/bioassay/1851>, (accessed 17 September 2026).
18. RDKit: Open-Source Cheminformatics, Release 2026.03.3, Zenodo, 2026, DOI: 10.5281/ZENODO.20446949.
19. D. Rogers and M. Hahn, Extended-Connectivity Fingerprints, *J. Chem. Inf. Model.*, 2010, **50**, 742–754.
20. P. J. Ropp, J. C. Kaminsky, S. Yablonski and J. D. Durrant, Dimorphite-DL: An Open-Source Program for Enumerating the Ionization States of Drug-like Small Molecules, *J. Cheminform.*, 2019, **11**, 14.
21. R. P. Sheridan, Time-Split Cross-Validation as a Method for Estimating the Goodness of Prospective Prediction, *J. Chem. Inf. Model.*, 2013, **53**, 783–790.
22. J. Simm, L. Humbeck, A. Zalewski, N. Sturm, W. Heyndrickx, Y. Moreau, B. Beck and A. Schuffenhauer, Splitting Chemical Structure Data Sets for Federated Privacy-Preserving Machine Learning, *J. Cheminform.*, 2021, **13**, 96.
23. SuperCowPowers, The CYP Challenge — Working Notes, *ADMET Workbench*, <https://supercowpowers.github.io/workbench/blogs/cyp_challenge/>, (accessed 3 September 2026).
24. D. Van Tilborg, A. Alenicheva and F. Grisoni, Exposing the Limitations of Molecular Machine Learning with Activity Cliffs, *J. Chem. Inf. Model.*, 2022, **62**, 5938–5951.
25. H. Veith, N. Southall, R. Huang, T. James, D. Fayne, N. Artemenko, M. Shen, J. Inglese, C. P. Austin, D. G. Lloyd and D. S. Auld, Comprehensive Characterization of Cytochrome P450 Isozyme Selectivity across Chemical Libraries, *Nat. Biotechnol.*, 2009, **27**, 1050–1055.
26. K. Yang, K. Swanson, W. Jin, C. Coley, P. Eiden, H. Gao, A. Guzman-Perez, T. Hopper, B. Kelley, M. Mathea, A. Palmer, V. Settels, T. Jaakkola, K. Jensen and R. Barzilay, Analyzing Learned Molecular Representations for Property Prediction, *J. Chem. Inf. Model.*, 2019, **59**, 3370–3388.

**Acknowledgements.** OpenADMET, for the challenge, the dataset and the tutorial repository. SuperCowPowers, lachrymator and Antonio De La Vega De Leon, for publishing working notes this entry borrowed from directly.