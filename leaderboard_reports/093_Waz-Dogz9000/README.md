# OpenADMET CYP Inhibition Blind Challenge — method report

Entry for the [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge),
Direct Inhibition (regression) track. HF username: `Waz-Dogz9000`.

This repository is the method report for the entry described under *Status*. It contains the **preregistration, the code,
the controls, the negative results and an adversarial review by a separate agent**. The preregistration
was frozen before any data was touched (by this repository's own recorded timestamps); the referee review began after the modelling lanes had handed back but ran against a tree that was still
changing (see *What the review covers* below for what it did and did not see).

## Status (2026-09-25)

- **Submission.** As of 25 September 2026, the public interim leaderboard records Waz-Dogz9000 as
  submitted on 18 September 2026 at 20:37 UTC with MA-ST-RAE 0.9437, linking this repository as its report.
  This independently corroborates the submission, but the interim leaderboard is not the preregistered final verdict. That
  the file submitted was the frozen `L3-6` CSV, and that the platform acknowledged receipt, are the
  author's own record and are not independently verified here; this repository holds no platform receipt.
  The hashes under *Predictions* commit the frozen local files; they do not by themselves show what the
  platform received.
- **No final result exists.** The verdict is decided by the FINAL leaderboard (published from 2026-11-04
  per the preregistration), under the conditions frozen in `prereg/PREREGISTRATION-CH01.md` §4, not by any
  interim figure. The bar is the lowest final MA-ST-RAE among four entries the preregistration treats as
  the organiser baseline set. That grouping is an inference the preregistration itself records as
  unresolved: the four share a 2026-08-13 04:21 UTC timestamp that predates the 2026-08-17 launch. At the
  2026-09-18 half-set snapshot the lowest of the four was TabICL-baseline, 0.6755. The conditions are:
  - **PASS**: final MA-ST-RAE strictly below the bar **and** final rank in the top half of ranked entrants.
  - **PASS_WITH_NARROWING**: strictly below the bar but ranked below the median.
  - **NEGATIVE_RESULT**: at or above the bar. This includes beating only the LGBM-baseline (0.8929 at the
    same 2026-09-18 half-set snapshot).

  No verdict is drawn here from the interim row.
- **Older status lines in this repository are freeze-time records, not current status.**
  `prereg/CHALLENGE-01-REPORT.md` ("NOT SUBMITTED"), `results/SERVE-PATH.md` and
  `results/SUBMISSION-MANIFEST.json` describe the state at the 2026-09-18 freeze, before submission.
  The first two are hash-committed in `prereg/HASHES-CH01.txt` and are left unedited.

## What was frozen for submission

One model: `L3-6`, a single LightGBM over long-format `(compound, endpoint)` rows.

- **Features (2273):** ECFP4 2048-bit ∪ 217 RDKit descriptors ∪ 4 out-of-fold predicted
  primary-screen `log2fc` values ∪ 4 endpoint one-hot.
- **Why long format:** the training inhibition matrix is only 33.3% filled (6525/19620 cells; just 41
  of 4905 compounds carry all four endpoints). An unmeasured cell contributes no row, so the mask is
  structural rather than imputed, and the four endpoints share one model.
- **Row weights:** `1/(1 + credible-interval width)`.
- Internal 5-fold scaffold-split macro RAE **0.8042 ± 0.0092**.

## Honest framing of that number

Internally there are no credible intervals, so ST-RAE **degenerates exactly to plain RAE**. The number
above is therefore *not* the leaderboard quantity and **must not be compared to any MA-ST-RAE on the
leaderboard**. RAE also normalises by the evaluated set's own dispersion — see "The risk we expect to
lose to" below.

## Method development, in order

| stage | result |
|---|---|
| B0 mean predictor | 1.0008 ± 0.0043 |
| B2a ECFP4 + LightGBM | 0.9169 ± 0.0176 |
| B2b RDKit descriptors + LightGBM | 0.8539 ± 0.0165 |
| Lane L3 — multi-task / cross-endpoint, 6 configs | best **0.8042 ± 0.0092** ← selected |
| Lane L4 — pretrained representations, 6 configs | best 0.8580 ± 0.0057 |

Budget was frozen in advance at 6 configurations per lane. All 12 ran; none was dropped.

## Negative results, kept

1. **Pretrained molecular representations did not beat hand-crafted descriptors.** Best embedding arm
   (MoLFormer-XL ∪ RDKit descriptors + LightGBM) 0.8580 vs 0.8539 for descriptors alone — short by less
   than its own fold sd, so *indistinguishable from, and certainly not better than*, B2b. Not inert
   either: every non-MLP embedding arm beat ECFP4 and the mean predictor. **The head mattered more
   than the checkpoint** — swapping LightGBM for RidgeCV on the same embedding moved MoLFormer
   0.9113 → 0.8647, while changing checkpoint at fixed head moved ≤ 0.006. A single
   embedding+LightGBM number would understate pretrained representations by ~0.05.
2. **A preregistered prediction of ours was wrong.** We predicted scaffold-split validation would be
   *materially harder* than random. Across all 15 scored arms the mean gap is +0.0004 and no arm's gap
   reaches one fold sd. Cause: 4363 of 4905 compounds (89%) are the sole member of their Bemis–Murcko
   scaffold, so scaffold grouping barely constrains anything. **Consequence we accept: no internal
   number here is evidence of generalisation to dissimilar chemistry**, and 88.5% of blind-set
   scaffolds are unseen in training. See `results/FALSIFIER-ADJUDICATION.md`.
3. **CYP2D6 resists everything.** No arm below 0.956; B2b itself 1.006 (worse than predicting the
   mean). Multi-task sharing moved it below 1.0 for the first time, which was the opposite of what the
   lane author predicted — 2D6 correlates ~0 with the other isoforms, and what transferred was
   structure–activity regularity, not the sibling value.

## Controls

| control | result | what it establishes |
|---|---|---|
| metric sanity | perfect 0.0000 / mean 1.0000 / corrupted(+3.0) 4.2887 | the scorer can actually go red |
| mean predictor | 1.0008 | harness wired correctly |
| train↔test overlap | 0 by name, canonical SMILES, InChIKey, connectivity layer | no contamination |
| shuffled labels | 1.0457 on the winner | split and scorer not broken |
| submission integrity | 25 assertions, 0 failed; organisers' own validator returns `(True, [])` | file is valid. Run by the author after the referee pass; not seen by the referee |

**We also record what a control does *not* establish.** The shuffled-label control permutes the target
only and does not regenerate the stacked out-of-fold features, so it is near-powerless against leakage
through that block, and we corrected our own record to say so. What actually establishes the stack is
clean is an exact mask audit (0 rows whose feature-producer saw them, all 5 folds) plus two canaries:
a noise-target stack scores 0.8208 and a shuffled-screen stack 0.8204, both *above* the no-stack
ablation at 0.8150, while the real stack reaches 0.8042.

## The risk we expect to lose to

RAE divides by the evaluated set's own dispersion. Holding this model completely fixed and only
narrowing the evaluated population moves its own score **0.8029 → 1.2266 (top-50% most active) →
1.8667 (top-25%)**. The blind test set is deliberately hit-enriched analog expansion around confirmed
hits, so its spread is plausibly narrower than training. No control above can see this, because they
all evaluate on the training set's own spread.

Our predictions are also compressed (pred sd ÷ train sd = 0.42–0.80), and the serve-time stacked
features sit 0.25–0.55 fit-sd off centre for the blind 750 (worst CYP1A2, −0.548) on a scale
1.12–1.24× different from fit time — a property of those 750 compounds, not of the construction.

**We therefore expect this entry to score worse than our internal number suggests.** That is written
here before any score exists.

## What the review covers, and what it does not

`referee/REFEREE-REPORT-CH01.md` is an adversarial review by a separate agent working from
`referee/REFEREE-BRIEF-CH01.md`, whose stated objective was to find the reason the submission should
not go out. Its brief told it to attack and not to be told which candidate was preferred. It is not a human, external or
organisational review, and this repository cannot show how separate its context was. It returned PASS_WITH_NARROWING.

**What the referee checked itself.** It re-applied the frozen selection rule across all 15 scored arms
and reached the same configuration; it re-executed the exact fold-mask audit and the two canaries
against the stacked out-of-fold features and could not break them; it recomputed the
scaffold-versus-random statistics.

**What it did not cover.**

- **The tree was not frozen.** Files under `results/` changed while the referee was reading them
  (finding F-13). The referee re-extracted every number it cites from the then-current files, so its
  numbers match the tree *as it stood during the pass*. This is not a review of one frozen snapshot,
  and it should not be read as final validation of the tree as published.
- **No submission artifact existed at review time** (F-8). The serve path, preregistered control C5
  (25 assertions), the run of the organisers' validator, and the submission file's hash were all
  produced afterwards by the author. The referee never saw them. That is **author-run closure
  evidence**, not independent evidence.
- **The two required corrections were executed by the author** (a record that falsely claimed a
  quarantined file was never opened, and a control whose stated power exceeded its actual power).
  This repository holds no record of the referee re-checking the corrected versions. Originals are retained, nothing deleted.

## Predictions

The prediction file is **deliberately not published until after the 2026-11-03 deadline**, so that it
cannot be copied into another entry while the challenge is live. Its hashes are published now as a
commitment to the frozen files, so that they cannot be altered unnoticed (the author's record says the CSV is the file that was submitted; see *Status*):

```
dd943ca3af2766106da457d384d0d01f4eeb24b3ddfc097ce3bec30b9ef34da7  submission_L3-6.csv
c200bb5ea5199b4ac2b8782cca38518e3fc89351eab5da736135ebb871cf0955  submission_L3-6.parquet
```

## Reproducing

Training data: [`openadmet/cyp-challenge-train-test`](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test)
at revision `3ac9c5dbb83eec5780ec7fa511908698cfe1396d` (Apache-2.0); hashes in `prereg/DATA-HASHES.txt`.
Scoring uses the organisers' own `rae_soft_threshold_absolute_error`, imported unmodified from the
[tutorial repo](https://github.com/OpenADMET/CYP-Challenge-Tutorial) at commit `858ae63c`.
Environment: `prereg/ENVIRONMENT.txt` (Python 3.12, rdkit 2026.3.6, lightgbm 4.7.0). Folds are frozen
at seed 20260918. Order: `code/audit_leakage.py` → `audit_data.py` → `splits.py` → `control_c4.py` →
`baselines.py` → `lane_l3.py` / `lane_l4.py` → `serve_l3_6.py`.

Predictions were byte-identical across two full runs on one machine; cross-platform determinism is not
claimed.

## Disclosures

- **No proprietary data.** Only the organisers' released dataset and public libraries/checkpoints.
- Pretrained checkpoints evaluated (and not used in the selected model):
  `DeepChem/ChemBERTa-77M-MLM` @ `ed8a5374f2024ec8da53760af91a33fb8f6a15ff`,
  `ibm/MoLFormer-XL-both-10pct` @ `361063d0ad524ef77cf39b08469f6be770dc550f`.
- No blind-set label was looked up in any external database, and the organisers' tutorial output files
  (which contain a third party's predictions over all 750 blind compounds) were quarantined and used by
  no modelling lane.
- Work was carried out with AI agents under a frozen protocol; the preregistration, worker brief and
  referee brief are all included so the process can be judged, not just the result.

Author: Warren Smith.
