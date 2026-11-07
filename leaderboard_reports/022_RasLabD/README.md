# OpenADMET CYP Inhibition Blind Challenge — direct-inhibition regression

Method, code and results for our entry to the [OpenADMET CYP inhibition blind
challenge](https://huggingface.co/spaces/openadmet/cyp-challenge) (direct-inhibition
regression track: pIC50 for CYP1A2, CYP2C9, CYP2D6 and CYP3A4).

**No proprietary data was used.** Every input is public.

## Result

Blind-set scores, 750 compounds, macro **MA-ST-RAE 0.4204** (lower is better):

| endpoint | ST-RAE | MAE | R² | Spearman ρ | rank |
|---|---|---|---|---|---|
| CYP1A2 | 0.4276 | 0.6959 | 0.6396 | 0.8318 | 17 / 266 |
| CYP2C9 | 0.3435 | 0.4052 | 0.7140 | 0.8493 | 13 / 266 |
| CYP2D6 | 0.5098 | 0.9302 | 0.4403 | 0.5388 | 23 / 266 |
| CYP3A4 | 0.4006 | 0.4945 | 0.7070 | 0.8429 | 35 / 266 |
| **macro** | **0.4204** | 0.6314 | 0.6252 | — | — |

The headline finding is in **[`methodology.pdf`](methodology.pdf)**.

## The short version

A compound only carries a dose-response pIC50 in the training release **if it
inhibited in the primary screen**. For CYP2D6 that selection is perfect — the
measured single-concentration readout separates labelled from unlabelled
compounds with **AUC 1.000**, and not one of the 2,648 compounds whose signal
barely moved has a curve. The model had therefore never seen a non-inhibitor,
while the blind set is roughly half non-inhibitors.

Per-isoform selection strength turns out to be the exact inverse of per-isoform
accuracy, which is the signature of the problem rather than a coincidence:

| isoform | selection AUC | our out-of-fold R² |
|---|---|---|
| CYP2D6 | **1.000** | 0.221 |
| CYP1A2 | 0.896 | 0.343 |
| CYP2C9 | 0.596 | 0.524 |
| CYP3A4 | 0.411 | 0.662 |

The fix gives each screened-but-uncurved compound a pIC50 **window** read off its
own log2 fold-change, rather than a value. Because the window comes from the
compound's own measurement, the same rule places CYP2D6's new rows low (median
centre 2.50) and CYP3A4's high (5.17) — which is why it is safe on an isoform
whose uncurved compounds are the *more* inhibited group, where a blanket
"unlabelled means inactive" collapses performance.

Measured effect on the ability to tell an inhibitor from a non-inhibitor,
out of fold on all 4,375 screened compounds:

| | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 |
|---|---|---|---|---|
| curve labels only | 0.7680 | 0.5479 | 0.7256 | 0.3638 |
| **with window labels** | **0.7997** | 0.5744 | **0.8335** | 0.3582 |

On the blind set this was worth **−0.027 macro ST-RAE**.

**This change looks like a disaster on the conventional validation set** — ST-RAE
measured on the compounds that *have* curves goes from 0.84 to 2.37 — because
that subset is the selected population. A model correctly centred on a population
that is half non-inhibitors must read low against a subset where everything
inhibits. At least one other published entry reports abandoning primary-screen
integration three times on exactly that evidence.

## Layout

```
src/                 the pipeline that produced the submission
  strae.py           exact MA-ST-RAE; unit-tested against the organisers' scorer
  fetch_data.py      re-download the official CSVs
  add_bounds.py      attach credible-interval bounds to the feature table
  build_windows.py   fit the readout -> pIC50 window map, emit window labels
  train_ensemble.py  4-member Chemprop D-MPNN ensemble, 5-fold scaffold CV
  pool_metrics.py    evaluation on the unselected screening pool  <- the key instrument
  compare_runs.py    paired bootstrap over scaffold groups
  placement.py       expected-ST-RAE decision layer
  make_submission.py build the submission file
tests/               55 tests; the metric is bit-exact against the organisers'
predictions/         out-of-fold and blind predictions for both label treatments
data/                official challenge CSVs + the merged feature table
results/             submitted file, blind scores, per-isoform leaderboards, figures
exploration/         measured negative results, with a note on each
figures/             figure and PDF sources
```

Model checkpoints (~350 MB) are not committed. The committed predictions are
enough to reproduce the submission exactly and to re-run every evaluation.

## Reproducing

```bash
pip install -r requirements.txt

# Reproduce the submitted file from the committed predictions (seconds, no GPU)
python src/make_submission.py
# -> results/final_submission_reproduced.csv, identical to results/final_submission.csv

# Check the metric implementation against the organisers' own scorer
pytest tests/ -q

# Re-run the evaluation that drove every decision
python src/pool_metrics.py --run predictions/window_labels
python src/pool_metrics.py --run predictions/curve_labels   # the before picture
```

Training from scratch needs a GPU (~1.5 h on one RTX 2080 Ti for all four
members × 5 folds):

```bash
python src/fetch_data.py
python src/add_bounds.py
python src/build_windows.py
python src/train_ensemble.py --spec all
python src/make_submission.py
```

## Disclosures

- **No proprietary data.** Inputs are the official challenge release, ChEMBL,
  and PubChem AID 1851 (Veith qHTS).
- **Public structures matching a blind-set compound were removed before
  training.** The blind set shares no molecule name and no InChIKey connectivity
  block with any training file.
- The placement step uses **blind-population mean and spread values published by
  the organisers**; it does not use any per-compound information about the test set.
- Credible intervals in the training release are **asymmetric about the fitted
  pIC50 on 71–84% of rows**, so this code uses the published `_conf_low` /
  `_conf_high` columns rather than a symmetric `±1.96·std` approximation. That
  alone is worth 0.016 macro in measurement error.

## Licence

Code: MIT. The official challenge data under `data/official/` is redistributed
under its original Apache-2.0 licence.
