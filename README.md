# openadmet-cyp

Code for the OpenADMET CYP Inhibition Blind Challenge.

**Contributing:** read [RULES.md](RULES.md) before opening a PR — how a PR should
look, what must never be committed, the rule that only Denali submits to the live
HuggingFace challenge, and how to use AI tools. Pick a workstream from
[tasks/round1/MANIFEST.md](tasks/round1/MANIFEST.md), which also holds the shared metric, the
pinned split and the join keys.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## Data

```bash
download-data              # challenge splits (inhibition, TDI, Emax, single-concentration) from HuggingFace
build-pubchem-aux          # PubChem AID 1851 (Veith CYP qHTS panel) -> per-isoform auxiliary heads
build-union                # join challenge + auxiliary sources into one wide multi-task table
check-overlap              # structural duplication within, and overlap between, any datasets
build-family-datasets      # ChEMBL + BindingDB affinity data across the P450 family (Pfam PF00067)
build-family-heads         # pivot the family datasets into one `{gene}_pIC50_{source}` head per protein per source
add-split-column           # write the pinned train/val/test `split` column into the union table
```

### External data as auxiliary heads

External CYP measurements are **never pooled into the scored pIC50 columns**.
Each source x isoform x readout becomes its own column, and therefore its own
head at training time, so no cross-assay calibration is required: a shared
D-MPNN encoder sees every molecule and every label while each head keeps its
own scale. Chemprop masks missing targets in the loss, so the sparse block
structure of a multi-source union costs nothing.

This replaces a pretrain-then-finetune split. One model is trained jointly on
all heads at once, which avoids both the pooling problem and catastrophic
forgetting during fine-tuning.

**PubChem AID 1851** contributes the signal a potency-only ingest throws away.
The panel screens five isoforms (1A2, 2C9, 2C19, 2D6, 3A4) in 15-point
dose-response. Of its 85,715 compound x isoform rows only 40,684 have a fitted
`Potency`, but **all** of them have `Max_Response` — including the 42,460 rows
the screen called Inactive, which have no potency at all. Since ChEMBL-style
potency data is truncated from below (no curve, no row), those negatives are
what a model trained on fitted pIC50 never sees. CYP2C19 is carried as an
extra head even though the challenge does not score it.

`Max_Response` is a PERCENT change in luminescence and is mostly negative,
because inhibiting pro-luciferin conversion *decreases* signal. It is emitted
negated, as `{ISO}_pctinhib_aid1851`, so that larger means more inhibition —
matching the direction of pIC50.

### Structural identity

Sources spell the same molecule differently (salt vs free base, explicit vs
implicit stereo), so raw SMILES comparison under-reports overlap. Every merge
and every overlap measurement keys on the **InChIKey connectivity block**
(first 14 characters), which is invariant to stereochemistry, protonation and
isotope labelling. See `data_tools.standardize`.

## Training

```bash
# joint multi-task: pass every head at once
train-chemprop --data-path data/union_train.csv --target-columns <all heads>
```

`build-union` prints the exact command with the head list filled in. All
`chemprop train` flags (`--batch-size`, `--accelerator`, `--split-type`,
`--task-weights`, ...) pass through.

```bash
head-search                # search auxiliary-head subsets on the pinned split (single / greedy / ablation / exhaustive)
evaluate-models            # train/val metrics for the sklearn-style models
generate-results           # predictions for submission
```

## Representations

`evaluate-models` and `generate-results` take `--input` with one or more of the
names below; several names are hstacked into one feature matrix.

| `--input` | Class | Dim | What it is |
|---|---|---:|---|
| `rdkit` | `RDKitDescriptors` | 217 | RDKit 2D physicochemical descriptors |
| `chemeleon` | `CheMeleon` | 2048 | fixed pretrained CheMeleon embedding; checkpoint downloads on first use |
| `morgan` | `MorganFingerprint` | 2048 | Morgan/ECFP4 bits, radius 2 |
| `count_morgan` | `CountMorganFingerprint` | 2048 | same generator, substructure counts |
| `maccs` | `MACCS` | 167 | MACCS structural keys |
| `mordred` | `MordredDescriptors` | 1613 | Mordred 2D descriptors (`mordredcommunity`); missing values encoded as 0 |

```bash
evaluate-models --input morgan                          # one representation
evaluate-models --input count_morgan rdkit --split butina --seed 0
evaluate-models --input all                             # every representation on its own, one table
```

Unparsable SMILES give an all-NaN row, which `evaluate-models` drops. Features
are cached per representation in `data/features/{name}.npy`, keyed on SMILES, so
each molecule is featurized once across every split and seed. To add one,
subclass `Representation` in `src/representations/` and append it to
`INPUT_REGISTRY` in `src/data_tools/inputs.py`; see
`reports/t2_solo_representations.md` for how the current six compare.

## Models

Registered models accept precomputed feature matrices through `fit(X, y)` and
`predict(X)`. The evaluator applies `models.utils.drop_nan_rows` before fitting
and scoring. The three ensemble wrappers do not add filtering, feature generation,
scaling or PCA. Select models with `evaluate-models --models <name> ...`.

| Name | Estimator | Default settings |
|---|---|---|
| `decision_tree` | sklearn DecisionTreeRegressor | depth=5, min split=2, min leaf=1, seed=42 |
| `ridge` | sklearn StandardScaler + Ridge | alpha=1.0 |
| `random_forest` | sklearn RandomForestRegressor | 200 trees, unlimited depth, min leaf=1, all features |
| `lightgbm` | LightGBM LGBMRegressor | 200 trees, rate=0.05, 31 leaves, unlimited depth, min child=20 |
| `xgboost` | XGBoost XGBRegressor | 200 trees, rate=0.05, depth=6, hist trees |
| `tabpfn` | TabPFN TabPFNRegressor | n_estimators=auto, temperature=auto, device=auto, seed=42 |

The three ensemble wrappers default to CPU, `random_state=42` and `n_jobs=1`.
Random Forest uses `criterion=squared_error`, `bootstrap=True` and `max_features=1.0`.
LightGBM uses `boosting_type=gbdt`, `objective=regression`, `device_type=cpu`
and `verbosity=-1`. Its unlimited depth is `max_depth=-1`.
XGBoost uses `objective=reg:squarederror`, `booster=gbtree`, `tree_method=hist`,
`device=cpu`, `subsample=1.0` and `colsample_bytree=1.0`.

Constructors expose tree count, complexity controls, seed and worker count;
the boosting wrappers also expose learning rate. The evaluator uses constructor
defaults. These are baseline settings, not tuned CYP results.

TabPFN does no gradient training: `fit` stores the training rows and the
transformer conditions on them at predict time. Its constructor exposes
`n_estimators`, `softmax_temperature`, `ignore_pretraining_limits`, `device`,
`model_path` and `random_state`; it defaults to `device="auto"` rather than CPU,
since the others are tree models and this one is a transformer. Set
`ignore_pretraining_limits=True` to feed it representations wider than the
pretraining feature limit.

`tabpfn` is an optional dependency, registered only when the package is
installed, so a missing install drops it from `--models all` instead of
breaking the CLI. It is untuned and has not been run on CYP data yet.

## Splits

`data_tools.load.load_data` supports `random`, `scaffold` (Bemis-Murcko) and
`butina` (Tanimoto cluster) train/val splits; `evaluate-models` exposes them as
`--split` and `--seed`.

Chemprop multi-task runs use a different mechanism: `add-split-column` writes
one pinned `split` column into `data/union_train.csv`, and `head-search`
refuses to run without it.
