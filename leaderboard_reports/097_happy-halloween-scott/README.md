# 🎃 Spooky CYP 🎃

**Direct-inhibition regression and time-dependent inhibition (TDI) for the OpenADMET CYP Inhibition Blind Challenge.**

## The human part 💀

Here are the two final models for the blind challenge: _a simple LightGBM and GNN blend_ for regression and _LightGBM and histogram boosting_ for TDI. More useful than the models themselves are probably the lessons about what absolutely did not work. These are all documented in [EXPERIMENTS](https://github.com/HauserGroup/cyp-challenge-experiments/blob/main/README.md). This is a good starting point for bad ideas. Shoutout to [jeremycheminf](https://github.com/jeremycheminf) and [briford](https://supercowpowers.github.io/workbench/blogs/cyp_challenge/) for doing the same. Noww, before you ask "Jakob, how did you burn that many UCPH leftover tokens and ended up with a triple digit rank?" just remember that the real MA-ST-RAEs were the friends we made all along. 

Thank you to the organisers and apologies for not making my submission Bayesian 👻

## Here starts the robot part ⚰️

Two separate models are frozen in this repository:

| Track | Final model | Inputs | Selected output |
|:--|:--|:--|:--|
| Regression | JEEL + three-model SimpleGINE average, with per-CYP weights and offsets | Molecular structure; pretrained Atom-JEPA features in JEEL | [regression.csv](results/submission/regression.csv) |
| TDI | Fingerprint LightGBM + descriptor/screen histogram boosting, with per-CYP weights and thresholds | Molecular structure; auxiliary models trained on challenge screening responses | [tdi.csv](results/submission/tdi.csv) |

Both use compound-level out-of-fold (OOF) predictions to fit their final decision layer. Regression uses one five-fold partition; TDI uses three five-fold repetitions. Supervised fitting uses challenge data. Regression also uses a frozen pretrained Atom-JEPA encoder; TDI has no pretrained component.

- [Regression recipe](#regression-jeel--simplegine)
- [TDI recipe](#tdi-structure--screen)
- [Recorded results](#recorded-results)
- [Run-the-model commands](#running-the-models)

## Regression: JEEL + SimpleGINE

For each CYP, average the three graph models, combine that average with JEEL, and add a fitted offset:

```text
GINE_c = mean(GINE_seed0,c, GINE_seed1,c, GINE_seed2,c)
prediction_c = w_c * JEEL_c + (1 - w_c) * GINE_c + b_c
```

The two branches use different representations and different treatments of assay uncertainty: JEEL fits point targets with inverse-interval-width weights, while SimpleGINE fits the intervals directly.

### JEEL: a 1,296-feature LightGBM regressor

**JEEL = JEPA + descriptors + ECFP4 + LightGBM.** Concatenate these blocks and fit one regressor per CYP:

| Feature block | Width | Construction |
|:--|--:|:--|
| Atom-JEPA | 256 | Mean of final-layer scalar atom features from the frozen molecular encoder; no CYP fine-tuning |
| Physicochemical descriptors | 16 | Size, lipophilicity, polarity, hydrogen bonding, flexibility, charge and simple substructure summaries |
| ECFP4 | 1,024 | Binary Morgan fingerprint, radius 2, chirality disabled |

Each regressor uses **400 trees, learning rate 0.05, 31 leaves, minimum leaf size 20, feature fraction 0.4, row fraction 0.8 with bagging frequency 1, L1 0.8 and L2 5.0**. Fits use one CPU thread and deterministic column-wise LightGBM.

The objective is LightGBM Huber with `alpha=0.5`. Observed point targets receive weights:

```text
q_i = 1 / clip(upper_i - lower_i, 0.05, 4.0)
weight_i = q_i / mean(q)
```

Normalize weights within each fitting partition and endpoint. This combines robust point-target fitting with greater weight on narrower assay intervals; it is an inverse-width heuristic, not an inverse-variance likelihood.

#### Why Atom-JEPA?

[Atom-JEPA](https://huggingface.co/atom-jepa/atom-jepa) looked cool, and trying its molecular embeddings was an appealing experiment. That is the honest motivation, rather than a claim that CYP inhibition requires this particular representation.

The model would probably be fine either way. Fingerprints, descriptors and the graph branch already provide substantial molecular information. We have not established that Atom-JEPA is essential to the final regression ensemble, or that replacing or omitting it would materially change blind performance. Its inclusion is a modelling choice, not a mechanistic explanation for the predictions.

### SimpleGINE: an interval-trained molecular graph network

**GINE = Graph Isomorphism Network with Edge features.** Atoms have 36 features and bonds have 6. A linear/SiLU encoder maps atoms to 128 dimensions; four residual GINE layers combine neighbouring atom representations with bond features. Scalar attention pooling produces one molecular vector, followed by a linear head with four potency outputs.

| Setting | Value |
|:--|:--|
| Hidden width / message-passing layers | 128 / 4 |
| Dropout | 0.15 |
| Epochs / batch size | 35 / 64 |
| Optimizer | AdamW; learning rate 0.001; weight decay 0.0001 |
| Schedule | Cosine annealing over 35 epochs |
| Stochastic weight averaging | Epoch indices 20–34 |
| Independent initializations | Seeds 0, 1 and 2; predictions averaged |

For prediction `p` and observed interval `[l, u]`, set `d = max(l - p, p - u, 0)`. The loss is:

```text
loss = 0.5 * d² / beta    if d < beta
       d - 0.5 * beta     otherwise
beta = 0.1
```

This gives zero loss inside the interval and a smooth-to-linear penalty outside it. Missing endpoints are masked; batch loss averages observed entries. The graph loss does not use JEEL's inverse-width weights. The full training registry is retained, with unmeasured targets masked.

### Folds and final combination

All regression components share the same five random compound folds. Each endpoint of a molecule stays in the same fold. Query predictions average the five fits for each component, then the three graph initializations, then apply the endpoint blend.

Fit `w_c` and `b_c` jointly on aligned OOF predictions by minimizing total distance outside the observed intervals, subject to `0 <= w_c <= 1`. This is a small linear program, implemented with SciPy HiGHS. The offset is unconstrained; there is no fitted output slope or clipping.

| CYP | JEEL weight | GINE weight | Offset, pIC50 |
|:--|--:|--:|--:|
| 1A2 | 0.408129 | 0.591871 | −0.043611 |
| 2C9 | 0.445807 | 0.554193 | +0.003199 |
| 2D6 | 0.365055 | 0.634945 | −0.081064 |
| 3A4 | 0.513231 | 0.486769 | −0.034836 |

Full-precision values are in [parameters.json](results/submission/parameters.json). Only direct-inhibition potencies and their intervals supervise regression. Screening and TDI tables contribute molecule identities to the shared loader, not auxiliary regression targets.

Source, equations and complete settings: [MODELS.md](MODELS.md).

## TDI: structure + screen

**FinalTDI-StructureScreen** blends two binary tree classifiers, fitted separately for CYP2D6 and CYP3A4. The screen-informed branch gives the classifier a compact summary learned from the primary screen; inference still requires only molecular structure.

### The two components

| Component | Features | Fixed learner |
|:--|:--|:--|
| Structural classifier | 4,263 fingerprint columns + 16 descriptors = **4,279 inputs** | LightGBM: binary log loss, 200 trees, learning rate 0.03, depth 7, 63 leaves, minimum child samples 20, feature fraction 0.4, L1 0, L2 0.5 |
| Descriptor/screen classifier | The same 16 descriptors + one predicted same-CYP screening response = **17 inputs** | Histogram gradient boosting: binary log loss, 150 iterations, learning rate 0.1, 31 leaves, minimum leaf size 20, L2 0, 255 bins |

The structural fingerprints are **MACCS 167 + topological torsion 1,024 + atom pair 1,024 + Avalon 1,024 + RDKit Pattern 1,024**. This matrix contains neither Morgan fingerprints nor Atom-JEPA embeddings. Both classifiers use equal sample weights and disabled early stopping; the structural LightGBM has no row bagging.

The auxiliary screen predictor is a LightGBM regressor from the 16 descriptors to the same CYP's observed single-concentration log2 fold-change response: **200 trees, learning rate 0.05, 15 leaves, minimum child samples 20, feature fraction 0.5, L1/L2 0**, with no row bagging or early stopping. Tree fitting runs with one CPU thread.

The descriptor/screen classifier was called “hurdle” during development, but its TDI head is a direct binary classifier. No potency gate, continuous-assay subtraction or kinetic model is involved.

### Supervision and nested screen features

| Endpoint | Observed binary labels | Positives |
|:--|--:|--:|
| CYP2D6 | 1,497 | 324 |
| CYP3A4 | 3,584 | 764 |

Use every observed binary label, including CYP3A4 negatives without paired continuous measurements. Missing TDI labels remain missing. Screen-only compounds can train the auxiliary regressors, but are never assigned assumed negative TDI labels.

For each outer fold:

1. Withhold all labels and screening responses for its validation compounds.
2. Fit the structural classifier on the outer-training compounds with observed binary labels.
3. Use three inner compound folds to generate cross-fitted screening predictions for the outer-training compounds; train the descriptor/screen classifier on those predicted features.
4. Refit each screen regressor on the full outer-training partition to generate screening features for validation and query compounds.

Training and query rows always use **predicted** screening responses. Do not substitute a training compound's measured screen response or generate screen features globally before splitting.

### Averaging, weights and cutoffs

Use **three repetitions of five compound folds**, with partition seeds 0, 1 and 2. Both endpoints share assignments within a repetition. Identical canonical structures stay together; assignments balance the joint positive, negative and missing-label patterns.

Each labelled compound's OOF branch score averages three held-out predictions. Each query branch score averages all 15 fold models. Blend those probabilities with fixed weights, then apply the fitted cutoff:

| Endpoint | Structural weight | Descriptor/screen weight | Cutoff | Positive calls on the 750-compound panel |
|:--|--:|--:|--:|--:|
| CYP2D6 | 0.95 | 0.05 | 0.277519153001473 | 90 |
| CYP3A4 | 0.60 | 0.40 | 0.3084748264872564 | 234 |

The weights were carried forward from the earlier fingerprint/screen blend and fixed before this run. Each cutoff maximizes MCC on the endpoint's observed OOF scores, with equal scores kept together and exact ties resolved toward the higher threshold. Cutoffs are frozen at inference; there is no positive-call quota or cohort-dependent rank threshold.

The retained ensemble contains 60 TDI classifiers and 30 inference-time screen regressors. Source, seed formulas and artifact provenance: [TDI model details](MODELS.md#final-tdi-classifier), [models/final_tdi.py](models/final_tdi.py).

## Recorded results

### Regression

| Prediction | OOF macro ST-RAE ↓ |
|:--|--:|
| JEEL | 0.7533 |
| SimpleGINE average | 0.7275 |
| Final fitted combination | **0.6986** |

### TDI

| Endpoint | OOF calibration MCC ↑ | OOF AUC ↑ |
|:--|--:|--:|
| CYP2D6 | 0.1657 | 0.5989 |
| CYP3A4 | 0.4063 | 0.8185 |
| Macro average | **0.2860** | 0.7087 |

**The regression weights/offsets and TDI cutoffs were fitted on the same OOF labels used for these final scores.** These are fitted calibration results, not independent assessments of the complete ensembles. Random folds also do not reproduce the blind panel's analog selection. Treat small differences between development recipes accordingly; these tables do not predict blind leaderboard performance.

## Running the models

### Install

```bash
uv sync --locked
```

The default profile has ten direct runtime dependencies and supports regression export and fitted TDI inference. Fresh training uses `--extra train`. Rebuilding Atom-JEPA embeddings uses `--extra embeddings`; the selected embedding cache is included. The `dev` group contains pytest and Ruff.

### Export predictions

The selected CSV and Parquet files for both tracks are already in **[results/submission/](results/submission/)**. To reproduce them in fresh directories:

```bash
uv run python scripts/finalize_regression.py --out-dir results/regression_export
uv run python scripts/predict_tdi.py --out-dir results/tdi_export
```

Both exporters validate the challenge-panel submissions. Fresh exports go to `--out-dir`; they do not replace the selected release files.

The regression exporter uses the retained component predictions and fitted blend parameters. Regression base checkpoints were not retained, so this export is specific to the challenge panel. TDI retains its trained ensemble and can predict another CSV containing `SMILES` and `Molecule_Name`:

```bash
uv run python scripts/predict_tdi.py --input molecules.csv --out-dir results/new_tdi
```

### Retrain

```bash
uv run --extra train python scripts/train_final_regression.py \
  --data-dir data/challenge --out-dir results/new_regression_components

uv run --extra train python scripts/train_final_tdi.py \
  --out-dir results/tdi_runs/retrained
```

Use fresh output directories. The regression command trains components; fitting a new final blend requires aligning their OOF predictions and fitting `FinalRegressionBlend` against the real intervals. The TDI command fits its complete fixed procedure, including cutoffs, and exports its predictions.

To regenerate Atom-JEPA features instead of using the included cache:

```bash
uv run --extra embeddings python scripts/atom_jepa_embeddings.py \
  --out data/new_atom_jepa_embeddings.parquet
```

### Files worth opening

| Path | Contents |
|:--|:--|
| [MODELS.md](MODELS.md) | Regression and TDI recipes, equations, training settings and artifact provenance |
| `data/components/` | Regression component OOF and challenge-panel predictions |
| `data/tdi/` | Fitted TDI ensemble, feature manifest, cutoffs and fold assignments |
| `data/challenge/` | The four supplied challenge CSVs |
| `results/submission/` | Selected regression and TDI submission files; regression blend parameters |
| `results/tdi/` | TDI component scores, summary and integration verification |
