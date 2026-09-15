# openadmet-cyp

Code for the OpenADMET CYP Inhibition Blind Challenge.

**Contributing:** read [RULES.md](RULES.md) before opening a PR — how a PR should
look, what must never be committed, the rule that only Denali submits to the live
HuggingFace challenge, and how to use AI tools. Pick a workstream from
[tasks/MANIFEST.md](tasks/MANIFEST.md), which also holds the shared metric, the
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
evaluate-models            # train/val metrics for the sklearn-style models
generate-results           # predictions for submission
build-family-datasets      # ChEMBL + BindingDB affinity data across the P450 family (Pfam PF00067)
```

## Splits

`data_tools.load.load_data` supports `random`, `scaffold` (Bemis-Murcko) and
`butina` (Tanimoto cluster) train/val splits.
