# OpenADMET CYP direct-inhibition regression — round 2

Author/account: `cylovelife`; public display alias: `aidd-agent`. Independent personal
project. Open-source code: <https://github.com/cylovelife/aidd-agent>. No proprietary
data, external training datasets, pretrained weights, GPU or paid compute were used.

## Data and problem

Use only the official direct-inhibition training CSV at Hugging Face dataset
`openadmet/cyp-challenge-train-test`, revision
`d7eb3c973dca01aad8e33c91c29e5123fd925d8f`; source and hashes are recorded in
`docs/data-manifest.json`. There are 4,905 training molecules, with 1,412/1,285/1,493/
2,335 measured labels for CYP1A2/2C9/2D6/3A4. Only 41 molecules have all four labels.
Missing labels are excluded endpoint by endpoint, never replaced with zero.
The official 750-molecule test file supplies identities and SMILES only. No test
labels or leaderboard feedback enter fitting, candidate selection or ensemble weights.

The TDI, Emax and single-concentration training tables at the same revision were
audited but not used. TDI contains no additional direct-inhibition pIC50 labels;
experimental auxiliary outcomes are not available features for the blinded test set.

## Features and candidates

Label-free features: RDKit Morgan radius-2, 2,048-bit fingerprints; log1p transformed
Morgan counts with 2,048 bins; and the 217 registered RDKit 2D descriptors, including
physicochemical, topology and functional-group descriptors. Exact names and RDKit
version are saved in each run. No assay, measured cross-target activity, molecule
identifier or label confidence bound is used as a prediction feature. No learned
preprocessor is fitted globally; molecular descriptors and fingerprints are deterministic.
Submitted SMILES remain byte-for-byte identical to the official input strings.

Ten candidates were fixed before training: a median reference, the original
Ridge(alpha=10), two bit-Tanimoto SVRs(C=1/10), a log-count Tanimoto SVR(C=10),
a centered log-count kernel ridge(alpha=1), a log-count interval SVR(C=5),
ExtraTrees on counts plus descriptors, HistGradientBoosting on descriptors, and
a fixed 0.5/0.25/0.25 blend of count SVR/ExtraTrees/HistGradientBoosting.
Ordinary SVRs use epsilon=0.1. The interval SVR uses epsilon=0 and duplicates each
training point at its measured lower/upper credible bounds with half weight each;
the resulting absolute-error objective equals the distance outside the training
credible interval plus a constant, with the ordinary SVR regularizer.
It is an evaluated candidate, not necessarily a component of the final submission.

ExtraTrees: 240 trees, min_samples_leaf=3, max_features=0.5, random_state=42,
n_jobs=2. HistGradientBoosting: absolute_error, 200 iterations, max_leaf_nodes=15,
learning_rate=0.05, l2_regularization=2, early_stopping=False, random_state=42.
Tanimoto uses x·z/(||x||²+||z||²−x·z); this is binary Tanimoto for bits and a
continuous extension for log-count vectors. Kernel ridge centers only fit labels.

## Validation, selection and limitations

Three fixed 80/20 scaffold group holdouts use seeds 42/43/44. Bemis–Murcko scaffolds
define cyclic groups and canonical identity defines acyclic groups. Identical molecules
remain together; no scaffold group crosses a fit/validation boundary. The original
seed=42 split is deliberately retained for direct comparison. All three splits are
development data, with overlapping training/validation membership between repetitions;
they are not an independent sealed test or three independent tests.

The exact pinned official ST-RAE function computes endpoint validation scores using
the provided credible bounds. Select the nonconstant candidate with minimum mean
ST-RAE across the three splits independently for each endpoint. Fixed blend weights
are never fitted or changed. The upload gate requires at least 5% improvement over
the old Ridge macro score, improvement over the median reference, and all submission
validators. Selection among multiple candidates can make the development estimate
optimistic; it is not a forecast of the official blinded score or final rank.

Final choices: CYP1A2 — descriptor HistGradientBoosting; CYP2C9 — fixed blend;
CYP2D6 — bit-Tanimoto SVR(C=1); CYP3A4 — fixed blend. Each selected model or blend
member is refitted once using all measured training labels for that endpoint.

The training set was enriched for primary-screen hits, whereas the test set is an
analog expansion measured densely across all four CYPs. This sampling difference,
missing training endpoints, unseen chemical structures and possible activity cliffs
limit generalization. CYP2D6 remains weakest in local development. No top-three
performance claim is made.

## Reproduction and disclosure

```bash
uv sync --frozen
uv run aidd fetch
uv run aidd optimize --output-dir state/runs/round2-reproduction
uv run aidd validate state/runs/round2-reproduction/regression.csv
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/aidd_agent
```

Each fresh run saves the frozen protocol, exact source/dependency/data hashes,
group partitions, candidate scores, validation predictions, selected models,
final 750×6 predictions, elapsed time and process peak RSS under ignored `state/`.
The budget is 1,800 seconds and 4 GiB RSS per run with at most two computation
threads. No large-model training or automated leaderboard optimization is performed.

Submission validation checks exact identities/original SMILES, all finite target
predictions, per-column sample standard deviation at least 0.01 and the official
validator. Platform receipt and score are separate evidence, published only after
they exist. The first official score was 0.9284982814; it does not score this method.
See `docs/run-2026-10-29.md` and the second-round submission record for actual
validation and platform outcomes.
