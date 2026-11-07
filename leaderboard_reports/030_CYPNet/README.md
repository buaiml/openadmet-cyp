# OpenADMET CYP Challenge

Chemprop models for predicting direct-inhibition pIC50 for CYP1A2, CYP2C9, CYP2D6 and CYP3A4.

**Best reported public leaderboard score: 0.4635 MA-ST-RAE** (lower is better).

The corresponding four-member submission is included as [`submissions/best_submission.csv`](submissions/best_submission.csv).

## Approach

Four Chemprop models use complementary assay heads:

| Member | Targets | Full-data epochs |
|---|---|---:|
| `aux` | Four direct-inhibition endpoints and four single-concentration log2 fold changes | 30 |
| `union` | The eight challenge heads above, five ChEMBL pIC50 heads and five PubChem/Veith response heads | 30 |
| `2d6_single` | CYP2D6 direct-inhibition pIC50 | 20 |
| `2d6_isoform` | CYP2D6 direct inhibition, log2 fold change, TDI, Emax, ChEMBL and Veith | 12 |

CYP1A2, CYP2C9 and CYP3A4 average the `aux` and `union` predictions. CYP2D6 averages all four members. Missing assay values remain missing; TDI and public activity labels are not substituted for direct-inhibition measurements.

The networks use MAE loss, message-passing width 700 and depth 5, dropout 0.1, two feed-forward layers of width 300, batch size 64 and seed 42. Primary-task weights are proportional to inverse label counts; auxiliary heads receive weight 0.3 relative to the primary average. Each full-data model uses every row with at least one observed head.

The submitted predictions also use fixed historical target moments:

| CYP | Mean pIC50 | Sample standard deviation |
|---|---:|---:|
| CYP1A2 | 4.412 | 1.169709 |
| CYP2C9 | 4.830 | 0.922638 |
| CYP2D6 | 3.570 | 0.900000 |
| CYP3A4 | 4.880 | 1.085472 |

These values are in [`config/placement.json`](config/placement.json). Placement rescales each prediction column to those moments. It was leaderboard-informed.

## Setup

Python 3.12 is recommended. From the repository directory:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The cached-prediction build and tests run on CPU. Training requires a CUDA-enabled PyTorch installation; follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) for your system. Training refuses to start without CUDA.

## Rebuild the submission without training

The original member prediction tables and their checksums are included, so the ensemble can be rebuilt without checkpoints, raw training data or a GPU:

```bash
python train.py build --predictions-dir artifacts/member_predictions --output-dir outputs/rebuilt
python -m pytest -q
```

This writes:

- `outputs/rebuilt/cyp_full_four_raw.csv`
- `outputs/rebuilt/cyp_full_four_placed.csv`
- `outputs/rebuilt/build_four.json`

The tests compare the rebuilt placed predictions with the archived submission to a tolerance of 1e-12 pIC50. The archived CSV itself is byte-preserved. CSV formatting and floating-point differences between library versions can prevent a regenerated file from having the same checksum.

Existing output files are never overwritten. Nothing is uploaded automatically.

## Prepare data and train

Place the seven source CSVs listed in [`data/README.md`](data/README.md) under `data/raw/`, then run:

```bash
python prepare.py prepare
python train.py fit aux
python train.py fit union
python train.py fit 2d6_single
python train.py fit 2d6_isoform
python train.py build --output-dir outputs/retrained
```

Prepared tables go to `data/prepared/`; training runs go to `runs/full/`. Model weights and raw training data are not bundled with this repository. Retraining is computationally expensive and is not guaranteed to reproduce the original weights or leaderboard result bit-for-bit.

Data preparation joins external assays by InChIKey connectivity, clips Veith responses to [-150, 50] and removes external structures overlapping the blinded test set. For the pinned input files, the union contains 31,669 rows and 161,395 observed targets. The CYP2D6 specialist table contains 32,876 rows, including 1,240 additional TDI-only challenge IDs. Prepared-data counts and source checksums are recorded in [`artifacts/data_manifest.json`](artifacts/data_manifest.json).

Optional scaffold-fold experiments use:

```bash
python prepare.py fit 2d6_single --fold 0 --epochs 400
```

Repeat with the desired model and fold (0–4). Fold boundaries depend on each model's observed target rows. Validation selects checkpoints, so these results are not an untouched independent test.

## Interpretation and limitations

- This is a four-full-fit implementation, not the original Workbench five-fold endpoint ensemble.
- The specialist epoch counts were selected using earlier scaffold validation. The 30-epoch auxiliary and union fits were compute-limited choices, not validated optima.
- There is no complete, matched OOF ensemble for this submitted model. In-sample full-fit predictions must not be presented as OOF validation.

## Layout

```text
prepare.py                       Dataset joins and optional scaffold CV
train.py                         Full-data training and ensemble assembly
utils.py                         Submission validation and fixed placement
config/placement.json            Historical target moments
artifacts/blind_compounds.csv    Original 750 test IDs and structures
artifacts/member_predictions/    Cached predictions and member metadata
artifacts/submission_manifest.json
submissions/best_submission.csv  Original scored candidate
tests/                           CPU regression tests
```
