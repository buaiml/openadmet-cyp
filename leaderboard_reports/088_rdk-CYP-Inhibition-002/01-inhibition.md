# Direct inhibition — molecular representation ensemble

> [!NOTE]
> **TL;DR**
>
> **Molecular structure → four inhibition-potency predictions**
>
> - **Targets:** CYP1A2, CYP2C9, CYP2D6 and CYP3A4 pIC50 for **750 compounds**.
> - **Model:** a weighted blend of random forests, descriptor boosting, CheMeleon neural models and two ridge-regression branches.
> - **Neural branches:** three descriptor-adapted CheMeleon models and one interval-L1 CheMeleon model.
> - **Ridge inputs:** Morgan counts, 100 graph descriptors and Snakes in one branch; a frozen 2,304-dimensional ShEPhERD2 representation in the other.

## Inputs and training data

The model predicts **pIC50 = −log₁₀(IC50 in mol/L)** from molecular structures. Training uses 4,834 compounds from the [official challenge release](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test/tree/d7eb3c973dca01aad8e33c91c29e5123fd925d8f), with 5,230 available measurements:

| Endpoint | Training measurements |
|---|---:|
| CYP1A2 | 1,129 |
| CYP2C9 | 1,056 |
| CYP2D6 | 1,209 |
| CYP3A4 | 1,836 |

Reserved confirmation compounds and 107 additional compounds sharing accepted chemical cores with them are excluded. Each endpoint uses only its observed labels. Test compounds supply structures and identifiers only; neither preprocessing nor encoder adaptation is fitted on them.

## Model and representations

Predictions are weighted averages in pIC50 units. These are the **final weights**, including all nested averaging:

| Component | CYP1A2 / CYP2C9 / CYP3A4 | CYP2D6 |
|---|---:|---:|
| Random forest | 21.09375% | 56.25% |
| Three descriptor-adapted CheMeleon models, averaged equally | 21.09375% | — |
| Mordred histogram gradient boosting | 7.03125% | — |
| Morgan/RDKit histogram gradient boosting | 7.03125% | — |
| Interval-L1 CheMeleon | 18.75% | 18.75% |
| Morgan counts + graph descriptors + Snakes ridge | 18.75% | 18.75% |
| ShEPhERD2 ridge | 6.25% | 6.25% |

**Tree models.** The forest uses 2,048 Feature-Morgan bits, 12 compact RDKit descriptors and four ADME prediction features. The two boosting models use 1,613 Mordred descriptors and 2,048 chiral Morgan bits plus 12 RDKit descriptors, respectively.

**Neural models.** CheMeleon is a pretrained molecular graph encoder. Three models learn training-set molecular descriptors before supervised inhibition fine-tuning. A separate model uses *interval-L1*: zero loss inside the supplied response confidence interval and absolute distance outside it, averaged over observed endpoints.

**Graph descriptors.** These 100 features describe rotatable bonds and their connectivity, fragments formed by cutting those bonds, ring sizes and fused/spiro connections, branching, graph distances, symmetry and stereochemistry. They are calculated from atoms and bonds without generating 3D conformers. Two features are published Chan LR1/LR2 conformational-entropy estimates at a 298.15 K reference condition; these and the other state-count proxies are not measured solution or binding entropies.

**Snakes** is a separate in-house molecular descriptor, used here as fixed ridge-regression inputs. **ShEPhERD2** provides a frozen 2,304-dimensional 3D molecular representation with xTB charges. Each ridge branch has four endpoint-specific models. Imputation and scaling use training data only, with missingness indicators retained.

<details>
<summary><strong>Technical settings</strong> — learners and feature preparation</summary>

- **Forest:** 100 trees, minimum leaf size 2, square-root feature sampling; median imputation, missingness indicators and StandardScaler. Endpoint seeds are 20261102–20261105.
- **Boosting:** absolute-error loss, 300 iterations, learning rate 0.05, 15 leaves, L2 regularization 10, minimum leaf size 20, no early stopping, seed 20261103. Entirely missing training columns are zeroed; other missing values remain native NaNs.
- **Descriptor-adapted CheMeleon:** three seeds, 20261103–20261105; nine descriptor-head warmup epochs, ten joint descriptor-adaptation epochs, then nine supervised frozen-encoder epochs and three fine-tuning epochs. The interval model uses nine supervised frozen-encoder epochs followed by three interval-L1 fine-tuning epochs. Readouts are 2,049→64→4 with ReLU.
- **Descriptor ridge:** 2,048 radius-2 chiral Morgan counts, five 20-feature graph blocks and Snakes. The separate 12-feature size-control block is excluded. Counts undergo `log1p`; continuous features are standardized. Both use training-only median imputation. Ridge alpha is 10, LSQR tolerance 0.0001, maximum 2,000 iterations; empty features are retained.
- **ShEPhERD2 ridge:** the same ridge settings, using the append readout with xTB charges and sigma 0.1. One ETKDGv3 conformer attempt per structure, up to 500 UFF optimization iterations, base seed 17 and surface seed 20261102. Unsupported structures and extraction failures remain missing.

</details>

## Development evaluation

Five fixed folds group compounds by inferred chemical cores. Held-out compounds do not enter fold-specific preprocessing or model fitting. **ST-RAE** measures error outside reported response intervals relative to a mean-prediction baseline; lower is better. Folds and endpoints receive equal weight.

The **tree/neural-only control** excludes both ridge branches and renormalizes the remaining weights to sum to one. All component weights and ridge settings are fixed for this comparison.

| Endpoint | Tree/neural-only control | Full model |
|---|---:|---:|
| CYP1A2 | 0.801068 | 0.809805 |
| CYP2C9 | 0.634501 | 0.618193 |
| CYP2D6 | 0.883095 | 0.879384 |
| CYP3A4 | 0.516897 | 0.494444 |
| **Macro ST-RAE ↓** | **0.708890** | **0.700456** |

The macro gain is 0.008434, positive in four of five folds. Its conditional 95% interval is [0.001241, 0.014972], or [−0.002740, +0.018115] after adjustment for 15 comparisons. Removing Snakes from the descriptor branch gives 0.703301; the incremental Snakes gain is 0.002844, with adjusted interval [−0.002172, +0.007788]. CYP1A2 error increases by 0.008737.

Intervals resample chemical allocation groups within folds with predictions fixed. They omit retraining uncertainty and adaptive development-set reuse. Neither overall superiority nor incremental Snakes benefit is established by these results.

## Inference, limitations and availability

Final models use all admitted training data, with unavailable endpoint labels masked. Complete ShEPhERD2 vectors cover 735/750 test compounds; the fingerprint/graph/Snakes matrix is fully finite for 737/750. Training-derived imputation provides predictions for all compounds. No clipping or output adjustment is applied.

**Proprietary-data use is declared:** the forest's four ADME features have private-teacher training ancestry. Upstream encoder training overlap with challenge compounds is unresolved. Code, fitted models and provenance records remain private.

No official score is reported here.

**Acknowledgment.** Reports by adlvdl and HauserGroup helped motivate the interval-aware loss used in the CheMeleon branch.

---

**Accepted:** 3 November 2026, 23:56:16 UTC.

**SHA-256 (`predictions.csv`):** `a7928c76a9898aa294591788fc68ec5be2d97a67729f1916ac19740aab3786e8`.
