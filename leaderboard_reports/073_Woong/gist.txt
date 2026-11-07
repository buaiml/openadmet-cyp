# OpenADMET CYP Challenge: Woong method report

**Byungwoong Yoo — Independent Researcher**  
Report date: 27 September 2026  
Model version: R1 regression merger and retained TDI classifier, 15 September 2026

## Scope

This report describes the predictions prepared for the 750 challenge compounds and submitted under the display name **Woong** on 15 September 2026. It covers direct-inhibition regression for CYP1A2, CYP2C9, CYP2D6 and CYP3A4, and time-dependent inhibition (TDI) classification for CYP2D6 and CYP3A4. The report accompanies the original prediction files; later experimental models are outside its scope.

## Data and molecular representation

The models use the [official OpenADMET CYP training data](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test), including direct-inhibition measurements and their uncertainty intervals, TDI labels, and auxiliary assay measurements. The assembled production cohort contains 6,146 molecules, with different targets observed for different subsets. Missing targets are masked rather than filled with zero or treated as negative labels. No proprietary assay data or external pretrained model weights were used in this submission pipeline.

Inference requires only molecular identifiers and SMILES. RDKit generates radius-2 Morgan bit fingerprints, log-transformed count fingerprints, feature-based Morgan fingerprints, atom-pair fingerprints, and physicochemical descriptors. The mixed similarity kernel averages generalized Tanimoto kernels over the Morgan-count, feature-count and atom-pair representations. Auxiliary assay values for a query molecule are predicted from its structure; experimental test-compound assay values are not inputs.

## Direct-inhibition regression

Each endpoint combines six branches:

1. An interval-aware LightGBM model using Morgan bits and a compact descriptor set.
2. An interval-aware LightGBM model using count fingerprints and descriptors.
3. A mixed-Tanimoto interval support-vector regressor, with C = 1.
4. A masked multi-assay multilayer perceptron.
5. A single-concentration (SC) prior: mixed-Tanimoto kernel ridge regression, with alpha = 0.3, predicts SC signals; a spline-based mapping relates these cross-fitted predictions to direct-inhibition intervals.
6. A multi-assay model, named `dose8`, using 768 training-derived Nyström landmarks and RDKit descriptors. It has a shared rank-8 component, task-specific residuals, and a monotone link between predicted direct potency and SC response.

The final prediction is a weighted sum plus an intercept. Weights are nonnegative and sum to one. They are fitted with a smoothed interval-distance loss, smoothing parameter 0.035, and L2 regularization. This objective penalizes deviations beyond the training uncertainty bounds while regularizing the combination.

The frozen production parameters are shown below, rounded to six decimal places. Columns follow the branch order above; near-zero stored weights are displayed as zero.

| Endpoint | 1 | 2 | 3 | 4 | 5 | 6 | Intercept |
|---|---:|---:|---:|---:|---:|---:|---:|
| CYP1A2 | 0.118133 | 0.186089 | 0.013411 | 0.120709 | 0.300121 | 0.261536 | 0.030867 |
| CYP2C9 | 0.140443 | 0 | 0 | 0.190678 | 0.311973 | 0.356906 | 0.012487 |
| CYP2D6 | 0 | 0.166999 | 0.163590 | 0.143675 | 0.403031 | 0.122706 | -0.003758 |
| CYP3A4 | 0.101480 | 0.267356 | 0 | 0.269817 | 0.177647 | 0.183700 | 0.039218 |

## TDI classification

The retained classifier combines five branches: two LightGBM classifiers using different molecular representations, a kernel ridge model, a radius-2 fingerprint support-vector classifier, and the auxiliary multi-assay neural model. Each branch receives a monotone Platt calibration fitted to training out-of-fold predictions. Calibrated outputs are combined with nonnegative, sum-to-one weights by regularized log-loss minimization. An endpoint-specific threshold is selected on training out-of-fold predictions using smoothed Matthews correlation coefficient (MCC) values to favor a stable cutoff. Predictions at or above that threshold are reported as TDI-positive.

The R1 regression update retained this earlier TDI policy and its prediction file unchanged. The experimental R1 context-based TDI challenger was not used.

## Model selection and production fitting

Development used 5,378 molecules. The primary evaluation consisted of three repeated four-fold cluster splits, with a separate three-fold scaffold split used as a stress test. Within each outer training partition, three-fold inner predictions were used to fit combination weights, calibration and classification thresholds. Targets and auxiliary measurements belonging to an excluded molecule were excluded together from the corresponding model fit.

The recorded mean regression MA-ST-RAE across the three cluster repeats was **0.649026789697**; the retained TDI policy's mean macro MCC was **0.241152169**. These are historical development results, not official test scores. Model families were explored and selected on the same development data, so the nested fitting of weights does not make the full research process an independent confirmatory evaluation.

For production, four previously fitted full-cohort regression branches were reused and the SC-prior and `dose8` branches were refitted on the 6,146-molecule cohort. The stacking policy selected from development out-of-fold predictions was copied unchanged. The production model therefore includes molecules excluded from particular development fits and must not be evaluated on those internal partitions as though they were unseen.

## Reproducibility and limitations

The archived execution environment was CPU-based: Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, RDKit 2025.09.4, LightGBM 4.6.0 and PyTorch 2.10.0+cpu. The preserved package contains training and inference code, fitted models, frozen policies and validation records. The package is retained locally.

The archived validation records report successful CSV checks and reproduction of the regression output from the saved production model in the original runtime. Report preparation recovered the original files and checked their hashes, schema and identifiers; it did not rerun all historical training experiments. The original prediction-file SHA-256 identifiers are:

| Track | SHA-256 |
|---|---|
| Regression | `bb8cd816aee35e22de6430d4fd5f6289cd66d58a8b04da0ce3697d87c9a33cce` |
| TDI | `e8cc64853b456c39f96fa95bc85f9884a8eecb63dc458296dc89c2315065ca77` |

The challenge's hit-selected training measurements and analogue-expanded test set may differ from the internal validation distribution. Development performance does not establish test-set accuracy or clinical utility. No test labels were used to fit the reported models.

AI tools assisted with code development, experiment execution and preparation of this report.

## References

- [OpenADMET CYP Challenge and submission requirements](https://huggingface.co/spaces/openadmet/cyp-challenge)
- [Official training and blinded-test dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test)
- [Official CYP Challenge tutorial and validation guidance](https://github.com/OpenADMET/CYP-Challenge-Tutorial)
