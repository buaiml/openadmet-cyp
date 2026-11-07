# CYP Inhibition Prediction — Open ADMET Challenge

Model report for the [Open ADMET CYP Challenge](https://openadmet.org/),
predicting direct inhibition pIC50 (regression) and time-dependent inhibition
(TDI classification) across four CYP isoforms: CYP1A2, CYP2C9, CYP2D6, CYP3A4.

**Team:** thinkMolecular

---

## Pipeline Overview

Both tracks share the same features and model families. The difference is in the
prediction target and how predictions are combined.

```
                         ┌──────────────────────────────┐
                         │       Shared Features         │
SMILES ──────────────────┤  ECFP4 (2048-bit)            │
                         │  RDKit 2D descriptors (200)   │
                         │  pKa (acidic/basic)           │
                         └──────────┬───────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼                               ▼
          Chemprop D-MPNN                      LightGBM
          (molecular graph                  (tabular features)
           + descriptors)
                    │                               │
        ┌───────────┴──────────┐          ┌────────┴────────┐
        ▼                      ▼          ▼                 ▼
   Regression              TDI        Regression          TDI
   (pIC50, multi-task)  (classifier   (custom losses)   (classifier
                         + shift-                        + shift-
                         regressor)                      regressor)
        │                      │          │                 │
        └──────┬───────────────┘          └────────┬────────┘
               │                                   │
               └─────────────┬─────────────────────┘
                             ▼
                    Per-isoform ensemble
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
        Regression track            TDI track
        (Caruana selection          (threshold selection)
         + calibration)              
```

---

## 1. Training Data and Label Types

The challenge provides two types of measurements per compound per isoform:

- **Dose-response curves** — yield an exact pIC50 value. These are the primary training signal.
- **Single-concentration screen** — yields only a pass/fail at a fixed concentration. Compounds that do not inhibit at the screen concentration are **censored**: we know only that their true pIC50 is below the screen's detection ceiling (~4.3), not the exact value. These are the majority of the data for some isoforms (e.g., ~65% of CYP2D6).

Both label types are used in training. The key modelling challenge is handling censored labels without either ignoring them (losing information) or pretending they are exact measurements at the ceiling (introducing bias).

---

## 2. Model Families

### 2.1 Chemprop D-MPNN

A bond message-passing neural network that learns directly from the molecular graph.

- Architecture: hidden=300, depth=3, 2-layer feed-forward head
- **Multi-task**: all four isoforms share a single graph encoder trunk. This allows the model to learn shared molecular representations across isoforms.
- **Censored label handling**: uses `bounded-mae` loss — the loss is dropped entirely when a prediction for a censored compound is already on the correct side of the censoring bound (i.e., predicted below the ceiling). This avoids penalizing the model for placing a censored compound anywhere below the detection limit.
- **Auxiliary heads**: during training, additional prediction heads are attached to the shared trunk. These are discarded at inference — only the pIC50 or TDI heads produce the final predictions. The auxiliary targets used:
  - **Screening log2-fold-change**  — single-concentration screen results covering thousands of additional compounds not in the dose-response data
  - **Emax**  — maximum efficacy under both direct-inhibition and TDI conditions
  - **TDI-condition pIC50**  — inhibition potency measured under the pre-incubation condition


### 2.2 LightGBM

Gradient-boosted trees on tabular features (ECFP4 fingerprints, RDKit 2D descriptors, predicted pKa). Multiple feature set combinations were evaluated via out-of-fold cross-validation; the ensemble selects the best-performing members per isoform.

For **regression**, three custom loss functions produce distinct models that contribute diversity to the ensemble:

- **Interval loss**: Directly aligned with the ST-RAE metric. The loss is zero when a prediction falls inside the label's credible interval (exact value for curves, `[floor, ceiling]` for censored compounds), and applies a smooth L1 penalty outside it. This handles censored labels natively — the model is free to predict anything below the ceiling for a censored compound.
- **Tobit loss**: A left-censored normal likelihood for censored compounds. Where interval loss provides no gradient at all below the ceiling (leaving the model unconstrained for ~65% of CYP2D6 data), Tobit provides a decaying gradient that gently pushes predictions downward — encouraging the model to separate censored from measured compounds without collapsing all censored predictions to a single value.
- **L2 (MSE)**: Standard regression loss as a diversity baseline.

For **TDI classification**, standard binary classification with `is_unbalance=True` to handle class imbalance.

---

## 3. Regression Track

Models are trained with 8-fold cross-validation across multiple random seeds (bagging). The best-performing combination of Chemprop and LightGBM models is selected per isoform based on out-of-fold performance, then averaged into an ensemble. A per-isoform affine recalibration is applied as a final postprocessing step.

---

## 4. TDI Classification Track

TDI (time-dependent inhibition) labels are derived from paired measurements — a compound is TDI-positive if its pre-incubated potency exceeds its direct potency. Only compounds with direct-inhibition dose-response curves are used for label construction.


Both Chemprop D-MPNN and LightGBM classifiers are trained alongside a shift-regressor. The final predictions are a blend of classifier and regressor scores from both model families.


---

## Data

Training and test data provided by the [Open ADMET CYP Challenge](https://openadmet.org/) via Hugging Face. No proprietary or external data was used.

