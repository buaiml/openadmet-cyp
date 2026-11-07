# OpenADMET CYP Challenge: preliminary model report

**Status:** preliminary. This report will be expanded after the challenge closes.

## Tracks

- **Regression:** direct-inhibition pIC50 for CYP1A2, CYP2C9, CYP2D6 and CYP3A4.
- **Classification:** time-dependent inhibition (TDI) calls for CYP2D6 and CYP3A4.

## Approach in brief

- **Regression:** for each endpoint, an ensemble of tabular models (gradient-boosted trees and
  tabular foundation models) over molecular descriptors, fingerprints and frozen pretrained
  molecular representations. Ensemble members are chosen with series-grouped cross-validation, and
  the final predictions are rescaled to a fixed mean and spread.
- **Classification:** model scores are rank-combined and converted to calls.

## Data

The challenge training data were used. Individual ensemble members may also use publicly released
pretrained models and public datasets. No proprietary data were used.

## Code

Not yet public. Release will be considered after the challenge closes.
