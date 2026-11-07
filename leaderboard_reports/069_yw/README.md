---
tags:
- chemistry
- openadmet
- cyp-inhibition
---
# yw — OpenADMET CYP direct-inhibition method report

Updated: 2026-11-03T16:48:24.354536+00:00

## Submission

This report describes the final public challenge submission under the anonymous leaderboard
alias **yw**. It uses no proprietary training data and no external assay data. No code is published.

Prediction file SHA-256: `80695a19012518a3fbfd5ec3086dd112701b307b27abacba41b9472ced05ed0f`.

The file combines, per enzyme, prediction columns from earlier model runs (no joint refit and no new
training). Each enzyme's column was chosen by comparing the public live-leaderboard ST-RAE of earlier
submissions that contained it. The live leaderboard covers only part of the blinded series, so these
choices are tuned to that subset.

## Live leaderboard result

Overall MA-ST-RAE of this file: observed **0.7404**.

| Enzyme | Source run | Live ST-RAE of the column |
| --- | --- | ---: |
| CYP1A2 | `20261031T032022-cyp1a2-nested-support-conditioned-family-gate` | 0.772 |
| CYP2C9 | `20261029T175417-cyp1a2-cyp3a4-fixed-descriptor-graph-ensemble` | 0.5217 |
| CYP2D6 | `20261017T004159-cyp2d6-assay-tagged-nonhit-calibration` | 1.172 |
| CYP3A4 | `20261029T175417-cyp1a2-cyp3a4-fixed-descriptor-graph-ensemble` | 0.4958 |

## Data and method of each source run

Each description is quoted from that run's own record. Only the enzymes listed for a run are taken
from it, so statements about its other enzymes do not apply to this file.

### `20261031T032022-cyp1a2-nested-support-conditioned-family-gate` (used for CYP1A2)

Model revision: `c232bdc94f87c336f3253b2e51c52bae87474d32`.

Method: CYP1A2 p=L+clip(0.5+Z beta,0,1)*(J-L), fixed eight-column support gate. Deterministic interval-error LP plus .01 L1 beta penalty, training weights constrained [0,1]. Full real-support gate is sole candidate; no control selection. Three other endpoints exactly F.

Features: intercept; m_z; d_z; abs_d_z; similarity_z; scaffold; m_z*d_z; similarity_z*d_z

Data sources: Challenge direct confidence intervals; Archived L tree and J descriptor-graph nested predictions; separately tagged TDI-condition screening supervision only in base models

Validation: Controller canonical random/scaffold five-fold, testlike holdout and archived Butina cluster five-fold. Gate training uses only each population inner OOF archive and permitted fitting labels; blinded fit uses full inner OOF. All 85 inner fits and 17 refits audited before gate training; no global outer OOF input.

### `20261029T175417-cyp1a2-cyp3a4-fixed-descriptor-graph-ensemble` (used for CYP2C9, CYP3A4)

Model revision: `911812692f61783085b34085e0fb498683c28dcc`.

Method: Fixed (L + GJ)/2 for CYP1A2 and CYP3A4; C CYP2C9/CYP2D6 copied exactly. GJ is the archived equal three-seed epoch-100 graph. No training, calibration, learned weights, checkpoint selection, or control substitution.

Features: Chemprop 2.3.1 depth-3 256-dimensional BondMessagePassing, default atom/bond features, fixed half-mean/half-componentwise-max pooling; X_d concatenation to 268-input two-layer eight-output RegressionFFN, hidden256, dropout0.1, no batch normalization; ['MolWt', 'MolLogP', 'TPSA', 'MolMR', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'RingCount', 'NumAromaticRings', 'FractionCSP3', 'HeavyAtomCount', 'NumHeteroatoms']; Historical LightGBM: chiral Morgan counts, RDKit descriptors, nested predicted TDI-condition screening responses

Data sources: Challenge direct-inhibition intervals; Separate TDI-condition single-concentration screen tasks; Hash-pinned completed L, C, and J archives; no new assay data

Validation: Unchanged canonical random/scaffold five-fold and testlike holdout; C fixed Butina cluster diagnostic. Every graph fit/held index and fitting-only descriptor/target normalization verified; all 153 saved inference arrays exactly reproduce seed-average exports. Historical stacking control flow and saved cluster screen memberships audited. TDI/Emax labels were not used in these components. All four predictions finite for every held row. No held assay enters a fitted component.

### `20261017T004159-cyp2d6-assay-tagged-nonhit-calibration` (used for CYP2D6)

Model revision: `06e1089342f4ec8297830b103935684e73668812`.

Method: Interval-bound LightGBM champion with a CYP2D6 assay-tagged screen nonhit probability correction

Features: Morgan radius-2 count fingerprints with chirality, RDKit descriptors, a CYP2D6 screen nonhit classifier and separate hit/nonhit L1 response models

Data sources: OpenADMET direct-inhibition training data (champion); OpenADMET single-concentration log2 fold change under TDI condition; Local screen-multitask-features champion predictions

Validation: Controller molecule-grouped random, scaffold, test-like direct splits and five-fold screen splits; all screen data from held-out canonical molecules excluded

Compounds were separated before fitting any model component or generating out-of-fold assay
features. Direct, TDI, and screening readouts were treated as distinct assay observations. CYP2D6
direct measurements are selected from screening hits, limiting what random-fold direct validation
establishes.
