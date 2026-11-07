# OpenADMET CYP regression: second controlled D6 location experiment

This is a prospective, explicitly adaptive public-feedback calibration experiment. It is not yet accepted or scored, not a validated predictive improvement, and not an estimate of missing weak CYP2D6 labels or hidden population moments. Earlier methods files in this gist remain unchanged.

## Parent and exact operation

Generation parent: our accepted October 31 16:54:04.522566 UTC file, SHA256 `96cef9cc9f9cbb84cd782acc61fa1da88e6b89db04e9e08f1c5709dae1d9f17e`. Its learned models, public external data, training-only calibration and fixed ensemble are documented in `big_union_methods_20261031.md`. The distinct qualified nested-B file and any newer experimental model are not incorporated here.

For every one of the 750 official test rows:

`new D6 pIC50 = binary64(float(original accepted96 serialized D6) - 1.0)`

Scale is exactly 1. Altered values use 17 significant digits and CSV CRLF. IDs, raw SMILES, row/header order and the other three endpoint field strings remain exact. There is no clipping, row-specific map, new model inference/training, source-label acquisition or change to CYP1A2/CYP2C9/CYP3A4. Observed predicted rank/tie groups remain exact in the constructed file; this was checked rather than assumed from positive-affine mathematics.

Prospective candidate SHA256: `96b8606287ef6cbb1478a352ecc40cefee530d39afb92e00b7657ab47bdd26d6`.

Frozen protocol SHA256: `5ffcfebbd16e0b267529dc8a084533ffa2471f6c11ad6f14acfb839995e8ab57`.

## Observed motivation, not an extrapolated score claim

Our previous, separate D6-minus0.5 experiment was accepted November 1 04:56:04.198113 UTC, SHA256 `2051dfb208c8fe04925ba31c54f9b6d0d8739b255f2b96e1bf27b667ca43dfca`. Its posted macro ST-RAE was .6778 and D6 .9707, versus .7943 and 1.4369 for the original parent. Other endpoint scores remained C1 .7671, C9 .4571 and A4 .5161. That actual result motivates one further half-unit location question; it does not establish the optimal shift or guarantee continued improvement. The new file is generated directly from the original parent, not by changing its learned model or using competitor predictions/constants.

Primary interpretation compares the new official D6 ST-RAE with immediately preceding .9707 and original 1.4369; macro and all endpoint controls are reported. MAE/R² are secondary and cannot rescue primary harm. Predicted rank/tie and public rank-metric stability are checked. No reconstructed labels, hidden biological means/variances, claimed missing-tail identification or future score/rank forecast is used.

The published endpoint scorer is separable, but errors and public adaptation are not statistically independent. Same frontend evidence and unchanged controls cannot prove unchanged backend scored rows/bands/bootstrap. Live-subset gains may not transfer to final evaluation. All earlier failed donor/model studies remain failed under their own frozen criteria.

## Operations and risk

The sole Big submission owner will require the existing anonymous account/settings, exact candidate/methods/schema hashes, local and independent file checks, and fresh official cooldown/account/rules checks immediately before any upload. Earliest planned attempt is November 1 16:58 UTC, later than the preceding actual acceptance plus 12 hours. This description is not an acceptance receipt and does not alter prior files.

The new shift can worsen performance. Restoring a prior file requires another permitted window and successful acceptance. No automatic later ladder, scale change, additional endpoint map or subsequent upload follows from this note. The final available slot is reserved for selecting/restoring a known qualified combination rather than blind exploration. A public methods description does not claim the full codebase is open; anonymity, proprietary-data No and open-code No remain unchanged.
