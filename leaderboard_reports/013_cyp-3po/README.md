---
license: mit
tags:
  - chemistry
  - admet
  - cytochrome-p450
  - openadmet
---

# cyp-3po — method report

OpenADMET CYP Inhibition Challenge, Regression Prediction track.

Submission macro **MA-ST-RAE 0.4048**.

| endpoint | ST-RAE | Spearman ρ |
|---|---:|---:|
| CYP1A2 | 0.4339 | 0.820 |
| CYP2C9 | 0.3469 | 0.862 |
| CYP2D6 | 0.4666 | 0.543 |
| CYP3A4 | 0.3718 | 0.863 |

## Summary

An ensemble of 76 independently trained MolE graph-transformer checkpoints
across six architecturally heterogeneous arms, spliced per endpoint, with
location and spread calibrated against the leaderboard's own returns.

**The full method report — data, architecture, ensemble, calibration, negative
results and limitations — is being finalised and will replace this page before
the end of the challenge.**
