# OpenADMET CYP Challenge: Model Report (KNIMEST)

A short model report for the regression track of the OpenADMET CYP Inhibition Blind Challenge.

My final submission is a two-stage TabICL model: CheMeleon embeddings (PCA, 256 dims) plus log2FC values predicted from the primary screen. I built it as a diagnostic control and chose it after seeing the blind results.

| Model | Role | Blind MA-ST-RAE |
|---|---|---|
| A (in-sample log2FC) | Final submission | 0.5977 |
| T (out-of-fold log2FC) | Main model, chosen on local validation | 0.8111 |

See [MODEL_REPORT.md](MODEL_REPORT.md) for the report and [results/submissions.csv](results/submissions.csv) for all blind results. Details will be published after the challenge closes.

## Acknowledgements

This work was inspired by N283T's publicly shared model report for the OpenADMET PXR Blind Challenge ([report](https://n283t.github.io/openadmet-pxr-model-report/ja/), [repository](https://github.com/N283T/openadmet-pxr-model-report)), in particular by the idea of using predicted primary-screen log2FC as auxiliary features with a tabular foundation model. I also thank OpenADMET for organizing the challenge and for the article "Are tabular foundation models all you need?" ([link](https://openadmet.ghost.io/are-tabular-foundation-models-all-you-need/)), which describes a reproducible version of this approach. Thanks also to the CheMeleon and TabICL teams.

Code and analyses were developed with the assistance of Claude Code (Anthropic); all decisions and the final submission are my own.

**Related write-up.** A personal account (in Japanese) of how this work started: [note article](https://note.com/knimesupportteam/n/nef47ec12f777?hl=en).
