# Does Enzyme Similarity Explain Auxiliary-Head Gain?

`reports/head_search_analysis.md` Section 4 left the bimodal split unexplained:
a solo aux head either buys ~0.15-0.21 macro RMSE or buys nothing, and row
count does not predict which. One candidate explanation is protein
relatedness — an aux isoform that is *like* the scored enzymes should transfer,
one that is not should not. This report tests that directly.

Source: `scripts/protein_similarity.py` (run 2026-09-21, local: 35 s with the
UniProt/AlphaFold cache warm, ~2 min cold).
Sequences from UniProt (reviewed human entries), structures from AlphaFold DB
v6. Gains are the solo-head rows of `results/head_search_results.csv`
(baseline 0.9430 ± 0.0018). Per-gene table: `results/protein_similarity.csv`.

## TL;DR

**No.** Neither sequence identity nor structural similarity to the four scored
isoforms predicts gain, on any slice.

| Slice | Sequence identity | TM-score |
|---|---|---|
| All 20 aux heads, mean similarity to the scored four | rho = **-0.00**, p = 0.995 | rho = **+0.03**, p = 0.885 |
| Family tier only (n = 15) | rho = -0.27, p = 0.321 | rho = -0.24, p = 0.383 |
| Every (aux head, scored task) pair (n = 76) | rho = -0.09, p = 0.423 | rho = +0.01, p = 0.941 |

Where a trend shows at all it points the **wrong way**: inside the family tier
the inert heads are the *more* similar ones (median max identity 48.6% vs.
30.4% for the gainers; median max TM 0.954 vs. 0.846; Mann-Whitney p = 0.15 and
p = 0.11, so not significant either, but certainly not the hypothesised
direction).

![Sequence](similarity_vs_gain_sequence.png)
![Structure](similarity_vs_gain_structure.png)

## Method

- **Sequence.** Global Needleman-Wunsch (BLOSUM62, -11/-1, end gaps free)
  between canonical UniProt sequences; percent identity over the shorter
  sequence. End gaps are unpenalised because several P450s differ mainly by a
  longer N-terminal membrane anchor, which is not a specificity difference.
- **Structure.** TM-align over AlphaFold CA traces, reported as the mean of the
  two length normalisations. P450 lengths differ by at most ~15%, so the two
  normalisations are close and the mean is symmetric.
- Each aux head is measured against all four scored isoforms, then summarised
  as the mean (the metric is a macro average over those same four tasks) and
  the max (a head could help by resembling just one). Self-pairs are dropped
  from the summary: CYP1A2-aux is 100% identical to the CYP1A2 task by
  construction. Those four self-aux heads are drawn as stars and are confounded
  anyway — they carry a supervision channel no external head has.

Observed ranges: identity 19.5-91.4%, TM-score 0.75-1.00. The structural axis
is nearly saturated, as expected inside one fold family, so a flat trend there
is weaker evidence than the flat trend on sequence.

## The counterexamples are the whole story

| Gene | Max identity to a scored isoform | Max TM | Solo gain |
|---|---|---|---|
| CYP3A5 | 84.3% (CYP3A4) | 0.987 | **+0.001** (inert) |
| CYP2C8 | 78.0% (CYP2C9) | 0.978 | **-0.002** (inert) |
| CYP2E1 | 57.1% | 0.969 | +0.001 (inert) |
| CYP1A1 | 72.7% (CYP1A2) | 0.946 | +0.150 |
| CYP11B1 | 24.4% | 0.798 | **+0.163** (best family head) |
| CYP19A1 | 23.1% | 0.819 | +0.154 |

CYP3A5 is the near-twin of a scored isoform and does nothing. CYP11B1 is a
mitochondrial steroid 11-beta-hydroxylase sharing under a quarter of its
sequence with anything scored, and it is the strongest family head in the
search. CYP1A1 shows the hypothesis is not merely inverted — a highly similar
head *can* gain — the axis just does not separate the two groups.

Volume does not rescue the split either: the six family gainers all have
≥ 856 rows, but so do inert TBXAS1 (1628), CYP2C8 (888), CYP4A11 (872) and
CYP4F2 (872).

## Why this is the expected answer, in hindsight

The model has no protein channel. A shared D-MPNN encoder sees SMILES and
nothing else; each head is a readout on top of it, and the protein identity of
a head is just a column name. Enzyme similarity could therefore only act
*indirectly* — similar enzymes get assayed against similar compound libraries,
so a related isoform's molecules would regularise the encoder in a useful
direction. What the data says is that this indirect path is swamped by which
library a head's rows actually came from: CYP2C8 and CYP3A5 are family heads
built from small BindingDB/ChEMBL med-chem series, while CYP11B1 and CYP19A1
are family heads built from different, larger series. Provenance of the ligand
set dominates homology of the protein.

This strengthens rather than weakens the chemical-space hypothesis in
`head_search_analysis.md` Section 4 and the maps in `scripts/chemspace.py`:
having ruled out row count, nearest-neighbour Tanimoto to the eval set, and now
protein relatedness on two independent axes, the remaining explanations are
properties of each head's molecule *distribution* (coverage, diversity,
label consistency), not of its enzyme.

## Caveats

- n = 20 heads (15 external), one split, solo configs only. Underpowered for
  anything but a strong effect; a strong effect is what the hypothesis
  predicted and it is absent.
- Gains below ~0.004 are seed noise, which is the entire inert group — they are
  "no effect", not small effects to be ranked among themselves.
- Similarity is whole-protein. A pocket-only measure (SRS residues, or the
  substrate-recognition sites lining the heme cavity) would be the sharper
  test of the same idea and is not ruled out by this result. That needs the T5
  structures, not just AlphaFold monomers.
