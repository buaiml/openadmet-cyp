# Does Enzyme Similarity Explain Auxiliary-Head Gain?

Some solo aux heads measurably improve macro RMSE and others do nothing, and
row count does not predict which (`reports/head_search_analysis.md`). One
candidate explanation is protein relatedness — an aux isoform that is *like* the scored enzymes should transfer,
one that is not should not. This report tests that directly.

Source: `scripts/protein_similarity.py` (first run 2026-09-21, re-run
2026-09-22; local, ~50 s with the UniProt/AlphaFold/RCSB cache warm, ~2 min
cold). Sequences from UniProt (reviewed human entries), models from AlphaFold
DB v6, heme donors from RCSB. Gains are the solo-head entries of
`results/head_search.json` from the fixed head-search run (baseline
0.7557 ± 0.0036). Per-gene table: `results/protein_similarity.csv`.

> **Re-run 2026-09-22 against the fixed head-search.** The first version of
> this report regressed against gains from the run whose baseline died at
> epoch 1 (`reports/head_search_baseline_bug.md`). The similarity measurements
> are unchanged; every gain and every correlation below is new. The answer is
> still no, but the trends are no longer flat or inverted: all four axes now
> lean weakly positive, and none reaches significance.

## TL;DR

**No, not detectably.** No similarity axis predicts gain at this sample
size: not whole-protein sequence identity, not fold similarity, and not the
active site, which is the version of the question that actually had a
mechanism behind it. Every axis leans the expected way (more similar, slightly
more gain), with |ρ| around 0.3 and p = 0.14-0.20.

Spearman rho of solo-head macro gain against each similarity axis
(mean over the four scored isoforms, self-pairs excluded):

| Slice | Sequence identity | TM-score | **Pocket identity** | **Pocket CA RMSD** |
|---|---|---|---|---|
| All 20 aux heads | +0.30 (p = 0.195) | +0.34 (p = 0.139) | **+0.32 (p = 0.164)** | **-0.34 (p = 0.148)** |
| Family tier only (n = 15) | +0.24 (p = 0.383) | +0.37 (p = 0.173) | **+0.29 (p = 0.294)** | **-0.45 (p = 0.089)** |
| Every (aux head, scored task) pair (n = 76) | +0.05 (p = 0.668) | +0.16 (p = 0.160) | **+0.18 (p = 0.116)** | **-0.15 (p = 0.196)** |

Pocket RMSD is lower-is-more-similar, so its negative ρ points the same way as
the others. Pair-level values are computed from the rounded
`results/protein_similarity.csv` and can differ from the figure annotations in
the second decimal.

The per-head trend is consistent in sign across all four axes but weak, and it
mostly disappears at the pair level (ρ ≤ 0.18). A head does not
preferentially help the scored isoform it resembles most. The strongest single
number, family-only pocket shape (ρ = -0.45, p = 0.089), is one of twelve tests
in the table and does not survive any multiple-comparison correction.

![Pocket sequence](similarity_vs_gain_pocket_sequence.png)
![Pocket shape](similarity_vs_gain_pocket_shape.png)
![Sequence](similarity_vs_gain_sequence.png)
![Structure](similarity_vs_gain_structure.png)

## The pocket axes, and why they are the ones that matter

Whole-chain TM-score is the weakest of the four tests by construction: every
P450 shares the triangular-prism fold, so the observed range is 0.75-1.00 and
almost all of it is the fold, not the chemistry. Whole-protein identity has the
same problem in milder form — it is dominated by the conserved core, the
I-helix and the cysteine ligand loop. What decides whether two P450s turn over
the same molecules is the ~20 residues lining the cavity above the heme.

So the pocket is measured directly:

1. A ligand-bound crystal structure of each scored isoform (CYP1A2 2HI4,
   CYP2C9 1R9O, CYP2D6 4WNV, CYP3A4 1TQN) is superposed on that isoform's
   AlphaFold model and its heme carried into the model frame. Fit RMSDs:
   0.39, 2.11, 0.63, 1.52 A.
2. Pocket = every residue with a heavy atom within 6 A of the heme **on the
   distal side** of the porphyrin plane (plane from the four pyrrole
   nitrogens, oriented away from the axial cysteine). The distal filter is the
   point: the proximal shell is the Cys ligand loop, invariant across the whole
   superfamily, and including it would flatten the axis by construction.
3. Residue correspondence comes from the TM-align structural alignment, not a
   sequence alignment — at 20-30% global identity a sequence alignment of the
   cavity is not trustworthy. Coverage is >= 92% of pocket positions for every
   pair.

The definition recovers the textbook active sites: CYP3A4 gives R105, S119,
I120, F302, A305, T309, I369, A370, R372, L373, E374; CYP2C9 gives R97, F114,
T301/T302, L362, S365, L366. Pocket sizes are 19-24 residues.

This axis has the dynamic range the global ones lack — pocket identity spans
16-89% across non-self pairs, versus 19.5-91.4% for whole-protein identity but
with the conserved bulk stripped out. It is still not significant against
gain, and the result holds at an 8 A cutoff (36-39 residue pockets: pocket
identity ρ = +0.25 over all 20, +0.20 family-only; pocket RMSD ρ = -0.39 and
-0.44, p ≈ 0.09 both), so it is not an artefact of where the boundary was
drawn.

## Method

- **Sequence.** Global Needleman-Wunsch (BLOSUM62, -11/-1, end gaps free)
  between canonical UniProt sequences; percent identity over the shorter
  sequence. End gaps are unpenalised because several P450s differ mainly by a
  longer N-terminal membrane anchor, which is not a specificity difference.
- **Structure.** TM-align over AlphaFold CA traces, reported as the mean of the
  two length normalisations. P450 lengths differ by at most ~15%, so the two
  normalisations are close and the mean is symmetric.
- **Pocket sequence.** Percent identity over the scored isoform's pocket
  positions, read off the TM-align correspondence (see the section above).
- **Pocket shape.** CA RMSD over those same matched positions after a Kabsch
  fit on the pocket pairs alone, so it is cavity geometry rather than whether
  the two folds superpose — they always do. Lower is more similar.
- Each aux head is measured against all four scored isoforms, then summarised
  as the mean (the metric is a macro average over those same four tasks) and
  the max (a head could help by resembling just one). Self-pairs are dropped
  from the summary: CYP1A2-aux is 100% identical to the CYP1A2 task by
  construction. Those four self-aux heads are drawn as stars and are confounded
  anyway — they carry a supervision channel no external head has.

Observed ranges over non-self pairs: identity 19.5-91.4%, TM-score 0.75-1.00,
pocket identity 16-89%, pocket RMSD 0.17-2.0 A. The whole-chain structural axis
is nearly saturated, as expected inside one fold family, so a flat trend there
carries less weight than the flat trends on the other three.

## The counterexamples are the whole story

| Gene | Max identity to a scored isoform | Max TM | Max pocket identity | Best pocket RMSD | Solo gain |
|---|---|---|---|---|---|
Solo gains are `baseline - solo` from the fixed run. "Above noise" means more
than 2 combined seed SDs (baseline and solo) above zero.

| Gene | Max identity to a scored isoform | Max TM | Max pocket identity | Best pocket RMSD | Solo gain |
|---|---|---|---|---|---|
| CYP2C8 | 78.0% (CYP2C9) | 0.978 | 73.7% (CYP2C9) | 0.39 A | **+0.019** (best family head, above noise) |
| CYP3A5 | 84.3% (CYP3A4) | 0.987 | 90.5% (CYP3A4) | 0.18 A | +0.007 (not above noise) |
| CYP1A1 | 72.7% (CYP1A2) | 0.946 | 79.2% (CYP1A2) | 0.17 A | +0.005 (not above noise) |
| CYP2B6 | 48.6% | 0.960 | 73.7% | 0.46 A | +0.005 (not above noise) |
| CYP2E1 | 57.1% | 0.969 | 84.2% | 0.41 A | +0.003 (not above noise) |
| CYP19A1 | 23.1% | 0.819 | 28.6% | 1.12 A | **+0.014** (above noise) |
| CYP11B1 | 24.4% | 0.798 | 39.1% | 0.98 A | **+0.012** (above noise) |

The top of the table fits the hypothesis: CYP2C8, the closest non-scored
relative of CYP2C9, is the best family head. The rest of it does not. CYP3A5
shares 90% of CYP3A4's active-site residues, superposes on that cavity to
0.18 A, and is not distinguishable from noise. CYP1A1 is almost as close to
CYP1A2 and does no better. Meanwhile CYP19A1 (aromatase) and CYP11B1 (a
mitochondrial steroid 11-beta-hydroxylase) share under a quarter of their
sequence with anything scored, and both clear the noise bar. High similarity is
neither necessary nor sufficient.

Volume does not separate them either: CYP2C8 (888 molecules) beats CYP11B2
(1,659, +0.001) and TBXAS1 (907, -0.001).

## Why this is the expected answer, in hindsight

The model has no protein channel. A shared D-MPNN encoder sees SMILES and
nothing else; each head is a readout on top of it, and the protein identity of
a head is just a column name. Enzyme similarity could therefore only act
*indirectly* — similar enzymes get assayed against similar compound libraries,
so a related isoform's molecules would regularise the encoder in a useful
direction. The weak positive lean is consistent with that indirect path
existing, but it is small next to the differences between the ligand sets
themselves: CYP3A5 and CYP2C8 are both close relatives of a scored enzyme,
and only one of them helps.

This points back to the ligand side. In the fixed run, nearest-neighbour
ECFP4 Tanimoto from the eval set to a head's training molecules does
correlate with gain (`results/aux_similarity.csv`: ρ = +0.49, p = 0.027 over
all 20 heads; +0.60, p = 0.018 over the 15 family heads), which is stronger
than any protein axis here. The explanation is more likely a property of each
head's molecule *distribution* (coverage, proximity, label consistency) than
of its enzyme. `scripts/chemspace.py` draws those distributions.

## Caveats

- n = 20 heads (15 external), one split, solo configs only. Underpowered for
  anything but a strong effect; a strong effect is what the hypothesis
  predicted and it is absent.
- Every gain here is under 0.04, and 13 of 20 are within 2 combined seed SDs
  of zero. Those are "no effect", not small effects to be ranked among
  themselves. The SDs are also optimistic: head-search varies only
  `--pytorch-seed`, so all three replicates see the same batch order.
- Pockets are defined on AlphaFold apo models with a heme borrowed from a
  crystal structure, and P450 cavities are famously plastic — CYP3A4's expands
  by hundreds of cubic angstroms on binding. A static first-shell definition
  cannot see that. The T5 co-folded complexes would allow a ligand-aware
  version (cavity volume, per-ligand contact fingerprints) rather than a
  residue list, and only that would fully close the question.
- The 1R9O -> AlphaFold fit for CYP2C9 is 2.11 A, noticeably worse than the
  other three (0.39-1.52 A); the heme placement is still consistent with the
  known 2C9 active site, but this pocket is the least precisely located of the
  four.
