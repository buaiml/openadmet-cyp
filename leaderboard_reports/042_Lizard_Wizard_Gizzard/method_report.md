# Method report — OpenADMET CYP Inhibition Blind Challenge

*Outward-facing document, written for the organisers and for anyone who wants to reuse the negative
results. English by intent; the rest of `docs/` is the team's internal Russian-language artefact.*

*This header claimed to be "linked from the leaderboard's Model Report field" until 21 September. It
was not: the board pointed at `METHOD.md` in every snapshot committed up to that date. It points at
THIS file in the snapshot of 11 October, so the field was corrected at a later upload and the
sentence above is kept only to date the error. `METHOD.md` remains public and carries the same
disclosure, corrected in the same commit as this one.*

Code: this repository, public. Proprietary training data: none. External data: the organisers' four
dose-response tables and the single-concentration screen, **and one third-party pretrained encoder**
— `chemprop_medium.pt`, committed as a plain file in another open entrant's public repository
(`jeremycheminf/openadmet_scripts`) and pretrained on CYP and ADME activity. It is used frozen, never
fine-tuned: a ridge head on its 600 output dimensions enters the regression ensemble as a sixth
member (§1). A second use of the same encoder in the TDI rule is measured and built but is NOT on
the board at the time of writing — the flag that enables it is off by default, and this sentence will
be wrong the moment that changes, so it is dated: 15 October.

*This line read "the organisers' own files only" from 26 September, the day that member shipped,
until 15 October. The member and this sentence landed in the same commit and only one of them was
updated. We have not checked whether the 750 blind compounds appear in that checkpoint's pretraining
data, and we will not, because establishing it would mean looking the blind set up in external
databases.*

---

## 0. Against the Innovation in ML criteria, in your words

The award names *architectural novelty, creative use of available data, novel training or
uncertainty quantification strategies, and simple yet effective ideas*. Answered in that order, one
measured number each, and the first answer is a concession.

**Architectural novelty — no.** A gradient-boosting ensemble and a small two-head neural trunk. We
do not compete on this axis and will not dress it up. What follows is where we think the entry
earns its place.

**Creative use of available data — the credible band is a training target, not only a scoring
device.** ST-RAE's per-compound loss is `L(p) = |p − clip(p, lo, hi)|`; absolute error against
`clip(p, lo, hi)` majorises it and is tight at the optimum, so the organisers' published bands can
be optimised against directly rather than merely scored against. That is the "dead zone", and it is
worth **+0.0197 of rank, sign 16/16 per-enzyme cells** (items 164, 204, 205, 213). Separately, the
TDI pre-incubation arm carries its own bands and so is usable as a second regression target
(item 243).

**Novel uncertainty quantification — in the units the metric pays in.** Not an interval around the
prediction, which ST-RAE does not charge for, but the per-compound probability of scoring exactly
zero, `P(hit_i) = P(lo_i ≤ ŷ_i ≤ hi_i)`. Section 4 gives it in full, including the part most entries
would omit: it beats a base-rate constant on **one enzyme of four** (CYP3A4, AUC 0.750), and the
mechanism says which enzyme before the model is fitted.

**Simple yet effective — a closed form for the scoring function itself.** The credible band is a
deterministic function of the label: isotonic regression from the label alone onto band width
recovers it at **R² 0.928–0.970** on all four enzymes (§9). The rank-11 open-code entrant names this
exact gap in their own public write-up, listing the ST-RAE optimum as *"known by sampling placements
against the board, not derived"* and noting that a credible-interval width model would give it
directly (SuperCowPowers, Workbench CYP challenge blog).
Two further results about the competition's own measurement, not about molecules: ST-RAE and R²
do not want the same prediction spread (the ST-RAE-optimal is 0.69–0.88 of the R²-optimal, on every
enzyme and every feature set, item 313), and the leaderboard's published bootstrap means carry an
exact bias `−(1−R²)(1+κ)/n` with `κ = Var((y−ȳ)²)/S⁴`, validated bit-for-bit against
`evaluation.utils.bootstrap_sampling` (item 312).

**Exploration over exploitation** is the award's stated purpose, so the catalogue of what we closed
is offered as part of the entry rather than hidden: §6, and 301 numbered items in
[`verify/README.md`](../verify/README.md), each with the measurement that shut it.

**And one practice we would most want borrowed, in §5: a failed prediction built in advance to work
as a measuring instrument.** Where a decision turns on a quantity you cannot measure without
spending something irreversible — for us, the width of the blind credible bands, readable only by
using up one of a rate-limited sequence of submissions — commit the prediction's SENSITIVITY MAP to
that quantity alongside the prediction. Then the outcome identifies the quantity whether the
prediction holds or not. Ours did not hold: the committed band was [0.6146, 0.6623], the board
returned 0.6805, and the same map read the unmeasurable multiplier back as ≈0.78×. The map also
chose the placement, on worst case across it rather than on expectation, and the realised value
landed inside exactly the margin that choice bought.

---

## 1. What we submit

Two files, one per track.

**Direct inhibition.** An ensemble over a shared feature matrix (2048 Morgan counts at r=2, 217
RDKit descriptors, 30 mechanistic columns), with the member set chosen **per enzyme** by a search
over all thirty-one subsets, and a dead-zone refit applied inside every member. Five members come
from that matrix — per-enzyme gradient boosting, a contrast-pooled model, a Gaussian process, a
neural trunk, and a re-projection of that trunk — and a **sixth** does not: a ridge head on 600
frozen dimensions of the third-party pretrained encoder declared above. The sixth member is worth
**+0.0138 of macro rank, sd 0.0006, sign 8/8 over eight seeds**, with the pair improving by 0.0197,
against a discriminating control (the same head and the same alpha grid on our own descriptor block)
at −0.0151 and sign 0/4. The per-enzyme compositions below are the search's answer over the first
five; the sixth is added to all four.

*This paragraph described a five-member ensemble until 15 October, three weeks after the sixth
member shipped. The out-of-fold table that follows is likewise the five-member configuration: it is
left as measured rather than restated, and the sixth member's increment is the figure above.*

    CYP1A2   per-enzyme + GP + trunk
    CYP2C9   GP + trunk
    CYP2D6   all five members
    CYP3A4   GP + trunk

**Time-dependent inhibition.** Not a classifier on the label. The shipped rule builds the label's
two conjuncts separately — a gate probability from the pre-incubation arm predicted by a four-member
ensemble, times a shift probability from a classifier on Δ > log₁₀2 — and thresholds the calibrated
product.

Out-of-fold performance, Butina-clustered five-fold split, four seeds:

    track           metric                  value
    regression      macro ST-RAE            0.6459
    regression      macro Spearman          0.6342
    TDI             MCC CYP3A4              0.3510
    TDI             MCC CYP2D6              0.1235
    TDI             macro MCC               0.2373

From a conventional starting point (same features, per-enzyme boosting, no ensemble) at 0.7150 pair
and 0.5651 rank, the trajectory is **+0.0579 of rank and −0.0583 of ST-RAE**.

We will say the uncomfortable part first: **the architecture is the least novel thing in this
report.** Gradient boosting, a Gaussian process and a small trunk are 2015 technology. What we think
is worth reading is where the gains came from, and — more of the document — where they did not.

## 2. The finding we would most want cited

**Physics enters this problem through the measurement model and returns zero through the feature
matrix.**

That sentence is not a slogan; it is what the log says once the interventions are sorted into two
piles. Everything we tried that modelled *how the assay produces a number* paid. Everything we tried
that added a *descriptor of the molecule* did not.

    shipped (the measurement model)                    rank gain   sign
    dead zone in every member                            +0.0197    4/4
    mechanistic block                                    +0.0163    4/4
    contrast pooling                                     +0.0141    4/4
    trunk as a fifth member                              +0.0054    4/4
    instrument equation (screening head)                 +0.0308    4/4

    refused (the feature matrix)              over ensemble  floor  as member   item
    3D shape: PBF, NPR, asphericity, PMI (16)       +0.0004 0.0049    +0.0087    281
    hand-built active-site blocks, 2D6              −0.0094 0.0049          —    179
    overlay with the co-crystal pose (contrast)     +0.0017 0.0061          —    296
    external CYP panels, auxiliary head             −0.0104      —    +0.0085    290
    multi-task trees                                −0.0062      —    +0.0059    292
    SMARTCyp, sites of metabolism (16)              +0.0009      —          —    179
    quantum block: HOMO/LUMO/gap/Fukui (10)             ~0       —          —    186
    CYP2C19 as a fifth-isoform donor                −0.0082      —          —    153

The right-hand columns are the point. Three of these arms carry a **real** signal as a standalone
member — the shape block +0.0087 on CYP2D6, the external-panel channel +0.0085 at sign 4/4, the
multi-task channel +0.0059 at sign 4/4 — and every one of them arrives at or below its floor once it
has to share an ensemble with five members that already correlate 0.91–0.99 with each other.

The dead zone is the largest single effect in the project and the clearest example of the law. The
organisers' metric forgives any prediction inside the credible interval, so the loss a model should
minimise is not the distance to the point estimate — it is the distance to the nearest bound. We
refit every member by majorise–minimise against `clip(p, lo, hi)`. That is not a feature and not an
architecture; it is reading the metric as a statement about the instrument.

The instrument equation is the same move applied to the screening data. The single-concentration
screen and the dose-response tables measure the same enzymes under different conditions; writing
down the relation between them, rather than concatenating one onto the other, gave +0.0308 of macro
rank at 4/4 seeds, t = +18.61, p = 0.0003.

Contrast pooling is the third. Pooling enzymes helps — but we spent five items on *why*, and the
answer is neither borrowed neighbours nor shared function nor sample size. Zeroing the enzyme
indicator drops the model 0.076 of rank **below** training each enzyme separately, and handing the
model each enzyme's mean for free recovers none of the gain. It works by **contrast**, and the
contrast is purchased with residual magnitude: switching the pooled member to absolute error
destroys 95 per cent of it, and the effect is **entirely CYP2D6** — the enzyme whose pooled member
does the work is also the only one that loses it.

## 3. Only rank survives, and this is a warning to the field

Before writing predictions out we fit an affine pair — a per-fold shrink and shift — to the metric.
It is strictly increasing, so it preserves Spearman exactly and rewrites everything else.

We recommend anyone reading this check their own gains against that transform, because we did it
late. Five separately measured improvements in our own log collapse into a band of 0.013 once the
pair runs — none clears the 0.007 noise floor and **three of the five change sign to harmful.** The
largest finding of that day was worth 0.046 of raw ST-RAE and 0.002 after.

The sharpest single case was a concatenated Chemprop embedding. Raw macro ST-RAE improved by 0.0084
(0.7673 → 0.7589). After the affine pair it **lost** 0.0113 (0.7150 → 0.7263). By rank it lost
0.0075, and under the test set's regime it lost 0.0182 — more than twice as much. The raw number and
the post-processed number carry opposite signs, and we published the raw one first. A raw
improvement in this challenge is not evidence of anything until it has been through the
post-processing that ships.

**And the closure has a boundary, which we found by crossing it.** What died above is a frozen
embedding used as COLUMNS CONCATENATED INTO THE SAME LEARNER. A separate ridge head on a frozen
encoder, whose *prediction* enters the mean as its own member, is a different object and it survives
the same post-processing: +0.0138 of macro rank, sign 8/8, with the affine pair improving rather than
absorbing it. It ships, and §1 and the disclosure above name it. The checkpoint differs too — the
encoder that failed was pretrained on the very RDKit descriptors our feature block already contains,
which is its best case as a substitute for those columns and its worst as a decorrelated member. So
the paragraph above is a warning about raw numbers and about one vehicle, not a verdict on frozen
encoders.

The mechanism is worth stating because it generalises: the affine pair is a shrinkage fitted
directly to the metric, per fold, so anything that merely reduces prediction variance or removes a
systematic offset is something the pair already does. It does not add to such improvements — it
substitutes for them.

The practical consequence for the leaderboard's secondary metrics: the organisers publish Spearman
ρ and Kendall τ with bootstrap intervals, and those are the columns an affine transform cannot
touch. We optimised them on purpose.

## 4. Uncertainty, in the units the metric pays in — and over our own conclusions

Two objects, and the first is derived from the competition's metric rather than imported.

**Per-compound, we predict the probability of not paying at all.** Conventional UQ puts an interval
around the prediction. Under ST-RAE that is the wrong object: a prediction anywhere inside the
compound's published credible band scores exactly zero, so distance *within* the band is not paid
for, and an interval straddling the band edge says nothing about what it will cost. The quantity the
metric cares about is

    P(hit_i) = P( lo_i <= ŷ_i <= hi_i )

the natural partner to the dead zone — one trains predictions *into* the band, the other says
whether they landed. Out of fold, 31.3 % of our predictions already score exactly zero, and the rate
is strongly structured by a quantity the model itself emits: band width is a decreasing function of
the label, so on CYP3A4 the hit rate runs 0.788 in the lowest potency quintile against 0.161 in the
highest — a five-fold spread visible entirely from the model's own output.

    фермент    база     AUC    Brier    Brier константы на базе
    CYP1A2    0.247   0.534   0.1928                    0.1861
    CYP2C9    0.439   0.597   0.2490                    0.2463
    CYP2D6    0.198   0.486   0.1635                    0.1585
    CYP3A4    0.369   0.750   0.1814                    0.2328

**On three enzymes of four the Brier score is worse than quoting the base rate, and we report it as
such.** It is real on CYP3A4 only: AUC 0.750, Brier beating the constant by 0.051. The part we would
defend is that the mechanism says WHICH enzyme it will work on before the model is fitted — CYP3A4's
bands span a nineteen-fold range between the tenth and ninetieth percentiles against five to eight
for the others, and where the band barely varies there is nothing for a hit probability to
discriminate. The better estimator is a single constructed feature, the signal-to-noise ratio
`z = ŵ(ŷ) / (2·σ̂)` calibrated by isotonic regression, not a learned classifier. Full derivation and
tables in `METHOD.md`.

**Over our own conclusions**, separately, we have a quantified answer to "is this number real", used
as a gate on every claim.

    noise floor (four seeds)                          value
    macro, by rank                                   0.0036
    CYP1A2 / CYP2C9 / CYP2D6 / CYP3A4     0.0061 / 0.0071 / 0.0049 / 0.0033
    paired (difference of two submissions)           0.0052
    MCC: CYP3A4 / CYP2D6 / macro          0.0281 / 0.0419 / 0.0076

Nothing below its floor is reported as a result. The split seed alone moves macro ST-RAE by 0.016 —
more than a typical effect — which is why every claim is quoted over four seeds and why the sign
count (how many of four seeds agree) is printed beside every mean.

We also pre-registered what the reveal should show, before it happens. On the arm that ships:
macro ST-RAE inside **[0.6114, 0.6997]** at n = 750 (half-width 0.0442) and **[0.5945, 0.7265]** at
n = 375; macro Spearman half-width **0.0157** at n = 750 and **0.0270** at n = 375. The n = 375 rows
are stated as *lower bounds on the spread*, because the live half is split by chemical series and a
clustered split has strictly higher variance than the random resampling those intervals assume.

One consequence we want on the record because it cuts against us: our absolute score is
unpredictable to about ±0.04, while our position relative to a *similar* submission is pinned to
about ±0.02. Anyone comparing two entries a few thousandths apart on this leaderboard is reading
noise, ours included.

**What we do ship per compound is a reliability flag, built from two statistics that need no
labels.** For each of the 750 blinded molecules: how often the five members disagree about its
ordering against the others, and how far it sits from the training rows labelled for that enzyme.
Neither uses a label, a leaderboard score or an external lookup — only the test structures and our
own committed predictions. Zero model fits; it is arithmetic over predictions already on disk.

Its control matters more than its output. The aggregate contested rate it computes must reproduce a
figure we published earlier from a cache that no longer exists, and an independent implementation
returns **0.2820 / 0.2683 / 0.3139 / 0.2319** against the recorded 0.2820 / 0.2683 / 0.3139 /
0.2319 — exact on all four enzymes. A second agreement arrived unasked: its nearest-neighbour
similarities (0.517 / 0.518 / 0.471 / 0.538) match those from a different script, written days
apart for a different question, to every printed digit.

    enzyme    median contested   median similarity   share below 0.4
    CYP1A2               0.295               0.517             0.287
    CYP2C9               0.281               0.518             0.351
    CYP2D6               0.317               0.471             0.420
    CYP3A4               0.234               0.538             0.215

The two share no input beyond the molecules — one is built from where five models argue, the other
from fingerprint distance — and both rank **CYP2D6 worst**. So do the rank (0.480), the band widths
(narrowest, so the metric forgives least), the TDI cell, and the train-to-test consistency ratio.
Five unrelated measurements landing on one enzyme is the clearest signal in the project about where
its remaining difficulty lives.

It changes no prediction, and we would rather say that plainly than let a diagnostic be mistaken for
a gain: the pairs where members disagree are **hard, not mis-ordered** — the ensemble mean already
scores 0.58 there, well above a coin — so repairing them would demand new information rather than a
rearrangement of what we hold.

## 5. How claims were tested

Every intervention in this project went through the same harness, and the harness is the part we
would defend as the contribution.

- **Pre-registration.** The acceptance threshold and a written prediction are committed **before**
  the run. When the prediction is wrong, the item says so — including the cases where we were right
  about the outcome and wrong about the mechanism.
- **Wrong-isoform controls, not permutation.** For any quantity derived from a (ligand, cavity) pair,
  the control is the same quantity computed against a *different* enzyme's cavity. A permutation
  control cannot separate real pocket information from a disguised volume descriptor; the wrong-cavity
  control can. On CYP3A4 our cheap structural proxy correlates +0.283 with heavy-atom count even
  after centring, and a permutation test never noticed.
- **Harness controls that must read exactly zero.** Where an arm rebuilds a component that is kept on
  only some enzymes, the untouched enzymes are required to print `0.000000` to every digit. Twice
  this caught nothing and confirmed the rig; that is what it is for.
- **Fresh seeds for confirmation.** A composition chosen on seeds 0–3 is confirmed on seeds 4–7, and
  the item states which seeds generated the hypothesis.
- **Knockout, not addition.** Named contributions sum to +0.0555 when each is measured by *adding* it
  to its own base, against a measured total of +0.0579. Measured by *removing* each from the shipped
  configuration they sum to **+0.0315** — and contrast pooling, worth +0.0141 by addition, is worth
  **+0.0008 at sign 0/4** as a marginal member. The table of contributions is not additive and we
  stopped presenting it as though it were.

The journal is public and runs past item 300. Most of it records something that did not work, and we
think that is the useful part — a correct idea takes one item, a wrong one takes three (proposal,
refutation, correction to the refutation).

Item numbers in this report are checkable: the affine-pair collapse is 77 and its by-rank re-reading
is 117; pooling's mechanism is 131, 132 and 264; the shape block is 281; the estimator-family screen
is 289; transfer and multi-task are 290 and 292; the structural proxy is 296; and the pre-registered
reveal bands are 294.

### A failed prediction as a measuring instrument

This is the practice we would most want borrowed, and it is one step past pre-registration.

Some decisions turn on a quantity you cannot measure without spending something irreversible. Ours
was the width of the credible bands on the blind half. The metric is a hinge that charges nothing
inside a compound's band, so where to place a prediction depends on how wide those bands are — and
the bands are not published. The only instrument that could read them was a submission, and
submissions are rate-limited to one per twelve hours with the latest one counting.

The usual move is to pick the most likely value, predict an outcome, and find out afterwards that a
missed prediction taught you nothing about which assumption failed. Instead:

> **Before acting, map the prediction's sensitivity to the unmeasurable quantity, and commit the map
> with the prediction. Then the outcome identifies that quantity whether or not the prediction
> holds.**

Committed before the upload, alongside a predicted cell of 0.6246 and an acceptance band:

    множитель ширины полос   0.4x     0.6x     0.8x     1.0x     1.5x
    предсказанная CYP2D6   0.8701   0.7653   0.6709   0.6246   0.6046

The board returned **0.6805**. The prediction failed — 0.6805 sits outside the committed
[0.6146, 0.6623] — and in the same stroke the map reads the multiplier back as **≈0.78×** of our own
training band widths. A quantity that had been unmeasurable for the whole competition became a
number, and it became one *because* the prediction was wrong in a direction the map could resolve.
Had we committed only the prediction, the same result would have been a puzzle.

**The reading checks out to four decimals, which is the control that makes it a measurement rather
than a story.** Interpolating the committed map at the recovered 0.78× returns **0.6803** against the
board's observed **0.6805** — a miss of 0.0002. The map was not adjusted afterwards; it is the one
in the commit that precedes the upload. So the model was never wrong about the mechanism. It was
wrong about one number, and that number is now known to within the width of the check.

The map does a second job, and this is what makes the practice pay rather than merely console. The
placement was chosen on **worst case across the map, not on expectation**. Two candidates were in
hand: a half step and a fuller one. The fuller step was better at 1.0× and turned negative between
0.8× and 0.6×; the half step never lost a place down to 0.6×. We shipped the half step, and the
realised 0.78× landed inside exactly that margin — at the true multiplier the fuller candidate would
have scored about three board places *worse*. The map priced the decision before the fact and read
the parameter after it.

Two honest limits. The parameter must be identifiable from the observable — the map has to be
monotone in it and the observable has to move enough, which is a property to check rather than
assume. And the reading inherits whatever the map was built on: ours rests on a world model that
still cannot reproduce all six of that cell's published numbers, an inconsistency we record and have
not resolved (journal item 312). So the 0.78× is a measurement with a stated model attached, not a
constant of nature.

Full derivation, the pre-registration as committed, and the scored verdict: journal items 320, 321,
and the checker `verify/k104_prereg320.py`, whose rules and sensitivity map are in the commit that
precedes the upload.

## 6. What we refused, and what that closed

Five conclusions we would hand to anyone continuing this problem.

1. **Descriptors of the enzyme are closed by arithmetic, not by measurement.** Four enzymes, all four
   seen in training, no zero-shot requirement — so a one-hot indicator is a sufficient statistic for
   enzyme identity, and any vector of protein descriptors is a four-row table, i.e. a change of
   basis. Measured: +0.0007. A fifth isoform as donor: −0.0082. What works instead is the
   *interaction* form: ligand columns conditioned on a known site.
2. **Ensemble redundancy is the binding constraint, not accuracy.** Member error correlations run
   0.91–0.99 across every pair (item 280). Four separate arms, measured in four separate items, each
   produced a genuine single-model gain that died on insertion: the largest was the shape block's
   **+0.0087 on CYP2D6 diluting to +0.0004** at sign 2/4 against that enzyme's floor of 0.0049
   (item 281), and CYP2D6's per-enzyme member effect of **+0.0149** at 4/4 — three times its floor —
   reached **+0.0011** over the ensemble. Item 269 sharpened the mechanism and is the one to read:
   cross-enzyme stacking arrived at **−0.0007** over the ensemble and died of **redundancy, not
   dilution** — the other members already carried the information. Escaping dilution is therefore
   necessary and not sufficient, which is the sentence we would put on the wall.
3. **No estimator family is both accurate and decorrelated** (item 289). Random forest and
   extra-trees decorrelate to only 0.95–0.96; SVR-RBF reaches 0.898 and kernel ridge 0.661, and both
   fail on accuracy instead. Sweeping the kernel width does not find a point that is both: kernel
   ridge peaks at macro rank 0.5417 and never reaches the weakest shipped member at 0.5776.
4. **The (ligand, cavity) pair-function class is not empty, but its magnitude is small.** Our targeted
   structural proxy beat its wrong-isoform control on CYP1A2 at sign 4/4 — the discrimination a
   permutation test cannot supply — and still came in at +0.0017 against that enzyme's floor of
   0.0061, from a univariate precondition of +0.173. A roughly hundred-fold collapse between a raw
   correlation with the label and a rank gain over a three-member mean.
5. **Transfer and multi-task both fail the same way.** An auxiliary-head trunk trained on external CYP
   panels and multi-task trees each showed a real channel over the ensemble (+0.0085 and +0.0059, 4/4)
   and each lost by rank once inserted (−0.0104 and −0.0062). The channel is real; the vehicle is
   redundant.

Several of these were reached independently by other entrants whose code is public, which we take as
corroboration rather than coincidence: 3D shape descriptors net-negative for an ensemble, tabular
models on ECFP4 correlating r ≈ 0.9 with a gradient-boosting baseline, and pseudo-labelling from the
single-concentration screen degrading accuracy monotonically with pseudo-weight.

## 7. Reproducibility

- Environment pinned by `uv.lock`; the `scikit-learn` ceiling is measured, not cautious — 1.3.2
  through 1.8.0 regenerate the saved out-of-fold predictions bit for bit, while 1.9.0 moves every
  prediction by ~0.135 pIC50.
- The split lives in one module and is pinned to a golden digest (`2d93c19815e14261`, 4703 clusters)
  by a test that CI runs on every push and every pull request. If the digest moves, the published
  tables no longer describe the code.
- CI additionally re-scores the committed predictions against documented values and imports every
  script, so a file that stops running is caught rather than discovered later.
- Structural inputs that are too large to commit are rebuilt by committed scripts, both default-safe:
  `src/prep_receptors.py` (chain A plus its own heme, parsed by column) and `src/prep_lig3d.py`
  (ETKDGv3, fixed seed, MMFF cleanup, names carrying the row index).
- Four people work from four machines; the protocol for the unmergeable prediction files, and the
  tripwire that verifies every commit on `main` arrived through a pull request with a green split
  guard, are in `CONTRIBUTING.md`.

## 8. Limitations

- **The architecture is conventional.** If the award weighs architectural novelty, this entry does not
  compete on that axis and we are not going to pretend otherwise.
- **The per-compound hit probability works on one enzyme of four.** Section 4 reports it in full: on
  CYP1A2, CYP2C9 and CYP2D6 its Brier score is worse than quoting the base rate, and we say so rather
  than quoting the CYP3A4 number alone. This sentence previously read "No predictive uncertainty",
  which was simply wrong about our own work — the object exists, it is derived from the metric, and
  its failure on three enzymes is a measured property of those assays' band widths rather than a
  reason to disclaim it.
- **One enzyme carries the rank.** CYP3A4 reaches Spearman 0.822 out of fold while CYP2D6 reaches
  0.480. Part of that gap is metric structure rather than model quality — CYP2D6 has the narrowest
  credible intervals of the four, so the soft threshold forgives least there, and CYP3A4 has two
  fifths of its labels below pIC50 4, where the metric downweights.
- **CYP2D6's TDI cell is weak** (MCC 0.1235 against a floor of 0.0419) and we have not fixed it. Public
  entries suggest it is hard for everyone, which is an explanation and not an excuse.
- **A campaign is still running.** Docking 5652 ligands into four cavities was pre-registered blind,
  with a written prediction that nothing passes. Whatever it returns will be in the journal.

## 9. A closed form for the metric, and the correction that goes with it

Section 2 carries the finding we would most want cited, because it is about the problem. This one is
about the *scoring function*, it took five minutes and fitted nothing, and it is the part most
directly reusable by anyone else working on this challenge.

**The credible band is a deterministic function of the label.** Isotonic regression from the label
alone onto the band width:

    enzyme        n   rho(y, width)   R2 isotonic   sd of label position in band
    CYP1A2     1412          -0.885         0.963                          0.057
    CYP2C9     1285          -0.899         0.928                          0.063
    CYP2D6     1493          -0.558         0.955                          0.087
    CYP3A4     2335          -0.928         0.970                          0.088

The width is recovered to within three per cent of its variance, and the label sits at a nearly
fixed relative position inside its own band — the standard deviation of `(y − lo)/w` is under 0.09
everywhere. So `lo` and `hi` are the label plus and minus a function of the label.

**Therefore ST-RAE is a potency-weighted absolute error.** The forgiveness threshold is about
**1.25 pIC50 below a label of 3.5 and about 0.13 above 4.6** — a fifty-fold difference in what
counts as a miss, determined entirely by how potent the compound is. That is a much simpler
statement than "soft-thresholded error against a confidence band", and it has a direct consequence
for modelling: a model fitted to minimise unweighted error is optimising the wrong loss, which is
why our dead-zone refit against `clip(p, lo, hi)` is the largest single effect in the project.

**And the obvious inference from it is wrong, which is why this section has two halves.** Reading
the above as "get the potent compounds right, the weak ones are nearly free" is natural and false.
Share of the ST-RAE numerator by potency quartile, on the submitted ensemble:

    enzyme     q1 weakest      q2       q3   q4 potent   half-threshold q1   half-threshold q4
    CYP1A2         37.1 %   9.6 %   12.5 %      40.8 %               0.537               0.098
    CYP2C9         27.9 %   9.9 %   12.8 %      49.5 %               0.546               0.128
    CYP2D6         29.2 %   9.6 %   10.2 %      51.0 %               0.245               0.105
    CYP3A4         16.0 %  20.9 %   22.4 %      40.7 %               1.233               0.069
    mean           27.6 %  12.5 %   14.4 %      45.5 %

**The penalty is U-shaped, not monotone.** The potent quartile dominates at 45.5 per cent, as the
closed form predicts — there is almost no forgiveness there. But the weak quartile is second at
27.6 per cent, and the two middle quartiles together supply only 26.9. The weak end is not free in
practice because the median absolute residual there runs 1.27 to 1.67 pIC50, which overruns even a
threshold of half a log unit. The forgiveness at that end differs **fivefold between enzymes** —
a half-threshold of 0.245 on CYP2D6 against 1.233 on CYP3A4 — and the cost tracks it exactly:
CYP3A4, the most forgiven, is the only enzyme whose weak quartile is cheap at 16.0 per cent, while
CYP2D6 with the tightest bands pays 29.2 per cent there despite being the enzyme where every model
does worst.

So effort belongs at **both** ends and the middle can be left alone — a different prescription from
the one the closed form alone implies. We record the pair together because the first half without
the second would have sent us, and anyone reading it, to optimise the wrong region.

### Reading the blind band geometry off the leaderboard, with no model at all

The organisers publish MAE and ST-RAE side by side for every entrant. Board MAE is plain
`sklearn.mean_absolute_error` (`evaluation/config.py`), and ST-RAE's denominator is a hinge on the
constant predictor at `mean(y_true)` and so **contains no `y_pred`**. The denominator is therefore
one constant shared by the whole field, and the ratio

    MAE / ST-RAE

is a pure measure of how much of an entrant's raw error the credible bands absorb — comparable
across every submission on the board, on the same hidden labels and the same fixed bootstrap
resamples. Writing `C` for that shared denominator and `n` for the scored count, two per-compound
facts turn the field into bounds rather than estimates:

    hinge_i <= |e_i|            =>   C/n  <=  min over entrants of  MAE/ST-RAE
    |e_i| - hinge_i <= W_i      =>   Wbar >=  max over entrants of  (MAE - (C/n)·ST-RAE)

Nothing is fitted. **The blind band geometry — a quantity the organisers never published — is read
directly off the published columns**, and any entrant can do it for their own placement. Our own
absorption, stated as the bound the board supports:

    фермент   MAE/ST-RAE    C/n <=    Wbar >=   поглощаем не менее
    CYP1A2        1.3425    0.7822     0.4132              41.7 %
    CYP2C9        1.0151    0.5081     0.3380              49.9 %
    CYP2D6        1.5622    0.5641     0.9300              63.9 %
    CYP3A4        1.1112    0.7491     0.2100              32.6 %

Two controls. The formula returns exactly 0 % for an entrant whose predictions lie outside every
band (their ratio *is* the minimum) and tends to 100 % as ST-RAE tends to zero at finite MAE — both
known a priori. And the ordering by recovered band width matches the ordering by absorption at the
extremes: CYP2D6 has the widest bands and absorbs most, CYP3A4 the narrowest and least. It is not
perfectly monotone in between — CYP2C9 absorbs more than CYP1A2 with narrower bands — because
absorption depends on where an entrant's errors fall, not on width alone, and we report that rather
than smoothing it.

**What it caught.** On 21 September this report's own journal concluded the blind CYP2D6 bands were
negligible, from the same statistic computed on **two** of our own submissions — a 2.2 per cent
spread. Computed on the field, CYP2D6 spans 0.5641 to 2.2625, a four-fold range and the largest of
the four enzymes. The two-point test had no power because both our placements sat in a locally flat
window near the bottom of a curve with an 80 per cent range. **Sixty-nine entrants were posting a
strictly worse CYP2D6 MAE than ours and a strictly better ST-RAE** — arithmetically impossible
without band absorption, since with no bands ST-RAE is `MAE/MAD` and strictly increasing in MAE.
Acting on that reading moved the cell from 0.7707 to 0.6805 and the entry ten board places; the
derivation and its scored pre-registration are journal items 320 and 321.

**Why our own optimiser had missed it, which is the part that generalises.** Band width is close to
a step function of the label: below pIC50 ≈ 4.13 the mean width on our own data is 2.379, above it
0.21–0.44. Weak compounds are cheap to miss and potent ones are not, so the ST-RAE optimum sits
*above* the label mean by an amount set by how much of the population lies under the step. Our
placement is fitted out of fold against our own labels — and the direct-inhibition assay was run on
compounds a screen had already flagged, so 9.8 per cent of our CYP2D6 training mask lies under the
step against 66–73 per cent of the blind set. **The optimiser had the right bands and the wrong
population.** Any entrant fitting a placement on a selection-biased training marginal has the same
exposure, which is why this is offered as a result about the metric rather than about our model.

## 10. A note on metric robustness

Offered as an observation about the scoring function, not as a complaint about anyone.

ST-RAE clips each error to the credible interval and divides by the same quantity for a
mean-predictor baseline. Both halves therefore depend on where the test labels sit and how wide their
intervals are, and neither depends on ordering. Across our four enzymes, out-of-fold ST-RAE runs
almost inverse to median interval width. The practical consequence is that **placement** — the mean
and spread a submission is centred on — moves the primary metric a long way independently of whether
the predictions are correctly ordered, and the leaderboard displays rows where ST-RAE is good while
Spearman is approximately zero.

We treated the test distribution as unknown and estimated it the only way that touches no leaderboard
feedback: nearest-training-neighbour label transfer for the 750 test compounds, with the estimator's
bias calibrated leave-one-out on the training set, where the true moments are known. The gate passed
(mean shifts within 0.062, sd ratios 0.948–1.021), and the estimate puts the blind set at or above
training potency on all four enzymes — which is what an analog expansion of the top hits implies by
construction. We pre-registered, before running it, that a disagreement with any other estimate would
be reported as *ambiguous* rather than as a refutation, because the live half is series-split and
need not represent the full 750. It disagreed, and we are holding to that.

If the organisers find the placement sensitivity worth addressing in future challenges, a secondary
metric that is invariant to a monotone transform — Spearman, which they already publish — does the
job.

**Correction, 28 October.** Until 27 October this section ended with the sentence "Nothing in our
submission is calibrated onto any estimate of the blind label distribution." That is no longer true
and the sentence is withdrawn rather than quietly deleted. Our current submission expands the CYP2D6
column by a factor chosen against the blind set's label spread as bounded by the leaderboard's own
published columns (section 9's `sd(y_test) ≥ MAE/√(1−R²)`). No test label was used and none is
available to us; the quantity is derived from numbers the board publishes about every entrant.
Section 11 records what that is worth and why we think it is a result about the metric rather than
about our model.


## 11. Two results from the final fortnight, offered because they are about the data and the metric

### 11.1 The blind set's construction is recoverable from the released data, and we recovered it

The challenge page states the design: hit expansion on the top 25 hits of CYP1A2, CYP2C9 and CYP3A4,
ten ECFP4 chemisimilars per hit from the Enamine catalogue, 25 × 3 × 10 = 750. Every clause of that
is checkable against the released SMILES, and all of it holds:

| | measured |
|---|---|
| distinct nearest training neighbours of the 750 | 132 |
| share of the test covered by the top 75 of them | 87.1 % |
| same statistic for 750 random training compounds (null) | 16.7 % |
| children per parent | median exactly 10, range 4–14 |
| CYP2D6 members of its own top 25 among the parents | 1, against 21 for CYP2C9 and 16 for CYP3A4 |

We report it because of its consequence rather than for its own sake. For any single isoform, two
thirds of the 750 were selected for a *different* isoform, and **CYP2D6 has no series of its own at
all**. A model trained on compounds promoted to dose-response — which is to say, on hits — predicts
at the level of hits, while the blind set is dominated by compounds chosen for other enzymes. We
spent three days trying to turn that into a corrected level and could not: the derivation holds on
CYP1A2 and CYP2D6, where the single-concentration screen separates actives (Youden 0.814 and 0.892),
and is refused on CYP2C9 and CYP3A4, where it does not (0.139 and 0.151). The limit is structural,
not a shortage of effort: **the released data contain exactly one measurement taken before
selection** — the single-concentration screen — because pIC50, Emax and the TDI curves all exist
only for compounds promoted *because* they screened active. Any question of the form "what would an
untested analogue do?" is answerable only through that one readout.

This is also why the analogue structure is harder than its similarity suggests. At the blind set's
own Tanimoto band, two compounds differ in activity about as much as two random ones — on CYP1A2,
1.023 of pIC50 against 1.090 — so the density is not a source of easy predictions. The organisers'
own phrase for this, "activity cliffs", is the right one, and we can put a number on the cost: our
pairwise sign accuracy falls monotonically with structural similarity on CYP1A2 and CYP2D6, from
0.699 and 0.669 at Tanimoto below 0.20 to 0.608 and 0.555 in the band the test occupies, while
CYP2C9 and CYP3A4 stay flat.

### 11.2 Placement moves the primary metric independently of ordering, demonstrated rather than argued

Section 10 makes this case correlationally. We can now make it by controlled experiment, because we
ran one.

Two submissions, twelve hours apart, differing in exactly one column. The CYP2D6 predictions were
multiplied by 1.5 about their own mean — a strictly increasing map, so the ordering cannot change —
and the other three isoforms were left bit for bit identical.

| | before | after |
|---|---|---|
| MA-ST-RAE | 0.5137 | **0.4954** |
| MA-MAE | 0.7297 | 0.7054 |
| MA-R² | 0.5166 | 0.5374 |
| MA-Spearman ρ | 0.7212 | **0.7212** |
| CYP2D6 ST-RAE | 0.6808 | **0.6078** |

Spearman did not move to four decimal places, which is the control confirming the transform did on
the live set what it claims. The CYP2D6 tab moved by 0.0730 and the macro by 0.0183 — and
0.0730 / 4 = 0.01825, so the whole macro movement is attributable to the single column that changed.
**A monotone rescaling of one isoform, carrying no new chemistry and no new information of any kind,
is worth 0.0183 of the primary metric.**

The converse also holds and we think it is the more interesting half. An earlier pair of
submissions shared a marginal distribution *exactly* — one was a rank map of the other onto its
predecessor's per-isoform values, bit for bit — and differed only in which compound received which
value. That was worth 0.0220 of MA-ST-RAE at identical Spearman. So the metric is not purely a
placement score either: **at a fixed marginal, a better ordering pays, and at a fixed ordering,
placement pays about as much.** Both halves are real, and an entrant who optimises only one of them
leaves the other on the table.

We offer this as calibration data for a future challenge rather than as a criticism. If the
intention is that the primary metric should reward chemistry, then the 0.0183 above was obtained
without any, and we would have been unable to find it without the MAE and R² columns the board
publishes — which is itself a reason to keep publishing them.

### 11.3 Four more things that did not work, with the reason each failed

Recorded because the reasons are reusable even where the results are not.

**A joint multi-head fine-tune over challenge + ChEMBL + qHTS**, warm-started from the published
CheMeleon CYP checkpoint, one head per source with the label scales never merged, and the panel's
28 595 censored cells entering as one-sided bounds. Five folds, four nets averaged in the
configuration the best open-code entrant describes. Solo macro Spearman 0.5634 for one net and
0.5741 for the average, against 0.6416 for our own five-member ensemble, and −0.0020 at sign 0/4
when nested into it. The representation is genuinely decorrelated — error correlation 0.846 against
the 0.93 we usually see — and that did not rescue it.

**Left-censored bounds from our own screen**, the device the three open-code entrants above us all
use. We hold 129 CYP2D6 labels below pIC50 4.0 and 2 882 screened compounds with no curve at all, a
22-fold increase in evidence of inactivity, at a bound the protocol fixes exactly (the screen's
single concentration, 4.95 × 10⁻⁵ M, is pC₀ = 4.305). On a single-stage per-enzyme model it is worth
+0.0125 of macro rank with a permutation null at 0/6. Inside the member we actually ship it is worth
nothing, and the null and the real bounds become indistinguishable — −0.0015 against −0.0014. The
reason is that our shipped member already runs a dead-zone pass that projects its prediction into
the credible band, and the band for a low-activity compound is seven to ten times wider than for an
active one. **The two devices say the same thing, and ours said it first.**

**A mixture placement** that sends the predicted-inactive fraction to the bottom of the range, which
the band geometry appears to license: −0.0918 of macro penalty against the affine pair at sign 0/4,
measured with oracle pools and the oracle active fraction taken fold-wise. The wide band for
inactives is only free to someone who knows which compounds are inactive, and at Spearman 0.72 we do
not.

**Placement fitted on our own labels.** Our only simulation of the transform in 11.2 recommended
expansion by 1.40 with a shift, stable across folds and out of fold. It was void, and the control
that caught it is worth repeating: the submission's own CYP2D6 column scores 2.3761 against *our*
training labels where the board scores it 0.6808. A placement fitted to the training marginal is
fitted to the wrong population, which is the same trap section 9 records from the other side.
