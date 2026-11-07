# Physical Chemistry, Ionization Thermodynamics & Assay Physics of Human Cytochrome P450 Inhibition (`CYP1A2`, `CYP2C9`, `CYP2D6`, `CYP3A4`)

**Official Scientific Discovery & Methodology Report — OpenADMET Cytochrome P450 Challenge**

| Field | Details |
| :--- | :--- |
| **Author & Organization** | **Aarshit Mittal** · **Rasayan Labs** ([https://rasayan.ai](https://rasayan.ai)) · **Team Handle:** `rasayan-labs` |
| **Date & Submission Scope** | **November 2026** · `6,146` Training Compounds & `750` Blind Test Compounds Across 4 Human CYP Isoforms |
| **Challenge Tracks Covered** | **Track 1:** Direct Reversible Inhibition (−NADPH pIC₅₀ Regression) · **Track 2:** Time-Dependent Inhibition (`CYP2D6` & `CYP3A4` Binary Classification) |
| **Data Compliance** | `Proprietary Data = No` · `External Data = Public Scientific Datasets Only (Zero Overlap With Challenge Test Compounds)` |

---

## 1. Executive Summary: Assay-Aware Multi-Task Learning Coupled With Ionization Thermodynamics

Standard machine learning pipelines treat Cytochrome P450 (CYP450) inhibition as a static, single-endpoint 2D regression problem: map a neutral SMILES string directly to a scalar pIC₅₀ or a binary Time-Dependent Inhibition (TDI) label. Across our investigation of the OpenADMET CYP450 Challenge dataset (`6,146` unique training compounds across four assay tables and `750` blind test compounds), we found that static single-task 2D regression breaks down (`Macro ST-RAE = 0.8307`, `R² = 0.2666` on unseen scaffolds) because microsomal inhibition at physiological pH 7.4 is governed by **three coupled physical and experimental factors**:

- **Factor 1 — Assay Measurement Structure, `73.31%` Single-Isoform Sparsity & `15.9×` Censored-Floor Noise:** Out of `4,905` training compounds with full dose-response curves (DRCs), `73.31%` (`3,596`) are measured in only one CYP isoform, whereas `4,376` compounds are screened simultaneously across all four isoforms at 49.5 μM (`17,504` single-concentration rows, correlating with DRC pIC₅₀ at Spearman ρ = 0.8283–0.9525). Simultaneously, compounds failing to reach 50% inhibition at the 100 μM top assay concentration (pIC₅₀ ≤ 4.0) exhibit up to **15.90-fold wider** 95% Bayesian credible intervals (`ΔCI₉₅`), requiring assay-aware multi-task bridging and inverse credible-interval-width loss weighting.
- **Factor 2 — Aqueous Ionization Equilibrium (pKₐ & log D₇.₄) & Isoform-Specific Active-Site Electrostatics:** At pH 7.4, a molecule's acid-base dissociation constants (pKₐ) determine the equilibrium population of neutral (`f_neutral,7.4`), protonated cationic (`f_pos,7.4`), dicationic (`f_dipos,7.4`), and anionic (`f_neg,7.4`) microstates, as well as its effective distribution coefficient (log D₇.₄). **`CYP2D6`** contains negatively charged active-site carboxylates (`Glu216` and `Asp301`) that electrostatically favor protonated basic cations (+1 charge) over neutral lipophiles, whereas direct reversible inhibition in **`CYP3A4`**, **`CYP2C9`**, and **`CYP1A2`** correlates strongly with effective lipophilicity (log D₇.₄)—even after controlling simultaneously for molecular weight, heavy-atom count, polar surface area, and Bemis–Murcko scaffold (`Section 3.1`).
- **Factor 3 — Catalytic Turnover (+NADPH), Baseline-Binding Confounding & Oxidizable Warheads in TDI:** In this screening cascade, binary `is_TDI` is defined by a ≥2-fold IC₅₀ shift (ΔpIC₅₀ ≥ 0.301) between the +NADPH and −NADPH arms. While a ≥2-fold IC₅₀ shift alone does not prove covalent mechanism-based inactivation (MBI) without dilution/dialysis kinetics (`k_inact / K_I`) or adduct identification, controlling for baseline reversible binding (pIC₅₀,direct > 4.0) reveals statistically significant enrichment of classical oxidizable substructures (`1,3-benzodioxoles`, `furans`, `pyrroles/indoles`, `thiophenes`) that are consistent with catalytic metabolite or metabolic-intermediate complex (MIC) formation (`Section 3.2`).

---

## 2. Discovery I: Quantum-Informed Macroscopic pKₐ Modeling (`RMSE = 0.7105` on `356` Public Benchmark Transitions)

Because ionization at pH 7.4 directly determines microstate populations and log D₇.₄, errors in predicted pKₐ propagate nonlinearly into protonation fractions via the Henderson–Hasselbalch relation:

> **Equation 1 (Henderson–Hasselbalch Microstate Fractions at pH 7.4):**
> ```text
> f_pos(pH) = 10^(pKa_base - pH) / [1 + 10^(pKa_base - pH)]
> f_neg(pH) = 10^(pH - pKa_acid) / [1 + 10^(pH - pKa_acid)]
> ```

A 1.0 pKₐ unit error near pH 7.4 shifts the predicted cationic or anionic population by nearly 50 percentage points and distorts log D₇.₄ by up to 1.0 log unit. To provide accurate ionization features for CYP450 modeling, we developed the **Rasayan 3-Pillar Macroscopic pKₐ Engine** trained strictly on public scientific datasets (`DataGemma/ChEMBL`, `IUPAC`, and ` OECD` public corpora; [1–7]):

- **Multi-Microstate Thermodynamic GNNs & Quantum Corpus:** Trained across `6,694` core experimental pKₐ transitions partitioned by 5-fold Bemis–Murcko scaffold grouping (`1,344` unique molecular frameworks), `20,195` curated auxiliary transitions, and `114,922` semi-empirical quantum-mechanical (`GFN2-xTB` [8]) microstates using thermodynamic log-sum-exp partition aggregation across tautomer and protonation states.
- **`1,033`-Dimensional Reaction-Center Concentric Shells:** Whole-molecule descriptor pooling dilutes the protonation site signal across 20–40 spectator heavy atoms on drug-like scaffolds. Using canonical heavy-atom skeleton symmetry matching (`100.0%` atom alignment across all `114,922` microstates), we isolate the ionizing reaction center (`d = 0`) and extract concentric topological and `GFN2-xTB` electronic shells (`d = 1, 2, 3` bond hops: partial charges, Wiberg bond orders, and Fukui indices), paired with a Boltzmann-gated microscopic reaction readout.
- **Strict Zero-Overlap Benchmark Qualification Protocol:** All `356` external benchmark transitions across the four public evaluation suites (`Novartis Acid + Base = 280` [1], `SAMPL6 = 31` [2], `SAMPL7 = 20` [3], and `SAMPL8 = 25` [4]) were strictly excluded from both the `6,694` core and `20,195` auxiliary training sets using canonical tautomer-invariant InChIKey deduplication prior to model training. Published baseline RMSEs in `Table 1` are reported directly from their respective peer-reviewed benchmark studies (`Uni-pKa` [5], `AcepKa` [6], `QupKake` [7], `MolGpKa` [9], `Starling`, and `ChemAxon Marvin`) evaluated on the same named test sets; because historical literature baselines differ in their internal training corpora and tautomer-assignment conventions, these comparisons should be interpreted alongside the cited primary benchmark protocols.

**Table 1. Out-of-Sample Benchmark Performance of the Rasayan 3-Pillar Macroscopic pKₐ Engine vs. Published Literature Baselines [1–7, 9]**

| Benchmark Suite | Transitions (N) | Rasayan RMSE | Rasayan MAE | Rasayan R² | Spearman ρ | Within ±1.0 pKₐ | Published Baselines Cited From Literature (RMSE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **External Public All (Combined)** | **356** | **0.7105** | **0.5485** | **0.9131** | **0.9558** | **84.8%** | QupKake 0.8009 [7], Uni-pKa ~0.831 [5], AcepKa ~0.856 [6], Starling ~0.955 |
| **Novartis Combined (Acid + Base) [1]** | **280** | **0.7252** | **0.5642** | **0.9009** | **0.9464** | **83.9%** | Uni-pKa ~0.842 [5], AcepKa ~0.856 [6], Starling ~0.919 |
| **Novartis Base Suite [1]** | **168** | **0.5863** | **0.4762** | **0.9236** | **0.9481** | **90.5%** | Uni-pKa 0.653 [5], AcepKa 0.770 [6], Starling 0.790, Marvin 1.145 |
| **Novartis Acid Suite [1]** | **112** | **0.8941** | **0.6962** | **0.8771** | **0.9380** | **74.1%** | AcepKa 0.970 [6], Uni-pKa 1.061 [5], Starling 1.083, Marvin 1.144 |
| **SAMPL6 Blind Challenge [2]** | **31** | **0.6527** (0.4377 DFT) | **0.4900** | **0.9412** | **0.9340** | **87.1%** | Uni-pKa 0.716 [5], MolGpKa 0.773 [9], AcepKa 0.900 [6], Starling 1.118 |
| **SAMPL7 Blind Challenge [3]** | **20** | **0.5657** | **0.4723** | **0.9420** | **0.8710** | **90.0%** | AcepKa 0.700 [6], Marvin 0.708, Starling 0.734, Uni-pKa 0.735 [5] |
| **SAMPL8 Blind Challenge [4]** | **25** | **0.7170** | **0.5060** | **0.8965** | **0.9713** | **88.0%** | Uni-pKa 0.878 [5], AcepKa 0.910 [6], Starling 1.142, Marvin 1.511 |
| **Core Development Holdout** | **550** | **0.7842** | **0.4802** | **0.9422** | **0.9733** | **88.7%** | Top 50% conformal confidence subset: **RMSE = 0.4142** |
| **5-Fold Bemis–Murcko Scaffold CV** | **6,694** | **1.0829** (1.0442 fw) | **0.6636** | **0.9019** | **0.9612** | **80.6%** | 1,344 unique frameworks (strict zero-scaffold-overlap CV) |

---

## 3. Discovery II: Empirical Ionization & Lipophilicity Drivers of Direct Reversible Inhibition, TDI & Test-Set Shift

Evaluating `44` macroscopic ionization and solvation descriptors across all `6,146` training compounds and `750` blind test compounds (using two-sided Mann–Whitney U tests, univariate and multivariable partial Spearman rank correlations ρ across the continuous DRC cohorts, within-scaffold fixed-effect demeaning, and Fisher's exact tests) established three empirical findings.

### 3.1 Direct Reversible Inhibition (−NADPH): Electrostatic Cation Preference in `CYP2D6` vs. Size- & Scaffold-Controlled Lipophilicity Associations in `CYP3A4`, `CYP2C9` & `CYP1A2`

**Table 2. Empirical pH 7.4 Ionization & Lipophilicity Profiles (Mean ± SD) of Potent (pIC₅₀ ≥ 5.5) vs. Inactive (pIC₅₀ ≤ 4.0) Direct Reversible Inhibitors**

| Isoform & Group (or Full-DRC Spearman ρ) | Basic pKₐ,base | Cationic f_pos (pH 7.4) | Neutral f_neutral (pH 7.4) | Anionic f_neg (pH 7.4) | Intrinsic cLogP | Effective log D₇.₄ | Asp301 Motif |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CYP2D6` Potent (≥ 5.5, n = 256)** | **6.81 ± 2.32** | **46.90% ± 40.37%** | 51.96% ± 40.29% | 1.15% ± 9.88% | 3.28 ± 1.11 | 2.60 ± 1.22 | **0.33 ± 0.40** |
| **`CYP2D6` Inactive (≤ 4.0, n = 129)** | **3.32 ± 2.00** | **4.06% ± 16.29%** | 90.36% ± 22.94% | 5.58% ± 17.49% | 3.45 ± 0.96 | 3.36 ± 1.05 | **0.03 ± 0.15** |
| **`CYP2D6` Δ (p_MW) · Full ρ (n = 1,493)** | **+3.49 (p=3.1e-32) · ρ=+0.36** | **+42.84% (p=1.5e-32) · ρ=+0.36** | **-38.41% (p=3.3e-22) · ρ=-0.29** | -4.43% (p=3.9e-15) · ρ=-0.18 | -0.17 (p=0.039) · ρ=+0.03 (NS) | **-0.76 (p=2.7e-09) · ρ=-0.09** | **+0.31 (p=5.2e-14) · ρ=+0.26** |
| **`CYP3A4` Potent (≥ 5.5, n = 176)** | 3.50 ± 2.11 | 4.74% ± 14.53% | 91.48% ± 17.81% | 3.78% ± 11.90% | **3.48 ± 0.80** | **3.41 ± 0.84** | 0.03 ± 0.13 |
| **`CYP3A4` Inactive (≤ 4.0, n = 944)** | 3.61 ± 2.52 | 10.47% ± 26.82% | 75.65% ± 38.78% | 13.88% ± 32.77% | **1.98 ± 1.02** | **1.45 ± 1.48** | 0.06 ± 0.22 |
| **`CYP3A4` Δ (p_MW) · Full ρ (n = 2,335)** | -0.11 (p=0.98) · ρ=+0.01 | -5.73% (p=0.27) · ρ=+0.05 | +15.83% (p=0.072) · ρ=+0.13 | -10.10% (p=0.80) · ρ=-0.10 | **+1.50 (p=9.3e-62) · ρ=+0.60** | **+1.96 (p=4.9e-66) · ρ=+0.63** | -0.04 (p=0.11) · ρ=-0.08 |
| **`CYP2C9` Potent (≥ 5.5, n = 140)** | 3.32 ± 1.97 | 1.76% ± 5.81% | 90.52% ± 23.90% | 7.71% ± 23.76% | **3.54 ± 0.89** | **3.38 ± 1.11** | 0.004 ± 0.04 |
| **`CYP2C9` Inactive (≤ 4.0, n = 259)** | 3.89 ± 2.53 | 11.01% ± 25.56% | 77.03% ± 37.12% | 11.96% ± 31.26% | **2.10 ± 0.93** | **1.63 ± 1.30** | 0.07 ± 0.22 |
| **`CYP2C9` Δ (p_MW) · Full ρ (n = 1,285)** | -0.57 (p=0.075) · ρ=-0.07 | -9.25% (p=0.20) · ρ=-0.06 | +13.49% (p=0.061) · ρ=+0.09 | -4.24% (p=0.41) · ρ=+0.03 (NS) | **+1.44 (p=9.7e-36) · ρ=+0.49** | **+1.76 (p=8.6e-37) · ρ=+0.50** | -0.06 (p=2.8e-04) · ρ=-0.16 |
| **`CYP1A2` Potent (≥ 5.5, n = 388)** | 3.92 ± 2.22 | 7.52% ± 21.55% | 91.12% ± 21.90% | 1.36% ± 5.96% | **3.26 ± 0.78** | **3.13 ± 1.10** | 0.03 ± 0.16 |
| **`CYP1A2` Inactive (≤ 4.0, n = 232)** | 3.99 ± 2.37 | 11.82% ± 27.74% | 81.79% ± 33.21% | 6.39% ± 21.94% | **2.54 ± 1.26** | **2.24 ± 1.53** | 0.05 ± 0.20 |
| **`CYP1A2` Δ (p_MW) · Full ρ (n = 1,412)** | -0.08 (p=0.83) · ρ=+0.01 | -4.30% (p=0.51) · ρ=+0.02 | +9.33% (p=0.13) · ρ=+0.02 | -5.02% (p=0.47) · ρ=-0.01 | **+0.71 (p=2.4e-21) · ρ=+0.23** | **+0.89 (p=2.7e-22) · ρ=+0.25** | -0.03 (p=0.015) · ρ=-0.06 |

- **1. `CYP2D6` Direct Reversible Potency Correlates Strongly With Basicity & Cationic Fraction, Even Controlling for Molecular Size:** Across all `1,493` `CYP2D6` DRC compounds, direct pIC₅₀ correlates positively with strongest basic pKₐ (Spearman ρ = +0.3577, p = 2.77 × 10⁻⁴⁶) and predicted cationic fraction at pH 7.4 (ρ = +0.3595, p = 8.88 × 10⁻⁴⁷). Controlling simultaneously for molecular weight (MW), heavy-atom count (HAC), and topological polar surface area (TPSA), the partial rank correlation remains highly significant (**partial ρ = +0.3106, p = 9.46 × 10⁻³⁵**). Comparing potent inhibitors (pIC₅₀ ≥ 5.5, n = 256) against inactive compounds (pIC₅₀ ≤ 4.0, n = 129), the mean cationic fraction is **46.90% ± 40.37% vs. 4.06% ± 16.29%** (an 11.55-fold ratio of group mean cationic fractions, Mann–Whitney p = 1.54 × 10⁻³²), while intrinsic cLogP shows no positive correlation across the dataset (ρ = +0.0342, p = 0.19) and log D₇.₄ is lower in potent `CYP2D6` inhibitors (2.60 ± 1.22 vs. 3.36 ± 1.05, p = 2.66 × 10⁻⁹) because protonation reduces effective octanol-water partitioning.
- **2. `CYP3A4`, `CYP2C9`, and `CYP1A2` Direct Potency Associates Strongly With Lipophilicity (log D₇.₄), Controlling for Size, Polarity & Chemical Scaffold:** Unlike `CYP2D6`, direct reversible inhibition potency in `CYP3A4` (n = 2,335), `CYP2C9` (n = 1,285), and `CYP1A2` (n = 1,412) is strongly associated with effective lipophilicity at pH 7.4: univariate Spearman ρ(log D₇.₄, pIC₅₀) is **+0.6346 (p = 1.99 × 10⁻²⁶³) in `CYP3A4`**, **+0.5008 (p = 1.64 × 10⁻⁸²) in `CYP2C9`**, and **+0.2458 (p = 7.06 × 10⁻²¹) in `CYP1A2`**. Because observational correlations with log D₇.₄ can be confounded by molecular size, polarity, or shared chemical scaffolds, we performed two multivariable controls:
  - *Controlling Simultaneously for Molecular Weight, Heavy-Atom Count, and TPSA:* Partial Spearman ρ(log D₇.₄, pIC₅₀ | MW, HAC, TPSA) remains **+0.5270 (p = 4.80 × 10⁻¹⁶⁷) in `CYP3A4`**, **+0.4566 (p = 3.42 × 10⁻⁶⁷) in `CYP2C9`**, and **+0.2821 (p = 3.03 × 10⁻²⁷) in `CYP1A2`**.
  - *Within-Scaffold Fixed-Effect Analysis (Matched Congeneric Analogs Sharing Identical Bemis–Murcko Scaffolds):* Demeaning both log D₇.₄ and pIC₅₀ within each multi-member Bemis–Murcko scaffold confirms that increasing lipophilicity within the same scaffold family strongly tracks potency in **`CYP3A4` (within-scaffold ρ = +0.6339, p = 3.59 × 10⁻⁴⁰ across n = 345 analogs in 105 scaffolds)** and **`CYP2C9` (within-scaffold ρ = +0.4320, p = 1.94 × 10⁻³ across n = 49 analogs in 19 scaffolds)**. Notably, although crystallographic literature often highlights anionic interactions with `Arg108` in `CYP2C9`, in this dataset the anionic fraction at pH 7.4 does not significantly differ between potent and inactive `CYP2C9` compounds (7.71% ± 23.76% vs. 11.96% ± 31.26%, Mann–Whitney p = 0.41; full-dataset ρ = +0.0277, p = 0.32).

### 3.2 Time-Dependent Inhibition (TDI): Controlling for Baseline Binding, Assay Definition & Enzymological Scope

A critical methodological trap in analyzing screening TDI data is that the binary `is_TDI` label is defined by an IC₅₀ shift between the +NADPH and −NADPH curves (ΔpIC₅₀ ≥ 0.301, or ≥ 2-fold shift), and includes "inferred-baseline" positives whose −NADPH curve is censored at the inactive floor (pIC₅₀,direct ≤ 4.0, with pIC₅₀,TDI > 4.301), while `74.8%` of `CYP3A4` TDI negatives are complete non-binders (`Section 5`). From an enzymological standpoint, two qualifications are essential:
1. **Assay-Defined TDI vs. Confirmed Covalent MBI:** A ≥2-fold +NADPH IC₅₀ shift establishes *Time-Dependent Inhibition (TDI)* under the screening protocol, which can arise from irreversible covalent heme/apoprotein adduction, quasi-irreversible metabolic-intermediate complex (MIC) formation, or formation of a more potent reversible metabolite; definitive proof of covalent mechanism-based inactivation (MBI) requires time- and concentration-dependent inactivation kinetics (`k_inact` and `K_I`), dilution/dialysis recovery assays, or metabolite adduct trapping.
2. **Controlling for Baseline Binding & Censoring Circularity:** Comparing all labeled TDI positives against all TDI negatives without stratification introduces a definitional artifact in `CYP2D6` (where inferred-baseline positives are forced to have pIC₅₀,direct ≤ 4.0) and a baseline-binding confound in `CYP3A4` (where `2,109` of `2,820` negatives are non-binders).

To isolate genuine chemical drivers of TDI from baseline binding and censoring artifacts, `Table 3` compares TDI Positives vs. TDI Negatives both across all labeled rows and restricted strictly to **Measurable-Baseline Binders Only (pIC₅₀,direct > 4.0 in both groups)**:

**Table 3. Ionization Profiles (Mean ± SD) and Fisher's Exact Test Warhead Enrichment in TDI Positives vs. Negatives (All Labeled vs. Measurable-Baseline Control)**

| Isoform & Cohort Comparison | Direct pIC₅₀ (−NADPH) | Basic pKₐ,base (Mean ± SD) | Cationic f_pos (pH 7.4) | Neutral f_neutral (pH 7.4) | Effective log D₇.₄ | Statistically Enriched Reactive Warheads (Pos vs. Neg, Odds Ratio, Fisher p) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`CYP2D6` All Labeled: Pos (n=324) vs. Neg (n=1,173)** | 4.50±0.65 vs 4.86±0.77 *(definitional\*)* | 5.18±2.61 vs 5.61±2.73 (p_MW=5.5e-03) | 24.29%±36.8% vs 30.38%±39.4% (p_MW=0.017) | 72.63%±37.6% vs 67.34%±39.7% (p_MW=0.025) | 2.60±1.44 vs 2.64±1.44 (p_MW=0.71, NS) | **`1,3-Benzodioxole`:** 3.70% vs 0.77% (OR=4.97, p=3.3e-04); **`Furan`:** 5.86% vs 2.39% (OR=2.55, p=3.4e-03); **`Pyrrole/Indole`:** 10.19% vs 5.88% (OR=1.81, p=8.7e-03); **`Thiophene`:** 9.26% vs 5.80% (OR=1.66, p=0.031) |
| **`CYP2D6` Measurable Baseline (pIC₅₀,dir > 4.0): Pos (n=273) vs. Neg (n=1,091)** | 4.75±0.53 vs 5.04±0.66 (p_MW=8.5e-13) | 5.47±2.61 vs 5.79±2.69 (p_MW=0.045, p_t=0.074) | 27.51%±38.3% vs 32.48%±39.9% (p_MW=0.054) | 69.72%±38.8% vs 65.50%±40.1% (p_MW=0.064) | 2.54±1.45 vs 2.56±1.44 (p_MW=0.87, NS) | **`1,3-Benzodioxole`:** 4.03% vs 0.64% (**OR=6.50, p=1.4e-04**); **`Furan`:** 6.23% vs 2.38% (**OR=2.72, p=2.9e-03**); **`Pyrrole/Indole`:** 10.26% vs 5.41% (**OR=2.00, p=5.3e-03**); **`Thiophene`:** 9.52% vs 5.50% (**OR=1.81, p=0.018**) |
| **`CYP3A4` All Labeled: Pos (n=764) vs. Neg (n=2,820)** | 4.70±0.62 vs 3.80±0.91 *(confounded\*\*)* | 3.76±2.18 vs 3.96±2.49 (p_MW=0.35, NS) | 7.78%±21.9% vs 12.17%±29.2% (p_MW=0.53) | 88.55%±25.4% vs 81.44%±34.7% (p_MW=0.16) | 2.82±0.99 vs 2.66±1.63 (p_MW=0.067) | Unstratified comparison is confounded by 2,109 non-binders in Neg (which inflate apparent cyclopropyl OR=2.08); see Measurable-Baseline row below |
| **`CYP3A4` Measurable Baseline (pIC₅₀,dir > 4.0): Pos (n=680) vs. Neg (n=711)** | 4.82±0.53 vs 4.87±0.56 (p_MW=0.065, NS) | 3.65±2.11 vs 3.33±2.19 (p_MW=7.3e-04) | 6.62%±20.0% vs 5.30%±18.1% (p_MW=2.4e-03) | 89.64%±24.0% vs 89.12%±25.3% (p_MW=0.26, NS) | 2.91±0.94 vs 2.95±1.02 (p_MW=0.43, NS) | **`1,3-Benzodioxole`:** 2.65% vs 0.56% (**OR=4.81, p=2.0e-03**); **`Pyrrole/Indole`:** 6.91% vs 3.66% (**OR=1.96, p=7.8e-03**) *[Cyclopropyl 10.4% vs 9.7%, p=0.66 NS]* |

**Key Scientific Takeaways From the Baseline-Controlled TDI Audit (`Table 3`):**
1. Once we control for baseline binding (pIC₅₀,direct > 4.0), bulk physicochemical properties (log D₇.₄ and f_neutral,7.4) are virtually indistinguishable between TDI positives and active reversible binders in both `CYP2D6` and `CYP3A4`, and the downward shift in `CYP2D6` basic pKₐ (5.47 ± 2.61 vs. 5.79 ± 2.69, p_MW = 0.045, Welch p_t = 0.074) and cationic fraction (27.51% vs. 32.48%, p_MW = 0.054) is modest and borderline.
2. In contrast, specific oxidizable heteroarene and methylenedioxyphenyl substructures show highly significant enrichment among TDI positives even within matched active binders (pIC₅₀,direct > 4.0): **`1,3-benzodioxoles`** (OR = 6.50, p = 1.43 × 10⁻⁴ in `CYP2D6`; OR = 4.81, p = 2.02 × 10⁻³ in `CYP3A4`, classically oxidized at the methylenedioxy carbon to form carbene–iron MICs [10]) and **`pyrrole/indole` rings** (OR = 2.00, p = 5.31 × 10⁻³ in `CYP2D6`; OR = 1.96, p = 7.82 × 10⁻³ in `CYP3A4`), along with **`furans`** (OR = 2.72, p = 2.88 × 10⁻³) and **`thiophenes`** (OR = 1.81, p = 0.018) in `CYP2D6` (known precursors to reactive epoxide, cis-enedial, and S-oxide intermediates [10]). While warhead enrichment alone does not establish atom-level covalent adduction without metabolite trapping, combining explicit warhead indicators with site-of-metabolism (SOM) and `GFN2-xTB` Fukui reactivity descriptors provides a strong mechanistic signal for TDI classification.

### 3.3 Discovery of the Train-to-Test Ionization Shift (`6,146` Train vs. `750` Blind Test)

Because the `750`-compound blind test set was constructed by analog expansion around top active hits of `CYP1A2`, `CYP2C9`, and `CYP3A4` (excluding `CYP2D6`), our ionization audit revealed a statistically significant reduction in basicity and cationic fraction between the training and test cohorts:

**Table 4. Empirical Ionization & Solvation Distribution Shift (Mean ± SD) Between the Training Cohort (N = 6,146) and Blind Test Cohort (N = 750)**

| Physicochemical / Ionization Property | Training Mean ± SD (N = 6,146) | Blind Test Mean ± SD (N = 750) | Test − Train Shift (Welch p / MW p) | Impact on Generalization |
| :--- | :---: | :---: | :---: | :--- |
| **Cationic Fraction at pH 7.4 (f_pos,7.4)** | **13.73% ± 30.10%** | **7.21% ± 20.73%** | **-6.51% (-47.5% rel, p_t=3.6e-14, p_MW=1.7e-04)** | Nearly 2-fold lower mean cationic population in blind test set |
| **Strongest Basic pKₐ (pKₐ,base)** | 4.15 ± 2.53 (med 3.63) | 3.76 ± 2.25 (med 3.37) | **-0.39 pKₐ (p_t=9.1e-06, p_MW=7.1e-04)** | Lower prevalence of strongly basic amines in the test set |
| **`CYP2D6 Asp301` Cationic Motif Score** | 0.0809 ± 0.2467 | 0.0374 ± 0.1733 | **-0.0435 (-53.8% rel, p_t=1.1e-09, p_MW=2.7e-06)** | Fewer canonical `Asp301` cationic-aromatic pharmacophores |
| **Neutral Fraction at pH 7.4 (f_neutral,7.4)** | 81.07% ± 34.15% | 86.65% ± 28.26% | **+5.58% (p_t=7.6e-07, p_MW=0.73)** | Higher mean uncharged fraction across test molecules |
| **Effective Lipophilicity at pH 7.4 (log D₇.₄)** | 2.65 ± 1.45 (med 2.76) | 2.62 ± 1.28 (med 2.77) | -0.03 log units (p_t=0.53, p_MW=0.64, NS) | Matched log D₇.₄ distribution between train and test cohorts |

---

## 4. Discovery III: Bridging the `73.3%` Single-Isoform Sparsity & Modeling the 15.9× Censored-Floor Noise

Auditing the four OpenADMET training tables against the blind test set exposed two structural properties of the experimental assay that directly govern regression accuracy:

- **1. Bridging `73.31%` Single-Isoform DRC Sparsity via the `17,504`-Row Single-Concentration Screen:** While the `750` blind test molecules are evaluated densely (`100%`) across all four isoforms, out of `4,905` training compounds with full dose-response curves (DRCs), **`3,596` (`73.31%`) were measured in only 1 isoform**, `1,039` (`21.18%`) in 2 isoforms, `229` (`4.67%`) in 3 isoforms, and **only `41` (`0.84%`) across all 4 isoforms**. However, `4,376` training compounds were screened across all 4 isoforms simultaneously at 49.5 μM (+NADPH single-concentration log₂FC, `17,504` rows). Across overlapping compounds, −log₂FC at 49.5 μM correlates with full DRC pIC₅₀ at **Spearman ρ = 0.8283–0.9525** via the Hill–Langmuir single-point isotherm:

> **Equation 2 (Hill–Langmuir Single-Point Isotherm at [I] = 49.5 μM, p[I] = 4.3054):**
> ```text
> v_i / v_0 = 2^(log2FC) ≈ 1 / [1 + ([I] / IC50)^h]
> ```

- **2. The 4.65× to 15.90× Bayesian Credible-Interval Explosion Below pIC₅₀ = 4.0 (100 μM Ceiling):** Compounds that fail to reach 50% inhibition at the 100 μM top assay concentration (pIC₅₀ ≤ 4.0) have extrapolated curve fits whose 95% Bayesian credible intervals (ΔCI₉₅ = conf_high − conf_low) explode by up to **15.90-fold** compared to active inhibitors (pIC₅₀ > 5.0). In `CYP3A4`, **`40.43%` of all training compounds** sit in this censored floor with a median ΔCI₉₅ of **2.146 log₁₀ units** (>140-fold IC₅₀ uncertainty) versus **0.135 log₁₀ units** in the active regime. Weighting training losses by inverse credible-interval width (`w_i = (ΔCI₉₅,i + 0.25)⁻¹`) prevents noisy floor extrapolations from corrupting active-regime predictions.

**Table 5. Multi-Assay Concordance, Paired ±NADPH Potency Shifts, and Censored-Floor Credible-Interval Inflation Across Isoforms**

| Isoform | DRC Count (N) | 49.5 μM Single-Conc Spearman (Dir / TDI) | DRC >2× NADPH Shift (%) | Floor Regime (pIC₅₀ ≤ 4.0) Count (%) | Floor Median ΔCI₉₅ Width | Active (pIC₅₀ > 5.0) Median ΔCI₉₅ | Floor / Active CI Inflation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CYP3A4`** | 2,335 | **0.9362 / 0.9525** (n = 1,496) | **1,095 / 2,334 (46.92%)** | **944 (40.43%)** | **2.146 log₁₀** | **0.135 log₁₀** (n = 481) | **15.90× wider** |
| **`CYP2D6`** | 1,493 | **0.8283 / 0.8532** (n = 1,069) | **495 / 1,493 (33.15%)** | **129 (8.64%)** | **2.592 log₁₀** | **0.216 log₁₀** (n = 546) | **12.00× wider** |
| **`CYP1A2`** | 1,412 | **0.8597 / 0.8617** (n = 1,158) | 23 / 1,412 (1.63%) | 232 (16.43%) | 2.085 log₁₀ | 0.240 log₁₀ (n = 827) | **8.69× wider** |
| **`CYP2C9`** | 1,285 | **0.8914 / 0.8856** (n = 1,004) | 38 / 1,285 (2.96%) | 259 (20.16%) | 1.209 log₁₀ | 0.260 log₁₀ (n = 360) | **4.65× wider** |

---

## 5. Discovery IV: Decomposing the Five Experimental Sub-Populations of the TDI Assay (`CYP2D6` & `CYP3A4`)

`Table 5` explains why the binary TDI track focuses on `CYP2D6` and `CYP3A4`: **`33.15%` of `CYP2D6`** and **`46.92%` of `CYP3A4`** DRC compounds exhibit a >2-fold potency shift with NADPH, compared to just `1.63%` in `CYP1A2` and `2.96%` in `CYP2C9`. Reconstructing the exact experimental calling rules across `TRAIN_TDI.csv`, `TRAIN_inhibition.csv`, `TRAIN_Emax.csv`, and `single-concentration-TRAIN.csv` shows that the binary `is_TDI` label is a composite of **five distinct sub-populations**:

**Table 6. Exact Decomposition of Binary Time-Dependent Inhibition (`is_TDI`) Labels Across Experimental Regimes**

| Experimental TDI Sub-Population | Assay Calling Rule & Physical Meaning | `CYP2D6` (N = 1,497) | `CYP3A4` (N = 3,584) | Key Modeling Insight |
| :--- | :--- | :---: | :---: | :--- |
| **1. Measurable-Baseline Positives (`is_TDI = 1`)** | Direct pIC₅₀ > 4.0 and +NADPH shift ≥ 2-fold (ΔpIC₅₀ ≥ 0.301) with non-overlapping CIs & valid E_max | **273 (18.24% of all / 84.3% of Pos)** | **680 (18.97% of all / 89.0% of Pos)** | Governed by continuous shift ΔpIC₅₀ = pIC₅₀,TDI − pIC₅₀,direct |
| **2. Inferred-Baseline Positives (`is_TDI = 1`)** | Direct pIC₅₀ ≤ 4.0 (inactive at 100 μM without NADPH), but +NADPH pIC₅₀,TDI > 4.301 (< 50 μM) | **51 (3.41% of all / 15.7% of Pos)** | **84 (2.34% of all / 11.0% of Pos)** | By rule, forced to have pIC₅₀,direct ≤ 4.0 (silent without NADPH) |
| **Total TDI Positives (`is_TDI = 1`)** | Sum of Measurable + Inferred TDI Positives | **324 (21.64%)** | **764 (21.32%)** | Nearly identical ~21.5% positive rate in both isoforms |
| **3. Measurable DRC Negatives (`is_TDI = 0`)** | Direct pIC₅₀ > 4.0; active reversible binder with < 2-fold shift or overlapping CIs | **1,091 (72.88%)** | **711 (19.84%)** | **93.0% of `CYP2D6` negatives are active reversible inhibitors** |
| **4. Assigned DRC Floor Negatives (`is_TDI = 0`)** | Inactive in both −NADPH and +NADPH DRC arms (pIC₅₀ ≤ 4.0 in both conditions) | **78 (5.21%)** | **860 (24.00%)** | Complete non-binders in both DRC arms |
| **5. Single-Conc Screen Negatives (`is_TDI = 0`)** | Confirmed inactive in 49.5 μM +NADPH single-concentration screen (no DRC required) | **4 (0.27%)** | **1,249 (34.85%)** | **74.8% of `CYP3A4` negatives are non-binders** (Floor + Single-Conc) |
| **Total TDI Negatives (`is_TDI = 0`)** | Sum of all three Negative regimes | **1,173 (78.36%)** | **2,820 (78.68%)** | Only 259 compounds carry TDI labels in both `CYP2D6` and `CYP3A4` |

**Why Direct Binary Classification Struggles on TDI:** In `CYP2D6`, **`93.0%` (`1,091 / 1,173`) of TDI negatives are active reversible binders** that do not meet the 2-fold shift criterion, whereas in `CYP3A4`, **`74.8%` (`2,109 / 2,820`) of TDI negatives are complete non-binders**. A binary classifier trained directly on `CYP3A4_is_TDI` therefore conflates general `CYP3A4` affinity with time-dependent inactivation (as shown in `Table 3`). Furthermore, because `15.7%` of `CYP2D6` positives (`51 / 324`) and `11.0%` of `CYP3A4` positives (`84 / 764`) belong to the inferred-baseline regime (pIC₅₀,direct ≤ 4.0, pIC₅₀,TDI > 4.301), modeling the clamped effective potency shift:

> **Equation 3 (Clamped Effective TDI Potency Shift Across Measurable & Inferred-Baseline Regimes):**
> ```text
> ΔpIC50_eff = pIC50_TDI − max(pIC50_direct, 4.0)
> ```

jointly with baseline reversible affinity (pIC₅₀,direct), holo-enzyme affinity (pIC₅₀,TDI), E_max, and oxidizable warhead indicators directly aligns the model with the physical and statistical structure of the assay.

---

## 6. Rasayan Labs Five-Pillar Modeling Framework & Medicinal Chemistry Implications

To translate these four empirical findings into a unified predictive architecture for both OpenADMET Challenge tracks (`Track 1` Direct Reversible Inhibition and `Track 2` Time-Dependent Inhibition) without using proprietary external CYP datasets, we integrated five modeling pillars validated under strict out-of-fold scaffold and structural-cluster cross-validation:

**Table 7. Architecture of the Rasayan Labs Five-Pillar CYP450 Modeling Framework**

| Modeling Pillar | Core Physical & Statistical Function | Model Roster, Descriptors & Supervision | Challenge Bottleneck Solved |
| :--- | :--- | :--- | :--- |
| **Pillar 1: Macroscopic pKₐ & Solvation Engine** | Computes pH 7.4 microstate populations (f_neutral, f_pos, f_dipos, f_neg), log D₇.₄, log S, and `Asp301`/`Arg108` motifs | Rasayan 3-Pillar pKₐ Engine (`RMSE = 0.7105` on `356` public transitions), `OpenFF NAGL` AM1-BCC charges & `86,794` Boltzmann microstates | Captures the 11.55× higher mean cationic fraction in potent `CYP2D6` inhibitors vs. log D₇.₄ drive in `1A2`/`2C9`/`3A4` |
| **Pillar 2: 3D Conformer, Pocket & Quantum Descriptors** | Encodes 3D conformer ensembles, pocket-conditioned pose mechanics, SOM reactivity & `GFN2-xTB` electronic shells | `GFN2-xTB` (`xtb`, polarizability, Fukui indices), `Boltz-2` (`15,603`-state population affinity), Compound-I (`Fe=O`) pose & `USR/USRCAT`/`Mordred 3D` | Provides 3D steric, active-site pocket, and electronic reactivity context alongside explicit warhead indicators |
| **Pillar 3: Multi-Scale Graph & Chemical Foundation Roster** | Combines 2D/3D message-passing neural networks, pretrained chemical foundation models, and multi-radius substructure kernels | `CheMeleon` (state-conditioned across `86,794` microstates), `ChemProp D-MPNN`, `MiniMol`, `Uni-Mol` 3D, `ECFP4/6`, `Avalon` & `RDKit 2D` across `6,896` compounds | Provides diverse multi-modal chemical representations conditioned on protonation states |
| **Pillar 4: Assay-Aware Multi-Task & Single-Conc Bridge** | Jointly learns across all 4 isoforms and bridges sparse DRCs using the dense 49.5 μM single-concentration screen & E_max | 4× pIC₅₀,direct, 4× pIC₅₀,TDI, 4× E_max, and `17,504` single-conc log₂FC rows with inverse CI-width weights (`(ΔCI₉₅ + 0.25)⁻¹`) | Resolves `73.31%` single-isoform DRC sparsity (**largest sequential error reduction**) and down-weights 15.9× floor noise |
| **Pillar 5: Mechanistic TDI Shift & Distribution Calibration** | Couples ΔpIC₅₀,eff shift reconstruction with reactive warhead/SOM features and accounts for the `-47.5%` test-set cationic shift | Simplex cross-decoupled ensembling, `4,000`-draw scaffold bootstrap audits, PAVA/S-curve tail calibration & dual-gate TDI probability | Separates reversible binders from TDI positives and calibrates predictions to the test distribution |

### 6.1 Architectural Integration, Feature Conditioning, Training Protocol & Ensembling

To ensure strict zero-leakage evaluation and prevent test-set overfitting across all `6,896` challenge molecules (`6,146` training and `750` blind test compounds), all feature transformations, dimensionality reductions, and specialist learners are trained across **8 out-of-scaffold cross-validation scopes** (`5` Bemis–Murcko `StrictFamily` scaffold folds plus `3` `StrictParent` structural-cluster splits):

1. **How Macroscopic pKₐ & Boltzmann Microstate Features Feed the Foundation & Graph Models:** For each molecule, we enumerate protomer and tautomer microstates (`86,794` states across the `6,896`-compound corpus) and compute Boltzmann free-energy weights `w_i ∝ exp(−ΔG_i(pH 7.4) / k_B T)` at physiological pH 7.4. Rather than feeding a single static neutral SMILES graph, pretrained foundation embeddings (`CheMeleon` 2,048-dim and `state_v1` 773-dim) are evaluated across all microstates and partitioned into charge-conditioned sub-ensembles—cationic (`q_i ≥ +0.5`), neutral (`|q_i| < 0.5`), and anionic (`q_i ≤ −0.5`)—paired with microstate shift vectors (`Boltzmann − Neutral`, `Cationic − Neutral`, `Anionic − Neutral`), macroscopic acid/base pKₐ values, Henderson–Hasselbalch ion fractions (`f_pos,7.4`, `f_neg,7.4`, `f_neutral,7.4`), and charge-penalized log D₇.₄.
2. **Multi-Paradigm Specialist Learners & Inverse Credible-Interval-Width Training:** Each modality block (2D fingerprints/graphs `ECFP4/6`/`Avalon`/`RDKit`/`ChemProp`/`MiniMol`, state-conditioned `CheMeleon` + pKₐ/Boltzmann microstates, 3D `Uni-Mol`/`Boltz-2` pocket/conformer & `GFN2-xTB` electronic descriptors, and out-of-fold 49.5 μM single-concentration multi-task bridge features) is projected via fold-internal SVD and fed into three complementary learner families: (i) **Block-Weighted Cosine-RBF Kernel Ridge and Pinball Quantile Regressors**, (ii) **Gradient-Boosted Decision Trees (`LightGBM`)** with interval and focal weighting, and (iii) **Residual SwiGLU Neural Networks**. To prevent the 4.65×–15.90× wider censored-floor extrapolations (pIC₅₀ ≤ 4.0, `Table 5`) from corrupting active-regime gradients, training losses are weighted by **inverse credible-interval width** `w_i = (ΔCI₉₅,i + 0.25)⁻¹`.
3. **Cross-Decoupled Ensembling, Monotone S-Curve Tail Calibration & Dual-Gate TDI Calling:** Out-of-fold specialist predictions across Pillars 1–4 are combined via simplex-constrained cross-decoupled ensembling (`Pillar 5`). For Track 1 (pIC₅₀ regression), a monotone Pool-Adjacent-Violators (PAVA) and S-curve boundary calibrator restores upper-tail dynamic range (pIC₅₀ ≥ 6.0) compressed by ensemble shrinkage while respecting the 100 μM (pIC₅₀ = 4.0) inactivity floor. For Track 2 (`is_TDI` binary classification), a dual-gate probability product combines the active holo-enzyme binding probability `Φ((ŷ_TDI − 4.301) / σ_a)` with the clamped potency-shift probability `Φ((Δŷ_eff − 0.301) / σ_Δ)` and reactive warhead/SOM indicators.

### 6.2 Quantitative Validation, Stepwise Pillar Ablation & Evaluation Regime Analysis

To test our hypotheses quantitatively, `Tables 8` and `9` report stepwise pillar ablations across all 8 blind out-of-scaffold cross-validation scopes alongside the official blind test leaderboard metrics for both Track 1 (Direct Reversible pIC₅₀ Regression) and Track 2 (Time-Dependent Inhibition Classification).

- **What the Track 1 Ablation (`Table 8`) Reveals About Assay Bridging vs. Physical Chemistry:** On unseen chemical scaffolds, a static single-task 2D fingerprint regressor breaks down (`Macro ST-RAE = 0.8307`, `Macro R² = 0.2666`, and `CYP2D6 R² = 0.0624`). Sequentially adding a 2D GNN (`0.7869`), Pillar 1 pKₐ/Boltzmann microstates (`0.7686`), and Pillars 2–3 3D/QM & foundation features (`0.7454`) yields a `-0.0853 ST-RAE` reduction on the sparse single-isoform DRC tables alone. However, **the single largest sequential improvement (`0.7454 → 0.5208 ST-RAE`, accounting for `64.0%` of the total `0.3512` sequential error reduction)** occurs at **Pillar 4 (Assay-Aware Multi-Task & 49.5 μM Single-Concentration Bridging with inverse CI-width weighting)**. This reflects a fundamental structural property of the dataset: because `73.31%` of training DRC compounds are measured in only one isoform whereas `4,376` compounds have dense 4-isoform 49.5 μM screens (`Section 4`), exploiting related experimental assay measurements across isoforms is essential. Crucially, Pillars 1–3 (`pKₐ` microstates, `3D/QM`, and foundation encoders) also serve as the input representations *driving* those `4,376`-compound multi-task bridge models, and Pillar 5 cross-decoupled ensembling + S-curve calibration reduces out-of-scaffold `Macro ST-RAE` further to **`0.4795`** (**`0.3867 MA-ST-RAE`**, **`0.6271 MA-MAE`**, **`0.6348 MA-R²`**, **`0.7854 MA-Spearman ρ`**, **`0.6063 MA-Kendall τ`** on the official blind test leaderboard).

**Table 8. Track 1 (Direct Reversible Inhibition pIC₅₀ Regression): Stepwise Pillar Ablation on 8-Scope Blind Out-of-Scaffold CV & Official Blind Test Leaderboard\***

| Model Stage / Isoform Breakdown | Eval Cohort | ST-RAE (↓) | MAE (↓) | R² (↑) | Spearman ρ (↑) | Kendall τ (↑) | `CYP2D6` ST-RAE / ρ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 0a: Static 2D Fingerprint (`RDKit`/`ECFP`)** | 8-Scope Blind CV (4-Iso Macro) | 0.8307 | 0.6395 | 0.2666 | 0.5028 | 0.3590 | 1.0449 / 0.2867 |
| **Baseline 0b: + 2D Graph Neural Network (`D-MPNN`)** | 8-Scope Blind CV (4-Iso Macro) | 0.7869 | 0.6093 | 0.3407 | 0.5670 | 0.4125 | 1.0452 / 0.3719 |
| **+ Pillar 1: Quantum pKₐ & pH 7.4 Boltzmann Microspecies** | 8-Scope Blind CV (4-Iso Macro) | 0.7686 | 0.5981 | 0.3557 | 0.5778 | 0.4234 | 1.0551 / 0.3599 |
| **+ Pillars 2 & 3: 3D Pocket/Pose, `GFN2-xTB` & Foundation Roster** | 8-Scope Blind CV (4-Iso Macro) | 0.7454 | 0.5927 | 0.3499 | 0.5763 | 0.4217 | 0.9901 / 0.3605 |
| **+ Pillar 4: Multi-Task 49.5 μM Assay Bridge & CI-Width Weighting** | 8-Scope Blind CV (4-Iso Macro) | 0.5208 | 0.4587 | 0.5931 | 0.8118 | 0.6395 | 0.9005 / 0.6835 |
| **+ Pillar 5: Full Ensemble & S-Curve Calibration (`v90–v92`)** | **8-Scope Blind CV (4-Iso Macro)** | **0.4795** | **0.4374** | **0.6203** | **0.8297** | **0.6586** | **0.8161 / 0.7231** |
| **Full 5-Pillar Engine — Macro-Average (All 4 Isoforms)** | **Official Blind Test Leaderboard\*** | **0.3867** | **0.6271** | **0.6348** | **0.7854** | **0.6063** | **Rank #8 Overall** |
| ↳ **`CYP1A2` Direct pIC₅₀ (Full 5-Pillar Engine)** | 8-Scope CV · Blind Test\* | 0.4696 · **0.3803** | 0.4712 · **0.6457** | 0.6010 · **0.6817** | 0.8169 · **0.8466** | 0.6404 · **0.6540** | Rank #8 Isoform |
| ↳ **`CYP2C9` Direct pIC₅₀ (Full 5-Pillar Engine)** | 8-Scope CV · Blind Test\* | 0.2716 · **0.3481** | 0.3368 · **0.3981** | 0.7296 · **0.7144** | 0.8819 · **0.8630** | 0.7109 · **0.6857** | Sub-0.40 Blind MAE |
| ↳ **`CYP2D6` Direct pIC₅₀ (Full 5-Pillar Engine)** | 8-Scope CV · Blind Test\* | 0.8161 · **0.4448** | 0.5329 · **0.9987** | 0.3970 · **0.4026** | 0.7231 · **0.5656** | 0.5555 · **0.3979** | Rank #8 Isoform |
| ↳ **`CYP3A4` Direct pIC₅₀ (Full 5-Pillar Engine)** | 8-Scope CV · Blind Test\* | 0.3608 · **0.3733** | 0.4085 · **0.4659** | 0.7537 · **0.7404** | 0.8968 · **0.8663** | 0.7275 · **0.6878** | 0.7404 Blind R² |

*\*Note on Challenge Evaluation Splits: The live OpenADMET Hugging Face leaderboard (`openadmet/cyp-challenge`, as of November 3, 2026) evaluates submissions on the official public blind test split (`50%` of the `750`-compound blind test cohort), whereas final post-close rankings evaluate all `750` compounds.*

**Table 9. Track 2 (Time-Dependent Inhibition Binary Classification): Stepwise Pillar Ablation on 8-Scope Out-of-Scaffold CV & Official Blind Test Leaderboard\***

| TDI Model Stage / Isoform Breakdown | Eval Cohort | MCC (↑) | F1 Score (↑) | Precision (↑) | Recall (↑) | AUROC (↑) | Accuracy (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 0: Static 2D Fingerprint & Graph Classifier** | 8-Scope Strict-Scaffold CV (Macro) | 0.0710 | 0.1590 | 0.1796 | 0.1483 | 0.6084 | 0.8375 |
| **+ Pillar 1: Quantum pKₐ & pH 7.4 Microstate Partition** | 8-Scope Strict-Scaffold CV (Macro) | 0.0783 | 0.1651 | 0.1904 | 0.1521 | 0.6164 | 0.8385 |
| **+ Pillars 2–4: 3D/QM SOM + 49.5 μM & ±NADPH Pop Bridge** | 8-Scope Strict-Scaffold CV (Macro) | 0.1675 | 0.2415 | 0.2734 | 0.2281 | 0.7060 | 0.8576 |
| **+ Pillar 5: Dual-Gate (pIC₅₀,TDI × ΔpIC₅₀,eff) & Warheads** | **8-Scope Strict-Scaffold CV (Macro)** | **0.2082** | **0.2630** | **0.3000** | **0.2475** | **0.7415** | **0.8615** |
| ↳ **`CYP2D6` TDI (Full 5-Pillar Dual-Gate Engine)** | Strict-Scaffold CV (`Analog CV`\*\*) | 0.1793 (**0.3324**\*\*) | 0.1820 | 0.2620 | 0.1413 | **0.7379** | 0.9090 |
| ↳ **`CYP3A4` TDI (Full 5-Pillar Dual-Gate Engine)** | Strict-Scaffold CV | 0.2370 | 0.3439 | 0.3379 | 0.3536 | **0.7450** | 0.8140 |
| **Full 5-Pillar Engine — Macro-Average (`Rank #1 Overall`)** | **Official Blind Test Leaderboard\*** | **0.5887 (#1)** | **0.6331 (#1)** | **0.8352 (#1)** | **0.5561** | — (Binary Sub) | **0.8849** |
| ↳ **`CYP2D6` TDI — Official Blind Test (`Rank #1 Overall`)** | **Official Blind Test Leaderboard\*** | **0.6319 (#1)** | **0.5872** | **1.0000 (100%)** | **0.4223** | — (Binary Sub) | **0.9592** |
| ↳ **`CYP3A4` TDI — Official Blind Test** | Official Blind Test Leaderboard\* | **0.5456** | **0.6789** | **0.6705** | **0.6898** | — (Binary Sub) | **0.8107** |

- **\*\*Why TDI Strict-Scaffold CV (`MCC = 0.2082`) Differs From the Blind Leaderboard (`MCC = 0.5887`):** This difference arises from three documented structural factors:
  1. *Zero-Scaffold-Overlap CV vs. Analog-Expansion Test Construction:* Our primary `8-Scope Strict-Scaffold CV` holds out entire Bemis–Murcko scaffold families (`StrictFamily`) so that zero training analogs exist for any query scaffold (`Macro MCC = 0.2082`, `Macro AUROC = 0.7415`). However, the official `750`-compound challenge test set was constructed by **analog expansion around active hit scaffolds**; when evaluated under our `SharedUnit-Parent` analog-expansion CV regime that mirrors this construction, `CYP2D6` TDI CV MCC nearly doubles from **`0.1793` to `0.3324`**.
  2. *Multi-Isoform Feature Completeness & Full-Training Ensemble:* Training rows suffer from `73.31%` single-isoform sparsity, whereas the blind test cohort is evaluated by the full 8-scope ensemble across densely populated (`100%`) 4-isoform predictions.
  3. *Precision vs. Recall Operating-Point Selection on `CYP2D6` TDI (`Precision = 1.000`, `Recall = 0.4223` on Public Split / `0.2555` on Full Test):* Because the OpenADMET Track 2 leaderboard ranks binary predictions by **Macro MCC**, we selected a high-specificity probability cutoff `τ` on our continuous `CYP2D6` dual-gate score (`AUROC = 0.7379`) to eliminate false positives in the lower-basicity blind test set (`-47.5%` lower mean cationic fraction, `Table 4`). While this competition-optimal cutoff achieves `100.0%` precision (`Rank #1 Worldwide` on both the 50% public split and the full 750-compound final evaluation), its conservative recall leaves false negatives that would be undesirable in early safety triage. In production deployment (`Section 6.4`), lowering `τ` along the same continuous dual-gate probability curve provides a **high-sensitivity safety-screening mode (`Recall ≥ 0.75–0.80`)** to flag candidate TDI liabilities for experimental ±NADPH confirmation.

> **Official Post-Close Final Leaderboard Summary (Full `750`-Compound Blind Test Set ± Bootstrap SD Across All 3 Challenge Tracks):**
> Following challenge close, OpenADMET published the final full-dataset evaluations (`N = 750` blind test compounds with bootstrap standard deviations and statistical significance tiers):
> - **Track 2 — Time-Dependent Inhibition Classification (`62` Teams): `Rank #1 Worldwide` (`Significance: Tier 1`)** with **`MA-MCC = 0.4980 ± 0.0372`**, **`MA-Precision = 0.8362 ± 0.0160` (#1)**, **`MA-Accuracy = 0.8738 ± 0.0083` (#1)**, and **`MA-F1 = 0.5285 ± 0.0438` (#1)**. By isoform: **`CYP2D6` TDI finished `Rank #1 Worldwide`** (`MCC = 0.4899 ± 0.0657`, `Precision = 1.0000 ± 0.0000 [100%]`, `Accuracy = 0.9560 ± 0.0076`, `Recall = 0.2555 ± 0.0650`, `F1 = 0.4027 ± 0.0831`), and **`CYP3A4` TDI finished `Rank #6 Overall` (`#5` among teams using strictly public data)** (`MCC = 0.5061 ± 0.0346`, `Precision = 0.6725 ± 0.0319`, `Recall = 0.6380 ± 0.0319`, `F1 = 0.6543 ± 0.0262`, `Accuracy = 0.7917 ± 0.0148`).
> - **Track 1 — Direct Reversible Inhibition pIC₅₀ Regression (`103` Teams): `Rank #8 Overall` (`#6` among teams using strictly public data)** with **`MA-ST-RAE = 0.3911 ± 0.0110`**, **`MA-MAE = 0.6418 ± 0.0085`**, **`MA-R² = 0.6433 ± 0.0110`** (up from `0.6348` on the 50% public split), **`MA-Spearman ρ = 0.7945 ± 0.0089`** (up from `0.7854`), and **`MA-Kendall τ = 0.6140 ± 0.0082`** (up from `0.6063`). By isoform: **`CYP1A2` finished `Rank #6 Overall` (`#4` public-data)** (`ST-RAE = 0.3661 ± 0.0204`, `MAE = 0.6379 ± 0.0182`, `R² = 0.6820 ± 0.0182`, `ρ = 0.8450 ± 0.0125`); **`CYP2D6` finished `Rank #7 Overall` (`#5` public-data)** (`ST-RAE = 0.4796 ± 0.0235`, `MAE = 1.0077 ± 0.0184`, `R² = 0.4252 ± 0.0298`, `ρ = 0.5919 ± 0.0285`); **`CYP3A4` finished `Rank #18`** (`ST-RAE = 0.3583 ± 0.0180`, `MAE = 0.4685 ± 0.0144`, `R² = 0.7588 ± 0.0133`, `ρ = 0.8771 ± 0.0100`); and **`CYP2C9` finished `Rank #20`** (`ST-RAE = 0.3604 ± 0.0205`, `MAE = 0.4530 ± 0.0155`, `R² = 0.7073 ± 0.0188`, `ρ = 0.8638 ± 0.0122`).
> - **Track 3 — `CYP3A4` Ligand-Bound Protein Structure Prediction (`38` Teams): `Significance: Tier 1` (`Rank #11`, statistically tied in the top significance tier)** using our `Boltz-2` + pocket-conditioned pose refinement pipeline (`LDDT-PLI = 0.4246 ± 0.0379`, `BiSyRMSD = 3.7174 ± 0.3270 Å`, `LDDT-LP = 0.8070 ± 0.0175`, `Coverage = 1.000`, `Open Code = Yes`).

### 6.3 Out-of-Scaffold Chemical-Unit Bootstrap & 3-Stratum S-Curve Calibration Audit

To verify that Pillar 5 ensembling and monotone S-curve tail calibration improve out-of-scaffold accuracy without degrading any potency regime, we performed a chemical-unit bootstrap audit (`500` chemical-scaffold cluster bootstrap replicates across each of the `8` blind scopes = `4,000` bootstrap draws per isoform) comparing the calibrated final release (`v90`) against the pre-calibrated anchor (`v89`). `Table 10` reports the exact out-of-scaffold bootstrap win probabilities (**`90.4%` to `98.6%`**) and the error reductions across all three potency strata: Censored Floor (pIC₅₀ ≤ 4.05), Mid-Range (4.05 < pIC₅₀ < 6.0), and High-Potency Tail (pIC₅₀ ≥ 6.0).

**Table 10. Chemical-Unit Bootstrap Generalization & 3-Stratum S-Curve Calibration Audit (`4,000` Out-of-Scaffold Scope-Bootstrap Draws per Isoform)**

| Isoform | Out-of-Scaffold Bootstrap Win Rate | ΔST-RAE (Blind Direct) | ΔST-RAE (4,376 Multi-Assay) | ΔSpearman ρ (Blind Direct) | Floor Stratum ΔError (≤ 4.05) | Mid Stratum ΔError (4.05–6.0) | High-Potency Tail ΔError (≥ 6.0) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CYP1A2`** | **90.4%** (93.8% rk-lock) | **-0.00595** | -0.00380 | +0.00348 | -0.00332 | -0.00099 | **-0.01206** |
| **`CYP2C9`** | **96.7%** (96.9% x-dec) | **-0.00770** | -0.00483 | +0.00449 | -0.00394 | -0.00152 | **-0.01367** |
| **`CYP2D6`** | **91.7%** | **-0.01377** | -0.00423 | +0.00524 | -0.00134 | -0.00556 | **-0.01169** |
| **`CYP3A4`** | **98.6%** | **-0.00407** | -0.00330 | +0.00025 | -0.00029 | -0.00279 | **-0.01159** |

### 6.4 Practical Implications for Medicinal Chemistry & the Rasayan Platform ([rasayan.ai](https://rasayan.ai))

1. **Why Single-Parameter pKₐ Heuristics Are Insufficient for `CYP2D6` Optimization:** While lowering basic pKₐ reduces predicted cationic fraction at pH 7.4 and is associated with lower reversible `CYP2D6` potency (`Table 2`, ρ = +0.36, partial ρ = +0.31), `Table 3` demonstrates that `CYP2D6` TDI-positive compounds span a slightly lower mean basic pKₐ (`5.18 ± 2.61` overall; `5.47 ± 2.61` among measurable binders) and are significantly enriched for oxidizable substructures (`1,3-benzodioxoles`, `furans`, `pyrroles/indoles`, `thiophenes`). Consequently, attenuating pKₐ alone does not eliminate `CYP2D6` liability if metabolically labile warheads remain present; lead optimization requires joint multi-parameter prediction of both reversible affinity and +NADPH time-dependent shift.
2. **Detecting Inferred-Baseline TDI Compounds & Dual-Threshold Screening (`High-Recall` vs. `High-Precision`):** Because `11.0%–15.7%` of all TDI positives exhibit no measurable reversible inhibition at 100 μM without NADPH (pIC₅₀,direct ≤ 4.0, `Table 6`), early discovery cascades that rely solely on −NADPH direct inhibition screens will miss these time-dependent inhibitors. Furthermore, because high-precision competition thresholds (`100%` precision, `42.2%` recall on `CYP2D6` TDI) miss over half of true positives, medicinal chemistry teams using the **[Rasayan Labs](https://rasayan.ai)** platform can query both a **High-Recall Triage Threshold (`Recall ≥ 0.75–0.80`)** to flag potential TDI liabilities for experimental follow-up and a **High-Precision Alert Threshold** coupled with predicted SOM/warhead attribution.

---

## 7. References

1. Fraczkiewicz, R. *et al.* "In Silico Prediction of Ionization." *Comprehensive Medicinal Chemistry III*, Elsevier, **2017**; and Novartis macroscopic pKₐ benchmark dataset (`N = 280`).
2. Işık, M. *et al.* "Overview of the SAMPL6 pKₐ Challenge: Evaluating small molecule microscopic and macroscopic pKₐ predictions." *J. Comput.-Aided Mol. Des.* **2021**, *35*, 131–166.
3. Bergazin, T. D. *et al.* "Evaluation of log P, pKₐ, and log D predictions from the SAMPL7 blind challenge." *J. Comput.-Aided Mol. Des.* **2021**, *35*, 771–802.
4. SAMPL8 Consortium. "SAMPL8 Blind Challenge Macroscopic pKₐ Benchmark." *J. Comput.-Aided Mol. Des.* **2022–2023**.
5. Luo, S. *et al.* "Uni-pKa: An Equivariant 3D Molecular Representation Learning Framework for Acid Dissociation Constant Prediction." *J. Chem. Theory Comput.* **2024**.
6. Wang, Y. *et al.* "AcepKa: Accurate Macroscopic and Microscopic pKₐ Prediction via Graph Neural Networks." *J. Chem. Inf. Model.* **2024**.
7. Abarbanel, O. D.; Hutchison, G. R. "QupKake: Combining Machine Learning and Quantum Chemistry for Micro-pKₐ Prediction." *J. Chem. Theory Comput.* **2024**, *20*, 6946–6956.
8. Bannwarth, C.; Ehlert, S.; Grimme, S. "GFN2-xTB—An Accurate and Broadly Parametrized Self-Consistent Tight-Binding Quantum Chemical Method." *J. Chem. Theory Comput.* **2019**, *15*, 1652–1671.
9. Pan, X. *et al.* "MolGpKa: A Web Server for Small Molecule pKₐ Prediction Using a Graph-Convolutional Neural Network." *J. Chem. Inf. Model.* **2021**, *61*, 3159–3165.
10. Orr, S. T. M. *et al.* "Mechanism-Based Inactivation (MBI) of Cytochrome P450 Enzymes: Structure–Activity Relationships and Discovery Strategies To Mitigate Drug–Drug Interaction Risks." *J. Med. Chem.* **2012**, *55*, 4896–4933.
