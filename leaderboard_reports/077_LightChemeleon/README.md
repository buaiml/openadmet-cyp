# OpenADMET CYP Inhibition Blind Challenge — Method Report

**Leaderboard name / alias:** LightChemeleon  
**Tracks submitted:**  
- Direct inhibition prediction (regression)  
- Time-dependent inhibition (TDI) classification  
- Structure prediction (pose)  

**Proprietary data:** No  
**Code:** Available on request 

---

## 1. Overview

We developed fully public-data pipelines for all three OpenADMET CYP tracks using the official Hugging Face dataset `openadmet/cyp-challenge-train-test`. Work was implemented primarily in Google Colab notebooks:

- Regression and TDI: fingerprint/descriptor gradient boosting, Chemprop message-passing models with CheMeleon-style pretraining, scaffold-aware validation, and ST-RAE / MCC evaluation.
- Structure: Boltz-2 co-folding of CYP3A4 + heme (CCD `HEM`) + ligand SMILES, with a fast AutoDock Vina docking fallback that produced a complete 20-complex submission.

No external labeled CYP inhibition datasets beyond the official challenge pack were used in the submitted models.

---

## 2. Data

| Resource | Role |
|----------|------|
| `cyp-challenge-TRAIN_inhibition.csv` | Direct-inhibition pIC50 labels and confidence bounds |
| `cyp-challenge-TRAIN_TDI.csv` | Binary TDI labels for CYP2D6 and CYP3A4 |
| `cyp-challenge-TEST-BLINDED.csv` | Blinded test SMILES / molecule names (regression & TDI) |
| `cyp-challenge-TEST-BLINDED_structures.csv` | 20 ligands for the structure track |

Molecules were validated with RDKit; invalid SMILES were excluded from training. For regression, rows with at least one isoform label were retained (~4905 compounds). TDI training used the official TDI table (~6145 rows; isoform-wise labeled subsets ~1497 for 2D6 and ~3584 for 3A4, ~21% positive each).

---

## 3. Direct inhibition (regression)

### 3.1 Features
- **Morgan fingerprints** (ECFP-like, radius 2, 2048 bits)
- **MACCS keys** (where used in multi-feature runs)
- **RDKit 2D descriptors** (e.g. MolWt, LogP, TPSA, HBD/HBA, rotatable bonds, aromatic rings), standardized on the training fold

### 3.2 Models
1. **Per-isoform LightGBM regressors**  
   - Morgan-only baseline  
   - Morgan + descriptors (stronger local validation)  
2. **Chemprop directed message-passing neural network (D-MPNN)**  
   - Multitask / multi-isoform setups  
   - CheMeleon foundation-style initialization / fine-tuning when weights were available  
   - GPU training in Colab with early stopping on validation loss  
3. **Simple ensembles** of tabular and Chemprop predictions (validation-informed blending)

Isoforms modeled: CYP1A2, CYP2C9, CYP2D6, CYP3A4 (`*_pIC50_direct_inhibition`).

### 3.3 Validation and metrics
- **Bemis–Murcko scaffold split** (group-disjoint train/validation) to reduce analogue leakage versus random splits  
- Metrics reported internally: MAE, \(R^2\), Spearman ρ  
- Primary challenge metric: **soft-threshold RAE (ST-RAE)** using experimental `*_conf_low` / `*_conf_high` intervals (zero error if the prediction lies inside the interval; otherwise distance to the nearest bound, normalized)

### 3.4 Test submission
Models were refit using available training labels and applied to the 750-compound blinded test set.  
**Format:** CSV with columns  
`SMILES`, `Molecule_Name`,  
`CYP1A2_pIC50_direct_inhibition`, `CYP2C9_pIC50_direct_inhibition`,  
`CYP2D6_pIC50_direct_inhibition`, `CYP3A4_pIC50_direct_inhibition`.

---

## 4. Time-dependent inhibition (classification)

### 4.1 Features
- Morgan fingerprints, MACCS keys, RDKit descriptors  
- **SMARTS structural alerts** associated with mechanism-based inactivation chemistry (e.g. methylenedioxy, aniline/phenol-related patterns, furan/thiophene, selected Michael-acceptor-like motifs), used as binary indicator features  

### 4.2 Models
- **Per-isoform LightGBM classifiers** with `scale_pos_weight` for class imbalance  
- Exploratory **Chemprop classification** heads on the same labels (CLI / package path constrained by Colab environment)

Targets: `CYP2D6_is_TDI`, `CYP3A4_is_TDI`.

### 4.3 Threshold selection
Probability thresholds were chosen to maximize **Matthews correlation coefficient (MCC)** on scaffold-aware held-out or multi-fold out-of-fold scores.  
Because unconstrained MCC search can select unrealistically low thresholds (especially for CYP2D6), final rules applied **constraints** on maximum predicted positive rate and minimum precision so test positive rates stay near training prevalence rather than extreme recall-only operating points.

### 4.4 Test submission
**Format:** CSV with  
`SMILES`, `Molecule_Name`, `CYP2D6_is_TDI`, `CYP3A4_is_TDI` (boolean).

---

## 5. Structure prediction

Two complementary approaches were implemented for the 20 structure-track compounds.

### 5.1 Boltz-2 co-folding (primary ML structure model)
- Isolated **Python 3.12** environment with `boltz[cuda]` (Boltz-2)  
- Input YAML per ligand:
  - **Protein:** human CYP3A4 sequence (chain A)
  - **Ligand B:** heme as CCD code `HEM`
  - **Ligand C:** challenge ligand SMILES  
- Inference settings used in runs: `--diffusion_samples 1`, `--recycling_steps 3`, `--use_msa_server`, `--no_kernels`  
- Outputs written under a Drive-backed directory; existing CIF/PDB products skipped on re-runs  

### 5.2 AutoDock Vina docking baseline (complete submission path)
Used when full co-folding of all 20 was limited by runtime:

1. Receptor: public **CYP3A4** structure **PDB 1TQN** (protein + heme)  
2. Ligands: RDKit embed + minimization → Open Babel PDBQT  
3. **AutoDock Vina** docking, box centered on the heme centroid (~22 Å), exhaustiveness 8  
4. Best pose converted to PDB, ligand residue named **`LIG`**, assembled as protein + HEM + LIG  
5. All **20/20** complexes packaged as a zip for the structure track  

**Submission format:** zip of 20 PDB files named by `Molecule_Name`, each a full complex with ligand residue `LIG`.

---

## 6. Software

| Component | Role |
|-----------|------|
| RDKit | Featurization, scaffolds, ligand prep |
| LightGBM / scikit-learn | Tabular regression & classification |
| Chemprop + PyTorch / Lightning | Graph neural models; CheMeleon-related fine-tuning |
| Hugging Face `datasets` | Official challenge tables |
| Boltz-2 | Structure co-folding |
| Open Babel + AutoDock Vina | Docking-based structure submission |
| Google Colab (GPU when available) | Training and inference |

---

## 7. Limitations

- Scaffold splits reduce leakage but do not fully match the challenge’s analog-expansion construction of the blind set; a train–blind gap remains, especially for CYP2D6.  
- TDI signal for CYP2D6 is weaker than for CYP3A4 under our features; threshold discipline matters as much as model choice.  
- Boltz-2 co-folding is closer to state-of-the-art pose generation but is slower; the Vina path is a rigid-receptor approximation without protein remodeling.  
- Chemprop CLI packaging differed across Colab images (`python -m chemprop` vs entry points); final tabular TDI models did not depend on that path.

---

## 8. Acknowledgments

We thank the OpenADMET organizers for the dataset, ST-RAE / MCC / LDDT-PLI metrics, and challenge platform, and the developers of RDKit, Chemprop, LightGBM, Boltz, and AutoDock Vina.

---

## 9. Contact

- Hugging Face username: **srajall**  
- Email: st735@snu.ed.in
