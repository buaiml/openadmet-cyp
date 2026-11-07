# OpenADMET Blind Challenge: Predicting CYP Inhibition

Team name: **nova**

- This is our model report for the [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge).
- We participated in all tracks, but we focused on TDI classification and structure prediction.

### TDI classification (CYP2D6, CYP3A4)
- **Data sources**
  - Challenge data and curated public data
  - No proprietary data used
- **Molecular features**
  - ECFP and Avalon fingerprints
  - RDKit descriptors
  - MiniMol, Monroe, CheMeleon, Atom-JEPA embeddings
  - Out-of-fold surrogate outputs from the external data
- **Model architectures**
  - SVC
  - LGBM 
  - ChemProp-MT (CheMeleon)
  - AutoGluon
  - Kumo
- **Ensembling strategy (weighted average)**
  - Caruana-selected weighted rank average across feature and model families
  - Top-K positive set per endpoint from inferred prevalence
 
 ### Direct inhibition (pIC50: CYP1A2, 2C9, 2D6, 3A4; ST-RAE)
- **Data sources**
  - Challenge data and curated public data
  - No proprietary data used
- **Molecular features**
  - ECFP4, FCFP4, Avalon
  - RDKit descriptors, ADMET-AI
  - CheMeleon, MiniMol, Monroe, CLAMP
  - Mol-JEPA / Atom-JEPA, SAND (shape-aware)
  - Out-of-fold surrogate outputs from the external data
- **Model architectures**
  - LGBM, RF, ExtraTrees, SVR
  - AutoGluon `extreme_quality` (GBM, CatBoost, RealMLP, TabICL, TabDPT), RealTabPFN-v2
  - ChemProp multitask, CheMeleon-initialised, with auxiliary heads (MAE, MSE, evidential and interval-hinge losses)
- **Ensembling strategy (weighted average)**
  - MPO-weighted average of family-representative members, with a CheMeleon member always included


 ### Structure (CYP3A4 protein-ligand poses, 20 ligands)
- **Data sources**
  - Challenge ligand SMILES
  - CYP3A4 protein sequence
  - Public CYP3A4 structures
- **Model architectures**
  - ESMFold2 (`esm` 3.3.0)
  - OpenFold3 (ob0 checkpoint)
  - Boltz-2 2.2.1 with Boltzina 1.0.1
- **Pose selection strategy**
  - Highest-ipTM PoseBusters-passing sample per ligand followed by OpenMM minimization
  - Visual inspection
 
