# cyp-challenge

## 1. Data source


### 1.1 Official challenge data

cyp-challenge-TRAIN_inhibition.csv 
cyp-challenge-single-concentration-TRAIN.csv
cyp-challenge-TRAIN_TDI.csv
cyp-challenge-TRAIN_Emax.csv
cyp-challenge-TEST-BLINDED.csv

### 1.2 Public data
PubChem
Octant CYP3A4
ChEMBL CYP2C8
ChEMBL CYP1A1
ChEMBL CYP1B1
ChEMBL CYP2B6

### 1.3 Simulated Analog Validation Set (valBig)

Used for model selection and evaluation of method effectiveness

**Design Principle**: Simulates the generation method of the actual TEST - the actual TEST consists of 75 strong-inhibitory parent molecules (25 each for CYP1A2/2C9/3A4). An identical "parent → analog" validation set is constructed within the official TRAIN to test the model's generalization ability for "structural variants of strong inhibitors"

**Specific Steps**:
1. Select the 25 molecules with the highest pIC50 for each of CYP1A2/2C9/3A4 as "parents" (sampled randomly with seed 45)
2. Calculate Tanimoto similarity using Morgan fingerprints (radius = 2, 2048 bits)
3. For each parent, find up to 20 analogs with a similarity of ≥ 0.35 in the remaining TRAIN molecules (only considering structural similarity, not activity values - to avoid using labels to select validation samples)
4. The selected 311 analogs (438 labels) are removed from the entire training data by connecting blocks based on InChIKey (including duplicate rows with the same structure in public data and all auxiliary labels), ensuring zero leakage
5. The parents remain in the training set

**Output**: 311 holdout molecules, with label counts: 1A2 = 107 / 2C9 = 86 / 2D6 = 110 / 3A4 = 135

## 2. Model architecture

Chemprop v2 D-MPNN
| Model | Heads | hidden_dim | strategy | enzyme |
|---|---:|---:|---|---|
| cp_h500 | 16 | 500 | D-MPNN | 3A4 |
| cp_h600 | 16 | 600 | D-MPNN | 2C9 |
| cp_h800 | 16 | 800 | D-MPNN | 1A2 |
| cpext | 23 | 700 | D-MPNN | 1A2 |
| SFLcp | 16 | 700 | SFL+ood early stopping | 2D6, 3A4 |
| SFLcp_ext | 23 | 700 | SFL+ood early stopping | 2D6 |



## 3. SFL training enhancement

### 3.1 Principle

SFL (Stretching Features and Labels) generates pseudo-OUT-of-Distribution (OOD) samples in the latent space: Instead of altering the molecular input structure, it extrapolates the embedding vectors output by the encoder and proportionally adjusts the corresponding labels - **the labels are stretched more severely** (doubly magnified) compared to the features, creating "samples with properties beyond the training range".

### 3.2 Specific Operations (For Each Batch)

1. **XOOD**: Find a partner for each molecule based on the similarity of features (labels are similar × structures are the farthest apart)
2. **YOOD**: Feature = Feature + λ·(Feature - Batch Mean), Label = Label + (1 + 0.5λ²)·λ·(Label - Mean)
3. **mixup**: Standard Beta(0.5, 0.5) feature mixing
4. **Loss**: 0.75·(ID Loss + 0.5·mixup Loss) + 0.25·(0.25·YOOD₁ + 0.75·YOOD₂)

### 3.3 Hyperparameter

lam_id:0.75
lam_mixup:0.5
lam_yood:0.25
lam_xood:0.75
lam_label_mix:0.5
xood_beta:(0.5, 0.5)
yood_beta:(5.0, 2.0)

### 3.4 Early Stopping Mode

SFL-A employs **pseudo-OOD validation scores** for early stopping (performing fixed λ = 0.5 and 0.7 extrapolations on the embedded/labelled elements of the validation set, and taking the masked MAE)

## 4. Integration and Sequential Enzyme Assembly

### 4.1 5-Fold Integration

Each model undergoes 5-fold scaffold grouping cross-validation for training, and during prediction, the predictions from the 5 groups are averaged arithmetically.

### 4.2 Sequential Enzyme Mixed Weights

Different enzymes have different responses to SFL, so each enzyme uses an independent mixed weight:
1A2: 50%·cp_h800 + 50%·cpext
2C9: cp_h600
2D6: 70%·SFLcp + 30%·SFLcp_ext
3A4: 70%·cp_h500 + 30%·SFLcp


## 5. Exploratory Directions That Were Not Adopted

The following directions were all systematically tested (using the valBig validation set and partial real-world measurements) and were excluded due to insufficient effectiveness or negative results: 

- Average of Multiple Seeds
- Tox21
- 2D6 pooling tags
- XGBoost/Morgan fingerprints
- ChemBERTa embeddings
